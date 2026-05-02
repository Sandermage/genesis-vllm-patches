#!/usr/bin/env python3
"""
Genesis Quality Harness v1.0
Automated regression test suite for Qwen3.6-35B-A3B-FP8 on 2xA5000.

Runs before/after every vLLM change (patch, config, model, quant).
Produces GO/NO-GO verdict + JSON for baseline comparison.

Usage:
    python genesis_quality_harness.py --host localhost --port 8000
    python genesis_quality_harness.py --label pre-mtp --baseline baseline.json

Exit codes:
    0  all critical tests passed
    1  at least one critical test failed
    2  harness error (network, server down)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field

import httpx


# ─────────────────────────────────────────────────────────────────────
# Test case definitions
# ─────────────────────────────────────────────────────────────────────

@dataclass
class TestCase:
    """Single probe against the model."""

    name: str
    prompt: str
    expect: str | list[str]   # substring or list of acceptable substrings (OR)
    critical: bool = False    # if True, failure -> NO-GO verdict
    max_tokens: int = 50
    thinking: bool = False
    temperature: float = 0.0
    extra_body: dict = field(default_factory=dict)

    def passes(self, reply: str) -> bool:
        """Accepts if any expected substring found (case-insensitive)."""
        r = reply.lower()
        if isinstance(self.expect, str):
            return self.expect.lower() in r
        return any(e.lower() in r for e in self.expect)


# ─────────────────────────────────────────────────────────────────────
# Test suites
# ─────────────────────────────────────────────────────────────────────

MATH_TESTS = [
    TestCase("math_mul_1", "Compute 47*13. Reply ONLY the number.", "611"),
    TestCase("math_mul_2", "Compute 123*7. Reply ONLY the number.", "861"),
    TestCase("math_add",   "Compute 4528+1973. Reply ONLY the number.", "6501"),
    TestCase("math_sub",   "Compute 1000-347. Reply ONLY the number.", "653"),
    TestCase("math_div",   "Compute 144/12. Reply ONLY the number.", "12"),
    TestCase("math_order", "Compute (3+4)*2. Reply ONLY the number.", "14"),
    TestCase("math_pct",   "What is 25% of 240? Reply ONLY the number.", "60"),
    TestCase("math_fact",  "What is 5 factorial? Reply ONLY the number.", "120"),
]

FACTUAL_TESTS = [
    TestCase("fact_capital_jp", "Capital of Japan? One word.", "tokyo"),
    TestCase("fact_capital_de", "Capital of Germany? One word.", "berlin"),
    TestCase("fact_capital_fr", "Capital of France? One word.", "paris"),
    TestCase("fact_author_1984", "Author of '1984' (last name only)?", "orwell"),
    TestCase("fact_year_ww2",   "Year World War II ended (4 digits)?", "1945"),
    TestCase("fact_speed_light", "Speed of light in vacuum in m/s, order of magnitude (10^?)? Just the exponent number.", "8"),
]

LOGIC_TESTS = [
    TestCase("logic_syllogism",
             "All cats are mammals. All mammals are animals. Are cats animals? yes/no.",
             "yes"),
    TestCase("logic_contrapositive",
             "If it rains, the ground is wet. The ground is NOT wet. Is it raining? yes/no.",
             "no"),
    TestCase("logic_and",
             "A is true. B is false. Is (A and B) true? yes/no.",
             "no"),
    TestCase("logic_or",
             "A is true. B is false. Is (A or B) true? yes/no.",
             "yes"),
]

CODE_TESTS = [
    TestCase("code_fib",
             "Write a Python one-line expression that computes the 10th Fibonacci number (F(0)=0, F(1)=1). Expression only, no explanation, no markdown.",
             ["55"], max_tokens=80),
    TestCase("code_sq_sum",
             "Python one-line expression: sum of squares of 1..5. Expression only.",
             ["55", "sum(i*i", "sum(i**2"], max_tokens=60),
    TestCase("code_reverse",
             "Write Python: reverse the string 'hello'. Expression only, no explanation.",
             ["'olleh'", '"olleh"', "[::-1]"], max_tokens=60),
    TestCase("code_primes",
             "How many primes below 10? ONLY the number.",
             "4"),
]

MULTISTEP_TESTS = [
    TestCase(
        "mstep_apples",
        "Sarah has 3 times as many apples as Tom. Tom has 5 more than Bob. Bob has 4 apples. "
        "How many apples does Sarah have? Number only.",
        "27",
        max_tokens=800,
        thinking=True,
    ),
    TestCase(
        "mstep_age",
        "Alice is twice as old as Bob. Bob is 5 years younger than Carol. Carol is 15. "
        "How old is Alice? Number only.",
        "20",
        max_tokens=600,
        thinking=True,
    ),
]

TOOL_CALL_TESTS = [
    # These validate that the qwen3_coder parser fires and produces a valid tool call.
    # The "expect" field matches the reply content OR tool_calls structure.
    TestCase(
        "tool_get_weather",
        "What is the weather in Tokyo?",
        "tokyo",
        max_tokens=300,
        extra_body={
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }],
            "tool_choice": "auto",
        },
    ),
    TestCase(
        "tool_calculator",
        "Compute 17 multiplied by 23 using the calculator tool.",
        ["17", "23", "multiply"],
        max_tokens=300,
        extra_body={
            "tools": [{
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Perform arithmetic",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["op", "a", "b"],
                    },
                },
            }],
            "tool_choice": "auto",
        },
    ),
]

MULTILINGUAL_TESTS = [
    TestCase(
        "multi_ru",
        "Переведи на английский: 'Привет, как дела?' Only translation, no quotes.",
        ["hello, how are you", "hi, how are you"],
    ),
    TestCase(
        "multi_zh",
        "Translate to English: 你好吗？ Only translation.",
        ["how are you", "are you well"],
    ),
]

COHERENCE_TESTS = [
    # Ensures model doesn't loop / emit garbage on a known-problematic prompt
    TestCase(
        "coherence_cn_loop_probe",
        "请为我设计一个适合团队玩的趣味游戏，游戏规则要详细，并且要有明确的胜利条件。",
        # Accept any substantial Chinese output that isn't a loop
        ["游戏", "规则", "团队"],
        max_tokens=2500,
        temperature=0.7,
        extra_body={"presence_penalty": 1.5},  # Qwen3.6 team recommendation
        critical=True,  # KILLSWITCH: if loops, whole harness fails
    ),
]

JSON_TESTS = [
    TestCase(
        "json_simple",
        'Output ONLY this JSON (no markdown): {"status": "ok", "value": 42}',
        ['"status"', '"ok"', "42"],
    ),
    TestCase(
        "json_nested",
        'Output ONLY a JSON object with: name="Alice", age=30, hobbies=["reading","chess"]. '
        'No markdown, no explanation.',
        ["Alice", "30", "reading", "chess"],
    ),
]

# Long-context: needle-in-haystack at various prompt sizes.
# Note: Qwen3.6-35B-A3B needle recall becomes non-deterministic beyond ~50k.
# We use 10k/30k for reliable regression detection and 80k as a stretch goal.
# Each entry: (name, target_prompt_tokens, needle_string, required)
LONG_CTX_TESTS = [
    ("needle_10k",  10_000,  "SENTINEL-BRAVO-77", True),   # must pass
    ("needle_30k",  30_000,  "SENTINEL-BRAVO-77", True),   # must pass
    ("needle_80k",  80_000,  "SENTINEL-BRAVO-77", False),  # advisory only
]


ALL_SUITES: dict[str, list[TestCase]] = {
    "math":         MATH_TESTS,
    "factual":      FACTUAL_TESTS,
    "logic":        LOGIC_TESTS,
    "code":         CODE_TESTS,
    "multistep":    MULTISTEP_TESTS,
    "tool_call":    TOOL_CALL_TESTS,
    "multilingual": MULTILINGUAL_TESTS,
    "coherence":    COHERENCE_TESTS,
    "json":         JSON_TESTS,
}


# ─────────────────────────────────────────────────────────────────────
# Thresholds (per-suite pass rate required for GO verdict)
# ─────────────────────────────────────────────────────────────────────

THRESHOLDS = {
    "math":         0.90,
    "factual":      0.80,
    "logic":        0.75,
    "code":         0.75,
    "multistep":    0.50,  # harder, thinking-mode dependent
    "tool_call":    0.90,  # critical for genesis-aggregator
    "multilingual": 0.50,
    "coherence":    1.00,  # must not loop
    "json":         0.75,
    "long_context": 0.90,  # needle recall
}


# ─────────────────────────────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────────────────────────────

def call_model(client: httpx.Client, base_url: str, api_key: str,
               model: str, tc: TestCase) -> dict:
    """Run one test case, return {content, reasoning, finish, latency, error}."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": tc.prompt}],
        "max_tokens": tc.max_tokens,
        "temperature": tc.temperature,
    }
    if not tc.thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    body.update(tc.extra_body)

    t0 = time.time()
    try:
        r = client.post(f"{base_url}/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json=body, timeout=180.0)
        latency = time.time() - t0
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}", "latency": latency}
        d = r.json()
        msg = d["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        return {
            "content": (msg.get("content") or "").strip(),
            "reasoning": msg.get("reasoning_content") or "",
            "tool_calls": tool_calls,
            "finish": d["choices"][0]["finish_reason"],
            "latency": latency,
            "usage": d.get("usage", {}),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "latency": time.time() - t0}


def detect_loop(text: str) -> bool:
    """Detect if text contains a repetitive loop (>5 occurrences of any 80-char window)."""
    if len(text) < 500:
        return False
    # Simple: sliding 80-char windows with step 40
    from collections import Counter
    windows = [text[i:i + 80] for i in range(0, len(text) - 80, 40)]
    if not windows:
        return False
    top = Counter(windows).most_common(1)[0]
    return top[1] >= 5


def run_suite(client: httpx.Client, base_url: str, api_key: str, model: str,
              suite_name: str, tests: list[TestCase], verbose: bool) -> dict:
    """Run all tests in a suite."""
    print(f"\n── {suite_name.upper()} ──")
    results = []
    for tc in tests:
        r = call_model(client, base_url, base_url and api_key, model, tc)
        if "error" in r:
            passed = False
            reply = r["error"]
        else:
            combined = (r["content"] + " " + " ".join(
                json.dumps(tcl.get("function", {}).get("arguments", "")) if isinstance(tcl, dict) else ""
                for tcl in r.get("tool_calls") or []
            )).lower()
            passed = tc.passes(combined)
            reply = r["content"][:80]

            # Coherence suite: also check for loops
            if suite_name == "coherence" and tc.critical:
                all_text = r.get("reasoning", "") + r["content"]
                if detect_loop(all_text):
                    passed = False
                    reply = "LOOP DETECTED"

        status = "✅" if passed else "❌"
        lat = f"{r.get('latency', 0):.1f}s"
        print(f"  {status} {tc.name:<28} ({lat}) {reply!r:<80}")
        results.append({
            "name": tc.name,
            "passed": passed,
            "critical": tc.critical,
            "latency": r.get("latency", 0),
            "reply": reply[:200],
            "error": r.get("error"),
        })

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"  → {passed}/{total} passed")
    return {"results": results, "passed": passed, "total": total}


def run_long_context(client: httpx.Client, base_url: str, api_key: str, model: str) -> dict:
    """Needle-in-haystack at specific prompt-token sizes."""
    print(f"\n── LONG_CONTEXT ──")
    results = []

    for name, target_tokens, needle, required in LONG_CTX_TESTS:
        # Build prompt: padding + needle buried at ~60% depth + padding + question
        pad_chunk = "The research team discussed various findings. The results were inconclusive. "
        tokens_per_chunk = 12
        needed_before = int(target_tokens * 0.6 / tokens_per_chunk)
        needed_after = int(target_tokens * 0.4 / tokens_per_chunk)
        before = pad_chunk * needed_before
        after = pad_chunk * needed_after
        # Visual delimiters + strict instruction — empirically most reliable
        # format for Qwen3.6 needle recall at variable context depths.
        prompt = (
            before
            + f"\n\n>>> The access code is {needle}. <<<\n\n"
            + after
            + "\n\nWhat is the access code mentioned in the text? Just the code, nothing else."
        )

        tc = TestCase(name, prompt, needle, max_tokens=30)
        r = call_model(client, base_url, api_key, model, tc)
        if "error" in r:
            passed = False
            reply = r["error"][:80]
            actual_tokens = 0
        else:
            passed = tc.passes(r["content"])
            reply = r["content"][:80]
            actual_tokens = r.get("usage", {}).get("prompt_tokens", 0)

        tag = "" if required else "(advisory)"
        status = "✅" if passed else ("❌" if required else "⚠️ ")
        print(f"  {status} {name:<15} ~{actual_tokens:>6}tok ({r.get('latency',0):.1f}s) {tag:<10} {reply!r}")
        results.append({
            "name": name,
            "target_tokens": target_tokens,
            "actual_tokens": actual_tokens,
            "passed": passed,
            "required": required,
            "reply": reply,
        })

    # Only required tests count toward pass/total for threshold evaluation
    required_results = [r for r in results if r["required"]]
    passed = sum(1 for r in required_results if r["passed"])
    total = len(required_results)
    advisory_passed = sum(1 for r in results if not r["required"] and r["passed"])
    advisory_total = sum(1 for r in results if not r["required"])
    print(f"  → {passed}/{total} required passed ({advisory_passed}/{advisory_total} advisory)")
    return {"results": results, "passed": passed, "total": total,
            "advisory_passed": advisory_passed, "advisory_total": advisory_total}


# ─────────────────────────────────────────────────────────────────────
# Baseline comparison + verdict
# ─────────────────────────────────────────────────────────────────────

def verdict(suites: dict) -> tuple[str, list[str]]:
    """Return (GO/NO-GO, list of failure reasons)."""
    reasons = []
    for suite_name, s in suites.items():
        total = s["total"]
        if total == 0:
            continue
        rate = s["passed"] / total
        threshold = THRESHOLDS.get(suite_name, 0.70)
        if rate < threshold:
            reasons.append(
                f"{suite_name}: {s['passed']}/{total} = {rate:.0%} < required {threshold:.0%}"
            )
        # Critical test failure anywhere = NO-GO
        for r in s["results"]:
            if r.get("critical") and not r["passed"]:
                reasons.append(f"CRITICAL: {r['name']} failed")
    return ("GO" if not reasons else "NO-GO"), reasons


def compare_baseline(current: dict, baseline_path: str | None) -> list[str]:
    """Compare current run vs baseline JSON, report regressions."""
    if not baseline_path:
        return []
    try:
        base = json.load(open(baseline_path))
    except Exception as e:
        return [f"baseline load error: {e}"]

    regressions = []
    for suite_name in current["suites"]:
        cur = current["suites"][suite_name]
        basin = base.get("suites", {}).get(suite_name)
        if not basin:
            continue
        cur_rate = cur["passed"] / cur["total"] if cur["total"] else 0
        base_rate = basin["passed"] / basin["total"] if basin["total"] else 0
        if cur_rate < base_rate - 0.05:  # >5 pp regression
            regressions.append(
                f"{suite_name}: {cur_rate:.0%} (was {base_rate:.0%}, Δ={cur_rate-base_rate:+.0%})"
            )
    return regressions


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Genesis Quality Harness v1.0")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="qwen3.6-35b-a3b")
    ap.add_argument("--api-key", default="genesis-local")
    ap.add_argument("--label", default="unlabeled")
    ap.add_argument("--baseline", default=None, help="Path to prior result JSON for regression compare")
    ap.add_argument("--skip-long", action="store_true", help="Skip long-context needle tests")
    ap.add_argument("--suite", action="append", help="Run only specific suite(s)")
    ap.add_argument("--output", default=None, help="Write results JSON here (default: auto-named)")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    print(f"Genesis Quality Harness v1.0")
    print(f"Endpoint: {base_url}")
    print(f"Model:    {args.model}")
    print(f"Label:    {args.label}")
    print(f"=" * 60)

    # Pre-flight
    with httpx.Client(timeout=10.0) as c:
        try:
            h = c.get(f"{base_url}/health")
            if h.status_code != 200:
                print(f"❌ Server not healthy: {h.status_code}")
                return 2
        except Exception as e:
            print(f"❌ Server unreachable: {e}")
            return 2

    results = {
        "harness_version": "1.0",
        "label": args.label,
        "timestamp": time.time(),
        "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "endpoint": base_url,
        "model": args.model,
        "suites": {},
    }

    suites_to_run = args.suite or list(ALL_SUITES.keys())

    with httpx.Client(timeout=300.0) as client:
        for suite_name in suites_to_run:
            if suite_name not in ALL_SUITES:
                print(f"⚠️  Unknown suite: {suite_name}")
                continue
            s = run_suite(client, base_url, args.api_key, args.model,
                          suite_name, ALL_SUITES[suite_name], args.verbose)
            results["suites"][suite_name] = s

        if not args.skip_long and (not args.suite or "long_context" in args.suite):
            s = run_long_context(client, base_url, args.api_key, args.model)
            results["suites"]["long_context"] = s

    # Verdict
    v, reasons = verdict(results["suites"])
    regressions = compare_baseline(results, args.baseline)

    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    total_passed = sum(s["passed"] for s in results["suites"].values())
    total_total = sum(s["total"] for s in results["suites"].values())
    print(f"  Overall: {total_passed}/{total_total}")
    for sn, s in results["suites"].items():
        mark = "✅" if (s["passed"] / max(s["total"], 1)) >= THRESHOLDS.get(sn, 0.7) else "❌"
        print(f"  {mark} {sn:<15} {s['passed']}/{s['total']}")
    print()
    if reasons:
        print("  Failure reasons:")
        for r in reasons:
            print(f"    • {r}")
    if regressions:
        print("  🔴 Regressions vs baseline:")
        for r in regressions:
            print(f"    • {r}")
    print()
    print(f"  {'✅' if v == 'GO' else '❌'} {v}")
    print("=" * 60)

    # Save
    out_path = args.output or f"harness_{args.label}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    results["verdict"] = v
    results["failure_reasons"] = reasons
    results["regressions"] = regressions
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    return 0 if v == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
