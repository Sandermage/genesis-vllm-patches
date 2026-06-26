#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Genesis comprehensive benchmark — README-ready data.

Measures the full operator-relevant matrix on a live vLLM container:

  1. Cold-warm latency (5 single-shot runs, drop fastest+slowest)
  2. Sustained TPS (10×400-token generation, mean + CV%)
  3. Tool-call clean rate (10 different tool prompts)
  4. Multi-turn stability (10-turn soak, turns survived + per-turn delta)
  5. VRAM steady-state (post-warmup)
  6. Long-context needle (1K / 10K / 50K / 90K depths)

Output: markdown table block ready to paste into README.

Usage::

    GENESIS_ENDPOINT=http://192.0.2.10:8000 \\
    GENESIS_MODEL=qwen3.6-27b \\
    python3 tests/bench/comprehensive_bench.py

Author: Sandermage 2026-05-05.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any


API_KEY = os.environ.get("GENESIS_API_KEY", "genesis-local")
ENDPOINT = os.environ.get(
    "GENESIS_ENDPOINT", "http://192.0.2.10:8000/v1/chat/completions"
)
MODEL = os.environ.get("GENESIS_MODEL", "qwen3.6-27b")
SSH_HOST = os.environ.get("GENESIS_SSH_HOST", "user@192.0.2.10")


def _api_post(payload: dict, timeout: float = 600.0) -> dict | None:
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"  [WARN] API error: {e}", file=sys.stderr)
        return None


def _vram_used_per_gpu() -> list[int]:
    """Returns list of MiB per GPU via ssh+nvidia-smi."""
    try:
        out = subprocess.run(
            ["ssh", SSH_HOST, "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return [int(x.strip()) for x in out.stdout.strip().splitlines() if x.strip()]
    except Exception as e:
        print(f"  [WARN] VRAM probe failed: {e}", file=sys.stderr)
        return []


# ─── Bench 1: cold-warm latency ─────────────────────────────────────────


def bench_cold_warm_latency() -> dict:
    """5 single-shot 400-token runs with same short prompt; drop min+max."""
    print("\n[1/6] Cold-warm latency (5×400t single-shot)")
    times: list[float] = []
    for i in range(5):
        t0 = time.perf_counter()
        resp = _api_post({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Tell me a story about a brave fox."}],
            "max_tokens": 400, "temperature": 0,
        })
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  run {i+1}: {dt:.2f}s")
        if resp is None:
            print(f"  [WARN] run {i+1} failed", file=sys.stderr)
    if len(times) < 5:
        return {"status": "incomplete", "samples": times}
    sorted_t = sorted(times)
    trimmed = sorted_t[1:-1]  # drop fastest + slowest
    return {
        "status": "ok",
        "min": min(times),
        "max": max(times),
        "trimmed_mean": statistics.mean(trimmed),
        "all_samples": times,
    }


# ─── Bench 2: sustained TPS ─────────────────────────────────────────────


def bench_sustained_tps() -> dict:
    """10 sequential 400-token generations; report TPS mean + CV%."""
    print("\n[2/6] Sustained TPS (10×400t generation)")
    tps_samples: list[float] = []
    for i in range(10):
        t0 = time.perf_counter()
        resp = _api_post({
            "model": MODEL,
            "messages": [{"role": "user", "content": f"Iteration {i+1}: write a paragraph about Quantum mechanics."}],
            "max_tokens": 400, "temperature": 0.7,
        })
        dt = time.perf_counter() - t0
        if resp is None:
            print(f"  [WARN] iter {i+1} failed")
            continue
        usage = resp.get("usage") or {}
        completion_tokens = usage.get("completion_tokens", 0)
        if completion_tokens > 0 and dt > 0:
            tps = completion_tokens / dt
            tps_samples.append(tps)
            print(f"  iter {i+1:2d}: {completion_tokens:3d}t in {dt:5.2f}s = {tps:6.1f} tok/s")
    if not tps_samples:
        return {"status": "fail"}
    mean_tps = statistics.mean(tps_samples)
    stdev_tps = statistics.stdev(tps_samples) if len(tps_samples) > 1 else 0.0
    cv_pct = (stdev_tps / mean_tps * 100.0) if mean_tps > 0 else 0.0
    return {
        "status": "ok",
        "mean_tps": mean_tps,
        "min_tps": min(tps_samples),
        "max_tps": max(tps_samples),
        "cv_pct": cv_pct,
        "n_samples": len(tps_samples),
    }


# ─── Bench 3: tool-call clean rate ──────────────────────────────────────


_TOOL_PROMPTS = [
    "Get the weather for Berlin",
    "Check weather in Tokyo",
    "What's the weather in Sydney?",
    "Use get_weather for New York",
    "Weather in London please",
    "Tell me weather of Paris",
    "Get_weather Madrid",
    "Weather check for Moscow",
    "What's the weather in Shanghai",
    "Check weather: Mumbai",
]
_WEATHER_TOOL = {
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
}


def bench_tool_call_clean_rate() -> dict:
    """10 different tool prompts; count how many emit a clean tool_call."""
    print("\n[3/6] Tool-call clean rate (10 different prompts)")
    clean = 0
    failed = 0
    fail_details = []
    for i, prompt in enumerate(_TOOL_PROMPTS):
        resp = _api_post({
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200, "temperature": 0,
            "tools": [_WEATHER_TOOL], "tool_choice": "auto",
        })
        if resp is None:
            failed += 1
            fail_details.append(f"prompt {i+1}: API error")
            continue
        ch = resp["choices"][0]
        tc = ch["message"].get("tool_calls") or []
        if tc and tc[0]["function"]["name"] == "get_weather":
            try:
                args = json.loads(tc[0]["function"]["arguments"])
                if "city" in args and len(args["city"]) > 0:
                    clean += 1
                    print(f"  prompt {i+1:2d}: ✓ {args}")
                    continue
            except Exception:
                pass
        failed += 1
        fail_details.append(f"prompt {i+1}: tc={tc}")
        print(f"  prompt {i+1:2d}: ✗ no clean tool_call")
    return {
        "status": "ok",
        "clean": clean,
        "total": len(_TOOL_PROMPTS),
        "rate_pct": (clean / len(_TOOL_PROMPTS) * 100.0),
        "fail_details": fail_details[:3],
    }


# ─── Bench 4: multi-turn stability ──────────────────────────────────────


def bench_multi_turn_stability(n_turns: int = 10) -> dict:
    """N-turn soak with growing context; track per-turn latency."""
    print(f"\n[4/6] Multi-turn stability ({n_turns}-turn soak)")
    history = [{"role": "system", "content": "You are a helpful assistant."}]
    turn_latencies: list[float] = []
    survived = 0
    for turn in range(n_turns):
        history.append({
            "role": "user",
            "content": f"Turn {turn+1}: tell me one fact about chemistry, briefly."
        })
        t0 = time.perf_counter()
        resp = _api_post({
            "model": MODEL, "messages": history,
            "max_tokens": 200, "temperature": 0.7,
        })
        dt = time.perf_counter() - t0
        if resp is None:
            print(f"  turn {turn+1}: ✗ API error")
            break
        survived += 1
        content = resp["choices"][0]["message"].get("content") or \
                  resp["choices"][0]["message"].get("reasoning") or ""
        history.append({"role": "assistant", "content": content[:200]})
        turn_latencies.append(dt)
        print(f"  turn {turn+1:2d}: {dt:.2f}s")
    return {
        "status": "ok" if survived == n_turns else "partial",
        "survived": survived,
        "total": n_turns,
        "mean_latency": statistics.mean(turn_latencies) if turn_latencies else 0,
        "max_latency": max(turn_latencies) if turn_latencies else 0,
    }


# ─── Bench 5: VRAM steady-state ─────────────────────────────────────────


def bench_vram_steady() -> dict:
    """Sample VRAM after warmup."""
    print("\n[5/6] VRAM steady-state")
    vram = _vram_used_per_gpu()
    return {"status": "ok", "vram_mib_per_gpu": vram, "total_mib": sum(vram)}


# ─── Bench 6: long-context needle ───────────────────────────────────────


_NEEDLE_STORY_TEMPLATE = (
    "A research project investigated the migration patterns of arctic terns "
    "across the North Atlantic during the spring of 2023. The study used "
    "satellite tracking devices and recorded daily flight distances. "
    "Researchers also collected weather data, sea surface temperatures, and "
    "feeding patterns. The methodology was developed by a team at the "
    "University of Copenhagen, with funding from the European Research Council. "
    "Results were published in the Journal of Avian Biology in November 2024. "
)
_NEEDLE_FACT = (
    "Buried in the appendix of the paper is one operational detail not "
    "mentioned in the abstract: the reference identifier assigned to the "
    "primary tagged tern was BANDIT-K9-7821, with mass at tagging recorded "
    "as 142 grams. This identifier was used to cross-reference all "
    "subsequent observations in the public dataset. "
)
_NEEDLE_QUESTION = (
    "Question: What was the reference identifier assigned to the primary "
    "tagged tern in this research, and what was its mass at tagging? "
    "Answer in one sentence."
)


def bench_needle_ladder(skip: bool = False) -> dict:
    """4-depth needle ladder using a realistic research-paper context.

    Audit fix 2026-05-05 (genesis_deep_cross_audit + bench feedback): the
    previous lorem-ipsum filler scored 0/4 because the model latched onto
    the meaningless filler instead of the needle. The new approach uses a
    coherent multi-paragraph research-paper-style filler (in-vocabulary
    English with stable topic), inserts a single specific factual claim
    (identifier + mass at tagging), and asks for both fields by name.
    """
    print("\n[6/6] Long-context needle (1K / 10K / 50K / 90K)")
    if skip:
        print("  SKIPPED (--skip-needle flag)")
        return {"status": "skipped"}
    found = 0
    details = []
    expected_id = "BANDIT-K9-7821"
    expected_mass = "142"
    for ctx_target in (1024, 10240, 51200, 92160):
        # Filler ≈ ctx_target tokens (~4 chars/token rough estimate)
        target_chars = ctx_target * 4
        block = _NEEDLE_STORY_TEMPLATE
        fill_count = max(1, target_chars // len(block))
        filler = block * fill_count
        # Insert needle at depth 0.5 (middle of context)
        insert_at = len(filler) // 2
        prompt = (
            filler[:insert_at] + "\n\n" + _NEEDLE_FACT + "\n\n" +
            filler[insert_at:] + "\n\n" + _NEEDLE_QUESTION
        )
        resp = _api_post({
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            # Qwen3 thinking mode can burn 200+ tokens reasoning before content;
            # also disable thinking via chat_template_kwargs so the answer
            # lands in content directly. Both belt-and-suspenders.
            "max_tokens": 600, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }, timeout=300.0)
        if resp is None:
            print(f"  ctx={ctx_target:6d}t  API error")
            details.append({"ctx": ctx_target, "found": False, "reason": "api_error"})
            continue
        ch = resp["choices"][0]
        out = ((ch["message"].get("content") or "") +
               (ch["message"].get("reasoning") or "")).upper()
        # Score: both expected_id AND expected_mass → FOUND; one → PARTIAL; none → MISS
        id_hit = expected_id.upper() in out or "K9-7821" in out or "K9 7821" in out
        mass_hit = expected_mass in out
        if id_hit and mass_hit:
            found += 1
            mark = "FOUND"
        elif id_hit or mass_hit:
            mark = f"PARTIAL ({'id' if id_hit else 'mass'} only)"
        else:
            mark = "MISS"
        print(f"  ctx={ctx_target:6d}t  {mark}")
        details.append({
            "ctx": ctx_target,
            "id_hit": id_hit, "mass_hit": mass_hit,
            "found": id_hit and mass_hit,
        })
    return {"status": "ok", "found": found, "total": 4, "details": details}


# ─── Composer ───────────────────────────────────────────────────────────


def render_markdown_block(results: dict) -> str:
    """Render the bench results as a README-ready markdown block."""
    lines = []
    lines.append(f"### Genesis comprehensive bench — {MODEL}")
    lines.append("")
    lines.append(f"_Measured 2026-05-05 on {SSH_HOST.split('@')[-1]}, vLLM 0.20.2rc1.dev9+g01d4d1ad3, "
                 f"Genesis v7.70-dev (live PROD config), 2× RTX A5000_")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")

    # 1. Cold-warm latency
    cw = results.get("cold_warm_latency", {})
    if cw.get("status") == "ok":
        lines.append(f"| Cold-warm latency (400t single-shot, trimmed mean of 5) | **{cw['trimmed_mean']:.2f}s** "
                     f"(min {cw['min']:.2f}s, max {cw['max']:.2f}s) |")

    # 2. Sustained TPS
    tps = results.get("sustained_tps", {})
    if tps.get("status") == "ok":
        lines.append(f"| Sustained generation TPS (10×400t mean) | **{tps['mean_tps']:.1f} tok/s** "
                     f"(CV {tps['cv_pct']:.2f}%) |")
        lines.append(f"| TPS range | min {tps['min_tps']:.1f} – max {tps['max_tps']:.1f} tok/s |")

    # 3. Tool-call
    tc = results.get("tool_call", {})
    if tc.get("status") == "ok":
        lines.append(f"| Tool-call clean rate (10 prompts) | **{tc['clean']}/{tc['total']} "
                     f"({tc['rate_pct']:.0f}%)** |")

    # 4. Multi-turn
    mt = results.get("multi_turn", {})
    if mt.get("status") in ("ok", "partial"):
        lines.append(f"| Multi-turn stability (10 turns) | **{mt['survived']}/{mt['total']}** "
                     f"survived, avg {mt['mean_latency']:.1f}s |")

    # 5. VRAM
    vr = results.get("vram", {})
    if vr.get("status") == "ok":
        per = " / ".join(f"{m} MiB" for m in vr["vram_mib_per_gpu"])
        lines.append(f"| VRAM steady-state (per GPU) | {per} (total **{vr['total_mib']} MiB**) |")

    # 6. Needle
    nd = results.get("needle", {})
    if nd.get("status") == "ok":
        lines.append(f"| Long-context needle (1K / 10K / 50K / 90K, depth 0.5) | "
                     f"**{nd['found']}/{nd['total']}** found |")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Genesis comprehensive benchmark for README data",
    )
    parser.add_argument("--skip-needle", action="store_true",
                        help="skip stage 6 (long-context needle)")
    parser.add_argument("--turns", type=int, default=10,
                        help="number of turns for multi-turn stability bench")
    parser.add_argument("--out", type=str, default=None,
                        help="path to write markdown block")
    args = parser.parse_args()

    print(f"═══════════════════════════════════════════════════════════════════════")
    print(f"  Genesis comprehensive bench")
    print(f"═══════════════════════════════════════════════════════════════════════")
    print(f"  Endpoint: {ENDPOINT}")
    print(f"  Model:    {MODEL}")
    print(f"  Skip needle: {args.skip_needle}")

    results: dict[str, Any] = {}
    results["cold_warm_latency"] = bench_cold_warm_latency()
    results["sustained_tps"] = bench_sustained_tps()
    results["tool_call"] = bench_tool_call_clean_rate()
    results["multi_turn"] = bench_multi_turn_stability(args.turns)
    results["vram"] = bench_vram_steady()
    results["needle"] = bench_needle_ladder(skip=args.skip_needle)

    print(f"\n═══════════════════════════════════════════════════════════════════════")
    print(f"  README-ready markdown block:")
    print(f"═══════════════════════════════════════════════════════════════════════\n")

    md = render_markdown_block(results)
    print(md)

    if args.out:
        with open(args.out, "w") as f:
            f.write(md + "\n")
        print(f"\nWrote markdown to: {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
