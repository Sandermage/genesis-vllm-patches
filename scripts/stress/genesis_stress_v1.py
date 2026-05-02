#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Genesis stress test harness — multi-dimensional perf + correctness eval.

Runs against any vllm-compatible OpenAI server. Defaults match the
Genesis dev rig but every value is overridable via env var or CLI flag:

  GENESIS_STRESS_URL    — server URL (default `http://localhost:8000`)
  GENESIS_STRESS_MODEL  — `served-model-name` (default `qwen3.6-35b-a3b`)
  GENESIS_STRESS_KEY    — API key (default `genesis-local`)

CLI flags `--url`, `--model`, `--key` override env. Rig-specific
defaults like `192.168.1.10` were removed (G-010 audit fix 2026-05-02);
operators on a different host just `export GENESIS_STRESS_URL=...`
or pass `--url`.

Categories tested:
  1. Stability  — 100 sustained requests, 1024 tok each, count failures
  2. Speed      — TPS scan over output sizes {512, 1024, 2048, 4096}
  3. CV         — 30 deterministic temp=0 runs to characterize variance
  4. TTFT       — first-token latency on prompt sizes {1K, 8K, 32K, 100K} chars
  5. Long-ctx   — full prompt at {50K, 100K, 200K} char prompts
  6. Tool-call  — minimal Hermes-style tool-call rate verification

Each category produces a numerical metric. Output is structured JSON suitable
for diff against a baseline. Designed so that two runs (baseline + post-patch)
can be diffed cleanly to detect regressions.

Author: Sandermage (Sander) Barzov Aleksandr.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request

import os

# G-010 audit fix (2026-05-02): no rig-specific IPs in source; env override.
DEFAULT_URL = os.environ.get("GENESIS_STRESS_URL", "http://localhost:8000")
DEFAULT_MODEL = os.environ.get("GENESIS_STRESS_MODEL", "qwen3.6-35b-a3b")
DEFAULT_KEY = os.environ.get("GENESIS_STRESS_KEY", "genesis-local")


def _post_chat(
    url: str, model: str, key: str,
    prompt: str, max_tokens: int,
    *,
    temperature: float = 0.0,
    stream: bool = False,
    timeout: float = 600.0,
) -> dict:
    """POST a single chat completion. Returns parsed JSON or raises."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "stream": stream,
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _wait_ready(url: str, key: str, timeout_s: int = 600) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                f"{url}/v1/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            urllib.request.urlopen(req, timeout=5)
            return True
        except Exception:
            time.sleep(2)
    return False


# ─── Test categories ───────────────────────────────────────────────────────


def test_stability(url, model, key, n=100) -> dict:
    """Run N requests sequentially, count failures and exceptions."""
    p = "Write 1024 tokens of detailed technical analysis on attention mechanisms."
    failures = 0
    exceptions = 0
    durations = []
    t0 = time.time()
    for i in range(n):
        try:
            j = _post_chat(url, model, key, p, 1024)
            if j["usage"]["completion_tokens"] < 100:
                failures += 1
        except Exception as e:
            exceptions += 1
            print(f"  stability run {i} EXCEPTION: {type(e).__name__}: {e}",
                  file=sys.stderr)
        durations.append(time.time() - t0)
    return {
        "n": n,
        "failures": failures,
        "exceptions": exceptions,
        "wall_clock_s": round(time.time() - t0, 2),
        "success_rate": round((n - failures - exceptions) / n, 4),
    }


def test_speed_scan(url, model, key) -> dict:
    """TPS scan across output sizes."""
    p = ("Write a thorough technical analysis on transformer attention, "
         "covering MTP, FlashAttention, KV cache, and cudagraph capture.")
    out_sizes = [512, 1024, 2048, 4096]
    results = {}
    for n in out_sizes:
        # 2 warmup + 3 measure
        for _ in range(2):
            _post_chat(url, model, key, p, n)
        runs = []
        for _ in range(3):
            t0 = time.time()
            j = _post_chat(url, model, key, p, n)
            dt = time.time() - t0
            tok = j["usage"]["completion_tokens"]
            runs.append(tok / dt)
        results[f"out_{n}"] = {
            "mean_tps": round(statistics.mean(runs), 2),
            "median_tps": round(statistics.median(runs), 2),
            "cv_pct": round(
                statistics.stdev(runs) / statistics.mean(runs) * 100, 2
            ) if len(runs) > 1 else 0.0,
        }
    return results


def test_cv(url, model, key, n=30) -> dict:
    """Long-running CV characterization at temp=0 deterministic."""
    p = ("Write 1024 tokens of detailed technical analysis on transformer "
         "attention mechanisms, covering FlashAttention, KV cache, MTP "
         "speculation, GQA, and cudagraph capture in vLLM. Be thorough.")
    # 3 warmup
    for _ in range(3):
        _post_chat(url, model, key, p, 1024)
    runs = []
    for _ in range(n):
        t0 = time.time()
        j = _post_chat(url, model, key, p, 1024)
        dt = time.time() - t0
        tok = j["usage"]["completion_tokens"]
        runs.append(tok / dt)
    return {
        "n": n,
        "mean": round(statistics.mean(runs), 2),
        "median": round(statistics.median(runs), 2),
        "stdev": round(statistics.stdev(runs), 3),
        "cv_pct": round(
            statistics.stdev(runs) / statistics.mean(runs) * 100, 2
        ),
        "min": round(min(runs), 2),
        "max": round(max(runs), 2),
    }


def test_ttft(url, model, key) -> dict:
    """TTFT (time to first token) across prompt sizes via streaming."""
    base = ("In the deep learning landscape of 2026, transformer architectures "
            "continue to evolve with novel attention and quantization. ")
    sizes_chars = {"1k": 1000, "8k": 8000, "32k": 32000, "100k": 100000}
    results = {}
    for label, char_n in sizes_chars.items():
        prompt = (base * (char_n // len(base) + 1))[:char_n] + "\n\nSummarize."
        # 1 warmup
        try:
            _post_chat(url, model, key, prompt, 32, timeout=120)
        except Exception as e:
            results[label] = {"error": str(e)[:100]}
            continue
        # 3 streaming measures
        ttfts = []
        for _ in range(3):
            body = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 32,
                "temperature": 0.0,
                "stream": True,
            }).encode()
            req = urllib.request.Request(
                f"{url}/v1/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
            )
            t0 = time.time()
            try:
                resp = urllib.request.urlopen(req, timeout=120)
                for line in resp:
                    if line.startswith(b"data: ") and b"content" in line:
                        ttfts.append(time.time() - t0)
                        resp.close()
                        break
            except Exception:
                ttfts.append(float("nan"))
        finite = [t for t in ttfts if not (t != t)]
        results[label] = {
            "mean_ttft_s": round(statistics.mean(finite), 3) if finite else None,
            "median_ttft_s": round(statistics.median(finite), 3) if finite else None,
            "n_ok": len(finite),
        }
    return results


def test_long_ctx(url, model, key) -> dict:
    """Validate model handles long prompts without hang/crash; report TPS."""
    base = ("In the deep learning landscape of 2026, transformer architectures "
            "continue to evolve with novel attention and quantization. ")
    sizes_chars = {"50k": 50000, "100k": 100000, "200k": 200000}
    results = {}
    for label, char_n in sizes_chars.items():
        prompt = (base * (char_n // len(base) + 1))[:char_n] + "\n\nSummarize concisely."
        try:
            t0 = time.time()
            j = _post_chat(url, model, key, prompt, 256, timeout=600)
            dt = time.time() - t0
            in_tok = j["usage"]["prompt_tokens"]
            out_tok = j["usage"]["completion_tokens"]
            results[label] = {
                "in_tok": in_tok,
                "out_tok": out_tok,
                "wall_s": round(dt, 2),
                "tps": round(out_tok / dt, 2),
            }
        except Exception as e:
            results[label] = {"error": f"{type(e).__name__}: {str(e)[:100]}"}
    return results


def test_tool_call(url, model, key) -> dict:
    """OpenAI-tools API style tool-call rate test (qwen3_coder-compatible).

    Uses the OpenAI `tools` array so vllm's tool-call-parser processes
    model output correctly regardless of model-specific format (qwen3_coder
    XML, Hermes JSON, etc.). Pass criteria: model emits structured tool_call
    in response.choices[0].message.tool_calls, OR mentions get_weather in
    content (loose fallback).
    """
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'Paris'",
                    },
                },
                "required": ["city"],
            },
        },
    }]
    queries = [
        "What is the weather in Paris?",
        "How is the weather looking in Tokyo right now?",
        "Tell me the current weather in San Francisco.",
        "I want to know weather in Moscow, please.",
        "What's the temperature outside in New York?",
    ]
    used = 0
    failed = 0
    samples = []
    for q in queries:
        try:
            body = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": q}],
                "tools": tools,
                "tool_choice": "auto",
                "max_tokens": 256,
                "temperature": 0.0,
            }).encode()
            req = urllib.request.Request(
                f"{url}/v1/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
            )
            j = json.loads(urllib.request.urlopen(req, timeout=120).read())
            msg = j["choices"][0]["message"]
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                used += 1
                samples.append({"tool_calls": len(tool_calls)})
            elif "get_weather" in content or "weather" in content.lower()[:120]:
                # loose fallback for models that emit weather info in content
                samples.append({"content_excerpt": content[:80]})
        except Exception as e:
            failed += 1
            samples.append({"error": str(e)[:60]})
    return {
        "n_queries": len(queries),
        "tool_used": used,
        "failed": failed,
        "tool_use_rate": round(used / len(queries), 4),
        "samples": samples,
    }


# ─── Main ──────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--key", default=DEFAULT_KEY)
    ap.add_argument("--out", default="-",
                    help="output JSON file (default: stdout)")
    ap.add_argument("--skip", default="",
                    help="comma-separated category list to skip "
                         "(stability,speed,cv,ttft,long_ctx,tool_call)")
    ap.add_argument("--stability-n", type=int, default=100)
    ap.add_argument("--cv-n", type=int, default=30)
    args = ap.parse_args()
    skip = set(s.strip() for s in args.skip.split(",") if s.strip())

    print(f"[stress] target {args.url} model {args.model}", file=sys.stderr)
    if not _wait_ready(args.url, args.key):
        print("[stress] server not ready", file=sys.stderr)
        sys.exit(1)

    report = {
        "model": args.model,
        "url": args.url,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "categories": {},
    }

    for name, fn, kwargs in [
        ("speed", test_speed_scan, {}),
        ("cv", test_cv, {"n": args.cv_n}),
        ("ttft", test_ttft, {}),
        ("long_ctx", test_long_ctx, {}),
        ("tool_call", test_tool_call, {}),
        ("stability", test_stability, {"n": args.stability_n}),
    ]:
        if name in skip:
            print(f"[stress] skip {name}", file=sys.stderr)
            continue
        print(f"[stress] running {name} ...", file=sys.stderr)
        t0 = time.time()
        try:
            report["categories"][name] = fn(args.url, args.model, args.key, **kwargs)
            report["categories"][name]["__elapsed_s"] = round(time.time() - t0, 1)
        except Exception as e:
            report["categories"][name] = {
                "error": f"{type(e).__name__}: {str(e)[:200]}",
                "__elapsed_s": round(time.time() - t0, 1),
            }
        print(f"[stress] {name} done in {time.time() - t0:.1f}s", file=sys.stderr)

    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if args.out == "-":
        print(json.dumps(report, indent=2))
    else:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[stress] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
