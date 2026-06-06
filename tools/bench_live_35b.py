#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Quick 35B bench: 5 prompts × 3 runs, measure TPS / TTFT / TPOT.

Run on server::

    python3 tools/bench_live_35b.py

Targets: a live vllm-qwen3.6-35b-balanced-k3 on port 8102 with
``--api-key genesis-local``. Streaming SSE is parsed for per-token timing.
"""
from __future__ import annotations

import json
import statistics
import time
import urllib.request

BASE = "http://localhost:8102"
KEY = "genesis-local"
MODEL = "qwen3.6-35b-a3b"

PROMPTS = [
    "Explain quicksort in one paragraph.",
    "Write a Python function to compute fibonacci(n).",
    "What is the difference between gRPC and REST?",
    "Compose a haiku about distributed systems.",
    "List 5 design patterns and their use cases.",
]
MAX_TOKENS = 200
N_RUNS = 3


def bench_one(prompt: str, run_idx: int) -> dict:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.3,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
        },
    )
    t_start = time.perf_counter()
    t_first = None
    n_tokens = 0
    last_token_time = None
    inter_token_times = []
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content")
            if content:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                n_tokens += 1
                if last_token_time is not None:
                    inter_token_times.append(now - last_token_time)
                last_token_time = now
    t_end = time.perf_counter()
    tps = round(n_tokens / (t_end - t_first), 2) if t_first and t_end > t_first else None
    return {
        "run": run_idx,
        "prompt": prompt[:30],
        "tokens": n_tokens,
        "wall_s": round(t_end - t_start, 3),
        "ttft_ms": round((t_first - t_start) * 1000, 1) if t_first else None,
        "tpot_ms": round(statistics.mean(inter_token_times) * 1000, 2) if inter_token_times else None,
        "tps": tps,
    }


def stats(name: str, vals: list[float]) -> None:
    vals = [v for v in vals if v is not None]
    if not vals:
        print(f"  {name}: n/a")
        return
    sd = statistics.stdev(vals) if len(vals) > 1 else 0
    print(f"  {name}: mean={statistics.mean(vals):.2f}  "
          f"median={statistics.median(vals):.2f}  "
          f"stdev={sd:.2f}  min={min(vals):.2f}  max={max(vals):.2f}  n={len(vals)}")


def main() -> None:
    print("=== Warmup ===")
    bench_one("Hi", 0)

    print("=== Bench ===")
    results = []
    for run in range(N_RUNS):
        for prompt in PROMPTS:
            r = bench_one(prompt, run)
            results.append(r)
            p = r["prompt"]
            print(f"  run={run} \"{p}\" wall={r['wall_s']}s tokens={r['tokens']} "
                  f"ttft={r['ttft_ms']}ms tpot={r['tpot_ms']}ms tps={r['tps']}")

    print()
    print("=== Summary ===")
    stats("wall_TPS", [r["tps"] for r in results])
    stats("TTFT_ms", [r["ttft_ms"] for r in results])
    stats("TPOT_ms", [r["tpot_ms"] for r in results])
    total = sum(r["tokens"] for r in results)
    print(f"  Total tokens generated: {total}")


if __name__ == "__main__":
    main()
