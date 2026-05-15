#!/usr/bin/env python3
"""Multi-concurrency throughput + latency bench for vllm OpenAI endpoint.

Measures across N concurrent requests:
  - Aggregate throughput (tokens/sec server-side)
  - Per-request TPS (each user's perceived rate)
  - TTFT (time to first token, streaming)
  - TPOT (time per output token, streaming, after first)
  - P50/P95 percentiles

Use this to validate that PN119 GQA-head grouping kernel and CUDA-graph
capture at batch>1 are actually firing.

Example:
  python3 tools/multi_conc_bench.py 8 3 384  # 8 concurrent, 3 rounds, 384 max_tok

  python3 tools/multi_conc_bench.py sweep    # full concurrency sweep 1/2/4/8
"""
import asyncio
import aiohttp
import time
import statistics
import sys
import json

URL = "http://localhost:8000/v1/chat/completions"
API_KEY = "genesis-local"

PROMPTS = [
    "Write a 200-word Python function that computes prime factorization.",
    "Explain TCP/IP four-layer model. Be detailed.",
    "Describe how RAFT consensus algorithm works in 200 words.",
    "Write a Python function that implements quicksort with explanation.",
    "What is monad in functional programming? Examples in Haskell.",
    "Compare Mutex vs Semaphore in 200 words with examples.",
    "Explain B-tree indexing in databases.",
    "Describe how a SAT solver works at high level.",
    "What is the CAP theorem? Examples of CA / CP / AP systems.",
    "Explain how Kubernetes scheduler picks a node.",
    "Write a recursive descent parser for arithmetic.",
    "Describe lock-free queue implementation.",
    "Explain protein folding briefly.",
    "What is JIT compilation? Examples in PyPy.",
    "Explain Bloom filters with use cases.",
    "Describe Paxos consensus step by step.",
]


async def stream_req(session, prompt, max_tok):
    body = {
        "model": "qwen3.6-35b-a3b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tok, "temperature": 0.7, "stream": True,
    }
    t0 = time.perf_counter()
    ttft, n_tokens, first_t, last_t = None, 0, None, t0
    async with session.post(URL, json=body,
                            headers={"Authorization": f"Bearer {API_KEY}"}) as resp:
        async for line in resp.content:
            if line.startswith(b"data: "):
                now = time.perf_counter()
                chunk = line[6:].strip()
                if chunk == b"[DONE]":
                    break
                try:
                    j = json.loads(chunk)
                except Exception:
                    continue
                delta = j.get("choices", [{}])[0].get("delta", {})
                tok = delta.get("content") or delta.get("reasoning")
                if tok:
                    n_tokens += 1
                    if first_t is None:
                        first_t = now
                        ttft = now - t0
                    last_t = now
    t1 = time.perf_counter()
    total = t1 - t0
    tpot_ms = ((last_t - first_t) * 1000 / max(1, n_tokens - 1)) if first_t and n_tokens > 1 else None
    return {"total_s": total, "ttft_ms": ttft * 1000 if ttft else None,
            "tokens": n_tokens, "tpot_ms": tpot_ms,
            "tps_per_req": n_tokens / total if total > 0 else 0}


async def non_stream_req(session, prompt, max_tok):
    body = {
        "model": "qwen3.6-35b-a3b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tok, "temperature": 0.7, "stream": False,
    }
    t0 = time.perf_counter()
    async with session.post(URL, json=body,
                            headers={"Authorization": f"Bearer {API_KEY}"}) as resp:
        data = await resp.json()
    t1 = time.perf_counter()
    n_tokens = data.get("usage", {}).get("completion_tokens", 0)
    total = t1 - t0
    return {"total_s": total, "tokens": n_tokens,
            "tps_per_req": n_tokens / total if total > 0 else 0}


async def run_round(session, n_conc, mode, max_tok):
    fn = stream_req if mode == "stream" else non_stream_req
    tasks = [fn(session, PROMPTS[i % len(PROMPTS)], max_tok) for i in range(n_conc)]
    t0 = time.perf_counter()
    results = await asyncio.gather(*tasks)
    t1 = time.perf_counter()
    return {"wall_s": t1 - t0,
            "tokens": sum(r["tokens"] for r in results),
            "aggregate_tps": sum(r["tokens"] for r in results) / (t1 - t0),
            "results": results}


def pct(data, p):
    if not data:
        return None
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


async def sweep_main():
    async with aiohttp.ClientSession() as session:
        print("== Concurrency sweep — non-streaming aggregate (warm) ==")
        for n in [1, 2, 4, 8]:
            await run_round(session, n, "non_stream", 384)
            agg = [(await run_round(session, n, "non_stream", 384))["aggregate_tps"] for _ in range(2)]
            print(f"  conc={n:2d}  agg_TPS={statistics.mean(agg):6.1f}")
        print()
        print("== Concurrency sweep — streaming (TTFT/TPOT, warm) ==")
        for n in [1, 2, 4, 8]:
            await run_round(session, n, "stream", 384)  # warmup
            ttfts, tpots, aggs = [], [], []
            for _ in range(2):
                rnd = await run_round(session, n, "stream", 384)
                aggs.append(rnd["aggregate_tps"])
                for r in rnd["results"]:
                    if r["ttft_ms"]:
                        ttfts.append(r["ttft_ms"])
                    if r["tpot_ms"]:
                        tpots.append(r["tpot_ms"])
            print(f"  conc={n:2d}  agg_TPS={statistics.mean(aggs):6.1f}  "
                  f"TTFT_med={statistics.median(ttfts):4.0f}ms  TTFT_P95={pct(ttfts, 95):4.0f}ms  "
                  f"TPOT_med={statistics.median(tpots):5.2f}ms  TPOT_P95={pct(tpots, 95):5.2f}ms")


async def single_main(n_conc, n_rounds, max_tok):
    async with aiohttp.ClientSession() as session:
        all_ttft, all_tpot, all_per_req, agg = [], [], [], []
        for i in range(n_rounds):
            r = await run_round(session, n_conc, "stream", max_tok)
            agg.append(r["aggregate_tps"])
            for r2 in r["results"]:
                if r2["ttft_ms"]:
                    all_ttft.append(r2["ttft_ms"])
                if r2["tpot_ms"]:
                    all_tpot.append(r2["tpot_ms"])
                all_per_req.append(r2["tps_per_req"])
            ttft_m = statistics.median([x["ttft_ms"] for x in r["results"] if x["ttft_ms"]]) if any(x["ttft_ms"] for x in r["results"]) else 0
            tpot_m = statistics.median([x["tpot_ms"] for x in r["results"] if x["tpot_ms"]]) if any(x["tpot_ms"] for x in r["results"]) else 0
            print(f"round {i+1}/{n_rounds}: agg_TPS={r['aggregate_tps']:.1f} "
                  f"ttft_med={ttft_m:.0f}ms tpot_med={tpot_m:.2f}ms wall={r['wall_s']:.2f}s")
        warm_agg = agg[1:] if len(agg) > 1 else agg
        warm_ttft = all_ttft[n_conc:] if n_rounds > 1 else all_ttft
        warm_tpot = all_tpot[n_conc:] if n_rounds > 1 else all_tpot
        warm_per = all_per_req[n_conc:] if n_rounds > 1 else all_per_req
        print(f"\n=== SUMMARY (n_conc={n_conc}, max_tok={max_tok}) ===")
        if warm_agg:
            print(f"  agg_TPS         mean={statistics.mean(warm_agg):.1f}")
        if warm_per:
            print(f"  per-req TPS     mean={statistics.mean(warm_per):.1f}")
        if warm_ttft:
            print(f"  TTFT_ms         median={statistics.median(warm_ttft):.0f}  P95={pct(warm_ttft, 95):.0f}")
        if warm_tpot:
            print(f"  TPOT_ms         median={statistics.median(warm_tpot):.2f}  P95={pct(warm_tpot, 95):.2f}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        asyncio.run(sweep_main())
    else:
        n_conc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
        n_rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        max_tok = int(sys.argv[3]) if len(sys.argv) > 3 else 384
        asyncio.run(single_main(n_conc, n_rounds, max_tok))


if __name__ == "__main__":
    main()
