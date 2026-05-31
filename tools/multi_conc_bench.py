#!/usr/bin/env python3
"""Multi-concurrency throughput + latency bench for vllm OpenAI endpoint.

Measures across N concurrent requests:
  - Aggregate throughput (tokens/sec server-side) — non-streaming
  - Per-request TPS (each user's perceived rate)
  - TTFT (time to first token, streaming)
  - TPOT (time per output token, streaming, after first)
  - P50/P95 percentiles

Use this to validate that PN119 GQA-head grouping kernel and CUDA-graph
capture at batch>1 are actually firing.

Examples:
  # Concurrency sweep — stdout-only quick recon
  python3 tools/multi_conc_bench.py sweep

  # Single-concurrency baseline capture with committed JSON
  python3 tools/multi_conc_bench.py \
      --model qwen3.6-35b-a3b --conc 8 --rounds 5 --max-tok 1024 \
      --out tools/bench_results/prod-qwen3.6-35b-multiconc_<pin>_<date>.json

JSON output (when ``--out`` is set) carries a ``decode_bench`` block
with field names compatible with ``scripts/attach_bench_proof.py``:
``wall_TPS`` (= non-stream aggregate TPS), ``decode_TPOT_ms`` (=
stream per-request TPOT), ``TTFT_ms`` (= stream per-request TTFT).
The richer per-mode breakdown lives in the ``multi_conc`` sub-block.
"""
import argparse
import asyncio
import aiohttp
import datetime as _dt
import json
import statistics
import sys
import time

URL = "http://localhost:8000/v1/chat/completions"
API_KEY = "genesis-local"
DEFAULT_MODEL = "qwen3.6-35b-a3b"

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


async def stream_req(session, model, prompt, max_tok):
    body = {
        "model": model,
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


async def non_stream_req(session, model, prompt, max_tok):
    body = {
        "model": model,
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


async def run_round(session, model, n_conc, mode, max_tok):
    fn = stream_req if mode == "stream" else non_stream_req
    tasks = [fn(session, model, PROMPTS[i % len(PROMPTS)], max_tok)
             for i in range(n_conc)]
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


def _stats(samples):
    """Return canonical {mean, std, cv, min, max, n} dict for a sample list."""
    if not samples:
        return {"mean": None, "std": None, "cv": None, "min": None, "max": None, "n": 0}
    n = len(samples)
    mean = statistics.mean(samples)
    std = statistics.stdev(samples) if n > 1 else 0.0
    cv = (std / mean) if mean else None
    return {
        "mean": round(mean, 4),
        "std":  round(std, 4),
        "cv":   round(cv, 4) if cv is not None else None,
        "min":  round(min(samples), 4),
        "max":  round(max(samples), 4),
        "n":    n,
    }


async def sweep_main():
    async with aiohttp.ClientSession() as session:
        print("== Concurrency sweep — non-streaming aggregate (warm) ==")
        for n in [1, 2, 4, 8]:
            await run_round(session, DEFAULT_MODEL, n, "non_stream", 384)
            agg = [(await run_round(session, DEFAULT_MODEL, n, "non_stream", 384))["aggregate_tps"]
                   for _ in range(2)]
            print(f"  conc={n:2d}  agg_TPS={statistics.mean(agg):6.1f}")
        print()
        print("== Concurrency sweep — streaming (TTFT/TPOT, warm) ==")
        for n in [1, 2, 4, 8]:
            await run_round(session, DEFAULT_MODEL, n, "stream", 384)
            ttfts, tpots, aggs = [], [], []
            for _ in range(2):
                rnd = await run_round(session, DEFAULT_MODEL, n, "stream", 384)
                aggs.append(rnd["aggregate_tps"])
                for r in rnd["results"]:
                    if r["ttft_ms"]:
                        ttfts.append(r["ttft_ms"])
                    if r["tpot_ms"]:
                        tpots.append(r["tpot_ms"])
            print(f"  conc={n:2d}  agg_TPS={statistics.mean(aggs):6.1f}  "
                  f"TTFT_med={statistics.median(ttfts):4.0f}ms  TTFT_P95={pct(ttfts, 95):4.0f}ms  "
                  f"TPOT_med={statistics.median(tpots):5.2f}ms  TPOT_P95={pct(tpots, 95):5.2f}ms")


async def single_main(model, n_conc, n_rounds, max_tok, out_path):
    started_iso = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    rounds_non_stream = []
    rounds_stream = []
    async with aiohttp.ClientSession() as session:
        # Non-streaming rounds — headline aggregate TPS.
        for i in range(n_rounds):
            r = await run_round(session, model, n_conc, "non_stream", max_tok)
            rounds_non_stream.append({
                "round": i + 1,
                "aggregate_tps": round(r["aggregate_tps"], 4),
                "wall_s": round(r["wall_s"], 4),
                "tokens": r["tokens"],
            })
            print(f"non_stream round {i+1}/{n_rounds}: "
                  f"agg_TPS={r['aggregate_tps']:.1f} wall={r['wall_s']:.2f}s")
        # Streaming rounds — TTFT/TPOT measurement.
        for i in range(n_rounds):
            r = await run_round(session, model, n_conc, "stream", max_tok)
            per_round_ttft = [x["ttft_ms"] for x in r["results"] if x["ttft_ms"]]
            per_round_tpot = [x["tpot_ms"] for x in r["results"] if x["tpot_ms"]]
            per_round_per_req = [x["tps_per_req"] for x in r["results"]]
            rounds_stream.append({
                "round": i + 1,
                "aggregate_tps": round(r["aggregate_tps"], 4),
                "wall_s": round(r["wall_s"], 4),
                "ttft_ms": per_round_ttft,
                "tpot_ms": per_round_tpot,
                "tps_per_req": per_round_per_req,
            })
            ttft_m = statistics.median(per_round_ttft) if per_round_ttft else 0
            tpot_m = statistics.median(per_round_tpot) if per_round_tpot else 0
            print(f"stream     round {i+1}/{n_rounds}: "
                  f"agg_TPS={r['aggregate_tps']:.1f} "
                  f"ttft_med={ttft_m:.0f}ms tpot_med={tpot_m:.2f}ms wall={r['wall_s']:.2f}s")

    # Drop first round of each mode as warmup if n_rounds > 1.
    warm_ns = rounds_non_stream[1:] if n_rounds > 1 else rounds_non_stream
    warm_s = rounds_stream[1:] if n_rounds > 1 else rounds_stream

    agg_ns = [r["aggregate_tps"] for r in warm_ns]
    agg_s = [r["aggregate_tps"] for r in warm_s]
    all_ttft = [t for r in warm_s for t in r["ttft_ms"]]
    all_tpot = [t for r in warm_s for t in r["tpot_ms"]]
    all_per_req = [t for r in warm_s for t in r["tps_per_req"]]

    summary = {
        "agg_TPS_non_stream": _stats(agg_ns),
        "agg_TPS_stream":     _stats(agg_s),
        "per_req_TPS":        _stats(all_per_req),
        "ttft_ms":            {**_stats(all_ttft),
                               "p50": pct(all_ttft, 50), "p95": pct(all_ttft, 95)},
        "tpot_ms":            {**_stats(all_tpot),
                               "p50": pct(all_tpot, 50), "p95": pct(all_tpot, 95)},
    }

    print(f"\n=== SUMMARY (model={model} conc={n_conc} max_tok={max_tok}) ===")
    print(f"  agg_TPS  non_stream   mean={summary['agg_TPS_non_stream']['mean']}  "
          f"cv={summary['agg_TPS_non_stream']['cv']}  n={summary['agg_TPS_non_stream']['n']}")
    print(f"  agg_TPS  stream       mean={summary['agg_TPS_stream']['mean']}  "
          f"cv={summary['agg_TPS_stream']['cv']}  n={summary['agg_TPS_stream']['n']}")
    print(f"  per-req TPS           mean={summary['per_req_TPS']['mean']}  "
          f"n={summary['per_req_TPS']['n']}")
    print(f"  TTFT_ms               p50={summary['ttft_ms']['p50']:.0f}  "
          f"p95={summary['ttft_ms']['p95']:.0f}")
    print(f"  TPOT_ms               p50={summary['tpot_ms']['p50']:.2f}  "
          f"p95={summary['tpot_ms']['p95']:.2f}")

    if out_path:
        # decode_bench headline keys match genesis_bench_suite.py shape so
        # scripts/attach_bench_proof.py can consume this as a baseline.
        # wall_TPS = non-stream aggregate TPS (operator policy for BENCHMARKS
        # row 25 headline). decode_TPOT_ms / TTFT_ms come from streaming since
        # non-stream cannot resolve per-token timing.
        decode_bench = {
            "wall_TPS":       summary["agg_TPS_non_stream"],
            "decode_TPOT_ms": {k: v for k, v in summary["tpot_ms"].items()
                              if k != "p50" and k != "p95"},
            "TTFT_ms":        {k: v for k, v in summary["ttft_ms"].items()
                              if k != "p50" and k != "p95"},
        }
        out = {
            "suite_version": "multi_conc_bench/1",
            "name": "multi_conc_bench",
            "started": started_iso,
            "mode": "multi_conc",
            "host": "localhost",
            "port": 8000,
            "model": model,
            "config": {
                "concurrency":  n_conc,
                "n_rounds":     n_rounds,
                "max_tokens":   max_tok,
                "warmup_round": 1 if n_rounds > 1 else 0,
            },
            "decode_bench": decode_bench,
            "multi_conc": {
                "concurrency": n_conc,
                "max_tokens":  max_tok,
                "non_stream": {
                    "rounds_all":  rounds_non_stream,
                    "rounds_warm": warm_ns,
                    "agg_TPS":     summary["agg_TPS_non_stream"],
                },
                "stream": {
                    "rounds_all":  rounds_stream,
                    "rounds_warm": warm_s,
                    "agg_TPS":     summary["agg_TPS_stream"],
                    "per_req_TPS": summary["per_req_TPS"],
                    "ttft_ms":     summary["ttft_ms"],
                    "tpot_ms":     summary["tpot_ms"],
                },
            },
        }
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nJSON: {out_path}")


def _parse_args(argv):
    p = argparse.ArgumentParser(
        description=("Multi-concurrency throughput + latency bench. "
                     "Default mode runs --conc parallel requests for --rounds "
                     "rounds (each: non-stream + stream); first round dropped "
                     "as warmup. Pass `sweep` as the first positional arg for "
                     "the legacy concurrency sweep."))
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"served-model-name to send in request body "
                        f"(default: {DEFAULT_MODEL})")
    p.add_argument("--conc", type=int, default=8,
                   help="concurrency level for single-mode (default: 8)")
    p.add_argument("--rounds", type=int, default=5,
                   help="rounds per mode (non_stream + stream); first round "
                        "dropped as warmup if rounds>1 (default: 5)")
    p.add_argument("--max-tok", type=int, default=1024, dest="max_tok",
                   help="max_tokens per request (default: 1024)")
    p.add_argument("--out", default=None,
                   help="optional JSON output path "
                        "(genesis_bench_suite-compatible decode_bench block)")
    return p.parse_args(argv)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        asyncio.run(sweep_main())
        return
    args = _parse_args(sys.argv[1:])
    asyncio.run(single_main(args.model, args.conc, args.rounds,
                            args.max_tok, args.out))


if __name__ == "__main__":
    main()
