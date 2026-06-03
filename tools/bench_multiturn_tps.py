#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Multi-turn TPS bench — measure per-turn wall_TPS / TTFT / decode_TPOT.

Designed for K_001 (SNDR_MTP_DYNAMIC_K_001) validation specifically:
the patch maintains per-seq SequenceState with a rolling acceptance-rate
window (len=10). The K hysteresis only triggers AFTER the window has
matured (10+ turns per session). genesis_bench_suite.py's `--ttft-turns`
mode runs single-session multi-turn but only reports TTFT — this bench
also reports per-turn wall_TPS, decode_TPOT, and aggregate stats so the
A/B comparison can see whether K_001 produces measurable wall_TPS
improvement once SequenceState matures.

Unlike bench_agentic.py this does NOT require tool-call support on the
endpoint. Plain chat-completions only. Each turn appends an assistant
+ synthetic user follow-up to the conversation history so the per-seq
state is exercised under realistic context growth.

Examples
--------
  # 20 turns × 3 sessions on 35B endpoint
  python3 tools/bench_multiturn_tps.py \\
      --url http://localhost:8102/v1 \\
      --model qwen3.6-35b-a3b \\
      --api-key genesis-local \\
      --turns 20 --sessions 3 \\
      --out /tmp/multiturn_35b_off.json

Output JSON shape
-----------------
{
  "model": "...",
  "url": "...",
  "config": {...},
  "per_turn": [
    {"turn": 1, "session": 1, "ttft_ms": ..., "decode_tps": ...,
     "decode_tpot_ms": ..., "completion_tokens": ..., "elapsed_s": ...},
    ...
  ],
  "by_session": {
    "1": {"wall_TPS_mean": ..., "TTFT_ms_mean": ..., ...},
    "2": {...}
  },
  "overall": {
    "wall_TPS": {"mean": ..., "std": ..., "cv": ..., "n": ...},
    "decode_TPOT_ms": {...},
    "TTFT_ms": {...}
  },
  "early_window": {"wall_TPS_mean": ...},  # turns 1-9 (pre-SequenceState-mature)
  "late_window":  {"wall_TPS_mean": ...},  # turns 10+ (K_001 hysteresis active)
  "k001_signal_delta_pct": ...             # (late - early) / early × 100
}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from typing import Any

import aiohttp


SYNTHETIC_FOLLOWUPS = [
    "Can you explain that in more detail?",
    "What about the edge cases?",
    "Is there a simpler way to express that?",
    "What are the tradeoffs of this approach?",
    "Can you walk through a concrete example?",
    "How does this scale with input size?",
    "What error conditions should I handle?",
    "Is there a standard library function for this?",
    "Could you compare this to the alternative approach?",
    "What testing strategy would you recommend?",
    "Are there any performance pitfalls?",
    "How would I extend this for the multi-threaded case?",
    "What logging would help debug this?",
    "Could you suggest a refactor to improve readability?",
    "What documentation should accompany this?",
    "How does this interact with caching?",
    "What about backward compatibility?",
    "Could you outline the security implications?",
    "What metrics would I track in production?",
    "How would you stage a rollout of this change?",
]


def _seed_prompt(session_idx: int) -> str:
    seeds = [
        "Write a Python function that computes the n-th Fibonacci number.",
        "Explain how a B-tree differs from a binary search tree.",
        "Sketch a producer-consumer pattern using asyncio queues.",
    ]
    return seeds[session_idx % len(seeds)]


async def _run_one_turn(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    api_key: str,
    history: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    request_timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": history,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    t_start = time.monotonic()
    ttft = None
    completion_text = ""
    completion_tokens = 0
    finish_reason: str | None = None
    error: str | None = None

    try:
        async with session.post(
            f"{url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=request_timeout),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return {
                    "error": f"HTTP {resp.status}: {body[:300]}",
                    "ttft_ms": None,
                    "decode_tps": None,
                    "decode_tpot_ms": None,
                    "wall_tps": None,
                    "completion_tokens": 0,
                    "completion_text": "",
                    "elapsed_s": time.monotonic() - t_start,
                    "finish_reason": None,
                }
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if ttft is None and chunk.get("choices"):
                    ttft = time.monotonic() - t_start
                choice = chunk["choices"][0] if chunk.get("choices") else {}
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    completion_text += content
                    completion_tokens += 1
                fr = choice.get("finish_reason")
                if fr:
                    finish_reason = fr
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    elapsed_s = time.monotonic() - t_start
    decode_s = max(elapsed_s - (ttft or 0.0), 1e-6)
    decode_tps = completion_tokens / decode_s if completion_tokens else None
    decode_tpot_ms = (decode_s / completion_tokens * 1000.0) if completion_tokens else None
    wall_tps = completion_tokens / elapsed_s if elapsed_s > 0 and completion_tokens else None

    return {
        "error": error,
        "ttft_ms": ttft * 1000.0 if ttft else None,
        "decode_tps": decode_tps,
        "decode_tpot_ms": decode_tpot_ms,
        "wall_tps": wall_tps,
        "completion_tokens": completion_tokens,
        "completion_text": completion_text,
        "elapsed_s": elapsed_s,
        "finish_reason": finish_reason,
    }


async def _run_session(
    session_idx: int,
    url: str,
    model: str,
    api_key: str,
    turns: int,
    max_tokens: int,
    temperature: float,
    request_timeout: float,
) -> list[dict[str, Any]]:
    history: list[dict[str, str]] = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": _seed_prompt(session_idx)},
    ]
    results: list[dict[str, Any]] = []
    async with aiohttp.ClientSession() as http:
        for turn in range(1, turns + 1):
            r = await _run_one_turn(
                http,
                url,
                model,
                api_key,
                history,
                max_tokens,
                temperature,
                request_timeout,
            )
            r["turn"] = turn
            r["session"] = session_idx + 1
            results.append(r)
            if r.get("error"):
                print(f"  [session {session_idx+1}] turn {turn} ERROR: {r['error'][:100]}")
                continue
            print(
                f"  [session {session_idx+1}] turn {turn}/{turns}: "
                f"ttft={r['ttft_ms']:.1f}ms  tps={r['wall_tps']:.2f}  "
                f"tok={r['completion_tokens']}  fr={r['finish_reason']}"
            )
            if r.get("completion_text"):
                history.append({"role": "assistant", "content": r["completion_text"]})
            history.append({
                "role": "user",
                "content": SYNTHETIC_FOLLOWUPS[(turn - 1) % len(SYNTHETIC_FOLLOWUPS)],
            })
    return results


def _summarize(values: list[float]) -> dict[str, float]:
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "cv": 0.0, "n": 0, "min": 0.0, "max": 0.0}
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    cv = std / mean if mean else 0.0
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "cv": round(cv, 4),
        "n": n,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def _aggregate(per_turn: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in per_turn if not r.get("error") and r.get("wall_tps")]
    wall_tps = [r["wall_tps"] for r in ok]
    tpot = [r["decode_tpot_ms"] for r in ok if r.get("decode_tpot_ms")]
    ttft = [r["ttft_ms"] for r in ok if r.get("ttft_ms")]

    by_session: dict[str, dict[str, Any]] = {}
    for r in ok:
        sk = str(r["session"])
        by_session.setdefault(sk, {"wall_tps": [], "ttft": [], "tpot": []})
        by_session[sk]["wall_tps"].append(r["wall_tps"])
        if r.get("ttft_ms"):
            by_session[sk]["ttft"].append(r["ttft_ms"])
        if r.get("decode_tpot_ms"):
            by_session[sk]["tpot"].append(r["decode_tpot_ms"])
    by_session_summary = {
        sk: {
            "wall_TPS": _summarize(d["wall_tps"]),
            "TTFT_ms": _summarize(d["ttft"]),
            "decode_TPOT_ms": _summarize(d["tpot"]),
        }
        for sk, d in by_session.items()
    }

    # K_001 SequenceState matures at turn 10 (window len=10). Compare
    # pre/post window. This is the specific signal K_001 claims to
    # produce.
    early = [r["wall_tps"] for r in ok if r["turn"] < 10]
    late = [r["wall_tps"] for r in ok if r["turn"] >= 10]
    early_mean = statistics.fmean(early) if early else 0.0
    late_mean = statistics.fmean(late) if late else 0.0
    k001_delta_pct = (
        100.0 * (late_mean - early_mean) / early_mean if early_mean else 0.0
    )

    return {
        "overall": {
            "wall_TPS": _summarize(wall_tps),
            "decode_TPOT_ms": _summarize(tpot),
            "TTFT_ms": _summarize(ttft),
        },
        "by_session": by_session_summary,
        "early_window_turns_1_9": _summarize(early),
        "late_window_turns_10plus": _summarize(late),
        "k001_signal_delta_pct": round(k001_delta_pct, 3),
        "error_count": sum(1 for r in per_turn if r.get("error")),
        "success_count": len(ok),
    }


async def _amain(args: argparse.Namespace) -> int:
    print(f"# Multi-turn TPS bench  url={args.url}  model={args.model}")
    print(f"# turns/session={args.turns}  sessions={args.sessions}  max_tok={args.max_tokens}")
    print()

    per_turn: list[dict[str, Any]] = []
    for sidx in range(args.sessions):
        print(f"== session {sidx+1}/{args.sessions} ==")
        results = await _run_session(
            session_idx=sidx,
            url=args.url,
            model=args.model,
            api_key=args.api_key,
            turns=args.turns,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            request_timeout=args.request_timeout,
        )
        per_turn.extend(results)

    agg = _aggregate(per_turn)
    print()
    print("# Summary")
    print(f"  overall wall_TPS:    {agg['overall']['wall_TPS']['mean']:.2f}  CV={agg['overall']['wall_TPS']['cv']:.4f}  n={agg['overall']['wall_TPS']['n']}")
    print(f"  overall TPOT_ms:     {agg['overall']['decode_TPOT_ms']['mean']:.2f}  CV={agg['overall']['decode_TPOT_ms']['cv']:.4f}")
    print(f"  overall TTFT_ms:     {agg['overall']['TTFT_ms']['mean']:.2f}  CV={agg['overall']['TTFT_ms']['cv']:.4f}")
    print(f"  early window (1-9):  {agg['early_window_turns_1_9']['mean']:.2f} TPS  (n={agg['early_window_turns_1_9']['n']})")
    print(f"  late window  (10+):  {agg['late_window_turns_10plus']['mean']:.2f} TPS  (n={agg['late_window_turns_10plus']['n']})")
    print(f"  K_001 signal Δ:      {agg['k001_signal_delta_pct']:+.2f}%  (late - early as fraction of early)")
    print(f"  success/error:       {agg['success_count']}/{agg['error_count']}")

    if args.out:
        out_path = args.out
        os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
        with open(out_path, "w") as f:
            json.dump({
                "model": args.model,
                "url": args.url,
                "config": {
                    "turns": args.turns,
                    "sessions": args.sessions,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                },
                "per_turn": per_turn,
                **agg,
            }, f, indent=2)
        print(f"\n# Wrote {out_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--url", default="http://localhost:8000/v1",
                   help="vLLM /v1 base URL")
    p.add_argument("--model", default="qwen3.6-35b-a3b")
    p.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "genesis-local"))
    p.add_argument("--turns", type=int, default=12,
                   help="Turns per session (>=10 needed to mature K_001 SequenceState window)")
    p.add_argument("--sessions", type=int, default=2,
                   help="Independent sessions (each starts with fresh history)")
    p.add_argument("--max-tokens", type=int, default=150)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--request-timeout", type=float, default=180.0)
    p.add_argument("--out", default=None,
                   help="Optional JSON output path")
    args = p.parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
