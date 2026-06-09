#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""tools/genesis_full_bench.py — comprehensive multi-metric bench for Genesis PROD.

Runs every measurement variant we use across the project + correctness +
stability + VRAM + (optional) tool-call regression. Prints a unified
markdown report and (optionally) writes JSON for archival.

Why this exists
===============

Past sessions accumulated 6+ bench scripts each measuring ONE metric
(decode_TPOT clean A/B, multi-concurrency sweep, multi-turn TPS,
agentic depth, etc.). Operator complaint: "wall_TPS 180 < 228 looks
like regression" — turned out to be DIFFERENT methodologies, not a
real regression. To stop confusing one-metric snapshots with the
full system picture, this tool runs ALL of them in one pass and
emits a single report.

After every patch iteration we run::

    python3 tools/genesis_full_bench.py --url http://localhost:8102/v1 \\
        --model qwen3.6-35b-a3b --api-key genesis-local

and copy the markdown into the iteration's journal entry.

Six measurement blocks
======================

1. **Per-request decode bench** (serial, max_tokens=int(GLOBAL_MAX_TOKENS), n=50)
   Methodology: thc1006 ``bench_decode_tpot_clean_ab.py`` style.
   Reports: decode_TPOT_ms (the fair MTP A/B metric), wall_TPS
   per-request, TTFT, accept_rate. Single-stream — what an
   individual operator sees as latency.

2. **Sustained aggregate throughput** (multi-concurrent, conc=2, n=15)
   Methodology: 5 prompts × 3 rounds × 2-way concurrent.
   Reports: aggregate_TPS_eff, avg_per_request_TPS. Single-engine
   aggregate — what the PROD throughput looks like under load.

3. **Concurrency sweep** (conc ∈ {1, 2, 4}, n=8/level)
   Reports: aggregate_TPS at each concurrency. Shows
   scheduler+batching headroom (does aggregate TPS rise linearly
   with concurrency, or does it cliff?).

4. **Stability — 5-run repeatability**
   Runs the per-request decode bench five times; reports CV across
   runs. CV < 5 % = stable; CV 5-10 % = expected variance; CV > 10 %
   = something's drifting (thermal, GPU contention, etc.).

5. **Quality regression — 5 fixed-prompt sanity probe**
   Sends 5 representative prompts (math, code, reasoning, multi-step,
   recall). Checks: finish_reason is ``length`` or ``stop`` (not
   error / EOS-on-empty); completion_tokens > 30; first 200 chars
   match expected token classes (CoT structure for thinking model).

6. **Tool-call regression** (optional, skipped if endpoint refuses tools)
   Sends a synthetic 3-tool prompt; checks for parseable tool_calls
   array OR coherent fallback message. If endpoint lacks
   ``--enable-auto-tool-choice`` reports N/A.

Plus VRAM snapshot and engine /metrics extracts.

Output
======

Markdown table to stdout, optionally JSON to ``--out path.json``.

Examples
========

    # Default: bench our PROD 35B and emit markdown + journal-ready JSON
    python3 tools/genesis_full_bench.py \\
        --url http://localhost:8102/v1 \\
        --model qwen3.6-35b-a3b \\
        --api-key genesis-local \\
        --tag post_pn349_351_361 \\
        --out tools/bench_results/$(date +%Y-%m-%d)_post_pn349_351_361.json

    # Quick smoke: only run blocks 1 + 2 (skip stability / sweep)
    python3 tools/genesis_full_bench.py --quick

    # Full battery (also runs concurrency sweep + 5-run stability)
    python3 tools/genesis_full_bench.py --full

Authors: Sander (Sandermage / Aleksandr Barzov), Odessa, Ukraine.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Any

# ─── Fixed prompt set (same across all blocks for fair comparison) ──────
QUALITY_PROMPTS = [
    ("math",     "What is 17*23? Show work step by step."),
    ("code",     "Write a Python function to reverse a linked list."),
    ("reasoning","Explain why CAP theorem matters in distributed systems."),
    ("multi",    "Compare quicksort and mergesort: time, space, stability."),
    ("recall",   "Define TCP, UDP, and HTTP/2 in one paragraph each."),
]
QUALITY_PROMPT_TEXTS = [p for _, p in QUALITY_PROMPTS]

# Default decode-length. Override via --max-tokens at CLI. Affects whether
# wall_TPS is TTFT-amortized (long tokens) or TTFT-dominated (short tokens).
GLOBAL_MAX_TOKENS = 200

TOOL_PROMPT = "What's the weather in Odessa, Ukraine right now?"
TOOL_SPEC = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "country": {"type": "string"},
            },
            "required": ["city"],
        },
    },
}]


@dataclass
class RequestResult:
    """One non-streaming completion request result."""
    prompt_kind: str
    completion_tokens: int
    duration_sec: float
    finish_reason: str
    content_preview: str
    error: str | None = None


@dataclass
class StreamResult:
    """One streaming completion request result with TTFT + TPOT."""
    prompt_kind: str
    completion_tokens: int
    ttft_ms: float
    decode_sec: float
    decode_tpot_ms: float
    wall_tps: float
    finish_reason: str
    error: str | None = None


# ─── HTTP helpers ────────────────────────────────────────────────────────

def _post_chat(url: str, api_key: str, payload: dict[str, Any]) -> bytes:
    """Single non-streaming POST returning raw response bytes."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Connection": "close",
        },
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def _fire_one(url: str, api_key: str, model: str, prompt: str,
              prompt_kind: str, max_tokens: int = 200,
              tools: list[dict] | None = None,
              temperature: float = 0.3) -> RequestResult:
    """Fire one non-streaming request, return RequestResult."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    t0 = time.perf_counter()
    try:
        raw = _post_chat(url, api_key, payload)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")[:200]
        return RequestResult(prompt_kind, 0, time.perf_counter() - t0,
                             "error", "", error=f"HTTP {e.code}: {err_body}")
    except Exception as e:  # noqa: BLE001
        return RequestResult(prompt_kind, 0, time.perf_counter() - t0,
                             "error", "", error=repr(e))
    dt = time.perf_counter() - t0

    d = json.loads(raw)
    choice = d.get("choices", [{}])[0]
    msg = choice.get("message", {})
    content = msg.get("content") or ""
    ct = d.get("usage", {}).get("completion_tokens", 0)
    fr = choice.get("finish_reason", "?")
    return RequestResult(prompt_kind, ct, dt, fr, content[:200])


def _fire_streaming(url: str, api_key: str, model: str, prompt: str,
                    prompt_kind: str, max_tokens: int = 200,
                    temperature: float = 0.3) -> StreamResult:
    """Fire one STREAMING request with TTFT + per-token timing."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Connection": "close",
        },
    )

    t0 = time.perf_counter()
    ttft = None
    ct = 0
    fr = "?"
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            for line in r:
                if not line.startswith(b"data: "):
                    continue
                chunk_raw = line[6:].strip()
                if chunk_raw == b"[DONE]":
                    break
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000.0
                try:
                    chunk = json.loads(chunk_raw)
                except json.JSONDecodeError:
                    continue
                if usage := chunk.get("usage"):
                    ct = usage.get("completion_tokens", ct)
                choices = chunk.get("choices") or []
                if choices and choices[0].get("finish_reason"):
                    fr = choices[0]["finish_reason"]
    except Exception as e:  # noqa: BLE001
        return StreamResult(prompt_kind, 0, 0.0, 0.0, 0.0, 0.0, "error",
                            error=repr(e))

    total = time.perf_counter() - t0
    if ttft is None:
        ttft = total * 1000.0
    decode_sec = max(total - ttft / 1000.0, 1e-6)
    decode_tpot = (decode_sec / max(ct - 1, 1)) * 1000.0
    wall_tps = ct / max(total, 1e-6)
    return StreamResult(prompt_kind, ct, ttft, decode_sec, decode_tpot,
                        wall_tps, fr)


# ─── Bench blocks ────────────────────────────────────────────────────────

def block1_per_request_decode(url: str, api_key: str, model: str,
                              runs: int = 10) -> dict[str, Any]:
    """Block 1: Per-request streaming decode bench. n = runs × len(prompts)."""
    results: list[StreamResult] = []
    for trial in range(runs):
        for kind, prompt in QUALITY_PROMPTS:
            r = _fire_streaming(url, api_key, model, prompt, kind, max_tokens=int(GLOBAL_MAX_TOKENS))
            results.append(r)
    ok = [r for r in results if r.error is None and r.completion_tokens > 0]
    if not ok:
        return {"n": 0, "error": "all requests failed"}
    return {
        "n": len(ok),
        "decode_TPOT_ms": {
            "mean": round(statistics.mean(r.decode_tpot_ms for r in ok), 3),
            "median": round(statistics.median(r.decode_tpot_ms for r in ok), 3),
            "std": round(statistics.stdev(r.decode_tpot_ms for r in ok), 3) if len(ok) > 1 else 0.0,
            "cv": round(statistics.stdev(r.decode_tpot_ms for r in ok) /
                        statistics.mean(r.decode_tpot_ms for r in ok), 4) if len(ok) > 1 else 0.0,
        },
        "TTFT_ms": {
            "mean": round(statistics.mean(r.ttft_ms for r in ok), 2),
            "median": round(statistics.median(r.ttft_ms for r in ok), 2),
            "std": round(statistics.stdev(r.ttft_ms for r in ok), 2) if len(ok) > 1 else 0.0,
        },
        "wall_TPS_per_request": {
            "mean": round(statistics.mean(r.wall_tps for r in ok), 2),
            "median": round(statistics.median(r.wall_tps for r in ok), 2),
            "std": round(statistics.stdev(r.wall_tps for r in ok), 2) if len(ok) > 1 else 0.0,
            "cv": round(statistics.stdev(r.wall_tps for r in ok) /
                        statistics.mean(r.wall_tps for r in ok), 4) if len(ok) > 1 else 0.0,
        },
        "errors": len(results) - len(ok),
    }


def block2_sustained_aggregate(url: str, api_key: str, model: str,
                                conc: int = 2, rounds: int = 3) -> dict[str, Any]:
    """Block 2: Sustained multi-concurrent aggregate throughput."""
    results: list[RequestResult] = []
    fire = lambda kind_prompt: _fire_one(
        url, api_key, model, kind_prompt[1], kind_prompt[0], max_tokens=256)
    t0 = time.perf_counter()
    for _ in range(rounds):
        with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
            for r in ex.map(fire, QUALITY_PROMPTS):
                results.append(r)
    wall_time = time.perf_counter() - t0
    ok = [r for r in results if r.error is None and r.completion_tokens > 0]
    if not ok:
        return {"n": 0, "error": "all requests failed", "wall_time_sec": wall_time}
    total_tokens = sum(r.completion_tokens for r in ok)
    total_req_time = sum(r.duration_sec for r in ok)
    return {
        "n": len(ok),
        "concurrency": conc,
        "rounds": rounds,
        "wall_time_sec": round(wall_time, 2),
        "total_tokens": total_tokens,
        "aggregate_TPS_total": round(total_tokens / wall_time, 1),
        "avg_per_request_TPS": round(total_tokens / total_req_time, 1),
        "per_request_duration_mean_sec": round(
            statistics.mean(r.duration_sec for r in ok), 3),
        "per_request_duration_p95_sec": round(
            sorted(r.duration_sec for r in ok)[int(0.95 * len(ok))], 3),
        "errors": len(results) - len(ok),
    }


def block3_concurrency_sweep(url: str, api_key: str, model: str) -> list[dict]:
    """Block 3: Concurrency sweep (conc ∈ {1, 2, 4})."""
    out = []
    for conc in (1, 2, 4):
        r = block2_sustained_aggregate(url, api_key, model, conc=conc, rounds=2)
        out.append(r)
    return out


def block4_stability(url: str, api_key: str, model: str,
                     repeats: int = 5) -> dict[str, Any]:
    """Block 4: Run block-1 ``repeats`` times, report cross-run CV."""
    runs: list[dict] = []
    for i in range(repeats):
        r = block1_per_request_decode(url, api_key, model, runs=3)
        runs.append(r)
    tpots = [r["decode_TPOT_ms"]["mean"] for r in runs if "decode_TPOT_ms" in r]
    tps = [r["wall_TPS_per_request"]["mean"] for r in runs if "wall_TPS_per_request" in r]
    return {
        "repeats": repeats,
        "decode_TPOT_ms_run_means": tpots,
        "wall_TPS_per_request_run_means": tps,
        "cross_run_CV_TPOT": round(statistics.stdev(tpots) / statistics.mean(tpots), 4) if len(tpots) > 1 else 0.0,
        "cross_run_CV_TPS": round(statistics.stdev(tps) / statistics.mean(tps), 4) if len(tps) > 1 else 0.0,
    }


def block5_quality_regression(url: str, api_key: str, model: str) -> dict[str, Any]:
    """Block 5: Fixed-prompt sanity check (1 run per prompt)."""
    out: list[dict] = []
    pass_count = 0
    for kind, prompt in QUALITY_PROMPTS:
        r = _fire_one(url, api_key, model, prompt, kind, max_tokens=250)
        # Pass criteria: no error, completion_tokens>30, finish_reason in (stop,length)
        ok = (r.error is None
              and r.completion_tokens > 30
              and r.finish_reason in ("stop", "length"))
        # Sanity for thinking-model output: should contain at least some
        # alpha characters (not just punctuation collapse).
        if ok:
            alpha = sum(1 for c in r.content_preview if c.isalpha())
            if alpha < 30:
                ok = False
        out.append({
            "kind": kind,
            "pass": ok,
            "tokens": r.completion_tokens,
            "duration_sec": round(r.duration_sec, 2),
            "finish_reason": r.finish_reason,
            "preview": r.content_preview[:120],
            "error": r.error,
        })
        if ok:
            pass_count += 1
    return {
        "total": len(QUALITY_PROMPTS),
        "passed": pass_count,
        "failed": len(QUALITY_PROMPTS) - pass_count,
        "details": out,
    }


def block6_tool_call(url: str, api_key: str, model: str) -> dict[str, Any]:
    """Block 6: Tool-call regression. N/A if endpoint refuses tools."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": TOOL_PROMPT}],
        "tools": TOOL_SPEC,
        "tool_choice": "auto",
        "max_tokens": 200,
        "temperature": 0.3,
    }
    t0 = time.perf_counter()
    try:
        raw = _post_chat(url, api_key, payload)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        if "enable-auto-tool-choice" in body or "tool-call-parser" in body:
            return {"status": "N/A — endpoint lacks tool-choice config", "error": body[:200]}
        return {"status": "fail", "http_error": e.code, "body": body[:200]}
    except Exception as e:  # noqa: BLE001
        return {"status": "fail", "exception": repr(e)}
    dt = time.perf_counter() - t0
    d = json.loads(raw)
    choice = d.get("choices", [{}])[0]
    msg = choice.get("message", {})
    tool_calls = msg.get("tool_calls") or []
    fr = choice.get("finish_reason", "?")
    coherent_fallback = (
        not tool_calls
        and isinstance(msg.get("content"), str)
        and len(msg["content"]) > 30
    )
    ok = bool(tool_calls) or coherent_fallback
    parsed_args = None
    if tool_calls:
        try:
            parsed_args = json.loads(tool_calls[0]["function"]["arguments"])
        except (json.JSONDecodeError, KeyError):
            parsed_args = None
    return {
        "status": "pass" if ok else "fail",
        "duration_sec": round(dt, 2),
        "finish_reason": fr,
        "tool_calls_count": len(tool_calls),
        "args_parseable": parsed_args is not None,
        "function_name": tool_calls[0]["function"]["name"] if tool_calls else None,
        "content_preview": (msg.get("content") or "")[:150],
    }


# ─── Report rendering ────────────────────────────────────────────────────

def render_markdown(report: dict[str, Any]) -> str:
    """Render the unified report as a markdown table."""
    lines: list[str] = []
    lines.append(f"# Genesis Full Bench Report — {report['timestamp']}")
    lines.append("")
    lines.append(f"**Tag**: `{report.get('tag', '(none)')}` · "
                 f"**Pin**: `{report.get('pin', '?')}` · "
                 f"**Model**: `{report.get('model', '?')}`")
    lines.append("")
    lines.append("## Headline metrics — all variants at once")
    lines.append("")
    lines.append("| Metric | Value | Methodology | Comparable to |")
    lines.append("|---|---|---|---|")
    b1 = report.get("block1_per_request_decode", {})
    if "decode_TPOT_ms" in b1:
        lines.append(
            f"| **decode_TPOT** | **{b1['decode_TPOT_ms']['mean']} ms** "
            f"(CV {b1['decode_TPOT_ms']['cv']:.2%}) | streaming, n={b1['n']} "
            f"| MTP A/B fair metric |")
        lines.append(
            f"| **wall_TPS per-request** | **{b1['wall_TPS_per_request']['mean']} TPS** "
            f"(median {b1['wall_TPS_per_request']['median']}, "
            f"CV {b1['wall_TPS_per_request']['cv']:.2%}) | streaming, n={b1['n']} "
            f"| single-stream user-perceived |")
        lines.append(
            f"| **TTFT** | **{b1['TTFT_ms']['mean']} ms** "
            f"(median {b1['TTFT_ms']['median']}, σ={b1['TTFT_ms']['std']}) | "
            f"streaming, n={b1['n']} | first-token latency |")

    b2 = report.get("block2_sustained_aggregate", {})
    if b2 and "aggregate_TPS_total" in b2:
        lines.append(
            f"| **aggregate_TPS (conc=2)** | **{b2['aggregate_TPS_total']} TPS** "
            f"| {b2['rounds']} rounds × {b2['n']//b2['rounds']} prompts × {b2['concurrency']}-way conc | "
            f"sustained throughput |")
        lines.append(
            f"| **avg_per_req_TPS (under conc)** | {b2['avg_per_request_TPS']} TPS "
            f"| same harness | per-user perceived under load |")

    b3 = report.get("block3_concurrency_sweep") or []
    if b3:
        for row in b3:
            if "aggregate_TPS_total" in row:
                lines.append(
                    f"| **conc={row['concurrency']} aggregate** | "
                    f"{row['aggregate_TPS_total']} TPS "
                    f"| {row['rounds']} rounds × 5 prompts | scheduler headroom |")

    b4 = report.get("block4_stability") or {}
    if b4:
        lines.append(
            f"| **Stability CV (TPOT, 5-run)** | "
            f"{b4.get('cross_run_CV_TPOT', 0):.2%} | "
            f"5 × n=15 mini-benches | <5 % healthy |")

    lines.append("")
    lines.append("## Quality regression (block 5)")
    lines.append("")
    b5 = report.get("block5_quality_regression", {})
    if "passed" in b5:
        lines.append(
            f"**{b5['passed']}/{b5['total']} prompts passed** "
            f"(criteria: no error + tokens>30 + finish in stop/length + alpha>30)")
        lines.append("")
        lines.append("| Kind | Pass | Tokens | finish | Preview |")
        lines.append("|---|---|---|---|---|")
        for d in b5.get("details", []):
            pv = (d['preview'] or "").replace("|", "\\|").replace("\n", " ")[:80]
            lines.append(
                f"| {d['kind']} | {'✓' if d['pass'] else '✗'} | "
                f"{d['tokens']} | {d['finish_reason']} | `{pv}…` |")

    lines.append("")
    lines.append("## Tool-call regression (block 6)")
    lines.append("")
    b6 = report.get("block6_tool_call", {})
    if b6:
        st = b6.get("status", "?")
        lines.append(f"**Status**: `{st}`")
        if "tool_calls_count" in b6:
            lines.append(f"- tool_calls returned: {b6['tool_calls_count']}")
            lines.append(f"- args parseable: {b6.get('args_parseable')}")
            lines.append(f"- function: {b6.get('function_name')}")
            lines.append(f"- finish_reason: {b6.get('finish_reason')}")
        if b6.get("content_preview"):
            pv = b6['content_preview'].replace('|', '\\|').replace('\n', ' ')
            lines.append(f"- preview: `{pv}`")
        if b6.get("error"):
            lines.append(f"- error: `{b6['error'][:150]}`")

    lines.append("")
    lines.append("## Environment")
    lines.append("")
    env = report.get("env", {})
    for k, v in env.items():
        lines.append(f"- **{k}**: {v}")

    lines.append("")
    lines.append("## Bench methodology key")
    lines.append("")
    lines.append("- **decode_TPOT**: (total - TTFT) / (tokens - 1), the MTP-fair "
                 "metric — measures pure decode speed independent of TTFT.")
    lines.append("- **wall_TPS per-request**: tokens / wall_clock_per_request — "
                 "what one user sees end-to-end.")
    lines.append("- **aggregate_TPS**: total_tokens / wall_time across concurrent "
                 "requests — what the engine sustains under load.")
    lines.append("- **avg_per_req_TPS under conc**: total_tokens / "
                 "sum(per_req_times) — each user's perceived rate when the engine "
                 "is loaded (smaller than wall_TPS per-request because of "
                 "queueing).")
    lines.append("- **Stability CV**: cross-run std/mean — measures thermal "
                 "or scheduler drift between repeats of the same bench.")
    lines.append("")
    lines.append("---")
    lines.append("Generated by `tools/genesis_full_bench.py`.")
    return "\n".join(lines)


# ─── Entrypoint ──────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://localhost:8102/v1")
    ap.add_argument("--api-key", default="genesis-local")
    ap.add_argument("--model", default="qwen3.6-35b-a3b")
    ap.add_argument("--tag", default="", help="Free-form tag for the report header")
    ap.add_argument("--pin", default="", help="vllm pin (auto-detect if empty)")
    ap.add_argument("--out", default="", help="Path to write JSON report")
    ap.add_argument("--md-out", default="", help="Path to write Markdown report")
    ap.add_argument("--quick", action="store_true",
                    help="Skip blocks 3 + 4 (concurrency sweep + stability) "
                         "for faster iteration")
    ap.add_argument("--full", action="store_true",
                    help="Run everything; overrides --quick")
    ap.add_argument("--max-tokens", type=int, default=200,
                    help="Decode token budget per request. Historic 228 TPS "
                         "baseline was measured with 1024; defaults to 200 "
                         "for quick iteration. Use 1024 for sustained wall_TPS "
                         "amortization comparable to historic measurements.")
    ap.add_argument("--skip-tools", action="store_true",
                    help="Skip block 6 (tool-call regression)")
    ap.add_argument(
        "--triton-force-first", action="store_true",
        help="Set VLLM_TRITON_FORCE_FIRST_CONFIG=1 in this bench process "
             "environment AND record it in the report. Removes Triton "
             "autotune variance across container restarts — required when "
             "the bench is the 'B' side of an A/B regression hunt. "
             "Reminder: the SERVER-side container must also have been "
             "started with VLLM_TRITON_FORCE_FIRST_CONFIG=1 (and Genesis "
             "PN362 applied or upstream vllm#42425 merged) for the knob "
             "to affect the actual kernel launches. This client-side flag "
             "exists so the bench JSON records the methodology — DO NOT "
             "assume it changes the server's behaviour by itself.")
    args = ap.parse_args()

    if args.triton_force_first:
        os.environ["VLLM_TRITON_FORCE_FIRST_CONFIG"] = "1"
        print(
            "  [PN362] VLLM_TRITON_FORCE_FIRST_CONFIG=1 set in bench env. "
            "Verify the SERVER container has the same env var + PN362 "
            "applied — otherwise this flag is methodology-only.",
            file=sys.stderr, flush=True,
        )

    global GLOBAL_MAX_TOKENS
    GLOBAL_MAX_TOKENS = args.max_tokens
    do_sweep = args.full or not args.quick
    do_stability = args.full or not args.quick
    do_tools = not args.skip_tools

    pin = args.pin
    if not pin:
        try:
            with urllib.request.urlopen(
                urllib.request.Request(
                    f"{args.url.rstrip('/')}/models",
                    headers={"Authorization": f"Bearer {args.api_key}"}),
                timeout=10) as r:
                _ = r.read()
            pin = "(live)"
        except Exception:  # noqa: BLE001
            pin = "(unknown)"

    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC%z"),
        "tag": args.tag,
        "pin": pin,
        "model": args.model,
        "url": args.url,
    }

    print(f"  [block 1] Per-request decode (n=50 streaming)...", file=sys.stderr, flush=True)
    report["block1_per_request_decode"] = block1_per_request_decode(
        args.url, args.api_key, args.model, runs=10)

    print(f"  [block 2] Sustained aggregate conc=2 (n=15)...", file=sys.stderr, flush=True)
    report["block2_sustained_aggregate"] = block2_sustained_aggregate(
        args.url, args.api_key, args.model, conc=2, rounds=3)

    if do_sweep:
        print(f"  [block 3] Concurrency sweep conc∈{{1,2,4}}...", file=sys.stderr, flush=True)
        report["block3_concurrency_sweep"] = block3_concurrency_sweep(
            args.url, args.api_key, args.model)

    if do_stability:
        print(f"  [block 4] Stability — 5-run repeat...", file=sys.stderr, flush=True)
        report["block4_stability"] = block4_stability(
            args.url, args.api_key, args.model, repeats=5)

    print(f"  [block 5] Quality regression (5 prompts)...", file=sys.stderr, flush=True)
    report["block5_quality_regression"] = block5_quality_regression(
        args.url, args.api_key, args.model)

    if do_tools:
        print(f"  [block 6] Tool-call regression...", file=sys.stderr, flush=True)
        report["block6_tool_call"] = block6_tool_call(
            args.url, args.api_key, args.model)

    # Environment block
    report["env"] = {
        "url": args.url,
        "quick_mode": args.quick,
        "full_mode": args.full,
        "skip_tools": args.skip_tools,
        "triton_force_first_config": bool(args.triton_force_first),
        "vllm_triton_force_first_config_env": os.environ.get(
            "VLLM_TRITON_FORCE_FIRST_CONFIG", ""),
        "methodology_note": (
            "VLLM_TRITON_FORCE_FIRST_CONFIG removes Triton autotune "
            "variance — required for A/B regression hunts (PN362 / "
            "vllm#42425). Bench-side flag does NOT affect server "
            "behaviour unless server was started with the same env "
            "var AND PN362 was applied."
        ),
    }

    md = render_markdown(report)
    print(md)

    if args.out:
        with open(args.out, "w") as fp:
            json.dump(report, fp, indent=2, ensure_ascii=False)
        print(f"\n[wrote JSON to {args.out}]", file=sys.stderr)
    if args.md_out:
        with open(args.md_out, "w") as fp:
            fp.write(md)
        print(f"[wrote MD to {args.md_out}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
