# SPDX-License-Identifier: Apache-2.0
"""Live micro-benchmark driven against a running vLLM engine.

This produces *real* TTFT / TPOT / throughput by streaming completions through
the engine's OpenAI API and timing them — closing the "dry-run only" gap for the
GUI. It is deliberately a **quick** bench with its parameters echoed in the
result, so two runs are comparable to each other (A/B), but it does **not** claim
to equal the canonical ``genesis_bench_suite`` Wave baselines (different prompt
set / protocol — iron rule #9). The GUI labels it accordingly.

stdlib only (urllib + threads); an unreachable engine raises so the route maps
it to a clean error.
"""
from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from statistics import mean, pstdev
from typing import Any, Callable, Iterable, Optional

from .engine_client import EngineError, resolve_engine

# Guards against an accidental heavy run from the GUI.
_MAX_REQUESTS = 64
_MAX_CONCURRENCY = 16
_MAX_TOKENS = 2048

_DEFAULT_PROMPT = (
    "Explain, in a few sentences, why batching improves LLM inference throughput."
)


def _percentile(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[rank]


def parse_sse_stream(lines: Iterable[bytes], *, t0: float, clock: Callable[[], float]) -> dict[str, Any]:
    """Consume an OpenAI streaming response into timing + token counts.

    ``lines`` yields raw ``data: {...}`` byte lines. ``clock`` returns the
    current time (injected for tests). Returns ttft/total/tokens/ok.
    """
    ttft: Optional[float] = None
    last_token_at = t0
    chunk_tokens = 0
    usage_tokens: Optional[int] = None
    for raw in lines:
        line = raw.decode("utf-8", "replace").strip() if isinstance(raw, (bytes, bytearray)) else raw.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        usage = payload.get("usage")
        if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
            usage_tokens = int(usage["completion_tokens"])
        for choice in payload.get("choices", []) or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                now = clock()
                if ttft is None:
                    ttft = now - t0
                last_token_at = now
                chunk_tokens += 1
    tokens = usage_tokens if usage_tokens is not None else chunk_tokens
    return {
        "ttft_s": ttft,
        "total_s": last_token_at - t0,
        "tokens": tokens,
        "ok": tokens > 0,
    }


def _stream_one(base_url: str, body: dict, *, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed scheme/port
            return parse_sse_stream(response, t0=t0, clock=time.time)
    except Exception as exc:  # noqa: BLE001 - per-request failure is recorded, not fatal
        return {"ttft_s": None, "total_s": time.time() - t0, "tokens": 0, "ok": False, "error": str(exc)}


def run_bench(params: dict[str, Any], *, host: Optional[str] = None, _runner: Optional[Callable] = None) -> dict[str, Any]:
    """Run a live micro-benchmark against the engine and aggregate KPIs."""
    num_requests = max(1, min(_MAX_REQUESTS, int(params.get("num_requests", 8))))
    concurrency = max(1, min(_MAX_CONCURRENCY, int(params.get("concurrency", 2))))
    max_tokens = max(1, min(_MAX_TOKENS, int(params.get("max_tokens", 128))))
    temperature = float(params.get("temperature", 0.7))
    prompt = str(params.get("prompt") or _DEFAULT_PROMPT)
    model = params.get("model") or "default"

    eng = resolve_engine(host)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    runner = _runner or (lambda: _stream_one(eng["base_url"], body, timeout=120.0))

    wall_start = time.time()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(runner) for _ in range(num_requests)]
        for future in futures:
            results.append(future.result())
    wall_clock = max(1e-6, time.time() - wall_start)

    ok = [r for r in results if r.get("ok")]
    failed = len(results) - len(ok)
    if not ok:
        raise EngineError("All benchmark requests failed — is the engine serving this model?")

    total_tokens = sum(int(r["tokens"]) for r in ok)
    ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
    # Per-request decode rate (tokens after the first), and TPOT.
    tpots: list[float] = []
    per_req_tok_s: list[float] = []
    for r in ok:
        tokens = int(r["tokens"])
        total_s = float(r["total_s"]) or 1e-6
        ttft = float(r["ttft_s"]) if r.get("ttft_s") is not None else 0.0
        decode_s = max(1e-6, total_s - ttft)
        if tokens > 1:
            tpots.append(decode_s / (tokens - 1))
        per_req_tok_s.append(tokens / total_s)

    throughput = total_tokens / wall_clock
    cv_pct = (pstdev(per_req_tok_s) / mean(per_req_tok_s) * 100.0) if len(per_req_tok_s) > 1 and mean(per_req_tok_s) else 0.0

    return {
        "ok": True,
        "params": {
            "num_requests": num_requests,
            "concurrency": concurrency,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "model": model,
        },
        "metrics": {
            "throughput_tok_s": round(throughput, 1),
            "ttft_avg_ms": round(mean(ttfts) * 1000, 1) if ttfts else None,
            "ttft_p50_ms": round(_percentile(ttfts, 50) * 1000, 1) if ttfts else None,
            "ttft_p90_ms": round(_percentile(ttfts, 90) * 1000, 1) if ttfts else None,
            "tpot_avg_ms": round(mean(tpots) * 1000, 2) if tpots else None,
            "cv_pct": round(cv_pct, 1),
            "total_tokens": total_tokens,
            "requests_ok": len(ok),
            "requests_failed": failed,
            "wall_clock_s": round(wall_clock, 2),
        },
        "methodology": "live quick-bench (GUI) — not the canonical Wave suite",
    }
