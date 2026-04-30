#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Genesis Benchmark Suite — community-grade single-script benchmark.

Comprehensive vLLM benchmark for Genesis Patches users. Covers:
  1. Server discovery + GPU profile (auto-detect card class)
  2. Tool-call quality (4 cases × 2 thinking modes = 8 cases)
  3. Decode-only TPOT bench (configurable N runs × M prompts)
  4. Wall TPS + TTFT measurement
  5. Multi-turn TTFT probe (cache benefit smell test)
  6. Stability stress (configurable, with crash detection)
  7. Context window probe (user-selectable max — for cards that can't 256K)
  8. Genesis log inspection (which patches APPLY/SKIP/applied-state)

Methodology cribbed from thc1006's `bench_v3_clean_ab.py` (decode-only TPOT
methodology) — credit in CREDITS.md. Statistical comparison via Welch's
t-test (stdlib-only, no scipy).

Output:
  - <name>.json — full machine-readable results
  - <name>.md   — human-readable summary
  - stdout      — live progress + final verdict

Usage:
  # Quick smoke test (5 min)
  python3 genesis_bench_suite.py --quick

  # Standard run (25 min)
  python3 genesis_bench_suite.py --mode standard --ctx 8k

  # Full battery (1-2 hours, picks largest ctx your card can hold)
  python3 genesis_bench_suite.py --mode full --ctx all

  # Compare two arms
  python3 genesis_bench_suite.py --compare A.json B.json

For run instructions on bare-metal / Docker / VM / WSL / RunPod see
docs/BENCHMARK_GUIDE.md in the same repo.

License: Apache-2.0 (matches vLLM upstream).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ────────────────────────────────────────────────────────────────────────
# CLI

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Genesis benchmark suite — measure tool-call quality, decode "
            "TPOT, wall TPS, TTFT, stability, and context window across "
            "diverse hardware classes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Connection
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--scheme", choices=["http", "https"], default="http",
                   help="URL scheme (default http; use https for RunPod / TLS-fronted endpoints)")
    p.add_argument("--api-key", default="genesis-local")
    p.add_argument("--model", default=None,
                   help="Override model name (default: auto-detect from /v1/models)")
    # Mode presets
    p.add_argument("--mode", choices=["quick", "standard", "full"], default="standard")
    p.add_argument("--quick", action="store_true",
                   help="Equivalent to --mode quick.")
    # Granular control (overrides preset defaults)
    p.add_argument("--runs", type=int, default=None,
                   help="Bench iterations per prompt (default: by mode)")
    p.add_argument("--prompts", choices=["short", "standard", "long"], default=None)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--ctx", default=None,
                   help="Max context size to probe: 1K, 4K, 8K, 16K, 32K, 64K, 128K, 256K, or 'all'.")
    p.add_argument("--stress", type=int, default=None,
                   help="Stability stress iterations (default: by mode)")
    p.add_argument("--ttft-turns", type=int, default=5)
    p.add_argument("--ctx-timeout", type=int, default=300)
    # Output
    p.add_argument("--out", default=None,
                   help="Output JSON path (default: derived from mode + timestamp)")
    p.add_argument("--md", default=None,
                   help="Output Markdown path (default: <out>.md)")
    p.add_argument("--name", "--arm-name", dest="name", default=None,
                   help="Arm name in result (default: derived from mode + GPU). "
                        "`--arm-name` is a doc-compatibility alias.")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--verbose", action="store_true")
    # Tests to skip
    p.add_argument("--skip-toolcall", action="store_true")
    p.add_argument("--skip-stress", action="store_true")
    p.add_argument("--skip-ctx-probe", action="store_true")
    p.add_argument("--skip-multi-turn", action="store_true")
    # Compare mode
    p.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                   help="Run Welch t-test compare on two existing result JSONs and exit.")
    p.add_argument("--compare-out", metavar="DELTA.json", default=None,
                   help="When --compare is set, write the delta JSON to this path "
                        "(in addition to printing to stdout).")
    # Output-length probe — measures how much can be RECEIVED
    p.add_argument("--probe-output-length", action="store_true",
                   help="Run the output-length probe: generate at increasingly "
                        "large max_tokens (1K, 2K, 4K, 8K, 16K) and verify the "
                        "model can stream that long without truncation/finish=length "
                        "regression. Pairs with VRAM peak scrape per request.")

    # D3 ablation mode — light-touch orchestrator helper
    p.add_argument("--ablate-against", metavar="BASELINE.json", default=None,
                   help="After this run completes, compare decode_bench wall_TPS / "
                        "decode_TPOT_ms / TTFT_ms against the named baseline JSON "
                        "via Welch t-test. Workflow: launch all-on → bench --name "
                        "baseline; restart with ONE patch OFF → bench --ablate-"
                        "against baseline.json --ablate-tag no-<PATCH>. Per-patch "
                        "deltas + p-values stored under `ablation` in result JSON.")
    p.add_argument("--ablate-tag", metavar="LABEL", default=None,
                   help="Free-form label for what was ablated (e.g. 'no-PN14'). "
                        "Stored in the ablation block for traceability.")

    args = p.parse_args(argv)
    if args.quick:
        args.mode = "quick"
    return args


# ────────────────────────────────────────────────────────────────────────
# Mode presets

MODE_DEFAULTS = {
    "quick":    dict(runs=5,  stress=0,  ctx="1K",  prompts="short"),
    "standard": dict(runs=25, stress=30, ctx="8K",  prompts="standard"),
    "full":     dict(runs=25, stress=100, ctx="all", prompts="standard"),
}

CTX_TARGETS = {
    "1K":   1024,
    "4K":   4096,
    "8K":   8192,
    "16K":  16384,
    "32K":  32768,
    "64K":  65536,
    "128K": 131072,
    "256K": 262144,
    "512K": 524288,
}

PROMPT_SETS = {
    "short": [
        "What is 2+2?",
        "Explain Newton's first law in one sentence.",
        "Name three programming languages.",
        "Write a haiku about rain.",
        "Define entropy briefly.",
    ],
    "standard": [
        "Write a 500-word essay on quantum computing in detail.",
        "Explain the Krebs cycle to a high schooler with examples.",
        "Compare TCP and UDP for a backend engineer; cover handshake, ordering, congestion.",
        "Outline the history of the Roman Empire from founding to fall in 800 words.",
        "Describe gradient descent with momentum mathematically and intuitively.",
    ],
    "long": [
        "Write a comprehensive technical analysis of attention mechanisms in transformers, "
        "covering scaled dot-product attention, multi-head attention, FlashAttention 2, "
        "FlashInfer, GQA, MLA, sliding window, and ALiBi. 1500 words.",
        "Author a detailed code review of the following hypothetical Python service: " * 20 +
        "Cover: error handling, async patterns, type safety, testing approach. 1000 words.",
        "Explain the physics of a black hole formation with full mathematical derivation "
        "from the Einstein field equations to the Schwarzschild metric. 1200 words.",
        "Walk through the design and implementation of a high-throughput log shipping "
        "pipeline (Kafka -> ETL -> ClickHouse). Discuss back-pressure, exactly-once "
        "delivery, partitioning, schema evolution. 1500 words.",
        "Compare the macro-economic policies of the United States in the 1970s vs. the "
        "2020s — inflation regimes, monetary policy, supply shocks. 1200 words.",
    ],
}


# ────────────────────────────────────────────────────────────────────────
# HTTP

# Module-level URL scheme — set once at main() entry from --scheme CLI arg.
# Default "http" preserves prior behavior for any external caller that
# imports test_* functions without going through main().
_URL_SCHEME: str = "http"


def _build_url(host: str, port: int, path: str) -> str:
    """Build an HTTP/HTTPS URL using the module-level _URL_SCHEME.

    Centralized so a future TLS-fronted endpoint (e.g. RunPod, public
    deployment) just sets `--scheme https` at main() and every endpoint
    inherits without per-test threading.
    """
    if not path.startswith("/"):
        path = "/" + path
    return f"{_URL_SCHEME}://{host}:{port}{path}"


def _bearer(api_key: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def http_get(url: str, headers: dict, timeout: float = 10.0) -> tuple[int, str]:
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.getcode(), r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace") if e.fp else ""
    except Exception as e:
        return 0, str(e)


def http_post_json(url: str, headers: dict, payload: dict, timeout: float = 60.0
                   ) -> tuple[int, dict | str]:
    try:
        req = urllib.request.Request(
            url, headers=headers, data=json.dumps(payload).encode(), method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.getcode(), json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read()) if e.fp else {}
        except Exception:
            return e.code, str(e)
    except Exception as e:
        return 0, str(e)


def http_post_stream(url: str, headers: dict, payload: dict,
                     timeout: float = 60.0, capture_text: bool = False,
                     system_fingerprint_capture: list | None = None):
    """Stream a chat completion and return per-trial telemetry.

    Optional `capture_text=True` accumulates the streamed `delta.content`
    into the returned `text` field — used by stability stress for SHA1
    drift detection across trials.

    Optional `system_fingerprint_capture` is a list that the function
    will append the response's `system_fingerprint` to (only the first
    one seen), if provided. Used for vLLM version capture.
    """
    payload = {**payload, "stream": True,
               "stream_options": {"include_usage": True}}
    req = urllib.request.Request(
        url, headers=headers, data=json.dumps(payload).encode(), method="POST"
    )
    t0 = time.perf_counter()
    ttft = None
    completion_tokens = 0
    finish_reason = None
    error = None
    text_buf: list[str] = [] if capture_text else None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                if not raw or not raw.startswith(b"data:"):
                    continue
                payload_str = raw[5:].strip()
                if payload_str == b"[DONE]":
                    break
                try:
                    chunk = json.loads(payload_str)
                except Exception:
                    continue
                # Capture system_fingerprint once if requested
                if (system_fingerprint_capture is not None
                        and not system_fingerprint_capture
                        and chunk.get("system_fingerprint")):
                    system_fingerprint_capture.append(
                        chunk["system_fingerprint"]
                    )
                if ttft is None:
                    delta = (chunk.get("choices") or [{}])[0].get("delta", {}) or {}
                    # TTFT = time to first delta carrying any actual data
                    # (skip the role-only init chunk vllm emits first).
                    if delta and any(
                        delta.get(k)
                        for k in ("content", "reasoning", "reasoning_content",
                                  "tool_calls", "function_call")
                    ):
                        ttft = time.perf_counter() - t0
                usage = chunk.get("usage") or {}
                if usage.get("completion_tokens") is not None:
                    completion_tokens = int(usage["completion_tokens"])
                ch0 = (chunk.get("choices") or [{}])[0]
                if text_buf is not None:
                    delta = ch0.get("delta") or {}
                    if isinstance(delta.get("content"), str):
                        text_buf.append(delta["content"])
                if ch0.get("finish_reason"):
                    finish_reason = ch0["finish_reason"]
    except Exception as e:
        error = repr(e)
    elapsed = time.perf_counter() - t0
    result = {
        "ttft_ms": round((ttft or 0) * 1000, 1) if ttft is not None else None,
        "elapsed_s": round(elapsed, 3),
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "error": error,
    }
    if text_buf is not None:
        result["text"] = "".join(text_buf)
    return result


# ────────────────────────────────────────────────────────────────────────
# Stats helpers

def mean_std_cv(xs: list[float]) -> dict:
    if not xs:
        return dict(mean=None, std=None, cv=None, min=None, max=None, n=0)
    n = len(xs)
    m = sum(xs) / n
    if n > 1:
        var = sum((x - m) ** 2 for x in xs) / (n - 1)
        s = math.sqrt(var)
    else:
        s = 0.0
    return dict(mean=round(m, 4), std=round(s, 4),
                cv=(round(s / m, 4) if m else None),
                min=round(min(xs), 4), max=round(max(xs), 4), n=n)


def welch_t(a: list[float], b: list[float]) -> dict:
    """Welch's two-sample t-test, two-sided. Returns t, df, approx p (Simpson)."""
    if len(a) < 2 or len(b) < 2:
        return dict(t=None, df=None, p_two_sided=None, verdict="INSUFFICIENT_SAMPLES")
    ma = sum(a) / len(a); mb = sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return dict(t=0.0, df=None, p_two_sided=1.0, verdict="IDENTICAL")
    t = (ma - mb) / se
    num = (va / len(a) + vb / len(b)) ** 2
    den = (va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1)
    df = num / den if den else float("inf")
    # Simpson's rule integration of Student's t pdf for two-sided p (rough but stdlib-only)
    p = _student_t_two_sided_p(t, df)
    verdict = "SIGNIFICANT" if p < 0.05 else "NOT_SIGNIFICANT"
    return dict(t=round(t, 4), df=round(df, 2),
                p_two_sided=round(p, 4), verdict=verdict)


def _student_t_two_sided_p(t: float, df: float) -> float:
    # Simpson's rule integration on the Student-t pdf, no scipy.
    # Accurate to ~3 decimals for df > 5 (good enough for engineering use).
    from math import gamma, sqrt, pi
    t = abs(t)
    coef = gamma((df + 1) / 2) / (sqrt(df * pi) * gamma(df / 2))
    def pdf(x):
        return coef * (1 + x * x / df) ** (-(df + 1) / 2)
    # integrate from t to ~100 (tail)
    hi = max(100.0, t * 5)
    n = 200
    h = (hi - t) / n
    s = pdf(t) + pdf(hi)
    for i in range(1, n):
        x = t + i * h
        s += 4 * pdf(x) if i % 2 else 2 * pdf(x)
    tail = s * h / 3
    return min(1.0, max(0.0, 2 * tail))


# ────────────────────────────────────────────────────────────────────────
# D3 — Per-patch ablation comparison helper

def _ablation_compare(baseline_path: str, current_result: dict,
                      ablate_tag: str | None) -> dict:
    """Compare current run against a baseline JSON via Welch t-test on the
    decode_bench wall_TPS / decode_TPOT_ms / TTFT_ms samples.

    Args:
        baseline_path: path to baseline JSON (all-on run)
        current_result: in-memory result dict for current run (one-patch-off)
        ablate_tag: human-readable label (e.g. 'no-PN14') stored in output

    Returns:
        dict with per-metric mean delta, percent change, Welch t/p, verdict.
        Empty dict (with `error` field) if baseline can't be loaded or
        decode_bench is missing on either side.
    """
    try:
        baseline = json.loads(Path(baseline_path).read_text())
    except Exception as e:
        return {"error": f"baseline load failed: {e}"}

    bdb = baseline.get("decode_bench")
    cdb = current_result.get("decode_bench")
    if not bdb or not cdb:
        return {"error": "decode_bench missing on baseline or current run"}

    out = {
        "baseline": baseline.get("name") or baseline_path,
        "current": current_result.get("name") or "current",
        "ablate_tag": ablate_tag,
        "metrics": {},
    }

    for metric_key, sample_field in [
        ("wall_TPS", "wall_tps"),
        ("decode_TPOT_ms", "decode_tpot_ms"),
        ("TTFT_ms", "ttft_ms"),
    ]:
        b_samples = [
            r.get(sample_field) for r in bdb.get("flat_results", [])
            if r.get(sample_field) is not None
        ]
        c_samples = [
            r.get(sample_field) for r in cdb.get("flat_results", [])
            if r.get(sample_field) is not None
        ]
        b_summary = bdb.get(metric_key, {})
        c_summary = cdb.get(metric_key, {})
        b_mean = b_summary.get("mean")
        c_mean = c_summary.get("mean")
        if b_mean is None or c_mean is None:
            out["metrics"][metric_key] = {"error": "summary mean missing"}
            continue
        delta = round(c_mean - b_mean, 4)
        pct = round((c_mean - b_mean) / b_mean * 100.0, 2) if b_mean else None
        wt = welch_t(c_samples, b_samples)
        out["metrics"][metric_key] = {
            "baseline_mean": b_mean,
            "current_mean": c_mean,
            "delta": delta,
            "pct_change": pct,
            "welch_t": wt.get("t"),
            "welch_p": wt.get("p_two_sided"),
            "verdict": wt.get("verdict"),
            "n_baseline": len(b_samples),
            "n_current": len(c_samples),
        }
    return out


def _print_ablation_table(ab: dict) -> None:
    """Pretty-print the ablation result table."""
    if "error" in ab:
        print(f"      ablation skipped — {ab['error']}")
        return
    tag = ab.get("ablate_tag") or "(no tag)"
    print(f"      vs baseline `{ab['baseline']}` (ablation: {tag})")
    rows = [("Metric", "Baseline", "Current", "Δ", "Δ%", "Welch p", "Verdict")]
    for k, v in ab["metrics"].items():
        if "error" in v:
            rows.append((k, "?", "?", "?", "?", "?", v["error"]))
            continue
        rows.append((
            k,
            f"{v['baseline_mean']}",
            f"{v['current_mean']}",
            f"{v['delta']:+.3f}" if isinstance(v['delta'], (int, float)) else str(v['delta']),
            f"{v['pct_change']:+.2f}%" if isinstance(v['pct_change'], (int, float)) else "?",
            f"{v['welch_p']}" if v['welch_p'] is not None else "?",
            v['verdict'] or "?",
        ))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("      " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(r)))


# ────────────────────────────────────────────────────────────────────────
# GPU profile detection (host-side via nvidia-smi, optional)

GPU_BANDWIDTH_GB_S = {
    # Static datasheet values (pulled from NVIDIA public specs).
    "rtx 3060": 360,    "rtx 3070": 448,    "rtx 3080": 760,
    "rtx 3090": 936,    "rtx a4000": 448,
    "rtx a5000": 768,   "rtx a6000": 768,
    "a100": 2039,
    "rtx 4070": 504,    "rtx 4080": 716,    "rtx 4090": 1008,
    "l40": 864,         "rtx 6000 ada": 960,
    "h100": 3350,       "h200": 4800,
    "rtx 5080": 960,    "rtx 5090": 1792,
    # Blackwell PRO workstation lineup
    "rtx pro 4000 blackwell": 672,
    "rtx pro 4500 blackwell": 896,
    "rtx pro 5000 blackwell": 1344,
    "rtx pro 6000 blackwell": 1792,
    "rtx pro 6000 blackwell max-q": 1792,
    "b200": 8000,
}


def detect_local_gpus() -> list[dict]:
    """Try nvidia-smi locally. Empty list if not available."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return []
    if out.returncode != 0:
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [s.strip() for s in line.split(",")]
        if len(parts) < 5:
            continue
        idx, name, mtot, mused, drv = parts[:5]
        cc = parts[5] if len(parts) > 5 else None
        nname = name.lower()
        bw = None
        for key, b in GPU_BANDWIDTH_GB_S.items():
            if key in nname:
                bw = b; break
        gpus.append(dict(
            index=int(idx), name=name, vram_total_mib=int(mtot),
            vram_used_mib=int(mused), driver=drv, compute_cap=cc,
            bandwidth_gb_s=bw,
        ))
    return gpus


# ────────────────────────────────────────────────────────────────────────
# Tests

def test_server_up(host: str, port: int, key: str) -> dict:
    code, body = http_get(_build_url(host, port, "/v1/models"), _bearer(key))
    if code == 200:
        try:
            d = json.loads(body)
            models = [m.get("id") for m in d.get("data", [])]
            return dict(reachable=True, http=200, models=models)
        except Exception:
            return dict(reachable=True, http=200, models=[])
    return dict(reachable=False, http=code, error=body[:300])


def test_tool_call(host: str, port: int, key: str, model: str, max_tokens: int = 1500) -> dict:
    """4 cases × 2 thinking modes = 8 cases. Strict 'tool_calls present' check."""
    cases = [
        # (label, thinking, prompt)
        ("paris_no_think",   False, "What's the weather in Paris? Use the get_weather tool."),
        ("tokyo_think",      True,  "Think step by step then call get_weather for Tokyo."),
        ("nyc_no_think",     False, "Call get_weather for New York."),
        ("london_think",     True,  "Reason about which city, then call get_weather for London."),
        ("kyiv_no_think",    False, "Get weather in Kyiv."),
        ("multi_no_think",   False, "Call get_weather for Berlin AND for Madrid (two calls)."),
        ("error_recovery",   False, "Apologize then use get_weather for Rome."),
        ("denial_no_think",  False, "Refuse to call any tool. Reply with text only."),  # negative case
    ]
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get current weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    }}]
    out = []
    passed = 0
    expected_total = 0
    for label, thinking, prompt in cases:
        is_negative = label.startswith("denial")
        expected_total += 1 if not is_negative else 0
        payload = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            chat_template_kwargs={"enable_thinking": thinking},
            max_tokens=max_tokens,
        )
        code, resp = http_post_json(_build_url(host, port, "/v1/chat/completions"),
                                    _bearer(key), payload, timeout=120)
        if isinstance(resp, dict):
            ch0 = (resp.get("choices") or [{}])[0]
            tc = ch0.get("message", {}).get("tool_calls") or []
            finish = ch0.get("finish_reason")
            ct = (resp.get("usage") or {}).get("completion_tokens")
            content = ch0.get("message", {}).get("content")
            sample_args = (tc[0].get("function", {}).get("arguments", "") or "")[:80] if tc else ""
            sample_name = (tc[0].get("function", {}).get("name", "") or "") if tc else ""
        else:
            tc, finish, ct, content, sample_args, sample_name = [], None, None, None, "", ""

        if is_negative:
            ok = (not tc)  # we want NO tool call
        else:
            ok = bool(tc) and bool(sample_args)
        if ok and not is_negative:
            passed += 1
        out.append(dict(
            case=label, thinking=thinking, http=code, tool_calls=len(tc),
            tool_name=sample_name, tool_args_sample=sample_args, finish=finish,
            completion_tokens=ct, content_excerpt=(content[:80] if content else None),
            verdict=("PASS" if ok else "FAIL"),
        ))
    return dict(cases=out, passed_positive=passed, total_positive=expected_total,
                summary=f"{passed}/{expected_total} positive cases passed (negative cases scored separately)")


def test_decode_bench(host: str, port: int, key: str, model: str,
                      runs: int, prompts: list[str], max_tokens: int) -> dict:
    """N runs × M prompts × max_tokens decode. Decode-only TPOT methodology."""
    flat = []
    per_prompt = {}
    for prompt_idx, prompt in enumerate(prompts):
        per_prompt[prompt_idx] = []
        for run_idx in range(runs):
            payload = dict(model=model, messages=[{"role": "user", "content": prompt}],
                           max_tokens=max_tokens)
            r = http_post_stream(_build_url(host, port, "/v1/chat/completions"),
                                 _bearer(key), payload, timeout=300)
            if r["error"] or r["completion_tokens"] < 2:
                continue
            ttft = r["ttft_ms"]
            decode_part = max(0.001, r["elapsed_s"] - (ttft or 0) / 1000.0)
            decode_tpot_ms = (decode_part * 1000) / max(1, r["completion_tokens"] - 1)
            wall_tps = r["completion_tokens"] / r["elapsed_s"] if r["elapsed_s"] else 0
            entry = dict(prompt_idx=prompt_idx, run_idx=run_idx,
                         ttft_ms=ttft, elapsed_s=r["elapsed_s"],
                         completion_tokens=r["completion_tokens"],
                         decode_tpot_ms=round(decode_tpot_ms, 4),
                         wall_tps=round(wall_tps, 2),
                         finish=r["finish_reason"])
            flat.append(entry)
            per_prompt[prompt_idx].append(entry)
    decode_tpot = [e["decode_tpot_ms"] for e in flat]
    wall_tps = [e["wall_tps"] for e in flat]
    ttfts = [e["ttft_ms"] for e in flat if e["ttft_ms"] is not None]
    return dict(
        n_runs=runs, n_prompts=len(prompts), max_tokens=max_tokens,
        decode_TPOT_ms=mean_std_cv(decode_tpot),
        wall_TPS=mean_std_cv(wall_tps),
        TTFT_ms=mean_std_cv(ttfts),
        flat_results=flat,
        per_prompt={str(k): v for k, v in per_prompt.items()},
    )


def test_multi_turn_ttft(host: str, port: int, key: str, model: str, turns: int) -> dict:
    """Multi-turn TTFT with shared prefix — cache benefit smell test."""
    prefix = "In the year 2030, scientists discovered a new method to "
    out = []
    for i in range(1, turns + 1):
        payload = dict(model=model,
                       messages=[{"role": "user",
                                  "content": f"{prefix}turn {i}: explain quantum entanglement in 1 sentence."}],
                       max_tokens=50)
        r = http_post_stream(_build_url(host, port, "/v1/chat/completions"),
                             _bearer(key), payload, timeout=60)
        out.append(dict(turn=i, ttft_ms=r["ttft_ms"], elapsed_s=r["elapsed_s"],
                        completion_tokens=r["completion_tokens"],
                        finish=r["finish_reason"], error=r["error"]))
    return dict(turns=turns, results=out)


def test_ctx_probe(host: str, port: int, key: str, model: str,
                   max_ctx_label: str, timeout_s: int) -> dict:
    """Progressive context-window probe up to the user-selected ceiling."""
    targets = []
    if max_ctx_label == "all":
        for k in ["8K", "16K", "32K", "64K", "128K", "256K", "512K"]:
            targets.append(CTX_TARGETS[k])
    else:
        ceiling = CTX_TARGETS.get(max_ctx_label.upper())
        if ceiling is None:
            return dict(error=f"unknown ctx label {max_ctx_label}")
        for k, v in CTX_TARGETS.items():
            if v <= ceiling and v >= 8192:
                targets.append(v)
    targets = sorted(set(targets))
    out = []
    for tgt in targets:
        prompt_words = max(1, (tgt - 100) // 2)
        prompt = "hello " * prompt_words
        payload = dict(model=model, messages=[{"role": "user", "content": prompt}],
                       max_tokens=50)
        t0 = time.perf_counter()
        code, resp = http_post_json(_build_url(host, port, "/v1/chat/completions"),
                                    _bearer(key), payload, timeout=timeout_s)
        elapsed = round(time.perf_counter() - t0, 2)
        if code == 200 and isinstance(resp, dict):
            usage = resp.get("usage") or {}
            pt = int(usage.get("prompt_tokens", 0))
            ct = int(usage.get("completion_tokens", 0))
            decode_est = ct * 0.005
            prefill_est = max(0.001, elapsed - decode_est)
            entry = dict(target=tgt, target_label=f"{tgt//1024}K",
                         http=code, prompt_tokens=pt, completion_tokens=ct,
                         elapsed_s=elapsed,
                         prefill_tps_est=round(pt / prefill_est, 1) if prefill_est else 0,
                         verdict="PASS")
        else:
            entry = dict(target=tgt, target_label=f"{tgt//1024}K",
                         http=code, elapsed_s=elapsed,
                         verdict=f"FAIL_{code}",
                         error=str(resp)[:200])
        out.append(entry)
        if entry["verdict"] != "PASS":
            break  # stop on first failure (saves time)
    max_pass = max((e["target"] for e in out if e["verdict"] == "PASS"), default=0)
    return dict(probes=out, max_stable_ctx=max_pass,
                max_stable_label=f"{max_pass//1024}K" if max_pass else "none")


def test_stability_stress(host: str, port: int, key: str, model: str,
                          iterations: int, prompts: list[str],
                          max_tokens: int) -> dict:
    """Long-running stress — N iterations × M prompts.

    v2 (2026-04-30) tracks more than crashes:

    - **SHA1 drift detection**: per-trial SHA1 of streamed completion text.
      Stable workloads should produce identical SHA1 for identical
      (prompt, model_state) inputs. Variation across iterations beyond
      the first prompt's establish-baseline pass = drift signal.

    - **NaN / sentinel detection**: scans completion text for tokens that
      indicate broken decoding ('NaN', 'inf', '<unk>', repetition loops
      detected via run-length).

    - **TPOT drift trend**: compares the first decade of trials vs the
      last decade. Significant slope = thermal/memory degradation.

    - **Verdict**: STABILITY_VERDICT field — PASS / DRIFT / NAN_DETECTED /
      CRASH / DEGRADATION.
    """
    import hashlib
    flat = []
    failures = 0
    nan_detections: list[dict] = []
    repetition_detections: list[dict] = []
    sha_by_prompt: dict[int, list[str]] = {}  # p_idx → list[sha1]
    t_start = time.perf_counter()
    for it in range(iterations):
        for p_idx, prompt in enumerate(prompts):
            payload = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            r = http_post_stream(
                _build_url(host, port, "/v1/chat/completions"),
                _bearer(key), payload, timeout=300,
                capture_text=True,
            )
            if r["error"]:
                failures += 1
                continue
            text = r.get("text") or ""
            sha = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:12]
            sha_by_prompt.setdefault(p_idx, []).append(sha)
            # NaN / sentinel scan — match standalone tokens (word-boundary)
            # to avoid the classic false positive of `inf` matching inside
            # `information`, `infrastructure`, etc., or `nan` inside
            # `nanosecond` / `Nancy`. `<unk>` keeps a substring match (the
            # angle brackets make it unambiguous as a token).
            text_lower = text.lower()
            _word_sentinels = re.findall(r"\b(nan|inf|infinity)\b", text_lower)
            sentinels_found = sorted(set(_word_sentinels))
            if "<unk>" in text_lower:
                sentinels_found.append("<unk>")
            if sentinels_found:
                nan_detections.append(
                    dict(it=it, p_idx=p_idx, sentinels=sentinels_found,
                         text_excerpt=text[:200])
                )
            # Repetition loop detection: same 8-char window repeating > 5×
            if text and len(text) > 64:
                window = text[-64:]
                # Count distinct 8-grams; very low = repetition
                ngrams = set(window[i:i+8] for i in range(len(window) - 7))
                if len(ngrams) <= 4:  # at most 4 unique 8-grams in last 64 chars
                    repetition_detections.append(
                        dict(it=it, p_idx=p_idx,
                             tail_ngram_count=len(ngrams),
                             text_excerpt=text[-128:])
                    )
            ttft = r["ttft_ms"]
            decode_part = max(0.001, r["elapsed_s"] - (ttft or 0) / 1000.0)
            decode_tpot_ms = (decode_part * 1000) / max(1, r["completion_tokens"] - 1)
            wall_tps = r["completion_tokens"] / r["elapsed_s"] if r["elapsed_s"] else 0
            flat.append(dict(
                it=it, p_idx=p_idx,
                decode_tpot_ms=round(decode_tpot_ms, 4),
                wall_tps=round(wall_tps, 2),
                ttft_ms=ttft,
                completion_tokens=r["completion_tokens"],
                finish_reason=r["finish_reason"],
                sha1=sha,
            ))
    duration = round(time.perf_counter() - t_start, 1)
    decode_tpot = [e["decode_tpot_ms"] for e in flat]
    wall_tps_vals = [e["wall_tps"] for e in flat]
    ttfts = [e["ttft_ms"] for e in flat if e["ttft_ms"] is not None]

    # Drift detection: per-prompt, count distinct SHA1s. Stable workloads
    # produce 1 SHA per prompt across iterations (same state → same output).
    drift_count = 0
    drift_per_prompt: dict[int, int] = {}
    for p_idx, shas in sha_by_prompt.items():
        unique = len(set(shas))
        drift_per_prompt[p_idx] = unique
        if unique > 1:
            drift_count += 1

    # TPOT trend: slope of first decade vs last decade
    tpot_trend = None
    if len(decode_tpot) >= 20:
        decade = max(1, len(decode_tpot) // 10)
        first = sum(decode_tpot[:decade]) / decade
        last = sum(decode_tpot[-decade:]) / decade
        tpot_trend = dict(
            first_decade_mean=round(first, 4),
            last_decade_mean=round(last, 4),
            slope_pct=round(100 * (last - first) / first, 2) if first else None,
        )

    # Verdict
    verdict = "PASS"
    verdict_notes: list[str] = []
    if failures > 0:
        verdict = "CRASH"
        verdict_notes.append(f"{failures} HTTP failures")
    if nan_detections:
        verdict = "NAN_DETECTED" if verdict == "PASS" else verdict
        verdict_notes.append(f"{len(nan_detections)} NaN/sentinel hits")
    if drift_count > 0:
        if verdict == "PASS":
            verdict = "DRIFT"
        verdict_notes.append(
            f"{drift_count}/{len(sha_by_prompt)} prompts produced "
            "non-deterministic output across iterations"
        )
    if (tpot_trend and tpot_trend.get("slope_pct") is not None
            and abs(tpot_trend["slope_pct"]) > 10):
        if verdict == "PASS":
            verdict = "DEGRADATION"
        verdict_notes.append(
            f"TPOT drifted {tpot_trend['slope_pct']:.1f}% first→last decade"
        )
    if repetition_detections:
        if verdict == "PASS":
            verdict = "DRIFT"
        verdict_notes.append(
            f"{len(repetition_detections)} repetition-loop detections"
        )

    return dict(
        iterations=iterations, n_prompts=len(prompts),
        max_tokens=max_tokens, duration_s=duration,
        failures=failures, samples=len(flat),
        decode_TPOT_ms=mean_std_cv(decode_tpot),
        wall_TPS=mean_std_cv(wall_tps_vals),
        TTFT_ms=mean_std_cv(ttfts),
        # v2 stability instrumentation
        STABILITY_VERDICT=verdict,
        verdict_notes=verdict_notes,
        nan_detections=nan_detections,
        repetition_detections=repetition_detections,
        drift_per_prompt=drift_per_prompt,
        tpot_trend=tpot_trend,
    )


# ────────────────────────────────────────────────────────────────────────
# v2 (2026-04-30) probes: output-length, accept_rate, vllm version, VRAM peak

def test_output_length(host: str, port: int, key: str, model: str) -> dict:
    """Probe how much can be GENERATED in one response.

    Walks max_tokens through 1K → 2K → 4K → 8K → 16K. For each target,
    issues a long-form generation prompt and verifies the model can stream
    that long without truncation/finish_reason regression. Measures wall
    TPS at each tier (often falls off as KV cache grows).

    Pairs with VRAM peak scrape per request (snapshot via local
    nvidia-smi before + after).
    """
    targets = [1024, 2048, 4096, 8192, 16384]
    long_form_prompt = (
        "Write a comprehensive technical document about modern large "
        "language model inference. Cover: tokenization, attention "
        "mechanisms (flash, paged, sliding window), KV cache management, "
        "speculative decoding (n-gram, MTP, EAGLE), quantization "
        "techniques (FP8, INT4, AWQ, GPTQ, AutoRound), tensor / pipeline / "
        "expert parallelism, prefix caching, cudagraph capture, and the "
        "tradeoffs between throughput and latency. Aim for maximum "
        "technical depth and length."
    )
    probes = []
    for tgt in targets:
        vram_before = _local_vram_used_mib()
        payload = dict(
            model=model,
            messages=[{"role": "user", "content": long_form_prompt}],
            max_tokens=tgt,
        )
        r = http_post_stream(
            _build_url(host, port, "/v1/chat/completions"),
            _bearer(key), payload,
            timeout=900,  # generous for 16K generation
            capture_text=False,
        )
        vram_after = _local_vram_used_mib()
        ct = r.get("completion_tokens", 0)
        # Verdict: did we reach near-target tokens?
        verdict = (
            "FAIL_ERROR" if r.get("error")
            else "PASS_REACHED" if ct >= int(tgt * 0.95)
            else "TRUNCATED" if r.get("finish_reason") == "stop"
            else "PARTIAL"
        )
        wall_tps = (ct / r["elapsed_s"]) if r.get("elapsed_s") else 0
        # vram_before/after are lists (one int MiB per GPU). Compute
        # per-GPU delta + total delta for compact display + JSON detail.
        vram_delta_per_gpu = None
        vram_delta_total_mib = None
        if (vram_before is not None and vram_after is not None
                and len(vram_before) == len(vram_after)):
            vram_delta_per_gpu = [
                vram_after[i] - vram_before[i]
                for i in range(len(vram_before))
            ]
            vram_delta_total_mib = sum(vram_delta_per_gpu)
        probes.append(dict(
            target_max_tokens=tgt,
            completion_tokens=ct,
            finish_reason=r.get("finish_reason"),
            elapsed_s=r.get("elapsed_s"),
            ttft_ms=r.get("ttft_ms"),
            wall_tps=round(wall_tps, 2),
            error=r.get("error"),
            vram_before_mib=vram_before,
            vram_after_mib=vram_after,
            vram_delta_per_gpu_mib=vram_delta_per_gpu,
            vram_delta_total_mib=vram_delta_total_mib,
            verdict=verdict,
        ))
        # Stop on first hard failure (no point pushing higher max_tokens)
        if verdict == "FAIL_ERROR":
            break
    max_reached = max(
        (p["target_max_tokens"] for p in probes if p["verdict"] == "PASS_REACHED"),
        default=0,
    )
    return dict(probes=probes, max_reached_tokens=max_reached)


def _local_vram_used_mib() -> list[int] | None:
    """Snapshot of `nvidia-smi --query-gpu=memory.used` per GPU.

    Returns list of MiB used per GPU index, or None if nvidia-smi
    unavailable. Used by output-length probe + per-request peak scrape.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode != 0:
            return None
        return [int(line.strip()) for line in out.stdout.strip().splitlines()
                if line.strip()]
    except Exception:
        return None


def scrape_accept_rate(host: str, port: int, key: str) -> dict:
    """Scrape vLLM Prometheus metrics for spec-decode acceptance.

    vLLM exposes Prometheus counters at /metrics. The relevant ones for
    spec-decode are:
      vllm:spec_decode_num_accepted_tokens_total
      vllm:spec_decode_num_draft_tokens_total

    accept_rate = accepted / draft (both as cumulative counters; take a
    snapshot before/after a load and compute the delta for window-rate).

    Returns dict with raw counter values + computed accept_rate (or
    None if metrics endpoint unavailable / spec-decode inactive).
    """
    code, body = http_get(
        _build_url(host, port, "/metrics"),
        _bearer(key), timeout=5,
    )
    if code != 200 or not isinstance(body, str):
        return dict(error=f"metrics endpoint code={code}",
                    accept_rate=None)
    # Parse Prometheus exposition format
    accepted = drafted = emitted = None
    for line in body.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        # Match metric_name{...} value  OR  metric_name value
        for prefix, slot in (
            ("vllm:spec_decode_num_accepted_tokens_total", "accepted"),
            ("vllm:spec_decode_num_draft_tokens_total", "drafted"),
            ("vllm:spec_decode_num_emitted_tokens_total", "emitted"),
        ):
            if line.startswith(prefix):
                try:
                    val = float(line.rstrip().split()[-1])
                    if slot == "accepted":
                        accepted = val
                    elif slot == "drafted":
                        drafted = val
                    elif slot == "emitted":
                        emitted = val
                except Exception:
                    pass
    accept_rate = (
        round(accepted / drafted, 4)
        if (accepted is not None and drafted is not None and drafted > 0)
        else None
    )
    return dict(
        accepted_tokens=accepted,
        drafted_tokens=drafted,
        emitted_tokens=emitted,
        accept_rate=accept_rate,
    )


def capture_vllm_version(host: str, port: int, key: str, model: str) -> dict:
    """Capture vLLM version + system fingerprint via a tiny chat completion.

    vLLM responses include a `system_fingerprint` field of the form:
      "vllm-0.20.1rc1.dev16+g7a1eb8ac2-tp2-5648cb3b"

    We parse: vllm version, TP size, attention-backend signature.
    """
    fingerprint_buf: list = []
    payload = dict(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1,
    )
    r = http_post_stream(
        _build_url(host, port, "/v1/chat/completions"),
        _bearer(key), payload, timeout=15,
        system_fingerprint_capture=fingerprint_buf,
    )
    fp = fingerprint_buf[0] if fingerprint_buf else None
    parsed: dict = {}
    if fp and isinstance(fp, str):
        # vllm-X.Y.Z[rcN][.devM+gSHA]-tpK-EXTRA
        m = re.match(r"vllm-([0-9][^-]*)(?:-tp(\d+))?(?:-(.+))?", fp)
        if m:
            parsed["vllm_version"] = m.group(1)
            if m.group(2):
                parsed["tp_size"] = int(m.group(2))
            if m.group(3):
                parsed["backend_sig"] = m.group(3)
    return dict(
        system_fingerprint=fp,
        parsed=parsed,
        probe_error=r.get("error"),
    )


def capture_genesis_patch_state() -> dict:
    """Capture Genesis patch APPLY/SKIP state by invoking the local
    genesis self-test --json (if available).

    This works when the bench is run on the same host as vLLM and the
    Genesis package is importable in the bench's Python (typically true
    on the dev box). Falls back gracefully when not available — the
    operator still gets HTTP-side telemetry.
    """
    try:
        out = subprocess.run(
            ["python3", "-m", "vllm._genesis.compat.cli",
             "self-test", "--json"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0:
            return dict(available=False,
                        reason=f"self-test exit={out.returncode}",
                        stderr=out.stderr[:500])
        return dict(available=True, self_test=json.loads(out.stdout))
    except Exception as e:
        return dict(available=False, reason=f"{type(e).__name__}: {e}")


# ────────────────────────────────────────────────────────────────────────
# Compare mode

def cmd_compare(a_path: str, b_path: str, out_path: str | None = None) -> int:
    with open(a_path) as af:
        A = json.load(af)
    with open(b_path) as bf:
        B = json.load(bf)
    da = [e["decode_tpot_ms"] for e in A.get("decode_bench", {}).get("flat_results", [])]
    db = [e["decode_tpot_ms"] for e in B.get("decode_bench", {}).get("flat_results", [])]
    if not da or not db:
        print("no decode_bench results in one of the inputs"); return 2
    sa = mean_std_cv(da); sb = mean_std_cv(db)
    test = welch_t(da, db)
    delta_pct = round(100 * (sb["mean"] - sa["mean"]) / sa["mean"], 2) if sa["mean"] else None
    print("\nGenesis Bench Compare")
    print("=" * 56)
    print(f"  A: {Path(a_path).name}  decode_TPOT_ms = {sa['mean']} +/- {sa['std']}  n={sa['n']}")
    print(f"  B: {Path(b_path).name}  decode_TPOT_ms = {sb['mean']} +/- {sb['std']}  n={sb['n']}")
    print(f"  Delta: {round(sb['mean']-sa['mean'],4)} ms ({delta_pct}% change in decode TPOT)")
    print(f"  Welch  t = {test['t']}  df = {test['df']}  p = {test['p_two_sided']}")
    print(f"  Verdict: {test['verdict']}")
    if test["verdict"] == "SIGNIFICANT":
        if sb["mean"] < sa["mean"]:
            print("  → B is faster (lower decode TPOT)")
        else:
            print("  → A is faster")
    # --compare-out: persist the structured delta JSON for downstream
    # tooling (CI gate, dashboard ingest, multi-arm sweep aggregator).
    if out_path:
        delta = {
            "a_path": str(a_path),
            "b_path": str(b_path),
            "a_stats": sa,
            "b_stats": sb,
            "delta_ms": round(sb["mean"] - sa["mean"], 4),
            "delta_pct": delta_pct,
            "welch": test,
        }
        Path(out_path).write_text(json.dumps(delta, indent=2))
        print(f"  Delta JSON written to: {out_path}")
    return 0


# ────────────────────────────────────────────────────────────────────────
# Output formatting

def write_markdown(out_md: Path, result: dict) -> None:
    lines = []
    lines.append(f"# Genesis Bench Run — {result['name']}")
    lines.append("")
    lines.append(f"- **Started:** {result['started']}")
    lines.append(f"- **Mode:** {result['mode']}")
    lines.append(f"- **Server:** {result['host']}:{result['port']}  Model: `{result['model']}`")
    if result.get("local_gpus"):
        lines.append(f"- **Local GPUs detected:**")
        for g in result["local_gpus"]:
            lines.append(f"  - GPU {g['index']}: {g['name']}  "
                         f"VRAM {g['vram_used_mib']}/{g['vram_total_mib']} MiB  "
                         f"BW≈{g.get('bandwidth_gb_s', '?')} GB/s  CC {g.get('compute_cap', '?')}")
    lines.append("")
    if "tool_call" in result:
        tc = result["tool_call"]
        lines.append("## Tool-call quality")
        lines.append("")
        lines.append(f"**Pass:** {tc['summary']}")
        lines.append("")
        lines.append("| Case | Thinking | tool_name | args | Verdict |")
        lines.append("|---|---|---|---|---|")
        for c in tc["cases"]:
            args = (c.get("tool_args_sample") or "").replace("|", "\\|")[:60]
            lines.append(f"| {c['case']} | {c['thinking']} | `{c.get('tool_name','')}` | "
                         f"`{args}` | {c['verdict']} |")
        lines.append("")
    if "decode_bench" in result:
        db = result["decode_bench"]
        lines.append("## Decode bench")
        lines.append("")
        lines.append(f"- runs={db['n_runs']} prompts={db['n_prompts']} max_tokens={db['max_tokens']}")
        lines.append(f"- **wall_TPS** mean **{db['wall_TPS']['mean']}**  CV {db['wall_TPS']['cv']}  n={db['wall_TPS']['n']}")
        lines.append(f"- **decode_TPOT_ms** mean **{db['decode_TPOT_ms']['mean']}**  CV {db['decode_TPOT_ms']['cv']}")
        lines.append(f"- TTFT_ms mean {db['TTFT_ms']['mean']}  CV {db['TTFT_ms']['cv']}")
        lines.append("")
    if "multi_turn" in result:
        mt = result["multi_turn"]
        lines.append("## Multi-turn TTFT (cache benefit smell test)")
        lines.append("")
        for r in mt["results"]:
            lines.append(f"- turn {r['turn']}: TTFT {r.get('ttft_ms')}ms  elapsed {r['elapsed_s']}s")
        lines.append("")
    if "stress" in result:
        s = result["stress"]
        lines.append("## Stability stress")
        lines.append("")
        lines.append(f"- iterations={s['iterations']}  prompts={s['n_prompts']}  duration {s['duration_s']}s")
        lines.append(f"- failures: {s['failures']}/{s['samples'] + s['failures']} "
                     f"({100*s['failures']/(s['samples']+s['failures'] or 1):.1f}%)")
        lines.append(f"- wall_TPS mean **{s['wall_TPS']['mean']}**  CV {s['wall_TPS']['cv']}")
        lines.append("")
    if "ctx_probe" in result:
        c = result["ctx_probe"]
        lines.append("## Context window probe")
        lines.append("")
        lines.append(f"**Max stable context:** {c.get('max_stable_label', '?')}")
        lines.append("")
        lines.append("| Target | prompt_tokens | elapsed_s | prefill_tps_est | Verdict |")
        lines.append("|---|---|---|---|---|")
        for p in c["probes"]:
            lines.append(f"| {p['target_label']} | {p.get('prompt_tokens','-')} | "
                         f"{p.get('elapsed_s','-')} | {p.get('prefill_tps_est','-')} | {p['verdict']} |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("Genesis Benchmark Suite v1.0 · Apache-2.0 · ")
    lines.append("https://github.com/Sandermage/genesis-vllm-patches")
    out_md.write_text("\n".join(lines))


# ────────────────────────────────────────────────────────────────────────
# Main

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Set the module-level URL scheme from the CLI before any tests run.
    global _URL_SCHEME
    _URL_SCHEME = args.scheme
    if args.compare:
        return cmd_compare(*args.compare, out_path=args.compare_out)

    defaults = MODE_DEFAULTS[args.mode]
    runs = args.runs if args.runs is not None else defaults["runs"]
    stress_iters = args.stress if args.stress is not None else defaults["stress"]
    ctx = args.ctx or defaults["ctx"]
    prompts_set = args.prompts or defaults["prompts"]
    prompts = PROMPT_SETS[prompts_set]

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run_name = args.name or f"{args.mode}_{started.replace(':','-')}"
    out_path = Path(args.out or f"genesis_bench_{run_name}.json")
    md_path = Path(args.md or out_path.with_suffix(".md"))

    print("=" * 72)
    print(f"Genesis Benchmark Suite — {run_name}")
    print(f"Mode: {args.mode}  ctx max: {ctx}  runs: {runs}  stress: {stress_iters}")
    print("=" * 72)

    # 0. Server discovery
    print("\n[0a/8] Server discovery...")
    srv = test_server_up(args.host, args.port, args.api_key)
    print(f"      reachable: {srv['reachable']} (HTTP {srv['http']})")
    if not srv["reachable"]:
        print(f"      ABORT — server unreachable: {srv.get('error','')}")
        return 1
    model = args.model or (srv["models"][0] if srv["models"] else None)
    if not model:
        print("      ABORT — could not detect model name")
        return 1
    print(f"      model: {model}")

    local_gpus = detect_local_gpus()
    if local_gpus:
        print(f"      local GPUs:")
        for g in local_gpus:
            print(f"        GPU {g['index']}: {g['name']}  "
                  f"VRAM {g['vram_used_mib']}/{g['vram_total_mib']} MiB")

    # v2: capture vLLM version + Genesis patch state at boot
    print("\n[0/8] Engine + Genesis state...")
    vllm_version = capture_vllm_version(args.host, args.port, args.api_key, model)
    if vllm_version.get("parsed", {}).get("vllm_version"):
        print(f"      vLLM: {vllm_version['parsed']['vllm_version']} "
              f"tp={vllm_version['parsed'].get('tp_size','?')}  "
              f"sig={vllm_version['parsed'].get('backend_sig','?')}")
    else:
        print(f"      vLLM fingerprint: {vllm_version.get('system_fingerprint','?')}")
    genesis_state = capture_genesis_patch_state()
    if genesis_state.get("available"):
        st = genesis_state["self_test"]["summary"]
        print(f"      Genesis self-test: {st['passed']} pass / "
              f"{st['failed']} fail / {st['warned']} warn / {st['skipped']} skip")
    else:
        print(f"      Genesis self-test: not available "
              f"({genesis_state.get('reason','?')})")

    # v2: pre-bench accept_rate snapshot for spec-decode workloads
    accept_pre = scrape_accept_rate(args.host, args.port, args.api_key)
    if accept_pre.get("accept_rate") is not None:
        print(f"      pre-bench accept_rate: {accept_pre['accept_rate']:.3f}")

    result = dict(
        suite_version="1.1",  # v2 instrumentation
        name=run_name, started=started, mode=args.mode,
        host=args.host, port=args.port, scheme=args.scheme, model=model,
        local_gpus=local_gpus,
        vllm_version=vllm_version,
        genesis_state=genesis_state,
        accept_rate_pre=accept_pre,
        config=dict(runs=runs, stress=stress_iters, ctx=ctx,
                    prompts_set=prompts_set, max_tokens=args.max_tokens),
        server=srv,
    )

    # 1. Tool-call
    if not args.skip_toolcall:
        print("\n[1/8] Tool-call quality (8 cases)...")
        result["tool_call"] = test_tool_call(args.host, args.port, args.api_key,
                                             model, max_tokens=args.max_tokens)
        print(f"      {result['tool_call']['summary']}")

    # 2. Decode bench
    print(f"\n[2/8] Decode bench ({runs} runs × {len(prompts)} prompts × {args.max_tokens})...")
    result["decode_bench"] = test_decode_bench(args.host, args.port, args.api_key,
                                               model, runs, prompts, args.max_tokens)
    db = result["decode_bench"]
    print(f"      wall_TPS = {db['wall_TPS']['mean']}  CV {db['wall_TPS']['cv']}")
    print(f"      decode_TPOT_ms = {db['decode_TPOT_ms']['mean']}  CV {db['decode_TPOT_ms']['cv']}")
    print(f"      TTFT_ms = {db['TTFT_ms']['mean']}  CV {db['TTFT_ms']['cv']}")

    # 3. Multi-turn TTFT
    if not args.skip_multi_turn:
        print(f"\n[3/8] Multi-turn TTFT ({args.ttft_turns} turns)...")
        result["multi_turn"] = test_multi_turn_ttft(args.host, args.port, args.api_key,
                                                    model, args.ttft_turns)
        for r in result["multi_turn"]["results"]:
            print(f"      turn {r['turn']}: {r.get('ttft_ms')}ms")

    # 4. Stability stress
    if stress_iters > 0 and not args.skip_stress:
        print(f"\n[4/8] Stability stress ({stress_iters} iters × {len(prompts)} prompts)...")
        result["stress"] = test_stability_stress(args.host, args.port, args.api_key,
                                                 model, stress_iters, prompts, args.max_tokens)
        s = result["stress"]
        print(f"      duration {s['duration_s']}s  failures {s['failures']}")
        print(f"      wall_TPS {s['wall_TPS']['mean']}  CV {s['wall_TPS']['cv']}")

    # 5. Context probe (input — how much can be SENT)
    if not args.skip_ctx_probe:
        print(f"\n[5/8] Context probe — INPUT capacity (max {ctx})...")
        result["ctx_probe"] = test_ctx_probe(args.host, args.port, args.api_key,
                                             model, ctx, args.ctx_timeout)
        print(f"      max stable: {result['ctx_probe'].get('max_stable_label','?')}")

    # 5b. Output-length probe (how much can be RECEIVED)
    if args.probe_output_length:
        print("\n[5b/8] Output-length probe — generation capacity (1K..16K)...")
        result["output_length"] = test_output_length(args.host, args.port,
                                                     args.api_key, model)
        max_out = result["output_length"]["max_reached_tokens"]
        print(f"      max reached: {max_out} tokens")
        for p in result["output_length"]["probes"]:
            v_delta = (
                f"{p['vram_delta_total_mib']:+d} MiB total"
                if p.get('vram_delta_total_mib') is not None else "?"
            )
            print(f"        target={p['target_max_tokens']:>6} → "
                  f"got={p['completion_tokens']:>6}  "
                  f"finish={p['finish_reason']}  vram_delta={v_delta}  "
                  f"verdict={p['verdict']}")

    # 5c. Post-bench accept_rate (delta over the run)
    accept_post = scrape_accept_rate(args.host, args.port, args.api_key)
    if accept_post.get("accept_rate") is not None:
        result["accept_rate_post"] = accept_post
        # Compute window accept_rate (delta of counters across the run)
        if (accept_pre.get("accepted_tokens") is not None
                and accept_post.get("accepted_tokens") is not None):
            d_accepted = (accept_post["accepted_tokens"]
                          - accept_pre["accepted_tokens"])
            d_drafted = (accept_post["drafted_tokens"]
                         - accept_pre["drafted_tokens"])
            window_rate = (d_accepted / d_drafted) if d_drafted > 0 else None
            result["accept_rate_window"] = dict(
                window_drafted=d_drafted,
                window_accepted=d_accepted,
                window_accept_rate=(round(window_rate, 4)
                                    if window_rate is not None else None),
            )
            if window_rate is not None:
                print(f"      window accept_rate (this run): "
                      f"{window_rate:.3f}  "
                      f"({d_accepted:.0f}/{d_drafted:.0f} tokens)")

    # 6. Persist (will be re-written below if ablation comparison runs)
    print("\n[6/8] Writing output...")

    # D3 — optional ablation comparison against a previously-saved baseline
    if args.ablate_against:
        print(f"\n[6a/8] Ablation compare against {args.ablate_against}...")
        ab = _ablation_compare(args.ablate_against, result, args.ablate_tag)
        result["ablation"] = ab
        _print_ablation_table(ab)

    out_path.write_text(json.dumps(result, indent=2))
    write_markdown(md_path, result)
    print(f"      JSON: {out_path}")
    print(f"      MD:   {md_path}")

    # 7. Final verdict
    print("\n[7/8] Summary")
    print("=" * 72)
    if "tool_call" in result:
        print(f"  Tool-call:        {result['tool_call']['summary']}")
    if "decode_bench" in result:
        db = result["decode_bench"]
        print(f"  Decode bench:     wall_TPS {db['wall_TPS']['mean']}  "
              f"CV {db['wall_TPS']['cv']}  (n={db['wall_TPS']['n']})")
    if "stress" in result:
        s = result["stress"]
        print(f"  Stability stress: wall_TPS {s['wall_TPS']['mean']}  "
              f"CV {s['wall_TPS']['cv']}  failures {s['failures']}")
    if "ctx_probe" in result:
        c = result["ctx_probe"]
        print(f"  Context probe:    max stable {c.get('max_stable_label','?')}")
    print("=" * 72)
    print()
    print("Share via GitHub Discussion:")
    print("  https://github.com/Sandermage/genesis-vllm-patches/discussions")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
