# SPDX-License-Identifier: Apache-2.0
"""Verify — bench running config, diff vs reference_metrics.

Layer 5 of the "100% close gaps" strategy. Catches drift over time:
patcher updated, vllm pin bumped, hardware swapped, but reference
hasn't been re-bench'd. Exit 1 if any verify_tolerance violated.

This is the canonical CI gate — `genesis model-config verify <key>`
should be green for every shipped builtin/community config.
"""
from __future__ import annotations

import json
import re
import statistics
import subprocess
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class VerifyResult:
    metric: str
    expected: str
    actual: str
    delta: str
    passed: bool
    severity: str  # 'error' / 'warning' / 'info'


def _run(cmd: list[str], timeout: int = 60) -> tuple[int, str, str]:
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        return p.returncode, p.stdout, p.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return -1, "", "command failed"


def _curl_post(url: str, headers: dict, body: dict, timeout: int = 60) -> dict:
    """Lightweight JSON POST returning parsed response."""
    cmd = ["curl", "-s", "-m", str(timeout)]
    for k, v in headers.items():
        cmd.extend(["-H", f"{k}: {v}"])
    cmd.extend(["-X", "POST", url, "-d", json.dumps(body)])
    rc, out, _ = _run(cmd, timeout=timeout + 5)
    if rc != 0:
        return {}
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {}


def _bench_long_gen(cfg, port: int, n_runs: int = 3) -> dict:
    """Run `n_runs` long_gen requests, return mean TPS + latency."""
    url = f"http://localhost:{port}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": cfg.served_model_name or "model",
        "messages": [{
            "role": "user",
            "content": "Write a detailed explanation of how transformers "
                       "work in neural networks. Include attention, "
                       "positional encoding, multi-head, layer norm, FFN.",
        }],
        "max_tokens": 1000,
        "temperature": 0,
    }
    times: list[float] = []
    tokens: list[int] = []
    for i in range(n_runs):
        t0 = time.time()
        resp = _curl_post(url, headers, body, timeout=120)
        t1 = time.time()
        if not resp:
            continue
        usage = resp.get("usage") or {}
        completion_toks = usage.get("completion_tokens", 0)
        if completion_toks > 0:
            times.append(t1 - t0)
            tokens.append(completion_toks)
    if not times:
        return {"error": "all bench requests failed"}
    return {
        "mean_lat_s": statistics.mean(times),
        "tok_avg": statistics.mean(tokens),
        "sustained_tps": statistics.mean(tokens) / statistics.mean(times),
    }


def _bench_tool_call(cfg, port: int, n_runs: int = 10) -> dict:
    """Tool-call quality bench."""
    url = f"http://localhost:{port}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": cfg.served_model_name or "model",
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
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
        # 800 (not 200) — Qwen3-thinking models emit reasoning ~250-470
        # tokens before tool_call; max_tokens=200 truncated some runs at
        # finish_reason=length BEFORE tool_call. Investigation 2026-05-07
        # found reasoning_len up to 471 in production runs; 800 budget
        # gives safe margin for outliers.
        "max_tokens": 800,
        "temperature": 0,
    }
    # Warmup: send 1 call to trigger CG capture for tool-call request shape
    # (different from plain chat — tool definitions in input change CG path).
    # Otherwise the first 'real' run can timeout / partial-output.
    _curl_post(url, headers, body, timeout=120)
    success = 0
    for _ in range(n_runs):
        resp = _curl_post(url, headers, body, timeout=120)
        if not resp:
            continue
        try:
            tc = (resp["choices"][0]["message"].get("tool_calls") or [])
            if tc and tc[0]["function"]["name"] == "get_weather":
                success += 1
        except (KeyError, IndexError):
            pass
    return {"success": success, "total": n_runs}


def _bench_stability(cfg, port: int, n_runs: int = 5) -> dict:
    """Same query × N times — measure CV of latency."""
    url = f"http://localhost:{port}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": cfg.served_model_name or "model",
        "messages": [{"role": "user", "content": "Count from 1 to 50."}],
        "max_tokens": 300,
        "temperature": 0,
    }
    times: list[float] = []
    for _ in range(n_runs):
        t0 = time.time()
        _curl_post(url, headers, body, timeout=30)
        t1 = time.time()
        times.append(t1 - t0)
    if len(times) < 2:
        return {"error": "stability bench failed"}
    mean = statistics.mean(times)
    std = statistics.stdev(times)
    return {
        "mean_s": mean,
        "cv_pct": 100 * std / mean if mean > 0 else 0.0,
    }


def _vram_usage(cfg, total_only: bool = False) -> dict:
    """Read nvidia-smi for current VRAM usage."""
    rc, out, _ = _run([
        "nvidia-smi", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ])
    if rc != 0:
        return {}
    per_gpu = [int(ln.strip()) for ln in out.strip().splitlines() if ln.strip()][
        : cfg.hardware.n_gpus
    ]
    return {
        "per_gpu_mib": per_gpu,
        "total_mib": sum(per_gpu),
    }


# ─── Verify ────────────────────────────────────────────────────────────


def _parse_tool_score(s: str) -> tuple[int, int]:
    m = re.match(r"^(\d+)\s*/\s*(\d+)$", s.strip())
    if not m:
        return (0, 1)
    return (int(m.group(1)), int(m.group(2)))


def verify(cfg, port: Optional[int] = None) -> list[VerifyResult]:
    """Run abbreviated bench + diff vs reference_metrics.

    Returns list of VerifyResult (one per metric checked). Caller can
    scan for `not r.passed and r.severity == "error"` to gate CI.
    """
    out: list[VerifyResult] = []
    if cfg.reference_metrics is None:
        out.append(VerifyResult(
            metric="reference_metrics", expected="present", actual="None",
            delta="—", passed=False, severity="error",
        ))
        return out

    p = port or (cfg.docker.port if cfg.docker else 8000)
    rm = cfg.reference_metrics
    tol = cfg.verify_tolerances

    # ── long_gen sustained TPS ──
    bench = _bench_long_gen(cfg, p, n_runs=3)
    if "error" in bench:
        out.append(VerifyResult(
            metric="long_gen_tps", expected=str(rm.long_gen_sustained_tps),
            actual="bench failed", delta="—",
            passed=False, severity="error",
        ))
    else:
        actual_tps = bench["sustained_tps"]
        drop_pct = 100 * (rm.long_gen_sustained_tps - actual_tps) / \
            rm.long_gen_sustained_tps if rm.long_gen_sustained_tps > 0 else 0
        passed = drop_pct <= tol.tps_drop_pct_max
        out.append(VerifyResult(
            metric="long_gen_tps",
            expected=f"{rm.long_gen_sustained_tps:.1f}",
            actual=f"{actual_tps:.1f}",
            delta=f"{-drop_pct:+.1f}% (tolerance ±{tol.tps_drop_pct_max}%)",
            passed=passed,
            severity="error" if not passed else "info",
        ))

    # ── tool_call quality ──
    tc = _bench_tool_call(cfg, p, n_runs=10)
    if "success" in tc:
        actual_score = f"{tc['success']}/{tc['total']}"
        ref_n, ref_d = _parse_tool_score(rm.tool_call_score)
        min_n, min_d = _parse_tool_score(tol.tool_call_min)
        actual_pct = tc["success"] / tc["total"]
        min_pct = min_n / min_d
        passed = actual_pct >= min_pct
        out.append(VerifyResult(
            metric="tool_call",
            expected=rm.tool_call_score,
            actual=actual_score,
            delta=f"min ≥ {tol.tool_call_min}",
            passed=passed,
            severity="error" if not passed else "info",
        ))

    # ── stability CV ──
    stab = _bench_stability(cfg, p, n_runs=5)
    if "cv_pct" in stab:
        actual_cv = stab["cv_pct"]
        passed = actual_cv <= tol.stability_cv_pct_max
        out.append(VerifyResult(
            metric="stability_cv",
            expected=f"{rm.stability_cv_pct:.2f}%",
            actual=f"{actual_cv:.2f}%",
            delta=f"max ≤ {tol.stability_cv_pct_max}%",
            passed=passed,
            severity="warning" if not passed else "info",
        ))

    # ── VRAM ──
    vram = _vram_usage(cfg)
    if "total_mib" in vram:
        actual_total = vram["total_mib"]
        delta_mib = actual_total - rm.vram_total_mib
        passed = delta_mib <= tol.vram_increase_mib_max
        out.append(VerifyResult(
            metric="vram_total",
            expected=f"{rm.vram_total_mib} MiB",
            actual=f"{actual_total} MiB",
            delta=f"{delta_mib:+d} MiB (max +{tol.vram_increase_mib_max})",
            passed=passed,
            severity="warning" if not passed else "info",
        ))

    return out


def has_blockers(results: list[VerifyResult]) -> bool:
    return any(not r.passed and r.severity == "error" for r in results)


def bench_metrics(cfg, port: Optional[int] = None) -> dict:
    """Run all benches against a live config and return raw metrics.

    Unlike `verify()` this does NOT require pre-existing reference_metrics
    — used by `bench-and-update` to capture a fresh baseline on a config
    where reference is null. Returns a flat dict keyed for direct write
    into ReferenceMetrics fields. On per-bench failure the corresponding
    keys are absent (operator can re-run rather than overwriting reference
    with bogus zeros).
    """
    p = port or (cfg.docker.port if cfg.docker else 8000)
    out: dict = {}

    lg = _bench_long_gen(cfg, p, n_runs=3)
    if "sustained_tps" in lg:
        out["long_gen_sustained_tps"] = round(lg["sustained_tps"], 1)
        out["long_gen_mean_lat_s"] = round(lg["mean_lat_s"], 2)

    tc = _bench_tool_call(cfg, p, n_runs=10)
    if "success" in tc:
        out["tool_call_score"] = f"{tc['success']}/{tc['total']}"

    stab = _bench_stability(cfg, p, n_runs=5)
    if "cv_pct" in stab:
        out["stability_mean_s"] = round(stab["mean_s"], 3)
        out["stability_cv_pct"] = round(stab["cv_pct"], 2)

    vram = _vram_usage(cfg)
    if "total_mib" in vram:
        out["vram_used_mib_per_gpu"] = vram["per_gpu_mib"]
        out["vram_total_mib"] = vram["total_mib"]

    return out
