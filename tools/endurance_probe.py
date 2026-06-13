#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Genesis endurance probe — multi-hour VRAM / RSS / KV-creep sampler.

Port of the ``vram_probe.py`` sidecar from upstream PR vllm#45022
(Voxtral realtime RFC; the feature itself fails all three gates on our
stack — the probe is the part worth keeping, per the 2026-06-11 50-PR
sweep roadmap chunk-3 Theme D). It catches the day-later-OOM class on
the 24 GB rig: a slow VRAM / EngineCore-RSS / KV-usage creep that a
5-minute smoke can never expose, surfacing it BEFORE a candidate pin is
promoted (docs/PIN_BUMP_PLAYBOOK.md, endurance step).

What it samples, once per ``--interval`` seconds:

- per-GPU VRAM used (``nvidia-smi`` CLI; graceful ``-1`` when absent)
- summed RSS of all EngineCore worker processes (``ps``)
- ``kv_cache_usage_perc`` / ``num_requests_running`` from ``/metrics``
- host MemAvailable (``/proc/meminfo``; ``-1`` on non-Linux)

Output: one JSON object per sample (JSONL, flushed every sample so a
killed run still leaves data) plus ``<output>.summary.json`` with
per-metric first/last/min/max and a least-squares slope per hour.
Verdict: ``PASS`` / ``CREEP`` / ``INSUFFICIENT_SAMPLES``; exit code 1
on CREEP so playbook automation can gate on it.

Genesis upgrades over the upstream sidecar: argparse CLI
(``--interval --duration --port --host --output`` + thresholds), JSONL
instead of CSV, multi-GPU VRAM (the upstream probe reads GPU 0 only —
wrong for the 2x A5000 TP=2 rig), sentinel-aware slope math, and the
machine-readable creep verdict.

Dependencies: pure stdlib + ``requests`` (urllib fallback). NO GPU
libraries — no torch, no pynvml; safe to run beside a loaded server.

Usage:
    python3 tools/endurance_probe.py --port 8101 --interval 30 \
        --duration 14400 --output /tmp/endurance_35b.jsonl

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Upstream provenance: https://github.com/vllm-project/vllm/pull/45022
(OPEN at port time, 2026-06-11; benchmarks/voxtral_realtime/vram_probe.py).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request

try:
    import requests
except ImportError:  # pragma: no cover - exercised on requests-less hosts
    requests = None  # type: ignore[assignment]

# Creep thresholds (per hour). VRAM/RSS in MiB/h; KV usage is the 0..1
# fraction from /metrics, so 0.005/h == +0.5 percentage points hourly.
DEFAULT_VRAM_MIB_PER_HOUR = 64.0
DEFAULT_RSS_MIB_PER_HOUR = 64.0
DEFAULT_KV_USAGE_PER_HOUR = 0.005

# A verdict needs enough signal: at least this many samples AND span.
MIN_SAMPLES_FOR_VERDICT = 10
MIN_SPAN_S_FOR_VERDICT = 600.0


def nvidia_vram_mib() -> list[int]:
    """Per-GPU used VRAM in MiB via the nvidia-smi CLI ([] on failure)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    values: list[int] = []
    for line in out.strip().splitlines():
        try:
            values.append(int(line.strip()))
        except ValueError:
            continue
    return values


def enginecore_rss_mib() -> float:
    """Summed RSS (MiB) of all EngineCore worker processes."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "rss,args"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return -1.0
    total_kib = 0
    for line in out.splitlines():
        if "EngineCore" in line or "from multiprocessing.spawn" in line:
            m = re.match(r"\s*(\d+)\s", line)
            if m:
                total_kib += int(m.group(1))
    return round(total_kib / 1024, 1)


def host_avail_mib() -> int:
    """Host MemAvailable in MiB (-1 when /proc/meminfo is unavailable)."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


def parse_metrics(body: str) -> tuple[float, float]:
    """Extract (kv_cache_usage_perc, num_requests_running) from a
    Prometheus /metrics body. Missing metrics return -1.0."""
    kv = running = -1.0
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        if "kv_cache_usage_perc" in line or "gpu_cache_usage_perc" in line:
            try:
                kv = float(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                pass
        elif "num_requests_running" in line:
            try:
                running = float(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                pass
    return kv, running


def fetch_metrics(host: str, port: int, timeout: float = 5.0) -> tuple[float, float]:
    """GET /metrics and parse it ((-1, -1) on any transport failure)."""
    url = f"http://{host}:{port}/metrics"
    try:
        if requests is not None:
            body = requests.get(url, timeout=timeout).text
        else:  # pragma: no cover - exercised on requests-less hosts
            body = urllib.request.urlopen(url, timeout=timeout).read().decode()
    except Exception:  # noqa: BLE001 - any transport failure is a soft miss
        return -1.0, -1.0
    return parse_metrics(body)


def slope_per_hour(points: list[tuple[float, float]]) -> float:
    """Least-squares slope of (elapsed_s, value) points, per hour."""
    n = len(points)
    if n < 2:
        return 0.0
    mean_t = sum(p[0] for p in points) / n
    mean_v = sum(p[1] for p in points) / n
    denom = sum((p[0] - mean_t) ** 2 for p in points)
    if denom == 0.0:
        return 0.0
    num = sum((p[0] - mean_t) * (p[1] - mean_v) for p in points)
    return (num / denom) * 3600.0


def take_sample(elapsed_s: float, host: str, port: int) -> dict:
    per_gpu = nvidia_vram_mib()
    kv, running = fetch_metrics(host, port)
    return {
        "elapsed_s": round(elapsed_s, 1),
        "vram_mib_per_gpu": per_gpu,
        "vram_mib_total": float(sum(per_gpu)) if per_gpu else -1.0,
        "enginecore_rss_mib": enginecore_rss_mib(),
        "kv_usage": kv,
        "running": running,
        "host_avail_mib": host_avail_mib(),
    }


def build_summary(
    samples: list[dict],
    *,
    vram_thresh: float = DEFAULT_VRAM_MIB_PER_HOUR,
    rss_thresh: float = DEFAULT_RSS_MIB_PER_HOUR,
    kv_thresh: float = DEFAULT_KV_USAGE_PER_HOUR,
    min_samples: int = MIN_SAMPLES_FOR_VERDICT,
    min_span_s: float = MIN_SPAN_S_FOR_VERDICT,
) -> dict:
    """Per-metric stats + slope/hour + creep verdict over the samples.

    Negative values are probe-failure sentinels and are excluded from
    the per-metric series so a flaky nvidia-smi cannot fake a slope.
    """
    thresholds = {
        "vram_mib_total": vram_thresh,
        "enginecore_rss_mib": rss_thresh,
        "kv_usage": kv_thresh,
    }
    metrics: dict[str, dict] = {}
    flagged: list[str] = []
    span_s = samples[-1]["elapsed_s"] - samples[0]["elapsed_s"] if samples else 0.0
    enough = len(samples) >= min_samples and span_s >= min_span_s

    for name, thresh in thresholds.items():
        points = [
            (float(s["elapsed_s"]), float(s[name]))
            for s in samples
            if float(s.get(name, -1.0)) >= 0.0
        ]
        if not points:
            metrics[name] = {
                "first": None,
                "last": None,
                "min": None,
                "max": None,
                "slope_per_hour": 0.0,
                "threshold_per_hour": thresh,
                "flagged": False,
            }
            continue
        slope = slope_per_hour(points)
        is_flagged = enough and slope > thresh
        metrics[name] = {
            "first": points[0][1],
            "last": points[-1][1],
            "min": min(v for _, v in points),
            "max": max(v for _, v in points),
            "slope_per_hour": slope,
            "threshold_per_hour": thresh,
            "flagged": is_flagged,
        }
        if is_flagged:
            flagged.append(name)

    creep = bool(flagged)
    if not enough:
        verdict = "INSUFFICIENT_SAMPLES"
    elif creep:
        verdict = "CREEP"
    else:
        verdict = "PASS"
    return {
        "num_samples": len(samples),
        "span_s": span_s,
        "metrics": metrics,
        "flagged_metrics": flagged,
        "creep_detected": creep,
        "verdict": verdict,
    }


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-hour VRAM/RSS/KV-creep endurance probe for a running "
            "vLLM server (port of the vllm#45022 vram_probe sidecar)."
        )
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=15.0,
        help="seconds between samples (default: 15)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=14400.0,
        help="total probe duration in seconds (default: 14400 = 4 h)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM HTTP port for /metrics (default: 8000)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="vLLM HTTP host for /metrics (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "JSONL output path (default: endurance_probe_<ts>.jsonl); the "
            "summary is written to <output>.summary.json"
        ),
    )
    parser.add_argument(
        "--vram-thresh-mib-per-hour",
        type=float,
        default=DEFAULT_VRAM_MIB_PER_HOUR,
        help="VRAM creep threshold in MiB/h (default: 64)",
    )
    parser.add_argument(
        "--rss-thresh-mib-per-hour",
        type=float,
        default=DEFAULT_RSS_MIB_PER_HOUR,
        help="EngineCore RSS creep threshold in MiB/h (default: 64)",
    )
    parser.add_argument(
        "--kv-thresh-per-hour",
        type=float,
        default=DEFAULT_KV_USAGE_PER_HOUR,
        help="KV usage creep threshold per hour, 0..1 scale (default: 0.005)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_arg_parser().parse_args(argv)
    output = args.output or time.strftime("endurance_probe_%Y%m%d_%H%M%S.jsonl")
    summary_path = output + ".summary.json"

    samples: list[dict] = []
    t0 = time.perf_counter()
    print(
        f"endurance_probe: sampling every {args.interval:.1f}s for "
        f"{args.duration:.0f}s against {args.host}:{args.port} -> {output}",
        flush=True,
    )
    try:
        with open(output, "w", encoding="utf-8") as f:
            while True:
                elapsed = time.perf_counter() - t0
                if elapsed > args.duration:
                    break
                sample = take_sample(elapsed, args.host, args.port)
                samples.append(sample)
                f.write(json.dumps(sample) + "\n")
                f.flush()
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("endurance_probe: interrupted - summarizing partial run", flush=True)

    summary = build_summary(
        samples,
        vram_thresh=args.vram_thresh_mib_per_hour,
        rss_thresh=args.rss_thresh_mib_per_hour,
        kv_thresh=args.kv_thresh_per_hour,
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(
        f"endurance_probe: {summary['verdict']} "
        f"({summary['num_samples']} samples over {summary['span_s']:.0f}s)",
        flush=True,
    )
    for name, m in summary["metrics"].items():
        print(
            f"  {name}: first={m['first']} last={m['last']} "
            f"slope/h={m['slope_per_hour']:.3f} "
            f"(threshold {m['threshold_per_hour']}) "
            f"{'FLAGGED' if m['flagged'] else 'ok'}",
            flush=True,
        )
    print(f"endurance_probe: summary written to {summary_path}", flush=True)
    return 1 if summary["creep_detected"] else 0


if __name__ == "__main__":
    sys.exit(main())
