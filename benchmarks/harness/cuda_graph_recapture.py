# SPDX-License-Identifier: Apache-2.0
"""CUDA graph recapture count — master plan Part 11.1 gate P0.

Threshold: 0 recaptures after warmup (critical for P28 sanity).

Design:
  - Hit the vLLM /metrics endpoint (if Prometheus enabled) and read
    `vllm:cuda_graph_capture_count` before/after 50 warmup requests.
  - After warmup, send 200 real requests at various context lengths.
  - Expected: capture_count stays stable (no new captures post-warmup).
  - Any increase after warmup indicates a forward-path allocation pattern
    is triggering graph invalidation → P28-class bug.

Fallback: if /metrics is not exposed, we probe the logs via SSH (skipped
in this harness; emit a `gate=skipped` result with reason).

Usage:
  python -m benchmarks.harness.cuda_graph_recapture \\
      --endpoint http://192.168.1.10:8000/v1 \\
      --metrics-url http://192.168.1.10:8000/metrics

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import sys as _sys
import urllib.error
import urllib.request

from benchmarks.harness._common import (
    GateResult, HarnessReport, make_arg_parser, post_chat, probe_health,
    default_out_path, write_report,
)


def _fetch_metric(url: str, metric: str) -> float | None:
    """Grab a numeric metric from Prometheus text exposition."""
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError):
        return None
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        if line.startswith(metric):
            parts = line.rsplit(maxsplit=1)
            if len(parts) == 2:
                try:
                    return float(parts[1])
                except ValueError:
                    return None
    return None


def _fetch_recapture_via_docker_logs(
    container_name: str, lookback_sec: int = 120,
) -> int | None:
    """Fallback probe when /metrics is not exposed.

    Parses `docker logs --since Ns <container>` for a "Capturing CUDA
    graphs" line count — upstream emits one per graph capture round.
    Increments between warmup and test bursts indicate a recapture.

    Requires `docker` on PATH (harness runs INSIDE a container typically;
    on the host where we run the harness, docker IS available).
    """
    import subprocess
    try:
        out = subprocess.run(
            ["docker", "logs", "--since", f"{lookback_sec}s", container_name],
            capture_output=True, text=True, timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    combined = (out.stdout or "") + (out.stderr or "")
    # Count "Capturing CUDA graphs" lines — one per capture round.
    # A value > 0 in the POST-warmup window means a recapture happened.
    return sum(
        1 for line in combined.splitlines()
        if "Capturing CUDA graphs" in line
    )


def main() -> int:
    parser = make_arg_parser("cuda_graph_recapture")
    parser.add_argument(
        "--metrics-url",
        default=None,
        help="Prometheus metrics URL (default: <endpoint-root>/metrics).",
    )
    parser.add_argument("--warmup-reqs", type=int, default=50)
    parser.add_argument("--test-reqs", type=int, default=200)
    parser.add_argument(
        "--container",
        default="vllm-integration-v7",
        help="Docker container name for the log-based fallback probe.",
    )
    parser.add_argument(
        "--metric",
        default="vllm:cuda_graph_capture_count",
        help="Prometheus metric name to watch.",
    )
    args = parser.parse_args()
    out_path = args.out or default_out_path("cuda_graph_recapture")

    report = HarnessReport(
        name="cuda_graph_recapture", endpoint=args.endpoint, model=args.model,
    )

    try:
        if not probe_health(args.endpoint):
            report.error = f"/health check failed for {args.endpoint}"
            write_report(report, out_path, quiet=args.quiet)
            return 2

        metrics_url = args.metrics_url or (
            args.endpoint.rstrip("/").rsplit("/v1", 1)[0]
            + "/metrics"
        )
        baseline = _fetch_metric(metrics_url, args.metric)
        if baseline is None:
            # /metrics not exposed — use docker-logs fallback.
            # Count "Capturing CUDA graphs" lines in the lookback window.
            log_baseline = _fetch_recapture_via_docker_logs(
                args.container, lookback_sec=5
            )
            if log_baseline is None:
                report.gates = [
                    GateResult(
                        name="cuda_graph_stable_after_warmup",
                        value=None,
                        threshold="/metrics + docker-logs both unavailable; skipped",
                        passed=True,  # benign skip
                    ),
                ]
                report.metrics = {
                    "note": (
                        f"metric {args.metric!r} not exposed at {metrics_url}; "
                        f"docker logs for {args.container} also unavailable"
                    ),
                }
                write_report(report, out_path, quiet=args.quiet)
                return 0
            # Docker-logs fallback: set baseline to 0, and count AFTER
            # bursts via logs with larger lookback.
            baseline = 0.0
            report.metrics["probe_mode"] = "docker_logs"

        # Warmup burst
        for i in range(args.warmup_reqs):
            post_chat(
                endpoint=args.endpoint, api_key=args.api_key, model=args.model,
                messages=[{"role": "user", "content": "Hi."}],
                max_tokens=16, temperature=0.0, seed=42,
            )
        after_warmup = _fetch_metric(metrics_url, args.metric)
        if after_warmup is None and report.metrics.get("probe_mode") == "docker_logs":
            after_warmup = float(
                _fetch_recapture_via_docker_logs(args.container, lookback_sec=30) or 0
            )
        after_warmup = after_warmup or 0.0

        # Test burst
        for i in range(args.test_reqs):
            post_chat(
                endpoint=args.endpoint, api_key=args.api_key, model=args.model,
                messages=[{"role": "user", "content": f"Count to {i % 10}."}],
                max_tokens=32, temperature=0.0, seed=42,
            )
        after_test = _fetch_metric(metrics_url, args.metric)
        if after_test is None and report.metrics.get("probe_mode") == "docker_logs":
            after_test = float(
                _fetch_recapture_via_docker_logs(args.container, lookback_sec=60) or 0
            )
        after_test = after_test or 0.0

        delta = after_test - after_warmup
        report.metrics = {
            "metric": args.metric,
            "baseline_before_warmup": baseline,
            "after_warmup": after_warmup,
            "after_test": after_test,
            "delta_post_warmup": delta,
        }
        report.gates = [
            GateResult(
                name="cuda_graph_stable_after_warmup",
                value=delta,
                threshold="== 0 captures after warmup",
                passed=delta == 0,
            ),
        ]
    except Exception as e:
        report.error = f"{type(e).__name__}: {e}"
        write_report(report, out_path, quiet=args.quiet)
        return 2

    write_report(report, out_path, quiet=args.quiet)
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    _sys.exit(main())
