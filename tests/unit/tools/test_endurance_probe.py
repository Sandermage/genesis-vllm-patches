# SPDX-License-Identifier: Apache-2.0
"""TDD for ``tools/endurance_probe.py`` — the multi-hour VRAM / RSS /
KV-creep sampler ported from the ``vram_probe.py`` sidecar of upstream
PR vllm#45022 (roadmap chunk-3 Theme D: do NOT vendor the Voxtral
feature — port the probe into the pin-validation playbook).

Contract (written BEFORE the implementation):

- CLI: ``--interval --duration --port`` (plus ``--host --output``),
  argparse with sane multi-hour defaults.
- Writes one JSON object per sample (JSONL) + a JSON summary with
  per-metric first/last/min/max and a least-squares slope per hour;
  flags creep when a slope exceeds its threshold; exit code 1 on
  detected creep (0 otherwise) so the pin-bump playbook can gate on it.
- Pure stdlib + requests against ``/metrics`` — NO GPU libraries
  (no torch, no pynvml); VRAM comes from the ``nvidia-smi`` CLI with a
  graceful fallback when absent.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "endurance_probe.py"


def _import_tool():
    spec = importlib.util.spec_from_file_location("endurance_probe", TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["endurance_probe"] = mod
    assert spec.loader is not None  # nosec
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────
# Source-level contract
# ─────────────────────────────────────────────────────────────────────


class TestSourceContract:
    def test_no_gpu_libraries(self):
        src = TOOL_PATH.read_text(encoding="utf-8")
        for line in src.splitlines():
            stripped = line.strip()
            assert not stripped.startswith(("import torch", "from torch")), line
            assert not stripped.startswith(("import pynvml", "from pynvml")), line

    def test_provenance_documented(self):
        src = TOOL_PATH.read_text(encoding="utf-8")
        assert "45022" in src
        assert "vram_probe" in src


# ─────────────────────────────────────────────────────────────────────
# /metrics parsing
# ─────────────────────────────────────────────────────────────────────


METRICS_BODY = """\
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="Qwen3.6-35B-A3B"} 0.4321
# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="Qwen3.6-35B-A3B"} 3.0
"""


class TestParseMetrics:
    def test_extracts_kv_usage_and_running(self):
        M = _import_tool()
        kv, running = M.parse_metrics(METRICS_BODY)
        assert kv == 0.4321
        assert running == 3.0

    def test_missing_metrics_return_minus_one(self):
        M = _import_tool()
        kv, running = M.parse_metrics("# nothing here\n")
        assert kv == -1.0
        assert running == -1.0

    def test_comment_lines_never_parsed(self):
        M = _import_tool()
        body = "# kv_cache_usage_perc 0.99\n"
        kv, running = M.parse_metrics(body)
        assert kv == -1.0


# ─────────────────────────────────────────────────────────────────────
# Slope / summary math
# ─────────────────────────────────────────────────────────────────────


class TestSlope:
    def test_linear_series_exact_slope(self):
        M = _import_tool()
        # 60 MiB growth over 1800 s == 120 MiB/h.
        points = [(float(t), 1000.0 + t * (60.0 / 1800.0)) for t in range(0, 1801, 60)]
        slope = M.slope_per_hour(points)
        assert abs(slope - 120.0) < 1e-6

    def test_flat_series_zero_slope(self):
        M = _import_tool()
        points = [(float(t), 5000.0) for t in range(0, 3600, 60)]
        assert abs(M.slope_per_hour(points)) < 1e-9

    def test_degenerate_series(self):
        M = _import_tool()
        assert M.slope_per_hour([]) == 0.0
        assert M.slope_per_hour([(0.0, 1.0)]) == 0.0


def _samples(*, vram_growth_mib_per_hour=0.0, n=30, span_s=3600.0):
    out = []
    for i in range(n):
        t = span_s * i / (n - 1)
        out.append(
            {
                "elapsed_s": t,
                "vram_mib_total": 20000.0 + vram_growth_mib_per_hour * t / 3600.0,
                "enginecore_rss_mib": 4000.0,
                "kv_usage": 0.30,
                "running": 2.0,
                "host_avail_mib": 10000.0,
            }
        )
    return out


class TestSummary:
    def test_flat_run_no_creep(self):
        M = _import_tool()
        summary = M.build_summary(_samples())
        assert summary["creep_detected"] is False
        assert summary["verdict"] == "PASS"
        assert summary["num_samples"] == 30
        assert summary["metrics"]["vram_mib_total"]["first"] == 20000.0
        assert summary["metrics"]["vram_mib_total"]["last"] == 20000.0

    def test_vram_creep_flagged(self):
        M = _import_tool()
        summary = M.build_summary(_samples(vram_growth_mib_per_hour=512.0))
        assert summary["creep_detected"] is True
        assert summary["verdict"] == "CREEP"
        assert "vram_mib_total" in summary["flagged_metrics"]
        assert (
            abs(summary["metrics"]["vram_mib_total"]["slope_per_hour"] - 512.0) < 1.0
        )

    def test_insufficient_samples_inconclusive(self):
        M = _import_tool()
        summary = M.build_summary(_samples(n=3))
        assert summary["creep_detected"] is False
        assert summary["verdict"] == "INSUFFICIENT_SAMPLES"

    def test_negative_sentinel_samples_ignored(self):
        """-1 sentinels (probe failures) must not poison the slopes."""
        M = _import_tool()
        samples = _samples()
        samples[5]["vram_mib_total"] = -1.0
        samples[6]["kv_usage"] = -1.0
        summary = M.build_summary(samples)
        assert summary["creep_detected"] is False


# ─────────────────────────────────────────────────────────────────────
# CLI contract
# ─────────────────────────────────────────────────────────────────────


class TestCli:
    def test_defaults(self):
        M = _import_tool()
        args = M.make_arg_parser().parse_args([])
        assert args.interval == 15.0
        assert args.duration == 14400.0
        assert args.port == 8000
        assert args.host == "127.0.0.1"

    def test_overrides(self):
        M = _import_tool()
        args = M.make_arg_parser().parse_args(
            [
                "--interval", "30",
                "--duration", "7200",
                "--port", "8101",
                "--host", "192.0.2.10",
                "--output", "/tmp/x.jsonl",
            ]
        )
        assert args.interval == 30.0
        assert args.duration == 7200.0
        assert args.port == 8101
        assert args.output == "/tmp/x.jsonl"


# ─────────────────────────────────────────────────────────────────────
# End-to-end main() with fake samplers
# ─────────────────────────────────────────────────────────────────────


class TestMainLoop:
    def _run(self, tmp_path, monkeypatch, *, vram_values):
        M = _import_tool()
        it = iter(vram_values)
        last = [vram_values[-1]]

        def fake_vram():
            try:
                v = next(it)
            except StopIteration:
                v = last[0]
            return [v]

        monkeypatch.setattr(M, "nvidia_vram_mib", fake_vram)
        monkeypatch.setattr(M, "enginecore_rss_mib", lambda: 4000.0)
        monkeypatch.setattr(M, "host_avail_mib", lambda: 9999)
        monkeypatch.setattr(
            M, "fetch_metrics", lambda host, port, timeout=5.0: (0.25, 1.0)
        )
        out = tmp_path / "probe.jsonl"
        rc = M.main(
            [
                "--interval", "0.01",
                "--duration", "0.15",
                "--port", "8101",
                "--output", str(out),
            ]
        )
        return M, out, rc

    def test_writes_jsonl_and_summary(self, tmp_path, monkeypatch):
        M, out, rc = self._run(
            tmp_path, monkeypatch, vram_values=[20000.0] * 100
        )
        assert rc == 0
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 2
        sample = json.loads(lines[0])
        for key in (
            "elapsed_s",
            "vram_mib_total",
            "vram_mib_per_gpu",
            "enginecore_rss_mib",
            "kv_usage",
            "running",
            "host_avail_mib",
        ):
            assert key in sample, key
        summary_path = Path(str(out) + ".summary.json")
        assert summary_path.is_file()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert "verdict" in summary
        assert "metrics" in summary
