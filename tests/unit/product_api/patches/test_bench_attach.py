# SPDX-License-Identifier: Apache-2.0
"""Tests for ``product_api.patches.bench_attach`` — M.6.2.

Uses ``tmp_path`` for the artefact directory and synthetic bench JSON
to keep the test laptop-only — no real bench runs and no on-tree
``evidence/patch_proof/`` mutations.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sndr.product_api.legacy.patches import bench_attach
from sndr.product_api.legacy.patches.bench_attach import BenchAttachResult


def _write_bench(p: Path) -> Path:
    payload = {
        "composed_key": "qwen-3.6-test:rtx3090:long-gen",
        "vllm_pin": "vllm@stub",
        "methodology_id": "M-v1",
        "methodology_sha": "deadbeef",
        "measured_at": "2026-05-27T00:00:00+00:00",
        "headline": {
            "median_tps": 42.5,
            "p95_tps": 38.1,
            "decode_TPOT_ms": 23.4,
            "TTFT_ms": 180.0,
            "cv_pct": 4.2,
            "tool_call_score": "A+",
        },
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _write_baseline(p: Path) -> Path:
    payload = {
        "composed_key": "qwen-3.6-test:rtx3090:long-gen",
        "headline": {
            "median_tps": 40.0,
            "p95_tps": 36.0,
            "decode_TPOT_ms": 25.0,
            "TTFT_ms": 200.0,
        },
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


class TestAttachBench:
    def test_attach_returns_typed_result(self, tmp_path):
        bench = _write_bench(tmp_path / "bench.json")
        result = bench_attach.attach_bench(
            "P67", bench, out_dir=tmp_path,
        )
        assert isinstance(result, BenchAttachResult)
        assert result.patch_id == "P67"
        assert result.artefact_path.exists()
        assert tmp_path in result.artefact_path.parents
        assert isinstance(result.bench_delta, dict)
        assert result.bench_delta.get("median_tps") == 42.5

    def test_attach_with_baseline_populates_delta(self, tmp_path):
        bench = _write_bench(tmp_path / "bench.json")
        baseline = _write_baseline(tmp_path / "baseline.json")
        result = bench_attach.attach_bench(
            "P67", bench, baseline_path=baseline, out_dir=tmp_path,
        )
        # Δ% keys populated against the baseline.
        assert "median_tps_delta_pct" in result.bench_delta
        assert "p95_tps_delta_pct" in result.bench_delta

    def test_bench_path_missing_raises_bench_attach_error(self, tmp_path):
        from sndr.proof.bench_attach import BenchAttachError

        with pytest.raises(BenchAttachError):
            bench_attach.attach_bench(
                "P67", tmp_path / "does_not_exist.json", out_dir=tmp_path,
            )
