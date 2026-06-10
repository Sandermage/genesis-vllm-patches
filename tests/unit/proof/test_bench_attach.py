# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr.proof.bench_attach` — §6.8 bench-delta ingest.

Contract:

  • `extract_headline_metrics` is tolerant of bench-suite schema drift —
    metrics may appear top-level or in `headline` / `summary` /
    `reference_metrics` / `metrics` sub-blocks, under any of the known
    aliases.
  • `compute_delta` populates `*_delta_pct` only when a baseline is
    supplied; absent values stay None.
  • `attach_bench` creates a new proof artefact when none exists, and
    updates the latest existing artefact when one does — keeping the
    static-check fields intact.
  • CLI exit codes: 0 on success, 2 on input errors.
"""
from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest


# ─── Helpers ───────────────────────────────────────────────────────────


def _bench_run(**overrides):
    base = {
        "composed_key": "qwen-3.6-2510:rtx3090:long-gen",
        "vllm_pin": "vllm@0.6.4-stub",
        "methodology_id": "M-v1",
        "methodology_sha": "deadbeef",
        "measured_at": "2026-05-12T10:00:00+00:00",
        "headline": {
            "median_tps": 42.5,
            "p95_tps": 38.1,
            "decode_TPOT_ms": 23.4,
            "TTFT_ms": 180.0,
            "cv_pct": 4.2,
            "tool_call_score": "A+",
        },
    }
    base.update(overrides)
    return base


def _bench_baseline(**overrides):
    base = {
        "composed_key": "qwen-3.6-2510:rtx3090:long-gen",
        "headline": {
            "median_tps": 40.0,
            "p95_tps": 36.0,
            "decode_TPOT_ms": 25.0,
            "TTFT_ms": 200.0,
            "cv_pct": 5.0,
        },
    }
    base.update(overrides)
    return base


def _write_json(p: Path, payload: dict) -> Path:
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


# ─── extract_headline_metrics ─────────────────────────────────────────


class TestExtractHeadlineMetrics:
    def test_headline_subblock(self):
        from sndr.proof.bench_attach import extract_headline_metrics
        out = extract_headline_metrics(_bench_run())
        assert out["median_tps"] == 42.5
        assert out["p95_tps"] == 38.1
        assert out["decode_tpot_ms"] == 23.4
        assert out["ttft_ms"] == 180.0
        assert out["cv_pct"] == 4.2
        assert out["tool_call_score"] == "A+"

    def test_top_level_metrics(self):
        from sndr.proof.bench_attach import extract_headline_metrics
        out = extract_headline_metrics({"median_tps": 99.9, "TTFT_ms": 12.0})
        assert out["median_tps"] == 99.9
        assert out["ttft_ms"] == 12.0

    def test_summary_subblock(self):
        from sndr.proof.bench_attach import extract_headline_metrics
        out = extract_headline_metrics({"summary": {"wall_TPS": 50}})
        # `wall_TPS` aliased to median_tps.
        assert out["median_tps"] == 50

    def test_alias_long_gen_sustained(self):
        from sndr.proof.bench_attach import extract_headline_metrics
        out = extract_headline_metrics(
            {"reference_metrics": {"long_gen_sustained_tps": 33.0}}
        )
        assert out["median_tps"] == 33.0

    def test_alias_precedence_first_wins(self):
        from sndr.proof.bench_attach import extract_headline_metrics
        # Both `median_tps` and `long_gen_sustained_tps` present at top
        # level — first alias in tuple wins.
        out = extract_headline_metrics(
            {"median_tps": 60.0, "long_gen_sustained_tps": 30.0}
        )
        assert out["median_tps"] == 60.0

    def test_missing_metrics_absent(self):
        from sndr.proof.bench_attach import extract_headline_metrics
        out = extract_headline_metrics({"headline": {"median_tps": 1}})
        # Only median_tps was provided.
        assert "p95_tps" not in out
        assert "ttft_ms" not in out

    def test_carry_through_identifiers(self):
        from sndr.proof.bench_attach import extract_headline_metrics
        out = extract_headline_metrics(_bench_run())
        assert out["composed_key"] == "qwen-3.6-2510:rtx3090:long-gen"
        assert out["vllm_pin"] == "vllm@0.6.4-stub"
        assert out["methodology_id"] == "M-v1"
        assert out["methodology_sha"] == "deadbeef"
        assert out["measured_at"] == "2026-05-12T10:00:00+00:00"

    def test_null_value_falls_through_to_alias(self):
        from sndr.proof.bench_attach import extract_headline_metrics
        # If first alias is null, fall through to next.
        out = extract_headline_metrics(
            {"median_tps": None, "wall_TPS_median": 77.0}
        )
        assert out["median_tps"] == 77.0


# ─── compute_delta ─────────────────────────────────────────────────────


class TestComputeDelta:
    def test_no_baseline_means_no_pct(self):
        from sndr.proof.bench_attach import compute_delta
        d = compute_delta(_bench_run())
        assert d.median_tps == 42.5
        assert d.median_tps_delta_pct is None
        assert d.p95_tps_delta_pct is None
        assert d.decode_tpot_delta_pct is None
        assert d.ttft_delta_pct is None
        assert d.baseline_path is None

    def test_baseline_yields_pct(self):
        from sndr.proof.bench_attach import compute_delta
        d = compute_delta(
            _bench_run(),
            baseline=_bench_baseline(),
            baseline_path="/tmp/baseline.json",
        )
        # (42.5 - 40.0) / 40.0 * 100 = 6.25
        assert d.median_tps_delta_pct == 6.25
        # (38.1 - 36.0) / 36.0 * 100 = 5.833... → 5.83
        assert d.p95_tps_delta_pct == 5.83
        # (23.4 - 25.0) / 25.0 * 100 = -6.4
        assert d.decode_tpot_delta_pct == -6.4
        # (180 - 200) / 200 * 100 = -10
        assert d.ttft_delta_pct == -10.0
        assert d.baseline_path == "/tmp/baseline.json"

    def test_zero_baseline_returns_none(self):
        from sndr.proof.bench_attach import compute_delta
        d = compute_delta(
            {"median_tps": 10.0},
            baseline={"median_tps": 0.0},
        )
        assert d.median_tps == 10.0
        assert d.median_tps_delta_pct is None

    def test_missing_current_metric_yields_none(self):
        from sndr.proof.bench_attach import compute_delta
        d = compute_delta(
            {"p95_tps": 10.0},
            baseline=_bench_baseline(),
        )
        assert d.median_tps is None
        # No current median → no delta either.
        assert d.median_tps_delta_pct is None
        assert d.p95_tps == 10.0

    def test_to_dict_drops_none(self):
        from sndr.proof.bench_attach import compute_delta
        d = compute_delta({"median_tps": 5.0})
        out = d.to_dict()
        assert out["median_tps"] == 5.0
        # `p95_tps_delta_pct` etc. must NOT appear when None.
        for k in ("p95_tps", "decode_tpot_ms", "ttft_ms",
                  "p95_tps_delta_pct", "decode_tpot_delta_pct",
                  "ttft_delta_pct", "median_tps_delta_pct",
                  "baseline_path"):
            assert k not in out, f"None value leaked: {k}"


# ─── attach_bench ──────────────────────────────────────────────────────


class TestAttachBench:
    def test_creates_new_artefact_when_none_exists(self, tmp_path):
        """No prior artefact → bench-attach builds a fresh proof with
        static checks + bench_delta."""
        from sndr.proof import load_proof_artefact
        from sndr.proof.bench_attach import attach_bench
        bench = _write_json(tmp_path / "run.json", _bench_run())
        out = tmp_path / "patch_proof"

        # P58 is a real PATCH_REGISTRY entry — static checks will run.
        target = attach_bench("P58", bench, out_dir=out)
        assert target.is_file()
        data = load_proof_artefact(target)
        assert data["patch_id"] == "P58"
        assert "static_checks" in data
        assert data["bench_delta"]["median_tps"] == 42.5

    def test_updates_existing_artefact(self, tmp_path):
        """Second attach → updates same artefact, keeps static_checks."""
        from sndr.proof import load_proof_artefact
        from sndr.proof.bench_attach import attach_bench

        bench1 = _write_json(tmp_path / "run1.json", _bench_run())
        bench2 = _write_json(
            tmp_path / "run2.json",
            _bench_run(headline={"median_tps": 100.0}),
        )
        out = tmp_path / "patch_proof"

        first = attach_bench("P58", bench1, out_dir=out)
        first_static = load_proof_artefact(first)["static_checks"]

        second = attach_bench("P58", bench2, out_dir=out)
        # Same file (same vllm_pin → same filename).
        assert first == second
        data = load_proof_artefact(second)
        # Updated metric, but static_checks unchanged.
        assert data["bench_delta"]["median_tps"] == 100.0
        assert data["static_checks"] == first_static

    def test_with_baseline(self, tmp_path):
        from sndr.proof import load_proof_artefact
        from sndr.proof.bench_attach import attach_bench

        bench = _write_json(tmp_path / "run.json", _bench_run())
        base = _write_json(tmp_path / "base.json", _bench_baseline())
        out = tmp_path / "patch_proof"

        target = attach_bench(
            "P58", bench, baseline_path=base, out_dir=out,
        )
        data = load_proof_artefact(target)
        assert data["bench_delta"]["median_tps_delta_pct"] == 6.25
        assert data["bench_delta"]["baseline_path"] == str(base)

    def test_missing_bench_raises(self, tmp_path):
        from sndr.proof.bench_attach import (
            BenchAttachError, attach_bench,
        )
        with pytest.raises(BenchAttachError, match="bench file not found"):
            attach_bench("P58", tmp_path / "nope.json",
                         out_dir=tmp_path / "patch_proof")

    def test_missing_baseline_raises(self, tmp_path):
        from sndr.proof.bench_attach import (
            BenchAttachError, attach_bench,
        )
        bench = _write_json(tmp_path / "run.json", _bench_run())
        with pytest.raises(BenchAttachError, match="baseline file not found"):
            attach_bench(
                "P58", bench,
                baseline_path=tmp_path / "nope.json",
                out_dir=tmp_path / "patch_proof",
            )

    def test_malformed_bench_raises(self, tmp_path):
        from sndr.proof.bench_attach import (
            BenchAttachError, attach_bench,
        )
        bad = tmp_path / "bad.json"
        bad.write_text("not valid {json", encoding="utf-8")
        with pytest.raises(BenchAttachError, match="could not parse"):
            attach_bench(
                "P58", bad, out_dir=tmp_path / "patch_proof",
            )

    def test_malformed_baseline_raises(self, tmp_path):
        from sndr.proof.bench_attach import (
            BenchAttachError, attach_bench,
        )
        bench = _write_json(tmp_path / "run.json", _bench_run())
        bad = tmp_path / "bad.json"
        bad.write_text("not valid {json", encoding="utf-8")
        with pytest.raises(BenchAttachError, match="could not parse"):
            attach_bench(
                "P58", bench, baseline_path=bad,
                out_dir=tmp_path / "patch_proof",
            )


# ─── CLI integration ──────────────────────────────────────────────────


def _make_args(**kw):
    defaults = dict(
        patch_id="P58",
        bench_path="",
        baseline=None,
        out_dir=None,
        json=False,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


class TestCLI:
    def test_cli_success_human(self, tmp_path, capsys):
        from sndr.cli.legacy.patches import _run_bench_attach
        bench = _write_json(tmp_path / "run.json", _bench_run())
        out = tmp_path / "patch_proof"
        rc = _run_bench_attach(_make_args(
            bench_path=str(bench), out_dir=str(out),
        ))
        captured = capsys.readouterr().out
        assert rc == 0
        assert "median_tps" in captured
        assert "bench_delta attached" in captured

    def test_cli_success_json(self, tmp_path, capsys):
        from sndr.cli.legacy.patches import _run_bench_attach
        bench = _write_json(tmp_path / "run.json", _bench_run())
        base = _write_json(tmp_path / "base.json", _bench_baseline())
        out = tmp_path / "patch_proof"
        rc = _run_bench_attach(_make_args(
            bench_path=str(bench),
            baseline=str(base),
            out_dir=str(out),
            json=True,
        ))
        captured = capsys.readouterr().out
        assert rc == 0
        payload = json.loads(captured)
        assert payload["patch_id"] == "P58"
        assert payload["bench_delta"]["median_tps"] == 42.5
        assert payload["bench_delta"]["median_tps_delta_pct"] == 6.25

    def test_cli_missing_bench_returns_2(self, tmp_path, capsys):
        from sndr.cli.legacy.patches import _run_bench_attach
        rc = _run_bench_attach(_make_args(
            bench_path=str(tmp_path / "nope.json"),
            out_dir=str(tmp_path / "patch_proof"),
        ))
        assert rc == 2

    def test_cli_malformed_bench_returns_2(self, tmp_path, capsys):
        from sndr.cli.legacy.patches import _run_bench_attach
        bad = tmp_path / "bad.json"
        bad.write_text("not valid {json", encoding="utf-8")
        rc = _run_bench_attach(_make_args(
            bench_path=str(bad),
            out_dir=str(tmp_path / "patch_proof"),
        ))
        assert rc == 2
