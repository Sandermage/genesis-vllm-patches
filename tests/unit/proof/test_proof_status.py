# SPDX-License-Identifier: Apache-2.0
"""Tests for `summarize_proof_status` + `sndr patches proof-status` —
§6.8 read-side reporting (Entry 20).

Contract: classify each PATCH_REGISTRY entry's freshest artefact into one
of five buckets (bench_with_baseline / bench_attached / static_only /
static_failed / dead) and aggregate the counts. The CLI surface is a
thin renderer on top of `summarize_proof_status`.
"""
from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest


# ─── classify_proof — single-artefact bucket logic ────────────────────


class TestClassifyProof:
    def test_static_failed_when_not_passed(self):
        from sndr.proof import classify_proof
        assert classify_proof({"static_passed": False}) == "static_failed"

    def test_static_only_when_no_bench_delta(self):
        from sndr.proof import classify_proof
        assert classify_proof(
            {"static_passed": True, "bench_delta": None}
        ) == "static_only"
        assert classify_proof(
            {"static_passed": True, "bench_delta": {}}
        ) == "static_only"

    def test_static_only_when_bench_has_only_identifiers(self):
        """Identifier fields like `composed_key` alone don't count as
        bench evidence — must have at least one real metric."""
        from sndr.proof import classify_proof
        assert classify_proof({
            "static_passed": True,
            "bench_delta": {"composed_key": "foo", "vllm_pin": "bar"},
        }) == "static_only"

    def test_bench_attached_when_metric_present(self):
        from sndr.proof import classify_proof
        assert classify_proof({
            "static_passed": True,
            "bench_delta": {"median_tps": 42.5},
        }) == "bench_attached"

    def test_bench_attached_with_tool_call_score(self):
        from sndr.proof import classify_proof
        # tool_call_score is a real metric even without TPS.
        assert classify_proof({
            "static_passed": True,
            "bench_delta": {"tool_call_score": "A+"},
        }) == "bench_attached"

    def test_bench_with_baseline_when_delta_pct_present(self):
        from sndr.proof import classify_proof
        assert classify_proof({
            "static_passed": True,
            "bench_delta": {
                "median_tps": 42.5,
                "median_tps_delta_pct": 6.25,
            },
        }) == "bench_with_baseline"

    def test_bench_with_baseline_takes_precedence_over_attached(self):
        """Even one *_delta_pct field upgrades the bucket."""
        from sndr.proof import classify_proof
        assert classify_proof({
            "static_passed": True,
            "bench_delta": {"ttft_delta_pct": -10.0},
        }) == "bench_with_baseline"


# ─── summarize_proof_status — aggregate logic ──────────────────────────


def _write_artefact(out_dir: Path, name: str, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _fake_registry(patch_ids: list[str]) -> dict:
    return {
        pid: {"lifecycle": "stable", "tier": "community", "family": "x"}
        for pid in patch_ids
    }


class TestSummarize:
    def test_empty_registry_yields_zero_counts(self, tmp_path):
        from sndr.proof import summarize_proof_status
        s = summarize_proof_status(registry={}, out_dir=tmp_path)
        assert s["total"] == 0
        assert all(v == 0 for v in s["counts"].values())
        assert s["patches"] == []

    def test_dead_when_no_artefact(self, tmp_path):
        from sndr.proof import summarize_proof_status
        s = summarize_proof_status(
            registry=_fake_registry(["P1", "P2"]),
            out_dir=tmp_path,
        )
        assert s["total"] == 2
        assert s["counts"]["dead"] == 2
        assert s["counts"]["static_only"] == 0
        assert {p["bucket"] for p in s["patches"]} == {"dead"}

    def test_mixed_buckets(self, tmp_path):
        from sndr.proof import summarize_proof_status
        _write_artefact(tmp_path, "P1__v1.json", {
            "static_passed": True, "bench_delta": None,
        })
        _write_artefact(tmp_path, "P2__v1.json", {
            "static_passed": True,
            "bench_delta": {"median_tps": 1.0},
        })
        _write_artefact(tmp_path, "P3__v1.json", {
            "static_passed": True,
            "bench_delta": {"median_tps": 1.0,
                            "median_tps_delta_pct": 2.0},
        })
        _write_artefact(tmp_path, "P4__v1.json", {
            "static_passed": False,
        })
        # P5 has no artefact → dead.
        s = summarize_proof_status(
            registry=_fake_registry(["P1", "P2", "P3", "P4", "P5"]),
            out_dir=tmp_path,
        )
        assert s["counts"] == {
            "bench_with_baseline": 1,
            "bench_attached": 1,
            "static_only": 1,
            "static_failed": 1,
            "dead": 1,
        }
        by_id = {p["patch_id"]: p["bucket"] for p in s["patches"]}
        assert by_id == {
            "P1": "static_only",
            "P2": "bench_attached",
            "P3": "bench_with_baseline",
            "P4": "static_failed",
            "P5": "dead",
        }

    def test_picks_best_bucket_across_pins(self, tmp_path):
        """Two artefacts (different vllm pins) — the better bucket wins."""
        from sndr.proof import summarize_proof_status
        _write_artefact(tmp_path, "P1__pin1.json", {
            "static_passed": True, "bench_delta": None,
        })
        _write_artefact(tmp_path, "P1__pin2.json", {
            "static_passed": True,
            "bench_delta": {"median_tps": 1.0,
                            "median_tps_delta_pct": 2.0},
        })
        s = summarize_proof_status(
            registry=_fake_registry(["P1"]),
            out_dir=tmp_path,
        )
        assert s["counts"]["bench_with_baseline"] == 1
        assert s["counts"]["static_only"] == 0

    def test_corrupt_artefact_falls_back(self, tmp_path):
        """A malformed JSON file shouldn't crash the summary; the patch
        stays in static_failed when no readable artefact passes."""
        from sndr.proof import summarize_proof_status
        bad = tmp_path / "P1__v1.json"
        bad.write_text("not valid {", encoding="utf-8")
        s = summarize_proof_status(
            registry=_fake_registry(["P1"]),
            out_dir=tmp_path,
        )
        # File exists, so it's not "dead". The classify_proof never
        # runs (load failed), so the default bucket is "static_failed".
        assert s["counts"]["static_failed"] == 1
        assert s["counts"]["dead"] == 0

    def test_uses_real_registry_when_none(self, tmp_path):
        """`registry=None` falls back to live PATCH_REGISTRY."""
        from sndr.proof import summarize_proof_status
        s = summarize_proof_status(out_dir=tmp_path)
        # ≥130 patches (matches PATCH_REGISTRY current baseline).
        assert s["total"] >= 130
        # No artefacts in tmp_path → everything is dead.
        assert s["counts"]["dead"] == s["total"]


# ─── CLI integration ──────────────────────────────────────────────────


def _make_args(**kw):
    defaults = dict(out_dir=None, bucket=None, json=False)
    defaults.update(kw)
    return argparse.Namespace(**defaults)


class TestCLI:
    def test_human_summary(self, tmp_path, capsys):
        from sndr.cli.legacy.patches import _run_proof_status
        rc = _run_proof_status(_make_args(out_dir=str(tmp_path)))
        out = capsys.readouterr().out
        assert rc == 0
        assert "proof-status" in out
        assert "bench_with_baseline" in out
        assert "dead" in out

    def test_json_payload_shape(self, tmp_path, capsys):
        from sndr.cli.legacy.patches import _run_proof_status
        rc = _run_proof_status(_make_args(
            out_dir=str(tmp_path), json=True,
        ))
        out = capsys.readouterr().out
        assert rc == 0
        payload = json.loads(out)
        assert "total" in payload
        assert "counts" in payload
        assert "patches" in payload
        # No filter → filter_buckets is None.
        assert payload["filter_buckets"] is None

    def test_bucket_filter_known(self, tmp_path, capsys):
        from sndr.cli.legacy.patches import _run_proof_status
        rc = _run_proof_status(_make_args(
            out_dir=str(tmp_path), bucket=["dead"], json=True,
        ))
        out = capsys.readouterr().out
        assert rc == 0
        payload = json.loads(out)
        assert payload["filter_buckets"] == ["dead"]
        for p in payload["patches"]:
            assert p["bucket"] == "dead"

    def test_bucket_filter_unknown_returns_2(self, tmp_path, capsys):
        from sndr.cli.legacy.patches import _run_proof_status
        rc = _run_proof_status(_make_args(
            out_dir=str(tmp_path), bucket=["nonsense"],
        ))
        assert rc == 2

    def test_bucket_filter_mixed_known_unknown_returns_2(self, tmp_path):
        from sndr.cli.legacy.patches import _run_proof_status
        rc = _run_proof_status(_make_args(
            out_dir=str(tmp_path), bucket=["dead", "nonsense"],
        ))
        assert rc == 2
