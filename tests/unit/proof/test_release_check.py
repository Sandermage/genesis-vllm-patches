# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr.proof.release_check` — §6.8 release-gate
consumer (Entry 21).

Contract:

  • ReleasePolicy validates mode + threshold at construction time.
  • Each policy mode allows exactly the buckets documented in
    `_MODE_ALLOWED`.
  • Regression check is direction-aware: TPS-drop and latency-rise
    both trigger; the symmetric (TPS-rise / latency-drop) never does.
  • Report-mode never blocks; tighter modes block when any
    considered patch is out-of-bucket.
  • CLI exit codes: 0 = passed/report; 1 = blocked; 2 = bad input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest


# ─── Helpers ───────────────────────────────────────────────────────────


def _write(out_dir: Path, name: str, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _reg(*ids: str) -> dict:
    return {
        pid: {"family": "x", "tier": "community", "lifecycle": "stable"}
        for pid in ids
    }


def _make_args(**kw):
    defaults = dict(
        mode="report",
        max_regression_pct=None,
        patch=None,
        tier=None,
        out_dir=None,
        json=False,
        show_passing=False,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


# ─── ReleasePolicy validation ──────────────────────────────────────────


class TestReleasePolicy:
    def test_default_is_report(self):
        from sndr.proof.release_check import ReleasePolicy
        p = ReleasePolicy()
        assert p.mode == "report"
        assert p.max_regression_pct is None

    def test_unknown_mode_raises(self):
        from sndr.proof.release_check import (
            ReleaseCheckError, ReleasePolicy,
        )
        with pytest.raises(ReleaseCheckError, match="unknown policy mode"):
            ReleasePolicy(mode="nonsense")

    def test_negative_threshold_raises(self):
        from sndr.proof.release_check import (
            ReleaseCheckError, ReleasePolicy,
        )
        with pytest.raises(ReleaseCheckError, match="must be"):
            ReleasePolicy(max_regression_pct=-1.0)

    def test_allowed_buckets_per_mode(self):
        from sndr.proof.release_check import ReleasePolicy
        assert "dead" in ReleasePolicy(mode="report").allowed_buckets
        assert "dead" not in ReleasePolicy(
            mode="require-static",
        ).allowed_buckets
        assert "static_only" in ReleasePolicy(
            mode="require-static",
        ).allowed_buckets
        assert "static_only" not in ReleasePolicy(
            mode="require-bench",
        ).allowed_buckets
        assert "bench_attached" in ReleasePolicy(
            mode="require-bench",
        ).allowed_buckets
        assert "bench_attached" not in ReleasePolicy(
            mode="require-baseline",
        ).allowed_buckets
        assert ReleasePolicy(
            mode="require-baseline",
        ).allowed_buckets == frozenset({"bench_with_baseline"})


# ─── evaluate_release — policy modes ──────────────────────────────────


class TestEvaluateMode:
    def test_report_mode_never_blocks(self, tmp_path):
        from sndr.proof.release_check import (
            ReleasePolicy, evaluate_release,
        )
        r = evaluate_release(
            ReleasePolicy(mode="report"),
            registry=_reg("P1"), out_dir=tmp_path,
        )
        # P1 is dead (no artefact) but report mode allows every bucket.
        assert r["passed_count"] == 1
        assert r["failed_count"] == 0
        assert r["release_blocked"] is False
        v = r["verdicts"][0]
        assert v["bucket"] == "dead"
        # In report mode every bucket is "allowed" → reasons empty.
        assert v["reasons"] == []

    def test_require_static_blocks_on_dead(self, tmp_path):
        from sndr.proof.release_check import (
            ReleasePolicy, evaluate_release,
        )
        r = evaluate_release(
            ReleasePolicy(mode="require-static"),
            registry=_reg("P1"), out_dir=tmp_path,
        )
        assert r["release_blocked"] is True
        assert r["failed_count"] == 1

    def test_require_static_passes_static_only(self, tmp_path):
        from sndr.proof.release_check import (
            ReleasePolicy, evaluate_release,
        )
        _write(tmp_path, "P1__v1.json", {"static_passed": True})
        r = evaluate_release(
            ReleasePolicy(mode="require-static"),
            registry=_reg("P1"), out_dir=tmp_path,
        )
        assert r["release_blocked"] is False
        assert r["passed_count"] == 1

    def test_require_bench_blocks_on_static_only(self, tmp_path):
        from sndr.proof.release_check import (
            ReleasePolicy, evaluate_release,
        )
        _write(tmp_path, "P1__v1.json", {"static_passed": True})
        r = evaluate_release(
            ReleasePolicy(mode="require-bench"),
            registry=_reg("P1"), out_dir=tmp_path,
        )
        assert r["release_blocked"] is True

    def test_require_bench_passes_bench_attached(self, tmp_path):
        from sndr.proof.release_check import (
            ReleasePolicy, evaluate_release,
        )
        _write(tmp_path, "P1__v1.json", {
            "static_passed": True,
            "bench_delta": {"median_tps": 1.0},
        })
        r = evaluate_release(
            ReleasePolicy(mode="require-bench"),
            registry=_reg("P1"), out_dir=tmp_path,
        )
        assert r["release_blocked"] is False

    def test_require_baseline_blocks_on_bench_attached(self, tmp_path):
        from sndr.proof.release_check import (
            ReleasePolicy, evaluate_release,
        )
        _write(tmp_path, "P1__v1.json", {
            "static_passed": True,
            "bench_delta": {"median_tps": 1.0},
        })
        r = evaluate_release(
            ReleasePolicy(mode="require-baseline"),
            registry=_reg("P1"), out_dir=tmp_path,
        )
        assert r["release_blocked"] is True

    def test_require_baseline_passes_baseline(self, tmp_path):
        from sndr.proof.release_check import (
            ReleasePolicy, evaluate_release,
        )
        _write(tmp_path, "P1__v1.json", {
            "static_passed": True,
            "bench_delta": {"median_tps": 1.0, "median_tps_delta_pct": 1.0},
        })
        r = evaluate_release(
            ReleasePolicy(mode="require-baseline"),
            registry=_reg("P1"), out_dir=tmp_path,
        )
        assert r["release_blocked"] is False


# ─── Regression detector ──────────────────────────────────────────────


class TestRegressionDetection:
    def _setup_baseline(self, tmp_path, **deltas):
        _write(tmp_path, "P1__v1.json", {
            "static_passed": True,
            "bench_delta": {
                "median_tps": 1.0,
                "median_tps_delta_pct": deltas.get("median", 0.0),
                "p95_tps_delta_pct": deltas.get("p95", 0.0),
                "decode_tpot_delta_pct": deltas.get("decode", 0.0),
                "ttft_delta_pct": deltas.get("ttft", 0.0),
            },
        })

    def test_tps_drop_triggers(self, tmp_path):
        from sndr.proof.release_check import (
            ReleasePolicy, evaluate_release,
        )
        self._setup_baseline(tmp_path, median=-10.0)
        r = evaluate_release(
            ReleasePolicy(mode="require-baseline", max_regression_pct=5.0),
            registry=_reg("P1"), out_dir=tmp_path,
        )
        assert r["release_blocked"] is True
        assert r["verdicts"][0]["regressions"][0]["metric"] == \
            "median_tps_delta_pct"

    def test_tps_rise_does_not_trigger(self, tmp_path):
        from sndr.proof.release_check import (
            ReleasePolicy, evaluate_release,
        )
        # +10% TPS is a WIN, not a regression — symmetric direction.
        self._setup_baseline(tmp_path, median=+10.0)
        r = evaluate_release(
            ReleasePolicy(mode="require-baseline", max_regression_pct=5.0),
            registry=_reg("P1"), out_dir=tmp_path,
        )
        assert r["release_blocked"] is False

    def test_latency_rise_triggers(self, tmp_path):
        from sndr.proof.release_check import (
            ReleasePolicy, evaluate_release,
        )
        self._setup_baseline(tmp_path, decode=+10.0)
        r = evaluate_release(
            ReleasePolicy(mode="require-baseline", max_regression_pct=5.0),
            registry=_reg("P1"), out_dir=tmp_path,
        )
        assert r["release_blocked"] is True

    def test_latency_drop_does_not_trigger(self, tmp_path):
        from sndr.proof.release_check import (
            ReleasePolicy, evaluate_release,
        )
        # -10% TPOT is a WIN; symmetric direction must not trigger.
        self._setup_baseline(tmp_path, decode=-10.0)
        r = evaluate_release(
            ReleasePolicy(mode="require-baseline", max_regression_pct=5.0),
            registry=_reg("P1"), out_dir=tmp_path,
        )
        assert r["release_blocked"] is False

    def test_threshold_boundary(self, tmp_path):
        """Exactly at threshold does NOT trigger (strict inequality)."""
        from sndr.proof.release_check import (
            ReleasePolicy, evaluate_release,
        )
        self._setup_baseline(tmp_path, median=-5.0)
        r = evaluate_release(
            ReleasePolicy(mode="require-baseline", max_regression_pct=5.0),
            registry=_reg("P1"), out_dir=tmp_path,
        )
        # -5.0% is exactly at the threshold — should not block.
        assert r["release_blocked"] is False

    def test_no_threshold_means_no_regression_check(self, tmp_path):
        from sndr.proof.release_check import (
            ReleasePolicy, evaluate_release,
        )
        self._setup_baseline(tmp_path, median=-50.0)
        # Without max_regression_pct, even -50% TPS is "passed".
        r = evaluate_release(
            ReleasePolicy(mode="require-baseline"),
            registry=_reg("P1"), out_dir=tmp_path,
        )
        assert r["release_blocked"] is False

    def test_regression_check_only_for_baseline_bucket(self, tmp_path):
        """`bench_attached` (no delta_pct) doesn't get regression-checked
        — there's nothing to compare."""
        from sndr.proof.release_check import (
            ReleasePolicy, evaluate_release,
        )
        _write(tmp_path, "P1__v1.json", {
            "static_passed": True,
            "bench_delta": {"median_tps": 1.0},   # no delta_pct
        })
        # require-bench (not -baseline) so P1 passes the bucket gate.
        r = evaluate_release(
            ReleasePolicy(mode="require-bench", max_regression_pct=1.0),
            registry=_reg("P1"), out_dir=tmp_path,
        )
        assert r["release_blocked"] is False


# ─── Filters ──────────────────────────────────────────────────────────


class TestFilters:
    def test_patch_filter(self, tmp_path):
        from sndr.proof.release_check import (
            ReleasePolicy, evaluate_release,
        )
        r = evaluate_release(
            ReleasePolicy(mode="require-static", patch_filter=frozenset({"P2"})),
            registry=_reg("P1", "P2"), out_dir=tmp_path,
        )
        assert r["considered"] == 1
        assert r["verdicts"][0]["patch_id"] == "P2"

    def test_tier_filter(self, tmp_path):
        from sndr.proof.release_check import (
            ReleasePolicy, evaluate_release,
        )
        reg = {
            "P1": {"family": "x", "tier": "release", "lifecycle": "stable"},
            "P2": {"family": "x", "tier": "community", "lifecycle": "stable"},
        }
        r = evaluate_release(
            ReleasePolicy(
                mode="require-static",
                tier_filter=frozenset({"release"}),
            ),
            registry=reg, out_dir=tmp_path,
        )
        assert r["considered"] == 1
        assert r["verdicts"][0]["patch_id"] == "P1"


# ─── CLI integration ──────────────────────────────────────────────────


class TestCLI:
    def test_cli_report_default_passes(self, tmp_path, capsys):
        from sndr.cli.legacy.patches import _run_release_check
        rc = _run_release_check(_make_args(out_dir=str(tmp_path)))
        out = capsys.readouterr().out
        assert rc == 0
        assert "report" in out

    def test_cli_require_baseline_blocks_when_no_artefacts(
        self, tmp_path, capsys,
    ):
        from sndr.cli.legacy.patches import _run_release_check
        rc = _run_release_check(_make_args(
            mode="require-baseline", out_dir=str(tmp_path),
        ))
        assert rc == 1
        assert "BLOCKED" in capsys.readouterr().out

    def test_cli_json_blocked_returns_1(self, tmp_path, capsys):
        from sndr.cli.legacy.patches import _run_release_check
        rc = _run_release_check(_make_args(
            mode="require-static",
            out_dir=str(tmp_path),
            json=True,
        ))
        out = capsys.readouterr().out
        assert rc == 1
        payload = json.loads(out)
        assert payload["release_blocked"] is True
        assert payload["policy"]["mode"] == "require-static"

    def test_cli_bad_threshold_returns_2(self, tmp_path, capsys):
        from sndr.cli.legacy.patches import _run_release_check
        rc = _run_release_check(_make_args(
            mode="report",
            max_regression_pct=-1.0,
            out_dir=str(tmp_path),
        ))
        assert rc == 2

    def test_cli_show_passing_flag(self, tmp_path, capsys):
        from sndr.cli.legacy.patches import _run_release_check
        rc = _run_release_check(_make_args(
            mode="report",
            out_dir=str(tmp_path),
            show_passing=True,
        ))
        out = capsys.readouterr().out
        assert rc == 0
        # Report mode → everything is "passing"; with --show-passing the
        # passing block must render.
        assert "passing" in out
