# SPDX-License-Identifier: Apache-2.0
"""Tests for ``sndr bench compare`` CLI (S2.5 audit closure 2026-05-08).

Covers:
  • Multi-metric extraction from genesis_bench_suite JSON shape
  • Verdict strings (WIN / REGRESS / ~ noise)
  • Exit code 2 when regression exceeds budget
  • --json structured output
  • Direction handling (lower_better vs higher_better)
  • Tolerance to missing optional metrics (accept_rate, tool_call)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sndr.cli.legacy.bench_compare import (
    main,
    render_human,
    render_json,
    _delta_pct,
    _verdict,
    _METRIC_SPEC,
)


# ─── Fixtures: synthetic bench JSONs ────────────────────────────────────


def _bench(name: str, tpot: float, wall_tps: float, ttft: float,
           accept: float | None = 0.815,
           tool_passed: int = 7, tool_total: int = 7,
           cv: float = 0.05) -> dict:
    return {
        "name": name,
        "decode_bench": {
            "decode_TPOT_ms": {"mean": tpot, "cv": cv, "n": 25, "std": 0.2},
            "wall_TPS": {"mean": wall_tps, "cv": cv, "n": 25, "std": 12.0},
            "TTFT_ms": {"mean": ttft, "cv": 0.3, "n": 25, "std": 30.0},
        },
        "multi_turn": {"window_accept_rate": accept} if accept is not None else {},
        "tool_call": {
            "passed_positive": tool_passed,
            "total_positive": tool_total,
            "summary": f"{tool_passed}/{tool_total} positive cases",
        },
    }


@pytest.fixture
def baseline_json(tmp_path: Path) -> Path:
    p = tmp_path / "baseline.json"
    p.write_text(json.dumps(_bench(
        "35b_wave3.1", tpot=3.92, wall_tps=236.24, ttft=110.77,
        accept=0.811, tool_passed=6,
    )))
    return p


@pytest.fixture
def candidate_win_json(tmp_path: Path) -> Path:
    """Candidate that's clearly better — lower TPOT, higher TPS, more tools."""
    p = tmp_path / "candidate_win.json"
    p.write_text(json.dumps(_bench(
        "35b_wave7", tpot=3.89, wall_tps=237.95, ttft=106.56,
        accept=0.806, tool_passed=7,
    )))
    return p


@pytest.fixture
def candidate_regress_json(tmp_path: Path) -> Path:
    """Candidate with clear TPS regression beyond 5% budget."""
    p = tmp_path / "candidate_regress.json"
    p.write_text(json.dumps(_bench(
        "broken", tpot=4.30, wall_tps=166.25, ttft=127.5,
        accept=0.814, tool_passed=7, cv=0.37,
    )))
    return p


# ─── Helper math ────────────────────────────────────────────────────────


class TestDeltaMath:
    def test_delta_pct_simple(self):
        assert _delta_pct(100.0, 110.0) == 10.0

    def test_delta_pct_negative(self):
        assert _delta_pct(100.0, 90.0) == -10.0

    def test_delta_pct_zero_baseline_returns_none(self):
        assert _delta_pct(0.0, 5.0) is None

    def test_delta_pct_none_input(self):
        assert _delta_pct(None, 5.0) is None
        assert _delta_pct(5.0, None) is None


class TestVerdicts:
    def test_lower_better_win_at_5pct(self):
        # decode_TPOT dropped 6% → WIN
        v = _verdict("lower_better", -6.0, budget=5.0)
        assert "WIN" in v

    def test_lower_better_regress_at_5pct(self):
        v = _verdict("lower_better", +7.0, budget=5.0)
        assert "REGRESS" in v

    def test_higher_better_win(self):
        v = _verdict("higher_better", +6.0, budget=5.0)
        assert "WIN" in v

    def test_higher_better_regress(self):
        v = _verdict("higher_better", -8.0, budget=5.0)
        assert "REGRESS" in v

    def test_noise_band_under_1pct(self):
        v = _verdict("lower_better", 0.5, budget=5.0)
        assert "noise" in v.lower()

    def test_within_budget_marked_neutral(self):
        v = _verdict("higher_better", -3.0, budget=5.0)
        assert "REGRESS" not in v
        assert "WIN" not in v


# ─── Render: human table ────────────────────────────────────────────────


class TestRenderHuman:
    def test_renders_all_metrics(self, baseline_json, candidate_win_json):
        a = json.loads(baseline_json.read_text())
        b = json.loads(candidate_win_json.read_text())
        text, has_regression = render_human(a, b, "A", "B", 5.0)
        assert "decode_TPOT_ms" in text
        assert "wall_TPS" in text
        assert "TTFT_ms" in text
        assert "spec_accept_rate" in text
        assert "tool_pass_pct" in text

    def test_win_case_no_regression_flag(
        self, baseline_json, candidate_win_json,
    ):
        a = json.loads(baseline_json.read_text())
        b = json.loads(candidate_win_json.read_text())
        _text, has_regression = render_human(a, b, "A", "B", 5.0)
        assert has_regression is False

    def test_regression_case_flagged(
        self, baseline_json, candidate_regress_json,
    ):
        a = json.loads(baseline_json.read_text())
        b = json.loads(candidate_regress_json.read_text())
        text, has_regression = render_human(a, b, "A", "B", 5.0)
        assert has_regression is True
        assert "REGRESS" in text


# ─── Render: structured JSON ────────────────────────────────────────────


class TestRenderJson:
    def test_json_structure(self, baseline_json, candidate_win_json):
        a = json.loads(baseline_json.read_text())
        b = json.loads(candidate_win_json.read_text())
        out = render_json(a, b, "A", "B", 5.0)
        assert out["a_name"] == "A"
        assert out["b_name"] == "B"
        assert "metrics" in out
        assert "decode_TPOT_ms" in out["metrics"]
        assert "delta_pct" in out["metrics"]["decode_TPOT_ms"]
        assert "verdict" in out["metrics"]["decode_TPOT_ms"]
        assert out["any_regression"] is False

    def test_any_regression_flag_when_regress(
        self, baseline_json, candidate_regress_json,
    ):
        a = json.loads(baseline_json.read_text())
        b = json.loads(candidate_regress_json.read_text())
        out = render_json(a, b, "A", "B", 5.0)
        assert out["any_regression"] is True


# ─── CLI integration ────────────────────────────────────────────────────


class TestCli:
    def test_cli_exit_0_on_win(
        self, baseline_json, candidate_win_json, capsys,
    ):
        rc = main([str(baseline_json), str(candidate_win_json)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "decode_TPOT_ms" in out

    def test_cli_exit_2_on_regression(
        self, baseline_json, candidate_regress_json, capsys,
    ):
        rc = main([str(baseline_json), str(candidate_regress_json)])
        out = capsys.readouterr().out
        assert rc == 2
        assert "Regression detected" in out

    def test_cli_json_mode(
        self, baseline_json, candidate_win_json, capsys,
    ):
        rc = main([str(baseline_json), str(candidate_win_json), "--json"])
        out = capsys.readouterr().out
        assert rc == 0
        # Output is JSON parseable
        data = json.loads(out)
        assert data["any_regression"] is False
        assert "metrics" in data

    def test_cli_missing_baseline_returns_1(self, tmp_path, capsys):
        rc = main([str(tmp_path / "nope.json"), str(tmp_path / "also-nope.json")])
        err = capsys.readouterr().err
        assert rc == 1
        assert "not found" in err

    def test_cli_custom_budget_changes_verdict(
        self, baseline_json, candidate_win_json, capsys,
    ):
        # With strict 0.1% budget, even small wins flag as WIN/regress
        rc = main([
            str(baseline_json), str(candidate_win_json),
            "--regression-budget", "0.1",
        ])
        # Win case → still rc=0 (no regression beyond 0.1%)
        # The accept_rate dropped 0.005 (0.6%), > 0.1% budget → REGRESS
        out = capsys.readouterr().out
        assert "REGRESS" in out
        assert rc == 2


# ─── Tolerance to missing optional fields ──────────────────────────────


class TestMissingFields:
    def test_missing_multi_turn_no_crash(self, tmp_path):
        a = _bench("a", tpot=4.0, wall_tps=200, ttft=100, accept=None)
        b = _bench("b", tpot=3.9, wall_tps=210, ttft=98, accept=None)
        # render must handle None accept_rate gracefully
        text, _ = render_human(a, b, "A", "B", 5.0)
        assert "spec_accept_rate" in text
        assert "n/a" in text

    def test_missing_tool_call_no_crash(self, tmp_path):
        a = _bench("a", tpot=4.0, wall_tps=200, ttft=100,
                   tool_passed=0, tool_total=0)
        b = _bench("b", tpot=3.9, wall_tps=210, ttft=98,
                   tool_passed=0, tool_total=0)
        text, _ = render_human(a, b, "A", "B", 5.0)
        assert "tool_pass_pct" in text
        assert "n/a" in text


# ─── Metric coverage contract ──────────────────────────────────────────


class TestMetricSpec:
    def test_all_metrics_have_direction(self):
        for label, extractor, direction, unit in _METRIC_SPEC:
            assert direction in ("lower_better", "higher_better"), label
            assert callable(extractor), label

    def test_minimum_six_metrics(self):
        # Audit S2.5 spec required: decode_TPOT, wall_TPS, TTFT,
        # accept, tool, CV — at least 6 metrics covered.
        labels = [m[0] for m in _METRIC_SPEC]
        assert len(labels) >= 6
        assert "decode_TPOT_ms" in labels
        assert "wall_TPS" in labels
        assert "TTFT_ms" in labels
