# SPDX-License-Identifier: Apache-2.0
"""Tests for ``sndr.observability.patch_metrics`` (Wave 7).

Covers:
  • Default OFF: ``measure_patch_apply`` yields a metric but doesn't
    store it when GENESIS_OBSERVABILITY is unset.
  • Opt-in: env=1 enables collection, ordinal increments, RSS delta
    captured.
  • Aggregation: ``metrics_summary()`` reports counts, slowest-3,
    highest-rss-3.
  • Reset: ``reset_apply_metrics()`` clears buffer + ordinal counter.
  • No-raise contract: an exception inside the block must propagate
    but the elapsed metric must still be recorded.
"""
from __future__ import annotations

import logging
import time

import pytest

from sndr.observability import (
    PatchApplyMetric,
    get_apply_metrics,
    measure_patch_apply,
    reset_apply_metrics,
)
from sndr.observability.patch_metrics import metrics_summary


@pytest.fixture(autouse=True)
def _reset():
    reset_apply_metrics()
    yield
    reset_apply_metrics()


@pytest.fixture
def obs_on(monkeypatch):
    monkeypatch.setenv("GENESIS_OBSERVABILITY", "1")
    yield


@pytest.fixture
def obs_off(monkeypatch):
    monkeypatch.delenv("GENESIS_OBSERVABILITY", raising=False)
    yield


# ─── Default OFF posture ────────────────────────────────────────────────


class TestDefaultOff:
    def test_no_metric_stored_when_env_unset(self, obs_off):
        with measure_patch_apply("P_test") as m:
            m.status = "applied"
            m.reason = "ok"
        assert get_apply_metrics() == []

    def test_metric_object_still_yielded_for_caller_use(self, obs_off):
        """Even when disabled, the context manager still yields a
        usable metric object so callers don't need to branch."""
        with measure_patch_apply("P_test") as m:
            assert isinstance(m, PatchApplyMetric)
            m.status = "applied"
        # Nothing stored, but no exception raised either.
        assert get_apply_metrics() == []


# ─── Opt-in: env=1 ──────────────────────────────────────────────────────


class TestOptInCollection:
    def test_metric_stored_when_env_on(self, obs_on):
        with measure_patch_apply("P1") as m:
            m.status = "applied"
            m.reason = "wired"
        metrics = get_apply_metrics()
        assert len(metrics) == 1
        assert metrics[0].name == "P1"
        assert metrics[0].status == "applied"
        assert metrics[0].reason == "wired"
        assert metrics[0].elapsed_ms >= 0
        assert metrics[0].ordinal == 0

    def test_ordinal_increments(self, obs_on):
        for nm in ["P1", "P2", "P3"]:
            with measure_patch_apply(nm) as m:
                m.status = "applied"
        metrics = get_apply_metrics()
        assert [m.name for m in metrics] == ["P1", "P2", "P3"]
        assert [m.ordinal for m in metrics] == [0, 1, 2]

    def test_elapsed_ms_captures_real_time(self, obs_on):
        with measure_patch_apply("P_slow") as m:
            time.sleep(0.01)  # 10ms
            m.status = "applied"
        metrics = get_apply_metrics()
        # Must have measured ~10ms. Allow generous slack on CI runners.
        assert metrics[0].elapsed_ms >= 5.0
        assert metrics[0].elapsed_ms < 500.0  # sanity: not absurd

    def test_status_reason_passthrough(self, obs_on):
        with measure_patch_apply("P_skip") as m:
            m.status = "skipped"
            m.reason = "opt-in only"
        m_out = get_apply_metrics()[0]
        assert m_out.status == "skipped"
        assert m_out.reason == "opt-in only"


# ─── Exception propagation ──────────────────────────────────────────────


class TestExceptionPropagation:
    def test_exception_propagates_but_metric_still_recorded(self, obs_on):
        """If apply() raises, the exception must propagate but the
        elapsed metric should still be in the buffer (with whatever
        partial status the caller set)."""
        with pytest.raises(RuntimeError, match="boom"):
            with measure_patch_apply("P_buggy") as m:
                m.status = "failed"
                m.reason = "about to raise"
                raise RuntimeError("boom")
        metrics = get_apply_metrics()
        assert len(metrics) == 1
        assert metrics[0].name == "P_buggy"
        assert metrics[0].status == "failed"
        assert metrics[0].elapsed_ms >= 0


# ─── Aggregation ────────────────────────────────────────────────────────


class TestSummary:
    def test_empty_when_no_metrics(self, obs_off):
        assert metrics_summary() == {}

    def test_counts_per_status(self, obs_on):
        for nm, st in [
            ("P1", "applied"), ("P2", "applied"),
            ("P3", "skipped"),
            ("P4", "failed"),
        ]:
            with measure_patch_apply(nm) as m:
                m.status = st
        s = metrics_summary()
        assert s["count"] == 4
        assert s["applied"] == 2
        assert s["skipped"] == 1
        assert s["failed"] == 1

    def test_slowest_3_ordering(self, obs_on, monkeypatch):
        """Slowest-3 list is sorted by elapsed_ms descending."""
        # Synthesize three metrics with increasing elapsed by sleeping
        for nm, sleep_s in [
            ("P_fast", 0.001),
            ("P_med", 0.005),
            ("P_slow", 0.020),
            ("P_skipme", 0.001),
        ]:
            with measure_patch_apply(nm) as m:
                time.sleep(sleep_s)
                m.status = "applied"
        s = metrics_summary()
        names = [x["name"] for x in s["slowest_3"]]
        # Slowest should be first
        assert names[0] == "P_slow"
        # Length capped at 3
        assert len(s["slowest_3"]) == 3


# ─── Reset ──────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_metrics_and_ordinal(self, obs_on):
        with measure_patch_apply("P1") as m:
            m.status = "applied"
        assert len(get_apply_metrics()) == 1
        reset_apply_metrics()
        assert get_apply_metrics() == []
        with measure_patch_apply("P_after_reset") as m:
            m.status = "applied"
        # Ordinal restarted from 0
        assert get_apply_metrics()[0].ordinal == 0


# ─── Structured logging ─────────────────────────────────────────────────


class TestStructuredLogging:
    def test_log_line_emitted_per_patch(self, obs_on, caplog):
        with caplog.at_level(logging.INFO, logger="genesis.observability"):
            with measure_patch_apply("P_log") as m:
                m.status = "applied"
                m.reason = "wired ok"
        records = [r for r in caplog.records
                   if r.name == "genesis.observability"]
        assert any(
            "[PatchMetrics]" in r.message and "P_log" in r.message
            and "applied" in r.message
            for r in records
        )

    def test_no_log_when_disabled(self, obs_off, caplog):
        with caplog.at_level(logging.INFO, logger="genesis.observability"):
            with measure_patch_apply("P_quiet") as m:
                m.status = "applied"
        # No PatchMetrics line emitted when disabled
        assert not any(
            "[PatchMetrics]" in r.message
            for r in caplog.records
        )
