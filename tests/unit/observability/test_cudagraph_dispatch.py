# SPDX-License-Identifier: Apache-2.0
"""Sprint 2.6: tests for CUDA graph dispatch hit-rate observability.

Covers:

  • Default OFF (no env): record_dispatch is a no-op
  • Opt-in via GENESIS_CUDAGRAPH_DISPATCH_TRACE=1
  • Hit + miss counters increment correctly
  • Snapshot reports hit_rate_pct / miss_rate_pct
  • Periodic emit fires every GENESIS_CUDAGRAPH_LOG_EVERY events
  • emit_summary() works on demand
  • reset_summary() / module reset for test isolation
  • Thread safety smoke (concurrent record_dispatch calls)
"""
from __future__ import annotations

import logging
import threading

import pytest

from sndr.observability import (
    CudagraphDispatchSummary,
    emit_cudagraph_summary,
    get_cudagraph_summary,
    record_cudagraph_dispatch,
    reset_cudagraph_summary,
)
from sndr.observability import cudagraph_dispatch as cgd


@pytest.fixture(autouse=True)
def _module_reset():
    """Wipe singleton state between tests."""
    cgd._reset_module_state()
    yield
    cgd._reset_module_state()


@pytest.fixture
def trace_on(monkeypatch):
    monkeypatch.setenv("GENESIS_CUDAGRAPH_DISPATCH_TRACE", "1")
    yield


@pytest.fixture
def trace_off(monkeypatch):
    monkeypatch.delenv("GENESIS_CUDAGRAPH_DISPATCH_TRACE", raising=False)
    yield


# ─── Default OFF posture ────────────────────────────────────────────────


class TestDefaultOff:
    def test_record_is_noop_without_env(self, trace_off):
        record_cudagraph_dispatch(matched=True)
        record_cudagraph_dispatch(matched=False)
        snap = get_cudagraph_summary()
        # No events recorded — counters stay at zero
        assert snap.hits == 0
        assert snap.misses == 0
        assert snap.total == 0
        assert snap.hit_rate_pct is None

    def test_get_summary_when_disabled_returns_zero(self, trace_off):
        snap = get_cudagraph_summary()
        assert isinstance(snap, CudagraphDispatchSummary)
        assert snap.hits == 0


# ─── Opt-in: counters work ──────────────────────────────────────────────


class TestCounters:
    def test_hit_increments(self, trace_on):
        record_cudagraph_dispatch(matched=True)
        snap = get_cudagraph_summary()
        assert snap.hits == 1
        assert snap.misses == 0
        assert snap.total == 1
        assert snap.hit_rate_pct == 100.0

    def test_miss_increments(self, trace_on):
        record_cudagraph_dispatch(matched=False)
        snap = get_cudagraph_summary()
        assert snap.hits == 0
        assert snap.misses == 1
        assert snap.miss_rate_pct == 100.0

    def test_mixed_hits_and_misses(self, trace_on):
        for _ in range(7):
            record_cudagraph_dispatch(matched=True)
        for _ in range(3):
            record_cudagraph_dispatch(matched=False)
        snap = get_cudagraph_summary()
        assert snap.hits == 7
        assert snap.misses == 3
        assert snap.total == 10
        assert snap.hit_rate_pct == 70.0
        assert snap.miss_rate_pct == 30.0

    def test_summary_dataclass_properties(self):
        s = CudagraphDispatchSummary(hits=80, misses=20)
        assert s.total == 100
        assert s.hit_rate_pct == 80.0
        assert s.miss_rate_pct == 20.0

    def test_zero_total_returns_none_rates(self):
        s = CudagraphDispatchSummary()
        assert s.hit_rate_pct is None
        assert s.miss_rate_pct is None


# ─── Periodic emit ──────────────────────────────────────────────────────


class TestPeriodicEmit:
    def test_emit_every_n_records(self, trace_on, monkeypatch, caplog):
        monkeypatch.setenv("GENESIS_CUDAGRAPH_LOG_EVERY", "5")
        with caplog.at_level(logging.INFO, logger="genesis.cudagraph"):
            for _ in range(5):
                record_cudagraph_dispatch(matched=True)
        # Exactly 1 summary line at the 5th record
        summaries = [r for r in caplog.records
                     if "dispatch hit-rate" in r.message]
        assert len(summaries) == 1

    def test_emit_does_not_fire_under_threshold(
        self, trace_on, monkeypatch, caplog,
    ):
        monkeypatch.setenv("GENESIS_CUDAGRAPH_LOG_EVERY", "100")
        with caplog.at_level(logging.INFO, logger="genesis.cudagraph"):
            for _ in range(50):
                record_cudagraph_dispatch(matched=True)
        summaries = [r for r in caplog.records
                     if "dispatch hit-rate" in r.message]
        assert len(summaries) == 0

    def test_log_every_zero_disables_periodic_emit(
        self, trace_on, monkeypatch, caplog,
    ):
        monkeypatch.setenv("GENESIS_CUDAGRAPH_LOG_EVERY", "0")
        with caplog.at_level(logging.INFO, logger="genesis.cudagraph"):
            for _ in range(100):
                record_cudagraph_dispatch(matched=True)
        # Zero means "never auto-emit" — operator must call emit_summary()
        summaries = [r for r in caplog.records
                     if "dispatch hit-rate" in r.message]
        assert len(summaries) == 0


# ─── On-demand emit ─────────────────────────────────────────────────────


class TestEmitSummary:
    def test_emit_summary_on_demand(self, trace_on, caplog):
        record_cudagraph_dispatch(matched=True)
        record_cudagraph_dispatch(matched=False)
        with caplog.at_level(logging.INFO, logger="genesis.cudagraph"):
            emit_cudagraph_summary()
        summaries = [r for r in caplog.records
                     if "dispatch hit-rate" in r.message]
        assert len(summaries) == 1
        # Hit-rate 50%
        assert "50.0" in summaries[0].message

    def test_emit_summary_no_events_is_silent(self, trace_on, caplog):
        with caplog.at_level(logging.INFO, logger="genesis.cudagraph"):
            emit_cudagraph_summary()
        summaries = [r for r in caplog.records
                     if "dispatch hit-rate" in r.message]
        assert summaries == []

    def test_emit_summary_when_counter_uninitialised_is_safe(self, trace_off):
        """Even with trace OFF and no counter created, emit_summary is
        a safe no-op (operator UX — calling it never raises)."""
        emit_cudagraph_summary()  # must not raise


# ─── Reset semantics ────────────────────────────────────────────────────


class TestReset:
    def test_reset_zeros_counters(self, trace_on):
        record_cudagraph_dispatch(matched=True)
        record_cudagraph_dispatch(matched=False)
        assert get_cudagraph_summary().total == 2
        reset_cudagraph_summary()
        assert get_cudagraph_summary().total == 0

    def test_reset_when_uninitialised_is_safe(self, trace_off):
        reset_cudagraph_summary()  # must not raise


# ─── Thread safety ──────────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_record_dispatch(self, trace_on, monkeypatch):
        # Disable periodic emit so concurrent calls don't slow on logging
        monkeypatch.setenv("GENESIS_CUDAGRAPH_LOG_EVERY", "0")

        N_THREADS = 8
        N_PER_THREAD = 250

        def worker(matched_ratio: float):
            for i in range(N_PER_THREAD):
                record_cudagraph_dispatch(matched=(i % 2 == 0))

        threads = [threading.Thread(target=worker, args=(0.5,))
                   for _ in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = get_cudagraph_summary()
        # All N_THREADS * N_PER_THREAD events recorded
        assert snap.total == N_THREADS * N_PER_THREAD
        # Hit rate should be ~50% (i % 2 == 0)
        assert 49.0 <= snap.hit_rate_pct <= 51.0


# ─── Env config helpers ────────────────────────────────────────────────


class TestEnvConfig:
    @pytest.mark.parametrize("v", ["1", "true", "yes", "y", "on", "TRUE"])
    def test_truthy_values_enable_trace(self, monkeypatch, v):
        monkeypatch.setenv("GENESIS_CUDAGRAPH_DISPATCH_TRACE", v)
        record_cudagraph_dispatch(matched=True)
        assert get_cudagraph_summary().hits == 1

    @pytest.mark.parametrize("v", ["0", "false", "no", "off", ""])
    def test_falsy_values_keep_disabled(self, monkeypatch, v):
        monkeypatch.setenv("GENESIS_CUDAGRAPH_DISPATCH_TRACE", v)
        record_cudagraph_dispatch(matched=True)
        assert get_cudagraph_summary().hits == 0

    def test_log_every_invalid_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("GENESIS_CUDAGRAPH_LOG_EVERY", "not-a-number")
        # Default 1000 used — module helper falls back gracefully
        from sndr.observability.cudagraph_dispatch import _log_every
        assert _log_every() == 1000

    def test_log_every_negative_clamped_to_zero(self, monkeypatch):
        monkeypatch.setenv("GENESIS_CUDAGRAPH_LOG_EVERY", "-5")
        from sndr.observability.cudagraph_dispatch import _log_every
        assert _log_every() == 0
