# SPDX-License-Identifier: Apache-2.0
"""PN282: tests for the spec-decode acceptance proxy metric module.

Covers:

  * Default OFF: record_acceptance is a no-op
  * Opt-in via SNDR_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC=1
  * Legacy alias GENESIS_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC accepted
  * Counters increment correctly per k bucket
  * calls_total increments once per record_acceptance call
  * Profile label resolves from env, frozen after first use
  * max_spec_len gauge tracks the most recent observation
  * Module-state reset isolates tests
  * No torch import required (offline collection safe)
"""
from __future__ import annotations

import pytest

prometheus_client = pytest.importorskip("prometheus_client")

from sndr.observability import spec_decode_metrics as sdm  # noqa: E402


@pytest.fixture(autouse=True)
def _module_reset(monkeypatch):
    """Wipe singleton state between tests."""
    # Reset module state including profile label cache + counter handles
    sdm._reset_module_state()
    # Ensure neither env flag is leaking from the host shell
    monkeypatch.delenv("SNDR_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC", raising=False)
    monkeypatch.delenv("GENESIS_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC", raising=False)
    monkeypatch.delenv("SNDR_SPEC_DECODE_PROFILE_LABEL", raising=False)
    yield
    sdm._reset_module_state()


@pytest.fixture
def metric_on(monkeypatch):
    monkeypatch.setenv("SNDR_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC", "1")
    yield


@pytest.fixture
def metric_on_legacy(monkeypatch):
    monkeypatch.setenv("GENESIS_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC", "1")
    yield


# Helper: read a counter's value for given labels from the global registry.
def _counter_value(name: str, **labels) -> float:
    from prometheus_client import REGISTRY
    val = REGISTRY.get_sample_value(name, labels) or 0.0
    return float(val)


# ─── Default OFF posture ────────────────────────────────────────────────


class TestDefaultOff:
    def test_record_is_noop_without_env(self):
        sdm.record_acceptance([0, 1, 2], max_spec_len=4)
        # No counter init expected; module state stays clean.
        assert sdm._ACCEPTED_COUNTER is None
        assert sdm._CALLS_COUNTER is None

    def test_is_enabled_false_without_env(self):
        assert sdm.is_enabled() is False


# ─── Opt-in via canonical env ───────────────────────────────────────────


class TestCanonicalEnv:
    def test_is_enabled_true_with_canonical_env(self, metric_on):
        assert sdm.is_enabled() is True

    def test_counter_increments_per_request(self, metric_on, monkeypatch):
        monkeypatch.setenv("SNDR_SPEC_DECODE_PROFILE_LABEL", "test-profile-A")
        sdm.record_acceptance([0, 2, 4], max_spec_len=4)
        profile = "test-profile-A"
        # Three requests → three increments across the k buckets.
        assert _counter_value(
            "sndr_spec_decode_accepted_per_call_total",
            k="0", profile=profile,
        ) == pytest.approx(1.0)
        assert _counter_value(
            "sndr_spec_decode_accepted_per_call_total",
            k="2", profile=profile,
        ) == pytest.approx(1.0)
        assert _counter_value(
            "sndr_spec_decode_accepted_per_call_total",
            k="4", profile=profile,
        ) == pytest.approx(1.0)
        # calls_total increments once per call, not per request.
        assert _counter_value(
            "sndr_spec_decode_calls_total", profile=profile,
        ) == pytest.approx(1.0)

    def test_calls_total_matches_record_count(self, metric_on, monkeypatch):
        monkeypatch.setenv("SNDR_SPEC_DECODE_PROFILE_LABEL", "test-profile-B")
        for _ in range(5):
            sdm.record_acceptance([1, 1], max_spec_len=4)
        assert _counter_value(
            "sndr_spec_decode_calls_total", profile="test-profile-B",
        ) == pytest.approx(5.0)

    def test_sum_of_k_counters_equals_total_request_outcomes(
        self, metric_on, monkeypatch,
    ):
        """Gate from user message: sum(k counters) == calls_total * batch_size
        when batch_size is constant; more generally, equals total request
        outcomes observed."""
        monkeypatch.setenv("SNDR_SPEC_DECODE_PROFILE_LABEL", "test-profile-C")
        # 3 calls with batch_size=4 each
        for _ in range(3):
            sdm.record_acceptance([0, 1, 2, 3], max_spec_len=4)
        profile = "test-profile-C"
        total_k = sum(
            _counter_value(
                "sndr_spec_decode_accepted_per_call_total",
                k=str(k), profile=profile,
            )
            for k in range(5)
        )
        # 3 calls * 4 requests = 12 total request outcomes
        assert total_k == pytest.approx(12.0)

    def test_max_spec_len_gauge_tracks_latest(self, metric_on, monkeypatch):
        monkeypatch.setenv("SNDR_SPEC_DECODE_PROFILE_LABEL", "test-profile-D")
        sdm.record_acceptance([0], max_spec_len=4)
        sdm.record_acceptance([0], max_spec_len=7)
        from prometheus_client import REGISTRY
        gauge_val = REGISTRY.get_sample_value(
            "sndr_spec_decode_max_spec_len",
            {"profile": "test-profile-D"},
        )
        assert gauge_val == pytest.approx(7.0)


# ─── Legacy alias ───────────────────────────────────────────────────────


class TestLegacyAlias:
    def test_is_enabled_true_with_legacy(self, metric_on_legacy):
        assert sdm.is_enabled() is True

    def test_legacy_warns_once(self, metric_on_legacy, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            # Call twice — should warn once.
            sdm.is_enabled()
            sdm.is_enabled()
        warnings = [
            r for r in caplog.records
            if "GENESIS_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC" in r.getMessage()
            and r.levelno == logging.WARNING
        ]
        assert len(warnings) == 1


# ─── Profile label resolution ───────────────────────────────────────────


class TestProfileLabel:
    def test_default_profile_when_unset(self, metric_on):
        assert sdm.get_profile_label() == "unknown"

    def test_profile_read_from_env(self, metric_on, monkeypatch):
        monkeypatch.setenv("SNDR_SPEC_DECODE_PROFILE_LABEL", "gemma4-test")
        assert sdm.get_profile_label() == "gemma4-test"

    def test_profile_frozen_after_first_use(self, metric_on, monkeypatch):
        monkeypatch.setenv("SNDR_SPEC_DECODE_PROFILE_LABEL", "first-profile")
        first = sdm.get_profile_label()
        # Change env after first read — should NOT propagate
        monkeypatch.setenv("SNDR_SPEC_DECODE_PROFILE_LABEL", "second-profile")
        second = sdm.get_profile_label()
        assert first == second == "first-profile"


# ─── Resilience ─────────────────────────────────────────────────────────


class TestResilience:
    def test_record_does_not_raise_on_empty_batch(self, metric_on):
        sdm.record_acceptance([], max_spec_len=4)
        # No counters touched (no request outcomes), but calls_total should
        # still increment because the call happened.
        # Note: calls_total uses the same profile resolution; verify via
        # the default "unknown" profile.
        assert _counter_value(
            "sndr_spec_decode_calls_total", profile="unknown",
        ) == pytest.approx(1.0)

    def test_record_does_not_raise_on_bad_values(self, metric_on):
        # Float values get cast via int(); negative values still cast fine.
        sdm.record_acceptance([0.0, 2.5], max_spec_len=4)
        # 2.5 -> int=2; 0.0 -> int=0
        assert _counter_value(
            "sndr_spec_decode_accepted_per_call_total",
            k="0", profile="unknown",
        ) == pytest.approx(1.0)
        assert _counter_value(
            "sndr_spec_decode_accepted_per_call_total",
            k="2", profile="unknown",
        ) == pytest.approx(1.0)
