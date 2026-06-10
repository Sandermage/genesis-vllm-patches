# SPDX-License-Identifier: Apache-2.0
"""Unit tests for WorkspaceFacade — extracted policy module for the 4
TurboQuant workspace patches (P98 / P99 / PN118 / SNDR_WORKSPACE_001).

These tests encode the byte-equivalence contract: every decision
function must produce the same verdict that the equivalent text-patch
code path produces. When v12.0.0 wires the patches to delegate into the
facade, these tests guarantee the delegation is byte-equivalent.
"""
from __future__ import annotations

import pytest


# ──────────────────────────────────────────────────────────────────────
# Test fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_facade_state():
    """Each test gets a clean WorkspaceFacade — memo cache cleared,
    counters zeroed."""
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    WorkspaceFacade.clear_for_tests()
    yield
    WorkspaceFacade.clear_for_tests()


class _FakeLayer:
    """Mimics a vLLM TQ attention layer with legacy _tq_*_buf attrs."""

    def __init__(self, mid_o=None, output=None, lse=None):
        self._tq_mid_o_buf = mid_o
        self._tq_output_buf = output
        self._tq_lse_buf = lse


# ──────────────────────────────────────────────────────────────────────
# P98 — decide_decode_path
# ──────────────────────────────────────────────────────────────────────


def test_p98_disabled_returns_use_manager(monkeypatch):
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    monkeypatch.delenv("GENESIS_ENABLE_P98", raising=False)
    d = WorkspaceFacade.decide_decode_path(layer=_FakeLayer("a", "b", "c"))
    assert d.verdict == "use_manager"
    assert "P98 disabled" in d.reason


def test_p98_enabled_non_decode_returns_use_manager(monkeypatch):
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    monkeypatch.setenv("GENESIS_ENABLE_P98", "1")
    d = WorkspaceFacade.decide_decode_path(
        layer=_FakeLayer("a", "b", "c"), is_decode=False,
    )
    assert d.verdict == "use_manager"
    assert "non-decode" in d.reason


def test_p98_enabled_decode_legacy_attrs_returns_fast_path(monkeypatch):
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    monkeypatch.setenv("GENESIS_ENABLE_P98", "1")
    layer = _FakeLayer(mid_o="MID", output="OUT", lse="LSE")
    d = WorkspaceFacade.decide_decode_path(layer=layer)
    assert d.verdict == "use_fast_path"
    assert d.extra == ("MID", "OUT", "LSE")


def test_p98_enabled_decode_no_legacy_attrs_falls_back_to_manager(monkeypatch):
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    monkeypatch.setenv("GENESIS_ENABLE_P98", "1")
    layer = _FakeLayer(mid_o=None, output=None, lse=None)
    d = WorkspaceFacade.decide_decode_path(layer=layer)
    assert d.verdict == "use_manager"
    assert "lacks legacy" in d.reason


def test_p98_enabled_decode_none_layer_falls_back(monkeypatch):
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    monkeypatch.setenv("GENESIS_ENABLE_P98", "1")
    d = WorkspaceFacade.decide_decode_path(layer=None)
    assert d.verdict == "use_manager"


def test_p98_counters_track_decisions(monkeypatch):
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    monkeypatch.setenv("GENESIS_ENABLE_P98", "1")
    layer = _FakeLayer("a", "b", "c")
    for _ in range(3):
        WorkspaceFacade.decide_decode_path(layer=layer)
    stats = WorkspaceFacade.stats()
    assert stats["decode_revert_fast_path"] == 3


# ──────────────────────────────────────────────────────────────────────
# P99 — memoization cache
# ──────────────────────────────────────────────────────────────────────


def test_p99_disabled_always_returns_miss(monkeypatch):
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    monkeypatch.delenv("GENESIS_ENABLE_P99", raising=False)
    d = WorkspaceFacade.lookup_memo(
        shapes_and_dtypes_key=("shape", "dtype"),
        ubatch_id=0, ws_ptr=12345,
    )
    assert d.verdict == "cache_miss"
    assert "P99 disabled" in d.reason


def test_p99_store_disabled_is_noop(monkeypatch):
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    monkeypatch.delenv("GENESIS_ENABLE_P99", raising=False)
    d = WorkspaceFacade.store_memo(("k",), 0, 1, ["t1", "t2"])
    assert d.verdict == "noop"
    assert WorkspaceFacade.memo_size() == 0


def test_p99_store_then_lookup_hits(monkeypatch):
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    monkeypatch.setenv("GENESIS_ENABLE_P99", "1")
    WorkspaceFacade.store_memo(("k",), 0, 1, ["t1", "t2"])
    d = WorkspaceFacade.lookup_memo(("k",), 0, 1)
    assert d.verdict == "cache_hit"
    assert d.extra == ["t1", "t2"]


def test_p99_different_ws_ptr_misses_cache(monkeypatch):
    """ws_ptr discrimination: a new workspace allocation invalidates
    cached views from the old allocation."""
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    monkeypatch.setenv("GENESIS_ENABLE_P99", "1")
    WorkspaceFacade.store_memo(("k",), 0, 1, ["t1"])
    d = WorkspaceFacade.lookup_memo(("k",), 0, 2)  # different ws_ptr
    assert d.verdict == "cache_miss"


def test_p99_different_ubatch_misses_cache(monkeypatch):
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    monkeypatch.setenv("GENESIS_ENABLE_P99", "1")
    WorkspaceFacade.store_memo(("k",), 0, 1, ["t1"])
    d = WorkspaceFacade.lookup_memo(("k",), 1, 1)  # different ubatch
    assert d.verdict == "cache_miss"


def test_p99_fifo_eviction_when_over_capacity(monkeypatch):
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    monkeypatch.setenv("GENESIS_ENABLE_P99", "1")
    # Shrink cache for the test
    original_max = WorkspaceFacade._MEMO_MAX_ENTRIES
    WorkspaceFacade._MEMO_MAX_ENTRIES = 3
    try:
        WorkspaceFacade.store_memo(("k1",), 0, 1, ["a"])
        WorkspaceFacade.store_memo(("k2",), 0, 1, ["b"])
        WorkspaceFacade.store_memo(("k3",), 0, 1, ["c"])
        assert WorkspaceFacade.memo_size() == 3
        # Insert 4th → evicts k1
        WorkspaceFacade.store_memo(("k4",), 0, 1, ["d"])
        assert WorkspaceFacade.memo_size() == 3
        assert WorkspaceFacade.lookup_memo(("k1",), 0, 1).verdict == "cache_miss"
        assert WorkspaceFacade.lookup_memo(("k4",), 0, 1).verdict == "cache_hit"
        stats = WorkspaceFacade.stats()
        assert stats["memo_evict"] >= 1
    finally:
        WorkspaceFacade._MEMO_MAX_ENTRIES = original_max


def test_p99_store_refresh_on_existing_key(monkeypatch):
    """Re-storing an existing key updates the stored views, doesn't
    create a duplicate entry."""
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    monkeypatch.setenv("GENESIS_ENABLE_P99", "1")
    WorkspaceFacade.store_memo(("k",), 0, 1, ["v1"])
    WorkspaceFacade.store_memo(("k",), 0, 1, ["v2"])
    assert WorkspaceFacade.memo_size() == 1
    d = WorkspaceFacade.lookup_memo(("k",), 0, 1)
    assert d.extra == ["v2"]


# ──────────────────────────────────────────────────────────────────────
# PN118 — try_acquire graceful fallback
# ──────────────────────────────────────────────────────────────────────


def test_pn118_disabled_returns_pass(monkeypatch):
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    monkeypatch.delenv("GENESIS_ENABLE_PN118", raising=False)
    d = WorkspaceFacade.decide_try_acquire(
        is_locked=True, current_size_bytes=100, required_total_bytes=200,
    )
    assert d.verdict == "pass"
    assert "PN118 disabled" in d.reason


def test_pn118_enabled_unlocked_passes(monkeypatch):
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    monkeypatch.setenv("GENESIS_ENABLE_PN118", "1")
    d = WorkspaceFacade.decide_try_acquire(
        is_locked=False, current_size_bytes=100, required_total_bytes=200,
    )
    assert d.verdict == "pass"
    assert "unlocked" in d.reason


def test_pn118_enabled_locked_fitting_request_passes(monkeypatch):
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    monkeypatch.setenv("GENESIS_ENABLE_PN118", "1")
    d = WorkspaceFacade.decide_try_acquire(
        is_locked=True, current_size_bytes=200, required_total_bytes=150,
    )
    assert d.verdict == "pass"
    assert "fits" in d.reason


def test_pn118_enabled_locked_undersized_graceful_fallback(monkeypatch):
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    monkeypatch.setenv("GENESIS_ENABLE_PN118", "1")
    d = WorkspaceFacade.decide_try_acquire(
        is_locked=True, current_size_bytes=100, required_total_bytes=200,
    )
    assert d.verdict == "graceful_fallback"
    assert "locked + undersized" in d.reason


# ──────────────────────────────────────────────────────────────────────
# SNDR_WORKSPACE_001 — grow-after-lock guard
# ──────────────────────────────────────────────────────────────────────


def test_sndr001_disabled_returns_raise(monkeypatch):
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    monkeypatch.delenv("GENESIS_ENABLE_SNDR_WORKSPACE_001", raising=False)
    d = WorkspaceFacade.decide_grow_after_lock(is_locked=True)
    assert d.verdict == "raise"


def test_sndr001_enabled_unlocked_returns_no_guard_needed(monkeypatch):
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    monkeypatch.setenv("GENESIS_ENABLE_SNDR_WORKSPACE_001", "1")
    d = WorkspaceFacade.decide_grow_after_lock(is_locked=False)
    assert d.verdict == "no_guard_needed"


def test_sndr001_enabled_locked_warn_and_grow(monkeypatch):
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    monkeypatch.setenv("GENESIS_ENABLE_SNDR_WORKSPACE_001", "1")
    d = WorkspaceFacade.decide_grow_after_lock(is_locked=True)
    assert d.verdict == "warn_and_grow"


# ──────────────────────────────────────────────────────────────────────
# Composition — decide_get_simultaneous
# ──────────────────────────────────────────────────────────────────────


def test_composition_p98_fast_path_short_circuits_everything(monkeypatch):
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    monkeypatch.setenv("GENESIS_ENABLE_P98", "1")
    layer = _FakeLayer("a", "b", "c")
    result = WorkspaceFacade.decide_get_simultaneous(
        is_decode=True, layer=layer,
        shapes_and_dtypes_key=("k",), ubatch_id=0, ws_ptr=1,
        is_locked=False, current_size_bytes=100, required_total_bytes=50,
    )
    assert result["next_action"] == "use_fast_path_buffers"
    assert result["P99"] is None
    assert result["PN118"] is None
    assert result["SNDR_WORKSPACE_001"] is None


def test_composition_p99_cache_hit_short_circuits_pn118_sndr(monkeypatch):
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    monkeypatch.delenv("GENESIS_ENABLE_P98", raising=False)
    monkeypatch.setenv("GENESIS_ENABLE_P99", "1")
    WorkspaceFacade.store_memo(("k",), 0, 1, ["t1"])
    result = WorkspaceFacade.decide_get_simultaneous(
        is_decode=True, layer=None,
        shapes_and_dtypes_key=("k",), ubatch_id=0, ws_ptr=1,
        is_locked=False, current_size_bytes=100, required_total_bytes=50,
    )
    assert result["next_action"] == "return_cached_views"
    assert result["P99"].verdict == "cache_hit"
    assert result["PN118"] is None


def test_composition_pn118_graceful_fallback_chain(monkeypatch):
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    monkeypatch.setenv("GENESIS_ENABLE_PN118", "1")
    result = WorkspaceFacade.decide_get_simultaneous(
        is_decode=True, layer=None,
        shapes_and_dtypes_key=("k",), ubatch_id=0, ws_ptr=1,
        is_locked=True, current_size_bytes=100, required_total_bytes=300,
    )
    assert result["next_action"] == "caller_uses_torch_empty"
    assert result["PN118"].verdict == "graceful_fallback"


def test_composition_sndr001_warn_grow_when_all_others_pass(monkeypatch):
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    monkeypatch.setenv("GENESIS_ENABLE_SNDR_WORKSPACE_001", "1")
    result = WorkspaceFacade.decide_get_simultaneous(
        is_decode=True, layer=None,
        shapes_and_dtypes_key=("k",), ubatch_id=0, ws_ptr=1,
        is_locked=True, current_size_bytes=300, required_total_bytes=100,
    )
    assert result["next_action"] == "log_warn_and_proceed_with_grow"


def test_composition_all_disabled_returns_normal_acquire(monkeypatch):
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    for flag in (
        "GENESIS_ENABLE_P98",
        "GENESIS_ENABLE_P99",
        "GENESIS_ENABLE_PN118",
        "GENESIS_ENABLE_SNDR_WORKSPACE_001",
    ):
        monkeypatch.delenv(flag, raising=False)
    result = WorkspaceFacade.decide_get_simultaneous(
        is_decode=True, layer=None,
        shapes_and_dtypes_key=("k",), ubatch_id=0, ws_ptr=1,
        is_locked=False, current_size_bytes=100, required_total_bytes=50,
    )
    assert result["next_action"] == "normal_acquire"


# ──────────────────────────────────────────────────────────────────────
# Observability + maintenance
# ──────────────────────────────────────────────────────────────────────


def test_summary_reports_env_flags_and_stats(monkeypatch):
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    monkeypatch.setenv("GENESIS_ENABLE_P99", "1")
    monkeypatch.setenv("GENESIS_ENABLE_PN118", "1")
    WorkspaceFacade.store_memo(("k",), 0, 1, ["t1"])
    WorkspaceFacade.lookup_memo(("k",), 0, 1)
    s = WorkspaceFacade.summary()
    assert s["env_flags"]["P99"] is True
    assert s["env_flags"]["PN118"] is True
    assert s["env_flags"]["P98"] is False
    assert s["env_flags"]["SNDR_WORKSPACE_001"] is False
    assert s["memo_size"] == 1
    assert s["stats"]["memo_hit"] == 1


def test_clear_for_tests_resets_state():
    from sndr.engines.vllm.kernels_legacy.workspace_facade import WorkspaceFacade
    WorkspaceFacade._STATS["memo_hit"] = 999
    WorkspaceFacade.clear_for_tests()
    assert WorkspaceFacade.stats()["memo_hit"] == 0
    assert WorkspaceFacade.memo_size() == 0
