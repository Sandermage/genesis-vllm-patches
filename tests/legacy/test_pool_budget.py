# SPDX-License-Identifier: Apache-2.0
"""TDD for vllm/_genesis/pool_budget.py — per-pool VRAM budget primitive.

Phase B of MEMORY_DEEP_PLAN architectural pass (Sander 2026-05-07).
Mirrors the buffer_mode.py pattern: env-gated, default OFF, dynamo-safe
caching, raise-on-overflow with caller fallback.
"""
from __future__ import annotations

import pytest

from sndr.runtime import pool_budget
from sndr.runtime.pool_budget import (
    PoolBudgetExceeded,
    assert_total_under_budget,
    check,
    deduct,
    max_mib_for,
    record,
    summary,
    total_max_mib,
    usage_bytes,
)


@pytest.fixture(autouse=True)
def _reset_budget(monkeypatch):
    """Clear all caches + usage between tests; clear all pool budget envs."""
    # Strip every env that affects pool_budget — be aggressive
    for env_name in (
        "GENESIS_POOL_MAX_MIB",
        "GENESIS_POOL_TOTAL_MAX_MIB",
        "GENESIS_POOL_MAX_MIB_PN59",
        "GENESIS_POOL_MAX_MIB_PN12",
        "GENESIS_POOL_MAX_MIB_P38",
        "GENESIS_POOL_MAX_MIB_P39A",
        "GENESIS_POOL_MAX_MIB_TESTPATCH",
    ):
        monkeypatch.delenv(env_name, raising=False)
    pool_budget._reset_caches()
    yield
    pool_budget._reset_caches()


# ─── Default unlimited (zero-risk pre-Phase-B behavior) ───────────────


class TestDefaultUnlimited:
    def test_max_mib_for_returns_none_when_unset(self):
        assert max_mib_for("PN59") is None

    def test_total_max_mib_returns_none_when_unset(self):
        assert total_max_mib() is None

    def test_check_no_raise_when_unlimited(self):
        # Should not raise even for absurd request
        check("PN59", 100 * 1024 * 1024 * 1024)  # 100 GiB

    def test_assert_total_no_raise_when_unlimited(self):
        record("PN59", 50 * 1024 * 1024 * 1024)  # 50 GiB recorded
        assert_total_under_budget()  # unlimited → no raise

    def test_invalid_env_treated_as_unlimited(self, monkeypatch):
        monkeypatch.setenv("GENESIS_POOL_MAX_MIB", "abc-not-int")
        pool_budget._reset_caches()
        assert max_mib_for("PN59") is None

    def test_zero_env_treated_as_unlimited(self, monkeypatch):
        monkeypatch.setenv("GENESIS_POOL_MAX_MIB", "0")
        pool_budget._reset_caches()
        assert max_mib_for("PN59") is None

    def test_negative_env_treated_as_unlimited(self, monkeypatch):
        monkeypatch.setenv("GENESIS_POOL_MAX_MIB", "-100")
        pool_budget._reset_caches()
        assert max_mib_for("PN59") is None


# ─── Per-pool cap precedence ──────────────────────────────────────────


class TestPerPoolPrecedence:
    def test_per_pool_specific_wins_over_global(self, monkeypatch):
        monkeypatch.setenv("GENESIS_POOL_MAX_MIB", "500")
        monkeypatch.setenv("GENESIS_POOL_MAX_MIB_PN59", "200")
        pool_budget._reset_caches()
        assert max_mib_for("PN59") == 200
        # Other pool falls through to global
        assert max_mib_for("PN12") == 500

    def test_global_only(self, monkeypatch):
        monkeypatch.setenv("GENESIS_POOL_MAX_MIB", "1024")
        pool_budget._reset_caches()
        assert max_mib_for("PN59") == 1024
        assert max_mib_for("ANYPATCH") == 1024

    def test_per_pool_only(self, monkeypatch):
        monkeypatch.setenv("GENESIS_POOL_MAX_MIB_PN12", "256")
        pool_budget._reset_caches()
        assert max_mib_for("PN12") == 256
        assert max_mib_for("PN59") is None  # not set, falls through to None

    def test_case_insensitive_patch_id(self, monkeypatch):
        monkeypatch.setenv("GENESIS_POOL_MAX_MIB_PN59", "200")
        pool_budget._reset_caches()
        assert max_mib_for("pn59") == 200
        assert max_mib_for("Pn59") == 200


# ─── Usage accounting ─────────────────────────────────────────────────


class TestUsageAccounting:
    def test_record_increments(self):
        record("PN59", 100 * 1024 * 1024)  # 100 MiB
        assert usage_bytes("PN59") == 100 * 1024 * 1024

    def test_record_accumulates(self):
        record("PN59", 50 * 1024 * 1024)
        record("PN59", 30 * 1024 * 1024)
        assert usage_bytes("PN59") == 80 * 1024 * 1024

    def test_deduct_subtracts(self):
        record("PN59", 100 * 1024 * 1024)
        deduct("PN59", 30 * 1024 * 1024)
        assert usage_bytes("PN59") == 70 * 1024 * 1024

    def test_deduct_floors_at_zero(self):
        record("PN59", 50 * 1024 * 1024)
        deduct("PN59", 100 * 1024 * 1024)  # over-deduct
        assert usage_bytes("PN59") == 0

    def test_total_usage_sums_all(self):
        record("PN59", 100 * 1024 * 1024)
        record("PN12", 50 * 1024 * 1024)
        record("P38", 30 * 1024 * 1024)
        assert usage_bytes() == 180 * 1024 * 1024

    def test_zero_record_noop(self):
        record("PN59", 0)
        record("PN59", -100)
        assert usage_bytes("PN59") == 0


# ─── Per-pool check gate ──────────────────────────────────────────────


class TestCheckGate:
    def test_under_cap_no_raise(self, monkeypatch):
        monkeypatch.setenv("GENESIS_POOL_MAX_MIB_PN59", "200")
        pool_budget._reset_caches()
        check("PN59", 100 * 1024 * 1024)  # 100 MiB < 200 cap

    def test_at_cap_no_raise(self, monkeypatch):
        monkeypatch.setenv("GENESIS_POOL_MAX_MIB_PN59", "200")
        pool_budget._reset_caches()
        check("PN59", 200 * 1024 * 1024)  # 200 MiB == cap (≤ cap)

    def test_over_cap_raises(self, monkeypatch):
        monkeypatch.setenv("GENESIS_POOL_MAX_MIB_PN59", "200")
        pool_budget._reset_caches()
        with pytest.raises(PoolBudgetExceeded, match="PN59"):
            check("PN59", 201 * 1024 * 1024)

    def test_check_accounts_for_existing_usage(self, monkeypatch):
        monkeypatch.setenv("GENESIS_POOL_MAX_MIB_PN59", "200")
        pool_budget._reset_caches()
        record("PN59", 150 * 1024 * 1024)  # already 150 MiB live
        # Adding 60 MiB would bring total to 210 > 200 cap → raise
        with pytest.raises(PoolBudgetExceeded):
            check("PN59", 60 * 1024 * 1024)
        # 30 MiB still fits (150+30=180 < 200)
        check("PN59", 30 * 1024 * 1024)


# ─── Cross-pool total budget ──────────────────────────────────────────


class TestTotalBudget:
    def test_under_total_no_raise(self, monkeypatch):
        monkeypatch.setenv("GENESIS_POOL_TOTAL_MAX_MIB", "1024")
        pool_budget._reset_caches()
        record("PN59", 200 * 1024 * 1024)
        record("PN12", 300 * 1024 * 1024)
        assert_total_under_budget()  # 500 < 1024

    def test_over_total_raises(self, monkeypatch):
        monkeypatch.setenv("GENESIS_POOL_TOTAL_MAX_MIB", "500")
        pool_budget._reset_caches()
        record("PN59", 300 * 1024 * 1024)
        record("PN12", 250 * 1024 * 1024)
        with pytest.raises(PoolBudgetExceeded, match="cumulative"):
            assert_total_under_budget()

    def test_error_message_includes_per_pool_breakdown(self, monkeypatch):
        monkeypatch.setenv("GENESIS_POOL_TOTAL_MAX_MIB", "100")
        pool_budget._reset_caches()
        record("PN59", 80 * 1024 * 1024)
        record("PN12", 60 * 1024 * 1024)
        try:
            assert_total_under_budget()
            assert False, "should have raised"
        except PoolBudgetExceeded as e:
            msg = str(e)
            assert "PN59" in msg
            assert "PN12" in msg


# ─── Summary diagnostics ──────────────────────────────────────────────


class TestSummary:
    def test_summary_empty(self):
        s = summary()
        assert s["per_pool"] == {}
        assert s["total_usage_mib"] == 0
        assert s["total_cap_mib"] is None

    def test_summary_with_usage(self, monkeypatch):
        monkeypatch.setenv("GENESIS_POOL_MAX_MIB_PN59", "1024")
        monkeypatch.setenv("GENESIS_POOL_TOTAL_MAX_MIB", "4096")
        pool_budget._reset_caches()
        record("PN59", 200 * 1024 * 1024)
        record("PN12", 100 * 1024 * 1024)
        s = summary()
        assert s["per_pool"]["PN59"]["usage_mib"] == 200.0
        assert s["per_pool"]["PN59"]["cap_mib"] == 1024
        assert s["per_pool"]["PN12"]["usage_mib"] == 100.0
        assert s["total_usage_mib"] == 300.0
        assert s["total_cap_mib"] == 4096


# ─── Composition with PN59 cap (Phase A) ──────────────────────────────


class TestComposition:
    def test_pool_budget_independent_of_pn59_cap(self, monkeypatch):
        """PN59 has its own GENESIS_PN59_O_MAX_T (T-cap, Phase A).
        Pool budget is separate (bytes-cap, Phase B). Both can coexist:
        T-cap rejects single huge allocations, byte-cap rejects cumulative."""
        # Don't set PN59 T cap; set byte cap
        monkeypatch.setenv("GENESIS_POOL_MAX_MIB_PN59", "100")
        pool_budget._reset_caches()
        # First small alloc fits
        check("PN59", 50 * 1024 * 1024)
        record("PN59", 50 * 1024 * 1024)
        # Second alloc would exceed → raise
        with pytest.raises(PoolBudgetExceeded):
            check("PN59", 60 * 1024 * 1024)
