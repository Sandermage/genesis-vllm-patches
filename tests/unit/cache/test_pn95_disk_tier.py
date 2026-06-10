# SPDX-License-Identifier: Apache-2.0
"""Tests for the PN95 Tier 3 disk-backed prefix store.

Covers the public API (`disk_tier_set` / `disk_tier_get` / etc.) plus
the spillover/promote integration with the in-memory Tier 2 prefix
store in `_pn95_runtime`.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def disk_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def disk_env(disk_dir, monkeypatch):
    """Enable disk tier with a tempdir + small capacity."""
    monkeypatch.setenv("GENESIS_PN95_DISK_TIER_ENABLE", "1")
    monkeypatch.setenv("GENESIS_PN95_DISK_TIER_DIR", str(disk_dir))
    monkeypatch.setenv("GENESIS_PN95_DISK_TIER_CAPACITY_GIB", "0.0001")  # ~100 KB
    from sndr.cache import _pn95_disk_tier as dt
    dt.reset_for_tests()
    yield dt
    dt.reset_for_tests()


# ─── Master gate ─────────────────────────────────────────────────────


class TestEnableGate:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("GENESIS_PN95_DISK_TIER_ENABLE", raising=False)
        from sndr.cache import _pn95_disk_tier as dt
        dt.reset_for_tests()
        assert dt._enabled() is False

    def test_explicit_enable(self, monkeypatch):
        monkeypatch.setenv("GENESIS_PN95_DISK_TIER_ENABLE", "1")
        from sndr.cache import _pn95_disk_tier as dt
        dt.reset_for_tests()
        assert dt._enabled() is True

    def test_disabled_set_returns_false(self, monkeypatch):
        monkeypatch.delenv("GENESIS_PN95_DISK_TIER_ENABLE", raising=False)
        from sndr.cache import _pn95_disk_tier as dt
        dt.reset_for_tests()
        assert dt.disk_tier_set("k", [("l", b"x")]) is False
        assert dt.disk_tier_get("k") is None


# ─── Roundtrip / persistence ─────────────────────────────────────────


class TestRoundtrip:
    def test_set_then_get(self, disk_env):
        payload = [("layer.0", b"\xab" * 64), ("layer.1", b"\xcd" * 64)]
        assert disk_env.disk_tier_set("hashA", payload) is True
        got = disk_env.disk_tier_get("hashA")
        assert got == payload

    def test_get_miss_returns_none(self, disk_env):
        assert disk_env.disk_tier_get("no-such-hash") is None

    def test_empty_layer_data_rejected(self, disk_env):
        assert disk_env.disk_tier_set("hash", []) is False

    def test_delete(self, disk_env):
        payload = [("l", b"x")]
        disk_env.disk_tier_set("k", payload)
        assert disk_env.disk_tier_get("k") is not None
        assert disk_env.disk_tier_delete("k") is True
        assert disk_env.disk_tier_get("k") is None
        # Second delete returns False (idempotent).
        assert disk_env.disk_tier_delete("k") is False

    def test_filename_collision_safe(self, disk_env):
        """Different block hashes hash to different filenames."""
        disk_env.disk_tier_set(("group1", 12345), [("l", b"A")])
        disk_env.disk_tier_set(("group2", 12345), [("l", b"B")])
        assert disk_env.disk_tier_get(("group1", 12345)) == [("l", b"A")]
        assert disk_env.disk_tier_get(("group2", 12345)) == [("l", b"B")]


# ─── Capacity / eviction ─────────────────────────────────────────────


class TestEviction:
    def test_evict_until_fit_under_pressure(self, disk_env):
        payload = [("layer.0", b"\xab" * 1024), ("layer.1", b"\xcd" * 1024)]
        # 80 × ~2 KB entries against a ~100 KB cap → multiple evictions
        for i in range(80):
            disk_env.disk_tier_set(f"hash{i:04d}", payload)
        stats = disk_env.disk_tier_stats()
        assert stats["disk_evictions_total"] > 0
        # Some entries remain (the most recent), but well below 80
        assert 0 < stats["disk_entries"] < 80

    def test_evict_oldest_manual(self, disk_env):
        for i in range(3):
            disk_env.disk_tier_set(f"k{i}", [("l", b"x" * 16)])
        before = disk_env.disk_tier_stats()["disk_entries"]
        evicted = disk_env.disk_tier_evict_oldest()
        after = disk_env.disk_tier_stats()["disk_entries"]
        assert evicted > 0
        assert after == before - 1

    def test_evict_oldest_empty(self, disk_env):
        # Empty tier returns 0 bytes evicted, no error.
        assert disk_env.disk_tier_evict_oldest() == 0


# ─── Stats integrity ─────────────────────────────────────────────────


class TestStats:
    def test_writes_and_bytes_counters(self, disk_env):
        payload = [("l", b"y" * 200)]
        for i in range(5):
            disk_env.disk_tier_set(f"h{i}", payload)
        stats = disk_env.disk_tier_stats()
        assert stats["disk_writes_total"] >= 5
        assert stats["disk_bytes_written_total"] > 0
        assert stats["disk_entries"] >= 1

    def test_read_hits_counter(self, disk_env):
        disk_env.disk_tier_set("h", [("l", b"x")])
        disk_env.disk_tier_get("h")
        disk_env.disk_tier_get("h")
        disk_env.disk_tier_get("nope")
        stats = disk_env.disk_tier_stats()
        assert stats["disk_reads_total"] == 3
        assert stats["disk_read_hits_total"] == 2


# ─── Integration with _pn95_runtime ──────────────────────────────────


class TestRuntimeSpillover:
    """When the CPU prefix store is at capacity and disk tier is on,
    LRU eviction must spill to disk before discarding."""

    def test_evict_spills_to_disk_when_enabled(
        self, disk_env, monkeypatch,
    ):
        from sndr.cache import _pn95_runtime as rt
        monkeypatch.setenv("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", "1")
        monkeypatch.setenv("GENESIS_PN95_PREFIX_STORE_GIB", "0.0000001")  # ~100 bytes
        rt.reset_for_tests()
        rt._PN95_PREFIX_STORE_MAX_BYTES_CACHED = None  # force re-read

        # Manually seed CPU store with an entry, then force evict.
        bh = ("group", 99)
        rt._PN95_PREFIX_STORE[bh] = [("layer.0", b"A" * 64), ("layer.1", b"B" * 64)]
        rt._PN95_PREFIX_STORE_BYTES_USED = 128
        # Now ask to fit something bigger than what remains → eviction
        rt._prefix_store_evict_until_fit(10_000)
        # CPU store cleared; entry spilled to disk
        assert bh not in rt._PN95_PREFIX_STORE
        got = disk_env.disk_tier_get(bh)
        assert got is not None
        assert got == [("layer.0", b"A" * 64), ("layer.1", b"B" * 64)]
        assert rt._PN95_STATS.get("ram_to_disk_spills_total", 0) >= 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
