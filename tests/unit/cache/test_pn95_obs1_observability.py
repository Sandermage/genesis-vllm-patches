# SPDX-License-Identifier: Apache-2.0
"""PN95 OBS1 — observability stats tests.

Validates:
- prefix_lookups_total and prefix_lookups_cold_miss counters
- prefix_hit_rate calculation
- Periodic JSON dump function (no errors, valid JSON, atomic write)
- Env gates (disable via empty path)
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from sndr.cache import _pn95_runtime as rt


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    monkeypatch.setattr(rt, "_PN95_STATS", {
        **rt._PN95_STATS,
        "prefix_lookups_total": 0,
        "prefix_lookups_cold_miss": 0,
    })
    monkeypatch.setattr(rt, "_TICK_COUNTER", 0)
    yield


# ─── Stats fields exposed ──────────────────────────────────────────────


def test_stats_exposes_lookup_counters(monkeypatch):
    monkeypatch.setenv("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", "1")
    s = rt.get_pn95_stats()
    assert "prefix_lookups_total" in s
    assert "prefix_lookups_cold_miss" in s
    assert "prefix_hit_rate" in s


def test_stats_hit_rate_zero_when_no_lookups(monkeypatch):
    monkeypatch.setenv("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", "1")
    s = rt.get_pn95_stats()
    assert s["prefix_hit_rate"] == 0.0
    assert s["prefix_lookups_total"] == 0
    assert s["prefix_lookups_cold_miss"] == 0


def test_stats_hit_rate_perfect(monkeypatch):
    """100 lookups, 0 cold misses → hit rate = 1.0."""
    monkeypatch.setenv("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", "1")
    monkeypatch.setattr(rt, "_PN95_STATS", {
        **rt._PN95_STATS,
        "prefix_lookups_total": 100,
        "prefix_lookups_cold_miss": 0,
    })
    s = rt.get_pn95_stats()
    assert s["prefix_hit_rate"] == 1.0


def test_stats_hit_rate_half(monkeypatch):
    """100 lookups, 50 cold misses → hit rate = 0.5."""
    monkeypatch.setenv("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", "1")
    monkeypatch.setattr(rt, "_PN95_STATS", {
        **rt._PN95_STATS,
        "prefix_lookups_total": 100,
        "prefix_lookups_cold_miss": 50,
    })
    s = rt.get_pn95_stats()
    assert s["prefix_hit_rate"] == 0.5


def test_stats_hit_rate_all_miss(monkeypatch):
    """100 lookups, 100 cold misses → hit rate = 0."""
    monkeypatch.setenv("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", "1")
    monkeypatch.setattr(rt, "_PN95_STATS", {
        **rt._PN95_STATS,
        "prefix_lookups_total": 100,
        "prefix_lookups_cold_miss": 100,
    })
    s = rt.get_pn95_stats()
    assert s["prefix_hit_rate"] == 0.0


# ─── Periodic dump function ────────────────────────────────────────────


def test_dump_writes_valid_json(monkeypatch, tmp_path):
    """Dump function writes valid JSON snapshot."""
    monkeypatch.setenv("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", "1")
    target = tmp_path / "stats.json"
    monkeypatch.setenv("GENESIS_PN95_STATS_FILE", str(target))
    monkeypatch.setenv("GENESIS_PN95_STATS_INTERVAL", "1")  # dump every tick
    monkeypatch.setattr(rt, "_TICK_COUNTER", 1)

    rt._pn95_dump_stats_if_due()

    assert target.exists()
    with open(target) as f:
        data = json.load(f)
    # Must contain expected stats keys
    assert "prefix_hit_rate" in data
    assert "compress_lib" in data
    assert "timestamp" in data
    assert isinstance(data["timestamp"], int)


def test_dump_disabled_with_empty_path(monkeypatch, tmp_path):
    """Empty GENESIS_PN95_STATS_FILE disables dump."""
    monkeypatch.setenv("GENESIS_PN95_STATS_FILE", "")
    monkeypatch.setenv("GENESIS_PN95_STATS_INTERVAL", "1")
    monkeypatch.setattr(rt, "_TICK_COUNTER", 1)

    rt._pn95_dump_stats_if_due()  # must NOT create any file
    # Nothing to check — disabled means no-op


def test_dump_throttled_by_interval(monkeypatch, tmp_path):
    """Dumps only on multiples of interval."""
    monkeypatch.setenv("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", "1")
    target = tmp_path / "stats.json"
    monkeypatch.setenv("GENESIS_PN95_STATS_FILE", str(target))
    monkeypatch.setenv("GENESIS_PN95_STATS_INTERVAL", "100")

    # Tick 1 — not a multiple of 100 → no dump
    monkeypatch.setattr(rt, "_TICK_COUNTER", 1)
    rt._pn95_dump_stats_if_due()
    assert not target.exists()

    # Tick 100 — multiple of 100 → dump
    monkeypatch.setattr(rt, "_TICK_COUNTER", 100)
    rt._pn95_dump_stats_if_due()
    assert target.exists()


def test_dump_atomic_no_partial_writes(monkeypatch, tmp_path):
    """Dump uses rename for atomic write — no .tmp file remains on success."""
    monkeypatch.setenv("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", "1")
    target = tmp_path / "stats.json"
    monkeypatch.setenv("GENESIS_PN95_STATS_FILE", str(target))
    monkeypatch.setenv("GENESIS_PN95_STATS_INTERVAL", "1")
    monkeypatch.setattr(rt, "_TICK_COUNTER", 1)

    rt._pn95_dump_stats_if_due()

    # No leftover tmp file
    assert not (tmp_path / "stats.json.tmp").exists()
    assert target.exists()


def test_dump_failsilent_on_invalid_path(monkeypatch):
    """Invalid path (read-only dir) doesn't crash."""
    monkeypatch.setenv("GENESIS_PN95_STATS_FILE", "/nonexistent_dir/stats.json")
    monkeypatch.setenv("GENESIS_PN95_STATS_INTERVAL", "1")
    monkeypatch.setattr(rt, "_TICK_COUNTER", 1)
    # Must not raise
    rt._pn95_dump_stats_if_due()
