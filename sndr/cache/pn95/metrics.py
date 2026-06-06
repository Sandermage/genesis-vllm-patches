# SPDX-License-Identifier: Apache-2.0
"""PN95 observability — stats snapshot + periodic dump + lookup hit tracker.

M.4.1 scope: function extraction only. The mutable state singletons
this module operates on (``_PN95_STATS``, ``_PN95_PREFIX_STORE``,
``_PN95_PREFETCH_STATS``, ``_PN95_LAYER_ACCESS_COUNTS``,
``_PN95_COMPRESS_LIB``, ``_PN95_HIT_COUNTS``, ``_PN95_HIT_TRACKER_MAX``,
``_TICK_COUNTER``) continue to live in ``_pn95_runtime`` because
existing tests rebind several of these via ``monkeypatch.setattr`` —
moving the ownership would break a documented test-contract alias.
Each function therefore late-imports its state via
``from vllm.sndr_core.cache import _pn95_runtime as _rt`` inside the
body, which resolves at call time after both modules are loaded.

Full state-ownership redistribution is deferred to M.4.2 where the
broader split can reorganize the cross-module dependency graph.

Extracted from ``_pn95_runtime.py`` in M.4.1. The legacy module
re-exports each function so the operator-facing
``sndr patches pn95-status --json`` contract stays byte-identical and
text-patch anchors are unaffected.
"""
from __future__ import annotations

import os
from typing import Any


def get_pn95_stats() -> dict:
    """Path C v1.0 Phase 3 observability — returns counter snapshot.

    Used by the `sndr report` CLI and operator-facing tools to surface
    PN95 activity without needing access to the EngineCore worker process.
    """
    from vllm.sndr_core.cache import _pn95_runtime as _rt
    from .gates import _pn95_async_enabled, _pn95_layer_aware_enabled

    snapshot = dict(_rt._PN95_STATS)
    # Phase 4 — augment with prefix-store stats
    snapshot["prefix_store_entries"] = len(_rt._PN95_PREFIX_STORE)
    snapshot["prefix_store_bytes_used"] = _rt._PN95_PREFIX_STORE_BYTES_USED
    snapshot["prefix_store_promote_hits"] = _rt._PN95_STATS.get(
        "prefix_promote_hits", 0
    )
    snapshot["prefix_store_demotes"] = _rt._PN95_STATS.get(
        "prefix_demote_count", 0
    )
    # A1 — compression stats
    raw = _rt._PN95_STATS.get("compress_raw_bytes_total", 0)
    stored = _rt._PN95_STATS.get("compress_stored_bytes_total", 0)
    snapshot["compress_raw_bytes_total"] = raw
    snapshot["compress_stored_bytes_total"] = stored
    snapshot["compress_ratio"] = round(raw / stored, 3) if stored > 0 else 1.0
    snapshot["compress_lib"] = _rt._PN95_COMPRESS_LIB or "uninit"
    # B1 — async stream stats
    snapshot["async_stream_enabled"] = _pn95_async_enabled()
    snapshot["async_demote_count"] = _rt._PN95_STATS.get("async_demote_count", 0)
    snapshot["async_promote_count"] = _rt._PN95_STATS.get("async_promote_count", 0)
    # B2 — batched demote ops (each batch processes N layers with 1 sync)
    snapshot["async_batch_demote_count"] = _rt._PN95_STATS.get(
        "async_batch_demote_count", 0
    )
    # B3 — batched promote ops (each batch processes N layers with 1 wait_stream)
    snapshot["async_batch_promote_count"] = _rt._PN95_STATS.get(
        "async_batch_promote_count", 0
    )
    # OBS1 — hit rate calculation for operator monitoring
    lookups_total = _rt._PN95_STATS.get("prefix_lookups_total", 0)
    cold_misses = _rt._PN95_STATS.get("prefix_lookups_cold_miss", 0)
    snapshot["prefix_lookups_total"] = lookups_total
    snapshot["prefix_lookups_cold_miss"] = cold_misses
    snapshot["prefix_hit_rate"] = (
        round((lookups_total - cold_misses) / lookups_total, 3)
        if lookups_total > 0 else 0.0
    )
    # Prefetch API stats — visibility into batched warm-up activity.
    # If prefetch_calls > 0, L2→L1 hits/misses indicate whether the
    # caller is correctly predicting near-future block accesses.
    for k, v in _rt._PN95_PREFETCH_STATS.items():
        snapshot[k] = v

    # Layer-aware demote priority — top-5 hottest layer access counts.
    # Useful for diagnosing whether some attention layers monopolize the
    # promote path (which justifies more aggressive cold-layer demote)
    # or whether access is uniform (then ordering is a no-op).
    snapshot["layer_aware_demote_enabled"] = _pn95_layer_aware_enabled()
    if _rt._PN95_LAYER_ACCESS_COUNTS:
        top_hot = sorted(
            _rt._PN95_LAYER_ACCESS_COUNTS.items(), key=lambda kv: -kv[1],
        )[:5]
        snapshot["layer_access_top5_hot"] = {k: v for k, v in top_hot}
        snapshot["layer_access_distinct"] = len(_rt._PN95_LAYER_ACCESS_COUNTS)
        snapshot["layer_access_total_observations"] = sum(
            _rt._PN95_LAYER_ACCESS_COUNTS.values()
        )
    # L1 pinned pool stats — surfaces hit-rate of the fast PCIe DMA path
    # vs the pageable L2 fallback. Operator can spot "pool too small" via
    # high l1_full_skips, or "no demote pressure" via zero l1_demote_writes.
    snapshot["l1_demote_writes"] = _rt._PN95_STATS.get("l1_demote_writes", 0)
    snapshot["l1_promote_hits"] = _rt._PN95_STATS.get("l1_promote_hits", 0)
    try:
        pool = _rt._pn95_l1_pool()
        if pool is not None:
            pool_stats = pool.stats()
            snapshot["l1_pool_enabled"] = True
            snapshot["l1_slots_capacity"] = pool_stats.get("slots_capacity", 0)
            snapshot["l1_slot_size_bytes"] = pool_stats.get("slot_size_bytes", 0)
            snapshot["l1_slots_used"] = pool_stats.get("slots_used", 0)
            snapshot["l1_bytes_used"] = pool_stats.get("bytes_used", 0)
            snapshot["l1_full_skips"] = pool_stats.get("l1_full_skips", 0)
            snapshot["l1_evictions"] = pool_stats.get("l1_evictions", 0)
        else:
            snapshot["l1_pool_enabled"] = False
    except Exception:
        snapshot["l1_pool_enabled"] = False
    return snapshot


def _pn95_dump_stats_if_due() -> None:
    """OBS1 — periodic stats dump to JSON file for operator visibility.

    Called from scheduler_tick. Throttled by tick counter and env-gated.
    Atomic write (tmp + rename) so operator can `cat` safely.

    Env vars:
      GENESIS_PN95_STATS_FILE=/tmp/pn95_stats.json (default; empty disables)
      GENESIS_PN95_STATS_INTERVAL=100 (dump every N ticks)
    """
    try:
        path = os.environ.get("GENESIS_PN95_STATS_FILE", "/tmp/pn95_stats.json")
        if not path:
            return
        try:
            interval = int(os.environ.get("GENESIS_PN95_STATS_INTERVAL", "50"))
        except (ValueError, TypeError):
            interval = 50
        from vllm.sndr_core.cache import _pn95_runtime as _rt
        if interval <= 0 or _rt._TICK_COUNTER % interval != 0:
            return
        import json
        snapshot = get_pn95_stats()
        snapshot["timestamp"] = int(__import__("time").time())
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(snapshot, f, indent=2, default=str)
        os.replace(tmp_path, path)
    except Exception:
        pass  # fail-silent — observability never breaks production


def _pn95_record_lookup(block_hash: Any) -> int:
    """Bump hit counter for a block_hash on every promote query. Returns
    post-bump count. Bounded by _PN95_HIT_TRACKER_MAX (LRU eviction)."""
    from vllm.sndr_core.cache import _pn95_runtime as _rt
    n = _rt._PN95_HIT_COUNTS.get(block_hash, 0) + 1
    if n == 1 and len(_rt._PN95_HIT_COUNTS) >= _rt._PN95_HIT_TRACKER_MAX:
        # FIFO drop — keep tracker bounded, lose oldest counter.
        try:
            oldest = next(iter(_rt._PN95_HIT_COUNTS))
            _rt._PN95_HIT_COUNTS.pop(oldest, None)
        except StopIteration:
            pass
    _rt._PN95_HIT_COUNTS[block_hash] = n
    return n
