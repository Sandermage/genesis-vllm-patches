# SPDX-License-Identifier: Apache-2.0
"""PN95 demote policy — pure decision helpers (what / when / order).

Four pure decision helpers, no transfer execution and no prefix-store
mutation. They answer:

  * ``_pn95_should_demote(block_hash)``  — gating predicate: has the
    block reached the GENESIS_PN95_STORE_THRESHOLD lookup count yet?
  * ``_pn95_record_layer_promote(layer)`` — bookkeeping write: bump
    the per-layer access counter (LRU-bounded by halving).
  * ``_pn95_sort_layers_cold_first(layers)`` — order eligible
    (layer_name, tensor_view) tuples cold-first by access count.
  * ``_select_cold_blocks_via_bpool_lru(target_count)`` — walk vllm's
    free_block_queue to pick demote candidates; respects hot-ring
    + already-in-prefix-store + spec-decode-hot exclusions.

Execution functions that ACT on those decisions — ``pn95_demote_batch``,
``_proactive_demote_cold``, ``worker_side_proactive_demote`` — stay
in ``_pn95_runtime`` because they call ``demote_on_evict`` (a
prefix-store mutation that lives there); they'll move with the
prefix-store split in a later slice.

M.4.2.D scope: function extraction only. State singletons stay in
``_pn95_runtime``:

  _PN95_HIT_COUNTS                   — read by _pn95_should_demote
  _PN95_LAYER_ACCESS_COUNTS          — read+rebind by _pn95_record_layer_promote
                                        and read by _pn95_sort_layers_cold_first
  _PN95_LAYER_ACCESS_RESET_THRESHOLD — read-only constant
  _PN95_BLOCK_POOL_REFS              — read by _select_cold_blocks_via_bpool_lru
  _PN95_PREFIX_STORE                 — read by _select_cold_blocks_via_bpool_lru
  _TM                                — read by _select_cold_blocks_via_bpool_lru

Inventory at extraction time showed zero test rebinds and zero direct
test references for any of these four functions OR for the
``_PN95_LAYER_ACCESS_*`` state — so the cross-module ``_rt.X`` access
pattern (now standard for the M.4.2 slices) is safe.

The ``global _PN95_LAYER_ACCESS_COUNTS`` rebind inside
``_pn95_record_layer_promote`` (the «halve all counters» path on
overflow) is replicated via explicit attribute mutation
``_rt._PN95_LAYER_ACCESS_COUNTS = {...}`` which rebinds the
``_pn95_runtime`` module attribute — the same slot the original
``global`` declaration mutated.

The legacy module re-exports all four functions so internal callers
(``pn95_demote_batch``, ``promote_on_miss``, ``demote_on_evict``,
``_proactive_demote_cold``) continue resolving through the shim
without edit; no text-anchor regen.
"""
from __future__ import annotations

from typing import Any

from .gates import _pn95_layer_aware_enabled, _pn95_store_threshold


def _pn95_should_demote(block_hash: Any) -> bool:
    """Apply store_threshold gate: skip demote if block hasn't reached
    threshold lookups yet. Returns True when demote should proceed."""
    from sndr.cache import _pn95_runtime as _rt
    thr = _pn95_store_threshold()
    if thr <= 1:
        return True  # default: every block demotes
    return _rt._PN95_HIT_COUNTS.get(block_hash, 0) >= thr


def _pn95_record_layer_promote(layer_name: str) -> None:
    """Bump access count for a layer on promote read. Cheap dict op."""
    from sndr.cache import _pn95_runtime as _rt
    n = _rt._PN95_LAYER_ACCESS_COUNTS.get(layer_name, 0) + 1
    if n > _rt._PN95_LAYER_ACCESS_RESET_THRESHOLD:
        # Halve all counters to preserve relative ordering without overflow.
        _rt._PN95_LAYER_ACCESS_COUNTS = {
            k: v // 2 for k, v in _rt._PN95_LAYER_ACCESS_COUNTS.items()
        }
        n = _rt._PN95_LAYER_ACCESS_COUNTS.get(layer_name, 0) + 1
    _rt._PN95_LAYER_ACCESS_COUNTS[layer_name] = n


def _pn95_sort_layers_cold_first(eligible_layers: list) -> list:
    """Sort (layer_name, tensor_view) tuples by ascending access count.

    Layers never observed in promote stay at the front (cold by default).
    Stable sort preserves the original block-pool ordering as the tiebreaker
    so behavior is deterministic when no promote history exists.

    No-op if GENESIS_ENABLE_PN95_LAYER_AWARE_DEMOTE != 1.
    """
    from sndr.cache import _pn95_runtime as _rt
    if not _pn95_layer_aware_enabled() or not _rt._PN95_LAYER_ACCESS_COUNTS:
        return eligible_layers
    return sorted(
        eligible_layers,
        key=lambda lv: _rt._PN95_LAYER_ACCESS_COUNTS.get(lv[0], 0),
    )


def _select_cold_blocks_via_bpool_lru(target_count: int) -> list:
    """Path C v1.0 Phase 4.1 — smart cold-block selection using vllm's
    own LRU (free_block_queue) instead of dummy block_idx=0 heuristic.

    Walks free_block_queue of registered BlockPools — these blocks are
    ALREADY in eviction order (head = most-likely-to-be-evicted-next).
    For each cached block (block_hash != None) we capture its ID + hash
    as a demote candidate.

    Returns list of (block_pool, block_id, block_hash) tuples.

    Skips:
    - Non-cached blocks (block_hash is None) — nothing to preserve
    - Null blocks (block.is_null) — Mamba alignment placeholders
    - Already-pre-demoted entries (in our prefix store)
    - Hot ring (last N admits — typically spec-decode targets)
    """
    from sndr.cache import _pn95_runtime as _rt
    candidates = []
    if not _rt._PN95_BLOCK_POOL_REFS:
        return candidates

    # Hot ring: last N admits never demote (typically spec-decode K+1
    # targets where the model just placed K speculative tokens). Reading
    # the tail of _admit_order on TM gives us the freshest activity.
    hot_keys = set()
    if _rt._TM is not None:
        ring_size = getattr(_rt._TM, "spec_decode_hot_ring", 0) or 0
        if ring_size > 0:
            try:
                hot_keys = set(_rt._TM._admit_order[-ring_size:])
            except (AttributeError, TypeError):
                hot_keys = set()

    for pool in _rt._PN95_BLOCK_POOL_REFS:
        try:
            queue = getattr(pool, "free_block_queue", None)
            if queue is None:
                continue
            # Iterate doubly-linked list head → tail (LRU order).
            # vllm's FreeKVCacheBlockQueue exposes .head / .next pointers.
            head = getattr(queue, "fake_free_list_head", None) or \
                   getattr(queue, "_fake_head", None)
            cur = getattr(head, "next_free_block", None) if head else None
            walked = 0
            max_walk = target_count * 8  # bound the scan
            while cur is not None and walked < max_walk:
                walked += 1
                if getattr(cur, "is_null", False):
                    cur = getattr(cur, "next_free_block", None)
                    continue
                blk_hash = getattr(cur, "block_hash", None)
                if blk_hash is None:
                    cur = getattr(cur, "next_free_block", None)
                    continue
                # Skip if already in CPU prefix store (don't re-copy)
                if blk_hash in _rt._PN95_PREFIX_STORE:
                    cur = getattr(cur, "next_free_block", None)
                    continue
                # Skip hot ring members
                blk_id = getattr(cur, "block_id", -1)
                if (id(pool), blk_id) in hot_keys:
                    cur = getattr(cur, "next_free_block", None)
                    continue
                candidates.append((pool, blk_id, blk_hash))
                if len(candidates) >= target_count:
                    return candidates
                cur = getattr(cur, "next_free_block", None)
        except Exception:
            continue
    return candidates
