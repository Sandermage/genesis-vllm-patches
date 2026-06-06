# SPDX-License-Identifier: Apache-2.0
"""PN95 L2 prefix store + demote/promote + execution helpers.

The CPU-side L2 prefix cache is the heart of PN95's tier-aware path:
when vllm evicts a cached GPU block (``_maybe_evict_cached_block``),
``demote_on_evict`` captures the bytes to ``_PN95_PREFIX_STORE``;
when vllm later reads a cache-miss hash (``get_cached_block``),
``promote_on_miss`` restores the bytes from the L2 store (or the
disk tier) into a freshly-allocated GPU block.

Ten helpers split across four concerns:

  Prefix-store accounting:
    _prefix_store_max_bytes         — env-cached GiB cap
    _prefix_store_evict_until_fit   — LRU eviction (with optional
                                       disk-tier spillover)
    _pn95_l1_pool                   — pinned-pool singleton accessor

  Block-pool registration:
    register_block_pool             — record a BlockPool ref for
                                       worker-side proactive demote

  Demote / promote — text-anchor entry points:
    demote_on_evict                 — GPU→L1+L2 capture on eviction
    promote_on_miss                 — L1/L2/disk→GPU restore on miss

  Execution / orchestration (parked here in M.4.2.H — they all call
  demote_on_evict / promote_on_miss):
    pn95_demote_batch               — super-block batched demote
                                       (calls demote_on_evict)
    _proactive_demote_cold          — scheduler-tick proactive path
                                       (calls demote_on_evict via
                                        _select_cold_blocks_via_bpool_lru)
    worker_side_proactive_demote    — worker-process driver
                                       (calls demote_on_evict +
                                        register_block_pool)
    pn95_materialize_virtual_block  — Phase 5 Anchor #12 donor swap
                                       (calls demote_on_evict; reads
                                        _PN95_BLOCK_METADATA virtual-
                                        block side-table)

M.4.2.H scope: function extraction only. State singletons stay in
``_pn95_runtime``:

  _PN95_PREFIX_STORE              — OrderedDict (LRU); test sites
                                     don't rebind, but the legacy
                                     module is the canonical owner
                                     because the disk_tier test (×1
                                     site) does direct
                                     ``rt._PN95_PREFIX_STORE_BYTES_USED = 128``
  _PN95_PREFIX_STORE_BYTES_USED   — int; REBOUND inside
                                     ``_prefix_store_evict_until_fit``
                                     and ``demote_on_evict`` /
                                     ``promote_on_miss``; rebind via
                                     ``_rt.X -= …`` attribute mutation
  _PN95_PREFIX_STORE_MAX_BYTES_CACHED — cached envvar; REBOUND inside
                                     ``_prefix_store_max_bytes``
  _PN95_BLOCK_POOL_REFS           — list (.append in register_block_pool)
  _PN95_PREFIX_STORE_LOCK         — threading.Lock (read-only ref)

The original ``global ... = …`` rebind sites are replicated via
explicit attribute mutation on ``_rt`` (lazy-import inside each body)
at the same module-attribute slot the original ``global`` declaration
mutated.

Sibling-patch text-anchor imports (CRITICAL — preserved via
``_pn95_runtime`` re-export shim, no anchor regen):

  pn95_tier_aware_cache.py:181 / :334 / :342
    ``from vllm.sndr_core.cache._pn95_runtime import register_block_pool
       as _g_pn95_regpool`` (×3)
  pn95_tier_aware_cache.py:213
    ``from vllm.sndr_core.cache._pn95_runtime import demote_on_evict
       as _g_pn95_demote_ev``
  pn95_tier_aware_cache.py:437
    ``from vllm.sndr_core.cache._pn95_runtime import promote_on_miss
       as _g_pn95_promote_m``

This file pulls in transfer / compression / demote_policy /
virtual_blocks via gates as needed — no circular import because
those modules already use lazy ``_rt`` access for their own state
references.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from .gates import _enabled, _phase5_virt_enabled

log = logging.getLogger("genesis.pn95")


# ─── Prefix-store accounting ───────────────────────────────────────────


def _prefix_store_max_bytes() -> int:
    """Read GENESIS_PN95_PREFIX_STORE_GIB env var (default 4 GiB)."""
    from vllm.sndr_core.cache import _pn95_runtime as _rt
    if _rt._PN95_PREFIX_STORE_MAX_BYTES_CACHED is None:
        gib = float(os.environ.get("GENESIS_PN95_PREFIX_STORE_GIB", "4"))
        _rt._PN95_PREFIX_STORE_MAX_BYTES_CACHED = int(gib * (1 << 30))
    return _rt._PN95_PREFIX_STORE_MAX_BYTES_CACHED


def _prefix_store_evict_until_fit(needed_bytes: int) -> None:
    """LRU evict from CPU prefix store until needed_bytes fits.

    When the disk tier (`_pn95_disk_tier`) is enabled, the LRU victim
    is spilled to disk before being dropped from RAM so future
    `promote_on_miss` can still recover the bytes. With the disk tier
    disabled (default) the behaviour matches the legacy implementation
    — victims are discarded.
    """
    from vllm.sndr_core.cache import _pn95_runtime as _rt
    max_bytes = _prefix_store_max_bytes()
    try:
        from vllm.sndr_core.cache import _pn95_disk_tier as _disk
    except ImportError:
        _disk = None
    disk_active = _disk is not None and _disk._enabled()
    while _rt._PN95_PREFIX_STORE_BYTES_USED + needed_bytes > max_bytes:
        if not _rt._PN95_PREFIX_STORE:
            return
        key, layer_data = _rt._PN95_PREFIX_STORE.popitem(last=False)
        freed = sum(len(b) for _name, b in layer_data)
        _rt._PN95_PREFIX_STORE_BYTES_USED -= freed
        # Spillover the evicted entry to the disk tier when enabled.
        # Failure is non-fatal — the victim is then discarded as before.
        if disk_active:
            try:
                if _disk.disk_tier_set(key, layer_data):
                    _rt._PN95_STATS.setdefault("ram_to_disk_spills_total", 0)
                    _rt._PN95_STATS["ram_to_disk_spills_total"] += 1
            except Exception:
                pass
        # L1 pinned pool: same entry may also have a slot reserved there.
        # Free the slot so the pool stays in sync with the L2 LRU. Reading
        # is best-effort — pool.evict tolerates absent keys.
        try:
            pool = _pn95_l1_pool()
            if pool is not None:
                pool.evict(key)
        except Exception:
            pass


def _pn95_l1_pool(slot_size_hint: int = 0):
    """Return the singleton pinned pool, or None when disabled / alloc-failed.

    Safe to call from hot paths — returns fast None when feature OFF.
    """
    try:
        from vllm.sndr_core.cache import _pn95_pinned_pool as _ppool
    except ImportError:
        return None
    return _ppool.get_pool(slot_size_hint)


# ─── Block-pool registration ───────────────────────────────────────────


def register_block_pool(block_pool: Any) -> None:
    """Phase 4: register a BlockPool instance so promote_on_miss can
    allocate fresh GPU blocks via block_pool.get_new_blocks(1).

    Multiple pools may register (one per KVCacheManager / kv_cache_group).
    Only attention groups will actually demote/promote — Mamba groups
    don't populate the prefix cache to begin with so their pools see
    no PN95 activity.
    """
    if not _enabled():
        return
    from vllm.sndr_core.cache import _pn95_runtime as _rt
    try:
        if block_pool not in _rt._PN95_BLOCK_POOL_REFS:
            _rt._PN95_BLOCK_POOL_REFS.append(block_pool)
    except Exception:
        pass


# ─── Demote / promote text-anchor entry points ─────────────────────────


def demote_on_evict(block_hash: Any, block_id: int) -> bool:
    """Phase 4: capture GPU block bytes to CPU pinned storage as vllm
    evicts the block from prefix cache. Called from BlockPool's
    _maybe_evict_cached_block before reset_hash().

    block_hash is the BlockHashWithGroupId key vllm uses internally.
    block_id is the GPU physical slot (0..num_gpu_blocks-1).

    Returns True iff bytes successfully captured.

    Safety: at this point the block has ref_cnt=0 and is not in any
    active block_table — no concurrent readers. The cudaMemcpyDeviceToHost
    is safe even if it takes 10-50ms.
    """
    from vllm.sndr_core.cache import _pn95_runtime as _rt
    if not _enabled() or _rt._TM is None:
        return False
    views = getattr(_rt._TM, "_attention_views", None)
    if not views:
        return False
    try:
        # Sprint Q1 B2 — collect all eligible views first, then ONE batched
        # GPU→CPU copy (single stream sync vs N).
        eligible_layers = []
        for layer_name, info in views.items():
            tensor = info.get("tensor")
            num_blocks = int(info.get("num_blocks", 0))
            if tensor is None or block_id < 0 or block_id >= num_blocks:
                continue
            eligible_layers.append((layer_name, tensor[block_id]))

        if not eligible_layers:
            return False

        # Layer-aware demote priority: when enabled, sort eligible_layers
        # ascending by promote-access count so cold layers go to CPU first.
        # No effect on the per-block all-or-nothing semantic (we still copy
        # every layer for this block); the ordering only matters when the
        # downstream pool is byte-budget capped (L1 pinned pool slot limit,
        # or future per-layer demote skip).
        from .demote_policy import _pn95_sort_layers_cold_first
        eligible_layers = _pn95_sort_layers_cold_first(eligible_layers)

        # ONE batched async copy for all N layer views (~16× less sync overhead).
        from .transfer import _pn95_gpu_to_cpu_bytes_batch
        layer_views = [v for _name, v in eligible_layers]
        raw_bytes_list = _pn95_gpu_to_cpu_bytes_batch(layer_views)

        # Sprint Q1 B4 — parallel compression (CPU work parallelizable since
        # zstd/lz4/zlib release GIL during compress). For 17 layers ~3-4× faster.
        from .compression import _pn95_compress_bytes_batch, _pn95_pack_layer_data
        compressed_list = _pn95_compress_bytes_batch(raw_bytes_list)

        # Assemble layer_data
        layer_data = []
        total_bytes = 0
        raw_total_bytes = 0
        for (layer_name, _v), cpu_bytes_raw, cpu_bytes_stored in zip(
            eligible_layers, raw_bytes_list, compressed_list,
        ):
            raw_total_bytes += len(cpu_bytes_raw)
            layer_data.append((layer_name, cpu_bytes_stored))
            total_bytes += len(cpu_bytes_stored)

        if not layer_data:
            return False

        # LRU eviction if at capacity (uses STORED size, not raw)
        _prefix_store_evict_until_fit(total_bytes)

        # Skip if even single entry doesn't fit (avoid memory pathology)
        if total_bytes > _prefix_store_max_bytes():
            return False

        # Insert (or move-to-end if duplicate hash — shouldn't happen
        # since vllm just removed it from cached_block_hash_to_block)
        if block_hash in _rt._PN95_PREFIX_STORE:
            old = _rt._PN95_PREFIX_STORE.pop(block_hash)
            _rt._PN95_PREFIX_STORE_BYTES_USED -= sum(len(b) for _n, b in old)

        # Two-phase commit (upstream PR #40020 prepare_store/complete_store
        # pattern): L1 write first, L2 write second, rollback L1 if L2 fails.
        # Keeps the two tiers from desyncing — a stale L1 slot pointing at
        # bytes whose L2 LRU position was never updated would let promote
        # return inconsistent data on next access.
        l1_acquired = False
        l1_pool = None
        try:
            blob = _pn95_pack_layer_data(layer_data)
            l1_pool = _pn95_l1_pool(slot_size_hint=len(blob))
            if l1_pool is not None and l1_pool.put(block_hash, blob):
                l1_acquired = True
                _rt._PN95_STATS.setdefault("l1_demote_writes", 0)
                _rt._PN95_STATS["l1_demote_writes"] += 1
        except Exception:
            l1_acquired = False  # L1 refused — proceed L2-only

        try:
            with _rt._PN95_PREFIX_STORE_LOCK:
                _rt._PN95_PREFIX_STORE[block_hash] = layer_data
                _rt._PN95_PREFIX_STORE_BYTES_USED += total_bytes
        except Exception:
            # L2 insert failed — rollback L1 to keep tiers consistent.
            if l1_acquired and l1_pool is not None:
                try:
                    l1_pool.evict(block_hash)
                except Exception:
                    pass
            _rt._PN95_STATS["demote_rollback_count"] = (
                _rt._PN95_STATS.get("demote_rollback_count", 0) + 1
            )
            return False

        _rt._PN95_STATS["prefix_demote_count"] = (
            _rt._PN95_STATS.get("prefix_demote_count", 0) + 1
        )
        # A1 stats — compression ratio tracking
        _rt._PN95_STATS["compress_raw_bytes_total"] = (
            _rt._PN95_STATS.get("compress_raw_bytes_total", 0) + raw_total_bytes
        )
        _rt._PN95_STATS["compress_stored_bytes_total"] = (
            _rt._PN95_STATS.get("compress_stored_bytes_total", 0) + total_bytes
        )
        return True
    except Exception:
        return False


def promote_on_miss(block_pool: Any, block_hash_with_group_id: Any) -> Any:
    """Phase 4.2: on get_cached_block cache miss, check our CPU prefix store
    and restore if present.

    CRITICAL FIX (Phase 4.2): byte reinterpret consistency. Previous version
    used `view.numel()` (element count, dtype-aware) when src is uint8 byte
    buffer — wrong for float16/bfloat16 tensors (numel = bytes/2, but src
    has all bytes → buffer overflow OR truncated copy → corrupt KV → tool
    call regression observed at 6/7 vs OFF baseline 7/7).

    Fix: reinterpret BOTH view and src as uint8 byte arrays before copy.
    This is dtype-agnostic and exact for any KV cache layout (TQ k8v4,
    fp8_e5m2, fp16, bf16).

    Returns a KVCacheBlock object (newly allocated, populated from CPU
    bytes, re-inserted into vllm's prefix cache) on hit, or None on miss.

    Safety: get_new_blocks may evict another cached block — that block's
    eviction will trigger our demote_on_evict (recursive but bounded by
    prefix store capacity). cudaMemcpyHostToDevice is sync.
    """
    from vllm.sndr_core.cache import _pn95_runtime as _rt
    if not _enabled() or _rt._TM is None:
        return None
    # OBS1 — track all lookup attempts (for hit_rate calculation)
    _rt._PN95_STATS["prefix_lookups_total"] = (
        _rt._PN95_STATS.get("prefix_lookups_total", 0) + 1
    )
    # store_threshold: bump per-hash hit counter so future demote knows
    # this block has been queried (review finding #1 — without this call
    # _pn95_should_demote always returns False and the gate blocks ALL
    # demotes when GENESIS_PN95_STORE_THRESHOLD>=2).
    from .metrics import _pn95_record_lookup
    _pn95_record_lookup(block_hash_with_group_id)

    # L1 pinned pool first — fastest path. If hit, unpack and use directly;
    # bypass the L2 OrderedDict read entirely. L2 still holds the entry for
    # disk-spillover bookkeeping, but the bytes we hand to the GPU come
    # from pinned memory (3-5x faster PCIe DMA).
    from .compression import _pn95_unpack_layer_data, _pn95_decompress_bytes
    from .demote_policy import _pn95_record_layer_promote
    from .transfer import _pn95_cpu_to_gpu_copy_batch
    layer_data = None
    try:
        pool = _pn95_l1_pool()
        if pool is not None and pool.has(block_hash_with_group_id):
            blob = pool.get_bytes(block_hash_with_group_id)
            layer_data = _pn95_unpack_layer_data(blob)
            if layer_data is not None:
                _rt._PN95_STATS.setdefault("l1_promote_hits", 0)
                _rt._PN95_STATS["l1_promote_hits"] += 1
    except Exception:
        layer_data = None

    if layer_data is None:
        layer_data = _rt._PN95_PREFIX_STORE.get(block_hash_with_group_id)
    if layer_data is None:
        # CPU prefix store miss — try the disk tier before giving up.
        # On disk hit, re-insert into the in-RAM store (LRU at the
        # warm end) so subsequent lookups stay fast and we don't pay
        # the unpickle cost twice.
        try:
            from vllm.sndr_core.cache import _pn95_disk_tier as _disk
        except ImportError:
            _disk = None
        if _disk is not None and _disk._enabled():
            disk_data = _disk.disk_tier_get(block_hash_with_group_id)
            if disk_data is not None:
                layer_data = disk_data
                # Insert back into CPU prefix store; evict to fit if
                # needed (which may itself spill an older entry to
                # disk per _prefix_store_evict_until_fit policy).
                total_bytes = sum(len(b) for _n, b in layer_data)
                _prefix_store_evict_until_fit(total_bytes)
                if total_bytes <= _prefix_store_max_bytes():
                    _rt._PN95_PREFIX_STORE[block_hash_with_group_id] = layer_data
                    _rt._PN95_PREFIX_STORE_BYTES_USED += total_bytes
                _rt._PN95_STATS.setdefault("disk_to_ram_promotes_total", 0)
                _rt._PN95_STATS["disk_to_ram_promotes_total"] += 1
    if layer_data is None:
        # OBS1 — cold miss: vllm asked and we had no data either
        _rt._PN95_STATS["prefix_lookups_cold_miss"] = (
            _rt._PN95_STATS.get("prefix_lookups_cold_miss", 0) + 1
        )
        return None
    try:
        new_blocks = block_pool.get_new_blocks(1)
        if not new_blocks:
            return None
        new_block = new_blocks[0]
        new_block_id = new_block.block_id

        views = getattr(_rt._TM, "_attention_views", None) or {}
        # Sprint Q1 B3 — collect all eligible (view, bytes) pairs, then ONE
        # batched async CPU→GPU copy (single wait_stream vs N).
        #
        # NOTE: Sequential decompress kept (measured B5 parallel slower on 17
        # layers — zstd decompress ~80μs total is too fast for ThreadPool overhead).
        # `_pn95_decompress_bytes_batch` available for future bulk warmup
        # scenarios (many small entries) where parallelism does pay off.
        eligible_views = []
        eligible_bytes = []
        for layer_name, cpu_bytes_stored in layer_data:
            cpu_bytes = _pn95_decompress_bytes(cpu_bytes_stored)
            info = views.get(layer_name)
            if info is None:
                continue
            tensor = info.get("tensor")
            num_blocks = int(info.get("num_blocks", 0))
            bytes_per_block = int(info.get("bytes_per_block", 0))
            if tensor is None or new_block_id < 0 or new_block_id >= num_blocks:
                continue
            if bytes_per_block > 0 and len(cpu_bytes) != bytes_per_block:
                continue
            try:
                view = tensor[new_block_id]
                eligible_views.append(view)
                eligible_bytes.append(cpu_bytes)
                # Layer-aware demote bookkeeping: record that this layer
                # was actually restored to GPU from PN95 — feeds the
                # cold/hot heatmap consumed by demote_on_evict ordering.
                _pn95_record_layer_promote(layer_name)
            except Exception:
                continue

        # Single batched copy for all eligible layers.
        n_layers_restored = _pn95_cpu_to_gpu_copy_batch(
            eligible_views, eligible_bytes
        )

        # Only mark as cached if we actually restored at least one layer.
        # Otherwise we'd be telling vllm "this hash is cached" while having
        # zero data — would corrupt subsequent attention reads.
        if n_layers_restored == 0:
            # Roll back: free the block we just allocated
            try:
                block_pool.free_blocks([new_block])
            except Exception:
                pass
            return None

        new_block.block_hash = block_hash_with_group_id
        try:
            block_pool.cached_block_hash_to_block.insert(
                block_hash_with_group_id, new_block
            )
        except Exception:
            pass

        old = _rt._PN95_PREFIX_STORE.pop(block_hash_with_group_id, None)
        if old is not None:
            _rt._PN95_PREFIX_STORE_BYTES_USED -= sum(len(b) for _n, b in old)

        _rt._PN95_STATS["prefix_promote_hits"] = (
            _rt._PN95_STATS.get("prefix_promote_hits", 0) + 1
        )
        _rt._PN95_STATS["blocks_promoted_total"] += 1
        return new_block
    except Exception:
        return None


# ─── Execution / orchestration (parked here in M.4.2.H) ────────────────


def pn95_demote_batch(block_id_hash_pairs: list) -> int:
    """Group-demote helper: batch `block_size_factor` (block_id, block_hash)
    tuples into a single super-block demote call.

    Caller (worker_side_proactive_demote, demote_on_evict-style hot path)
    passes a list of pairs in admit-LRU order. Implementation slices into
    groups of `factor` size and dispatches each via the per-block
    `demote_on_evict` (which already does atomic two-phase commit).

    Returns count of blocks successfully demoted.

    Note: this is a helper, not a replacement for demote_on_evict. The
    per-block path remains for vllm anchor-#9 entry points; this gives
    the proactive scheduler a way to amortize when it knows N blocks
    will be evicted as a batch.
    """
    from vllm.sndr_core.cache import _pn95_runtime as _rt
    from .gates import _pn95_block_size_factor
    from .demote_policy import _pn95_should_demote
    if not block_id_hash_pairs:
        return 0
    factor = _pn95_block_size_factor()
    n_demoted = 0
    # Slice into super-block groups.
    for i in range(0, len(block_id_hash_pairs), factor):
        group = block_id_hash_pairs[i : i + factor]
        for block_id, block_hash in group:
            # store_threshold gate — skip blocks below admission threshold.
            if not _pn95_should_demote(block_hash):
                _rt._PN95_STATS["store_threshold_skips"] = (
                    _rt._PN95_STATS.get("store_threshold_skips", 0) + 1
                )
                continue
            try:
                if demote_on_evict(block_hash, block_id):
                    n_demoted += 1
            except Exception:
                pass
    if factor > 1:
        _rt._PN95_STATS["super_block_demote_batches"] = (
            _rt._PN95_STATS.get("super_block_demote_batches", 0) + 1
        )
        _rt._PN95_STATS["block_size_factor"] = factor
    return n_demoted


def _proactive_demote_cold(target_count: int) -> int:
    """Path C v1.0 Phase 4.1 — opportunistic demote of cold cached blocks.

    Captures block bytes to CPU prefix store BEFORE vllm evicts them.
    When vllm later evicts via _maybe_evict_cached_block, our demote_on_evict
    anchor will see the entry already exists and skip (no double-copy).
    When request hits the same hash later, promote_on_miss restores it.

    Net effect: GPU pool throughput improves (vllm's eviction is a no-op
    for our purposes — bytes already saved). And on multi-turn workloads
    with re-occurring prefixes, we get sustained cache hits.

    Returns number of blocks captured.
    """
    from .demote_policy import _select_cold_blocks_via_bpool_lru
    candidates = _select_cold_blocks_via_bpool_lru(target_count)
    if not candidates:
        return 0
    n_captured = 0
    for _pool, blk_id, blk_hash in candidates:
        if demote_on_evict(blk_hash, blk_id):
            n_captured += 1
    return n_captured


def worker_side_proactive_demote(
    block_pool: Any,
    target_count: int = 8,
) -> int:
    """Worker-process entry point for proactive cold-block demote.

    The scheduler-tick path (`scheduler_tick`) runs in the EngineCore
    process. In a multiproc vLLM deploy that process never holds a
    BlockPool reference (those live in Worker processes), so the
    `_proactive_demote_cold` branch silently no-ops.

    This helper closes that gap. Callers from worker context — a
    BlockPool hot path, a worker rpc, or a manual operator probe — pass
    the locally-live `block_pool` so the LRU walk runs against the
    real free-block queue that owns the GPU bytes.

    Throttling is deliberately the caller's job: this function performs
    the requested work synchronously and returns the count of blocks
    captured. Callers in a hot path should rate-limit (e.g. only invoke
    when free queue length drops below a threshold).

    Returns the number of blocks whose bytes were captured to the CPU
    prefix store. Zero is returned when:
      - PN95 disabled
      - TierManager not installed in this process
      - block_pool has no free_block_queue / no cached blocks
      - all candidates already in CPU prefix store
    """
    from vllm.sndr_core.cache import _pn95_runtime as _rt
    if not _enabled() or _rt._TM is None:
        return 0
    if block_pool is None:
        return 0
    # Register the pool if the worker-side anchor (SITE6) hasn't run
    # yet for this pool. Idempotent — `register_block_pool` dedups.
    register_block_pool(block_pool)

    # Replay the same LRU walk `_select_cold_blocks_via_bpool_lru` does
    # but bounded to this single pool — caller knows which pool is
    # under pressure, so we don't scan unrelated pools.
    candidates: list = []
    hot_keys: set = set()
    try:
        ring_size = getattr(_rt._TM, "spec_decode_hot_ring", 0) or 0
        if ring_size > 0:
            hot_keys = set(_rt._TM._admit_order[-ring_size:])
    except (AttributeError, TypeError):
        hot_keys = set()
    try:
        queue = getattr(block_pool, "free_block_queue", None)
        if queue is None:
            return 0
        head = (
            getattr(queue, "fake_free_list_head", None)
            or getattr(queue, "_fake_head", None)
        )
        cur = getattr(head, "next_free_block", None) if head else None
        walked = 0
        max_walk = max(target_count * 8, 16)
        while cur is not None and walked < max_walk:
            walked += 1
            if getattr(cur, "is_null", False):
                cur = getattr(cur, "next_free_block", None)
                continue
            blk_hash = getattr(cur, "block_hash", None)
            if blk_hash is None:
                cur = getattr(cur, "next_free_block", None)
                continue
            if blk_hash in _rt._PN95_PREFIX_STORE:
                cur = getattr(cur, "next_free_block", None)
                continue
            blk_id = getattr(cur, "block_id", -1)
            if (id(block_pool), blk_id) in hot_keys:
                cur = getattr(cur, "next_free_block", None)
                continue
            candidates.append((blk_id, blk_hash))
            if len(candidates) >= target_count:
                break
            cur = getattr(cur, "next_free_block", None)
    except Exception:
        return 0

    if not candidates:
        return 0

    captured = 0
    for blk_id, blk_hash in candidates:
        try:
            if demote_on_evict(blk_hash, blk_id):
                captured += 1
        except Exception:
            continue

    if captured:
        _rt._PN95_STATS["blocks_demoted_total"] += captured
        _rt._PN95_STATS["last_demote_count"] = captured
        _rt._PN95_STATS.setdefault("worker_proactive_calls", 0)
        _rt._PN95_STATS["worker_proactive_calls"] += 1
        _rt._PN95_STATS.setdefault("worker_proactive_captured", 0)
        _rt._PN95_STATS["worker_proactive_captured"] += captured
    return captured


def pn95_materialize_virtual_block(
    pool: Any, virt_block: Any, exclude: Optional[list] = None,
) -> Optional[int]:
    """Path C v1.0 Phase 5 Anchor #12 — materialize a virtual block via
    swap-based virtualization.

    Strategy:
      1. Find a 'donor' block in pool.blocks: cached (block_hash != None),
         physical_resident=True, ref_cnt=0 (in free queue), not in exclude
         list (other blocks just popped by same get_new_blocks call).
      2. Demote donor's bytes to CPU prefix store via Phase 4 mechanism
         (so future cache hits can promote_on_miss restore).
      3. Adopt donor's physical_block_id for virt_block.
      4. Donor becomes virtual (physical_resident=False, physical_block_id=None).
      5. virt_block becomes physical (gets donor's block_id).

    Returns the new physical block_id assigned to virt_block, or None
    if no donor available (truly exhausted GPU).

    Caller (Anchor #12 in get_new_blocks) is responsible for mutating
    `virt_block.block_id` to this returned value. We don't mutate here
    to keep this helper testable.

    Race safety: relies on the ref_cnt=0 invariant — donors are not
    active in request block_tables. vllm v1 GIL serializes access.
    """
    from vllm.sndr_core.cache import _pn95_runtime as _rt
    if not _enabled() or not _phase5_virt_enabled():
        return None
    if _rt._TM is None:
        return None
    try:
        pool_id = id(pool)
        exclude_ids = set()
        if exclude is not None:
            for b in exclude:
                exclude_ids.add(id(b))

        # Find donor: physical_resident, ref_cnt=0, has block_hash (cached),
        # NOT in exclude list, NOT null_block.
        # Walk free_queue head→tail (LRU order = coldest first).
        free_q = getattr(pool, "free_block_queue", None)
        if free_q is None:
            return None

        head = getattr(free_q, "fake_free_list_head", None)
        cur = getattr(head, "next_free_block", None) if head else None
        donor = None
        max_walk = 256  # cap iteration cost
        walked = 0
        while cur is not None and walked < max_walk:
            walked += 1
            if id(cur) in exclude_ids:
                cur = getattr(cur, "next_free_block", None)
                continue
            if getattr(cur, "is_null", False):
                cur = getattr(cur, "next_free_block", None)
                continue
            # Skip non-cached (no bytes worth saving — but still valid donor)
            # Actually any physical_resident block is valid donor; cached
            # ones are PREFERRED because we can save their bytes for restore
            cur_id = getattr(cur, "block_id", -1)
            cur_meta = _rt._PN95_BLOCK_METADATA.get((pool_id, cur_id))
            if cur_meta is None or not cur_meta.get("physical_resident", False):
                cur = getattr(cur, "next_free_block", None)
                continue
            donor = cur
            break

        if donor is None:
            return None

        # Capture donor's bytes to CPU prefix store (only if cached)
        donor_hash = getattr(donor, "block_hash", None)
        donor_phys_id = donor.block_id  # physical_resident → block_id == physical_block_id
        if donor_hash is not None:
            # Best-effort capture — failure is not critical (just lose cache hit later)
            try:
                demote_on_evict(donor_hash, donor_phys_id)
            except Exception:
                pass

        # CRITICAL: side-table is keyed by block_id, but swap mutates ids.
        # After swap:
        #   - virt_block has new id = donor_phys_id, IS physical
        #   - donor has new id = virt_id, IS virtual
        # Metadata MUST be stored at NEW ids matching the NEW status.
        virt_id = virt_block.block_id

        # Build NEW metadata to attach AT THE NEW IDS:
        new_physical_meta = {
            "physical_resident": True,
            "physical_block_id": donor_phys_id,
            "last_access_tick": 0,
        }
        new_virtual_meta = {
            "physical_resident": False,
            "physical_block_id": None,
            "last_access_tick": 0,
        }
        # virt_block (which will get new id donor_phys_id) is now physical
        _rt._PN95_BLOCK_METADATA[(pool_id, donor_phys_id)] = new_physical_meta
        # donor (which will get new id virt_id) is now virtual
        _rt._PN95_BLOCK_METADATA[(pool_id, virt_id)] = new_virtual_meta

        # Donor block stays in free_queue. Mutate its block_id to virt_id
        # so future free_queue traversal sees the correct (mutated) id.
        # Caller mutates virt_block.block_id to donor_phys_id separately.
        try:
            donor.block_id = virt_id
            # Clear donor's hash — its bytes are now on CPU (or never were cached)
            if hasattr(donor, "_block_hash"):
                donor._block_hash = None
        except Exception:
            pass

        _rt._PN95_STATS["blocks_materialized_total"] = (
            _rt._PN95_STATS.get("blocks_materialized_total", 0) + 1
        )
        return donor_phys_id
    except Exception as e:
        log.warning("[PN95 v1.0 Phase 5] materialize_virtual_block failed: %s", e)
        return None
