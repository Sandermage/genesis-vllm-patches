# SPDX-License-Identifier: Apache-2.0
"""PN95 v7.73.x runtime hooks — notify_admit / notify_touch.

Module-level singleton TierManager. Both hooks are designed to be
**fail-silent**: if GENESIS_ENABLE_PN95_TIER_AWARE_CACHE is unset OR
the singleton hasn't been initialized OR any error occurs inside the
notification, the call must return cleanly so the surrounding vLLM
code path is never destabilized.

Public entry points:
  - `init_from_config(cfg)` — install the singleton from a ModelConfig.
    Idempotent. Called once at engine startup by the dispatcher hook.
  - `notify_admit(request, prev_n_cached, new_n_cached, group_id)` —
    called from the cache_blocks() text-patch site after vLLM's
    cache_full_blocks() returns.
  - `notify_touch(block_hash, group_ids, cached_blocks)` — called
    from the get_cached_block() text-patch site before return.
  - `tier_manager()` — accessor for live observability / tests.
  - `reset_for_tests()` — drop the singleton.

Vision-token tagging + Mamba exclusion plumbing wires through this
module so the text-patches stay tiny.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

log = logging.getLogger("genesis.pn95")

_LOCK = threading.Lock()
_TM: Optional[Any] = None  # vllm.sndr_core.cache.tier_manager.TierManager
_LAST_GROUP_IDS_BY_HASH: dict = {}  # cleared on reset_for_tests


def _enabled() -> bool:
    """True iff GENESIS_ENABLE_PN95_TIER_AWARE_CACHE=1."""
    return os.environ.get(
        "GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", "0"
    ).strip().lower() in ("1", "true", "yes", "on")


def _phase5_virt_enabled() -> bool:
    """Path C v1.0 Phase 5 master gate — KV pool virtualization.

    Default OFF for safety. When enabled:
      - Anchor #9 inflates available_memory in pre-flight check by CPU tier capacity
      - Anchor #10 caps physical KVCacheTensor allocation to GPU-only memory
      - Anchor #11+ (later sessions) wire up logical/physical block split

    Independent from GENESIS_ENABLE_PN95_TIER_AWARE_CACHE — Phase 5 virt
    requires both: PN95 base infrastructure AND VIRT opt-in.
    """
    if not _enabled():
        return False  # virt requires PN95 base
    return os.environ.get(
        "GENESIS_PN95_VIRT_ENABLE", "0"
    ).strip().lower() in ("1", "true", "yes", "on")


def pn95_extra_logical_memory_bytes() -> int:
    """Phase 5 Anchor #9 helper — bytes of CPU tier capacity to add to
    vllm's pre-flight available_memory check.

    Allows max_model_len > GPU hardware ceiling. Caller (Anchor #9)
    inflates available_memory by this amount, vllm then computes
    num_blocks based on inflated value, BUT Anchor #10 caps the
    physical GPU allocation separately to prevent CUDA OOM.

    Returns 0 if PN95 disabled OR Phase 5 virt disabled OR no TM
    installed yet (deferred boot-time call before TM init).

    Safety: env-gated (default OFF). Even if accidentally called when
    TM not ready, returns 0 → vllm sees unmodified available_memory →
    behavior identical to PN95 OFF.
    """
    if not _phase5_virt_enabled():
        return 0
    tm = _TM
    if tm is None:
        return 0
    extra = 0
    try:
        for tier_idx, tier in enumerate(tm.tiers):
            if tier_idx == 0:
                continue  # tier 0 is GPU, already counted by vllm
            device = getattr(tier, "device", "")
            if device == "cpu":
                cap_gib = float(getattr(tier, "capacity_gib", 0.0))
                extra += int(cap_gib * (1 << 30))
    except Exception:
        return 0
    return extra


_PN95_CUDA_STREAM: Optional[Any] = None
# Phase 5 Session 2 — side-table for block metadata.
# KVCacheBlock is @dataclass(slots=True) → cannot add fields directly.
# Side-table keyed by (id(pool), block_id) → {"physical_resident": bool,
# "physical_block_id": Optional[int], "last_access_tick": int}.
_PN95_BLOCK_METADATA: dict = {}
_PN95_POOL_LOGICAL_NUM_BLOCKS: dict = {}  # id(pool) → logical num_blocks


def _pn95_stream() -> Optional[Any]:
    """Path C v1.0 Phase 5 — lazy-init separate CUDA stream for PN95 ops.

    Used by Sessions 3-4 to overlap demote/promote PCIe transfers with
    attention compute on the default stream. Cuts visible promote latency
    from ~25 μs/block to ~0 (overlapped). Reduces TPS impact of demote
    cycles to negligible.

    Returns None when torch unavailable (no-op) — caller falls back to
    default stream (synchronous) which preserves current behavior.

    Zero overhead when not used (lazy init).
    """
    global _PN95_CUDA_STREAM
    if _PN95_CUDA_STREAM is None:
        try:
            import torch
            if torch.cuda.is_available():
                _PN95_CUDA_STREAM = torch.cuda.Stream()
        except Exception:
            return None
    return _PN95_CUDA_STREAM


def _pn95_async_enabled() -> bool:
    """Path C v1.0 Quality-First Sprint Q1 B1 — async stream gate.

    Default ON — async stream usage is lossless (stream sync ensures
    correctness). Operator can disable via GENESIS_PN95_ASYNC_STREAM=0
    for debugging or если cudagraph capture issues observed.
    """
    return os.environ.get(
        "GENESIS_PN95_ASYNC_STREAM", "1"
    ).strip().lower() in ("1", "true", "yes", "on")


def _pn95_gpu_to_cpu_bytes(view: Any) -> bytes:
    """Path C v1.0 Sprint Q1 B1 — async-aware GPU→CPU byte copy.

    Uses _pn95_stream когда available so demote PCIe transfer doesn't
    block default stream compute. Synchronous fallback preserves existing
    behavior (and correctness) когда CUDA unavailable.

    Returns bytes — caller may compress via _pn95_compress_bytes(...).

    Safety: stream.synchronize() before reading bytes ensures copy complete.
    Default stream NOT blocked during transfer (only synchronizes pn95 stream).
    """
    import torch
    stream = _pn95_stream() if _pn95_async_enabled() else None
    if stream is None:
        # Synchronous fallback — current behavior
        view_u8 = view.contiguous().view(torch.uint8).reshape(-1)
        return bytes(view_u8.cpu().numpy().tobytes())
    # Async — copy on _pn95_stream, frees default stream for compute.
    # Use .to("cpu", non_blocking=True) — universal torch API.
    # Must sync our stream before reading bytes (numpy access would sync
    # default stream, not our pn95 stream — explicit sync needed).
    with torch.cuda.stream(stream):
        view_u8 = view.contiguous().view(torch.uint8).reshape(-1)
        cpu_tensor = view_u8.to("cpu", non_blocking=True)
    stream.synchronize()
    _PN95_STATS["async_demote_count"] = (
        _PN95_STATS.get("async_demote_count", 0) + 1
    )
    return bytes(cpu_tensor.numpy().tobytes())


def _pn95_use_stream_pool() -> bool:
    """Stream-pool mode (upstream PR #40020-style event polling).

    When enabled, demote/promote submit work into the pooled-stream queue
    in _pn95_stream_pool and synchronize via end_event.synchronize() on
    that stream — no host-blocking torch.cuda.current_stream().synchronize()
    on the default stream. The default stream stays free to dispatch the
    next attention forward while PCIe DMA runs on the pooled stream.

    Default OFF (singleton stream path stays exact). Set
    GENESIS_PN95_USE_STREAM_POOL=1 to opt in.
    """
    return os.environ.get(
        "GENESIS_PN95_USE_STREAM_POOL", "0",
    ).strip().lower() in ("1", "true", "yes", "on")


def _pn95_gpu_to_cpu_bytes_batch_v2(views: list) -> list:
    """Stream-pool variant of the batched demote copy.

    Acquires a stream from the pool, queues all N copies on it, records
    end_event, calls end_event.synchronize() (waits ONLY on that event —
    default stream stays alive). Returns the bytes after sync.

    This is identical in I/O behaviour to the singleton-stream version
    but interoperable with submitted prefetch transfers (which use the
    same pool). It also paves the way for the next step where caller
    can submit() instead of synchronize() and check end_event.query()
    non-blockingly.
    """
    if not views:
        return []
    import torch
    from vllm.sndr_core.cache import _pn95_stream_pool as sp
    st = sp._state()
    stream = st.acquire_stream()
    end_evt = st.acquire_event()
    cpu_tensors = []
    try:
        with torch.cuda.stream(stream):
            for v in views:
                v_u8 = v.contiguous().view(torch.uint8).reshape(-1)
                cpu_tensors.append(v_u8.to("cpu", non_blocking=True))
            end_evt.record(stream)
        # Event sync — does NOT block the default stream's launch queue.
        end_evt.synchronize()
        _PN95_STATS["async_demote_count"] = (
            _PN95_STATS.get("async_demote_count", 0) + len(views)
        )
        _PN95_STATS["async_batch_demote_count"] = (
            _PN95_STATS.get("async_batch_demote_count", 0) + 1
        )
        _PN95_STATS["stream_pool_batches"] = (
            _PN95_STATS.get("stream_pool_batches", 0) + 1
        )
        return [bytes(t.numpy().tobytes()) for t in cpu_tensors]
    finally:
        st.release_stream(stream)
        st.release_event(end_evt)


def _pn95_cpu_to_gpu_copy_batch_v2(views: list, src_bytes_list: list) -> int:
    """Stream-pool variant of batched promote copy.

    Same correctness as v1 (default stream waits via wait_stream), but
    uses a freshly-acquired pooled stream and pooled end_event for
    interop with submit() prefetch work.
    """
    if not views or not src_bytes_list or len(views) != len(src_bytes_list):
        return 0
    import numpy as np
    import torch
    from vllm.sndr_core.cache import _pn95_stream_pool as sp
    st = sp._state()
    stream = st.acquire_stream()
    end_evt = st.acquire_event()
    n_total = 0
    try:
        with torch.cuda.stream(stream):
            for view, src_bytes in zip(views, src_bytes_list):
                src_arr = np.frombuffer(src_bytes, dtype=np.uint8).copy()
                # pin_memory() so cudaMemcpyAsync uses the fast PCIe path
                # (review finding #4 — without pinning we silently fall
                # back to the sync bounce-buffer copy).
                src_cpu = torch.from_numpy(src_arr).pin_memory()
                src_u8 = src_cpu.to(view.device, non_blocking=True)
                view_u8 = view.contiguous().view(torch.uint8).reshape(-1)
                n = min(view_u8.numel(), src_u8.numel())
                if n > 0:
                    view_u8[:n].copy_(src_u8[:n], non_blocking=True)
                    n_total += 1
            end_evt.record(stream)
        # Order: default stream consumes after our writes complete.
        torch.cuda.current_stream().wait_stream(stream)
        _PN95_STATS["async_promote_count"] = (
            _PN95_STATS.get("async_promote_count", 0) + n_total
        )
        _PN95_STATS["async_batch_promote_count"] = (
            _PN95_STATS.get("async_batch_promote_count", 0) + 1
        )
        _PN95_STATS["stream_pool_batches"] = (
            _PN95_STATS.get("stream_pool_batches", 0) + 1
        )
        return n_total
    finally:
        st.release_stream(stream)
        st.release_event(end_evt)


def _pn95_gpu_to_cpu_bytes_batch(views: list) -> list:
    """Path C v1.0 Sprint Q1 B2 — batched async GPU→CPU byte copy.

    Same effect as N calls к _pn95_gpu_to_cpu_bytes but with ONE
    stream.synchronize() instead of N. For 17-attention-layer demote,
    this saves ~16× stream sync overhead (~10-50 μs each → 160-800 μs total).

    Critical: PCIe DMA engine processes batched copies more efficiently
    too — multiple in-flight transfers overlap better than serial.

    Returns list of bytes в same order as input views. Empty list if
    views empty.

    Async stream usage controlled by GENESIS_PN95_ASYNC_STREAM (default ON).
    """
    if not views:
        return []
    # Stream-pool mode (env-gated, default OFF) — routes to v2 which uses
    # pooled streams + event-based sync. Interop with prefetch submit() work.
    if _pn95_use_stream_pool() and _pn95_async_enabled():
        return _pn95_gpu_to_cpu_bytes_batch_v2(views)
    import torch
    stream = _pn95_stream() if _pn95_async_enabled() else None
    if stream is None:
        # Synchronous fallback — equivalent к N sequential _pn95_gpu_to_cpu_bytes calls.
        # Each .cpu() triggers its own sync; same as before B2 introduced.
        return [
            bytes(v.contiguous().view(torch.uint8).reshape(-1).cpu().numpy().tobytes())
            for v in views
        ]
    # Async batched — queue ALL copies on _pn95_stream, single sync at end.
    cpu_tensors = []
    with torch.cuda.stream(stream):
        for v in views:
            v_u8 = v.contiguous().view(torch.uint8).reshape(-1)
            cpu_tensors.append(v_u8.to("cpu", non_blocking=True))
    # ONE sync для all N copies — saves (N-1) × ~10-50 μs overhead.
    stream.synchronize()
    _PN95_STATS["async_demote_count"] = (
        _PN95_STATS.get("async_demote_count", 0) + len(views)
    )
    _PN95_STATS["async_batch_demote_count"] = (
        _PN95_STATS.get("async_batch_demote_count", 0) + 1
    )
    return [bytes(t.numpy().tobytes()) for t in cpu_tensors]


def _pn95_cpu_to_gpu_copy_batch(views: list, src_bytes_list: list) -> int:
    """Path C v1.0 Sprint Q1 B3 — batched async CPU→GPU byte copy.

    Mirror of B2 (_pn95_gpu_to_cpu_bytes_batch) for the promote path.
    Same correctness primitive (current_stream.wait_stream(_pn95_stream))
    but ONE wait_stream call для N layer copies vs N individual calls.

    Args:
      views: list of GPU tensor views (one per layer)
      src_bytes_list: list of raw CPU bytes (decompressed уже), same length

    Returns: number of layers successfully copied (0 if mismatched lengths
    or empty input).

    Critical: like single-block helper, default stream waits for our copy
    via wait_stream() — no race против subsequent attention forward.
    """
    if not views or not src_bytes_list or len(views) != len(src_bytes_list):
        return 0
    # Stream-pool mode (env-gated, default OFF) — routes to v2.
    if _pn95_use_stream_pool() and _pn95_async_enabled():
        return _pn95_cpu_to_gpu_copy_batch_v2(views, src_bytes_list)
    import numpy as np
    import torch
    stream = _pn95_stream() if _pn95_async_enabled() else None

    if stream is None:
        # Synchronous fallback — equivalent к N sequential _pn95_cpu_to_gpu_copy calls.
        n_total = 0
        for view, src_bytes in zip(views, src_bytes_list):
            src_arr = np.frombuffer(src_bytes, dtype=np.uint8).copy()
            src_cpu = torch.from_numpy(src_arr)
            src_u8 = src_cpu.to(view.device)
            view_u8 = view.contiguous().view(torch.uint8).reshape(-1)
            n = min(view_u8.numel(), src_u8.numel())
            if n > 0:
                view_u8[:n].copy_(src_u8[:n], non_blocking=False)
                n_total += 1
        return n_total

    # Async batched — queue ALL N copies on _pn95_stream, ONE wait_stream at end.
    # Review finding #4: numpy.frombuffer produces pageable memory, so the
    # `.to(view.device, non_blocking=True)` falls back to sync bounce-buffer
    # DMA and the pinned-pool premise is defeated. We pin the source through
    # torch.from_numpy(...).pin_memory() so the DMA path is async pinned->GPU
    # like cudaMemcpyAsync expects (3-5 GB/s vs 600 MB/s pageable).
    n_total = 0
    with torch.cuda.stream(stream):
        for view, src_bytes in zip(views, src_bytes_list):
            src_arr = np.frombuffer(src_bytes, dtype=np.uint8).copy()
            src_cpu = torch.from_numpy(src_arr).pin_memory()
            src_u8 = src_cpu.to(view.device, non_blocking=True)
            view_u8 = view.contiguous().view(torch.uint8).reshape(-1)
            n = min(view_u8.numel(), src_u8.numel())
            if n > 0:
                view_u8[:n].copy_(src_u8[:n], non_blocking=True)
                n_total += 1
    # ONE wait_stream — saves (N-1) wait_stream calls.
    # Default stream waits for ALL our pn95-stream copies before next compute.
    torch.cuda.current_stream().wait_stream(stream)
    _PN95_STATS["async_promote_count"] = (
        _PN95_STATS.get("async_promote_count", 0) + n_total
    )
    _PN95_STATS["async_batch_promote_count"] = (
        _PN95_STATS.get("async_batch_promote_count", 0) + 1
    )
    return n_total


def _pn95_cpu_to_gpu_copy(view: Any, src_bytes: bytes) -> int:
    """Path C v1.0 Sprint Q1 B1 — async-aware CPU→GPU byte copy.

    Critical: writes к GPU view must be visible на default stream before
    subsequent attention forward reads it. Achieved via:
      `current_stream.wait_stream(_pn95_stream)` after copy.

    This makes default stream wait for our copy WITHOUT blocking CPU thread.

    Returns number of bytes copied. Synchronous fallback preserves current
    behavior когда CUDA unavailable.
    """
    import numpy as np
    import torch
    # np.frombuffer returns read-only array; torch.from_numpy then warns.
    # .copy() makes writable copy — safe для torch consumption.
    src_arr = np.frombuffer(src_bytes, dtype=np.uint8).copy()
    src_cpu = torch.from_numpy(src_arr)

    stream = _pn95_stream() if _pn95_async_enabled() else None
    if stream is None:
        # Synchronous fallback
        src_u8 = src_cpu.to(view.device)
        view_u8 = view.contiguous().view(torch.uint8).reshape(-1)
        n = min(view_u8.numel(), src_u8.numel())
        if n > 0:
            view_u8[:n].copy_(src_u8[:n], non_blocking=False)
        return n
    # Async — copy on _pn95_stream, default stream waits for completion
    # BEFORE attention reads the new block (correctness guarantee).
    with torch.cuda.stream(stream):
        src_u8 = src_cpu.to(view.device, non_blocking=True)
        view_u8 = view.contiguous().view(torch.uint8).reshape(-1)
        n = min(view_u8.numel(), src_u8.numel())
        if n > 0:
            view_u8[:n].copy_(src_u8[:n], non_blocking=True)
    # CRITICAL: default stream waits for our pn95 stream — ordering
    # without blocking. Subsequent attention forward gets correct data.
    torch.cuda.current_stream().wait_stream(stream)
    _PN95_STATS["async_promote_count"] = (
        _PN95_STATS.get("async_promote_count", 0) + 1
    )
    return n


def pn95_phase5_init_block_pool(pool: Any) -> None:
    """Path C v1.0 Phase 5 (Anchors #11 + extended for #12) — initialize
    side-table metadata + optionally inflate the pool with virtual blocks.

    Two modes:
      VIRT=0 (default): metadata side-table init only, NO behavior change.
      VIRT=1: also creates N_virtual extra KVCacheBlock objects with virtual
        block_ids ≥ num_physical, marks them physical_resident=False, adds
        to pool.blocks list AND free_block_queue. Scheduler sees inflated
        num_gpu_blocks. When get_new_blocks pops a virtual block, Anchor #12
        materializes it via swap (mutates block.block_id to physical id from
        a demoted donor block).

    Side-table per (id(pool), block_id):
      - physical_resident: bool
      - physical_block_id: Optional[int] — None for virtual, GPU slot otherwise
      - last_access_tick: int

    N_virtual computed from CPU tier capacity / page_size_bytes_per_block.

    Idempotent — safe to call multiple times on same pool.
    Fail-silent — returns immediately on any error.
    """
    if not _enabled():
        return
    try:
        pool_id = id(pool)

        # Initialize per-block metadata for ALL existing physical blocks
        blocks = getattr(pool, "blocks", None) or []
        n_physical = len(blocks)

        for blk in blocks:
            key = (pool_id, blk.block_id)
            if key not in _PN95_BLOCK_METADATA:
                _PN95_BLOCK_METADATA[key] = {
                    "physical_resident": True,
                    "physical_block_id": blk.block_id,
                    "last_access_tick": 0,
                }

        # VIRT=0: no inflation. Default behavior preserved.
        if not _phase5_virt_enabled():
            if pool_id not in _PN95_POOL_LOGICAL_NUM_BLOCKS:
                _PN95_POOL_LOGICAL_NUM_BLOCKS[pool_id] = pool.num_gpu_blocks
            return

        # VIRT=1: create virtual blocks. CONSERVATIVE inflation strategy
        # to prevent scheduler over-admission crash:
        #
        # The fundamental constraint: when single request grows past
        # physical pool size, vllm cannot preempt its own blocks (that
        # would lose in-progress work). Virtual blocks can only be safely
        # materialized when a DIFFERENT request's blocks are in free_queue
        # (ref_cnt=0). For single-request long-context, this never happens
        # → materialization fails → crash.
        #
        # SAFE INFLATION = inflate only by amount that scheduler will
        # admit cross-request rotation can absorb. Empirical safe ratio:
        # logical = physical × INFLATION_RATIO where ratio ≤ 1.5 typically
        # (operator override via GENESIS_PN95_VIRT_INFLATION_RATIO).
        #
        # For aggressive single-request expansion, scheduler-level
        # preemption is required — that's beyond text-patches.
        # Idempotent guard.
        if pool_id in _PN95_POOL_LOGICAL_NUM_BLOCKS:
            return

        # Compute N_virtual from tier capacity. Need per-block bytes
        # estimate — use TM's _attention_views first registered layer
        # bytes_per_block, fallback to 49664 (TQ k8v4 default for 27B PROD).
        bytes_per_block = 49664
        try:
            views = getattr(_TM, "_attention_views", None) or {}
            if views:
                first_info = next(iter(views.values()))
                bytes_per_block = int(first_info.get("bytes_per_block", 49664))
        except Exception:
            pass

        cpu_tier_bytes = pn95_extra_logical_memory_bytes()
        # Account for ALL eligible attention layers (each block's bytes
        # × n_layers must fit in CPU tier).
        n_attn_layers = max(1, len(getattr(_TM, "_attention_views", {}) or {}) or 17)
        bytes_per_full_block = bytes_per_block * n_attn_layers
        cpu_capacity_blocks = int(cpu_tier_bytes // bytes_per_full_block)

        # SAFE INFLATION RATIO — bounded by physical pool size to prevent
        # scheduler over-admission crash. Default 1.5x physical (operator
        # can tune via env GENESIS_PN95_VIRT_INFLATION_RATIO).
        try:
            inflation_ratio = float(os.environ.get(
                "GENESIS_PN95_VIRT_INFLATION_RATIO", "1.5"))
        except (ValueError, TypeError):
            inflation_ratio = 1.5
        inflation_ratio = max(1.0, min(inflation_ratio, 8.0))  # clamp [1, 8]

        # Max safe virtual count based on physical pool size
        max_safe_virtual = int(n_physical * (inflation_ratio - 1.0))

        # Take min of CPU capacity and safe inflation
        n_virtual = min(cpu_capacity_blocks, max_safe_virtual)

        # Operator override caps (avoid runaway memory)
        try:
            virt_max = int(os.environ.get(
                "GENESIS_PN95_VIRT_MAX_BLOCKS", "10000"))
        except (ValueError, TypeError):
            virt_max = 10000
        n_virtual = min(n_virtual, virt_max)

        if n_virtual <= 0:
            log.warning(
                "[PN95 v1.0 Phase 5] VIRT=1 but no CPU tier capacity "
                "available for virtual blocks — skipping inflation"
            )
            _PN95_POOL_LOGICAL_NUM_BLOCKS[pool_id] = pool.num_gpu_blocks
            return

        # Create virtual blocks. KVCacheBlock(block_id=int) constructor.
        # Need access to vllm's KVCacheBlock class — import lazily.
        try:
            from vllm.v1.core.kv_cache_utils import KVCacheBlock
        except ImportError:
            log.warning(
                "[PN95 v1.0 Phase 5] cannot import KVCacheBlock — virt skipped"
            )
            _PN95_POOL_LOGICAL_NUM_BLOCKS[pool_id] = pool.num_gpu_blocks
            return

        free_q = getattr(pool, "free_block_queue", None)
        if free_q is None:
            log.warning(
                "[PN95 v1.0 Phase 5] pool has no free_block_queue — virt skipped"
            )
            _PN95_POOL_LOGICAL_NUM_BLOCKS[pool_id] = pool.num_gpu_blocks
            return

        # Generate virtual blocks with synthetic block_ids starting from
        # n_physical. Mark each as virtual in side-table.
        virtual_blocks = []
        for i in range(n_virtual):
            virt_id = n_physical + i
            try:
                vblk = KVCacheBlock(block_id=virt_id)
                _PN95_BLOCK_METADATA[(pool_id, virt_id)] = {
                    "physical_resident": False,
                    "physical_block_id": None,
                    "last_access_tick": 0,
                }
                virtual_blocks.append(vblk)
            except Exception:
                continue

        # Append к pool.blocks list and free_block_queue.
        # FreeKVCacheBlockQueue uses doubly-linked-list пointers
        # (prev_free_block / next_free_block). Use append_n method.
        if hasattr(free_q, "append_n"):
            try:
                free_q.append_n(virtual_blocks)
            except Exception as e:
                log.warning(
                    "[PN95 v1.0 Phase 5] free_q.append_n failed: %s — virt skipped",
                    e,
                )
                _PN95_POOL_LOGICAL_NUM_BLOCKS[pool_id] = pool.num_gpu_blocks
                return

        try:
            pool.blocks.extend(virtual_blocks)
        except Exception:
            pass

        # Inflate pool.num_gpu_blocks для scheduler awareness
        new_logical = n_physical + len(virtual_blocks)
        try:
            pool.num_gpu_blocks = new_logical
        except Exception:
            pass

        _PN95_POOL_LOGICAL_NUM_BLOCKS[pool_id] = new_logical

        log.warning(
            "[PN95 v1.0 Phase 5 Anchor #11+] inflated BlockPool: "
            "physical=%d → logical=%d (+%d virtual blocks, "
            "CPU tier %.1f GiB / block %d B × %d layers = %d B/full)",
            n_physical, new_logical, len(virtual_blocks),
            cpu_tier_bytes / (1 << 30), bytes_per_block, n_attn_layers,
            bytes_per_full_block,
        )
    except Exception as e:
        log.warning("[PN95 v1.0 Phase 5] init_block_pool failed: %s", e)


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

    Race safety: relies on ref_cnt=0 invariant — donors не active в
    request block_tables. vllm v1 GIL serializes access.
    """
    if not _enabled() or not _phase5_virt_enabled():
        return None
    if _TM is None:
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
            # ones are PREFERRED because we can save их bytes для restore
            cur_id = getattr(cur, "block_id", -1)
            cur_meta = _PN95_BLOCK_METADATA.get((pool_id, cur_id))
            if cur_meta is None or not cur_meta.get("physical_resident", False):
                cur = getattr(cur, "next_free_block", None)
                continue
            donor = cur
            break

        if donor is None:
            return None

        # Capture donor's bytes к CPU prefix store (only if cached)
        donor_hash = getattr(donor, "block_hash", None)
        donor_phys_id = donor.block_id  # physical_resident → block_id == physical_block_id
        if donor_hash is not None:
            # Best-effort capture — failure не critical (just lose cache hit later)
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
        _PN95_BLOCK_METADATA[(pool_id, donor_phys_id)] = new_physical_meta
        # donor (which will get new id virt_id) is now virtual
        _PN95_BLOCK_METADATA[(pool_id, virt_id)] = new_virtual_meta

        # Donor block stays в free_queue. Mutate его block_id к virt_id
        # so future free_queue traversal sees correct (mutated) id.
        # Caller mutates virt_block.block_id к donor_phys_id separately.
        try:
            donor.block_id = virt_id
            # Clear donor's hash — его bytes теперь на CPU (or never were cached)
            if hasattr(donor, "_block_hash"):
                donor._block_hash = None
        except Exception:
            pass

        _PN95_STATS["blocks_materialized_total"] = (
            _PN95_STATS.get("blocks_materialized_total", 0) + 1
        )
        return donor_phys_id
    except Exception as e:
        log.warning("[PN95 v1.0 Phase 5] materialize_virtual_block failed: %s", e)
        return None


def pn106_get_gdn_h_buf(B: int, NT: int, H: int, V: int, K: int,
                         dtype: Any, device: Any):
    """Legacy entry-point — delegates to generic named-pool allocator."""
    return pn106_get_pooled_buf("gdn_h", (B, NT, H, V, K), dtype, device)


def _pn106_legacy_h_impl(B: int, NT: int, H: int, V: int, K: int,
                         dtype: Any, device: Any):
    """Return a view of the singleton GDN h-state pool sized (B, NT, H, V, K).

    The pool itself is grown on demand to the max (NT × H × V × K) seen
    so far. Same-shape requests reuse the same backing storage; bigger
    NT triggers a one-time pool re-grow (PyTorch caching allocator
    typically reuses the slab even on grow).

    Reused across all 48 GDN layers within a step AND across steps.
    Saves ~50-120 MiB alloc/free traffic per layer per call (Qwen3.6-27B
    has 48 GDN layers, so net ~2.4-5.7 GiB allocator traffic eliminated
    per chunked-prefill step). Steady-state ~200-400 MiB fragmentation
    reclaimed.

    Safety: this returns a VIEW into a shared buffer. Caller MUST
    fully overwrite via the downstream Triton kernel (which it does —
    `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` writes the full
    h tensor). No stale data risk because the kernel does not read
    h on input.
    """
    import torch
    elem_per_slot = B * NT * H * V * K
    elem_bytes = torch.empty(0, dtype=dtype).element_size()
    bytes_needed = elem_per_slot * elem_bytes

    key = (str(device), str(dtype))
    pool = _PN106_POOLS.get(key)
    if pool is None or pool.numel() * pool.element_size() < bytes_needed:
        # Grow (or allocate). Round up to 1.25x for headroom.
        target_elems = int(elem_per_slot * 1.25)
        try:
            pool = torch.empty(target_elems, dtype=dtype, device=device)
            _PN106_POOLS[key] = pool
            _PN95_STATS["pn106_pool_grows"] = (
                _PN95_STATS.get("pn106_pool_grows", 0) + 1
            )
            _PN95_STATS["pn106_pool_bytes"] = pool.numel() * pool.element_size()
        except Exception:
            return None

    view = pool[: elem_per_slot].view(B, NT, H, V, K)
    _PN95_STATS["pn106_h_slices_served"] = (
        _PN95_STATS.get("pn106_h_slices_served", 0) + 1
    )
    return view


_PN106_POOLS: dict = {}
_PN106_NAMED_POOLS: dict = {}  # name -> torch.Tensor (flat backing buffer)


def pn106_get_pooled_buf(name: str, shape: tuple, dtype: Any, device: Any,
                         zero: bool = False):
    """Generic named-pool allocator for hot-path scratch tensors.

    Replaces fresh `torch.empty(shape, ...)` / `torch.empty_like(t)` with
    a view into a persistent flat backing buffer that grows on demand
    and is reused across calls.

    Args:
      zero: if True, zero the returned slice before handing back. Use for
            `torch.zeros(...)` replacements where the kernel expects
            initialized memory (e.g. gdn core_attn_out — see vllm PR
            #28182 discussion). Adds ~5-15us overhead per call
            (memset bandwidth-bound, well under fragmentation cost).

    Args:
      name: stable pool identifier ('gdn_h', 'gdn_v_new', 'gdn_o', etc.)
      shape: requested tensor shape (any rank)
      dtype: torch.dtype
      device: torch.device

    Returns:
      A view-tensor of `shape` backed by the named pool, or None if
      allocation fails.

    Each (name, device, dtype) gets an independent backing buffer.
    Growth is by 1.25x to amortize re-alloc on slowly-growing peaks.

    Caller MUST overwrite the returned view before reading any element
    (which is the case for all our patched sites — Triton kernels are
    write-only on these scratch tensors). No correctness loss.
    """
    import torch
    n_elems = 1
    for d in shape:
        n_elems *= int(d)
    if n_elems <= 0:
        return None
    key = (name, str(device), str(dtype))
    pool = _PN106_NAMED_POOLS.get(key)
    if pool is None or pool.numel() < n_elems:
        target = max(n_elems, int(n_elems * 1.25))
        # Round up to 4K elements to dampen growth churn
        target = ((target + 4095) // 4096) * 4096
        try:
            pool = torch.empty(target, dtype=dtype, device=device)
        except Exception:
            return None
        _PN106_NAMED_POOLS[key] = pool
        _PN95_STATS[f"pn106_pool_{name}_grows"] = (
            _PN95_STATS.get(f"pn106_pool_{name}_grows", 0) + 1
        )
        _PN95_STATS[f"pn106_pool_{name}_bytes"] = (
            pool.numel() * pool.element_size()
        )
    view = pool[:n_elems].view(*shape)
    if zero:
        view.zero_()
    _PN95_STATS[f"pn106_pool_{name}_slices"] = (
        _PN95_STATS.get(f"pn106_pool_{name}_slices", 0) + 1
    )
    return view


_PN201_LAST_EMPTY_CACHE_TICK: int = 0

# PN203 cold-prefix offload settings (set by PN203 apply hook at boot).
# Read by scheduler_tick to decide whether to do window-aware demote
# of full-attention layer blocks older than _PN203_ACTIVE_WINDOW_TOKENS.
_PN203_ENABLED: bool = False
_PN203_ACTIVE_WINDOW_TOKENS: int = 32768
_PN203_ATTENTION_ONLY: bool = True


def pn203_cold_prefix_sweep() -> int:
    """Tier 3.A — sweep cold prefix blocks beyond active window to L2.

    Walks each registered BlockPool's free_block_queue and demotes
    cached blocks belonging to full-attention layers whose position
    in the request's KV is older than `_PN203_ACTIVE_WINDOW_TOKENS`.
    Mamba/GDN blocks left GPU-resident (state is fixed-size per layer
    regardless of position, and demoting them is unsafe per PN95 design).

    Returns count of blocks swept. Best-effort — fail-silent.

    Coordinates with existing PN95 path: demote_on_evict already captures
    bytes to L2 (pinned pool if PN95_PINNED_POOL enabled), so this
    function just adds the window-aware selection policy.
    """
    if not _PN203_ENABLED or not _enabled() or _TM is None:
        return 0
    swept = 0
    try:
        # Window-aware filtering: prefer blocks deep in admit_order (older
        # positions). We approximate "position" by admit-order index;
        # blocks admitted earlier are older in the request stream.
        # Hard mapping (per-request position) requires per-block metadata;
        # this approximation is good enough for cold-prefix detection.
        if not _PN95_BLOCK_POOL_REFS:
            return 0
        # Use existing LRU walker but cap to window-relative cold candidates.
        candidates = _select_cold_blocks_via_bpool_lru(target_count=16)
        for pool, block_id, block_hash in candidates:
            # Filter: attention-only mode skips Mamba groups (block_hash
            # carries group_id which we check against _mamba_excluded).
            if _PN203_ATTENTION_ONLY and _TM is not None:
                try:
                    gid_str = getattr(block_hash, "group_id", None)
                    if gid_str in getattr(_TM, "_mamba_excluded", set()):
                        continue
                except Exception:
                    pass
            try:
                if demote_on_evict(block_hash, block_id):
                    swept += 1
            except Exception:
                continue
        if swept > 0:
            _PN95_STATS["pn203_cold_prefix_sweeps"] = (
                _PN95_STATS.get("pn203_cold_prefix_sweeps", 0) + 1
            )
            _PN95_STATS["pn203_blocks_swept_total"] = (
                _PN95_STATS.get("pn203_blocks_swept_total", 0) + swept
            )
    except Exception as e:
        log.warning("[PN203] cold_prefix_sweep failed silently: %s", e)
    return swept


def pn201_maybe_empty_cache(free_mib: int, free_blocks: Optional[int] = None) -> bool:
    """Threshold-gated empty_cache call for scheduler_tick path (Tier 1.C).

    Defragments the PyTorch CUDA caching allocator when memory pressure
    is high. Returns True iff empty_cache was actually called this tick.

    Triggered when EITHER:
      - free_blocks < GENESIS_PN201_EMPTY_CACHE_FREE_BLOCKS_THRESHOLD
        (default 8 — matches PN95 proactive demote threshold scale)
      - free_mib < 256

    Cooldown: GENESIS_PN201_EMPTY_CACHE_COOLDOWN ticks (default 50, ~5s
    at default tick rate). Without cooldown, back-to-back chunks could
    fire empty_cache continuously, each blocking ~5 ms.

    Architectural note: this hook is the Tier-1.C piece — pure fragmentation
    reclaim. The Tier-3 CPU offload manager (PN203) will fire from the
    same scheduler_tick using the same pressure signal but doing real
    block migration instead of cache discard.
    """
    global _PN201_LAST_EMPTY_CACHE_TICK
    if os.environ.get(
        "GENESIS_ENABLE_PN201_SCHEDULER_EMPTY_CACHE", "0",
    ).strip().lower() not in ("1", "true", "yes", "on"):
        return False

    try:
        threshold_blocks = int(os.environ.get(
            "GENESIS_PN201_EMPTY_CACHE_FREE_BLOCKS_THRESHOLD", "8"))
        cooldown = int(os.environ.get(
            "GENESIS_PN201_EMPTY_CACHE_COOLDOWN", "50"))
    except (ValueError, TypeError):
        threshold_blocks, cooldown = 8, 50

    pressure = free_mib < 256
    if free_blocks is not None:
        pressure = pressure or free_blocks < threshold_blocks
    if not pressure:
        return False

    tick = _PN95_STATS.get("ticks_total", 0)
    if tick - _PN201_LAST_EMPTY_CACHE_TICK < cooldown:
        _PN95_STATS["pn201_empty_cache_cooldowns"] = (
            _PN95_STATS.get("pn201_empty_cache_cooldowns", 0) + 1
        )
        return False

    try:
        import torch
        torch.cuda.empty_cache()
        _PN201_LAST_EMPTY_CACHE_TICK = tick
        _PN95_STATS["pn201_empty_cache_calls"] = (
            _PN95_STATS.get("pn201_empty_cache_calls", 0) + 1
        )
        log.info(
            "[PN201] empty_cache fired at tick=%d free_mib=%d free_blocks=%s",
            tick, free_mib, free_blocks,
        )
        return True
    except Exception as e:
        log.warning("[PN201] empty_cache failed: %s", e)
        return False


def pn106_periodic_empty_cache() -> None:
    """Call `torch.cuda.empty_cache()` to defragment the allocator.

    Invoked sparingly (every Nth scheduler tick) to reclaim "reserved but
    unallocated" memory observed in the OOM crash log (~319 MiB
    fragmentation). The CUDA caching allocator does not give back
    reserved-but-free slabs until empty_cache is called. Critical for
    long-running deployments — fragmentation accumulates as variable-
    sized chunks pass through the GDN/attention path.

    Env-driven cadence: GENESIS_PN106_EMPTY_CACHE_EVERY_N_TICKS
    (default 0 = disabled — operator opts in when fragmentation
    actually hurts).
    """
    try:
        n = int(os.environ.get("GENESIS_PN106_EMPTY_CACHE_EVERY_N_TICKS", "0"))
    except (ValueError, TypeError):
        n = 0
    if n <= 0:
        return
    tick = _PN95_STATS.get("ticks_total", 0)
    if tick == 0 or tick % n != 0:
        return
    try:
        import torch
        torch.cuda.empty_cache()
        _PN95_STATS["pn106_empty_cache_calls"] = (
            _PN95_STATS.get("pn106_empty_cache_calls", 0) + 1
        )
    except Exception:
        pass


def pn97_physical_cap_bytes(n_tensors: int) -> Optional[int]:
    """Return per-KVCacheTensor byte cap for PN97 (Phase 7 PoC).

    Called from the PN97 anchor at `_allocate_kv_cache_tensors`. We
    compute the maximum bytes that fit in the physical GPU KV budget
    (gpu_memory_utilization × VRAM − model_weights − workspace),
    divided evenly across the `n_tensors` KVCacheTensor entries.

    Returns None when PN97 disabled OR VIRT_ENABLE off (no inflation
    happening, no cap needed) OR torch unavailable.

    Operator can override via `GENESIS_PN97_PHYSICAL_CAP_GIB` (single
    value, total across all tensors).
    """
    if os.environ.get("GENESIS_ENABLE_PN97_TENSOR_PHYSICAL_CAP", "0").strip().lower() not in (
        "1", "true", "yes", "on",
    ):
        return None
    if n_tensors <= 0:
        return None

    # Operator override (total bytes across all tensors).
    env_total = os.environ.get("GENESIS_PN97_PHYSICAL_CAP_GIB", "").strip()
    if env_total:
        try:
            total_bytes = int(float(env_total) * (1 << 30))
            return total_bytes // n_tensors
        except (ValueError, TypeError):
            pass

    # Auto-derive: query torch for free GPU memory and reserve 80% for KV.
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free_bytes, _total = torch.cuda.mem_get_info(0)
        # 80% of currently-free memory goes to KV (rest = workspace/activations).
        # This is conservative — operator may bump via env override.
        per_tensor = int(free_bytes * 0.80) // n_tensors
        return max(per_tensor, 1 << 30)  # at least 1 GiB per tensor entry
    except Exception:
        return None


def pn96_emergency_rescue(pool: Any, deficit: int) -> int:
    """Phase 6 PoC — emergency rescue when get_new_blocks would crash.

    Called from the PN96 anchor BEFORE vllm raises
    `ValueError("Cannot get N free blocks")`. Walks the pool's
    free_block_queue looking for ALREADY-CACHED blocks (block_hash
    set, ref_cnt=0). For each such block, captures the bytes to the
    PN95 L2 store via demote_on_evict, then marks the slot reusable
    (by clearing its hash — vllm will pop it from cached_block_hash_to_block
    on the next eviction pass).

    Returns the number of slots rescued. Best-effort: on any error
    returns 0 (caller falls through to the upstream ValueError).

    Why this matters: vllm's block pool can have free slots that
    are "free but reserved for cache reuse" — they show up in
    free_block_queue but with non-None block_hash. The eviction
    happens lazily inside `_maybe_evict_cached_block`. PN96 forces
    eager eviction with byte-preservation so the pool reports enough
    free slots to satisfy the current allocation request.

    Honest limitation: this ONLY rescues already-free cached blocks.
    Active blocks (ref_cnt>0, held by a running sequence) are NOT
    touched — those would need scheduler-level preemption which is
    Phase 7 work. So PN96 helps multi-prefix workloads but does
    NOT extend the single-user max_model_len above the GPU pool size.
    """
    if not _enabled() or deficit <= 0:
        return 0
    rescued = 0
    try:
        free_q = getattr(pool, "free_block_queue", None)
        if free_q is None:
            return 0
        head = (getattr(free_q, "fake_free_list_head", None)
                or getattr(free_q, "_fake_head", None))
        cur = getattr(head, "next_free_block", None) if head else None
        walked = 0
        max_walk = max(deficit * 4, 64)  # cap walk cost
        while cur is not None and walked < max_walk and rescued < deficit:
            walked += 1
            block_hash = getattr(cur, "block_hash", None)
            ref_cnt = getattr(cur, "ref_cnt", -1)
            block_id = getattr(cur, "block_id", -1)
            is_null = getattr(cur, "is_null", False)
            nxt = getattr(cur, "next_free_block", None)
            if (block_hash is not None and ref_cnt == 0
                    and block_id >= 0 and not is_null):
                # Preserve bytes BEFORE vllm reuses this slot.
                try:
                    if demote_on_evict(block_hash, block_id):
                        rescued += 1
                        # Clear hash so vllm sees it as a clean free slot.
                        try:
                            cur.block_hash = None
                        except Exception:
                            pass
                except Exception:
                    pass
            cur = nxt
        if rescued > 0:
            _PN95_STATS["pn96_emergency_rescues"] = (
                _PN95_STATS.get("pn96_emergency_rescues", 0) + 1
            )
            _PN95_STATS["pn96_blocks_rescued_total"] = (
                _PN95_STATS.get("pn96_blocks_rescued_total", 0) + rescued
            )
            log.info(
                "[PN96] emergency rescue: rescued %d slots (deficit was %d, walked %d)",
                rescued, deficit, walked,
            )
    except Exception as e:
        log.warning("[PN96] emergency_rescue failed silently: %s", e)
    return rescued


def pn95_block_is_physical_resident(pool: Any, block_id: int) -> bool:
    """Return True iff `block_id` exists in the physical KV pool range.

    Used by Anchor #14 (defensive guard at get_new_blocks) to detect a
    virtual block leaked out before the runtime is ready to materialize
    it. A virtual block_id >= physical_num_blocks would index past the
    KVCacheTensor → CUDA illegal memory access, which we want to convert
    into a clean ValueError instead of a worker-process crash.

    For pools without Phase-5 metadata (which is the normal case while
    VIRT_ENABLE=0) every block is physical by definition; we return True.
    """
    pool_id = id(pool)
    physical_num = _PN95_POOL_LOGICAL_NUM_BLOCKS.get(pool_id, -1)
    if physical_num <= 0:
        return True  # no virtual blocks ever created on this pool
    return 0 <= block_id < physical_num


def pn95_guard_get_new_blocks(pool: Any, blocks: list) -> None:
    """Phase 5 Anchor #14 — defensive guard: reject virtual blocks before
    they reach the GPU.

    Walks the just-popped block list; if any block has a block_id >=
    physical_num_blocks AND Phase-5 materialization is not ready (no
    donor available, VIRT disabled, etc.) — raise ValueError so the
    scheduler retries / preempts instead of letting the worker dereference
    invalid GPU memory.

    Best-effort materialization is still attempted for each suspicious
    block via pn95_materialize_virtual_block — only if THAT also fails
    do we surface the error. On normal (VIRT=0) deployments this is a
    fast no-op (the pool has no virtual blocks).
    """
    if not _enabled() or _TM is None:
        return
    pool_id = id(pool)
    if _PN95_POOL_LOGICAL_NUM_BLOCKS.get(pool_id, -1) <= 0:
        return  # no Phase-5 inflation on this pool
    for blk in blocks:
        bid = getattr(blk, "block_id", -1)
        if pn95_block_is_physical_resident(pool, bid):
            continue
        # Attempt materialization (swap with a donor physical).
        new_phys = pn95_materialize_virtual_block(pool, blk, exclude=blocks)
        if new_phys is None:
            _PN95_STATS["virtual_block_unmaterialized_total"] = (
                _PN95_STATS.get("virtual_block_unmaterialized_total", 0) + 1
            )
            raise ValueError(
                f"[Genesis PN95 Anchor #14] virtual block_id={bid} could "
                f"not be materialized (no free donor available). This is "
                f"the documented 'Phase 5 VIRT without scheduler "
                f"preemption' failure mode — set "
                f"GENESIS_PN95_VIRT_ENABLE=0 to disable, or wait for "
                f"scheduler-preemption work to land."
            )
        # Adopt donor's physical block_id.
        try:
            blk.block_id = new_phys
        except Exception:
            raise ValueError(
                f"[Genesis PN95 Anchor #14] materialization succeeded "
                f"(donor phys_id={new_phys}) but block_id assignment "
                f"failed; refusing to return virtual block to engine."
            )


def pn95_anchor12_post_popleft(pool: Any, popped_blocks: list) -> bool:
    """Path C v1.0 Phase 5 Anchor #12 — post-process popped blocks list.

    DESIGN UPDATE 2026-05-09: rollback to truly no-op behavior.

    Discovery from live testing: swap-based virtualization fundamentally
    unsafe without scheduler-level preemption. When all physical blocks
    are held by active requests (ref_cnt > 0), no donors available in
    free_queue → materialization fails → ValueError → engine_core dies.

    To safely reach 256K context via virtualization requires:
      1. Scheduler-side preemption (evict older requests' blocks)
      2. Request queueing on partial allocation failure
      3. Cross-step coordination of demote/promote with attention reads

    This is scheduler-level architecture work, NOT achievable via
    text-patches alone. Deferred to future sub-project.

    For now: this anchor returns True (no-op) when PN95/VIRT disabled.
    When VIRT=1 + virtual blocks present: WARNINGS but no crash —
    materialization attempted on best-effort basis. If donor unavailable,
    block stays virtual (will crash attention read) — but Anchor #11
    inflation also rolled back to no-virtual-creation, so this path
    never triggers in practice.
    """
    if not _enabled() or not _phase5_virt_enabled():
        return True
    # Best-effort materialization but NEVER raise — return True even on
    # partial failure to avoid crashing engine_core. With Anchor #11
    # rolled back to no-inflation, this loop typically finds no virtual
    # blocks and is a fast pass-through.
    try:
        pool_id = id(pool)
        for block in popped_blocks:
            meta = _PN95_BLOCK_METADATA.get((pool_id, block.block_id))
            if meta is None:
                continue
            if meta.get("physical_resident", True):
                continue
            new_phys_id = pn95_materialize_virtual_block(
                pool, block, exclude=popped_blocks,
            )
            if new_phys_id is None:
                # Best-effort: cannot materialize but cannot crash either.
                # Block stays virtual; if attention reads it, vllm crashes
                # at attention layer, not here. With Anchor #11 inflation
                # rolled back, this path is unreachable in normal use.
                continue
            try:
                block.block_id = new_phys_id
            except Exception:
                continue
        return True
    except Exception:
        return True


def pn95_block_metadata(pool: Any, block_id: int) -> Optional[dict]:
    """Phase 5 Session 2 helper — read PN95 metadata for (pool, block_id).

    Returns None when:
      - PN95 disabled
      - Pool not yet initialized via pn95_phase5_init_block_pool
      - block_id not in side-table (e.g., for null_block which has block_id=0
        but special semantics)
    """
    if not _enabled():
        return None
    return _PN95_BLOCK_METADATA.get((id(pool), block_id))


def pn95_pool_logical_num_blocks(pool: Any) -> Optional[int]:
    """Phase 5 helper — get logical num_blocks for a pool.

    Returns None if not initialized → caller treats as unmodified vllm
    behavior. When Session 3+ activates virtualization, this returns
    inflated count (physical + cpu_tier_blocks).
    """
    if not _enabled():
        return None
    return _PN95_POOL_LOGICAL_NUM_BLOCKS.get(id(pool))


def pn95_physical_num_blocks_cap() -> Optional[int]:
    """Phase 5 Anchor #10 helper — cap physical KVCacheTensor allocation
    to GPU-only memory budget.

    Without this cap, inflated available_memory in Anchor #9 would
    cause vllm to allocate `KVCacheTensor(size=page_size * num_blocks)`
    sized for inflated num_blocks → CUDA OOM at init.

    Returns the max block count that fits in tier 0 (GPU). vllm's
    allocation site uses min(num_blocks_logical, this_cap) for the
    actual torch.empty() call. The remaining (num_blocks_logical -
    cap) blocks become "virtual" — created in BlockPool's metadata
    pool but not backed by physical GPU memory until materialized
    via Anchor #12 (next session).

    Returns None when Phase 5 virt disabled — caller treats as
    "no cap" and uses original num_blocks (current behavior).

    NOTE: Returns BYTES budget, not block count. Caller (Anchor #10
    site) divides by page_size to get block count for KVCacheTensor.
    """
    if not _phase5_virt_enabled():
        return None
    tm = _TM
    if tm is None or len(tm.tiers) < 1:
        return None
    try:
        gpu_tier = tm.tiers[0]
        if getattr(gpu_tier, "device", "") != "gpu":
            return None
        cap_gib = float(getattr(gpu_tier, "capacity_gib", 0.0))
        if cap_gib <= 0:
            return None
        return int(cap_gib * (1 << 30))
    except Exception:
        return None


def _detect_upstream_offload_connector(cfg: Any) -> Optional[str]:
    """Return the upstream KV-offload connector class name when one is
    declared on the engine config, else None.

    Mirrors the upstream v1/kv_offload/ framework gate: when an
    OffloadingConnector is wired in, the upstream offload manager
    owns block residency. Running PN95 in parallel would double-manage
    the same blocks and lead to undefined behaviour.

    Probe order:
      1. cfg.kv_transfer_config.kv_connector  (vLLM v0/v1 standard)
      2. cfg.kv_transfer_config.connector_class  (newer attribute)
      3. cfg.cache_config.offload_connector  (PN95 sibling spec)

    A non-empty truthy string is treated as a positive detection; any
    other shape (None, empty, non-string) is treated as not present.
    """
    candidates = (
        ("kv_transfer_config", "kv_connector"),
        ("kv_transfer_config", "connector_class"),
        ("kv_transfer_config", "kv_connector_module_path"),  # PR #40020 surface
        ("cache_config", "offload_connector"),
        ("cache_config", "kv_connector"),                    # older alias
    )
    for top, attr in candidates:
        block = getattr(cfg, top, None)
        if block is None:
            continue
        value = getattr(block, attr, None)
        if isinstance(value, str) and value.strip():
            return value

    # Fallback scan: any *connector* attribute on kv_transfer_config with
    # a truthy-string value indicates an offload framework is wired in.
    # Catches custom user spec names that bypass the well-known attrs above.
    block = getattr(cfg, "kv_transfer_config", None)
    if block is not None:
        try:
            attrs = vars(block)
        except TypeError:
            attrs = {}
        for k, v in attrs.items():
            if "connector" in k.lower() and isinstance(v, str) and v.strip():
                return v
    return None


def init_from_config(cfg: Any) -> bool:
    """Install the TierManager singleton from a ModelConfig.

    Returns True iff a manager was installed; False when:
      - PN95 is disabled via env
      - cfg has no cache_config.tiers
      - upstream KV-offload connector is wired in (avoid double-manage)
      - import error / construction error (logged + swallowed)

    Idempotent: re-calling with the same cfg leaves the singleton
    untouched. Re-calling with a different cfg replaces it (with a
    warning logged).
    """
    if not _enabled():
        return False
    global _TM
    with _LOCK:
        # Upstream offload coexistence gate. The upstream v1/kv_offload
        # OffloadingManager assumes exclusive block residency. Letting
        # PN95 also touch / demote blocks would race the upstream
        # prepare_load / complete_store handshake. Detect and skip.
        upstream = _detect_upstream_offload_connector(cfg)
        if upstream is not None:
            log.warning(
                "[PN95] upstream KV offload connector detected (%s) — "
                "skipping PN95 install to avoid double-managing blocks. "
                "Disable the upstream connector or unset "
                "GENESIS_ENABLE_PN95_TIER_AWARE_CACHE to silence this.",
                upstream,
            )
            return False
        try:
            from vllm.sndr_core.cache.tier_manager import make_tier_manager
            new_tm = make_tier_manager(cfg)
        except Exception as e:
            log.warning("[PN95] init_from_config: import/build failed: %s", e)
            return False
        if new_tm is None:
            return False
        if _TM is not None and _TM is not new_tm:
            log.warning("[PN95] replacing existing TierManager singleton")
        _TM = new_tm
        log.info("[PN95] TierManager installed: %s", _TM.stats())
        return True


def _mm_block_overlap_set(
    mm_features: Any,
    block_range: range,
    block_size: int,
) -> set[int]:
    """Day 5 (UNIFIED_CONFIG plan 2026-05-09): per-block MM tagging.

    Given a list of MM features (each with a `mm_position.offset/length`
    placeholder range in token space) and a block_idx range, return the
    set of block indices whose token span overlaps any MM range.

    Defensive: if mm_features is None/empty/malformed → returns empty set.
    Callers should check `not result` before iterating.
    """
    if not mm_features or block_size <= 0:
        return set()
    # Build (start_tok, end_tok) tuples for every MM feature
    mm_ranges: list[tuple[int, int]] = []
    for f in mm_features:
        try:
            pos = getattr(f, "mm_position", None)
            if pos is None:
                continue
            offset = int(getattr(pos, "offset", 0))
            length = int(getattr(pos, "length", 0))
            if length > 0:
                mm_ranges.append((offset, offset + length))
        except Exception:
            continue
    if not mm_ranges:
        return set()
    # Per-block overlap test
    out: set[int] = set()
    for blk_idx in block_range:
        blk_start = blk_idx * block_size
        blk_end = blk_start + block_size
        for mm_start, mm_end in mm_ranges:
            # Half-open intervals: overlap iff
            #   blk_start < mm_end AND mm_start < blk_end
            if blk_start < mm_end and mm_start < blk_end:
                out.add(blk_idx)
                break
    return out


def notify_admit(request: Any, prev_n_cached: int, new_n_cached: int,
                 group_id: int, block_size: int = 0) -> None:
    """Hook called from the cache_blocks() text-patch.

    `request` is a vllm Request; `prev_n_cached`/`new_n_cached` are the
    block index range that just got cached (newly_cached =
    range(prev_n_cached, new_n_cached)). `group_id` is the KV cache
    group id for the manager that produced these blocks. `block_size`
    is the manager's per-block token count — required for Day 5
    per-block MM tagging.

    Day 5: per-block mm_origin computed from `request.mm_features` (the
    list of `MultiModalFeatureSpec` objects, each carrying
    `mm_position: PlaceholderRange(offset, length)`). Falls back to
    coarse `has_mm_input` boolean when block_size is 0 or mm_features
    is missing (callers from older patch versions get a clean degrade).
    """
    if _TM is None:
        return
    try:
        gid_str = f"g{group_id}"
        rid = getattr(request, "request_id", None) or getattr(
            request, "id", None) or "unknown"
        blk_range = range(prev_n_cached, new_n_cached)

        # Day 5 fast-path: real per-block MM tagging if data available
        mm_block_set: set[int] = set()
        mm_features = getattr(request, "mm_features", None)
        if mm_features and block_size > 0:
            mm_block_set = _mm_block_overlap_set(
                mm_features, blk_range, block_size)
        else:
            # Coarse fallback (skeleton behavior — whole request marked
            # mm_origin if any MM input present)
            coarse_mm = bool(getattr(request, "has_mm_input", False)
                              or getattr(request, "mm_inputs", None)
                              or getattr(request, "multi_modal_inputs", None))
            if coarse_mm:
                mm_block_set = set(blk_range)

        for blk_idx in blk_range:
            key = (rid, gid_str, blk_idx)
            _TM.admit(key, mm_origin=(blk_idx in mm_block_set),
                       group_id=gid_str)

        # Auto-warm L1 from L2/disk for predicted-near neighbors. The admit
        # call just observed a real prefix-cache event, which is the cheapest
        # signal we have that this request stream will keep traversing the
        # adjacent block_hashes. We pull the trailing N entries from
        # _admit_order — those are the freshest hits, most likely co-locality
        # candidates — and ask pn95_prefetch_blocks to move them L2->L1.
        # Pure host-side memcpy; no GPU touch. Skipped when env-gated off.
        if _pn95_prefetch_neighbors_enabled():
            window = _pn95_prefetch_window()
            if window > 0:
                try:
                    tail = _TM._admit_order[-window:]
                    if tail:
                        pn95_prefetch_blocks(list(tail))
                except Exception:
                    pass
    except Exception as e:  # pragma: no cover — defensive
        log.warning("[PN95] notify_admit failed silently: %s", e)


def _pn95_prefetch_neighbors_enabled() -> bool:
    """Env gate: GENESIS_PN95_PREFETCH_NEIGHBORS=1 enables auto-warm.

    When a vllm prefix-cache hit lands in notify_touch, we know the
    request will likely traverse adjacent block_hashes next. If any of
    those neighbors are in L2 / disk (not yet in L1 pinned), pre-warm
    them into the pinned pool so the next promote_on_miss is fast.
    Default OFF — operators turn on when running multi-stream workloads
    where adjacency matters more than one-shot.
    """
    return os.environ.get(
        "GENESIS_PN95_PREFETCH_NEIGHBORS", "0",
    ).strip().lower() in ("1", "true", "yes", "on")


def _pn95_prefetch_window() -> int:
    """How many trailing block_hashes from the admit-order tail to pre-warm
    around a touch event. Default 8 — small enough that the warm-up cost
    stays under typical attention compute time.
    """
    try:
        return max(0, min(64, int(os.environ.get(
            "GENESIS_PN95_PREFETCH_WINDOW", "8"))))
    except (ValueError, TypeError):
        return 8


def notify_touch(block_hash: Any, group_ids: list,
                 cached_blocks: Optional[list]) -> None:
    """Hook called from the get_cached_block() text-patch.

    Records that `block_hash` was hit. The skeleton just records via
    the TierManager.touch(); promote-on-hit logic stays inside the
    manager (returns demoted bytes on tier-1 hit; caller promotes).

    For the skeleton we don't actually do GPU promotion since that
    requires a real cuda buffer reference — Day 7 (live integration)
    swaps in the real promote path.
    """
    if _TM is None:
        return
    try:
        # We don't have the (request, group_idx, block_idx) triple at
        # this site; instead use the block_hash as the key. The Day 5
        # plumbing canonicalizes (admit uses one key shape, touch
        # uses another) — for skeleton we record the touch by hash.
        # When a tier-aware system is fully wired, admit + touch
        # share the same key namespace via canonical_block_key().
        key = ("h", block_hash) if not isinstance(block_hash, tuple) \
            else block_hash
        # Best-effort: TierManager.touch returns bytes if demoted.
        # In the skeleton the caller can't do anything with bytes;
        # just record and discard.
        _TM.touch(key)
    except Exception as e:  # pragma: no cover — defensive
        log.warning("[PN95] notify_touch failed silently: %s", e)


def register_kv_caches(kv_caches: Any, kv_cache_groups: Any) -> int:
    """Path C v1.0 Phase 1 (UNIFIED_CONFIG plan 2026-05-09): bridge from
    vLLM worker-level GPU tensor refs to the TierManager.

    Called from the 4th PN95 text-patch in `gpu_model_runner.py`
    immediately after `kv_caches = self.initialize_kv_cache_tensors(...)`.

    `kv_caches` is the vLLM worker's per-layer KV tensor list (or dict)
    — typically `dict[layer_name, Tensor]` or `list[Tensor]`. Each
    tensor has shape `(2, num_blocks, block_size, num_kv_heads, head_dim)`
    for attention layers, or `(num_blocks, conv_state_dim, ...)` for
    Mamba SSM layers (which we already exclude via Day 6).

    Phase 1 records the shape + tensor refs into TierManager metadata
    so Phase 2 can later cudaMemcpyAsync slices to/from CPU pinned slots.
    Phase 1 is observability-only — no actual copies happen yet.

    Returns the count of layer tensors successfully registered.
    Fail-silent: never raises.
    """
    global _TM
    # DEBUG sentinel for live verification
    try:
        with open("/tmp/pn95_init_called.log", "a") as fh:
            import os as _os
            shape_repr = (
                f"dict[{len(kv_caches)}]" if isinstance(kv_caches, dict)
                else f"list[{len(kv_caches)}]" if isinstance(kv_caches, (list, tuple))
                else type(kv_caches).__name__
            )
            fh.write(
                f"[{_os.getpid()}] register_kv_caches called: kv_caches={shape_repr} "
                f"enabled={_enabled()} tm={'set' if _TM else 'None'}\n"
            )
    except Exception:
        pass
    log.warning(
        "[PN95 v1.0] register_kv_caches called: PN95 enabled=%s, "
        "TierManager=%s",
        _enabled(), "installed" if _TM else "None",
    )
    if not _enabled():
        return 0
    # Lazy install of singleton if missing — workers spawn fresh Python
    # so the EngineCore-side init from init_mamba_exclusions doesn't
    # propagate. Re-do it here from the same env var.
    if _TM is None:
        cfg_key = os.environ.get("GENESIS_PN95_CONFIG_KEY", "").strip()
        if cfg_key:
            try:
                from vllm.sndr_core.model_configs.registry import get
                cfg = get(cfg_key)
                if cfg is not None:
                    init_from_config(cfg)
            except Exception as e:
                log.warning(
                    "[PN95 v1.0] register_kv_caches lazy-init failed: %s", e,
                )
    if _TM is None:
        return 0
    try:
        # vLLM stores kv_caches in different shapes depending on version.
        # Common shapes:
        #   - list[torch.Tensor]: indexed by layer index
        #   - dict[str, torch.Tensor]: keyed by layer name
        # We handle both.
        # Phase 2 (UNIFIED_CONFIG plan 2026-05-09): vllm dev93 stores
        # per-layer KV caches in two distinct shapes:
        #   - Attention layers (`*self_attn.attn`): bare torch.Tensor
        #     of shape (num_blocks, block_size, K_or_V, packed_features)
        #     dtype=uint8 (TQ packed) — ELIGIBLE for demote.
        #   - Mamba/linear_attn layers: list[2 torch.Tensor]
        #     of shape (num_blocks, hidden_dim, conv_state_dim) fp16 —
        #     EXCLUDE from demote (SSM state stays GPU-resident).
        #
        # We register both shapes for observability but only attention
        # layers get the per-layer view registry that demote_block()
        # uses. Mamba layers are tracked by group_id only.
        n_registered = 0
        n_attention_eligible = 0
        per_layer_meta: dict = {}
        # Per-attention-layer view registry: {layer_name: {tensor, num_blocks, bytes_per_block}}
        attention_views: dict = {}

        if isinstance(kv_caches, dict):
            iterable = kv_caches.items()
        elif isinstance(kv_caches, (list, tuple)):
            iterable = enumerate(kv_caches)
        else:
            log.warning(
                "[PN95 v1.0] register_kv_caches: unrecognized kv_caches "
                "shape %s — skipping", type(kv_caches).__name__,
            )
            return 0

        for layer_id, val in iterable:
            try:
                layer_key = str(layer_id)
                # Mamba/linear_attn = list[Tensor]
                if isinstance(val, (list, tuple)):
                    inner_shapes = []
                    for t in val:
                        if hasattr(t, "shape"):
                            inner_shapes.append(tuple(t.shape))
                    per_layer_meta[layer_key] = {
                        "kind": "mamba_list",
                        "n_inner": len(val),
                        "inner_shapes": inner_shapes,
                        "demote_eligible": False,
                    }
                    n_registered += 1
                    continue
                # Attention bare Tensor — Phase 2 demote target
                shape = tuple(getattr(val, "shape", ()))
                dtype = str(getattr(val, "dtype", "?"))
                device = str(getattr(val, "device", "?"))
                if not shape or len(shape) < 2:
                    per_layer_meta[layer_key] = {
                        "kind": "unknown",
                        "shape": shape, "demote_eligible": False,
                    }
                    n_registered += 1
                    continue
                # Convention from dev93: shape[0] = num_blocks (TQ k8v4)
                num_blocks = int(shape[0])
                # Per-block byte size = product of remaining dims × elem_size
                elem_size = getattr(val, "element_size", lambda: 1)()
                tail_elems = 1
                for d in shape[1:]:
                    tail_elems *= int(d)
                bytes_per_block = tail_elems * elem_size
                per_layer_meta[layer_key] = {
                    "kind": "attention_tensor",
                    "shape": shape, "dtype": dtype, "device": device,
                    "num_blocks": num_blocks,
                    "bytes_per_block": bytes_per_block,
                    "demote_eligible": True,
                }
                # Stash the live tensor ref for demote_block / promote_block
                attention_views[layer_key] = {
                    "tensor": val,
                    "num_blocks": num_blocks,
                    "bytes_per_block": bytes_per_block,
                    "device": str(device),
                }
                n_registered += 1
                n_attention_eligible += 1
            except Exception as e:
                log.warning(
                    "[PN95 v1.0] register_kv_caches: layer %s failed: %s",
                    layer_id, e,
                )

        # Stash on the TierManager for Phase 2 demote/promote bridge
        _TM._kv_caches_ref = kv_caches  # type: ignore[attr-defined]
        _TM._kv_caches_meta = per_layer_meta  # type: ignore[attr-defined]
        _TM._attention_views = attention_views  # type: ignore[attr-defined]
        log.warning(
            "[PN95 v1.0] register_kv_caches: %d layers (mamba+attn), "
            "%d attention layers eligible for demote",
            n_registered, n_attention_eligible,
        )
        # Sentinel for live integration verification — RICH dump of
        # actual structure (Phase 2 inspection): we need to know what
        # vllm dev93 puts in kv_caches[layer_name] since shape came
        # back () in v1.0 Phase 1.
        try:
            with open("/tmp/pn95_init_called.log", "a") as fh:
                fh.write(f"  → registered {n_registered} layers\n")
                # Dump first 2 entries with FULL introspection
                # Pick samples: 2 mamba layers + 2 attention layers
                items_iter = []
                if isinstance(kv_caches, dict):
                    all_items = list(kv_caches.items())
                    mamba_items = [(k, v) for k, v in all_items
                                    if "linear_attn" in k][:2]
                    attn_items = [(k, v) for k, v in all_items
                                   if "self_attn" in k or "attn.attn" in k][:2]
                    items_iter = mamba_items + attn_items
                    if not items_iter:
                        items_iter = all_items[:2]
                else:
                    items_iter = list(enumerate(kv_caches))[:2]
                for key, val in items_iter:
                    fh.write(f"    [{key}] type={type(val).__name__}\n")
                    if hasattr(val, "shape"):
                        fh.write(f"      shape={tuple(val.shape)}\n")
                    if hasattr(val, "dtype"):
                        fh.write(f"      dtype={val.dtype}\n")
                    if hasattr(val, "device"):
                        fh.write(f"      device={val.device}\n")
                    # Show available attrs (filter to non-dunder)
                    attrs = [a for a in dir(val) if not a.startswith("_")][:25]
                    fh.write(f"      attrs(first 25): {attrs}\n")
                    # If it's a list/tuple/dict-like, dig 1 level deeper
                    if isinstance(val, (list, tuple)) and len(val) > 0:
                        fh.write(f"      (list[{len(val)}] of {type(val[0]).__name__})\n")
                        if hasattr(val[0], "shape"):
                            fh.write(f"      [0].shape={tuple(val[0].shape)}\n")
                            fh.write(f"      [0].dtype={val[0].dtype}\n")
                    elif isinstance(val, dict) and val:
                        first_k = next(iter(val))
                        fh.write(f"      (dict[{len(val)}], first key={first_k!r}, val type={type(val[first_k]).__name__})\n")
        except Exception as e:
            try:
                with open("/tmp/pn95_init_called.log", "a") as fh:
                    fh.write(f"    SENTINEL DUMP FAILED: {e}\n")
            except Exception:
                pass
        return n_registered
    except Exception as e:  # pragma: no cover — defensive
        log.warning("[PN95 v1.0] register_kv_caches failed silently: %s", e)
        return 0


def init_mamba_exclusions_from_kv_groups(kv_cache_groups: Any) -> int:
    """Day 6 (UNIFIED_CONFIG plan 2026-05-09): walk KVCacheGroupSpec list,
    register every MambaSpec group as excluded from demotion.

    Returns the count of groups marked excluded. Idempotent (safe to
    re-call). Fail-silent: never raises — all errors logged + swallowed.

    Called from the PN95 text-patch in `KVCacheManager.__init__`. ALSO
    triggers lazy TierManager init from env (`GENESIS_PN95_CONFIG_KEY`)
    if no manager has been installed yet — so workers spawned with
    `VLLM_WORKER_MULTIPROC_METHOD=spawn` get the singleton on first use.
    """
    n_groups = len(list(kv_cache_groups or []))
    # DEBUG sentinel — writes to /tmp to prove the hook fired
    try:
        with open("/tmp/pn95_init_called.log", "a") as fh:
            import os as _os
            fh.write(
                f"[{_os.getpid()}] init_mamba called n_groups={n_groups} "
                f"enabled={_enabled()}\n"
            )
    except Exception:
        pass
    log.warning(
        "[PN95] init_mamba_exclusions_from_kv_groups called: %d groups, "
        "PN95 enabled=%s",
        n_groups, _enabled(),
    )
    if not _enabled():
        return 0
    try:
        # Lazy install of singleton if missing — read config from env.
        global _TM
        if _TM is None:
            cfg_key = os.environ.get("GENESIS_PN95_CONFIG_KEY", "").strip()
            log.info("[PN95] lazy init: cfg_key=%r", cfg_key)
            if cfg_key:
                try:
                    from vllm.sndr_core.model_configs.registry import get
                    cfg = get(cfg_key)
                    if cfg is not None:
                        init_from_config(cfg)
                        log.info("[PN95] singleton installed: %s",
                                 _TM.stats() if _TM else "FAILED")
                except Exception as e:
                    log.warning(
                        "[PN95] lazy init from GENESIS_PN95_CONFIG_KEY=%s "
                        "failed: %s", cfg_key, e,
                    )

        if _TM is None:
            return 0

        n_excluded = 0
        for idx, group in enumerate(kv_cache_groups or []):
            spec = getattr(group, "kv_cache_spec", None)
            cls_name = type(spec).__name__ if spec is not None else "<None>"
            log.warning(
                "[PN95] group %d: spec_class=%s layers=%s",
                idx, cls_name, getattr(group, "layer_names", "?"),
            )
            if spec is None:
                continue
            # Detect MambaSpec by name + check known mamba-spec classes
            # in case vllm renamed (Mamba2Spec, ShortConvSpec, etc.)
            mamba_class_names = (
                "MambaSpec", "Mamba2Spec", "ShortConvSpec",
                "GdnAttentionSpec", "MambaAttentionSpec",
            )
            if cls_name in mamba_class_names:
                gid = f"g{idx}"
                _TM.register_mamba_excluded(gid)
                n_excluded += 1
                log.warning(
                    "[PN95] excluding %s group %s (layers=%s) from demotion",
                    cls_name, gid, getattr(group, "layer_names", "?"),
                )

        if n_excluded > 0:
            log.info(
                "[PN95] Mamba exclusion init complete — %d groups excluded "
                "out of %d total. TierManager stats: %s",
                n_excluded, len(list(kv_cache_groups or [])), _TM.stats(),
            )
        return n_excluded
    except Exception as e:  # pragma: no cover — defensive
        log.warning("[PN95] init_mamba_exclusions failed silently: %s", e)
        return 0


_TICK_COUNTER = 0
_TICK_LAST_FREE_MIB = 0
# Path C v1.0 Phase 3 — observability counters
_PN95_STATS = {
    "ticks_total": 0,
    "ticks_pressure_check": 0,
    "ticks_demote_triggered": 0,
    "blocks_demoted_total": 0,
    "blocks_promoted_total": 0,
    "last_free_mib": 0,
    "last_demote_count": 0,
}
# Cache config envs — read once at module init, not on every tick
# (was causing measurable overhead per call). Override via reset_env_cache().
_TICK_EVERY_CACHED: Optional[int] = None
_THRESHOLD_CACHED: Optional[int] = None
_DEMOTE_BATCH_CACHED: Optional[int] = None
_FREE_MIB_CACHE_TTL: int = 5  # cache mem_get_info for N consecutive ticks
_FREE_MIB_CACHE_VALID: int = 0


def _read_env_int(name: str, default: int) -> int:
    """Read int env var with fail-silent fallback."""
    try:
        return int(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        return default


def _refresh_env_cache() -> None:
    """Re-read env vars into module-local cache. Called once on first tick."""
    global _TICK_EVERY_CACHED, _THRESHOLD_CACHED, _DEMOTE_BATCH_CACHED
    # Path C Phase 3 default: TICK_EVERY=10 (was 100 — too slow for single-stream
    # workloads where Scheduler.schedule() fires only ~30 times per long request).
    _TICK_EVERY_CACHED = max(1, _read_env_int("GENESIS_PN95_TICK_EVERY", 10))
    _THRESHOLD_CACHED = _read_env_int("GENESIS_PN95_DEMOTE_FREE_MIB_THRESHOLD", 2048)
    _DEMOTE_BATCH_CACHED = max(1, _read_env_int("GENESIS_PN95_DEMOTE_BATCH", 8))


def _gpu_free_mib() -> int:
    """Best-effort: returns GPU 0 free VRAM in MiB. 0 if torch/cuda missing.

    Note: torch.cuda.mem_get_info costs ~800-1200 μs per call (cudaMemGetInfo
    syscall round-trip). Caller responsible for caching across multiple ticks
    via _FREE_MIB_CACHE_VALID counter.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 0
        free, _total = torch.cuda.mem_get_info(0)
        return free // (1 << 20)
    except Exception:
        return 0


def get_pn95_stats() -> dict:
    """Path C v1.0 Phase 3 observability — returns counter snapshot.

    Used by the `sndr report` CLI and operator-facing tools to surface
    PN95 activity without needing access to the EngineCore worker process.
    """
    snapshot = dict(_PN95_STATS)
    # Phase 4 — augment with prefix-store stats
    snapshot["prefix_store_entries"] = len(_PN95_PREFIX_STORE)
    snapshot["prefix_store_bytes_used"] = _PN95_PREFIX_STORE_BYTES_USED
    snapshot["prefix_store_promote_hits"] = _PN95_STATS.get(
        "prefix_promote_hits", 0
    )
    snapshot["prefix_store_demotes"] = _PN95_STATS.get(
        "prefix_demote_count", 0
    )
    # A1 — compression stats
    raw = _PN95_STATS.get("compress_raw_bytes_total", 0)
    stored = _PN95_STATS.get("compress_stored_bytes_total", 0)
    snapshot["compress_raw_bytes_total"] = raw
    snapshot["compress_stored_bytes_total"] = stored
    snapshot["compress_ratio"] = round(raw / stored, 3) if stored > 0 else 1.0
    snapshot["compress_lib"] = _PN95_COMPRESS_LIB or "uninit"
    # B1 — async stream stats
    snapshot["async_stream_enabled"] = _pn95_async_enabled()
    snapshot["async_demote_count"] = _PN95_STATS.get("async_demote_count", 0)
    snapshot["async_promote_count"] = _PN95_STATS.get("async_promote_count", 0)
    # B2 — batched demote ops (each batch processes N layers with 1 sync)
    snapshot["async_batch_demote_count"] = _PN95_STATS.get(
        "async_batch_demote_count", 0
    )
    # B3 — batched promote ops (each batch processes N layers с 1 wait_stream)
    snapshot["async_batch_promote_count"] = _PN95_STATS.get(
        "async_batch_promote_count", 0
    )
    # OBS1 — hit rate calculation для operator monitoring
    lookups_total = _PN95_STATS.get("prefix_lookups_total", 0)
    cold_misses = _PN95_STATS.get("prefix_lookups_cold_miss", 0)
    snapshot["prefix_lookups_total"] = lookups_total
    snapshot["prefix_lookups_cold_miss"] = cold_misses
    snapshot["prefix_hit_rate"] = (
        round((lookups_total - cold_misses) / lookups_total, 3)
        if lookups_total > 0 else 0.0
    )
    # Prefetch API stats — visibility into batched warm-up activity.
    # If prefetch_calls > 0, L2→L1 hits/misses indicate whether the
    # caller is correctly predicting near-future block accesses.
    for k, v in _PN95_PREFETCH_STATS.items():
        snapshot[k] = v

    # Layer-aware demote priority — top-5 hottest layer access counts.
    # Useful for diagnosing whether some attention layers monopolize the
    # promote path (which justifies more aggressive cold-layer demote)
    # or whether access is uniform (then ordering is a no-op).
    snapshot["layer_aware_demote_enabled"] = _pn95_layer_aware_enabled()
    if _PN95_LAYER_ACCESS_COUNTS:
        top_hot = sorted(
            _PN95_LAYER_ACCESS_COUNTS.items(), key=lambda kv: -kv[1],
        )[:5]
        snapshot["layer_access_top5_hot"] = {k: v for k, v in top_hot}
        snapshot["layer_access_distinct"] = len(_PN95_LAYER_ACCESS_COUNTS)
        snapshot["layer_access_total_observations"] = sum(
            _PN95_LAYER_ACCESS_COUNTS.values()
        )
    # L1 pinned pool stats — surfaces hit-rate of the fast PCIe DMA path
    # vs the pageable L2 fallback. Operator can spot "pool too small" via
    # high l1_full_skips, or "no demote pressure" via zero l1_demote_writes.
    snapshot["l1_demote_writes"] = _PN95_STATS.get("l1_demote_writes", 0)
    snapshot["l1_promote_hits"] = _PN95_STATS.get("l1_promote_hits", 0)
    try:
        pool = _pn95_l1_pool()
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
    """OBS1 — periodic stats dump к JSON file для operator visibility.

    Called from scheduler_tick. Throttled by tick counter и env-gated.
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
        if interval <= 0 or _TICK_COUNTER % interval != 0:
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


# ─── Path C v1.0 Phase 4 — prefix-cache extension to CPU pinned RAM ──────
#
# Strategy: instead of demoting ARBITRARY GPU blocks (race-prone), we
# intercept exactly two BlockPool events that are already safe:
#
#   1. demote_on_evict — called from `_maybe_evict_cached_block` AFTER
#      the block has been removed from `cached_block_hash_to_block` and
#      BEFORE `block.reset_hash()`. At this moment the block has ref_cnt=0
#      (no readers), is not in any active `block_table`, and vllm is
#      about to recycle the GPU slot. We safely copy the bytes to CPU.
#
#   2. promote_on_miss — called from `get_cached_block` when vllm's own
#      lookup returned None (cache miss). We check our CPU store; if the
#      hash is there, we allocate a fresh GPU block via `get_new_blocks(1)`,
#      copy CPU→GPU, re-insert into vllm's prefix cache, and return it.
#      vllm sees a normal cache hit — no further changes needed.
#
# Effect: prefix cache effective capacity = N_gpu_blocks + N_cpu_entries
# Multi-turn / long-history workloads see dramatically higher hit rate
# without any CUDA OOM risk and without any hot-path overhead (no polling,
# no per-tick mem_get_info — the path only fires on actual eviction events).
#
# Compatible with hybrid-GDN models (Mamba SSM groups never enter the
# prefix cache to begin with — only attention groups have block hashes).
# Compatible with TP=2+ — each worker has its own _PN95_PREFIX_STORE
# scoped to that worker's GPU.

from collections import OrderedDict as _OrderedDict  # noqa: E402  — block-local import after PN95 section header
_PN95_PREFIX_STORE: "_OrderedDict[Any, list]" = _OrderedDict()
_PN95_PREFIX_STORE_BYTES_USED: int = 0
_PN95_PREFIX_STORE_MAX_BYTES_CACHED: Optional[int] = None
_PN95_BLOCK_POOL_REFS: list = []
# Lock protecting concurrent writers to _PN95_PREFIX_STORE +
# _PN95_PREFIX_STORE_BYTES_USED. Multiple paths can mutate the store:
# demote_on_evict (scheduler thread), prefetch_blocks (prefetch worker
# thread), _prefix_store_evict_until_fit (called recursively from demote).
# Pre-PN95 the dict was single-threaded so a lock would have been overhead;
# with the new prefetch API + worker_side_proactive_demote we explicitly
# advertise thread-safety, so the lock is required (review finding #12).
_PN95_PREFIX_STORE_LOCK: threading.Lock = threading.Lock()


# ── L1 pinned host cache (optional, gated by GENESIS_ENABLE_PN95_PINNED_POOL).
# Held in a separate module (_pn95_pinned_pool) so the heavy import (torch
# pin_memory) doesn't run at sndr_core boot when the feature is OFF.
#
# Layer payload (list of (layer_name, bytes)) is serialized to a single bytes
# blob via pickle.HIGHEST_PROTOCOL before being placed in the pool — the pool
# itself works on byte slabs of equal slot size. Unpack reverses pickle.
# Pickle overhead is ~5-10 μs per blob, dwarfed by the PCIe transfer savings
# from non-pageable memory (3-5 GB/s pinned vs ~600 MB/s pageable bounce).
def _pn95_pack_layer_data(layer_data: list) -> bytes:
    import pickle
    return pickle.dumps(layer_data, protocol=pickle.HIGHEST_PROTOCOL)


def _pn95_unpack_layer_data(blob: bytes) -> Optional[list]:
    """Unpickle the layer-data blob from a pinned-pool slot or disk tier.

    Uses a strict allow-list of class lookups (review finding #16):
    a corrupted slot or a maliciously-crafted disk file MUST NOT be able
    to invoke arbitrary code via pickle's `__reduce__` / `find_class`.
    KV payloads are pure (str, bytes) tuples in a list, no custom classes
    needed; everything else is rejected.
    """
    if not blob:
        return None
    import io
    import pickle

    class _PN95SafeUnpickler(pickle.Unpickler):
        _ALLOWED = frozenset({
            ("builtins", "str"),
            ("builtins", "bytes"),
            ("builtins", "list"),
            ("builtins", "tuple"),
            ("builtins", "int"),
            ("builtins", "float"),
            ("builtins", "bool"),
            ("builtins", "dict"),
            ("builtins", "NoneType"),
        })

        def find_class(self, module: str, name: str):
            if (module, name) in self._ALLOWED:
                return super().find_class(module, name)
            raise pickle.UnpicklingError(
                f"[PN95] pickle class not allow-listed: {module}.{name}"
            )

    try:
        obj = _PN95SafeUnpickler(io.BytesIO(blob)).load()
    except (pickle.UnpicklingError, EOFError, TypeError, ValueError):
        return None
    return obj if isinstance(obj, list) else None


def _pn95_l1_pool(slot_size_hint: int = 0):
    """Return the singleton pinned pool, or None when disabled / alloc-failed.

    Safe to call from hot paths — returns fast None when feature OFF.
    """
    try:
        from vllm.sndr_core.cache import _pn95_pinned_pool as _ppool
    except ImportError:
        return None
    return _ppool.get_pool(slot_size_hint)


# ── Prefetch / warmup API ───────────────────────────────────────────────
# Inspired by SGLang HiCache layer-by-layer prefetch overlap: the engine
# tells PN95 which block_hashes are about to be needed; PN95 warms up the
# fast L1 pinned pool from the slow L2 OrderedDict (or, if not in L2, the
# disk tier) so the actual `promote_on_miss` call lands in L1.
#
# Without prefetch the path on a cold block is:
#   promote_on_miss → L2 OrderedDict.get → numpy.frombuffer → torch.from_numpy
#   → .to(cuda, non_blocking=True from pageable mem) → bounce-buffer copy
#   ~400 μs for a 32 KB block (single attention layer's K+V for one block).
#
# With prefetch the L1 slot is already pinned by the time vllm calls
# promote_on_miss; the GPU read is a single pinned-host DMA at PCIe Gen4
# line rate, ~80 μs.
#
# Stats track hits/misses so operators can see whether prefetch is paying
# off (vs raw L1 demote-side fills).
_PN95_PREFETCH_STATS = {
    "prefetch_calls": 0,
    "prefetch_block_hashes": 0,
    "prefetch_l2_hits_promoted": 0,  # L2 entry copied into L1 pinned
    "prefetch_l2_already_in_l1": 0,  # L1 already warm — no-op
    "prefetch_missing": 0,           # not in L2 or disk — nothing to do
    "prefetch_disk_hits_promoted": 0,
    "prefetch_pool_full_skips": 0,
}


def pn95_prefetch_blocks(block_hashes: list) -> dict:
    """Warm up the L1 pinned pool for the given block_hashes BEFORE the
    engine needs them. Returns a stats delta dict.

    Strategy per block_hash:
      1. If L1 already has it → no-op (count as already-warm).
      2. If L2 OrderedDict has it → pack + put into L1 pinned pool.
      3. If disk tier has it → pull from disk into L2 (re-fill RAM cache),
         then pack + put into L1.
      4. If none → record as miss; nothing to do.

    Cheap on the happy path: dict lookup + pickle.dumps + memcpy into
    pinned slot. Steps 2/3 do NOT touch the GPU — pure host-side moves.
    The slow GPU DMA happens later on `promote_on_miss`, but from the
    pinned slot (3-5× faster than pageable bytes).

    Safe to call from any thread; the pool is mutex-protected internally.
    Returns {} when PN95 disabled or pool unavailable.
    """
    if not _enabled():
        return {}
    pool = _pn95_l1_pool()
    if pool is None:
        return {}

    delta = {
        "blocks_warmed_from_l2": 0,
        "blocks_already_warm": 0,
        "blocks_missing": 0,
        "blocks_warmed_from_disk": 0,
        "blocks_pool_full": 0,
    }

    _PN95_PREFETCH_STATS["prefetch_calls"] += 1
    _PN95_PREFETCH_STATS["prefetch_block_hashes"] += len(block_hashes)

    # Lazy disk import — only when we actually need disk fallback below.
    _disk = None

    for h in block_hashes:
        # 1) Already in L1?
        if pool.has(h):
            delta["blocks_already_warm"] += 1
            _PN95_PREFETCH_STATS["prefetch_l2_already_in_l1"] += 1
            continue

        # 2) In L2?
        layer_data = _PN95_PREFIX_STORE.get(h)

        # 3) Else try disk.
        if layer_data is None:
            if _disk is None:
                try:
                    from vllm.sndr_core.cache import _pn95_disk_tier as _disk_mod
                    _disk = _disk_mod
                except ImportError:
                    _disk = False
            if _disk and _disk._enabled():
                try:
                    layer_data = _disk.disk_tier_get(h)
                except Exception:
                    layer_data = None
            if layer_data is not None:
                # Re-insert into L2 so future direct hits stay fast too.
                try:
                    global _PN95_PREFIX_STORE_BYTES_USED
                    total_bytes = sum(len(b) for _n, b in layer_data)
                    if total_bytes <= _prefix_store_max_bytes():
                        with _PN95_PREFIX_STORE_LOCK:
                            _PN95_PREFIX_STORE[h] = layer_data
                            _PN95_PREFIX_STORE_BYTES_USED += total_bytes
                except Exception:
                    pass
                delta["blocks_warmed_from_disk"] += 1
                _PN95_PREFETCH_STATS["prefetch_disk_hits_promoted"] += 1

        if layer_data is None:
            delta["blocks_missing"] += 1
            _PN95_PREFETCH_STATS["prefetch_missing"] += 1
            continue

        # 4) Pack + put into L1 pinned slot.
        try:
            blob = _pn95_pack_layer_data(layer_data)
            if pool.put(h, blob):
                if delta["blocks_warmed_from_disk"]:
                    pass  # already counted as disk
                else:
                    delta["blocks_warmed_from_l2"] += 1
                    _PN95_PREFETCH_STATS["prefetch_l2_hits_promoted"] += 1
            else:
                delta["blocks_pool_full"] += 1
                _PN95_PREFETCH_STATS["prefetch_pool_full_skips"] += 1
        except Exception:
            pass

    return delta


def pn95_get_prefetch_stats() -> dict:
    """Snapshot of prefetch API counters — surfaced via sndr patches pn95-status."""
    return dict(_PN95_PREFETCH_STATS)


# ── Layer-aware demote priority ──────────────────────────────────────────
# Tracks per-layer access frequency from the promote path so demote can
# prioritize cold layers when capacity is constrained. Implementation is a
# small dict keyed by layer_name; on a 17-attention-layer Qwen3.6 27B model
# the structure stays trivial (<200 bytes). Single-process, single-rank —
# no cross-worker sync needed (each rank decides its own demote order).
#
# Update on every promote restoration; read on every demote sort. The
# heuristic is intentionally simple: layers with the highest cumulative
# promote-read count are deemed "hot" and pushed to the end of the demote
# queue. Cold layers (low counts) are demoted first, freeing GPU memory
# along the path the GPU's attention forward least frequently touches.
#
# Bounded growth: counts are reset on overflow (>10M) to prevent integer
# bloat. The relative ordering is what matters, not absolute values.
_PN95_LAYER_ACCESS_COUNTS: dict = {}
_PN95_LAYER_ACCESS_RESET_THRESHOLD = 10_000_000


# ── store_threshold reuse-frequency gate (upstream PR #40020 pattern) ─────
#
# Tracks how many times each block_hash has been *looked up* during
# promote_on_miss. Blocks with hits below GENESIS_PN95_STORE_THRESHOLD
# are NOT demoted on evict — the engine pays no compression/copy cost
# on a block that's about to disappear from the request stream forever.
#
# Inspired by upstream `FilterReusedOffloadingManager` in cpu/manager.py
# (only stores keys observed `store_threshold` times via lookup).
#
# Default off (threshold=0). Operators set >=2 when serving chat workloads
# where most prefill blocks are one-shot.
_PN95_HIT_COUNTS: dict = {}
_PN95_HIT_TRACKER_MAX = 64_000


def _pn95_store_threshold() -> int:
    try:
        return max(0, int(os.environ.get("GENESIS_PN95_STORE_THRESHOLD", "0")))
    except (ValueError, TypeError):
        return 0


def _pn95_record_lookup(block_hash: Any) -> int:
    """Bump hit counter for a block_hash on every promote query. Returns
    post-bump count. Bounded by _PN95_HIT_TRACKER_MAX (LRU eviction)."""
    global _PN95_HIT_COUNTS
    n = _PN95_HIT_COUNTS.get(block_hash, 0) + 1
    if n == 1 and len(_PN95_HIT_COUNTS) >= _PN95_HIT_TRACKER_MAX:
        # FIFO drop — keep tracker bounded, lose oldest counter.
        try:
            oldest = next(iter(_PN95_HIT_COUNTS))
            _PN95_HIT_COUNTS.pop(oldest, None)
        except StopIteration:
            pass
    _PN95_HIT_COUNTS[block_hash] = n
    return n


def _pn95_should_demote(block_hash: Any) -> bool:
    """Apply store_threshold gate: skip demote if block hasn't reached
    threshold lookups yet. Returns True when demote should proceed."""
    thr = _pn95_store_threshold()
    if thr <= 1:
        return True  # default: every block demotes
    return _PN95_HIT_COUNTS.get(block_hash, 0) >= thr


# ── block_size_factor — PCIe transaction amortization ────────────────────
#
# Upstream PR #40020 lets the offload layer operate on `block_size_factor`
# adjacent KV blocks as a single super-block. This amortizes the PCIe
# transaction setup cost (~10-20us per DMA submit) over a larger payload,
# critical when each KV block is small (Qwen3.6 fp8 32KB/block).
#
# At factor=4 we batch four ~32KB blocks into one ~128KB transfer:
#   - submit/sync overhead drops 4×
#   - PCIe is more BW-efficient on larger packets (closer to line rate)
#   - tradeoff: the L1 pinned pool slot_size auto-derives from first
#     super-block payload, so 4× larger slots → fewer slots within
#     GENESIS_PN95_PINNED_POOL_MB budget
#
# Default 1 (no grouping). 2-4 typical sweet spots for production.
def _pn95_block_size_factor() -> int:
    try:
        v = int(os.environ.get("GENESIS_PN95_BLOCK_SIZE_FACTOR", "1"))
    except (ValueError, TypeError):
        v = 1
    # Clamp to sensible range; 8+ rarely helps and uses lots of pinned slots.
    return max(1, min(8, v))


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
                _PN95_STATS["store_threshold_skips"] = (
                    _PN95_STATS.get("store_threshold_skips", 0) + 1
                )
                continue
            try:
                if demote_on_evict(block_hash, block_id):
                    n_demoted += 1
            except Exception:
                pass
    if factor > 1:
        _PN95_STATS["super_block_demote_batches"] = (
            _PN95_STATS.get("super_block_demote_batches", 0) + 1
        )
        _PN95_STATS["block_size_factor"] = factor
    return n_demoted


def _pn95_layer_aware_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN95_LAYER_AWARE_DEMOTE", "0",
    ).strip().lower() in ("1", "true", "yes", "on")


def _pn95_record_layer_promote(layer_name: str) -> None:
    """Bump access count for a layer on promote read. Cheap dict op."""
    global _PN95_LAYER_ACCESS_COUNTS
    n = _PN95_LAYER_ACCESS_COUNTS.get(layer_name, 0) + 1
    if n > _PN95_LAYER_ACCESS_RESET_THRESHOLD:
        # Halve all counters to preserve relative ordering without overflow.
        _PN95_LAYER_ACCESS_COUNTS = {
            k: v // 2 for k, v in _PN95_LAYER_ACCESS_COUNTS.items()
        }
        n = _PN95_LAYER_ACCESS_COUNTS.get(layer_name, 0) + 1
    _PN95_LAYER_ACCESS_COUNTS[layer_name] = n


def _pn95_sort_layers_cold_first(eligible_layers: list) -> list:
    """Sort (layer_name, tensor_view) tuples by ascending access count.

    Layers never observed in promote stay at the front (cold by default).
    Stable sort preserves the original block-pool ordering as the tiebreaker
    so behavior is deterministic when no promote history exists.

    No-op if GENESIS_ENABLE_PN95_LAYER_AWARE_DEMOTE != 1.
    """
    if not _pn95_layer_aware_enabled() or not _PN95_LAYER_ACCESS_COUNTS:
        return eligible_layers
    return sorted(
        eligible_layers,
        key=lambda lv: _PN95_LAYER_ACCESS_COUNTS.get(lv[0], 0),
    )

# Path C v1.0 Quality-First Sprint Q1 A1 — lossless CPU prefix compression.
# Reduces effective CPU tier capacity 2-3× via zstd (or 1.5-2× via lz4).
# Detection at decompress is via magic bytes — no per-entry header overhead.
# Quality: 100% (lossless by construction).
_PN95_COMPRESS_LIB: Optional[str] = None  # 'zstd'|'lz4'|'zlib'|'none'|None
_PN95_COMPRESS_LEVEL: Optional[int] = None
_PN95_COMPRESS_MIN_BYTES = 256  # entries smaller skip compression (overhead)
# Sprint Q1 B6 — per-thread cached compressor/decompressor instances.
# threading.local ensures each ThreadPool worker has own cached instance
# (avoids race in singleton init AND any potential thread-safety nuance
# of underlying C library context state).
_PN95_ZSTD_TL = threading.local()


def _pn95_init_compression() -> None:
    """Lazy-init compression backend on first use.

    Reads GENESIS_PN95_CPU_COMPRESS env: 'zstd'|'lz4'|'zlib'|'none'|'auto'.
    Default 'auto' = prefer zstd > lz4 > zlib > none.

    GENESIS_PN95_COMPRESS_LEVEL controls compression level:
      zstd: 1-22 (default 3 = balanced speed/ratio)
      zlib: 1-9 (default 1 = fast)
      lz4: ignored (single level)
    """
    global _PN95_COMPRESS_LIB, _PN95_COMPRESS_LEVEL
    if _PN95_COMPRESS_LIB is not None:
        return
    requested = os.environ.get("GENESIS_PN95_CPU_COMPRESS", "auto").strip().lower()
    if requested in ("none", "off", "0", "disabled"):
        _PN95_COMPRESS_LIB = "none"
        return
    # Try zstd first (best ratio + decent speed)
    if requested in ("auto", "zstd"):
        try:
            import zstandard  # noqa: F401
            _PN95_COMPRESS_LIB = "zstd"
            try:
                _PN95_COMPRESS_LEVEL = int(os.environ.get(
                    "GENESIS_PN95_COMPRESS_LEVEL", "3"))
            except (ValueError, TypeError):
                _PN95_COMPRESS_LEVEL = 3
            log.info("[PN95 A1] CPU compression: zstd level=%d",
                     _PN95_COMPRESS_LEVEL)
            return
        except ImportError:
            if requested == "zstd":
                log.warning("[PN95 A1] zstandard not installed, trying lz4")
    # Try lz4 (faster, less compression)
    if requested in ("auto", "lz4"):
        try:
            import lz4.frame  # noqa: F401
            _PN95_COMPRESS_LIB = "lz4"
            log.info("[PN95 A1] CPU compression: lz4")
            return
        except ImportError:
            if requested == "lz4":
                log.warning("[PN95 A1] lz4 not installed, trying zlib")
    # Fallback to stdlib zlib (always available)
    if requested in ("auto", "zlib"):
        _PN95_COMPRESS_LIB = "zlib"
        try:
            _PN95_COMPRESS_LEVEL = int(os.environ.get(
                "GENESIS_PN95_COMPRESS_LEVEL", "1"))
        except (ValueError, TypeError):
            _PN95_COMPRESS_LEVEL = 1
        log.info("[PN95 A1] CPU compression: zlib level=%d (stdlib fallback)",
                 _PN95_COMPRESS_LEVEL)
        return
    _PN95_COMPRESS_LIB = "none"


def _pn95_compress_bytes(data: bytes) -> bytes:
    """Compress bytes via configured backend. Returns compressed OR original
    if compression disabled / failed / no benefit.

    Compression backend writes a magic header (zstd/lz4/zlib all do); the
    symmetric _pn95_decompress_bytes auto-detects via magic check.
    """
    _pn95_init_compression()
    lib = _PN95_COMPRESS_LIB
    if lib in ("none", None):
        return data
    if len(data) < _PN95_COMPRESS_MIN_BYTES:
        return data
    try:
        if lib == "zstd":
            # Sprint Q1 B6 — per-thread cached compressor (avoid alloc per call,
            # avoid race in B4 ThreadPool path).
            cctx = getattr(_PN95_ZSTD_TL, "cctx", None)
            if cctx is None:
                import zstandard as zstd
                cctx = zstd.ZstdCompressor(level=_PN95_COMPRESS_LEVEL or 3)
                _PN95_ZSTD_TL.cctx = cctx
            compressed = cctx.compress(data)
        elif lib == "lz4":
            import lz4.frame
            compressed = lz4.frame.compress(data)
        elif lib == "zlib":
            import zlib
            compressed = zlib.compress(data, _PN95_COMPRESS_LEVEL or 1)
        else:
            return data
    except Exception:
        return data
    # Only use compression if it saved >5% (avoid overhead на already-compressed data)
    if len(compressed) >= int(len(data) * 0.95):
        return data
    return compressed


_PN95_COMPRESS_POOL: Optional[Any] = None  # ThreadPoolExecutor для parallel compress


def _pn95_compress_pool() -> Optional[Any]:
    """Path C v1.0 Sprint Q1 B4 — lazy-init ThreadPoolExecutor для parallel
    compression. zstd/lz4/zlib release GIL during compression — multiple
    threads truly parallel.

    Returns None если threading unavailable (which doesn't happen in CPython).
    Default 4 workers (env GENESIS_PN95_COMPRESS_THREADS).
    """
    global _PN95_COMPRESS_POOL
    if _PN95_COMPRESS_POOL is None:
        try:
            from concurrent.futures import ThreadPoolExecutor
            try:
                workers = int(os.environ.get("GENESIS_PN95_COMPRESS_THREADS", "4"))
            except (ValueError, TypeError):
                workers = 4
            workers = max(1, min(workers, 16))  # clamp [1, 16]
            _PN95_COMPRESS_POOL = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="pn95-compress"
            )
        except Exception:
            return None
    return _PN95_COMPRESS_POOL


def _pn95_compress_bytes_batch(data_list: list) -> list:
    """Path C v1.0 Sprint Q1 B4 — parallel batched compression.

    Compress N bytes objects concurrently через ThreadPool. zstd/lz4/zlib
    release Python GIL during native compression → real parallelism.

    For 17-layer demote with ~100KB blocks: sequential = ~1.7ms total,
    parallel (4 threads) = ~0.5ms total = ~3-4× speedup.

    Returns list of compressed bytes в same order. Empty list if input empty.
    Falls back к sequential если pool unavailable.
    """
    if not data_list:
        return []
    pool = _pn95_compress_pool()
    if pool is None or len(data_list) <= 1:
        return [_pn95_compress_bytes(d) for d in data_list]
    # Parallel — submit all, collect ordered results
    futures = [pool.submit(_pn95_compress_bytes, d) for d in data_list]
    return [f.result() for f in futures]


def _pn95_decompress_bytes_batch(data_list: list) -> list:
    """Path C v1.0 Sprint Q1 B5 — parallel batched decompression.

    Mirror of B4 (_pn95_compress_bytes_batch) для promote path. zstd/lz4/zlib
    release Python GIL during decompression → real parallelism.

    For 17-layer promote with mixed compressed sizes:
    sequential ~340μs total, parallel (4 threads) ~85μs total = ~4× speedup.

    Returns list of decompressed bytes в same order. Backward-compatible:
    uncompressed entries pass through unchanged (auto-detected via magic bytes
    в underlying _pn95_decompress_bytes).
    """
    if not data_list:
        return []
    pool = _pn95_compress_pool()
    if pool is None or len(data_list) <= 1:
        return [_pn95_decompress_bytes(d) for d in data_list]
    # Parallel — submit all, collect ordered results
    futures = [pool.submit(_pn95_decompress_bytes, d) for d in data_list]
    return [f.result() for f in futures]


def _pn95_decompress_bytes(data: bytes) -> bytes:
    """Auto-detect compression via magic bytes and decompress. Returns
    original bytes if no compression detected (backward-compatible —
    handles uncompressed entries from before A1, mixed-format stores).
    """
    if len(data) < 4:
        return data
    # zstd frame magic: 28 b5 2f fd
    if data[:4] == b'\x28\xb5\x2f\xfd':
        try:
            # Sprint Q1 B6 — per-thread cached decompressor.
            dctx = getattr(_PN95_ZSTD_TL, "dctx", None)
            if dctx is None:
                import zstandard as zstd
                dctx = zstd.ZstdDecompressor()
                _PN95_ZSTD_TL.dctx = dctx
            return dctx.decompress(data)
        except Exception:
            return data
    # lz4 frame magic: 04 22 4d 18
    if data[:4] == b'\x04\x22\x4d\x18':
        try:
            import lz4.frame
            return lz4.frame.decompress(data)
        except Exception:
            return data
    # zlib header (RFC 1950): 0x78 (CMF) + check byte (variable)
    # Common values: 0x78 0x01, 0x78 0x5e, 0x78 0x9c, 0x78 0xda
    if data[0] == 0x78 and data[1] in (0x01, 0x5e, 0x9c, 0xda):
        try:
            import zlib
            return zlib.decompress(data)
        except Exception:
            return data
    return data


def _prefix_store_max_bytes() -> int:
    """Read GENESIS_PN95_PREFIX_STORE_GIB env var (default 4 GiB)."""
    global _PN95_PREFIX_STORE_MAX_BYTES_CACHED
    if _PN95_PREFIX_STORE_MAX_BYTES_CACHED is None:
        gib = float(os.environ.get("GENESIS_PN95_PREFIX_STORE_GIB", "4"))
        _PN95_PREFIX_STORE_MAX_BYTES_CACHED = int(gib * (1 << 30))
    return _PN95_PREFIX_STORE_MAX_BYTES_CACHED


def _prefix_store_evict_until_fit(needed_bytes: int) -> None:
    """LRU evict from CPU prefix store until needed_bytes fits.

    When the disk tier (`_pn95_disk_tier`) is enabled, the LRU victim
    is spilled to disk before being dropped from RAM so future
    `promote_on_miss` can still recover the bytes. With the disk tier
    disabled (default) the behaviour matches the legacy implementation
    — victims are discarded.
    """
    global _PN95_PREFIX_STORE_BYTES_USED
    max_bytes = _prefix_store_max_bytes()
    try:
        from vllm.sndr_core.cache import _pn95_disk_tier as _disk
    except ImportError:
        _disk = None
    disk_active = _disk is not None and _disk._enabled()
    while _PN95_PREFIX_STORE_BYTES_USED + needed_bytes > max_bytes:
        if not _PN95_PREFIX_STORE:
            return
        key, layer_data = _PN95_PREFIX_STORE.popitem(last=False)
        freed = sum(len(b) for _name, b in layer_data)
        _PN95_PREFIX_STORE_BYTES_USED -= freed
        # Spillover the evicted entry to the disk tier when enabled.
        # Failure is non-fatal — the victim is then discarded as before.
        if disk_active:
            try:
                if _disk.disk_tier_set(key, layer_data):
                    _PN95_STATS.setdefault("ram_to_disk_spills_total", 0)
                    _PN95_STATS["ram_to_disk_spills_total"] += 1
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
    try:
        if block_pool not in _PN95_BLOCK_POOL_REFS:
            _PN95_BLOCK_POOL_REFS.append(block_pool)
    except Exception:
        pass


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
    if not _enabled() or _TM is None:
        return False
    views = getattr(_TM, "_attention_views", None)
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
        eligible_layers = _pn95_sort_layers_cold_first(eligible_layers)

        # ONE batched async copy для all N layer views (~16× less sync overhead).
        layer_views = [v for _name, v in eligible_layers]
        raw_bytes_list = _pn95_gpu_to_cpu_bytes_batch(layer_views)

        # Sprint Q1 B4 — parallel compression (CPU work parallelizable since
        # zstd/lz4/zlib release GIL during compress). For 17 layers ~3-4× faster.
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
        if block_hash in _PN95_PREFIX_STORE:
            old = _PN95_PREFIX_STORE.pop(block_hash)
            global _PN95_PREFIX_STORE_BYTES_USED
            _PN95_PREFIX_STORE_BYTES_USED -= sum(len(b) for _n, b in old)

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
                _PN95_STATS.setdefault("l1_demote_writes", 0)
                _PN95_STATS["l1_demote_writes"] += 1
        except Exception:
            l1_acquired = False  # L1 refused — proceed L2-only

        try:
            with _PN95_PREFIX_STORE_LOCK:
                _PN95_PREFIX_STORE[block_hash] = layer_data
                _PN95_PREFIX_STORE_BYTES_USED += total_bytes
        except Exception:
            # L2 insert failed — rollback L1 to keep tiers consistent.
            if l1_acquired and l1_pool is not None:
                try:
                    l1_pool.evict(block_hash)
                except Exception:
                    pass
            _PN95_STATS["demote_rollback_count"] = (
                _PN95_STATS.get("demote_rollback_count", 0) + 1
            )
            return False

        _PN95_STATS["prefix_demote_count"] = (
            _PN95_STATS.get("prefix_demote_count", 0) + 1
        )
        # A1 stats — compression ratio tracking
        _PN95_STATS["compress_raw_bytes_total"] = (
            _PN95_STATS.get("compress_raw_bytes_total", 0) + raw_total_bytes
        )
        _PN95_STATS["compress_stored_bytes_total"] = (
            _PN95_STATS.get("compress_stored_bytes_total", 0) + total_bytes
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
    global _PN95_PREFIX_STORE_BYTES_USED
    if not _enabled() or _TM is None:
        return None
    # OBS1 — track all lookup attempts (for hit_rate calculation)
    _PN95_STATS["prefix_lookups_total"] = (
        _PN95_STATS.get("prefix_lookups_total", 0) + 1
    )
    # store_threshold: bump per-hash hit counter so future demote knows
    # this block has been queried (review finding #1 — without this call
    # _pn95_should_demote always returns False and the gate blocks ALL
    # demotes when GENESIS_PN95_STORE_THRESHOLD>=2).
    _pn95_record_lookup(block_hash_with_group_id)

    # L1 pinned pool first — fastest path. If hit, unpack and use directly;
    # bypass the L2 OrderedDict read entirely. L2 still holds the entry for
    # disk-spillover bookkeeping, but the bytes we hand to the GPU come
    # from pinned memory (3-5x faster PCIe DMA).
    layer_data = None
    try:
        pool = _pn95_l1_pool()
        if pool is not None and pool.has(block_hash_with_group_id):
            blob = pool.get_bytes(block_hash_with_group_id)
            layer_data = _pn95_unpack_layer_data(blob)
            if layer_data is not None:
                _PN95_STATS.setdefault("l1_promote_hits", 0)
                _PN95_STATS["l1_promote_hits"] += 1
    except Exception:
        layer_data = None

    if layer_data is None:
        layer_data = _PN95_PREFIX_STORE.get(block_hash_with_group_id)
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
                    _PN95_PREFIX_STORE[block_hash_with_group_id] = layer_data
                    _PN95_PREFIX_STORE_BYTES_USED += total_bytes
                _PN95_STATS.setdefault("disk_to_ram_promotes_total", 0)
                _PN95_STATS["disk_to_ram_promotes_total"] += 1
    if layer_data is None:
        # OBS1 — cold miss: vllm asked, мы тоже не имели данных
        _PN95_STATS["prefix_lookups_cold_miss"] = (
            _PN95_STATS.get("prefix_lookups_cold_miss", 0) + 1
        )
        return None
    try:
        new_blocks = block_pool.get_new_blocks(1)
        if not new_blocks:
            return None
        new_block = new_blocks[0]
        new_block_id = new_block.block_id

        views = getattr(_TM, "_attention_views", None) or {}
        # Sprint Q1 B3 — collect all eligible (view, bytes) pairs, then ONE
        # batched async CPU→GPU copy (single wait_stream vs N).
        #
        # NOTE: Sequential decompress kept (measured B5 parallel slower на 17
        # layers — zstd decompress ~80μs total too fast для ThreadPool overhead).
        # `_pn95_decompress_bytes_batch` available для future bulk warmup
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

        old = _PN95_PREFIX_STORE.pop(block_hash_with_group_id, None)
        if old is not None:
            _PN95_PREFIX_STORE_BYTES_USED -= sum(len(b) for _n, b in old)

        _PN95_STATS["prefix_promote_hits"] = (
            _PN95_STATS.get("prefix_promote_hits", 0) + 1
        )
        _PN95_STATS["blocks_promoted_total"] += 1
        return new_block
    except Exception:
        return None


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
    candidates = []
    if not _PN95_BLOCK_POOL_REFS:
        return candidates

    # Hot ring: last N admit'ов never demote (typically spec-decode K+1
    # targets where the model just placed K speculative tokens). Reading
    # the tail of _admit_order on TM gives us the freshest activity.
    hot_keys = set()
    if _TM is not None:
        ring_size = getattr(_TM, "spec_decode_hot_ring", 0) or 0
        if ring_size > 0:
            try:
                hot_keys = set(_TM._admit_order[-ring_size:])
            except (AttributeError, TypeError):
                hot_keys = set()

    for pool in _PN95_BLOCK_POOL_REFS:
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
                if blk_hash in _PN95_PREFIX_STORE:
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
    if not _enabled() or _TM is None:
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
        ring_size = getattr(_TM, "spec_decode_hot_ring", 0) or 0
        if ring_size > 0:
            hot_keys = set(_TM._admit_order[-ring_size:])
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
            if blk_hash in _PN95_PREFIX_STORE:
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
        _PN95_STATS["blocks_demoted_total"] += captured
        _PN95_STATS["last_demote_count"] = captured
        _PN95_STATS.setdefault("worker_proactive_calls", 0)
        _PN95_STATS["worker_proactive_calls"] += 1
        _PN95_STATS.setdefault("worker_proactive_captured", 0)
        _PN95_STATS["worker_proactive_captured"] += captured
    return captured


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
    candidates = _select_cold_blocks_via_bpool_lru(target_count)
    if not candidates:
        return 0
    n_captured = 0
    for _pool, blk_id, blk_hash in candidates:
        if demote_on_evict(blk_hash, blk_id):
            n_captured += 1
    return n_captured


def scheduler_tick() -> None:
    """Path C v1.0 Phase 4.1 — smart proactive scheduler-tick hook.

    Strategy:
      1. Fast-path early return (~50 ns) when disabled
      2. Throttled by GENESIS_PN95_TICK_EVERY (default 10)
      3. Cached _gpu_free_mib (TTL=5 ticks → amortizes cudaMemGetInfo)
      4. When pressure detected (free < threshold), select COLD cached
         blocks via BlockPool's own LRU queue (head of free_block_queue
         = next-to-evict). These blocks are ref_cnt=0 = no readers =
         safe to copy. Skip hot-ring members (last N spec-decode targets).
      5. demote_on_evict captures bytes BEFORE vllm's own eviction —
         turns vllm's reset_hash into a no-op (bytes already preserved).

    Result: real LRU-based demote instead of dummy block_idx=0. Released
    GPU memory comes from vllm's normal eviction path (no race).

    Fail-silent — never raises into scheduler hot path.
    """
    if not _enabled() or _TM is None:
        return
    global _TICK_COUNTER, _TICK_LAST_FREE_MIB, _FREE_MIB_CACHE_VALID
    _TICK_COUNTER += 1
    _PN95_STATS["ticks_total"] += 1

    # OBS1 — periodic stats dump к JSON file для operator visibility
    # Throttled by GENESIS_PN95_STATS_INTERVAL (default 100 ticks),
    # disabled via GENESIS_PN95_STATS_FILE="" env. Fail-silent.
    _pn95_dump_stats_if_due()

    if _TICK_EVERY_CACHED is None:
        _refresh_env_cache()

    if _TICK_COUNTER % _TICK_EVERY_CACHED != 0:
        return

    _PN95_STATS["ticks_pressure_check"] += 1
    try:
        if _FREE_MIB_CACHE_VALID <= 0:
            free_mib = _gpu_free_mib()
            _TICK_LAST_FREE_MIB = free_mib
            _PN95_STATS["last_free_mib"] = free_mib
            _FREE_MIB_CACHE_VALID = _FREE_MIB_CACHE_TTL
        else:
            free_mib = _TICK_LAST_FREE_MIB
            _FREE_MIB_CACHE_VALID -= 1

        if free_mib <= 0 or free_mib >= _THRESHOLD_CACHED:
            return

        _FREE_MIB_CACHE_VALID = 0

        # [Genesis PN203] cold-prefix offload sweep — Tier 3.A core.
        # Runs BEFORE empty_cache so the demote path can populate L2 (PN95
        # pinned pool) with bytes that would otherwise be discarded.
        # Requires per-layer KV split (PN202) for correctness on hybrid models.
        try:
            pn203_cold_prefix_sweep()
        except Exception:
            pass

        # [Genesis PN201] threshold-gated empty_cache for fragmentation
        # reclaim. Fires after PN203 has captured what's worth saving.
        try:
            pn201_maybe_empty_cache(free_mib)
        except Exception:
            pass

        # smart proactive demote via vllm LRU. Falls back to
        # legacy block_idx=0 path if no BlockPools registered (dispatcher
        # not wired) or no cached candidates found.
        target = _DEMOTE_BATCH_CACHED
        n_demoted = _proactive_demote_cold(target)

        if n_demoted == 0:
            # Legacy fallback — only fires if BlockPool refs not registered
            # or no cached blocks exist yet (cold start)
            views = getattr(_TM, "_attention_views", {}) or {}
            for layer_name, info in list(views.items())[:target]:
                num_blocks = int(info.get("num_blocks", 0))
                if num_blocks <= 0:
                    continue
                if _TM.demote_block(layer_name, 0):
                    n_demoted += 1
                if n_demoted >= target:
                    break

        if n_demoted > 0:
            _PN95_STATS["ticks_demote_triggered"] += 1
            _PN95_STATS["blocks_demoted_total"] += n_demoted
            _PN95_STATS["last_demote_count"] = n_demoted
            log.warning(
                "[PN95 v1.0 Phase 4.1] scheduler_tick: pressure (free=%d MiB "
                "< %d MiB) — demoted %d cold blocks via LRU "
                "(total demoted=%d, prefix_store_entries=%d)",
                free_mib, _THRESHOLD_CACHED, n_demoted,
                _PN95_STATS["blocks_demoted_total"],
                len(_PN95_PREFIX_STORE),
            )
    except Exception as e:  # pragma: no cover — defensive
        log.warning("[PN95 v1.0 Phase 4.1] scheduler_tick failed: %s", e)


def tier_manager() -> Optional[Any]:
    """Accessor for tests + observability."""
    return _TM


def reset_for_tests() -> None:
    """Drop the singleton + cached state. Used by pytest fixtures."""
    global _TM, _LAST_GROUP_IDS_BY_HASH
    with _LOCK:
        _TM = None
        _LAST_GROUP_IDS_BY_HASH = {}
