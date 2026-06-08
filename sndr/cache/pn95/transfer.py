# SPDX-License-Identifier: Apache-2.0
"""PN95 transfer primitives — CUDA streams + GPU↔CPU byte copy.

Eight low-level helpers consumed by the demote / promote / prefetch
hot paths:

  Stream + singleton:
    * ``_pn95_stream``                  — lazy-init dedicated CUDA
                                           stream for PN95 transfers
                                           (returns None on no-CUDA)

  Single-block primitives (Sprint Q1 B1):
    * ``_pn95_gpu_to_cpu_bytes``        — async-aware demote copy
    * ``_pn95_cpu_to_gpu_copy``         — async-aware promote copy

  Batched primitives (Sprint Q1 B2 / B3):
    * ``_pn95_gpu_to_cpu_bytes_batch``  — N-layer demote, 1 sync
    * ``_pn95_cpu_to_gpu_copy_batch``   — N-layer promote, 1 wait_stream

  Stream-pool variants (Sprint Q1 B4 routing):
    * ``_pn95_gpu_to_cpu_bytes_batch_v2`` — pool-stream + end_event
    * ``_pn95_cpu_to_gpu_copy_batch_v2``  — pool-stream + wait_stream

  Pure helper (no CUDA / no torch):
    * ``_mm_block_overlap_set``         — Day 5: which block indices
                                           overlap given MM placeholder
                                           token ranges (used by
                                           notify_admit)

M.4.2.G scope: function extraction only. Every CUDA/torch dependency
remains lazy — ``import torch`` lives inside each function body, never
at module level — so importing this module on a no-CUDA host (Mac
laptop) succeeds and the disabled-path returns work without side
effects.

The single mutable state singleton this code touches stays defined in
``_pn95_runtime``:

  _PN95_CUDA_STREAM   — lazy-init dedicated stream; REBOUND inside
                         ``_pn95_stream`` (originally via ``global``;
                         replaced with ``_rt._PN95_CUDA_STREAM = …``
                         attribute write at the same module-attribute
                         slot)

Inventory at extraction time:

  * 5 test sites rebind ``rt._PN95_CUDA_STREAM`` via
    ``monkeypatch.setattr`` — same alias-fragility class first
    surfaced as ``_PN95_STATS`` in M.4.1 — so the state name must
    stay anchored in ``_pn95_runtime`` to keep the monkeypatch path
    visible to readers.

  * Sibling-patch text-anchor imports: ZERO references to any
    transfer symbol. No anchor regen, no shim-string constraint.

  * Test direct-call surface: ``_pn95_cpu_to_gpu_copy_batch``,
    ``_pn95_gpu_to_cpu_bytes_batch``, ``_mm_block_overlap_set`` are
    invoked via ``rt.<name>`` from
    tests/unit/cache/test_pn95_b2/b3/day5_*.py — all keep resolving
    through the re-export shim with no edit.

  * Internal callers within ``_pn95_runtime`` (notify_admit at
    line 662, demote_on_evict at line 1495, promote_on_miss at
    line 1699) STAY in _pn95_runtime — they call transfer functions
    via the re-export shim. No anchor regen needed.

The ``_PN95_STATS`` counter dict is also referenced (write-only) from
the moved functions. It stays in ``_pn95_runtime`` (from M.4.1) and
is accessed via lazy ``_rt._PN95_STATS``.
"""
from __future__ import annotations

from typing import Any, Optional

from .gates import _pn95_async_enabled, _pn95_use_stream_pool


# ─── Stream singleton ──────────────────────────────────────────────────


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
    # State `_PN95_CUDA_STREAM` stays in `_pn95_runtime` (5 test sites
    # rebind it via monkeypatch). Late-import keeps the cross-module
    # rebind path visible to those tests.
    from sndr.cache import _pn95_runtime as _rt
    if _rt._PN95_CUDA_STREAM is None:
        try:
            import torch
            if torch.cuda.is_available():
                _rt._PN95_CUDA_STREAM = torch.cuda.Stream()
        except Exception:
            return None
    return _rt._PN95_CUDA_STREAM


# ─── Single-block GPU↔CPU primitives (Sprint Q1 B1) ────────────────────


def _pn95_gpu_to_cpu_bytes(view: Any) -> bytes:
    """Path C v1.0 Sprint Q1 B1 — async-aware GPU→CPU byte copy.

    Uses _pn95_stream when available so demote PCIe transfer does not
    block default-stream compute. Synchronous fallback preserves existing
    behaviour (and correctness) when CUDA is unavailable.

    Returns bytes — caller may compress via _pn95_compress_bytes(...).

    Safety: stream.synchronize() before reading bytes ensures copy complete.
    Default stream NOT blocked during transfer (only synchronizes pn95 stream).
    """
    import torch
    from sndr.cache import _pn95_runtime as _rt
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
    _rt._PN95_STATS["async_demote_count"] = (
        _rt._PN95_STATS.get("async_demote_count", 0) + 1
    )
    return bytes(cpu_tensor.numpy().tobytes())


def _pn95_cpu_to_gpu_copy(view: Any, src_bytes: bytes) -> int:
    """Path C v1.0 Sprint Q1 B1 — async-aware CPU→GPU byte copy.

    Critical: writes to the GPU view must be visible on the default
    stream before the subsequent attention forward reads it. Achieved
    via `current_stream.wait_stream(_pn95_stream)` after copy.

    This makes the default stream wait for our copy WITHOUT blocking
    the CPU thread.

    Returns number of bytes copied. Synchronous fallback preserves
    current behaviour when CUDA is unavailable.
    """
    import numpy as np
    import torch
    from sndr.cache import _pn95_runtime as _rt
    # np.frombuffer returns read-only array; torch.from_numpy then warns.
    # .copy() makes writable copy — safe for torch consumption.
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
    _rt._PN95_STATS["async_promote_count"] = (
        _rt._PN95_STATS.get("async_promote_count", 0) + 1
    )
    return n


# ─── Batched primitives (Sprint Q1 B2 / B3) ────────────────────────────


def _pn95_gpu_to_cpu_bytes_batch(views: list) -> list:
    """Path C v1.0 Sprint Q1 B2 — batched async GPU→CPU byte copy.

    Same effect as N calls to _pn95_gpu_to_cpu_bytes but with ONE
    stream.synchronize() instead of N. For 17-attention-layer demote,
    this saves ~16× stream sync overhead (~10-50 μs each → 160-800 μs total).

    Critical: PCIe DMA engine processes batched copies more efficiently
    too — multiple in-flight transfers overlap better than serial.

    Returns list of bytes in same order as input views. Empty list if
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
    from sndr.cache import _pn95_runtime as _rt
    stream = _pn95_stream() if _pn95_async_enabled() else None
    if stream is None:
        # Synchronous fallback — equivalent to N sequential _pn95_gpu_to_cpu_bytes calls.
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
    # ONE sync for all N copies — saves (N-1) × ~10-50 μs overhead.
    stream.synchronize()
    _rt._PN95_STATS["async_demote_count"] = (
        _rt._PN95_STATS.get("async_demote_count", 0) + len(views)
    )
    _rt._PN95_STATS["async_batch_demote_count"] = (
        _rt._PN95_STATS.get("async_batch_demote_count", 0) + 1
    )
    return [bytes(t.numpy().tobytes()) for t in cpu_tensors]


def _pn95_cpu_to_gpu_copy_batch(views: list, src_bytes_list: list) -> int:
    """Path C v1.0 Sprint Q1 B3 — batched async CPU→GPU byte copy.

    Mirror of B2 (_pn95_gpu_to_cpu_bytes_batch) for the promote path.
    Same correctness primitive (current_stream.wait_stream(_pn95_stream))
    but ONE wait_stream call for N layer copies vs N individual calls.

    Args:
      views: list of GPU tensor views (one per layer)
      src_bytes_list: list of raw CPU bytes (already decompressed), same length

    Returns: number of layers successfully copied (0 if mismatched lengths
    or empty input).

    Critical: like the single-block helper, the default stream waits for
    our copy via wait_stream() — no race against the subsequent attention
    forward.
    """
    if not views or not src_bytes_list or len(views) != len(src_bytes_list):
        return 0
    # Stream-pool mode (env-gated, default OFF) — routes to v2.
    if _pn95_use_stream_pool() and _pn95_async_enabled():
        return _pn95_cpu_to_gpu_copy_batch_v2(views, src_bytes_list)
    import numpy as np
    import torch
    from sndr.cache import _pn95_runtime as _rt
    stream = _pn95_stream() if _pn95_async_enabled() else None

    if stream is None:
        # Synchronous fallback — equivalent to N sequential _pn95_cpu_to_gpu_copy calls.
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
    _rt._PN95_STATS["async_promote_count"] = (
        _rt._PN95_STATS.get("async_promote_count", 0) + n_total
    )
    _rt._PN95_STATS["async_batch_promote_count"] = (
        _rt._PN95_STATS.get("async_batch_promote_count", 0) + 1
    )
    return n_total


# ─── Stream-pool variants (Sprint Q1 B4 routing) ───────────────────────


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
    from sndr.cache import _pn95_runtime as _rt
    from sndr.cache import _pn95_stream_pool as sp
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
        _rt._PN95_STATS["async_demote_count"] = (
            _rt._PN95_STATS.get("async_demote_count", 0) + len(views)
        )
        _rt._PN95_STATS["async_batch_demote_count"] = (
            _rt._PN95_STATS.get("async_batch_demote_count", 0) + 1
        )
        _rt._PN95_STATS["stream_pool_batches"] = (
            _rt._PN95_STATS.get("stream_pool_batches", 0) + 1
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
    from sndr.cache import _pn95_runtime as _rt
    from sndr.cache import _pn95_stream_pool as sp
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
        _rt._PN95_STATS["async_promote_count"] = (
            _rt._PN95_STATS.get("async_promote_count", 0) + n_total
        )
        _rt._PN95_STATS["async_batch_promote_count"] = (
            _rt._PN95_STATS.get("async_batch_promote_count", 0) + 1
        )
        _rt._PN95_STATS["stream_pool_batches"] = (
            _rt._PN95_STATS.get("stream_pool_batches", 0) + 1
        )
        return n_total
    finally:
        st.release_stream(stream)
        st.release_event(end_evt)


# ─── Day 5 — per-block MM tagging helper (pure, no torch/CUDA) ─────────


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
