# SPDX-License-Identifier: Apache-2.0
"""PN95 — per-worker pinned host RAM pool for CPU prefix store.

Two-level CPU tier:
  L1 — pinned (page-locked, non-swappable) — fast PCIe DMA, fixed budget
  L2 — pageable bytes (existing _PN95_PREFIX_STORE OrderedDict) — slow but unlimited

Demote path:
  GPU → L1 if a slot is free → bytes copy is direct via cudaMemcpy from pinned host
  GPU → L2 (slow path) if L1 full
  L1 → L2 on eviction (move bytes out of pinned slot to pageable dict)

Promote path:
  L1 hit → GPU DMA directly from pinned slot (fastest)
  L2 hit (no L1) → copy bytes to L1 slot first (if available), then GPU DMA
  Pure L2 → GPU via numpy.frombuffer + non-blocking .to(device) (existing path)

Why pinned for L1:
  torch's .to('cuda', non_blocking=True) launches async cudaMemcpyAsync ONLY
  when source is pinned (page-locked). From regular Python bytes, torch falls
  back to a sync bounce-buffer copy that blocks the host thread until DMA
  completes. Pinning is the difference between 3-5 GB/s sustained PCIe Gen4
  (pinned) and 600-800 MB/s effective (pageable + bounce buffer).

Why two-level (not all-pinned):
  Linux pinned memory is bounded by RLIMIT_MEMLOCK. Default ulimit -l on most
  hosts is 64 KB → only ~120 KV blocks fit. Docker default is usually
  unlimited but not guaranteed. Bounding L1 keeps the patch operator-friendly:
  it always works (degrades to L2-only when L1 budget exceeded).

Storage layout:
  Single large pinned tensor `_pool_buf: (capacity_slots × slot_size, uint8)`.
  Slot i occupies `_pool_buf[i*slot_size : i*slot_size + n_bytes_used_i]`.
  One big cudaHostAlloc (~50ms total) instead of N small ones (~10ms each ×
  N would be seconds at thousands of slots).

Capacity:
  Env-driven (`GENESIS_PN95_PINNED_POOL_MB`, default 256 MB).
  slot_size auto-derived from observed first-demote block size (which is
  stable across calls — all layers same shape per block).

This is opt-in via `GENESIS_ENABLE_PN95_PINNED_POOL=1` (default OFF).
Falls back to existing pageable-only path transparently when disabled
or when allocation fails (RLIMIT_MEMLOCK hit on operator's box).

Per-rank: each Worker process has its own pool (singleton). The pool
is initialized lazily on first demote so we know the actual slot_size
needed (depends on block_size × num_kv_heads × head_dim × kv_dtype).
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

log = logging.getLogger("genesis.pn95.pinned_pool")


def _enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN95_PINNED_POOL", "0",
    ).strip().lower() in ("1", "true", "yes", "on")


def _budget_bytes() -> int:
    """Pinned L1 budget in bytes. Default 256 MB."""
    raw = os.environ.get("GENESIS_PN95_PINNED_POOL_MB", "256")
    try:
        mb = int(raw)
    except ValueError:
        mb = 256
    return max(mb, 16) * 1024 * 1024  # min 16 MB


# Per-process singleton (per-rank since each Worker is its own process).
_POOL: Optional["PinnedHostPool"] = None
_POOL_LOCK = threading.Lock()
# Sticky-disable flag: once cudaHostAlloc fails (RLIMIT_MEMLOCK / host OOM),
# we don't retry on every demote. Module-local so we don't mutate os.environ
# (process-wide, leaks into child procs, races with operator readers — review
# finding #6).
_POOL_ALLOC_FAILED: bool = False


class PinnedHostPool:
    """Fixed-size pinned host RAM pool of equal-sized slots.

    Single big pinned tensor backs all slots. Slot bookkeeping is in-process
    (lists + dict, no torch ops). Thread-safe via _lock; safe to call from
    PN95 scheduler tick + worker forward thread concurrently.

    Stats are exposed via `stats()` for sndr patches pn95-status integration.
    """

    def __init__(self, slot_size: int, capacity_slots: int):
        import torch
        self._slot_size = int(slot_size)
        self._capacity = int(capacity_slots)
        # Single large pinned alloc (one cudaHostAlloc).
        # NOTE: this can fail with `RuntimeError: CUDA error: out of memory`
        # if RLIMIT_MEMLOCK is hit. Caller catches and falls back to pageable.
        self._buf = torch.empty(
            self._capacity * self._slot_size,
            dtype=torch.uint8,
            pin_memory=True,
        )
        self._free: list[int] = list(range(self._capacity))
        # slot_id -> n_bytes used (≤ slot_size)
        self._used: dict[int, int] = {}
        # block_hash -> slot_id (reverse lookup for L1 hit check)
        self._hash_to_slot: dict[Any, int] = {}
        self._lock = threading.Lock()
        # Stats
        self._stats = {
            "slots_capacity": self._capacity,
            "slot_size_bytes": self._slot_size,
            "slots_used": 0,
            "bytes_used": 0,
            "l1_demote_writes": 0,
            "l1_promote_reads_gpu": 0,
            "l1_promote_reads_bytes": 0,
            "l1_evictions": 0,
            "l1_full_skips": 0,
        }

    def capacity(self) -> int:
        return self._capacity

    def slot_size(self) -> int:
        return self._slot_size

    def has(self, block_hash: Any) -> bool:
        with self._lock:
            return block_hash in self._hash_to_slot

    def put(self, block_hash: Any, data: bytes) -> bool:
        """Write `data` into a free slot keyed by block_hash.

        Returns True on success, False on L1 full (caller falls back to L2).
        If block_hash already present, updates existing slot.
        """
        import torch
        if len(data) > self._slot_size:
            # Block bigger than slot — never fits; falls back to L2.
            return False
        with self._lock:
            # Update existing slot.
            slot = self._hash_to_slot.get(block_hash)
            if slot is None:
                if not self._free:
                    self._stats["l1_full_skips"] += 1
                    return False
                slot = self._free.pop()
                self._hash_to_slot[block_hash] = slot
            n = len(data)
            offset = slot * self._slot_size
            # Copy bytes into pinned buffer slice. torch.frombuffer on
            # immutable `bytes` raises a UserWarning (and on some torch
            # builds errors); bytearray gives us a single writable view
            # with one alloc, then from_numpy adapts to a torch tensor.
            # The host-to-host memcpy that copy_() emits is unavoidable.
            import numpy as _np
            src_np = _np.frombuffer(memoryview(data), dtype=_np.uint8)
            src = torch.from_numpy(_np.array(src_np, copy=True))
            self._buf[offset : offset + n].copy_(src)
            self._used[slot] = n
            self._stats["slots_used"] = len(self._used)
            self._stats["bytes_used"] = sum(self._used.values())
            self._stats["l1_demote_writes"] += 1
            return True

    def get_view(self, block_hash: Any):
        """Return torch tensor view of the pinned slot — for direct GPU DMA.

        Returns (view, n_bytes) or (None, 0) on miss.
        Caller does `gpu_view.copy_(view.to(gpu_view.device, non_blocking=True))`.
        """
        with self._lock:
            slot = self._hash_to_slot.get(block_hash)
            if slot is None:
                return None, 0
            n = self._used.get(slot, 0)
            if n == 0:
                return None, 0
            offset = slot * self._slot_size
            self._stats["l1_promote_reads_gpu"] += 1
            return self._buf[offset : offset + n], n

    def get_bytes(self, block_hash: Any) -> bytes:
        """Read slot contents as bytes (compat with existing bytes API).

        Slower than get_view (extra copy), but matches the existing PN95
        protocol where layer payloads are bytes.
        """
        with self._lock:
            slot = self._hash_to_slot.get(block_hash)
            if slot is None:
                return b""
            n = self._used.get(slot, 0)
            if n == 0:
                return b""
            offset = slot * self._slot_size
            self._stats["l1_promote_reads_bytes"] += 1
            # bytes() of a sliced pinned tensor goes through .tobytes() —
            # zero-copy on CPU since memory is already host-resident.
            return bytes(self._buf[offset : offset + n].numpy().tobytes())

    def evict(self, block_hash: Any) -> bytes:
        """Remove `block_hash` from L1, return its bytes for L2 spillover.

        If block_hash not present, returns b''. Slot is added back to free list.
        """
        with self._lock:
            slot = self._hash_to_slot.pop(block_hash, None)
            if slot is None:
                return b""
            n = self._used.pop(slot, 0)
            self._free.append(slot)
            self._stats["slots_used"] = len(self._used)
            self._stats["bytes_used"] = sum(self._used.values())
            self._stats["l1_evictions"] += 1
            if n == 0:
                return b""
            offset = slot * self._slot_size
            return bytes(self._buf[offset : offset + n].numpy().tobytes())

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)


def get_pool(slot_size_hint: int = 0) -> Optional[PinnedHostPool]:
    """Lazy init + return singleton pool, or None if disabled / alloc failed.

    `slot_size_hint` is the observed block byte-size on first demote. It's
    used only at first-time init; subsequent calls ignore it (pool size
    is fixed once allocated).

    Operator escape: setting GENESIS_PN95_PINNED_POOL=0 (or unset) returns
    None and PN95 stays on pageable-only path. Sticky-disable: if the first
    alloc raised RLIMIT_MEMLOCK / OOM, subsequent calls also return None
    (module flag _POOL_ALLOC_FAILED is sticky for the process lifetime).
    """
    global _POOL, _POOL_ALLOC_FAILED
    if not _enabled() or _POOL_ALLOC_FAILED:
        return None
    if _POOL is not None:
        return _POOL
    with _POOL_LOCK:
        if _POOL is not None:
            return _POOL
        # Slot size: prefer explicit env override (avoids the "first demote
        # was lucky-compressible → pool locked into tiny slots" footgun from
        # review finding #5). Default falls back to caller's hint if env
        # not set. We also bump by 25% headroom because the pickle envelope
        # plus zstd worst-case-incompressible payloads can drift up.
        env_slot = os.environ.get("GENESIS_PN95_PINNED_SLOT_BYTES", "").strip()
        if env_slot:
            try:
                slot_bytes = int(env_slot)
            except ValueError:
                slot_bytes = slot_size_hint
        else:
            if slot_size_hint <= 0:
                # First demote hasn't happened yet; we can't size the pool.
                # Caller will retry on next demote with a real hint.
                return None
            slot_bytes = max(slot_size_hint, int(slot_size_hint * 1.25))
        budget = _budget_bytes()
        capacity = max(1, budget // slot_bytes)
        # Keep slot_size_hint name in the PinnedHostPool ctor for back-compat.
        slot_size_hint = slot_bytes
        try:
            pool = PinnedHostPool(slot_size_hint, capacity)
        except Exception as e:
            # RLIMIT_MEMLOCK hit or torch failed pin_memory allocation.
            # Sticky-disable via module flag, NOT env mutation (review #6).
            log.warning(
                "[PN95-PINNED] failed to allocate %d MB pinned pool (slot=%d, "
                "capacity=%d): %s — falling back to pageable-only L2.",
                budget // (1024 * 1024), slot_size_hint, capacity, e,
            )
            _POOL_ALLOC_FAILED = True
            return None
        log.info(
            "[PN95-PINNED] allocated %d MB pinned host pool: %d slots × %d bytes each.",
            budget // (1024 * 1024), capacity, slot_size_hint,
        )
        _POOL = pool
        return _POOL


def reset_pool_for_tests() -> None:
    """Release the singleton — test-only helper."""
    global _POOL
    with _POOL_LOCK:
        _POOL = None
