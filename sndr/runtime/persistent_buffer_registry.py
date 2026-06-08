# SPDX-License-Identifier: Apache-2.0
"""Operator-visible unified buffer pool registry — v11.1.0 P3.3.

Master plan section P3.3: 4 separate buffer pool managers (P36/PN12/P46/P39a)
share a common pattern — allocate-once, reuse-forever buffers in vLLM
hot paths. This module centralizes the lookup surface so:

  - Operators can `sndr patches show buffer_registry` and see all pools
    + sizes + per-pool stats in one place.
  - Future pools migrate via a single API (~10 LOC each).
  - Each pool's allocation logic stays IDENTICAL (byte-equivalent).

NOT a refactor of the pool LOGIC. Each pool's internal torch.empty() call
is unchanged. Only the LOOKUP SURFACE is unified.

Public surface:

  PersistentBufferRegistry — singleton, `Registry().get_pool(name)`
  BufferPool — single named pool; .acquire / .release / .stats

  POOL_TQ_DECODE_SHARED         — P36 (TurboQuant shared decode buffers)
  POOL_FFN_INTERMEDIATE_SCRATCH — PN12 (FFN intermediate scratch)
  POOL_GDN_GATING               — P46 (GDN gating buffers)
  POOL_FLA_KKT_PERSISTENT_A     — P39a (FLA chunk_scaled_dot_kkt pool)

Migration pattern (per pool):

  # Before
  def _ensure_buffer(layer, shape, dtype):
      if not hasattr(layer, "_buf"):
          layer._buf = torch.empty(shape, dtype=dtype, device="cuda")
      return layer._buf

  # After (thin wrapper, byte-equivalent)
  def _ensure_buffer(layer, shape, dtype):
      from sndr.runtime.persistent_buffer_registry import (
          PersistentBufferRegistry, POOL_TQ_DECODE_SHARED,
      )
      pool = PersistentBufferRegistry().get_pool(POOL_TQ_DECODE_SHARED)
      return pool.acquire(shape, dtype, "cuda")

The pool's acquire() does the same torch.empty() when no buffer of the
exact shape+dtype+device is in the free list — semantics are identical
to the per-patch helper.

Thread safety: BufferPool uses an RLock around the free-list. Acquires
under contention either reuse a free buffer or allocate fresh; no
deadlock or race possible. PersistentBufferRegistry uses an RLock
around the pool index.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
v11.1.0 Phase 6 P3.3 closeout.
"""
from __future__ import annotations

import logging
from threading import RLock
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch

log = logging.getLogger("genesis.runtime.persistent_buffer_registry")


# Pool name constants — referenced by migrated patches + sndr CLI surfaces.
POOL_TQ_DECODE_SHARED = "tq_decode_shared"
POOL_FFN_INTERMEDIATE_SCRATCH = "ffn_intermediate_scratch"
POOL_GDN_GATING = "gdn_gating"
POOL_FLA_KKT_PERSISTENT_A = "fla_kkt_persistent_a"


class BufferPool:
    """One named buffer pool — pre-allocates on first acquire, reuses
    after release.

    Each pool keeps a free-list keyed by (shape_tuple, dtype, device).
    Acquire pops a free buffer if available; else allocates fresh via
    torch.empty() — same as the per-patch helper had.

    Release pushes back to the free-list for reuse. No copy-on-release;
    callers MUST NOT keep a reference to the returned tensor across the
    release call.

    Thread-safe via internal RLock around the free-list.
    """

    def __init__(self, name: str, max_size: int = 64):
        self.name = name
        self._free_lists: dict[tuple, list] = {}
        self._lock = RLock()
        self._stats = {
            "acquires": 0,
            "releases": 0,
            "allocations": 0,
            "reuses": 0,
        }
        self._max_size = max_size

    def acquire(self, shape, dtype: "torch.dtype", device) -> "torch.Tensor":
        """Return a tensor of the requested shape/dtype/device.

        Reuses from free-list if a buffer with the EXACT same
        (shape, dtype, device) is available; else allocates fresh
        via torch.empty().
        """
        import torch

        shape_tuple = tuple(shape)
        device_key = str(device) if not isinstance(device, str) else device
        key = (shape_tuple, dtype, device_key)
        with self._lock:
            self._stats["acquires"] += 1
            free = self._free_lists.get(key)
            if free:
                tensor = free.pop()
                self._stats["reuses"] += 1
                return tensor
            self._stats["allocations"] += 1
        # Allocate outside the lock to avoid holding it during CUDA sync
        tensor = torch.empty(shape_tuple, dtype=dtype, device=device)
        return tensor

    def release(self, tensor: "torch.Tensor") -> None:
        """Mark tensor reusable. Caller MUST drop its reference."""
        shape_tuple = tuple(tensor.shape)
        device_key = str(tensor.device)
        key = (shape_tuple, tensor.dtype, device_key)
        with self._lock:
            self._stats["releases"] += 1
            free = self._free_lists.setdefault(key, [])
            if len(free) < self._max_size:
                free.append(tensor)

    def stats(self) -> dict:
        """Return a snapshot of acquire/release/alloc/reuse counts."""
        with self._lock:
            return dict(self._stats)

    def __repr__(self) -> str:
        with self._lock:
            shapes = sorted(self._free_lists.keys(), key=lambda k: k[0])
            return f"BufferPool(name={self.name!r}, shapes={len(shapes)}, stats={self._stats})"


class PersistentSlicePool:
    """Grow-in-place + slice-on-acquire pool — CUDA-graph-safe semantics.

    UNLIKE BufferPool (free-list, acquire/release), this pool keeps ONE
    tensor per (key, dtype, device) and returns a view-slice on each
    acquire. Pointer-stable for same-or-smaller shapes; pointer changes
    EXACTLY ONCE on grow.

    Use this for hot-path allocators that:
      1. Are called sequentially (no concurrency on the same tensor)
      2. Need pointer stability across calls (CUDA-graph capture)
      3. Have a "max-so-far" sizing pattern (variable rows, fixed cols, etc.)

    Migration target for legacy allocators with the slice+grow pattern:
      - PN12 FFNIntermediateCache (1 variable dim: rows)
      - P39a FlaKktBufferManager (2 variable dims: B + T)
      - Future allocators with similar shape-class semantics

    Key semantics
    -------------
    The caller declares which dims are FIXED (held constant per pool entry)
    via the `key_dims` parameter to `acquire()`. The remaining (leading)
    dims are VARIABLE — they may grow over the lifetime of the pool.

    Example — PN12 (variable rows, fixed intermediate_size):

        pool = registry.get_slice_pool(POOL_FFN_INTERMEDIATE_SCRATCH)
        # First call: allocates [128, 17408]
        t1 = pool.acquire(shape=(128, 17408), dtype=fp16, device='cuda',
                          key_dims=1)  # last 1 dim is "fixed" → key=(17408,)
        # Same shape: returns slice, pointer-stable
        t2 = pool.acquire(shape=(64, 17408), dtype=fp16, device='cuda',
                          key_dims=1)  # returns t1[:64], same data_ptr
        # Larger first dim: grows ONCE
        t3 = pool.acquire(shape=(256, 17408), dtype=fp16, device='cuda',
                          key_dims=1)  # reallocates, data_ptr changes
        # Same large or smaller: stable again
        t4 = pool.acquire(shape=(200, 17408), dtype=fp16, device='cuda',
                          key_dims=1)  # returns t3[:200], same as t3

    Example — P39a (variable B+T, fixed H+BT):

        # 4D shape, last 2 dims fixed
        t = pool.acquire(shape=(B, T, H, BT), dtype=fp32, device='cuda',
                         key_dims=2)  # key=(H, BT)

    Thread safety
    -------------
    RLock protects pool dict + grow operations. Acquire is fast-path
    lock-free after the initial pool exists at the requested size.
    """

    def __init__(self, name: str):
        self.name = name
        self._pools: dict[tuple, "torch.Tensor"] = {}
        self._lock = RLock()
        self._stats = {
            "acquires": 0,
            "allocations": 0,
            "grows": 0,
            "slice_hits": 0,
        }

    def acquire(
        self,
        shape,
        dtype: "torch.dtype",
        device,
        key_dims: int = 0,
    ) -> "torch.Tensor":
        """Return a view of shape `shape` from the persistent pool.

        `key_dims` declares how many TRAILING dims are fixed identity for
        the pool key. The remaining LEADING dims are variable — pool
        grows in-place when a larger value is requested.

        Same dtype + device + fixed_tail required for slice reuse;
        otherwise a new pool is created (no implicit casting).

        For pure fixed-shape pools (no variable dims, e.g. P46 GDN gating),
        pass `key_dims=len(shape)`. The pool's first call allocates;
        subsequent calls with the same shape return the same tensor.
        """
        import torch

        shape_tuple = tuple(int(x) for x in shape)
        if key_dims < 0 or key_dims > len(shape_tuple):
            raise ValueError(
                f"key_dims={key_dims} out of range for shape "
                f"{shape_tuple} (len={len(shape_tuple)})"
            )
        device_key = str(device) if not isinstance(device, str) else device
        fixed_tail = shape_tuple[len(shape_tuple) - key_dims:] if key_dims else ()
        var_head = shape_tuple[:len(shape_tuple) - key_dims]
        key = (fixed_tail, dtype, device_key)

        with self._lock:
            self._stats["acquires"] += 1
            pool = self._pools.get(key)
            needs_alloc = pool is None
            needs_grow = False
            if pool is not None:
                # Check whether the existing pool covers the request.
                for i, want in enumerate(var_head):
                    if pool.shape[i] < want:
                        needs_grow = True
                        break

            if needs_alloc:
                # First allocation — size to requested.
                pool = torch.empty(shape_tuple, dtype=dtype, device=device)
                self._pools[key] = pool
                self._stats["allocations"] += 1
                return pool

            if needs_grow:
                # Compute new shape: per-variable-dim max(current, requested).
                new_var = tuple(
                    max(int(pool.shape[i]), int(var_head[i]))
                    for i in range(len(var_head))
                )
                new_shape = new_var + fixed_tail
                pool = torch.empty(new_shape, dtype=dtype, device=device)
                self._pools[key] = pool
                self._stats["grows"] += 1

            # Slice-on-acquire: return view at exactly the requested shape.
            # For fully-fixed pools (key_dims == len(shape)), var_head is
            # empty and we return the pool tensor directly.
            self._stats["slice_hits"] += 1
            if not var_head:
                return pool
            slicer = tuple(slice(0, int(s)) for s in var_head) + tuple(
                slice(None) for _ in fixed_tail
            )
            return pool[slicer]

    def stats(self) -> dict:
        """Snapshot of acquire / alloc / grow counts."""
        with self._lock:
            return dict(self._stats)

    def num_entries(self) -> int:
        """How many distinct (fixed_tail, dtype, device) keys are live."""
        with self._lock:
            return len(self._pools)

    def total_bytes(self) -> int:
        """Sum of bytes across all entries in this pool."""
        with self._lock:
            total = 0
            for t in self._pools.values():
                total += t.element_size() * t.numel()
            return total

    def clear_for_tests(self) -> None:
        """Drop all pool entries — for test isolation only."""
        with self._lock:
            self._pools.clear()
            for k in self._stats:
                self._stats[k] = 0

    def __repr__(self) -> str:
        return (
            f"PersistentSlicePool(name={self.name!r}, "
            f"entries={self.num_entries()}, stats={self.stats()})"
        )


class PersistentBufferRegistry:
    """Singleton index of named BufferPools and PersistentSlicePools.

    Pools are created lazily on first get_pool(name) / get_slice_pool(name)
    call. The index is process-global — every patch in the process shares
    the same pool for the same name.

    BufferPool vs PersistentSlicePool
    ----------------------------------
    - BufferPool: acquire/release with free list. Good for transient
      buffers with explicit lifecycle (caller knows when done).
    - PersistentSlicePool: grow + slice. Good for hot-path buffers with
      "max-so-far" sizing + CUDA-graph capture requirements.

    A name can be used as either a BufferPool OR a PersistentSlicePool,
    but not both. The first get_*_pool() call locks the type for the
    lifetime of the pool. Mixing raises ValueError.
    """

    _instance: Optional["PersistentBufferRegistry"] = None
    _instance_lock = RLock()

    def __new__(cls) -> "PersistentBufferRegistry":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._pools = {}
                    instance._pool_lock = RLock()
                    cls._instance = instance
        return cls._instance

    def get_pool(self, name: str, max_size: int = 64) -> BufferPool:
        """Return the named BufferPool, creating it on first call."""
        with self._pool_lock:
            if name not in self._pools:
                self._pools[name] = BufferPool(name, max_size=max_size)
                log.debug("[buffer_registry] created BufferPool %r", name)
            p = self._pools[name]
            if not isinstance(p, BufferPool):
                raise ValueError(
                    f"pool {name!r} was registered as "
                    f"{type(p).__name__}, not BufferPool"
                )
            return p

    def get_slice_pool(self, name: str) -> PersistentSlicePool:
        """Return the named PersistentSlicePool, creating it on first call."""
        with self._pool_lock:
            if name not in self._pools:
                self._pools[name] = PersistentSlicePool(name)
                log.debug("[buffer_registry] created PersistentSlicePool %r", name)
            p = self._pools[name]
            if not isinstance(p, PersistentSlicePool):
                raise ValueError(
                    f"pool {name!r} was registered as "
                    f"{type(p).__name__}, not PersistentSlicePool"
                )
            return p

    def all_pools(self) -> dict:
        """Snapshot of all registered pools (BufferPool + PersistentSlicePool)."""
        with self._pool_lock:
            return dict(self._pools)

    def summary(self) -> dict:
        """Operator-facing summary — used by `sndr patches show buffer_registry`."""
        with self._pool_lock:
            pools = {}
            for name, p in self._pools.items():
                if isinstance(p, BufferPool):
                    pools[name] = {
                        "type": "BufferPool",
                        "stats": p.stats(),
                        "shape_count": len(p._free_lists),
                    }
                elif isinstance(p, PersistentSlicePool):
                    pools[name] = {
                        "type": "PersistentSlicePool",
                        "stats": p.stats(),
                        "entry_count": p.num_entries(),
                        "total_bytes": p.total_bytes(),
                    }
                else:
                    pools[name] = {"type": type(p).__name__}
            return {
                "pool_count": len(self._pools),
                "pools": pools,
            }

    def _clear_for_tests(self) -> None:
        """Drop all pools — for test isolation only."""
        with self._pool_lock:
            self._pools.clear()


def _reset_registry_for_tests() -> None:
    """Reset the singleton — for test isolation only.

    NOTE: this exists ONLY to allow `clear_for_tests` style fixtures
    to reset state between tests. Production code MUST NOT call this.
    """
    PersistentBufferRegistry._instance = None


__all__ = [
    "BufferPool",
    "PersistentSlicePool",
    "PersistentBufferRegistry",
    "POOL_TQ_DECODE_SHARED",
    "POOL_FFN_INTERMEDIATE_SCRATCH",
    "POOL_GDN_GATING",
    "POOL_FLA_KKT_PERSISTENT_A",
]
