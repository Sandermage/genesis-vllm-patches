# SPDX-License-Identifier: Apache-2.0
"""P46 — Persistent buffers for `fused_gdn_gating` `g` / `beta_output`.

Problem
-------
`vllm/model_executor/layers/mamba/gdn_linear_attn.py:1195-1196`
allocates two tensors per call:

    g = torch.empty(1, batch, num_heads, dtype=torch.float32, ...)
    beta_output = torch.empty(1, batch, num_heads, dtype=b.dtype, ...)

Qwen3.6-35B-A3B has 48 GDN-bearing layers. On decode (batch=1) these
are TINY (~kilobytes each). Bytes saved per step = negligible. But
the ALLOCATOR overhead — two fresh `torch.empty` calls per layer per
step × 48 layers × 250 tok/s = **~24 000 allocator ops/sec** just
for these two tensors.

Fix
---
Module-level pool keyed by `(batch, num_heads, dtype, device)`. On
first call at a given shape, allocate both `g` and `beta_output`
persistent. Subsequent calls with the same shape (overwhelmingly
common for stable workloads — batch size + head count rarely shift
per forward) return the SAME tensors, kernel writes into them in-place.
No `.zero_()` needed — Triton kernel unconditionally writes every
position.

CUDA graph safety
-----------------
- Pool allocation happens at FIRST call → if that's during warmup
  (before capture), pointer is stable across all captured graphs.
- `_resolve_default_shape_once()` helper lets wiring pre-warm
  during profile_run so the pool is profiler-visible AND pointer
  already stable before any capture.
- Pool "grows" (i.e. re-allocates into different key) only if a
  DIFFERENT (batch, num_heads, dtype) is seen — at which point the
  OLD pool stays live under the old key; we don't mutate the existing
  tensor, so any captured graph holding it continues to work.

Scope notes
-----------
- byte-exact output vs upstream: ✅ yes — Triton kernel writes all
  positions unconditionally, allocated-content doesn't matter (as
  with `torch.empty`).
- Platform guard: NVIDIA CUDA SM 8.0+ (same as rest of P2x).
- Default-on: this patch has NO semantic change, only allocator
  reduction. Active by default on NVIDIA.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
Status: v7.7 implementation (default-on on NVIDIA)
"""
from __future__ import annotations

import logging

import torch

log = logging.getLogger("genesis.gdn_gating_buffer")


class GdnGatingBufferManager:
    """Module-level pool for `fused_gdn_gating`'s two output tensors.

    Thread-safety: class-level mutable dicts; each TP worker process
    has its own state (vLLM uses spawn). No cross-process sync needed.

    Lifecycle per shape:
      first call at (B, H, dt_g, dt_b, device) → alloc g + beta_output
      second call same key → return cached tensors (identity `is`)
      different key → alloc new pair under new key, old pair stays live
    """

    # v11.2.0 P3.3 migration: storage now backed by PersistentSlicePool.
    # The legacy `_G_POOLS` and `_BETA_POOLS` dicts are kept as operator-
    # visible mirrors so existing introspection (memory_metrics.py +
    # get_registry_info()) keeps working byte-equivalently. The
    # underlying tensors are SHARED IDENTITIES with what the unified
    # PersistentSlicePool exposes via the registry.
    _G_POOLS: dict[tuple, torch.Tensor] = {}
    _BETA_POOLS: dict[tuple, torch.Tensor] = {}

    @classmethod
    def should_apply(cls) -> bool:
        """Same platform gate as rest of P2x — NVIDIA SM ≥ 8.0."""
        from sndr.engines.vllm.detection.guards import is_nvidia_cuda, is_sm_at_least
        if not is_nvidia_cuda():
            return False
        if not is_sm_at_least(8, 0):
            return False
        return True

    @classmethod
    def _get_backing_pool(cls):
        """Return the PersistentSlicePool that backs both g + beta pools."""
        from sndr.runtime.persistent_buffer_registry import (
            PersistentBufferRegistry, POOL_GDN_GATING,
        )
        return PersistentBufferRegistry().get_slice_pool(POOL_GDN_GATING)

    @classmethod
    def _acquire_gating_buffer(
        cls,
        mirror_dict: dict,
        kind: str,
        batch: int,
        num_heads: int,
        device,
        dtype: "torch.dtype",
    ) -> "torch.Tensor":
        """Shared core for acquire_g + acquire_beta — fixed shape `(1, B, H)`
        keyed by (B, H, dtype, device). First call allocates via the
        unified PersistentSlicePool; subsequent calls return the same
        tensor (pointer-stable identity for CUDA-graph capture)."""
        if not cls.should_apply():
            return torch.empty(
                (1, batch, num_heads), dtype=dtype, device=device,
            )
        key = (batch, num_heads, str(dtype), str(device))
        t = mirror_dict.get(key)
        if t is not None:
            return t
        pool = cls._get_backing_pool()
        # Distinguish g vs beta in the slice-pool key by appending a kind
        # discriminator via the fixed-tail (last) dim multiplier. We use
        # the natural shape (1, B, H) and rely on dtype to separate g
        # (typically fp32) from beta (typically fp16/bf16). When both
        # share the same dtype/B/H/device (unusual in practice), we tag
        # via key_dims=3 + namespace separation through entry mirroring.
        t = pool.acquire(
            (1, batch, num_heads),
            dtype=dtype, device=device, key_dims=3,
        )
        mirror_dict[key] = t
        log.info(
            "[P46] allocated persistent `%s` buffer (1,%d,%d) dtype=%s "
            "device=%s (backed by PersistentSlicePool)",
            kind, batch, num_heads, dtype, device,
        )
        return t

    @classmethod
    def acquire_g(
        cls,
        batch: int,
        num_heads: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return the persistent `g` buffer for this shape key.

        Shape returned: `(1, batch, num_heads)` — matches upstream
        `fused_gdn_gating` contract line-for-line.

        v11.2.0 P3.3: backed by unified PersistentSlicePool. Byte-equivalent
        semantics — same identity across calls, same shape/dtype/device.

        On platform-skip returns a fresh `torch.empty` (preserves
        upstream semantics on CPU / ROCm / pre-Ampere).
        """
        return cls._acquire_gating_buffer(
            cls._G_POOLS, "g", batch, num_heads, device, dtype,
        )

    @classmethod
    def acquire_beta(
        cls,
        batch: int,
        num_heads: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the persistent `beta_output` buffer for this shape key.

        Shape: `(1, batch, num_heads)`; dtype matches caller's `b.dtype`
        (typically FP16 or BF16).

        v11.2.0 P3.3: backed by unified PersistentSlicePool. Byte-equivalent
        semantics. g vs beta are distinguished by dtype in production
        (g=fp32, beta=fp16/bf16); when the same dtype is used for both
        (test-only edge case) the per-mirror-dict caching ensures
        operational identity is preserved.
        """
        return cls._acquire_gating_buffer(
            cls._BETA_POOLS, "beta_output", batch, num_heads, device, dtype,
        )

    @classmethod
    def get_registry_info(cls) -> dict:
        """Diagnostic snapshot — used by `memory_metrics.py`."""
        def _bytes(t: torch.Tensor) -> int:
            return t.element_size() * t.numel()
        g_bytes = sum(_bytes(t) for t in cls._G_POOLS.values())
        beta_bytes = sum(_bytes(t) for t in cls._BETA_POOLS.values())
        return {
            "num_g_pools": len(cls._G_POOLS),
            "num_beta_pools": len(cls._BETA_POOLS),
            "total_bytes": g_bytes + beta_bytes,
            "g_entries": [
                {"key": k, "bytes": _bytes(t)}
                for k, t in cls._G_POOLS.items()
            ],
            "beta_entries": [
                {"key": k, "bytes": _bytes(t)}
                for k, t in cls._BETA_POOLS.items()
            ],
        }

    @classmethod
    def clear_for_tests(cls) -> None:
        cls._G_POOLS.clear()
        cls._BETA_POOLS.clear()
