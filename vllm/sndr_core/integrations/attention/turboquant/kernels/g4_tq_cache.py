# SPDX-License-Identifier: Apache-2.0
"""KVCache wrapper class for G4-TurboQuant + vLLM v1 integration.

Provides a drop-in replacement for vLLM's KV cache layout that stores
compressed indices + per-vector scale instead of raw fp16/bf16.

================================================================
MEMORY LAYOUT
================================================================

For each transformer layer L with num_kv_heads H, block_size B (typically
16 tokens per page), and head_dim D:

  K_indices:   (num_blocks, B, H, D)   uint8
  K_scale:     (num_blocks, B, H)      fp32
  V_indices:   (num_blocks, B, H, D)   uint8
  V_scale:     (num_blocks, B, H)      fp32

Compression vs fp16:
  fp16:        2 × B × H × D bytes per token
  TQ 4-bit:    1 × B × H × D + 4 × B × H bytes per token
  Compression: 16 / (4 + 4·D⁻¹) ≈ 4×

For D=256: TQ 4-bit = 4× compression vs fp16.

================================================================
INTEGRATION POINTS WITH vLLM v1
================================================================

The wrapper hooks into vLLM v1's KV cache management:

  1. **Allocation**: `KVCacheSpec.build` — we provide a custom spec
     that reports the compressed memory layout instead of fp16.
  2. **Write path**: attention forward calls `write_kv_cache(layer_idx,
     k, v, slot_mapping)` — we intercept, run `g4_tq_write()`, store
     indices+scale in our buffers.
  3. **Read path**: attention forward calls `read_kv_cache(layer_idx,
     block_ids)` — we intercept, run `g4_tq_read()`, return fp16/bf16
     tensors for attention math.

================================================================
SLIDING WINDOW HANDLING (Gemma 4-specific)
================================================================

Gemma 4 has interleaved sliding (window=1024) + global layers. For
sliding layers we can use **higher precision** (e.g. 4-bit) since the
KV cache is small; for global layers we use **lower precision** (3-bit)
to maximize 256K context capacity.

The wrapper accepts per-layer-type config:
  bits_sliding: int = 4   # sliding layers: high quality, small cache
  bits_global:  int = 3   # global layers: lower bits, max compression

================================================================
ROTATION SEED MANAGEMENT
================================================================

Each layer has a distinct sign vector / Clifford rotor:
  signs[layer_idx] = build_randomized_hadamard_seed(head_dim, layer_idx, seed_base)

Seeds are baked at apply() time and stored on device. Identical seeds
across server restarts → reproducible KV writes (important for
prefix caching!).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .g4_tq_codebook import get_centroids
from .g4_tq_rotor import build_randomized_hadamard_seed

log = logging.getLogger("genesis.g4_tq.cache")

GENESIS_G4_TQ_CACHE_MARKER = (
    "Genesis G4-TurboQuant KVCache wrapper v1 (drop-in vLLM v1 layer cache)"
)


# ─── Config dataclass ────────────────────────────────────────────────


@dataclass
class G4TurboQuantConfig:
    """Per-model config for G4-TurboQuant KV cache.

    Attributes:
        head_dim: per-head dimension (256 for Gemma 4).
        bits_sliding: bits per coord for sliding-attention layers (default 4).
        bits_global: bits per coord for full_attention layers (default 3).
        block_size: WHT block size for RHT (default 128, must divide head_dim).
        rotation_method: 'rht' or 'clifford'.
        seed_base: model-wide RNG seed; deterministic across restarts.
        sliding_window: sliding window size in tokens (1024 for Gemma 4).
        per_layer_types: list of "sliding_attention" / "full_attention"
                         strings, length = num_layers.
        pack_mode: 'uint32' (default, 4× compression), 'tight' (5.33×
                   compression but harder to access), 'uint8' (legacy
                   unpacked, 2× compression — kept for back-compat).
        wht_mode: 'signs_only' (default — fast, original placeholder
                   path with sign-flip only) or 'full_wht' (slower but
                   ~6× lower quantization MSE; applies real Walsh-Hadamard
                   rotation in addition to signs). Opt-in via env
                   ``GENESIS_G4_TQ_WHT_MODE=full_wht``. Buffer layout is
                   IDENTICAL between modes — operators can flip the flag
                   between restarts without cache migration.
    """
    head_dim: int = 256
    bits_sliding: int = 4
    bits_global: int = 3
    block_size: int = 128
    rotation_method: str = "rht"
    seed_base: int = 0xC0FFEE
    sliding_window: int = 1024
    per_layer_types: Optional[list[str]] = None
    pack_mode: str = "uint32"  # uint32 | tight | uint8
    wht_mode: str = "signs_only"  # signs_only | full_wht

    def __post_init__(self):
        assert self.head_dim % self.block_size == 0, (
            f"head_dim {self.head_dim} not divisible by block_size "
            f"{self.block_size}"
        )
        assert self.bits_sliding in (3, 4, 5), (
            f"bits_sliding must be 3/4/5, got {self.bits_sliding}"
        )
        assert self.bits_global in (3, 4, 5), (
            f"bits_global must be 3/4/5, got {self.bits_global}"
        )
        assert self.rotation_method in ("rht", "clifford"), (
            f"rotation_method must be 'rht' or 'clifford', got {self.rotation_method}"
        )
        assert self.wht_mode in ("signs_only", "full_wht"), (
            f"wht_mode must be 'signs_only' or 'full_wht', got {self.wht_mode}"
        )

    def bits_for_layer(self, layer_idx: int) -> int:
        """Get bit-width for a specific layer."""
        if self.per_layer_types is None:
            return self.bits_global
        layer_type = self.per_layer_types[layer_idx]
        return self.bits_sliding if layer_type == "sliding_attention" else self.bits_global


# ─── Memory layout helper ────────────────────────────────────────────


def kv_cache_size_bytes(
    config: G4TurboQuantConfig,
    num_layers: int,
    num_kv_heads: int,
    num_blocks: int,
    block_size_tokens: int = 16,
) -> dict:
    """Estimate KV cache memory footprint.

    Returns dict with sliding_bytes, global_bytes, total_bytes, fp16_equivalent_bytes.
    """
    if config.per_layer_types is None:
        # Assume all global
        sliding_count = 0
        global_count = num_layers
    else:
        sliding_count = sum(1 for t in config.per_layer_types if t == "sliding_attention")
        global_count = num_layers - sliding_count

    # Per-token, per-head storage depends on pack_mode:
    #   uint32: head_dim/8 × 4 bytes = head_dim/2 bytes per coord set
    #   tight:  head_dim/8 × 3 bytes = head_dim×3/8 bytes
    #   uint8:  head_dim bytes (legacy)
    pack_mode = getattr(config, "pack_mode", "uint32")
    if pack_mode == "uint32":
        bytes_per_token_per_head = (config.head_dim // 8) * 4 + 4
    elif pack_mode == "tight":
        bytes_per_token_per_head = (config.head_dim * 3 // 8) + 4
    else:  # uint8 legacy
        bytes_per_token_per_head = config.head_dim + 4

    sliding_bytes = (
        sliding_count
        * num_kv_heads
        * num_blocks
        * block_size_tokens
        * bytes_per_token_per_head
        * 2  # K + V
    )
    global_bytes = (
        global_count
        * num_kv_heads
        * num_blocks
        * block_size_tokens
        * bytes_per_token_per_head
        * 2
    )

    fp16_bytes_per_token = config.head_dim * 2  # bf16/fp16 = 2 bytes
    fp16_total = (
        num_layers
        * num_kv_heads
        * num_blocks
        * block_size_tokens
        * fp16_bytes_per_token
        * 2  # K + V
    )

    total = sliding_bytes + global_bytes

    return {
        "sliding_bytes": sliding_bytes,
        "global_bytes": global_bytes,
        "total_bytes": total,
        "fp16_equivalent_bytes": fp16_total,
        "compression_ratio": fp16_total / max(total, 1),
        "savings_gb": (fp16_total - total) / (1024**3),
    }


# ─── KVCache wrapper class ───────────────────────────────────────────


class G4TurboQuantKVCache:
    """Per-layer KV cache wrapper that stores compressed indices + scale.

    Designed to be installed at vLLM v1's layer-cache allocation hook.
    The class is **lazy-init**: actual buffer allocation happens at
    first write call to know exact `(M, num_kv_heads)`.

    Usage (server-side wiring via G4_19 patch):

        config = G4TurboQuantConfig(
            head_dim=256, bits_sliding=4, bits_global=3,
            per_layer_types=hf_config.layer_types,
        )
        for layer_idx in range(num_layers):
            cache[layer_idx] = G4TurboQuantKVCache(layer_idx, config)
    """

    def __init__(
        self,
        layer_idx: int,
        config: G4TurboQuantConfig,
        num_kv_heads: int,
        max_num_blocks: int,
        block_size_tokens: int = 16,
        device: Optional["torch.device"] = None,  # noqa: F821
    ):
        """Initialize per-layer compressed KV cache.

        Args:
            layer_idx: this layer's index (for seeding).
            config: G4TurboQuantConfig.
            num_kv_heads: per-shard KV heads after TP split.
            max_num_blocks: maximum number of cache blocks.
            block_size_tokens: tokens per block (typically 16).
            device: torch device.
        """
        import torch
        self.layer_idx = layer_idx
        self.config = config
        self.num_kv_heads = num_kv_heads
        self.max_num_blocks = max_num_blocks
        self.block_size_tokens = block_size_tokens
        self.device = device or torch.device("cpu")

        # Bits for this layer
        self.bits = config.bits_for_layer(layer_idx)

        # Sign vector (RHT) — same for K and V, deterministic per layer
        signs_np = build_randomized_hadamard_seed(
            head_dim=config.head_dim,
            layer_idx=layer_idx,
            seed_base=config.seed_base,
        )
        self.signs = torch.from_numpy(signs_np).to(self.device)

        # Lazy buffers
        self.k_indices = None
        self.k_scale = None
        self.v_indices = None
        self.v_scale = None
        self._allocated = False

    def allocate_(self) -> None:
        """Allocate full buffers up-front.

        Layout depends on pack_mode:
          uint32: (N, H, D//8) int32 — 8 indices per word
          tight:  (N, H, D*3//8) uint8 — tight 3-byte/8-index pack
          uint8:  (N, H, D) uint8 — legacy unpacked
        """
        import torch
        N = self.max_num_blocks * self.block_size_tokens
        H = self.num_kv_heads
        D = self.config.head_dim
        pack_mode = getattr(self.config, "pack_mode", "uint32")

        if pack_mode == "uint32":
            assert D % 8 == 0, f"head_dim {D} not div 8 for uint32 pack"
            cache_shape = (N, H, D // 8)
            cache_dtype = torch.int32
        elif pack_mode == "tight":
            assert D % 8 == 0
            cache_shape = (N, H, (D * 3) // 8)
            cache_dtype = torch.uint8
        else:  # uint8 legacy
            cache_shape = (N, H, D)
            cache_dtype = torch.uint8

        self.k_indices = torch.zeros(cache_shape, dtype=cache_dtype, device=self.device)
        self.v_indices = torch.zeros(cache_shape, dtype=cache_dtype, device=self.device)
        self.k_scale = torch.zeros((N, H), dtype=torch.float32, device=self.device)
        self.v_scale = torch.zeros((N, H), dtype=torch.float32, device=self.device)
        self._allocated = True
        # Memory accounting in MB
        kv_mb = (
            self.k_indices.numel() * self.k_indices.element_size() +
            self.v_indices.numel() * self.v_indices.element_size() +
            self.k_scale.numel() * 4 + self.v_scale.numel() * 4
        ) / (1024 ** 2)
        log.info(
            "[G4-TQ layer=%d] allocated KV cache: %d slots × %d heads × "
            "%d head_dim, bits=%d, pack=%s, mem=%.1f MB",
            self.layer_idx, N, H, D, self.bits, pack_mode, kv_mb,
        )

    def write_kv(
        self,
        slot_indices: "torch.Tensor",  # noqa: F821 — (M,) int64
        k: "torch.Tensor",             # noqa: F821 — (M, H, D)
        v: "torch.Tensor",             # noqa: F821 — (M, H, D)
    ) -> None:
        """Write K, V vectors at slot positions.

        Dispatches to packed (uint32) or unpacked (uint8) Triton kernel
        depending on config.pack_mode. Slot scatter uses simple indexing.
        """
        if not self._allocated:
            self.allocate_()

        pack_mode = getattr(self.config, "pack_mode", "uint32")
        wht_mode = getattr(self.config, "wht_mode", "signs_only")

        if pack_mode == "tight" and self.bits == 3:
            # 5.33× compression tight 3-byte packing kernel.
            # Same kernel implements both signs_only and full_wht via the
            # APPLY_WHT constexpr argument.
            from .g4_tq_tight_triton import g4_tq_write_tight_3bit

            apply_wht = (wht_mode == "full_wht")
            k_idx, k_sc = g4_tq_write_tight_3bit(
                k.contiguous(), self.signs,
                head_dim=self.config.head_dim,
                block_size=self.config.block_size,
                apply_wht=apply_wht,
            )
            v_idx, v_sc = g4_tq_write_tight_3bit(
                v.contiguous(), self.signs,
                head_dim=self.config.head_dim,
                block_size=self.config.block_size,
                apply_wht=apply_wht,
            )
        elif pack_mode == "uint32" and self.bits == 3:
            # Use packed 3-bit kernel. wht_mode picks the rotation impl;
            # both write to the IDENTICAL uint32 packed buffer format so
            # operators may toggle the flag between restarts safely.
            if wht_mode == "full_wht":
                from .g4_tq_packed_wht_triton import (
                    g4_tq_write_packed_wht_3bit as _write_fn,
                )
            else:
                from .g4_tq_packed_triton import (
                    g4_tq_write_packed_3bit as _write_fn,
                )

            k_idx, k_sc = _write_fn(
                k.contiguous(), self.signs,
                head_dim=self.config.head_dim,
                block_size=self.config.block_size,
            )
            v_idx, v_sc = _write_fn(
                v.contiguous(), self.signs,
                head_dim=self.config.head_dim,
                block_size=self.config.block_size,
            )
        else:
            # Legacy unpacked path
            from .g4_tq_write_triton import g4_tq_write

            k_idx, k_sc = g4_tq_write(
                k.contiguous(), self.signs,
                bits=self.bits,
                head_dim=self.config.head_dim,
                block_size=self.config.block_size,
            )
            v_idx, v_sc = g4_tq_write(
                v.contiguous(), self.signs,
                bits=self.bits,
                head_dim=self.config.head_dim,
                block_size=self.config.block_size,
            )

        # Scatter to slots
        self.k_indices[slot_indices] = k_idx
        self.k_scale[slot_indices] = k_sc
        self.v_indices[slot_indices] = v_idx
        self.v_scale[slot_indices] = v_sc

    def read_kv(
        self,
        slot_indices: "torch.Tensor",  # noqa: F821 — (N,) int64
        dtype: "torch.dtype" = None,   # noqa: F821
    ) -> tuple["torch.Tensor", "torch.Tensor"]:  # noqa: F821
        """Read K, V vectors from slot positions, dequantize to fp16/bf16.

        Returns:
            (k, v): each shape (N, num_kv_heads, head_dim).
        """
        import torch

        if dtype is None:
            dtype = torch.bfloat16

        k_idx = self.k_indices[slot_indices].contiguous()
        k_sc = self.k_scale[slot_indices].contiguous()
        v_idx = self.v_indices[slot_indices].contiguous()
        v_sc = self.v_scale[slot_indices].contiguous()

        pack_mode = getattr(self.config, "pack_mode", "uint32")
        wht_mode = getattr(self.config, "wht_mode", "signs_only")

        if pack_mode == "tight" and self.bits == 3:
            from .g4_tq_tight_triton import g4_tq_read_tight_3bit

            apply_wht = (wht_mode == "full_wht")
            k = g4_tq_read_tight_3bit(
                k_idx, k_sc, self.signs,
                head_dim=self.config.head_dim,
                block_size=self.config.block_size,
                apply_wht=apply_wht,
                dtype=dtype,
            )
            v = g4_tq_read_tight_3bit(
                v_idx, v_sc, self.signs,
                head_dim=self.config.head_dim,
                block_size=self.config.block_size,
                apply_wht=apply_wht,
                dtype=dtype,
            )
        elif pack_mode == "uint32" and self.bits == 3:
            if wht_mode == "full_wht":
                from .g4_tq_packed_wht_triton import (
                    g4_tq_read_packed_wht_3bit as _read_fn,
                )
            else:
                from .g4_tq_packed_triton import (
                    g4_tq_read_packed_3bit as _read_fn,
                )

            k = _read_fn(
                k_idx, k_sc, self.signs,
                head_dim=self.config.head_dim,
                block_size=self.config.block_size,
                dtype=dtype,
            )
            v = _read_fn(
                v_idx, v_sc, self.signs,
                head_dim=self.config.head_dim,
                block_size=self.config.block_size,
                dtype=dtype,
            )
        else:
            from .g4_tq_read_triton import g4_tq_read

            k = g4_tq_read(
                k_idx, k_sc, self.signs,
                bits=self.bits,
                head_dim=self.config.head_dim,
                block_size=self.config.block_size,
                dtype=dtype,
            )
            v = g4_tq_read(
                v_idx, v_sc, self.signs,
                bits=self.bits,
                head_dim=self.config.head_dim,
                block_size=self.config.block_size,
                dtype=dtype,
            )
        return k, v

    def memory_footprint_bytes(self) -> int:
        """Return current allocated memory size in bytes."""
        if not self._allocated:
            return 0
        n = self.max_num_blocks * self.block_size_tokens
        H = self.num_kv_heads
        D = self.config.head_dim
        # uint8 indices + fp32 scale, K+V
        return 2 * (n * H * D + n * H * 4)


__all__ = [
    "GENESIS_G4_TQ_CACHE_MARKER",
    "G4TurboQuantConfig",
    "G4TurboQuantKVCache",
    "kv_cache_size_bytes",
]
