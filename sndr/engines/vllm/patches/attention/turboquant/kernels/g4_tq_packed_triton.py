# SPDX-License-Identifier: Apache-2.0
"""Triton fused write/read kernels with 3-bit / 4-bit packing.

================================================================
KEY DIFFERENCE FROM g4_tq_write_triton.py
================================================================

The earlier kernels stored quantized indices as uint8 (1 byte per coord),
giving only 2× compression. This module packs indices into uint32 words:

  * **3-bit pack**: 8 indices per uint32 word (24 bits used + 8 padding)
    → 4 bytes per 8 coords → effective 0.5 byte/coord → 4× weight + scale = ~3.88× compression
  * **4-bit pack**: 8 indices per uint32 word (32 bits) → 4 bytes per 8 coords
    → effective 0.5 byte/coord → ~3.88× compression (no padding)

Note: 3-bit could theoretically achieve 5.33× with tight 24-bit packing
(3 bytes per 8 coords), but uint32 alignment is preferred for Triton's
memory access granularity. The trade-off is 8 bytes per 256-d vector
(scale + alignment) vs ~5 GB savings on a 60-layer 256K context cache —
still huge.

================================================================
KERNEL LAYOUT
================================================================

Output buffer:
  packed:  [M, num_kv_heads, head_dim // 8]   uint32  (was: [M, H, D] uint8)
  scale:   [M, num_kv_heads]                  fp32

For head_dim=256: 256/8 = 32 uint32 per (token, head) = 128 bytes
vs fp16: 512 bytes per (token, head) → **4× compression**
plus the 4-byte scale overhead.

================================================================
TRITON BIT OPS
================================================================

Triton supports basic integer arithmetic on int32 tensors. Pack:
  word = idx[0] | (idx[1] << 3) | (idx[2] << 6) | ...

Unpack:
  idx[i] = (word >> (3 * i)) & 0x7

For 4-bit:
  word = idx[0] | (idx[1] << 4) | (idx[2] << 8) | ...
  idx[i] = (word >> (4 * i)) & 0xF

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

GENESIS_G4_TQ_PACKED_MARKER = (
    "Genesis G4-TurboQuant PACKED write/read kernel v1 (3/4-bit uint32 pack)"
)


# Boundaries baked
_BOUNDARIES_3BIT: tuple[float, ...] = (
    -1.84375, -1.05860, -0.50977, -0.00000,
     0.50977,  1.05860,  1.84375,
)
_BOUNDARIES_4BIT: tuple[float, ...] = (
    -2.40052, -1.84372, -1.43737, -1.10422,
    -0.81739, -0.56030, -0.32171,  0.00000,
     0.32171,  0.56030,  0.81739,  1.10422,
     1.43737,  1.84372,  2.40052,
)


# Centroids baked
_CENTROIDS_3BIT: tuple[float, ...] = (
    -2.34375, -1.34375, -0.77344, -0.24609,
     0.24609,  0.77344,  1.34375,  2.34375,
)
_CENTROIDS_4BIT: tuple[float, ...] = (
    -2.73163, -2.06940, -1.61803, -1.25670,
    -0.95174, -0.68303, -0.43757, -0.20585,
     0.20585,  0.43757,  0.68303,  0.95174,
     1.25670,  1.61803,  2.06940,  2.73163,
)


# ─── Triton WRITE kernel (rotate → normalize → quantize → PACK) ──────


if _TRITON_AVAILABLE:

    @triton.jit
    def _g4_tq_write_packed_kernel_3bit(
        X_ptr,                 # [M, H, D] bf16/fp16 raw KV vector
        SIGNS_ptr,             # [D] fp32 ±1
        PACKED_ptr,            # [M, H, D//8] uint32
        SCALE_ptr,             # [M, H] fp32
        # boundaries (7 values for 3-bit)
        b0, b1, b2, b3, b4, b5, b6,
        # strides
        stride_xm, stride_xh, stride_xd,
        stride_pm, stride_ph, stride_pd,
        stride_sm, stride_sh,
        M, NUM_KV_HEADS,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,   # 128 — WHT block
    ):
        """Fused write: rotate → L2-normalize → quantize-3bit → pack uint32."""
        m = tl.program_id(0)
        h = tl.program_id(1)

        x_ptr = X_ptr + m * stride_xm + h * stride_xh
        p_ptr = PACKED_ptr + m * stride_pm + h * stride_ph
        scale_ptr = SCALE_ptr + m * stride_sm + h * stride_sh

        N_BLOCKS: tl.constexpr = HEAD_DIM // BLOCK_SIZE
        cols = tl.arange(0, BLOCK_SIZE)

        # PASS 1: compute L2 of rotated vector
        l2_sq = tl.zeros((), dtype=tl.float32)
        for b in tl.static_range(N_BLOCKS):
            block_off = b * BLOCK_SIZE
            xb = tl.load(x_ptr + (block_off + cols) * stride_xd).to(tl.float32)
            sb = tl.load(SIGNS_ptr + (block_off + cols)).to(tl.float32)
            # WHT butterfly placeholder — for now signs only (TODO: real WHT)
            xb_rot = xb * sb
            l2_sq = l2_sq + tl.sum(xb_rot * xb_rot, axis=0)

        scale = tl.sqrt(l2_sq / tl.full((), HEAD_DIM, tl.float32))
        # Stability guard: NaN → 1.0 (would propagate corruption otherwise)
        scale_clean = tl.where(scale == scale, scale, 1.0)
        scale_safe = tl.where(scale_clean > 1e-8, scale_clean, 1.0)
        tl.store(scale_ptr, scale_clean)

        # PASS 2: rotate, normalize, quantize, pack 8 indices → 1 uint32
        # We process HEAD_DIM coords in groups of 8, packing each group.
        N_PACKED: tl.constexpr = HEAD_DIM // 8

        # We iterate over packed words; each handles 8 coords
        for w in tl.static_range(N_PACKED):
            coord_off = w * 8
            # Load 8 coords (may span block boundary)
            i_arr = coord_off + tl.arange(0, 8)
            x_block = tl.load(x_ptr + i_arr * stride_xd).to(tl.float32)
            s_block = tl.load(SIGNS_ptr + i_arr).to(tl.float32)
            x_rot = x_block * s_block / scale_safe

            # Stability: defensive NaN/Inf cleanup before quantization.
            # NaN → 0 (centre of codebook); ±Inf-magnitude → clip to safe span.
            x_rot = tl.where(x_rot == x_rot, x_rot, 0.0)
            x_rot = tl.maximum(tl.minimum(x_rot, 100.0), -100.0)

            # Quantize: 7 boundaries → indices 0..7
            idx = tl.zeros((8,), dtype=tl.int32)
            idx = idx + (x_rot > b0).to(tl.int32)
            idx = idx + (x_rot > b1).to(tl.int32)
            idx = idx + (x_rot > b2).to(tl.int32)
            idx = idx + (x_rot > b3).to(tl.int32)
            idx = idx + (x_rot > b4).to(tl.int32)
            idx = idx + (x_rot > b5).to(tl.int32)
            idx = idx + (x_rot > b6).to(tl.int32)

            # Pack 8 × 3-bit into uint32:
            # packed = idx[0] | (idx[1] << 3) | ... | (idx[7] << 21)
            # Triton: we use reduce / sum-with-shift idiom
            shifts = tl.arange(0, 8) * 3
            packed_word = tl.sum(idx << shifts, axis=0)

            tl.store(p_ptr + w * stride_pd, packed_word.to(tl.uint32))


    @triton.jit
    def _g4_tq_read_packed_kernel_3bit(
        PACKED_ptr,            # [M, H, D//8] uint32
        SCALE_ptr,             # [M, H] fp32
        SIGNS_ptr,             # [D] fp32
        X_OUT_ptr,             # [M, H, D] bf16/fp16
        # Centroids (8 for 3-bit)
        c0, c1, c2, c3, c4, c5, c6, c7,
        # Strides
        stride_pm, stride_ph, stride_pd,
        stride_sm, stride_sh,
        stride_xm, stride_xh, stride_xd,
        M, NUM_KV_HEADS,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused read: unpack uint32 → 8×3bit → dequant → inverse rotate."""
        m = tl.program_id(0)
        h = tl.program_id(1)

        p_ptr = PACKED_ptr + m * stride_pm + h * stride_ph
        scale_ptr = SCALE_ptr + m * stride_sm + h * stride_sh
        x_ptr = X_OUT_ptr + m * stride_xm + h * stride_xh

        scale = tl.load(scale_ptr).to(tl.float32)
        # Stability: defensive NaN cleanup on the dequantization side.
        scale = tl.where(scale == scale, scale, 1.0)
        N_PACKED: tl.constexpr = HEAD_DIM // 8

        for w in tl.static_range(N_PACKED):
            coord_off = w * 8

            # Load one uint32 word
            word = tl.load(p_ptr + w * stride_pd).to(tl.int32)

            # Unpack 8 × 3-bit
            shifts = tl.arange(0, 8) * 3
            idx = (word >> shifts) & 0x7  # (8,)

            # Codebook lookup via cascade
            v = c0
            v = tl.where(idx == 1, c1, v)
            v = tl.where(idx == 2, c2, v)
            v = tl.where(idx == 3, c3, v)
            v = tl.where(idx == 4, c4, v)
            v = tl.where(idx == 5, c5, v)
            v = tl.where(idx == 6, c6, v)
            v = tl.where(idx == 7, c7, v)

            # Re-scale + inverse rotation (sign-flip; full WHT in future)
            i_arr = coord_off + tl.arange(0, 8)
            s_block = tl.load(SIGNS_ptr + i_arr).to(tl.float32)
            v = v * scale * s_block

            tl.store(
                x_ptr + i_arr * stride_xd,
                v.to(X_OUT_ptr.dtype.element_ty),
            )


def g4_tq_write_packed_3bit(
    x: torch.Tensor,
    signs: torch.Tensor,
    head_dim: int = 256,
    block_size: int = 128,
    out_packed: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton fused write with 3-bit uint32 packing.

    Args:
        x: ``(M, num_kv_heads, head_dim)`` bf16/fp16.
        signs: ``(head_dim,)`` fp32 ±1 (RHT).
        head_dim: must match x.shape[-1].
        block_size: WHT block (128 default).
        out_packed: optional pre-allocated ``(M, num_kv_heads, head_dim//8)`` uint32.
        out_scale: optional ``(M, num_kv_heads)`` fp32.

    Returns:
        (packed, scale):
          packed: ``(M, num_kv_heads, head_dim // 8)`` uint32
          scale: ``(M, num_kv_heads)`` fp32
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton not available")
    assert x.dim() == 3, f"expected (M, num_kv_heads, head_dim); got {x.shape}"
    M, num_kv_heads, hd = x.shape
    assert hd == head_dim, f"head_dim mismatch: {hd} != {head_dim}"
    assert head_dim % 8 == 0, f"head_dim {head_dim} must be div 8 for 3-bit pack"

    n_packed = head_dim // 8
    if out_packed is None:
        out_packed = torch.empty(
            (M, num_kv_heads, n_packed), dtype=torch.int32, device=x.device,
        )
    if out_scale is None:
        out_scale = torch.empty(
            (M, num_kv_heads), dtype=torch.float32, device=x.device,
        )

    grid = (M, num_kv_heads)
    _g4_tq_write_packed_kernel_3bit[grid](
        x, signs, out_packed, out_scale,
        *_BOUNDARIES_3BIT,
        x.stride(0), x.stride(1), x.stride(2),
        out_packed.stride(0), out_packed.stride(1), out_packed.stride(2),
        out_scale.stride(0), out_scale.stride(1),
        M, num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
    )
    return out_packed, out_scale


def g4_tq_read_packed_3bit(
    packed: torch.Tensor,
    scale: torch.Tensor,
    signs: torch.Tensor,
    head_dim: int = 256,
    block_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Triton fused read with 3-bit uint32 unpack.

    Args:
        packed: ``(M, num_kv_heads, head_dim//8)`` int32/uint32.
        scale: ``(M, num_kv_heads)`` fp32.
        signs: ``(head_dim,)`` fp32 ±1.
        head_dim, block_size: must match write.
        dtype: output dtype.
        out: optional pre-allocated output.

    Returns:
        ``(M, num_kv_heads, head_dim)`` decompressed.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton not available")
    assert packed.dim() == 3
    M, num_kv_heads, n_packed = packed.shape
    assert n_packed == head_dim // 8

    if out is None:
        out = torch.empty(
            (M, num_kv_heads, head_dim), dtype=dtype, device=packed.device,
        )

    grid = (M, num_kv_heads)
    _g4_tq_read_packed_kernel_3bit[grid](
        packed, scale, signs, out,
        *_CENTROIDS_3BIT,
        packed.stride(0), packed.stride(1), packed.stride(2),
        scale.stride(0), scale.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        M, num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
    )
    return out


__all__ = [
    "GENESIS_G4_TQ_PACKED_MARKER",
    "g4_tq_write_packed_3bit",
    "g4_tq_read_packed_3bit",
]
