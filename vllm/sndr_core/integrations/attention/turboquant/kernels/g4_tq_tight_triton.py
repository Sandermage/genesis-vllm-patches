# SPDX-License-Identifier: Apache-2.0
"""Triton fused write/read kernels for TIGHT 3-bit packing (5.33× compression).

================================================================
WHY TIGHT (vs uint32)
================================================================

uint32 packing: 8 × 3-bit indices in a 32-bit word (24 used + 8 wasted).
   storage  = (head_dim // 8) × 4 bytes  + 4 byte scale
   for D=256: 132 bytes/(token,head) → ~3.88× vs fp16

tight packing: 8 × 3-bit indices in 3 bytes (24 bits, 0 waste).
   storage  = (head_dim × 3 // 8) bytes + 4 byte scale
   for D=256: 100 bytes/(token,head) → ~5.12× vs fp16

For 60-layer 256K Gemma 4 KV cache the saving is ~3.4 GB across the
whole cache (per K + per V). Useful for squeezing prefix-caching or
batch=2 fitting into the same 48 GB budget.

================================================================
BYTE LAYOUT (PER GROUP OF 8 INDICES) — matches numpy reference
================================================================

bit:   23 22 21 20 19 18 17 16 | 15 14 13 12 11 10 09 08 | 07 06 05 04 03 02 01 00
idx:   [ idx7 ][ idx6 ][idx5_h]|[idx5_l][ idx4 ][ idx3 ][idx2_h]|[idx2_l][ idx1 ][ idx0 ]
byte:  |--byte 2 -----------|  |  --byte 1 ----------------|     | -- byte 0 -------|

  byte 0 = idx0          | (idx1 << 3) | ((idx2 & 0x03) << 6)
  byte 1 = (idx2 >> 2)   | (idx3 << 1) | (idx4 << 4) | ((idx5 & 0x01) << 7)
  byte 2 = (idx5 >> 1)   | (idx6 << 2) | (idx7 << 5)

================================================================
SUPPORTED MODES
================================================================

  signs-only:  fastest write/read, rotation = sign flip only
  full_wht:    in-tile FWHT butterfly (real Hadamard, +~22% MSE saving
               on heavy-tailed inputs, ~57× faster than v1 GEMV form)

Both modes share the same packed buffer layout — toggling between
restarts does NOT require cache migration.

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

# Boundaries / centroids are always defined; the JIT'd butterfly helper
# only exists when Triton is available. Import accordingly so this module
# imports cleanly on Triton-less hosts (CI, dev macs).
from .g4_tq_packed_wht_triton import _BOUNDARIES_3BIT, _CENTROIDS_3BIT

if _TRITON_AVAILABLE:
    from .g4_tq_packed_wht_triton import _fwht_butterfly_block

GENESIS_G4_TQ_TIGHT_MARKER = (
    "Genesis G4-TurboQuant TIGHT 3-bit packing kernel v1 "
    "(true 5.12× compression — 3 bytes per 8 indices)"
)


# ─── Triton kernels (signs-only & full-WHT, write + read) ───────────


if _TRITON_AVAILABLE:

    @triton.jit
    def _g4_tq_write_tight_kernel_3bit(
        X_ptr,                  # [M, H, D] bf16/fp16 raw KV
        SIGNS_ptr,              # [D] fp32 ±1
        PACKED_ptr,             # [M, H, D*3//8] uint8 output
        SCALE_ptr,              # [M, H] fp32
        b0, b1, b2, b3, b4, b5, b6,
        stride_xm, stride_xh, stride_xd,
        stride_pm, stride_ph, stride_pd,
        stride_sm, stride_sh,
        M, NUM_KV_HEADS,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        APPLY_WHT: tl.constexpr,
    ):
        """Fused write with TIGHT (3-byte-per-8-indices) packing.

        Steps per WHT block of BLOCK_SIZE coords:
          1. signs (D)
          2. optional FWHT butterfly (APPLY_WHT=True)
          3. divide by scale
          4. quantize to 3-bit
          5. tight-pack 8 indices → 3 bytes; store

        Output buffer is uint8 with stride
        ``(M, H, head_dim * 3 // 8)``. Per WHT block we write
        ``BLOCK_SIZE * 3 // 8`` bytes; per group of 8 indices we
        write 3 bytes.
        """
        m = tl.program_id(0)
        h = tl.program_id(1)

        x_ptr = X_ptr + m * stride_xm + h * stride_xh
        p_ptr = PACKED_ptr + m * stride_pm + h * stride_ph
        scale_ptr = SCALE_ptr + m * stride_sm + h * stride_sh

        N_BLOCKS: tl.constexpr = HEAD_DIM // BLOCK_SIZE
        cols = tl.arange(0, BLOCK_SIZE)

        # PASS 1: L2 norm (Hadamard is orthonormal, so safe to compute pre-rotation)
        l2_sq = tl.zeros((), dtype=tl.float32)
        for b in tl.static_range(N_BLOCKS):
            block_off = b * BLOCK_SIZE
            xb = tl.load(x_ptr + (block_off + cols) * stride_xd).to(tl.float32)
            l2_sq = l2_sq + tl.sum(xb * xb, axis=0)

        scale = tl.sqrt(l2_sq / tl.full((), HEAD_DIM, tl.float32))
        scale_clean = tl.where(scale == scale, scale, 1.0)
        scale_safe = tl.where(scale_clean > 1e-8, scale_clean, 1.0)
        tl.store(scale_ptr, scale_clean)

        # Per-block constants (tight packing layout)
        N_GROUPS: tl.constexpr = BLOCK_SIZE // 8       # groups per block
        BYTES_PER_BLOCK: tl.constexpr = N_GROUPS * 3   # = BLOCK_SIZE * 3 // 8

        # PASS 2: per WHT block — rotate, quantize, tight-pack
        for b in tl.static_range(N_BLOCKS):
            block_off = b * BLOCK_SIZE
            xb = tl.load(x_ptr + (block_off + cols) * stride_xd).to(tl.float32)
            sb = tl.load(SIGNS_ptr + (block_off + cols)).to(tl.float32)
            x_signed = xb * sb

            if APPLY_WHT:
                x_rot = _fwht_butterfly_block(x_signed, BLOCK_SIZE)
            else:
                x_rot = x_signed

            x_norm = x_rot / scale_safe
            # Stability
            x_norm = tl.where(x_norm == x_norm, x_norm, 0.0)
            x_norm = tl.maximum(tl.minimum(x_norm, 100.0), -100.0)

            # Quantize all coords at once
            idx = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
            idx = idx + (x_norm > b0).to(tl.int32)
            idx = idx + (x_norm > b1).to(tl.int32)
            idx = idx + (x_norm > b2).to(tl.int32)
            idx = idx + (x_norm > b3).to(tl.int32)
            idx = idx + (x_norm > b4).to(tl.int32)
            idx = idx + (x_norm > b5).to(tl.int32)
            idx = idx + (x_norm > b6).to(tl.int32)

            # Reshape (BLOCK,) → (N_GROUPS, 8) for tight packing
            idx_2d = tl.reshape(idx, (N_GROUPS, 8))
            slot_ids = tl.arange(0, 8)  # 0..7, per-group element index

            # Extract each slot via masked sum (Triton-friendly gather):
            # slot k = sum over axis 1 of (idx_2d * (slot_ids == k))
            i0 = tl.sum(tl.where(slot_ids[None, :] == 0, idx_2d, 0), axis=1)
            i1 = tl.sum(tl.where(slot_ids[None, :] == 1, idx_2d, 0), axis=1)
            i2 = tl.sum(tl.where(slot_ids[None, :] == 2, idx_2d, 0), axis=1)
            i3 = tl.sum(tl.where(slot_ids[None, :] == 3, idx_2d, 0), axis=1)
            i4 = tl.sum(tl.where(slot_ids[None, :] == 4, idx_2d, 0), axis=1)
            i5 = tl.sum(tl.where(slot_ids[None, :] == 5, idx_2d, 0), axis=1)
            i6 = tl.sum(tl.where(slot_ids[None, :] == 6, idx_2d, 0), axis=1)
            i7 = tl.sum(tl.where(slot_ids[None, :] == 7, idx_2d, 0), axis=1)

            # Tight pack (matches numpy reference pack_indices_3bit_tight):
            #   byte0 = i0 | (i1 << 3) | ((i2 & 0x03) << 6)
            #   byte1 = ((i2 >> 2) & 0x01) | (i3 << 1) | (i4 << 4)
            #                              | ((i5 & 0x01) << 7)
            #   byte2 = ((i5 >> 1) & 0x03) | (i6 << 2) | (i7 << 5)
            b0v = (i0 | (i1 << 3) | ((i2 & 0x03) << 6)) & 0xFF
            b1v = (((i2 >> 2) & 0x01) | (i3 << 1) | (i4 << 4) |
                   ((i5 & 0x01) << 7)) & 0xFF
            b2v = (((i5 >> 1) & 0x03) | (i6 << 2) | (i7 << 5)) & 0xFF

            # Store: for group g, bytes are at positions [3g, 3g+1, 3g+2]
            # relative to the WHT-block byte base (= b * BYTES_PER_BLOCK).
            group_ids = tl.arange(0, N_GROUPS)
            block_byte_off = b * BYTES_PER_BLOCK

            tl.store(
                p_ptr + (block_byte_off + group_ids * 3) * stride_pd,
                b0v.to(tl.uint8),
            )
            tl.store(
                p_ptr + (block_byte_off + group_ids * 3 + 1) * stride_pd,
                b1v.to(tl.uint8),
            )
            tl.store(
                p_ptr + (block_byte_off + group_ids * 3 + 2) * stride_pd,
                b2v.to(tl.uint8),
            )


    @triton.jit
    def _g4_tq_read_tight_kernel_3bit(
        PACKED_ptr,             # [M, H, D*3//8] uint8 input
        SCALE_ptr,              # [M, H] fp32
        SIGNS_ptr,              # [D] fp32 ±1
        X_OUT_ptr,              # [M, H, D] bf16/fp16 output
        c0, c1, c2, c3, c4, c5, c6, c7,
        stride_pm, stride_ph, stride_pd,
        stride_sm, stride_sh,
        stride_xm, stride_xh, stride_xd,
        M, NUM_KV_HEADS,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        APPLY_WHT: tl.constexpr,
    ):
        """Fused read with TIGHT (3-byte) unpacking.

        Mirror of write: load 3 bytes per group, unpack 8 indices,
        codebook lookup, optional inverse FWHT, apply signs.
        """
        m = tl.program_id(0)
        h = tl.program_id(1)

        p_ptr = PACKED_ptr + m * stride_pm + h * stride_ph
        scale_ptr = SCALE_ptr + m * stride_sm + h * stride_sh
        x_ptr = X_OUT_ptr + m * stride_xm + h * stride_xh

        scale = tl.load(scale_ptr).to(tl.float32)
        scale = tl.where(scale == scale, scale, 1.0)

        N_BLOCKS: tl.constexpr = HEAD_DIM // BLOCK_SIZE
        cols = tl.arange(0, BLOCK_SIZE)
        N_GROUPS: tl.constexpr = BLOCK_SIZE // 8
        BYTES_PER_BLOCK: tl.constexpr = N_GROUPS * 3

        for b in tl.static_range(N_BLOCKS):
            block_off = b * BLOCK_SIZE
            block_byte_off = b * BYTES_PER_BLOCK
            group_ids = tl.arange(0, N_GROUPS)

            # Load three byte arrays (each (N_GROUPS,)) — matches tight format
            b0v = tl.load(
                p_ptr + (block_byte_off + group_ids * 3) * stride_pd
            ).to(tl.int32)
            b1v = tl.load(
                p_ptr + (block_byte_off + group_ids * 3 + 1) * stride_pd
            ).to(tl.int32)
            b2v = tl.load(
                p_ptr + (block_byte_off + group_ids * 3 + 2) * stride_pd
            ).to(tl.int32)

            # Unpack 8 indices (matches numpy unpack_indices_3bit_tight)
            i0 =  b0v       & 0x07
            i1 = (b0v >> 3) & 0x07
            i2 = ((b0v >> 6) & 0x03) | ((b1v & 0x01) << 2)
            i3 = (b1v >> 1) & 0x07
            i4 = (b1v >> 4) & 0x07
            i5 = ((b1v >> 7) & 0x01) | ((b2v & 0x03) << 1)
            i6 = (b2v >> 2) & 0x07
            i7 = (b2v >> 5) & 0x07

            # Combine i0..i7 (each (N_GROUPS,)) into (N_GROUPS, 8) then (BLOCK,)
            slot_ids = tl.arange(0, 8)
            idx_2d = (
                tl.where(slot_ids[None, :] == 0, i0[:, None], 0) +
                tl.where(slot_ids[None, :] == 1, i1[:, None], 0) +
                tl.where(slot_ids[None, :] == 2, i2[:, None], 0) +
                tl.where(slot_ids[None, :] == 3, i3[:, None], 0) +
                tl.where(slot_ids[None, :] == 4, i4[:, None], 0) +
                tl.where(slot_ids[None, :] == 5, i5[:, None], 0) +
                tl.where(slot_ids[None, :] == 6, i6[:, None], 0) +
                tl.where(slot_ids[None, :] == 7, i7[:, None], 0)
            )
            idx = tl.reshape(idx_2d, (BLOCK_SIZE,))

            # Codebook lookup (rotated frame)
            v = tl.full((BLOCK_SIZE,), c0, dtype=tl.float32)
            v = tl.where(idx == 1, c1, v)
            v = tl.where(idx == 2, c2, v)
            v = tl.where(idx == 3, c3, v)
            v = tl.where(idx == 4, c4, v)
            v = tl.where(idx == 5, c5, v)
            v = tl.where(idx == 6, c6, v)
            v = tl.where(idx == 7, c7, v)
            v = v * scale

            if APPLY_WHT:
                u = _fwht_butterfly_block(v, BLOCK_SIZE)
            else:
                u = v

            # Apply signs
            sb = tl.load(SIGNS_ptr + (block_off + cols)).to(tl.float32)
            u = u * sb

            tl.store(
                x_ptr + (block_off + cols) * stride_xd,
                u.to(X_OUT_ptr.dtype.element_ty),
            )


def g4_tq_write_tight_3bit(
    x: torch.Tensor,
    signs: torch.Tensor,
    head_dim: int = 256,
    block_size: int = 128,
    apply_wht: bool = False,
    out_packed: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton fused write with TIGHT 3-byte-per-8-indices packing.

    Args:
        x: ``(M, num_kv_heads, head_dim)`` bf16/fp16.
        signs: ``(head_dim,)`` fp32 ±1.
        head_dim: must match x.shape[-1] and be divisible by 8.
        block_size: WHT block size (64/128/256 supported).
        apply_wht: if True, applies full Walsh-Hadamard butterfly
                   after signs; if False, only signs (placeholder rotation).
        out_packed: optional pre-allocated ``(M, H, head_dim*3//8)`` uint8.
        out_scale: optional pre-allocated ``(M, H)`` fp32.

    Returns:
        (packed, scale):
          packed shape (M, num_kv_heads, head_dim * 3 // 8) uint8.
          scale shape (M, num_kv_heads) fp32.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton not available")
    assert x.dim() == 3, f"expected (M, num_kv_heads, head_dim); got {x.shape}"
    M, num_kv_heads, hd = x.shape
    assert hd == head_dim
    assert head_dim % 8 == 0, f"head_dim {head_dim} must be divisible by 8"
    assert head_dim % block_size == 0
    assert block_size in (64, 128, 256), (
        f"block_size {block_size} must be in {{64, 128, 256}} for tight kernel "
        "(butterfly FWHT depends on these unrolled sizes)"
    )

    packed_dim = head_dim * 3 // 8
    if out_packed is None:
        out_packed = torch.empty(
            (M, num_kv_heads, packed_dim), dtype=torch.uint8, device=x.device,
        )
    if out_scale is None:
        out_scale = torch.empty(
            (M, num_kv_heads), dtype=torch.float32, device=x.device,
        )

    grid = (M, num_kv_heads)
    _g4_tq_write_tight_kernel_3bit[grid](
        x, signs, out_packed, out_scale,
        *_BOUNDARIES_3BIT,
        x.stride(0), x.stride(1), x.stride(2),
        out_packed.stride(0), out_packed.stride(1), out_packed.stride(2),
        out_scale.stride(0), out_scale.stride(1),
        M, num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        APPLY_WHT=bool(apply_wht),
    )
    return out_packed, out_scale


def g4_tq_read_tight_3bit(
    packed: torch.Tensor,
    scale: torch.Tensor,
    signs: torch.Tensor,
    head_dim: int = 256,
    block_size: int = 128,
    apply_wht: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Triton fused read for TIGHT 3-byte-per-8-indices packing."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton not available")
    assert packed.dim() == 3
    M, num_kv_heads, packed_dim = packed.shape
    assert packed_dim == head_dim * 3 // 8, (
        f"packed last dim {packed_dim} ≠ expected {head_dim * 3 // 8} "
        f"for head_dim={head_dim}"
    )
    assert block_size in (64, 128, 256)

    if out is None:
        out = torch.empty(
            (M, num_kv_heads, head_dim), dtype=dtype, device=packed.device,
        )

    grid = (M, num_kv_heads)
    _g4_tq_read_tight_kernel_3bit[grid](
        packed, scale, signs, out,
        *_CENTROIDS_3BIT,
        packed.stride(0), packed.stride(1), packed.stride(2),
        scale.stride(0), scale.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        M, num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        APPLY_WHT=bool(apply_wht),
    )
    return out


__all__ = [
    "GENESIS_G4_TQ_TIGHT_MARKER",
    "g4_tq_write_tight_3bit",
    "g4_tq_read_tight_3bit",
]
