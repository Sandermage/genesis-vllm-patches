# SPDX-License-Identifier: Apache-2.0
"""3-bit and 4-bit bit-packing for G4-TurboQuant KV cache indices.

================================================================
WHY PACKING
================================================================

Lloyd-Max quantization produces indices in [0..7] for 3-bit or [0..15]
for 4-bit per coordinate. Storing them as uint8 (1 byte each) wastes
5/8 = 62.5% of bits for 3-bit, or 4/8 = 50% for 4-bit.

For Gemma 4 head_dim=256 with 32K context and 60 layers, fp16 KV cache
is ~30 GB. With packed 3-bit it's ~5.6 GB (5.3× compression). With
unpacked uint8 it's only ~15 GB (2× compression). The 5.3× number we
advertise requires actual bit-packing — that's what this module
provides.

================================================================
LAYOUT
================================================================

**3-bit packing**: group 8 consecutive indices, pack into 24 bits =
3 bytes per group. For head_dim=256: 256/8 = 32 groups × 3 bytes =
96 bytes per (token, head). vs 512 bytes fp16 = 5.33× compression.

**4-bit packing**: group 2 consecutive indices, pack into 8 bits =
1 byte per pair. For head_dim=256: 256/2 = 128 bytes per (token, head).
vs 512 bytes fp16 = 4× compression.

================================================================
BIT ARRANGEMENT (3-bit, little-endian within byte)
================================================================

bit:    23 22 21 20 19 18 17 16 15 14 13 12 11 10 09 08 07 06 05 04 03 02 01 00
idx:    [ 7 ][ 6 ][ 5 ][ 4 ][ 3 ][ 2 ][ 1 ][ 0 ]
byte:   |--byte 2--|---byte 1---|--byte 0--|

Packed: byte0 = idx0 | (idx1 << 3) | ((idx2 & 0x03) << 6)
        byte1 = (idx2 >> 2) | (idx3 << 1) | (idx4 << 4) | ((idx5 & 0x01) << 7)
        byte2 = (idx5 >> 1) | (idx6 << 2) | (idx7 << 5)

For Triton efficiency we use **uint32 layout** instead — 8 indices ×
3 bits = 24 bits fit in one uint32 with 8 bits padding:

  packed_u32 = (idx0 << 0) | (idx1 << 3) | (idx2 << 6) | ... | (idx7 << 21)

This is the layout the kernels use. Storage: 4 bytes per 8 coords =
4 bytes per group. Net 4/8 = 0.5 byte/coord vs 0.375 for tight packing —
slight overhead for kernel simplicity.

Trade-off: 256 head_dim / 8 × 4 bytes = 128 bytes per (token, head) +
4 byte scale = 132 bytes vs 512 fp16 = **3.88× compression**.

For tighter packing (true 5.33×) we'd use 3-byte tight pack but Triton
load/store granularity is uint32 minimum so we accept 3.88×.

================================================================
HELPERS
================================================================

* ``pack_indices_3bit``   — numpy reference for write side
* ``unpack_indices_3bit`` — numpy reference for read side
* ``pack_indices_4bit``   — numpy 4-bit (2 per byte)
* ``unpack_indices_4bit`` — numpy 4-bit unpack
* Triton kernels use inline pack/unpack via shift+or arithmetic

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import numpy as np

GENESIS_G4_TQ_PACKING_MARKER = "Genesis G4-TurboQuant bit-packing v1"


# ─── 3-bit packing (8 indices → uint32) ──────────────────────────────


def pack_indices_3bit_uint32(indices: np.ndarray) -> np.ndarray:
    """Pack 3-bit indices into uint32 words.

    Args:
        indices: shape (..., 8*k), uint8 values in [0..7]. Last dim
                 must be divisible by 8.

    Returns:
        np.ndarray of shape (..., k), uint32 with 8 indices per word.
    """
    arr = np.asarray(indices, dtype=np.uint32)
    *batch, d = arr.shape
    if d % 8 != 0:
        raise ValueError(f"last dim {d} not divisible by 8 for 3-bit pack")
    k = d // 8

    arr = arr.reshape(*batch, k, 8)
    packed = (
        (arr[..., 0]) |
        (arr[..., 1] << 3) |
        (arr[..., 2] << 6) |
        (arr[..., 3] << 9) |
        (arr[..., 4] << 12) |
        (arr[..., 5] << 15) |
        (arr[..., 6] << 18) |
        (arr[..., 7] << 21)
    )
    return packed.astype(np.uint32)


def unpack_indices_3bit_uint32(packed: np.ndarray) -> np.ndarray:
    """Unpack uint32 words back to 3-bit indices (last dim ×8)."""
    p = np.asarray(packed, dtype=np.uint32)
    *batch, k = p.shape
    out = np.empty((*batch, k, 8), dtype=np.uint8)
    out[..., 0] = (p) & 0x7
    out[..., 1] = (p >> 3) & 0x7
    out[..., 2] = (p >> 6) & 0x7
    out[..., 3] = (p >> 9) & 0x7
    out[..., 4] = (p >> 12) & 0x7
    out[..., 5] = (p >> 15) & 0x7
    out[..., 6] = (p >> 18) & 0x7
    out[..., 7] = (p >> 21) & 0x7
    return out.reshape(*batch, k * 8)


# ─── 4-bit packing (2 indices per byte) ──────────────────────────────


def pack_indices_4bit_byte(indices: np.ndarray) -> np.ndarray:
    """Pack 4-bit indices into bytes (2 indices per byte).

    Args:
        indices: shape (..., 2*k), uint8 values in [0..15].

    Returns:
        np.ndarray of shape (..., k), uint8.
    """
    arr = np.asarray(indices, dtype=np.uint8)
    *batch, d = arr.shape
    if d % 2 != 0:
        raise ValueError(f"last dim {d} not divisible by 2 for 4-bit pack")
    k = d // 2
    arr = arr.reshape(*batch, k, 2)
    packed = arr[..., 0] | (arr[..., 1] << 4)
    return packed.astype(np.uint8)


def unpack_indices_4bit_byte(packed: np.ndarray) -> np.ndarray:
    """Unpack bytes back to 4-bit indices (last dim ×2)."""
    p = np.asarray(packed, dtype=np.uint8)
    *batch, k = p.shape
    out = np.empty((*batch, k, 2), dtype=np.uint8)
    out[..., 0] = p & 0x0F
    out[..., 1] = (p >> 4) & 0x0F
    return out.reshape(*batch, k * 2)


# ─── Tight 3-bit packing (true 5.33× — 3 bytes per 8 indices) ────────


def pack_indices_3bit_tight(indices: np.ndarray) -> np.ndarray:
    """Pack 3-bit indices into 3-byte groups (8 indices = 3 bytes).

    Tighter than uint32 packing (1 byte saved per 8 coords). Used when
    storage matters more than kernel simplicity. Read kernel must
    handle non-aligned byte access.

    Args:
        indices: shape (..., 8*k), uint8 values in [0..7].

    Returns:
        np.ndarray of shape (..., 3*k), uint8.
    """
    arr = np.asarray(indices, dtype=np.uint16)
    *batch, d = arr.shape
    if d % 8 != 0:
        raise ValueError(f"last dim {d} not divisible by 8 for tight 3-bit pack")
    k = d // 8
    arr = arr.reshape(*batch, k, 8)

    b0 = (arr[..., 0] | (arr[..., 1] << 3) | ((arr[..., 2] & 0x03) << 6)) & 0xFF
    b1 = (
        ((arr[..., 2] >> 2) & 0x01) |
        (arr[..., 3] << 1) |
        (arr[..., 4] << 4) |
        ((arr[..., 5] & 0x01) << 7)
    ) & 0xFF
    b2 = (
        ((arr[..., 5] >> 1) & 0x03) |
        (arr[..., 6] << 2) |
        (arr[..., 7] << 5)
    ) & 0xFF

    packed = np.stack([b0, b1, b2], axis=-1)  # (..., k, 3)
    return packed.reshape(*batch, k * 3).astype(np.uint8)


def unpack_indices_3bit_tight(packed: np.ndarray) -> np.ndarray:
    """Unpack tight 3-byte groups → 8 indices each."""
    p = np.asarray(packed, dtype=np.uint16)
    *batch, total = p.shape
    if total % 3 != 0:
        raise ValueError(f"last dim {total} not divisible by 3 for tight unpack")
    k = total // 3
    p = p.reshape(*batch, k, 3)
    b0 = p[..., 0]
    b1 = p[..., 1]
    b2 = p[..., 2]
    out = np.empty((*batch, k, 8), dtype=np.uint8)
    out[..., 0] = b0 & 0x07
    out[..., 1] = (b0 >> 3) & 0x07
    out[..., 2] = ((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2)
    out[..., 3] = (b1 >> 1) & 0x07
    out[..., 4] = (b1 >> 4) & 0x07
    out[..., 5] = ((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1)
    out[..., 6] = (b2 >> 2) & 0x07
    out[..., 7] = (b2 >> 5) & 0x07
    return out.reshape(*batch, k * 8)


# ─── Compression ratio helpers ───────────────────────────────────────


def compression_ratio(bits: int, head_dim: int, scale_bytes: int = 4) -> float:
    """Return fp16 / packed bytes ratio for given bit-width.

    Args:
        bits: per-coord bit-width (3 or 4).
        head_dim: per-head dimension.
        scale_bytes: per-vector scale storage (fp32 = 4 bytes typically).

    Returns:
        compression ratio vs fp16 baseline.
    """
    fp16_bytes = head_dim * 2  # 2 bytes per coord
    if bits == 3:
        # uint32 packing: 8 coords → 4 bytes → 0.5 byte/coord
        packed_bytes = (head_dim // 8) * 4 + scale_bytes
    elif bits == 4:
        # 2 coords/byte
        packed_bytes = (head_dim // 2) + scale_bytes
    elif bits == 5:
        # 8 coords → 5 bytes (uint32 + 1 byte spillover) — not optimal
        packed_bytes = (head_dim // 8) * 5 + scale_bytes
    else:
        # Fallback: uint8 = 1 byte/coord
        packed_bytes = head_dim + scale_bytes
    return fp16_bytes / packed_bytes


__all__ = [
    "GENESIS_G4_TQ_PACKING_MARKER",
    "pack_indices_3bit_uint32",
    "unpack_indices_3bit_uint32",
    "pack_indices_4bit_byte",
    "unpack_indices_4bit_byte",
    "pack_indices_3bit_tight",
    "unpack_indices_3bit_tight",
    "compression_ratio",
]
