# SPDX-License-Identifier: Apache-2.0
"""Unit tests for G4-TurboQuant bit-packing.

Tests round-trip equivalence for 3-bit and 4-bit pack/unpack operations.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_pack_3bit_uint32_round_trip():
    """8 × 3-bit indices → uint32 → 8 indices unchanged."""
    from sndr.engines.vllm.patches.attention.turboquant.kernels.g4_tq_packing import (
        pack_indices_3bit_uint32,
        unpack_indices_3bit_uint32,
    )
    rng = np.random.default_rng(0)
    # 10 batch × 256 head_dim → 32 packed words per batch element
    indices = rng.integers(0, 8, size=(10, 256), dtype=np.uint8)
    packed = pack_indices_3bit_uint32(indices)
    assert packed.shape == (10, 32)
    assert packed.dtype == np.uint32
    restored = unpack_indices_3bit_uint32(packed)
    assert restored.shape == indices.shape
    np.testing.assert_array_equal(indices, restored)


def test_pack_3bit_all_zero():
    from sndr.engines.vllm.patches.attention.turboquant.kernels.g4_tq_packing import (
        pack_indices_3bit_uint32,
        unpack_indices_3bit_uint32,
    )
    indices = np.zeros((4, 8), dtype=np.uint8)
    packed = pack_indices_3bit_uint32(indices)
    np.testing.assert_array_equal(packed, 0)
    restored = unpack_indices_3bit_uint32(packed)
    np.testing.assert_array_equal(restored, 0)


def test_pack_3bit_all_max():
    """All indices = 7 → packed word = 0x00FFFFFF (24 bits set)."""
    from sndr.engines.vllm.patches.attention.turboquant.kernels.g4_tq_packing import (
        pack_indices_3bit_uint32,
        unpack_indices_3bit_uint32,
    )
    indices = np.full((1, 8), 7, dtype=np.uint8)
    packed = pack_indices_3bit_uint32(indices)
    # 8 × 3 bits = 24 bits all set
    expected = (
        7 | (7 << 3) | (7 << 6) | (7 << 9) |
        (7 << 12) | (7 << 15) | (7 << 18) | (7 << 21)
    )
    assert packed[0, 0] == expected
    restored = unpack_indices_3bit_uint32(packed)
    np.testing.assert_array_equal(restored, indices)


def test_pack_3bit_specific_pattern():
    """Verify specific known-value placement."""
    from sndr.engines.vllm.patches.attention.turboquant.kernels.g4_tq_packing import (
        pack_indices_3bit_uint32,
    )
    indices = np.array([[1, 2, 3, 4, 5, 6, 7, 0]], dtype=np.uint8)
    packed = pack_indices_3bit_uint32(indices)
    # word = 1 | (2 << 3) | (3 << 6) | (4 << 9) | (5 << 12) | (6 << 15) | (7 << 18) | (0 << 21)
    expected = 1 | (2 << 3) | (3 << 6) | (4 << 9) | (5 << 12) | (6 << 15) | (7 << 18)
    assert packed[0, 0] == expected


def test_pack_4bit_byte_round_trip():
    from sndr.engines.vllm.patches.attention.turboquant.kernels.g4_tq_packing import (
        pack_indices_4bit_byte,
        unpack_indices_4bit_byte,
    )
    rng = np.random.default_rng(1)
    indices = rng.integers(0, 16, size=(8, 256), dtype=np.uint8)
    packed = pack_indices_4bit_byte(indices)
    assert packed.shape == (8, 128)
    assert packed.dtype == np.uint8
    restored = unpack_indices_4bit_byte(packed)
    assert restored.shape == indices.shape
    np.testing.assert_array_equal(indices, restored)


def test_pack_3bit_tight_round_trip():
    """Tight 3-byte/8-index packing."""
    from sndr.engines.vllm.patches.attention.turboquant.kernels.g4_tq_packing import (
        pack_indices_3bit_tight,
        unpack_indices_3bit_tight,
    )
    rng = np.random.default_rng(2)
    indices = rng.integers(0, 8, size=(5, 256), dtype=np.uint8)
    packed = pack_indices_3bit_tight(indices)
    assert packed.shape == (5, 96)  # 256 // 8 * 3 = 96 bytes
    restored = unpack_indices_3bit_tight(packed)
    np.testing.assert_array_equal(indices, restored)


def test_compression_ratio_3bit():
    from sndr.engines.vllm.patches.attention.turboquant.kernels.g4_tq_packing import (
        compression_ratio,
    )
    # head_dim=256, 3-bit uint32 packing: 32 × 4 bytes + 4 byte scale = 132 bytes
    # fp16: 512 bytes
    # ratio: 512 / 132 ≈ 3.88
    r = compression_ratio(3, 256, scale_bytes=4)
    assert 3.5 < r < 4.0, f"3-bit ratio={r}"


def test_compression_ratio_4bit():
    from sndr.engines.vllm.patches.attention.turboquant.kernels.g4_tq_packing import (
        compression_ratio,
    )
    # head_dim=256, 4-bit: 128 bytes + 4 = 132 bytes; same as 3-bit uint32
    r = compression_ratio(4, 256, scale_bytes=4)
    assert 3.5 < r < 4.0


def test_pack_modules_importable():
    """Just confirm module structure exists."""
    from sndr.engines.vllm.patches.attention.turboquant.kernels import g4_tq_packing
    assert hasattr(g4_tq_packing, "pack_indices_3bit_uint32")
    assert hasattr(g4_tq_packing, "unpack_indices_3bit_uint32")
    assert hasattr(g4_tq_packing, "pack_indices_4bit_byte")
    assert hasattr(g4_tq_packing, "compression_ratio")
    assert g4_tq_packing.GENESIS_G4_TQ_PACKING_MARKER.startswith("Genesis G4")


def test_packed_triton_module_importable():
    """Triton kernel module imports without crashing (kernel won't fire without CUDA)."""
    try:
        from sndr.engines.vllm.patches.attention.turboquant.kernels import (
            g4_tq_packed_triton,
        )
    except ImportError as e:
        if "torch" in str(e) or "triton" in str(e):
            pytest.skip(f"requires torch/triton: {e}")
        raise
    assert hasattr(g4_tq_packed_triton, "g4_tq_write_packed_3bit")
    assert hasattr(g4_tq_packed_triton, "g4_tq_read_packed_3bit")
    assert "PACKED" in g4_tq_packed_triton.GENESIS_G4_TQ_PACKED_MARKER
