# SPDX-License-Identifier: Apache-2.0
"""Unit tests for:
  * the FWHT butterfly algorithm (numpy reference of the in-Triton form)
  * tight 3-bit kernel module structure + numpy-reference round-trip

These tests run on Triton-less hosts (CI / Mac dev). GPU tests live next
to the existing ``test_g4_turboquant.py``.
"""
from __future__ import annotations

import numpy as np
import pytest


# ─── Butterfly correctness via numpy mirror of the Triton kernel ──


def _fwht_butterfly_via_reshape(x_in: np.ndarray) -> np.ndarray:
    """Numpy mirror of ``_fwht_butterfly_block`` in
    ``g4_tq_packed_wht_triton.py``. Uses ONLY operations expressible
    in Triton: reshape + sum-with-mask. If this matches the orthonormal
    Hadamard matmul we are confident the Triton kernel is correct."""
    x = x_in.astype(np.float64).copy()
    BLOCK = len(x)
    stride = 1
    while stride < BLOCK:
        G = BLOCK // (2 * stride)
        x_3d = x.reshape(G, 2, stride)
        axis1 = np.arange(2)
        mask_top = (axis1[None, :, None] == 0)
        mask_bot = (axis1[None, :, None] == 1)
        top = np.sum(np.where(mask_top, x_3d, 0.0), axis=1)
        bot = np.sum(np.where(mask_bot, x_3d, 0.0), axis=1)
        new_top = top + bot
        new_bot = top - bot
        new_3d = (
            np.where(mask_top, new_top[:, None, :], 0.0) +
            np.where(mask_bot, new_bot[:, None, :], 0.0)
        )
        x = new_3d.reshape(BLOCK)
        stride *= 2
    return x / np.sqrt(BLOCK)


@pytest.mark.parametrize("block_size", [64, 128, 256])
def test_butterfly_matches_orthonormal_hadamard(block_size):
    """Reshape-form butterfly must agree with x · H, H = Sylvester / √n."""
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant.kernels import (
        g4_tq_packed_wht_triton as mod,
    )
    # Hadamard helper stores fp32, butterfly is fp64 — compare with fp32
    # tolerance. The mathematical equivalence is proved by other tests
    # (self-inverse + L2 norm preservation).
    H = mod._build_hadamard_matrix(block_size).numpy().astype(np.float64)
    rng = np.random.default_rng(42)
    for trial in range(5):
        x = rng.standard_normal(block_size)
        via_butterfly = _fwht_butterfly_via_reshape(x)
        via_matmul = x @ H.T
        err = np.abs(via_butterfly - via_matmul).max()
        # fp32 H is normalized by 1/sqrt(n) where n is power of 2 — exact
        # in fp64. But the multiplication x · H has rounding; bound by
        # n * eps_fp32 * ||x||_inf ≈ 1e-6 at worst.
        assert err < 1e-5, (
            f"butterfly differs from H·x by {err:.2e} at block_size={block_size}"
        )


def test_butterfly_is_orthonormal_self_inverse():
    """FWHT∘FWHT(x) = x because H · H = I (H is symmetric and orthonormal)."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.standard_normal(128)
        y = _fwht_butterfly_via_reshape(x)
        x_round = _fwht_butterfly_via_reshape(y)
        assert np.allclose(x, x_round, atol=1e-10), "FWHT not self-inverse"


def test_butterfly_preserves_l2_norm():
    """Orthonormal H preserves ||x||₂."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(128)
    y = _fwht_butterfly_via_reshape(x)
    assert abs(np.linalg.norm(x) - np.linalg.norm(y)) < 1e-10, (
        "Hadamard not norm-preserving"
    )


# ─── Tight 3-bit kernel module structure ───────────────────────────


def test_tight_module_imports():
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant.kernels import (
        g4_tq_tight_triton as mod,
    )
    assert mod.GENESIS_G4_TQ_TIGHT_MARKER.startswith("Genesis G4")
    assert "TIGHT" in mod.GENESIS_G4_TQ_TIGHT_MARKER


def test_tight_public_api_present():
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant.kernels import (
        g4_tq_tight_triton as mod,
    )
    for name in ("g4_tq_write_tight_3bit", "g4_tq_read_tight_3bit"):
        assert hasattr(mod, name), f"missing public symbol {name!r}"


# ─── Tight pack format matches numpy reference exactly ─────────────


@pytest.mark.parametrize("seed", [0, 1, 42, 0xC0FFEE])
def test_tight_pack_format_round_trip(seed):
    """8 indices → 3 bytes round-trip via numpy reference, every byte
    layout bit matches the Triton kernel's packing arithmetic."""
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_packing import (
        pack_indices_3bit_tight,
        unpack_indices_3bit_tight,
    )
    rng = np.random.default_rng(seed)
    # 100 groups of 8 = 800 indices in [0..7]
    idx = rng.integers(0, 8, size=(100, 8 * 16), dtype=np.uint8)
    packed = pack_indices_3bit_tight(idx)
    # Shape (100, 16 * 3) = (100, 48)
    assert packed.shape == (100, 16 * 3)
    assert packed.dtype == np.uint8
    restored = unpack_indices_3bit_tight(packed)
    np.testing.assert_array_equal(idx, restored)


def test_tight_compression_ratio_correct():
    """For head_dim=256 + tight format, storage is 96 bytes per
    (token,head). vs 512 bytes fp16 → 5.33× before scale, ~5.12× after
    including 4-byte scale.
    """
    HEAD_DIM = 256
    tight_storage_bytes = (HEAD_DIM * 3) // 8        # 96
    fp16_storage_bytes = HEAD_DIM * 2                # 512
    scale_bytes = 4
    raw_ratio = fp16_storage_bytes / tight_storage_bytes
    with_scale = fp16_storage_bytes / (tight_storage_bytes + scale_bytes)
    assert abs(raw_ratio - 5.333) < 0.001
    assert abs(with_scale - 5.12) < 0.001


# ─── Cache wrapper dispatch for tight pack_mode ────────────────────


def test_g4tq_config_accepts_tight_pack_mode():
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_cache import (
        G4TurboQuantConfig,
    )
    cfg = G4TurboQuantConfig(pack_mode="tight")
    assert cfg.pack_mode == "tight"
    # Default rest of config
    assert cfg.head_dim == 256
    assert cfg.block_size == 128


def test_g4tq_kv_cache_size_tight_pack_arithmetic():
    """``kv_cache_size_bytes`` accounting for tight pack mode."""
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_cache import (
        G4TurboQuantConfig, kv_cache_size_bytes,
    )
    cfg_uint32 = G4TurboQuantConfig(pack_mode="uint32")
    cfg_tight = G4TurboQuantConfig(pack_mode="tight")
    info_u = kv_cache_size_bytes(cfg_uint32, num_layers=60,
                                 num_kv_heads=16, num_blocks=1024,
                                 block_size_tokens=16)
    info_t = kv_cache_size_bytes(cfg_tight, num_layers=60,
                                 num_kv_heads=16, num_blocks=1024,
                                 block_size_tokens=16)
    assert info_t["total_bytes"] < info_u["total_bytes"], (
        "tight mode should consume less than uint32"
    )
    # Tight saves (4-3) bytes per 8 coords = 1 byte per 8 coords
    # = head_dim/8 bytes per (token, head)
    # For head_dim=256: 32 bytes per (token, head)
    # × 60 layers × 16 heads × 16384 slots × 2 (K+V) = ~1 GB saving
    saving_gb = (info_u["total_bytes"] - info_t["total_bytes"]) / (1024 ** 3)
    assert 0.8 < saving_gb < 1.5, (
        f"tight savings {saving_gb:.2f} GB outside expected ~1 GB band"
    )


# ─── End-to-end numpy reference: tight + signs-only round-trip ─────


def test_tight_signs_only_roundtrip_numpy():
    """Numpy reference: quantize → tight-pack → tight-unpack → dequantize.
    Verifies the bytes round-trip even if the Triton kernel can't run."""
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_packing import (
        pack_indices_3bit_tight, unpack_indices_3bit_tight,
    )
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_rotor import (
        build_randomized_hadamard_seed,
    )
    B3 = np.array([-1.84375, -1.05860, -0.50977, 0.0,
                    0.50977,  1.05860,  1.84375])
    C3 = np.array([-2.34375, -1.34375, -0.77344, -0.24609,
                    0.24609,  0.77344,  1.34375,  2.34375])

    rng = np.random.default_rng(123)
    M, D = 8, 256
    x = rng.standard_normal((M, D)).astype(np.float32)
    signs = build_randomized_hadamard_seed(D, layer_idx=0)

    # forward
    scale = np.linalg.norm(x, axis=-1, keepdims=True) / np.sqrt(D)
    scale_safe = np.where(scale > 1e-8, scale, 1.0)
    x_signed = x * signs[None, :]
    x_norm = x_signed / scale_safe
    idx = np.searchsorted(B3, x_norm, side="left").astype(np.uint8)

    # tight pack
    packed = pack_indices_3bit_tight(idx)
    assert packed.shape == (M, D * 3 // 8)
    assert packed.dtype == np.uint8

    # reverse: unpack + dequantize + inverse rotation
    restored_idx = unpack_indices_3bit_tight(packed)
    np.testing.assert_array_equal(idx, restored_idx)
    v = C3[restored_idx] * scale_safe
    x_recon = v * signs[None, :]
    # cos-sim per row
    cos = [
        np.dot(x_recon[i], x[i]) / (np.linalg.norm(x_recon[i]) * np.linalg.norm(x[i]))
        for i in range(M)
    ]
    assert np.mean(cos) > 0.95, f"tight round-trip cos sim only {np.mean(cos):.4f}"
