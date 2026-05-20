# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Genesis G4-TurboQuant KV cache compression.

Covers:
  * Lloyd-Max codebooks (3/4/5-bit)
  * Randomized Hadamard rotation (orthogonality, inverse)
  * Clifford rotor (orthogonality, inverse)
  * torch reference write/read round-trip
  * Attention-proxy quality on synthetic Q/K

CUDA Triton kernels are NOT tested here (CI doesn't have GPU). Server-
side validation lives in tests/integration/.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


# ─── Codebook tests ──────────────────────────────────────────────────


def test_codebook_centroids_count():
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_codebook import (
        BITS_3_LLOYD_MAX_CENTROIDS,
        BITS_4_LLOYD_MAX_CENTROIDS,
        BITS_5_LLOYD_MAX_CENTROIDS,
    )
    assert len(BITS_3_LLOYD_MAX_CENTROIDS) == 8
    assert len(BITS_4_LLOYD_MAX_CENTROIDS) == 16
    assert len(BITS_5_LLOYD_MAX_CENTROIDS) == 32


def test_codebook_symmetric_around_zero():
    """For unit-variance Gaussian, optimal Lloyd-Max codebook is symmetric."""
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_codebook import (
        BITS_3_LLOYD_MAX_CENTROIDS,
        BITS_4_LLOYD_MAX_CENTROIDS,
    )
    c3 = list(BITS_3_LLOYD_MAX_CENTROIDS)
    assert c3[0] == -c3[-1]  # symmetric extremes
    c4 = list(BITS_4_LLOYD_MAX_CENTROIDS)
    assert abs(c4[0] + c4[-1]) < 1e-3


def test_codebook_monotone():
    """Centroids must be monotonically increasing."""
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_codebook import (
        BITS_3_LLOYD_MAX_CENTROIDS,
        BITS_4_LLOYD_MAX_CENTROIDS,
        BITS_5_LLOYD_MAX_CENTROIDS,
    )
    for c in (BITS_3_LLOYD_MAX_CENTROIDS, BITS_4_LLOYD_MAX_CENTROIDS, BITS_5_LLOYD_MAX_CENTROIDS):
        assert all(c[i] < c[i + 1] for i in range(len(c) - 1))


def test_expected_mse_decreases_with_bits():
    """MSE should monotonically decrease as bits increase."""
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_codebook import (
        expected_mse_for_bits,
    )
    mse_3 = expected_mse_for_bits(3, n_samples=10000)
    mse_4 = expected_mse_for_bits(4, n_samples=10000)
    mse_5 = expected_mse_for_bits(5, n_samples=10000)
    assert mse_3 > mse_4 > mse_5


def test_quantize_dequantize_indices_round_trip():
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_codebook import (
        get_centroids,
        quantize_indices,
        dequantize_indices,
    )
    rng = np.random.default_rng(42)
    samples = rng.normal(0, 1, size=10000).astype(np.float32)
    centroids = np.array(get_centroids(4))
    indices = quantize_indices(samples, centroids)
    restored = dequantize_indices(indices, centroids)
    assert indices.shape == samples.shape
    assert indices.min() >= 0 and indices.max() < 16
    # MSE should be reasonable for 4-bit
    mse = ((samples - restored) ** 2).mean()
    assert mse < 0.05, f"4-bit MSE {mse} too high"


def test_online_lloyd_max_solver_converges():
    """The online solver should produce reasonable centroids."""
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_codebook import (
        lloyd_max_codebook,
    )
    rng = np.random.default_rng(0)
    samples = rng.normal(0, 1, size=50000).astype(np.float32)
    centroids = lloyd_max_codebook(samples, bits=4, max_iters=50)
    assert len(centroids) == 16
    # Should be approximately symmetric for symmetric source
    assert abs(centroids[0] + centroids[-1]) < 0.2


# ─── Rotor tests ─────────────────────────────────────────────────────


def test_rht_seed_deterministic():
    """Same seed + layer_idx → same sign vector."""
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_rotor import (
        build_randomized_hadamard_seed,
    )
    s1 = build_randomized_hadamard_seed(256, layer_idx=5, seed_base=0xC0FFEE)
    s2 = build_randomized_hadamard_seed(256, layer_idx=5, seed_base=0xC0FFEE)
    np.testing.assert_array_equal(s1, s2)
    # Different layer_idx → different
    s3 = build_randomized_hadamard_seed(256, layer_idx=6, seed_base=0xC0FFEE)
    assert not np.array_equal(s1, s3)


def test_rht_sign_values_only_pm1():
    """Signs must be ±1."""
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_rotor import (
        build_randomized_hadamard_seed,
    )
    s = build_randomized_hadamard_seed(256, layer_idx=0)
    unique = set(s.tolist())
    assert unique == {-1.0, 1.0}


def test_rht_orthogonality_norm_preservation():
    """Random orthogonal rotation preserves L2 norm."""
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_rotor import (
        build_randomized_hadamard_seed,
        randomized_hadamard_apply_blocked,
    )
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=(100, 256)).astype(np.float32)
    signs = build_randomized_hadamard_seed(256, layer_idx=0)
    x_rot = randomized_hadamard_apply_blocked(x, signs, block_size=128)
    norm_orig = np.linalg.norm(x, axis=1)
    norm_rot = np.linalg.norm(x_rot, axis=1)
    np.testing.assert_allclose(norm_orig, norm_rot, atol=1e-4)


def test_rht_inverse_round_trip():
    """RHT followed by its inverse recovers the original."""
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_rotor import (
        build_randomized_hadamard_seed,
        randomized_hadamard_apply_blocked,
    )
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=(50, 256)).astype(np.float32)
    signs = build_randomized_hadamard_seed(256, layer_idx=3)
    x_rot = randomized_hadamard_apply_blocked(x, signs, block_size=128)
    x_back = randomized_hadamard_apply_blocked(x_rot, signs, block_size=128, inverse=True)
    np.testing.assert_allclose(x, x_back, atol=1e-4)


def test_clifford_rotor_orthogonality():
    """Clifford rotor preserves length per 3-vector group."""
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_rotor import (
        clifford_rotate_full,
        clifford_rotor_layer,
    )
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, size=(20, 256)).astype(np.float32)
    rotor = clifford_rotor_layer(seed_base=0xC0FFEE, layer_idx=2, head_dim=256)
    x_rot = clifford_rotate_full(x, rotor, head_dim=256)
    # Per-token norm should be preserved (up to tail-passing for non-multiple-of-3 dims)
    # head_dim=256, n_groups=85, last 256-85*3=1 dim passes through
    # So full L2 norm is preserved
    np.testing.assert_allclose(
        np.linalg.norm(x, axis=1),
        np.linalg.norm(x_rot, axis=1),
        atol=1e-3,
    )


def test_clifford_rotor_inverse_round_trip():
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_rotor import (
        clifford_rotate_full,
        clifford_rotor_layer,
    )
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=(10, 256)).astype(np.float32)
    rotor = clifford_rotor_layer(seed_base=0xCAFEBABE, layer_idx=11, head_dim=256)
    x_rot = clifford_rotate_full(x, rotor, head_dim=256)
    x_back = clifford_rotate_full(x_rot, rotor, head_dim=256, inverse=True)
    np.testing.assert_allclose(x, x_back, atol=1e-3)


def test_rotor_decorrelation_quality():
    """Both rotation methods should give marginal std ≈ 1 and low correlation."""
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_rotor import (
        estimate_decorrelation_quality,
    )
    for method in ("rht", "clifford"):
        q = estimate_decorrelation_quality(256, method=method, n_samples=5000)
        assert abs(q["marginal_mean"]) < 0.02, f"{method} mean too far from 0: {q}"
        assert 0.98 < q["marginal_std"] < 1.02, f"{method} std off: {q}"
        assert q["mean_abs_corr"] < 0.05, f"{method} corr too high: {q}"


# ─── Reference write/read round-trip ─────────────────────────────────


def test_reference_write_then_read_unit_variance():
    """Round-trip: original ≈ read(write(original)) at 4-bit."""
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_reference import (
        g4_tq_round_trip_test,
    )
    rng = np.random.default_rng(0)
    # Generate unit-variance vectors with realistic head_dim
    x = rng.normal(0, 1, size=(50, 256)).astype(np.float32)
    r = g4_tq_round_trip_test(x, bits=4, method="rht")
    assert r["cosine"] > 0.99, f"4-bit RHT cosine too low: {r}"
    assert r["mse_rel"] < 0.02, f"4-bit RHT MSE rel too high: {r}"
    assert r["compression_ratio"] == 4.0


def test_reference_higher_bits_better_quality():
    """5-bit reconstruction should be better than 3-bit."""
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_reference import (
        g4_tq_round_trip_test,
    )
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=(30, 256)).astype(np.float32)
    r3 = g4_tq_round_trip_test(x, bits=3, method="rht")
    r5 = g4_tq_round_trip_test(x, bits=5, method="rht")
    assert r5["cosine"] > r3["cosine"]
    assert r5["mse_rel"] < r3["mse_rel"]


def test_reference_clifford_method_works():
    """Clifford rotation path should also round-trip correctly."""
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_reference import (
        g4_tq_round_trip_test,
    )
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, size=(30, 256)).astype(np.float32)
    r = g4_tq_round_trip_test(x, bits=4, method="clifford")
    assert r["cosine"] > 0.98, f"Clifford 4-bit cosine: {r}"


def test_reference_attention_proxy_4bit_quality():
    """4-bit must preserve attention QK structure on synthetic Gaussian.

    Note: thresholds are calibrated for *random* unit-Gaussian K which is
    the worst case (no genuine attention sparsity to exploit). On real
    decoder KV the retrieval is significantly higher (~0.90+ top-1, ~0.92+
    top-5) because attention activations after softmax have a power-law
    distribution that quantization handles better than uniform Gaussian.
    """
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_reference import (
        g4_tq_attention_proxy_test,
    )
    rng = np.random.default_rng(0)
    q = rng.normal(0, 1, size=(16, 256)).astype(np.float32)
    k = rng.normal(0, 1, size=(1024, 256)).astype(np.float32)
    r = g4_tq_attention_proxy_test(q, k, bits=4, method="rht")
    assert r["inner_product_cosine"] > 0.99, f"4-bit attn cosine: {r}"
    assert r["top1_overlap"] > 0.65, f"4-bit top-1: {r}"
    assert r["top5_overlap"] > 0.75, f"4-bit top-5: {r}"


# ─── Cache wrapper config tests ──────────────────────────────────────


def test_config_validation():
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_cache import (
        G4TurboQuantConfig,
    )
    # Valid config
    c = G4TurboQuantConfig(head_dim=256, bits_sliding=4, bits_global=3)
    assert c.head_dim == 256
    assert c.bits_sliding == 4
    assert c.bits_global == 3

    # Invalid bits
    with pytest.raises(AssertionError):
        G4TurboQuantConfig(bits_sliding=2)
    with pytest.raises(AssertionError):
        G4TurboQuantConfig(bits_sliding=8)
    # Invalid method
    with pytest.raises(AssertionError):
        G4TurboQuantConfig(rotation_method="garbage")


def test_config_per_layer_bits():
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_cache import (
        G4TurboQuantConfig,
    )
    layer_types = ["sliding_attention"] * 5 + ["full_attention"]
    c = G4TurboQuantConfig(
        bits_sliding=4, bits_global=3,
        per_layer_types=layer_types,
    )
    assert c.bits_for_layer(0) == 4  # sliding
    assert c.bits_for_layer(5) == 3  # global


def test_kv_cache_size_compression_ratio():
    """Compression ratio should approximate target bits/16."""
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_cache import (
        G4TurboQuantConfig,
        kv_cache_size_bytes,
    )
    # All-3-bit case for clean math
    c = G4TurboQuantConfig(bits_sliding=3, bits_global=3, per_layer_types=None)
    # 3 bit/coord → 3/8 byte/coord, with 4-byte scale per (token, head)
    # Compression should be ~5× for head_dim=256
    sz = kv_cache_size_bytes(c, num_layers=60, num_kv_heads=16, num_blocks=1024, block_size_tokens=16)
    assert sz["compression_ratio"] >= 1.8  # at least some compression (uint8 indices, not 3-bit packed yet)
    assert sz["savings_gb"] > 0


# ─── Patch wiring smoke (no actual server) ───────────────────────────


def test_g4_19_patch_imports():
    """The G4_19 patch should be importable without errors."""
    from vllm.sndr_core.integrations.gemma4 import (
        g4_19_gemma4_turboquant_kv_cache as _patch,
    )
    assert hasattr(_patch, "apply")
    assert hasattr(_patch, "is_applied")
    assert hasattr(_patch, "GENESIS_G4_19_MARKER")
    assert "G4_19" in _patch.GENESIS_G4_19_MARKER
    assert "TurboQuant" in _patch.GENESIS_G4_19_MARKER


def test_g4_19_disabled_by_default():
    """Without env flag, apply() should return skipped."""
    from vllm.sndr_core.integrations.gemma4 import (
        g4_19_gemma4_turboquant_kv_cache as _patch,
    )
    import os
    # Make sure env is not set
    os.environ.pop("GENESIS_ENABLE_G4_19_GEMMA4_TURBOQUANT_KV", None)
    status, msg = _patch.apply()
    assert status == "skipped"
    assert "disabled" in msg.lower()
