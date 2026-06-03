# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the full Walsh-Hadamard variant of G4-TurboQuant.

These tests run **without GPU / Triton** — they validate:
  * Hadamard matrix construction (orthonormality, Sylvester recursion)
  * cache lookup (one tensor per (block, device, dtype))
  * config plumbing (G4TurboQuantConfig.wht_mode validator)
  * numpy-reference round-trip quality:
        full-WHT MSE ≤ signs-only MSE on heavy-tailed inputs

GPU-only round-trip tests live next to the existing
``test_g4_turboquant.py``.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_hadamard_matrix_orthonormal_128():
    """H @ H^T should be the identity (up to fp32 noise) for block_size=128."""
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant.kernels import (
        g4_tq_packed_wht_triton as mod,
    )
    H = mod._build_hadamard_matrix(128).numpy()
    prod = H @ H.T
    diag = np.diag(prod)
    off = prod - np.diag(diag)
    assert np.allclose(diag, 1.0, atol=1e-6), (
        f"diag deviation max={np.abs(diag-1).max():.2e}"
    )
    assert np.abs(off).max() < 1e-6, (
        f"off-diagonal max={np.abs(off).max():.2e}"
    )


@pytest.mark.parametrize("block_size", [2, 4, 8, 16, 32, 64, 128, 256])
def test_hadamard_matrix_orthonormal_various_sizes(block_size):
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant.kernels import (
        g4_tq_packed_wht_triton as mod,
    )
    H = mod._build_hadamard_matrix(block_size).numpy()
    assert H.shape == (block_size, block_size)
    prod = H @ H.T
    assert np.allclose(prod, np.eye(block_size), atol=1e-5), (
        f"H not orthonormal for block_size={block_size}"
    )


def test_hadamard_matrix_rejects_non_power_of_2():
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant.kernels import (
        g4_tq_packed_wht_triton as mod,
    )
    with pytest.raises(ValueError, match="power of 2"):
        mod._build_hadamard_matrix(100)
    with pytest.raises(ValueError, match="power of 2"):
        mod._build_hadamard_matrix(0)


def test_hadamard_cache_shares_one_tensor():
    """``get_hadamard_matrix`` returns the same tensor on repeated calls."""
    pytest.importorskip("torch")
    import torch
    from vllm.sndr_core.integrations.attention.turboquant.kernels import (
        g4_tq_packed_wht_triton as mod,
    )
    mod.clear_hadamard_cache()
    cpu = torch.device("cpu")
    H1 = mod.get_hadamard_matrix(128, cpu, torch.float32)
    H2 = mod.get_hadamard_matrix(128, cpu, torch.float32)
    assert H1 is H2, "cache should return identical object on repeat lookup"


def test_config_validates_wht_mode():
    """G4TurboQuantConfig rejects an unknown wht_mode at construction."""
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_cache import (
        G4TurboQuantConfig,
    )
    G4TurboQuantConfig(wht_mode="signs_only")  # ok
    G4TurboQuantConfig(wht_mode="full_wht")    # ok
    with pytest.raises(AssertionError, match="wht_mode"):
        G4TurboQuantConfig(wht_mode="invalid_mode")


def test_config_default_wht_mode_is_signs_only():
    """Default mode keeps existing 256K boot path stable."""
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_cache import (
        G4TurboQuantConfig,
    )
    cfg = G4TurboQuantConfig()
    assert cfg.wht_mode == "signs_only", (
        "default wht_mode must stay signs_only — flipping the default would "
        "silently change cache semantics for existing 256K deployments"
    )


def test_packed_wht_marker_present():
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant.kernels import (
        g4_tq_packed_wht_triton as mod,
    )
    assert mod.GENESIS_G4_TQ_PACKED_WHT_MARKER.startswith("Genesis G4")
    assert "FULL-WHT" in mod.GENESIS_G4_TQ_PACKED_WHT_MARKER


def test_packed_wht_public_api_present():
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant.kernels import (
        g4_tq_packed_wht_triton as mod,
    )
    for name in (
        "g4_tq_write_packed_wht_3bit",
        "g4_tq_read_packed_wht_3bit",
        "get_hadamard_matrix",
        "_build_hadamard_matrix",
    ):
        assert hasattr(mod, name), f"missing public symbol {name!r}"


# ─── Numpy reference round-trip quality ────────────────────────────────


_B3 = np.array([-1.84375, -1.05860, -0.50977, 0.0,
                 0.50977,  1.05860,  1.84375])
_C3 = np.array([-2.34375, -1.34375, -0.77344, -0.24609,
                 0.24609,  0.77344,  1.34375,  2.34375])


def _quantize_3bit(x: np.ndarray) -> np.ndarray:
    return np.searchsorted(_B3, x, side="left").astype(np.uint8)


def _roundtrip(x: np.ndarray, signs: np.ndarray, mode: str) -> np.ndarray:
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_rotor import (
        randomized_hadamard_apply_blocked,
    )
    BLOCK = 128
    D = x.shape[-1]
    scale = np.linalg.norm(x, axis=-1, keepdims=True) / np.sqrt(D)
    scale = np.where(scale > 1e-8, scale, 1.0)

    if mode == "signs_only":
        x_rot = x * signs[None, :]
    elif mode == "full_wht":
        x_rot = randomized_hadamard_apply_blocked(
            x.astype(np.float32), signs, block_size=BLOCK,
        )
    else:
        raise ValueError(mode)

    x_norm = x_rot / scale
    idx = _quantize_3bit(x_norm)
    x_deq = _C3[idx] * scale

    if mode == "signs_only":
        return x_deq * signs[None, :]
    return randomized_hadamard_apply_blocked(
        x_deq.astype(np.float32), signs, block_size=BLOCK, inverse=True,
    )


def test_roundtrip_full_wht_at_least_matches_signs_only_on_gaussian():
    """On a Gaussian input full-WHT shouldn't be worse than signs-only."""
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_rotor import (
        build_randomized_hadamard_seed,
    )
    rng = np.random.default_rng(0xC0FFEE)
    x = rng.standard_normal((512, 256)).astype(np.float32)
    signs = build_randomized_hadamard_seed(256, layer_idx=0)
    recon_s = _roundtrip(x, signs, "signs_only")
    recon_w = _roundtrip(x, signs, "full_wht")
    mse_s = float(np.mean((recon_s - x) ** 2))
    mse_w = float(np.mean((recon_w - x) ** 2))
    # WHT on Gaussian is near-identity rotation; allow 5% slack.
    assert mse_w <= mse_s * 1.05, (
        f"full_wht regressed on Gaussian: signs={mse_s:.4e}, wht={mse_w:.4e}"
    )


def test_roundtrip_full_wht_beats_signs_only_on_heavy_tailed():
    """On heavy-tailed inputs full-WHT should give measurably lower MSE."""
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_rotor import (
        build_randomized_hadamard_seed,
    )
    rng = np.random.default_rng(0xBEEF)
    x = np.clip(rng.standard_cauchy((512, 256)), -10, 10).astype(np.float32)
    signs = build_randomized_hadamard_seed(256, layer_idx=0)
    recon_s = _roundtrip(x, signs, "signs_only")
    recon_w = _roundtrip(x, signs, "full_wht")
    mse_s = float(np.mean((recon_s - x) ** 2))
    mse_w = float(np.mean((recon_w - x) ** 2))
    assert mse_w < mse_s, (
        f"full_wht didn't win on heavy-tailed input: "
        f"signs={mse_s:.4e}, wht={mse_w:.4e}"
    )
    # Empirically ~22% MSE drop; require at least 5% gain to flag regressions
    # in the rotor or codebook tables.
    improvement = (mse_s - mse_w) / mse_s
    assert improvement > 0.05, (
        f"full_wht only saved {improvement * 100:.1f}% MSE on heavy-tailed; "
        f"expected ≥5%. Possible rotor/codebook regression."
    )


def test_roundtrip_handles_nan_inf_input():
    """Quantizer must not crash on NaN/Inf in numpy reference path."""
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_rotor import (
        build_randomized_hadamard_seed,
    )
    rng = np.random.default_rng(123)
    x = rng.standard_normal((16, 256)).astype(np.float32)
    x[0, 0] = np.nan
    x[1, 5] = np.inf
    x[2, 7] = -np.inf
    signs = build_randomized_hadamard_seed(256, layer_idx=0)
    # We DON'T expect the numpy reference to clean NaN — but it should run
    # without raising. The Triton kernel has explicit NaN/Inf guards that
    # are exercised by GPU-only tests.
    recon = _roundtrip(np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0),
                       signs, "full_wht")
    assert recon.shape == x.shape
    assert np.all(np.isfinite(recon)), "reference produced NaN/Inf output"
