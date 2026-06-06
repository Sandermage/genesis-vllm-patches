# SPDX-License-Identifier: Apache-2.0
"""Torch reference implementations of G4-TurboQuant.

These are **slow** (no fusion, no GPU optimization) but **correct** —
they serve as the source-of-truth for unit-testing the Triton kernels.

The Triton kernels in ``g4_tq_write_triton.py`` and ``g4_tq_read_triton.py``
are validated against these references using:
  * Element-wise cosine similarity ≥ 0.9999
  * MSE ≤ 1e-4 (in unit-variance pre-rotation regime)
  * Top-k retrieval accuracy match at 1, 5, 10

These tests run in CI without CUDA.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .g4_tq_codebook import (
    boundaries_from_centroids,
    get_centroids,
)
from .g4_tq_rotor import (
    build_randomized_hadamard_seed,
    clifford_rotor_layer,
    clifford_rotate_full,
    randomized_hadamard_apply_blocked,
)

GENESIS_G4_TQ_REFERENCE_MARKER = "Genesis G4-TurboQuant reference impl v1"


# ─── Per-vector statistics (scale factor for unit-variance assumption) ─


def compute_scale_per_vector(x: np.ndarray) -> np.ndarray:
    """Return per-vector L2 norm / √head_dim (the scale to normalize).

    Lloyd-Max codebooks are designed for unit-variance marginals; we
    scale each vector to unit variance before quantization and store
    the scale alongside the indices.

    Args:
        x: shape (..., head_dim)

    Returns:
        np.ndarray of shape x.shape[:-1] — per-vector scale factor.
    """
    head_dim = x.shape[-1]
    return np.linalg.norm(x, axis=-1) / np.sqrt(head_dim)


# ─── Write path: rotate → quantize → pack ────────────────────────────


def g4_tq_write_reference(
    x: np.ndarray,
    signs: np.ndarray,
    bits: int,
    block_size: int = 128,
    rotor: Optional[np.ndarray] = None,
    method: str = "rht",
) -> tuple[np.ndarray, np.ndarray]:
    """Reference write path: rotate KV vector, quantize each coord.

    Args:
        x: shape (..., head_dim), raw KV vector (any dtype).
        signs: shape (head_dim,) — RHT sign vector (used if method='rht').
        bits: 3, 4, or 5 bits per coordinate.
        block_size: WHT block size (default 128 for head_dim=256).
        rotor: shape (head_dim//3, 4) — Clifford rotor (used if method='clifford').
        method: 'rht' or 'clifford'.

    Returns:
        Tuple of:
          * indices: shape (..., head_dim), dtype int8/uint8 — bin indices
          * scale: shape (...,), float32 — per-vector L2/√d scale factor
    """
    centroids = np.array(get_centroids(bits), dtype=np.float32)
    boundaries = boundaries_from_centroids(centroids)

    # Step 1: rotate
    x_f32 = x.astype(np.float32)
    if method == "rht":
        x_rot = randomized_hadamard_apply_blocked(x_f32, signs, block_size=block_size)
    elif method == "clifford":
        x_rot = clifford_rotate_full(x_f32, rotor, head_dim=x.shape[-1])
    else:
        raise ValueError(f"unknown method {method}")

    # Step 2: compute per-vector scale
    scale = compute_scale_per_vector(x_rot).astype(np.float32)
    # Avoid divide-by-zero on degenerate inputs
    scale_safe = np.where(scale > 1e-8, scale, 1.0).astype(np.float32)

    # Step 3: normalize
    x_norm = x_rot / scale_safe[..., None]

    # Step 4: quantize to bin indices via boundary lookup
    indices = np.searchsorted(boundaries, x_norm).clip(0, len(centroids) - 1)
    indices = indices.astype(np.uint8)

    return indices, scale


# ─── Read path: unpack → dequantize → unrotate ───────────────────────


def g4_tq_read_reference(
    indices: np.ndarray,
    scale: np.ndarray,
    signs: np.ndarray,
    bits: int,
    block_size: int = 128,
    rotor: Optional[np.ndarray] = None,
    method: str = "rht",
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Reference read path: dequantize then inverse-rotate.

    Args:
        indices: shape (..., head_dim), uint8 bin indices.
        scale: shape (...,), float32 per-vector L2/√d.
        signs: shape (head_dim,) — sign vector used in write.
        bits: must match write.
        block_size: WHT block size.
        rotor: Clifford rotor params.
        method: 'rht' or 'clifford'.
        dtype: output dtype (typically fp16/bf16/fp32).

    Returns:
        Reconstructed vector of shape indices.shape (last dim = head_dim).
    """
    centroids = np.array(get_centroids(bits), dtype=np.float32)

    # Step 1: dequantize
    x_norm = centroids[indices.astype(np.int64)]  # (..., head_dim)

    # Step 2: re-scale (undo per-vector normalization)
    x_rot = x_norm * scale[..., None]

    # Step 3: inverse rotation
    if method == "rht":
        x = randomized_hadamard_apply_blocked(
            x_rot, signs, block_size=block_size, inverse=True
        )
    elif method == "clifford":
        x = clifford_rotate_full(
            x_rot, rotor, head_dim=indices.shape[-1], inverse=True
        )
    else:
        raise ValueError(f"unknown method {method}")

    return x.astype(dtype)


# ─── Round-trip quality check ────────────────────────────────────────


def g4_tq_round_trip_test(
    x: np.ndarray,
    bits: int,
    method: str = "rht",
    layer_idx: int = 0,
    seed_base: int = 0xC0FFEE,
    block_size: int = 128,
) -> dict:
    """Encode then decode a tensor; return reconstruction quality metrics.

    Used in unit tests to verify codebook + rotation is well-tuned.

    Args:
        x: shape (..., head_dim) — original KV vectors.
        bits: 3, 4, or 5.
        method: 'rht' or 'clifford'.
        layer_idx: per-layer rotation seed offset.
        seed_base: model-wide seed.

    Returns:
        dict with metrics:
          * mse: mean squared error
          * cosine: per-vector cosine similarity (mean)
          * mse_rel: MSE / variance(x) — relative MSE
    """
    head_dim = x.shape[-1]

    if method == "rht":
        signs = build_randomized_hadamard_seed(head_dim, layer_idx, seed_base)
        rotor = None
    else:
        signs = None
        rotor = clifford_rotor_layer(seed_base, layer_idx, head_dim)

    indices, scale = g4_tq_write_reference(
        x, signs=signs, bits=bits,
        block_size=block_size, rotor=rotor, method=method,
    )
    x_recon = g4_tq_read_reference(
        indices, scale, signs=signs, bits=bits,
        block_size=block_size, rotor=rotor, method=method,
        dtype=np.float32,
    )

    diff = x.astype(np.float32) - x_recon
    mse = float((diff * diff).mean())
    var = float(x.astype(np.float32).var())

    # Per-vector cosine similarity
    x_flat = x.astype(np.float32).reshape(-1, head_dim)
    r_flat = x_recon.reshape(-1, head_dim)
    dot = (x_flat * r_flat).sum(axis=1)
    norm_x = np.linalg.norm(x_flat, axis=1)
    norm_r = np.linalg.norm(r_flat, axis=1)
    cosine = float((dot / (norm_x * norm_r + 1e-12)).mean())

    return {
        "mse": mse,
        "mse_rel": mse / (var + 1e-12),
        "cosine": cosine,
        "compression_ratio": 16.0 / bits,  # vs fp16
    }


# ─── Inner-product preservation test (attention math correctness) ────


def g4_tq_attention_proxy_test(
    q: np.ndarray,
    k: np.ndarray,
    bits: int,
    method: str = "rht",
    layer_idx: int = 0,
    seed_base: int = 0xC0FFEE,
) -> dict:
    """Test how well TurboQuant preserves Q·K^T inner products.

    This is the **actual** quality metric that matters for attention.
    Even if reconstruction MSE is small, attention quality depends on
    rank-ordering of dot products.

    Args:
        q: shape (M, head_dim), queries (NOT quantized).
        k: shape (N, head_dim), keys (will be quantized + restored).
        bits, method, layer_idx, seed_base: same as round_trip.

    Returns:
        dict with:
          * inner_product_cosine: cosine similarity of Q·K vs Q·K_recon
          * top1_overlap: fraction of (M) queries where top-1 K matches
          * top5_overlap: same for top-5
    """
    head_dim = k.shape[-1]
    if method == "rht":
        signs = build_randomized_hadamard_seed(head_dim, layer_idx, seed_base)
        rotor = None
    else:
        signs = None
        rotor = clifford_rotor_layer(seed_base, layer_idx, head_dim)

    indices, scale = g4_tq_write_reference(
        k, signs=signs, bits=bits, rotor=rotor, method=method
    )
    k_recon = g4_tq_read_reference(
        indices, scale, signs=signs, bits=bits, rotor=rotor, method=method,
        dtype=np.float32,
    )

    qk = q.astype(np.float32) @ k.astype(np.float32).T
    qk_recon = q.astype(np.float32) @ k_recon.T

    qk_flat = qk.ravel()
    qk_recon_flat = qk_recon.ravel()
    cos = float(
        (qk_flat * qk_recon_flat).sum()
        / (
            np.linalg.norm(qk_flat) * np.linalg.norm(qk_recon_flat) + 1e-12
        )
    )

    top1 = np.argmax(qk, axis=1)
    top1_recon = np.argmax(qk_recon, axis=1)
    top1_overlap = float((top1 == top1_recon).mean())

    top5 = np.argpartition(qk, -5, axis=1)[:, -5:]
    top5_recon = np.argpartition(qk_recon, -5, axis=1)[:, -5:]
    top5_overlap = float(
        np.mean([
            len(set(a) & set(b)) / 5
            for a, b in zip(top5, top5_recon)
        ])
    )

    return {
        "inner_product_cosine": cos,
        "top1_overlap": top1_overlap,
        "top5_overlap": top5_overlap,
    }


__all__ = [
    "GENESIS_G4_TQ_REFERENCE_MARKER",
    "compute_scale_per_vector",
    "g4_tq_write_reference",
    "g4_tq_read_reference",
    "g4_tq_round_trip_test",
    "g4_tq_attention_proxy_test",
]
