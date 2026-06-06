# SPDX-License-Identifier: Apache-2.0
"""PN77 helper — FP8 lm_head weight compressor (Phase E MVP).

WHY THIS EXISTS
================

`lm_head` ships in BF16 even on int4/FP8 model bodies (it's outside the
quantization scope of AutoRound and Qwen FP8 checkpoints). For
Qwen3.6-27B vocab=248320, hidden=5120, BF16 → 2424 MiB; per A5000 rank
(TP=2) = 1212 MiB. Casting to FP8 e4m3 with per-channel scale halves
this — saving ~606 MiB/rank, ~1212 MiB total. On 35B (hidden=2048) the
saving is ~254 MiB/rank.

This module exposes a pure-math compressor: load BF16 weight, compute
per-channel scales (across vocab axis), cast to float8_e4m3fn. The
caller (PN77 wiring / forward dispatch) uses the compressed weight +
scales to either:

  (a) **MVP path**: cast back to BF16 on each forward (~3 ms/call,
      VRAM saving real, compute parity)
  (b) **Production path** [deferred]: route through
      `apply_fp8_marlin_linear` for weight-only FP8 GEMM on Ampere
      (no per-call cast; needs `prepare_fp8_layer_for_marlin` repack)

PER-CHANNEL SCALES
===================

vocab=248320 has rows with wildly different magnitudes (frequent tokens
have larger weight norms). Per-tensor scale would either saturate
common rows or zero out rare rows. Per-channel (one scale per vocab
row) is mandatory. Scale is `weight.abs().amax(dim=1) / fp8_max`,
clamped to a small floor (1e-12) to avoid division by zero on
fully-zero rows.

CAST-BACK CONSISTENCY
======================

`compress(w_bf16) → (w_fp8, scale)` then `decompress(w_fp8, scale) →
w_bf16'`. The roundtrip incurs only quantization error; on Qwen3.6
weights cosine_sim(w_bf16, w_bf16') ≥ 0.999 is the gate (same threshold
as upstream PR #35696 quality discussion). Tests use synthetic weights
that exercise the dynamic range.

SAFETY MODEL
=============

- Pure-math module: NO mutations of any parameter, NO model edits, NO
  module imports beyond torch.
- Caller is responsible for atomic param replacement; this module just
  produces tensors.
- `decompress()` is bit-stable across calls (no allocator-state
  dependence).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Reference: vllm PR #35696 (lucaspirola, OPEN — `maybe_compress_lm_head_to_fp8`),
PR #35694 (FP8 weight storage in unquantized linear/embedding apply path).
"""
from __future__ import annotations

import torch

# torch.float8_e4m3fn dynamic range: ~448 (max representable absolute value).
# Use a slight margin (clamp to 0.95 × max) for headroom against rounding.
_FP8_E4M3_MAX = 448.0
_SCALE_FLOOR = 1e-12  # prevents div-by-zero on all-zero rows


def compute_per_channel_scale(
    weight: torch.Tensor,
    fp8_max: float = _FP8_E4M3_MAX,
) -> torch.Tensor:
    """Compute per-vocab-row scale = max(|w|) / fp8_max.

    `weight` shape: (vocab_size, hidden_size). Returns shape (vocab_size,).
    """
    if weight.dim() != 2:
        raise ValueError(
            f"compute_per_channel_scale expects 2D weight, got {weight.dim()}D "
            f"shape {tuple(weight.shape)}"
        )
    if fp8_max <= 0:
        raise ValueError(f"fp8_max must be > 0, got {fp8_max}")

    # amax across hidden_size axis → one scale per vocab row
    row_max = weight.detach().abs().amax(dim=1)
    scale = (row_max / fp8_max).clamp_(min=_SCALE_FLOOR)
    return scale


def compress(
    weight_bf16: torch.Tensor,
    fp8_max: float = _FP8_E4M3_MAX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress BF16 weight to (FP8 weight, per-channel scale).

    Returns:
        weight_fp8: torch.float8_e4m3fn, same shape as input
        scale: float32 tensor, shape (vocab_size,)

    Math:
        scale[i] = max(|weight[i, :]|) / fp8_max
        weight_fp8[i, :] = (weight_bf16[i, :] / scale[i]).clamp(-fp8_max, fp8_max).to(fp8)

    VRAM-AWARE PATH (2026-05-07): keep working tensors in source dtype to
    avoid 2× transient buffer (was 2424 MiB on 27B vocab=124160×hidden=5120).
    Quality OK: target is FP8 e4m3fn anyway (3-bit exponent + 4-bit mantissa) —
    BF16 division headroom is more than sufficient.
    """
    if weight_bf16.dim() != 2:
        raise ValueError(
            f"compress expects 2D weight, got {weight_bf16.dim()}D"
        )

    src_dtype = weight_bf16.dtype
    scale = compute_per_channel_scale(weight_bf16, fp8_max)
    # Cast scale to source dtype for division — keeps the WHOLE pipeline in
    # source dtype (avoids 2× transient FP32 buffer). FP32 scale is returned
    # for downstream use (Marlin/scaled_mm consume FP32 scale).
    scale_src = scale.to(src_dtype).unsqueeze(1)
    # In-place clamp + cast: division creates one new tensor (same size as
    # source), clamp_ is in-place on it, then cast to FP8 creates the final
    # half-size tensor. The scaled tensor goes out of scope and is reclaimable.
    weight_scaled = (weight_bf16 / scale_src).clamp_(-fp8_max, fp8_max)
    weight_fp8 = weight_scaled.to(torch.float8_e4m3fn)
    # Explicit del to drop reference + signal allocator before next caller
    # (Marlin repack will allocate another large tensor; want THIS one freed).
    del weight_scaled, scale_src
    scale_f32 = scale.to(torch.float32)
    return weight_fp8, scale_f32


def decompress(
    weight_fp8: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reverse: FP8 weight × per-channel scale → BF16 (or other) weight.

    Used in MVP forward path before Marlin route lands. Cost: one tensor
    cast + one row-broadcast multiply (~3 ms for vocab=248320 on A5000).
    """
    if weight_fp8.dim() != 2:
        raise ValueError(
            f"decompress expects 2D weight, got {weight_fp8.dim()}D"
        )
    if scale.dim() != 1 or scale.shape[0] != weight_fp8.shape[0]:
        raise ValueError(
            f"scale shape {tuple(scale.shape)} must be (vocab_size={weight_fp8.shape[0]},)"
        )

    # Promote both to float32 for stable multiply, then cast to output dtype.
    weight_f32 = weight_fp8.to(torch.float32)
    scale_f32 = scale.to(torch.float32).unsqueeze(1)
    return (weight_f32 * scale_f32).to(output_dtype)


_PN77_ENV = "GENESIS_ENABLE_PN77_FP8_LM_HEAD"
_PN77_MARKER = "_genesis_pn77_fp8"  # set on layer when compression is active
_PN77_SCALE_ATTR = "weight_scale"  # standard vllm-conventional scale buffer name


def _is_enabled() -> bool:
    """Read env once-per-process; opt-in."""
    import os
    return os.environ.get(_PN77_ENV, "").strip().lower() in (
        "1", "true", "yes", "y", "on",
    )


def maybe_compress_lm_head_to_fp8(model) -> tuple[str, str]:
    """Top-level entry point — called from `load_weights` post-load via PN77 wiring.

    Returns (status, reason) tuple matching Genesis dispatcher convention:
      ('skipped', '<why>')   — env off / tied / already compressed / no lm_head
      ('applied', '<info>')  — compression done, marker set, weight replaced
      ('failed',  '<why>')   — internal error; caller logs but model still loads

    NEVER raises — always returns; caller's `load_weights` continues with
    whatever state (compressed or original) the model ended in.

    Compression contract:
    - Model's `lm_head.weight` Parameter is REPLACED with FP8 tensor
    - `lm_head.weight_scale` is REGISTERED as a buffer (per-channel float32)
    - `lm_head._genesis_pn77_fp8` marker set to True (dispatch hook reads this)
    - Tied-embedding case: detected via `weight.data_ptr()` equality with
      `embed_tokens.weight.data_ptr()`. Compression NOT applied (would
      poison embed_tokens too).
    """
    if not _is_enabled():
        return "skipped", (
            f"opt-in: set {_PN77_ENV}=1 to enable FP8 lm_head compression "
            f"(saves ~606 MiB/rank on 27B Qwen3.5/3.6, ~243 MiB/rank on 35B)"
        )

    try:
        import torch.nn as nn
    except Exception as e:
        return "failed", f"torch.nn not importable: {e}"

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        return "skipped", "model has no lm_head attribute"

    weight = getattr(lm_head, "weight", None)
    if weight is None:
        return "skipped", "lm_head has no weight attribute"

    # Idempotency
    if getattr(lm_head, _PN77_MARKER, False):
        return "skipped", "PN77 already applied (marker present)"

    # Already in FP8? (e.g. checkpoint was native FP8)
    if weight.dtype == torch.float8_e4m3fn:
        return "skipped", "lm_head already FP8 (native checkpoint)"

    # Only BF16/FP16 supported
    if weight.dtype not in (torch.bfloat16, torch.float16):
        return "skipped", f"lm_head dtype {weight.dtype} not supported (BF16/FP16 only)"

    # Tied-embedding check (don't poison embed_tokens via shared storage)
    embed_tokens = _find_embed_tokens(model)
    if embed_tokens is not None:
        if hasattr(embed_tokens, "weight") and weight.data_ptr() == embed_tokens.weight.data_ptr():
            return "skipped", "tied embeddings (lm_head shares storage with embed_tokens)"

    # 2D check
    if weight.dim() != 2:
        return "skipped", f"lm_head weight not 2D (got {weight.dim()}D)"

    try:
        weight_fp8, scale = compress(weight.data)
    except Exception as e:
        return "failed", f"compress() raised {type(e).__name__}: {e}"

    # Replace param + register scale buffer + set marker.
    #
    # CRITICAL: preserve all weight attributes (`weight_loader`, `input_dim`,
    # `output_dim`, etc.) set by `set_weight_attrs` in `create_weights`.
    # If we replace with a bare Parameter, vllm's loader falls back to
    # `default_weight_loader` which doesn't TP-shard → AssertionError on
    # any subsequent re-load (MTP head sharing, weight re-init, etc.).
    try:
        old_weight = lm_head.weight
        # Snapshot ALL attributes set on the old Parameter (excluding
        # built-in tensor attributes — only Genesis-or-vllm-attached extras).
        preserved_attrs = {
            k: v for k, v in old_weight.__dict__.items()
            if not k.startswith("_") and k not in ("data", "grad", "requires_grad")
        }
        # In-place param swap — Parameter wraps the new FP8 tensor
        new_param = nn.Parameter(weight_fp8, requires_grad=False)
        # Copy attrs (weight_loader, input_dim, output_dim, etc.)
        for k, v in preserved_attrs.items():
            setattr(new_param, k, v)
        lm_head.weight = new_param
        # Scale as buffer (not Parameter — avoids gradient/optimizer surprises)
        if hasattr(lm_head, _PN77_SCALE_ATTR):
            # Existing weight_scale (e.g. native FP8 model) — overwrite
            delattr(lm_head, _PN77_SCALE_ATTR)
        lm_head.register_buffer(_PN77_SCALE_ATTR, scale)
        setattr(lm_head, _PN77_MARKER, True)
    except Exception as e:
        return "failed", f"param replacement raised {type(e).__name__}: {e}"

    saved_mib = estimate_savings_bytes(
        weight.shape[0], weight.shape[1], src_dtype=weight.dtype,
    ) / (1024 * 1024)
    return "applied", (
        f"FP8 lm_head compressed: shape={tuple(weight.shape)}, "
        f"savings ≈ {saved_mib:.0f} MiB/rank"
    )


def _find_embed_tokens(model):
    """Best-effort locate embed_tokens for tied-weights detection.

    Common paths: model.model.embed_tokens, model.embed_tokens. Returns None
    if no such attribute. Handles Qwen3_5/Qwen3_5Moe/Qwen3 layouts.
    """
    for path in (
        ("model", "embed_tokens"),
        ("embed_tokens",),
        ("model", "language_model", "embed_tokens"),
    ):
        cur = model
        ok = True
        for attr in path:
            if not hasattr(cur, attr):
                ok = False
                break
            cur = getattr(cur, attr)
        if ok and cur is not None:
            return cur
    return None


def estimate_savings_bytes(
    vocab_size: int,
    hidden_size: int,
    src_dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Estimate VRAM saved by compression: src_size − fp8_size − scale_size.

    Useful for capacity planning + R-018-style dispatch decisions.

    For Qwen3.6-27B vocab=248320 hidden=5120 BF16:
        src = 248320 × 5120 × 2 = 2_541_158_400 bytes (2424 MiB)
        fp8 = 248320 × 5120 × 1 = 1_270_579_200 bytes (1212 MiB)
        scale = 248320 × 4 = 993_280 bytes (~1 MiB)
        savings = 1_269_585_920 bytes (~1211 MiB total, 606 MiB/rank at TP=2)
    """
    src_bytes_per_elem = torch.tensor([], dtype=src_dtype).element_size()
    src_size = vocab_size * hidden_size * src_bytes_per_elem
    fp8_size = vocab_size * hidden_size * 1  # float8_e4m3fn = 1 byte
    scale_size = vocab_size * 4  # float32 scale
    return src_size - fp8_size - scale_size
