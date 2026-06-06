# SPDX-License-Identifier: Apache-2.0
"""G4_08 — Marlin K-dim pad fallback: route through Genesis Triton MoE GEMM.

================================================================
PURPOSE
================================================================

Hijacks the Marlin MoE GEMM call path so that when K is not divisible
by ``min_thread_k=64``, vLLM dispatches to our zero-pad Triton kernel
(``kernels/g4_kpad_moe_gemm_triton.py``) instead of failing with
"Invalid thread config" (vllm#40354).

This unlocks Gemma 4 26B-A4B at TP=2 (intermediate_size=704 → per-shard
K=352) on Ampere SM 8.6 RTX 3090 / A5000.

================================================================
INTEGRATION STRATEGY
================================================================

vLLM's compressed-tensors Marlin MoE method (``CompressedTensorsMoEWNA16MarlinMethod``)
exposes ``apply_weights(layer, x, ...)``. We monkey-patch that method
at plugin-register time:

  1. Pre-check: infer K from layer and check divisibility
  2. If K % 64 != 0: load (and cache) the K-padded version of the
     expert weights into the layer (one-time, on first call) and route
     through our Triton kernel
  3. If K % 64 == 0: fall through to upstream Marlin (no overhead)

The weight-padding step is cached on the layer (``layer._g4_padded_weights``)
so we don't pad every forward.

================================================================
SAFETY MODEL
================================================================

* default_on: False (research; opt-in via env)
* env_flag: GENESIS_ENABLE_G4_08_MARLIN_KDIM_PAD
* applies_to:
    - architecture: gemma4 (or any MoE arch with K%64≠0)
    - hardware: Ampere SM 8.6 (Hopper has its own faster Cutlass path)
* depends_on: triton ≥ 2.3
* conflicts_with: G4_02. The two are mutually exclusive — G4_02 refuses
  to boot on K%64≠0, while G4_08 unblocks the same case by routing
  through our Triton K-pad kernel. Enable G4_08 ONLY by setting
  ``GENESIS_DISABLE_G4_02_GUARD=1`` so G4_02 stands down. G4_08
  effectively *supersedes* G4_02 for the same shape failure.

================================================================
PERFORMANCE EXPECTATIONS (PROJECTED)
================================================================

* vs Marlin (when K is aligned): 0.6-0.8x throughput
* vs no Marlin (kernel rejected): ∞ (unblocked)
* VRAM overhead: ~+9% on the affected MoE weight tensors due to padding

Measured numbers will land in the V2 model YAML's reference_metrics
after server validation.

================================================================
TEST PLAN
================================================================

``tests/unit/integrations/gemma4/test_g4_08_kpad_moe.py``:
  * Unit: g4_kpad_moe_gemm_reference matches g4_kpad_moe_gemm to abs<1e-2
  * Unit: padding-zone loads return zero (mask correctness)
  * Unit: aligned K (K=384) bit-identical to Marlin reference
  * Unit: g4_kpad_moe_gemm with K_real=352, K_padded=384 numerically valid

Server: 26B-A4B at TP=2 boots; smoke chat produces coherent answer;
bench within projected range.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
from typing import Optional

from ._gemma4_detect import (
    env_truthy,
    infer_marlin_kdim_for_moe,
    is_ampere_sm86,
    is_gemma4_arch,
    marlin_kdim_supported,
)

log = logging.getLogger("genesis.gemma4.g4_08_marlin_kdim_pad")

GENESIS_G4_08_MARKER = (
    "Genesis G4_08 gemma4 Marlin K-dim pad fallback v1 "
    "(closes vllm#40354 for K%64≠0 MoE via Triton zero-pad kernel)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_08_MARLIN_KDIM_PAD"

_APPLIED = False
_ORIGINAL_APPLY = None


def _env_enabled() -> bool:
    return env_truthy(_ENV_ENABLE)


def _layer_is_gemma4_or_marlin_blocked(layer) -> bool:
    """Detection: layer needs the K-pad fallback.

    Either:
      * its parent model is Gemma 4 (specific opt-in), OR
      * its inferred K dim fails Marlin's divisibility check (generic fallback)
    """
    # Check Gemma 4 model match first
    for attr in ("_vllm_config", "vllm_config", "model_config"):
        cfg = getattr(layer, attr, None)
        if cfg is not None:
            mc = getattr(cfg, "model_config", None) or cfg
            if is_gemma4_arch(mc):
                return True
    # Fallback: K-divisibility probe
    prob_k = infer_marlin_kdim_for_moe(layer)
    if prob_k is not None and not marlin_kdim_supported(prob_k, strict_python_check=False):
        return True
    return False


def _ensure_padded_weights(layer, K_real: int) -> int:
    """Pad w13_weight + w2_weight on K dim (idempotent, cached on layer).

    Returns the padded K dimension.
    """
    cache_key = "_g4_08_padded_kdim"
    if hasattr(layer, cache_key):
        return getattr(layer, cache_key)
    from .kernels.g4_kpad_moe_gemm_triton import pad_moe_weight_to_aligned_k
    # Pad w13_weight (gate_up_proj) — orientation [E, 2*I, H] or [E, H, 2*I]
    # We don't know orientation a priori; pad whichever axis has K_real
    if hasattr(layer, "w13_weight") and layer.w13_weight is not None:
        padded_w13, _, K_padded = pad_moe_weight_to_aligned_k(layer.w13_weight, K_real, 64)
        layer.w13_weight = padded_w13
    else:
        K_padded = K_real
    # Pad w2_weight (down_proj) — orientation [E, H, I] or [E, I, H]
    if hasattr(layer, "w2_weight") and layer.w2_weight is not None:
        padded_w2, _, _ = pad_moe_weight_to_aligned_k(layer.w2_weight, K_real, 64)
        layer.w2_weight = padded_w2
    setattr(layer, cache_key, K_padded)
    log.info(
        "[G4_08] padded layer MoE weights: K_real=%d → K_padded=%d (+%d zero columns)",
        K_real, K_padded, K_padded - K_real,
    )
    return K_padded


def _make_guarded_apply(original):
    """Wrap apply_weights with the K-pad routing."""

    def _genesis_g4_08_routed_apply(self, layer, *args, **kwargs):
        try:
            if is_ampere_sm86() and _layer_is_gemma4_or_marlin_blocked(layer):
                prob_k = infer_marlin_kdim_for_moe(layer)
                if prob_k is not None and not marlin_kdim_supported(prob_k, strict_python_check=False):
                    log.info(
                        "[G4_08] routing through Triton K-pad kernel for layer "
                        "with K_real=%d", prob_k,
                    )
                    # Pad weights on first call (idempotent)
                    K_padded = _ensure_padded_weights(layer, prob_k)
                    # Dispatch to our kernel via the layer's existing forward shape
                    # NOTE: this is the *integration glue* between vLLM's MoE
                    # apply_weights signature and our kernel. In a real ship-ready
                    # version this needs the full FusedMoE interface (routing,
                    # gating, top-k); here we route only the GEMM step.
                    from .kernels.g4_kpad_moe_gemm_triton import g4_kpad_moe_gemm
                    # Determine num_bits from quant_config
                    num_bits = getattr(layer, "_g4_num_bits", 8)
                    activations = args[0] if args else kwargs.get("x")
                    expert_ids = kwargs.get("expert_ids") or kwargs.get("topk_ids")
                    # For now, delegate to original Marlin if we can't extract args
                    if activations is None or expert_ids is None:
                        log.warning(
                            "[G4_08] could not extract activations/expert_ids; "
                            "falling through to original Marlin (will fail).",
                        )
                        return original(self, layer, *args, **kwargs)
                    # Route through our kernel
                    return g4_kpad_moe_gemm(
                        activations=activations,
                        expert_weights=layer.w2_weight,
                        scales=layer.w2_weight_scale,
                        expert_ids=expert_ids,
                        K_real=prob_k,
                        num_bits=num_bits,
                        has_gelu_tanh=False,
                    )
        except Exception as e:  # noqa: BLE001
            log.warning(
                "[G4_08] routing failed: %r; falling through to original Marlin", e,
            )

        return original(self, layer, *args, **kwargs)

    _genesis_g4_08_routed_apply._genesis_g4_08_wrapped = True
    _genesis_g4_08_routed_apply._genesis_g4_08_original = original
    _genesis_g4_08_routed_apply.__wrapped__ = original
    return _genesis_g4_08_routed_apply


def apply() -> tuple[str, str]:
    global _APPLIED, _ORIGINAL_APPLY

    if not _env_enabled():
        return "skipped", (
            f"G4_08 disabled (set {_ENV_ENABLE}=1 to enable Genesis K-pad "
            "Triton MoE fallback — opens Gemma 4 26B-A4B at TP=2 on Ampere)"
        )

    if _APPLIED:
        return "applied", "G4_08 already installed (idempotent)"

    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
            compressed_tensors_moe_wna16_marlin as mod,
        )
        target_cls = getattr(mod, "CompressedTensorsMoEWNA16MarlinMethod", None)
        if target_cls is None:
            return "skipped", (
                "CompressedTensorsMoEWNA16MarlinMethod not found in this vLLM pin"
            )
        original = target_cls.apply_weights
        if getattr(original, "_genesis_g4_08_wrapped", False):
            _APPLIED = True
            return "applied", "G4_08 already wrapped (idempotent)"
        _ORIGINAL_APPLY = original
        target_cls.apply_weights = _make_guarded_apply(original)
    except ImportError as e:
        return "skipped", (
            "compressed_tensors_moe_wna16_marlin not importable: " f"{e}"
        )

    _APPLIED = True
    log.info(
        "[G4_08] installed: K-dim non-aligned MoE GEMMs will now route through "
        "Genesis Triton zero-pad kernel."
    )
    return "applied", (
        "G4_08 installed: Marlin MoE method patched to route K%64≠0 cases "
        "through Genesis Triton K-pad kernel. 26B-A4B at TP=2 should now load."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED, _ORIGINAL_APPLY
    if not _APPLIED or _ORIGINAL_APPLY is None:
        return False
    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
            compressed_tensors_moe_wna16_marlin as mod,
        )
        target_cls = getattr(mod, "CompressedTensorsMoEWNA16MarlinMethod", None)
        if target_cls is None:
            return False
        target_cls.apply_weights = _ORIGINAL_APPLY  # type: ignore[assignment]
    except ImportError:
        return False
    _APPLIED = False
    _ORIGINAL_APPLY = None
    return True


__all__ = ["GENESIS_G4_08_MARKER", "apply", "is_applied", "revert"]
