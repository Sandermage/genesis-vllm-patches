# SPDX-License-Identifier: Apache-2.0
"""PN125 — close upstream gap: Qwen3.5/3.6 hybrid skipped FULL_AND_PIECEWISE.

================================================================
THE GAP
================================================================

vLLM has two helpers for cudagraph mode selection on hybrid models:

  - `MambaModelConfig.verify_and_update_config(vllm_config)` —
     sets `compilation_config.cudagraph_mode = FULL_AND_PIECEWISE`
     for hybrid attention + Mamba/GDN models. PyTorch blog
     ("Hybrid Models as First-Class Citizens in vLLM", 2026)
     measures up to 91% throughput improvement and lower ITL at
     low concurrency vs PIECEWISE default.

  - `HybridAttentionMambaModelConfig.verify_and_update_config` —
     calls MambaModelConfig at the end. Designed for the
     attention+Mamba hybrid family.

The mapping table `_MODELS_CONFIG_MAP` in
`vllm/model_executor/models/config.py` registers
HybridAttentionMambaModelConfig for some architectures (Jamba etc.)
but `Qwen3_5MoeForConditionalGeneration` (our 35B-A3B-FP8) and
`Qwen3_5ForConditionalGeneration` (our 27B-INT4-AutoRound)
both map to `Qwen3_5ForConditionalGenerationConfig` which only
updates `mamba_ssm_cache_dtype` and never calls MambaModelConfig.

Net effect: hybrid Qwen3.5 / Qwen3.6 models run in default
PIECEWISE cudagraph mode, missing the GDN-targeted throughput
improvement other Mamba/Jamba models get for free.

Verified on dev338 (2026-05-14) container boot log:
  "Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%"

================================================================
THE FIX
================================================================

Runtime-hook (NOT text-patch): wrap
`Qwen3_5ForConditionalGenerationConfig.verify_and_update_config`
so it ALSO calls `MambaModelConfig.verify_and_update_config(vllm_config)`
on hybrid path. Implementation is a 5-line monkey-patch at apply
time, before any model loads. Idempotent (marker on the wrapped
classmethod).

================================================================
RISK / VALIDATION
================================================================

Default OFF — opt-in via `GENESIS_ENABLE_PN125_HYBRID_FULL_AND_PIECEWISE=1`.

Risks:
  - Adds ~1 GiB VRAM per worker (FULL graph captures alongside
    PIECEWISE). On our 2× A5000 24 GB at gpu_memory_utilization=0.9
    with 35B-A3B-FP8 + TQ k8v4 we have ~3 GiB headroom — fits.
  - Adds ~10 s to boot (extra graph capture pass).
  - Composes with P66 (cudagraph size filter — also active);
    P66's allowlist drops invalid sizes BEFORE capture, so the
    FULL graph capture still respects the filter.
  - Does NOT compose with `--enforce-eager` (graphs disabled).

Bench-gate: before promoting to default_on, validate on:
  - 35B-A3B 8K decode: TPS ≥ 216 (current baseline), p99 TPOT
    not worse than +5%, VRAM headroom ≥ 1 GiB after capture.
  - 27B INT4 8K decode: TPS ≥ 130, VRAM headroom ≥ 2 GiB.
  - 32K decode: same as 8K plus cudagraph_capture passes complete.

================================================================
COMPOSITION
================================================================

- Safe with P66, P95, P101 — they all operate at different
  layers (capture-size allowlist / Marlin TP cap / FlashInfer
  spec-decode wrapper). FULL_AND_PIECEWISE only changes WHICH
  graph mode runs, not WHAT shapes get captured.
- Mutually exclusive with `--enforce-eager` (operator decision;
  if eager is set, this patch is a no-op).
- Composes with PN119 (GQA grouping kernel) — orthogonal layers.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Source: PyTorch blog 2026-05 "Hybrid Models as First-Class Citizens
        in vLLM" (https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/)
        — measured up to 91% throughput gain on hybrid Mamba models.

References:
- vLLM upstream gap: `_MODELS_CONFIG_MAP` only registers
  `Qwen3_5ForConditionalGenerationConfig` for Qwen3.5/3.6 hybrid
  variants, which never calls `MambaModelConfig.verify_and_update_config`.
- Could also be fixed upstream by adding HybridAttentionMambaModelConfig
  to the Qwen3.5 mapping; we patch locally for now.

Status: experimental (lifecycle=experimental, default_on=False).
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.wiring.pn125_hybrid_full_and_piecewise")

GENESIS_PN125_MARKER = (
    "Genesis PN125 hybrid_gdn FULL_AND_PIECEWISE v1 (closes vLLM upstream "
    "gap: Qwen3.5/3.6 skipped from MambaModelConfig.verify_and_update_config)"
)

_ENV_ENABLE = "GENESIS_ENABLE_PN125_HYBRID_FULL_AND_PIECEWISE"
_ENV_DISABLE = "GENESIS_DISABLE_PN125_HYBRID_FULL_AND_PIECEWISE"

_APPLIED = False
_ORIGINAL_VERIFY: object = None


def _env_enabled() -> bool:
    """Default OFF — bench-gate required before flipping default."""
    if os.environ.get(_ENV_DISABLE, "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    val = os.environ.get(_ENV_ENABLE, "").strip().lower()
    return val in ("1", "true", "yes", "on")


def apply() -> tuple[str, str]:
    """Install the verify_and_update_config wrapper. Never raises."""
    global _APPLIED, _ORIGINAL_VERIFY

    if not _env_enabled():
        return "skipped", (
            f"PN125 disabled (set {_ENV_ENABLE}=1 to enable hybrid_gdn "
            f"FULL_AND_PIECEWISE on Qwen3.5/3.6 — see PyTorch blog 2026-05)"
        )

    if _APPLIED:
        return "applied", "PN125 already installed (idempotent)"

    try:
        from vllm.model_executor.models import config as _config_mod
    except ImportError as e:
        return "skipped", f"vllm.model_executor.models.config not importable: {e}"

    target_cls = getattr(_config_mod, "Qwen3_5ForConditionalGenerationConfig", None)
    if target_cls is None:
        return "skipped", (
            "Qwen3_5ForConditionalGenerationConfig not in vllm config — "
            "pin may predate Qwen3.5 hybrid support; PN125 is no-op"
        )

    mamba_cls = getattr(_config_mod, "MambaModelConfig", None)
    if mamba_cls is None or not hasattr(mamba_cls, "verify_and_update_config"):
        return "skipped", (
            "MambaModelConfig.verify_and_update_config not in vllm config "
            "— pin may predate hybrid auto-FULL_AND_PIECEWISE; PN125 is no-op"
        )

    # Idempotency: check marker attribute on the wrapped classmethod's __func__
    original = target_cls.verify_and_update_config
    if getattr(original, "_genesis_pn125_wrapped", False):
        _APPLIED = True
        return "applied", "PN125 already wrapped (idempotent)"

    # Save original for revert and potential drift inspection
    _ORIGINAL_VERIFY = original

    # Build wrapper that calls original AND mamba's verify
    def _genesis_pn125_wrapped_verify_and_update_config(vllm_config):
        """Genesis PN125 — original Qwen3.5 verify + MambaModelConfig setup.

        Original handles `mamba_ssm_cache_dtype` resolution; MambaModelConfig
        handles `cudagraph_mode = FULL_AND_PIECEWISE` plus prefix-caching
        consistency for Mamba layers. Both are required for hybrid_gdn_moe.
        Failures inside Mamba step are swallowed (logged) so this patch
        never breaks boot.
        """
        # Run original Qwen3.5 verify (idempotent in upstream)
        result = original(vllm_config)
        # Then run MambaModelConfig.verify_and_update_config to set
        # cudagraph_mode = FULL_AND_PIECEWISE and align prefix caching.
        try:
            mamba_cls.verify_and_update_config(vllm_config)
        except Exception as e:
            log.warning(
                "[PN125] MambaModelConfig.verify_and_update_config "
                "failed (%s); leaving cudagraph_mode at default. "
                "Genesis PN125 self-suppresses for this boot.",
                e,
            )
        return result

    # Mark inner wrapper for idempotency + revert lookup (must be set on
    # the function itself, NOT on the bound method — the latter is read-only
    # because Python re-creates the bound-method object on each attribute
    # access).
    _genesis_pn125_wrapped_verify_and_update_config._genesis_pn125_wrapped = True
    _genesis_pn125_wrapped_verify_and_update_config._genesis_pn125_original = original

    # Build the classmethod descriptor and stash the marker on its underlying
    # __func__ so idempotency check on the bound-method's __func__ works
    # next time apply() runs (`getattr(original, "_genesis_pn125_wrapped",
    # False)` reads via the descriptor → bound method → __func__).
    def _classmethod_shim(cls, vllm_config):
        return _genesis_pn125_wrapped_verify_and_update_config(vllm_config)
    _classmethod_shim._genesis_pn125_wrapped = True
    _classmethod_shim._genesis_pn125_original = original

    target_cls.verify_and_update_config = classmethod(_classmethod_shim)
    _APPLIED = True

    log.info(
        "[PN125] installed: Qwen3.5/3.6 hybrid config now calls "
        "MambaModelConfig.verify_and_update_config (sets "
        "cudagraph_mode=FULL_AND_PIECEWISE). Source: PyTorch blog 2026-05."
    )
    return "applied", (
        "PN125 installed: Qwen3.5/3.6 hybrid path now invokes "
        "MambaModelConfig.verify_and_update_config — closes vLLM upstream "
        "gap; cudagraph_mode auto-FULL_AND_PIECEWISE on hybrid_gdn_moe. "
        "Expected: lower TTFT/ITL at low concurrency (per PyTorch blog 2026-05)."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    """Restore upstream Qwen3_5ForConditionalGenerationConfig.verify_and_update_config."""
    global _APPLIED, _ORIGINAL_VERIFY
    if not _APPLIED or _ORIGINAL_VERIFY is None:
        return False
    try:
        from vllm.model_executor.models import config as _config_mod
    except ImportError:
        return False
    target_cls = getattr(_config_mod, "Qwen3_5ForConditionalGenerationConfig", None)
    if target_cls is None:
        return False
    target_cls.verify_and_update_config = _ORIGINAL_VERIFY  # type: ignore[assignment]
    _APPLIED = False
    return True
