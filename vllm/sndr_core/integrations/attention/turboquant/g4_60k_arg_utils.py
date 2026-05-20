# SPDX-License-Identifier: Apache-2.0
"""G4_60k — patch ``EngineArgs.create_engine_config`` for TQ skip-layers + FA2.

================================================================
PROBLEM
================================================================

vllm pin ``0.20.2rc1.dev371+gbf610c2f5`` does not pre-populate the
``cache_config.kv_cache_dtype_skip_layers`` set when ``cache_dtype``
starts with ``turboquant_``. As a result:

  * **Boundary layers** (first/last N decoder layers, model-defined)
    receive TQ-compressed KV by default. For Gemma 4, that quantizes
    the boundary-protected layers and degrades quality measurably
    (the upstream PR's empirical claim).

  * **KV-sharing target layers** (high-fanout layers whose K=V is
    reused by ≥50% of subsequent layers) also receive TQ-compressed
    KV. Quantizing such a target corrupts ALL downstream consumers
    that read its cache — the quality damage is multiplied.

  * **FlashAttention 3** path activates for any layer that picks
    ``FlashAttentionImpl`` after the TQ backend chain, but FA3's
    ``head_dim ≤ 512`` assertion conflicts with TurboQuant's metadata
    builder. Result: assertion crash at boot or first forward.

Upstream PR #42637 fixes all three at the engine-config phase:

  * Lines 1717-1732: collect ``boundary`` + ``kv_sharing_targets`` from
    ``TurboQuantConfig`` static helpers and union them into
    ``kv_cache_dtype_skip_layers``.
  * Lines 2050-2061: hard-set ``attention_config.flash_attn_version=2``
    with operator-facing warning when ``turboquant_*`` is selected.

================================================================
FIX
================================================================

Wrap ``EngineArgs.create_engine_config`` to apply both adjustments
**after** the config is built but **before** it returns. Mutates two
fields on the constructed ``VllmConfig``:

  1. ``vllm_config.cache_config.kv_cache_dtype_skip_layers`` —
     extended via ``set | boundary | kv_sharing_targets``, then aligned
     via ``TurboQuantConfig.align_kv_sharing_skip_layers``.

  2. ``vllm_config.attention_config.flash_attn_version`` — forced to 2
     if ``turboquant_*`` selected and version is None or ≥ 3.

================================================================
DEPENDENCIES
================================================================

  * **G4_60h** (turboquant/config.py overlay) STRONGLY recommended.
    Without it, ``TurboQuantConfig.get_boundary_skip_layers`` and
    ``get_kv_sharing_target_skip_layers`` may not exist. G4_60k
    gracefully degrades to FA2-force-only when those static methods
    are absent (skip-layers stay empty — original behaviour).

================================================================
SCOPE
================================================================

Active only when ``GENESIS_ENABLE_G4_60K_TQ_ENGINE_CONFIG=1``.

  * Non-TQ workloads: no-op (the wrapped method's TQ branch is gated
    on ``resolved_cache_dtype.startswith("turboquant_")``).

  * TQ workloads on Hopper/Blackwell with head_dim ≤ 512 where FA3
    could otherwise work: forces FA2 — same as upstream. Single line
    change at line 2061 in upstream; we mirror it.

================================================================
RISK
================================================================

  * **FA3 downgrade**: on Hopper/Blackwell where FA3 supports head_dim
    ≤ 512, this loses FA3 perf. cferra surfaced this as Gate 6 in
    issue #41403. Future Genesis patch G4_61X may gate the downgrade on
    ``compute_capability < (9, 0)`` — for now we match upstream exactly.

  * **Skip-layer set overflow**: if ``boundary ∪ kv_sharing_targets``
    is large (e.g. all 64 layers), TQ effectively disables. The
    ``align_kv_sharing_skip_layers`` helper trims overflow by warning;
    we preserve this behaviour.

================================================================
REFERENCES
================================================================

  * Upstream PR: https://github.com/vllm-project/vllm/pull/42637
  * Upstream lines (PR #42637 HEAD ``fdeb14981``):
    ``vllm/engine/arg_utils.py``
      - skip-layer collection  lines 1717-1732
      - FA2 force              lines 2050-2061
  * Related issue: https://github.com/vllm-project/vllm/issues/41403
    (Gate 6 — FA3 downgrade analysis by cferra)

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.gemma4.g4_60k_arg_utils")

GENESIS_G4_60K_MARKER = (
    "Genesis G4_60k EngineArgs.create_engine_config wrap: TQ skip-layer "
    "union + FA2 force (PR #42637 cherry-pick)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_60K_TQ_ENGINE_CONFIG"
_APPLIED = False
_ORIGINAL_CREATE_ENGINE_CONFIG = None


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Wrap EngineArgs.create_engine_config with TQ post-processing."""
    global _APPLIED, _ORIGINAL_CREATE_ENGINE_CONFIG

    if not _env_enabled():
        return "skipped", (
            f"G4_60k disabled (set {_ENV_ENABLE}=1 to apply TQ skip-layer "
            "+ FA2 force post-processing — PR #42637 cherry-pick)"
        )

    if _APPLIED:
        return "applied", "G4_60k already installed (idempotent)"

    try:
        from vllm.engine.arg_utils import EngineArgs
    except ImportError as e:
        return "skipped", f"vllm.engine.arg_utils not importable: {e}"

    original = EngineArgs.create_engine_config
    if getattr(original, "_genesis_g4_60k_wrapped", False):
        _APPLIED = True
        return "applied", (
            "EngineArgs.create_engine_config already wrapped (idempotent)"
        )
    _ORIGINAL_CREATE_ENGINE_CONFIG = original

    def _wrapped_create_engine_config(self, *args, **kwargs):
        """Apply TQ skip-layer union + FA2 force on the built VllmConfig."""
        vllm_config = original(self, *args, **kwargs)

        cache_dtype = getattr(
            vllm_config.cache_config, "cache_dtype", "auto"
        )
        if not cache_dtype.startswith("turboquant_"):
            return vllm_config

        # === Skip-layer union (PR #42637 lines 1717-1732) ===
        try:
            from vllm.model_executor.layers.quantization.turboquant.config import (
                TurboQuantConfig,
            )

            model_config = vllm_config.model_config
            cache_config = vllm_config.cache_config

            boundary_layers = []
            kv_sharing_targets = []
            if hasattr(TurboQuantConfig, "get_boundary_skip_layers"):
                boundary_layers = TurboQuantConfig.get_boundary_skip_layers(
                    model_config
                )
            if hasattr(TurboQuantConfig, "get_kv_sharing_target_skip_layers"):
                kv_sharing_targets = (
                    TurboQuantConfig.get_kv_sharing_target_skip_layers(
                        model_config
                    )
                )

            existing = set(
                getattr(cache_config, "kv_cache_dtype_skip_layers", []) or []
            )

            # [PN247] Allow operator to force-add layer indices to skip set.
            # Used for H6 hypothesis A/B: Gemma 4 MTP wires KV-sharing via
            # `kv_sharing_target_layer_name` on attention modules at
            # Gemma4Proposer._setup_gemma4_kv_sharing() time. The hf_text_config
            # path that boundary_skip/kv_sharing_target_skip rely on returns 0
            # for Gemma 4 MTP because num_kv_shared_layers=0. So G4_60h cannot
            # populate the skip set automatically. Manual override required.
            # Format: comma-separated layer indices as strings, e.g. "58,59".
            # Source of truth: live trace at PN241/PN246, see
            # `genesis_pn241_mtp_trace.log [PN246] drafter kv_sharing` lines.
            import os as _os_pn247
            _forced_raw = _os_pn247.environ.get(
                "GENESIS_G4_TQ_FORCE_SKIP_LAYERS", ""
            )
            _forced = {
                x.strip()
                for x in _forced_raw.replace(";", ",").split(",")
                if x.strip()
            }

            combined_set = (
                {str(x) for x in existing}
                | {str(x) for x in boundary_layers}
                | {str(x) for x in kv_sharing_targets}
                | _forced
            )
            combined = list(combined_set)

            if _forced:
                log.warning(
                    "[PN247] GENESIS_G4_TQ_FORCE_SKIP_LAYERS forced_skip=%s",
                    sorted(_forced),
                )

            if hasattr(TurboQuantConfig, "align_kv_sharing_skip_layers"):
                combined = TurboQuantConfig.align_kv_sharing_skip_layers(
                    model_config, combined
                )

            # Frozen-config-aware mutation. cache_config may be a Pydantic
            # model; use object.__setattr__ to bypass __setattr__ guards.
            try:
                cache_config.kv_cache_dtype_skip_layers = combined
            except (AttributeError, TypeError):
                object.__setattr__(
                    cache_config, "kv_cache_dtype_skip_layers", combined
                )

            log.info(
                "[G4_60k] kv_cache_dtype_skip_layers populated: "
                "boundary=%d, kv_sharing_targets=%d, final=%d",
                len(boundary_layers),
                len(kv_sharing_targets),
                len(combined),
            )
        except Exception as e:  # noqa: BLE001
            log.warning(
                "[G4_60k] skip-layer union skipped (G4_60h may not be "
                "applied): %r",
                e,
            )

        # === FA2 force (PR #42637 lines 2050-2061) ===
        try:
            attention_config = getattr(vllm_config, "attention_config", None)
            if attention_config is not None:
                current_fa = getattr(
                    attention_config, "flash_attn_version", None
                )
                if current_fa is None or current_fa >= 3:
                    log.warning(
                        "[G4_60k] TurboQuant is not yet compatible with "
                        "FlashAttention >= 3. Overriding flash_attn_version "
                        "to 2. To silence this warning, pass "
                        "--attention-config.flash_attn_version=2"
                    )
                    try:
                        attention_config.flash_attn_version = 2
                    except (AttributeError, TypeError):
                        object.__setattr__(
                            attention_config, "flash_attn_version", 2
                        )
        except Exception as e:  # noqa: BLE001
            log.warning("[G4_60k] FA2 force failed: %r", e)

        return vllm_config

    _wrapped_create_engine_config._genesis_g4_60k_wrapped = True  # type: ignore[attr-defined]
    EngineArgs.create_engine_config = _wrapped_create_engine_config  # type: ignore[method-assign]

    _APPLIED = True
    log.info(
        "[G4_60k] EngineArgs.create_engine_config wrapped: TQ skip-layer "
        "union + FA2 force active."
    )
    return "applied", (
        "G4_60k installed: TQ workloads now auto-populate "
        "kv_cache_dtype_skip_layers and force flash_attn_version=2."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    """Best-effort revert (test isolation only)."""
    global _APPLIED, _ORIGINAL_CREATE_ENGINE_CONFIG
    if not _APPLIED or _ORIGINAL_CREATE_ENGINE_CONFIG is None:
        return False
    try:
        from vllm.engine.arg_utils import EngineArgs

        EngineArgs.create_engine_config = _ORIGINAL_CREATE_ENGINE_CONFIG  # type: ignore[method-assign]
    except Exception:  # noqa: BLE001
        return False
    _APPLIED = False
    _ORIGINAL_CREATE_ENGINE_CONFIG = None
    return True


__all__ = [
    "GENESIS_G4_60K_MARKER",
    "apply",
    "is_applied",
    "revert",
]
