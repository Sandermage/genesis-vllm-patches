# SPDX-License-Identifier: Apache-2.0
"""G4_16 — force FULL_AND_PIECEWISE cudagraph mode for Gemma 4 dense paths.

================================================================
WHAT IT FIXES
================================================================

vLLM v1's compilation_config defaults to ``cudagraph_mode=PIECEWISE``
for most models, but **Gemma 4 dense decode benefits from
FULL_AND_PIECEWISE** because:

  1. The sliding-attention → global-attention → MLP triplet repeats 60
     layers in 31B (and 60 in 26B-A4B dense path) — capturing the whole
     triplet as one CUDAGraph eliminates 60× the per-layer launch
     overhead.
  2. PIECEWISE captures only the attention sub-graph; the MLP + residual
     joins (which are kernel-launch-bound for small batches) stay
     un-captured.
  3. FULL_AND_PIECEWISE captures the full per-layer block including
     RMSNorm + residual, so each decode step is **1 graph replay** instead
     of ~6 launches.

This patch parallels our existing PN125 (which targets Qwen3.5/3.6
``hybrid_gdn_moe``) but applies to ``model_type == gemma4``.

Expected gain: **10-30% TPS** on decode at batch=1, diminishing at
larger batch (where launch overhead is amortized over more useful
work).

================================================================
WHY UPSTREAM DOESN'T DO THIS
================================================================

Upstream's resolver in ``config/compilation.py`` checks
``splitting_ops_contain_attention()`` and, when true, defaults to
``FULL_AND_PIECEWISE``. The check fires for ``hybrid_gdn_moe`` but
NOT for ``gemma4`` because Gemma 4's attention layers are registered
under two distinct op names (sliding + full) — the heuristic only
matches one. Until upstream broadens the heuristic, we apply the
correct mode by hand.

================================================================
SAFETY MODEL
================================================================

* default_on: True (no-harm: if CG capture fails the engine falls
  back to eager; we only set the preferred mode)
* env_flag: GENESIS_ENABLE_G4_16_GEMMA4_FULL_AND_PIECEWISE
* applies_to:
    - architecture: gemma4 (BOTH 31B dense and 26B-A4B dense path)
    - cudagraph_mode != "NONE" (don't override eager mode)
* conflicts_with: PN125 (different model_type — orthogonal)
* superseded_by: when upstream broadens splitting_ops heuristic to
  cover gemma4

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
References:
  * vllm/config/compilation.py — splitting_ops_contain_attention()
  * Our existing PN125 (parallel pattern for hybrid_gdn_moe)
"""
from __future__ import annotations

import logging

from ._gemma4_detect import env_truthy, is_gemma4_arch

log = logging.getLogger("genesis.gemma4.g4_16_full_and_piecewise")

GENESIS_G4_16_MARKER = (
    "Genesis G4_16 gemma4 FULL_AND_PIECEWISE cudagraph mode v1 "
    "(parallel to PN125 for model_type=gemma4; +10-30% TPS at low batch)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_16_GEMMA4_FULL_AND_PIECEWISE"

_APPLIED = False
_ORIGINAL_VERIFY = None


def _env_enabled() -> bool:
    return env_truthy(_ENV_ENABLE)


def apply() -> tuple[str, str]:
    """Install cudagraph-mode override via Gemma4Config wrapper."""
    global _APPLIED, _ORIGINAL_VERIFY

    if not _env_enabled():
        return "skipped", (
            f"G4_16 disabled (set {_ENV_ENABLE}=1 to force FULL_AND_PIECEWISE "
            "cudagraph mode on Gemma 4 — expected +10-30% TPS at low batch)"
        )

    if _APPLIED:
        return "applied", "G4_16 already installed (idempotent)"

    try:
        from vllm.model_executor.models import gemma4 as _g4_mod
    except ImportError as e:
        return "skipped", f"vllm.model_executor.models.gemma4 not importable: {e}"

    target_cls = None
    for cls_name in ("Gemma4Config", "Gemma4TextConfig", "Gemma4ForConditionalGenerationConfig"):
        cls = getattr(_g4_mod, cls_name, None)
        if cls is not None and hasattr(cls, "verify_and_update_config"):
            target_cls = cls
            break
    if target_cls is None:
        return "skipped", (
            "No Gemma4Config-like class with verify_and_update_config found; "
            "G4_16 is no-op on this pin"
        )

    original = target_cls.verify_and_update_config
    if getattr(original, "_genesis_g4_16_wrapped", False):
        _APPLIED = True
        return "applied", "G4_16 already wrapped (idempotent)"
    _ORIGINAL_VERIFY = original

    def _genesis_g4_16_wrapped_verify(vllm_config):
        result = original(vllm_config)
        try:
            mc = getattr(vllm_config, "model_config", None)
            cc = getattr(vllm_config, "compilation_config", None)
            if mc is not None and cc is not None and is_gemma4_arch(mc):
                # Read current mode — only override when it's not explicitly
                # set to NONE (operator wants eager)
                current = getattr(cc, "cudagraph_mode", None)
                if current is not None and str(current).upper() == "NONE":
                    log.info(
                        "[G4_16] cudagraph_mode=NONE (eager) — not overriding"
                    )
                else:
                    # Set to FULL_AND_PIECEWISE
                    # Support both string and enum-style setters
                    try:
                        from vllm.config.compilation import CUDAGraphMode
                        cc.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
                    except (ImportError, AttributeError):
                        cc.cudagraph_mode = "FULL_AND_PIECEWISE"
                    log.warning(
                        "[G4_16] setting cudagraph_mode → FULL_AND_PIECEWISE "
                        "(was %s) for Gemma 4 dense path",
                        current,
                    )
                    # Mark we set this so PN125 / other patches don't fight
                    cc._g4_16_cudagraph_override = True
        except Exception as e:  # noqa: BLE001
            log.warning(
                "[G4_16] cudagraph_mode override failed: %r; leaving as-is", e
            )
        return result

    _genesis_g4_16_wrapped_verify._genesis_g4_16_wrapped = True
    _genesis_g4_16_wrapped_verify.__wrapped__ = original

    def _classmethod_shim(cls, vllm_config):
        return _genesis_g4_16_wrapped_verify(vllm_config)
    _classmethod_shim._genesis_g4_16_wrapped = True
    target_cls.verify_and_update_config = classmethod(_classmethod_shim)
    _APPLIED = True
    log.info(
        "[G4_16] installed: Gemma 4 will use cudagraph_mode=FULL_AND_PIECEWISE."
    )
    return "applied", (
        "G4_16 installed: Gemma 4 cudagraph_mode forced to FULL_AND_PIECEWISE. "
        "Expected +10-30% TPS on decode at low batch."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED, _ORIGINAL_VERIFY
    if not _APPLIED or _ORIGINAL_VERIFY is None:
        return False
    try:
        from vllm.model_executor.models import gemma4 as _g4_mod
        for cls_name in ("Gemma4Config", "Gemma4TextConfig", "Gemma4ForConditionalGenerationConfig"):
            cls = getattr(_g4_mod, cls_name, None)
            if cls is not None and getattr(
                cls.verify_and_update_config, "_genesis_g4_16_wrapped", False
            ):
                cls.verify_and_update_config = _ORIGINAL_VERIFY  # type: ignore[assignment]
                _APPLIED = False
                return True
    except ImportError:
        pass
    return False


__all__ = ["GENESIS_G4_16_MARKER", "apply", "is_applied", "revert"]
