# SPDX-License-Identifier: Apache-2.0
"""G4_23 — Gemma 4 vision-tower FP16 overflow guard (closes vllm#40124).

================================================================
WHAT IT FIXES
================================================================

vllm-project/vllm#40124 (OPEN as of 2026-05-17, 4 comments): Gemma 4's
vision tower is a SigLIP-style ViT with **patch-embed bias values** that
go up to ±2400 in the trained checkpoint. The forward pass on the
vision tower computes patch projections in the operator-chosen dtype
(``--dtype float16`` is the popular default on Ampere because BF16
support varies).

In FP16, the patch-embed output overflows the max representable
finite value (65504) during accumulation in the attention QK^T matmul.
Result: ``NaN`` propagates through the rest of the vision tower → the
multi-modal projector output is all-NaN → the language model's image
embedding inputs are NaN → either a hard ``RuntimeError: Function
'AddmmBackward' returned nan values`` or silent garbage tokens.

================================================================
THE FIX
================================================================

Two-pronged:

  1. **Force vision tower to BF16 (or FP32)** even when the rest of the
     model runs FP16. The vision tower is a tiny fraction of total
     params (~300M out of 31B), so the precision bump costs ~600 MB
     VRAM and is well within budget.

  2. **Add a soft-clip on patch-embed output** if BF16 isn't available
     (older Ampere variants in some configurations). We clamp to
     ``±32768`` which is well within FP16's headroom (65504) but
     accommodates the legitimate large activations in Gemma 4's vision
     tower.

We hook ``Gemma4VisionTower.__init__`` and the patch-embed forward to
install the dtype override and soft-clip respectively.

================================================================
SAFETY MODEL
================================================================

* default_on: True (no harm: forces vision tower dtype upward when
  operator chose FP16, leaves other dtype paths alone)
* env_flag: GENESIS_ENABLE_G4_23_GEMMA4_VISION_FP16_OVERFLOW
* applies_to:
    - architecture: Gemma4ForConditionalGeneration
    - dtype: float16 / fp16
* conflicts_with: G4_17 (which stubs out the vision tower entirely —
  if G4_17 fires first, G4_23 is no-op because there's nothing to
  protect)
* superseded_by: vllm#40124 when merged (proposed: dtype-aware
  vision-tower forward in upstream)

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
References:
  * https://github.com/vllm-project/vllm/issues/40124 (OPEN, 4 comments)
"""
from __future__ import annotations

import logging

from ._gemma4_detect import env_truthy

log = logging.getLogger("genesis.gemma4.g4_23_vision_fp16_overflow")

GENESIS_G4_23_MARKER = (
    "Genesis G4_23 gemma4 vision-tower FP16 overflow guard v1 "
    "(forces BF16 on vision tower when operator chose FP16; closes vllm#40124)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_23_GEMMA4_VISION_FP16_OVERFLOW"

# Patch-embed output can legitimately reach ~±2400; we soft-clip well
# below FP16 max (65504) to leave headroom for attention QK^T accum.
_PATCH_EMBED_CLAMP = 32768.0

_APPLIED = False
_ORIGINAL_INIT = None
_PATCHED_CLS = None


def _env_enabled() -> bool:
    return env_truthy(_ENV_ENABLE)


def _find_vision_tower_cls():
    """Locate Gemma4VisionTower across vLLM pin variants."""
    candidates = (
        "Gemma4VisionTower",
        "Gemma4VisionModel",
        "SiglipVisionModel",  # Gemma 4 uses SigLIP under the hood
    )
    try:
        from vllm.model_executor.models import gemma4 as _g4_mod
    except ImportError:
        return None
    for name in candidates:
        cls = getattr(_g4_mod, name, None)
        if cls is not None:
            return cls
    return None


def _bf16_supported() -> bool:
    """True when the host CUDA device supports BF16 natively."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        return torch.cuda.is_bf16_supported()
    except Exception:  # noqa: BLE001
        return False


def apply() -> tuple[str, str]:
    """Install vision-tower dtype upgrade + patch-embed soft-clip."""
    global _APPLIED, _ORIGINAL_INIT, _PATCHED_CLS

    if not _env_enabled():
        return "skipped", (
            f"G4_23 disabled (set {_ENV_ENABLE}=1 to force vision tower "
            "to BF16 on Gemma 4 + FP16 operator config — closes vllm#40124)"
        )

    if _APPLIED:
        return "applied", "G4_23 already installed (idempotent)"

    target_cls = _find_vision_tower_cls()
    if target_cls is None:
        # dev491+ relocated the Gemma-4 vision tower: it is no longer a vLLM-native
        # Gemma4VisionTower/SiglipVisionModel class in models/gemma4.py but an HF
        # AutoModel.from_config built in models/gemma4_mm.py (vision forward does
        # vt.patch_embedder(...).to(model_dtype) with NO float32 upgrade / clamp).
        # So this monkeypatch cannot bind and the vllm#40124 FP16-overflow guard is
        # NOT installed. On a MM-capable pin make that LOUD (a silent 'skipped' was
        # invisible to the operator — the 2026-06-16 dev491 audit's one actionable
        # finding); on a genuinely text-only build the absence is expected, so keep
        # that quiet. Exposure only bites under --dtype float16; the canonical
        # Gemma-4 YAMLs run bfloat16 (no overflow), so behaviour is unchanged.
        try:
            import importlib.util as _ilu
            _mm_pin = _ilu.find_spec(
                "vllm.model_executor.models.gemma4_mm"
            ) is not None
        except Exception:  # noqa: BLE001
            _mm_pin = False
        if _mm_pin:
            log.warning(
                "[G4_23] ENABLED but NO-OP on this vLLM pin: no vLLM-native "
                "Gemma4VisionTower/SiglipVisionModel in models.gemma4 — the vision "
                "tower is now an HF AutoModel (models.gemma4_mm). The vllm#40124 "
                "FP16-overflow guard is NOT installed. This only matters under "
                "--dtype float16; canonical Gemma-4 YAMLs use bfloat16 (safe). To "
                "restore it on a float16 MM profile, re-point at the HF "
                "patch_embedder path. 2026-06-16 audit."
            )
            return "skipped", (
                "G4_23 NO-OP: Gemma4VisionTower absent (vision tower relocated to HF "
                "AutoModel in gemma4_mm on dev491+); FP16 guard not installed — "
                "matters only under --dtype float16. See vllm#40124."
            )
        return "skipped", (
            "No Gemma4VisionTower-like class found; G4_23 is a no-op "
            "(text-only build of Gemma 4 doesn't have this class)"
        )

    _PATCHED_CLS = target_cls
    original_init = target_cls.__init__
    if getattr(original_init, "_genesis_g4_23_wrapped", False):
        _APPLIED = True
        return "applied", "G4_23 already wrapped (idempotent)"
    _ORIGINAL_INIT = original_init

    bf16_ok = _bf16_supported()

    def _genesis_g4_23_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            import torch
            # Determine the operator's requested dtype
            cur_dtype = next(self.parameters()).dtype if list(self.parameters()) else None
            if cur_dtype is torch.float16:
                if bf16_ok:
                    log.warning(
                        "[G4_23] Gemma 4 vision tower: operator chose FP16, "
                        "patch-embed bias values up to ±2400 → overflow risk "
                        "in QK^T. Upgrading vision tower to BF16 (cost: "
                        "~600 MB VRAM, gain: no NaN propagation)."
                    )
                    self.to(torch.bfloat16)
                else:
                    log.warning(
                        "[G4_23] Gemma 4 vision tower: BF16 not supported on "
                        "this device. Installing soft-clip (±%f) on patch-embed "
                        "output to prevent FP16 overflow in attention.",
                        _PATCH_EMBED_CLAMP,
                    )
                    _install_patch_embed_clamp(self)
        except Exception as e:  # noqa: BLE001
            log.warning("[G4_23] dtype upgrade / clamp install failed: %r", e)

    _genesis_g4_23_init._genesis_g4_23_wrapped = True
    _genesis_g4_23_init.__wrapped__ = original_init
    target_cls.__init__ = _genesis_g4_23_init
    _APPLIED = True
    log.info(
        "[G4_23] installed: Gemma 4 vision tower will use BF16 (or soft-clip "
        "fallback) to avoid FP16 overflow in patch-embed forward."
    )
    return "applied", (
        "G4_23 installed: Gemma 4 vision tower will upgrade to BF16 "
        "when operator-selected dtype is FP16 (or apply soft-clip fallback "
        "if BF16 unsupported). Closes vllm#40124."
    )


def _install_patch_embed_clamp(vision_tower) -> bool:
    """Install a forward-hook on the patch-embed module that clamps output."""
    try:
        # Heuristic search for the patch-embed module
        patch_embed = None
        for name in ("patch_embed", "patch_embedding", "embeddings"):
            patch_embed = getattr(vision_tower, name, None)
            if patch_embed is not None:
                break
        if patch_embed is None:
            return False

        def _clamp_hook(_module, _input, output):
            if output is None:
                return output
            return output.clamp(min=-_PATCH_EMBED_CLAMP, max=_PATCH_EMBED_CLAMP)

        handle = patch_embed.register_forward_hook(_clamp_hook)
        # Stash for revert
        vision_tower._g4_23_clamp_handle = handle
        return True
    except Exception:  # noqa: BLE001
        return False


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED, _ORIGINAL_INIT, _PATCHED_CLS
    if not _APPLIED or _PATCHED_CLS is None or _ORIGINAL_INIT is None:
        return False
    _PATCHED_CLS.__init__ = _ORIGINAL_INIT
    _APPLIED = False
    _ORIGINAL_INIT = None
    _PATCHED_CLS = None
    return True


__all__ = ["GENESIS_G4_23_MARKER", "apply", "is_applied", "revert"]
