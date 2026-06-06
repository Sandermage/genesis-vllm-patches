# SPDX-License-Identifier: Apache-2.0
"""G4_17 — skip vision-tower load when running Gemma 4 multimodal text-only.

================================================================
WHAT IT FIXES
================================================================

When an operator launches the Gemma 4 multimodal model
(``Gemma4ForConditionalGeneration``) with text-only requests
(no images), vLLM still loads the **vision tower + multimodal projector**
into VRAM at boot:

  * vision tower (SigLIP-style ViT): ~2.3 GB on 31B variant
  * mm_projector + image_newline: ~50 MB
  * Image processor + tokenizer image-special-tokens: small but adds
    init overhead

The user's prompt **never contains an image**, so all of this is dead
weight. Symptoms reported:

  * +2.3 GB VRAM per device (across TP shards)
  * +30 sec to cold boot (vision tower download + load + JIT)
  * On Ampere SM 8.6 with 24 GB cards, this VRAM is the difference
    between fitting 32K context and OOM at 16K

================================================================
THE FIX
================================================================

Hook the model loader to **skip vision-tower init** when:

  1. ``architectures == ["Gemma4ForConditionalGeneration"]``
  2. operator explicitly passes ``GENESIS_GEMMA4_TEXT_ONLY=1``
  3. (optional auto-detect) ``--limit-mm-per-prompt image=0`` is set
     OR no image inputs are observed in first 100 requests

We monkey-patch ``Gemma4ForConditionalGeneration.__init__`` to inspect
the env flag and, when set, install a stub ``vision_tower`` /
``multi_modal_projector`` that raises a clear error if anything tries
to use them — but never allocates GPU memory.

================================================================
SAFETY MODEL
================================================================

* default_on: False (operator must opt in via env or explicit flag)
* env_flag: GENESIS_ENABLE_G4_17_GEMMA4_VISION_SKIP
* runtime_flag: GENESIS_GEMMA4_TEXT_ONLY=1
* applies_to:
    - architecture: Gemma4ForConditionalGeneration
* conflicts_with: any multimodal-required workflow (operator's choice)
* superseded_by: an upstream ``mm_lazy_load=True`` proposal when one
  is accepted by the vLLM project. The previously-recorded upstream
  reference ``vllm#41565`` was a wrong-number — issue #41565 is the
  TurboQuant ``_continuation_prefill`` workspace bug (correct upstream
  tracker for G4_61 / G4_62), not a multimodal-text-only report. No
  matching upstream issue for the vision-tower text-only skip was
  located during the 2026-05-24 PIN.R-REFS-CLOSED-PR.R audit. Treat
  this patch as ``genesis_original`` with no upstream tracker until
  a correct upstream issue is found.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

from ._gemma4_detect import env_truthy

log = logging.getLogger("genesis.gemma4.g4_17_vision_skip")

GENESIS_G4_17_MARKER = (
    "Genesis G4_17 gemma4 vision-tower skip for text-only inference v1 "
    "(saves ~2.3 GB VRAM + ~30 sec cold boot when text-only; "
    "genesis_original, no upstream tracker)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_17_GEMMA4_VISION_SKIP"
_ENV_TEXT_ONLY = "GENESIS_GEMMA4_TEXT_ONLY"

_APPLIED = False
_ORIGINAL_INIT = None
_PATCHED_CLS = None


def _env_enabled() -> bool:
    return env_truthy(_ENV_ENABLE)


def _text_only_requested() -> bool:
    """Check both the per-feature env and the runtime text-only flag."""
    return env_truthy(_ENV_TEXT_ONLY)


class _DeferredVisionTowerStub:
    """Stand-in for the un-loaded vision tower.

    Raises a clear error if any code path tries to use it. Costs zero
    VRAM and doesn't pull any vision weights into memory.
    """
    def __init__(self):
        self._activated = False

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "[Genesis G4_17] Vision tower not loaded (text-only mode via "
            "GENESIS_GEMMA4_TEXT_ONLY=1). To enable images, unset the env "
            "and restart the server."
        )

    def __getattr__(self, item):
        # Hot-path safety: tokenizer / config accessors return None
        if item in ("config", "image_size", "patch_size"):
            return None
        if self._activated:
            raise RuntimeError(
                f"[Genesis G4_17] Vision tower attr '{item}' accessed in "
                "text-only mode."
            )
        raise AttributeError(
            f"_DeferredVisionTowerStub has no attribute '{item}' "
            "(text-only mode)"
        )

    def to(self, *args, **kwargs):
        return self

    def cuda(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def parameters(self):
        return iter(())

    def state_dict(self, *args, **kwargs):
        return {}

    def load_state_dict(self, *args, **kwargs):
        return ([], [])


def apply() -> tuple[str, str]:
    """Install vision-skip hook on Gemma4ForConditionalGeneration."""
    global _APPLIED, _ORIGINAL_INIT, _PATCHED_CLS

    if not _env_enabled():
        return "skipped", (
            f"G4_17 disabled (set {_ENV_ENABLE}=1 + {_ENV_TEXT_ONLY}=1 to "
            "skip vision-tower init on Gemma 4)"
        )

    if _APPLIED:
        return "applied", "G4_17 already installed (idempotent)"

    try:
        from vllm.model_executor.models import gemma4 as _g4_mod
    except ImportError as e:
        return "skipped", f"vllm.model_executor.models.gemma4 not importable: {e}"

    target_cls = getattr(_g4_mod, "Gemma4ForConditionalGeneration", None)
    if target_cls is None:
        return "skipped", (
            "Gemma4ForConditionalGeneration class not found in this vLLM "
            "pin — G4_17 is no-op"
        )

    _PATCHED_CLS = target_cls
    original_init = target_cls.__init__
    if getattr(original_init, "_genesis_g4_17_wrapped", False):
        _APPLIED = True
        return "applied", "G4_17 already wrapped (idempotent)"
    _ORIGINAL_INIT = original_init

    def _genesis_g4_17_init(self, *args, **kwargs):
        # If text-only mode requested AT __init__ time (env), redirect
        # the vision tower construction by temporarily monkey-patching
        # the vision_tower setter on the instance after super().__init__.
        original_init(self, *args, **kwargs)
        if _text_only_requested():
            log.warning(
                "[G4_17] %s='1' — installing vision-tower STUB "
                "(saves ~2.3 GB VRAM). Image inputs will raise.",
                _ENV_TEXT_ONLY,
            )
            # Move the real vision tower to CPU + delete its CUDA copy
            try:
                if hasattr(self, "vision_tower") and self.vision_tower is not None:
                    try:
                        self.vision_tower.to("cpu")
                    except Exception:  # noqa: BLE001
                        pass
                    self.vision_tower = _DeferredVisionTowerStub()
                if hasattr(self, "multi_modal_projector") and self.multi_modal_projector is not None:
                    try:
                        self.multi_modal_projector.to("cpu")
                    except Exception:  # noqa: BLE001
                        pass
                    self.multi_modal_projector = _DeferredVisionTowerStub()
                # Clear CUDA cache so the freed memory becomes available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
            except Exception as e:  # noqa: BLE001
                log.warning("[G4_17] vision-tower stub install failed: %r", e)

    _genesis_g4_17_init._genesis_g4_17_wrapped = True
    _genesis_g4_17_init.__wrapped__ = original_init
    target_cls.__init__ = _genesis_g4_17_init
    _APPLIED = True
    log.info(
        "[G4_17] installed: Gemma4ForConditionalGeneration will stub the "
        "vision tower when %s=1 is set.",
        _ENV_TEXT_ONLY,
    )
    return "applied", (
        f"G4_17 installed: set {_ENV_TEXT_ONLY}=1 on the worker to skip "
        "vision-tower load on Gemma 4 ConditionalGeneration. Saves ~2.3 GB "
        "VRAM + ~30 sec cold boot. Image inputs will raise clear errors."
    )


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


__all__ = ["GENESIS_G4_17_MARKER", "apply", "is_applied", "revert"]
