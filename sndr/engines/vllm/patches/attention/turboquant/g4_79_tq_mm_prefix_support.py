# SPDX-License-Identifier: Apache-2.0
"""G4_79 — declare ``supports_mm_prefix()`` on TurboQuantAttentionBackend.

================================================================
PROBLEM (pin 0.22.1rc1.dev259+g303916e93, fleet validation 2026-06-11)
================================================================

Booting Gemma-4-31B (multimodal) with ``--kv-cache-dtype
turboquant_4bit_nc --attention-backend TURBOQUANT`` fails worker init::

    ValueError: Selected backend AttentionBackendEnum.TURBOQUANT is not
    valid for this configuration. Reason: ['kv_cache_dtype not supported',
    'partial multimodal token full attention not supported']

The mm half is a NEW upstream validity gate on this pin
(``v1/attention/backend.py:301-304``): for models with
``model_config.is_mm_prefix_lm`` (hard architecture-level flag for
Gemma 4 MM — ``config/model.py:1274``), every candidate backend must
declare ``supports_mm_prefix()``. The base default is False
(``backend.py:217``) and the TQ backend does not override it.

================================================================
WHY DECLARING SUPPORT IS CORRECT (verified 2026-06-11, read-only)
================================================================

Gemma 4's vision and audio towers run in EAGER mode via
``AutoModel.from_config()`` (pristine ``gemma4_mm.py:1037/1053``) —
they never pass through vLLM's attention-backend machinery. The TQ
backend only serves the TEXT decoder's KV cache; multimodal embeddings
are already merged into the token stream before any TQ-quantized layer
sees them. Backends that DO declare support on this pin (Triton,
FlexAttention — ``flex_attention.py:112``) implement no mm-specific
logic in their decode paths either: the flag gates full-attention
treatment of image-token prefixes, which is a property of the
attention MASK handling shared by all decoder backends, not of the KV
storage format that TQ changes.

================================================================
SCOPE AND THE dtype HALF
================================================================

This patch fixes ONLY the mm_prefix reason. The sibling
``'kv_cache_dtype not supported'`` reason is suspected to be the AWQ
``kv_cache_scheme`` dtype override reaching the selection-time
validator (the G4_31 class of problem, one stage earlier than G4_31's
``Attention.__init__`` hook can reach). The first instrumented 31B
boot discriminates:

  - if the dtype reason disappears with G4_79 + G4_31 → done;
  - if it persists → either extend G4_31 to the selection-time
    resolution or fall back to G4_32 (blanket validator bypass,
    dev371-era empirical workaround) for the boot, then root-cause.

Opt-in (``GENESIS_ENABLE_G4_79_TQ_MM_PREFIX=1``); no-op for
non-multimodal models (the validator never consults mm_prefix when
``is_mm_prefix_lm`` is False).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.gemma4.g4_79_tq_mm_prefix")

GENESIS_G4_79_MARKER_ATTR = "_genesis_g4_79_mm_prefix"
_ORIG_ATTR = "_genesis_g4_79_orig_supports_mm_prefix"


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_G4_79_TQ_MM_PREFIX", ""
    ).strip().lower() in ("1", "true", "yes", "on")


def _cls_is_patched(cls: type) -> bool:
    return bool(cls.__dict__.get(GENESIS_G4_79_MARKER_ATTR, False))


def inject_mm_prefix_support(cls: type) -> bool:
    """Install ``supports_mm_prefix -> True`` on *cls*.

    Returns True when the class was patched by this call, False when it
    already carried the Genesis override (idempotent re-apply).
    """
    if _cls_is_patched(cls):
        return False

    # Keep the pre-patch classmethod (may live on a base class) so
    # revert restores the exact prior behavior.
    setattr(cls, _ORIG_ATTR, cls.__dict__.get("supports_mm_prefix"))

    @classmethod
    def supports_mm_prefix(_cls) -> bool:  # noqa: ANN001 — vllm signature
        return True

    cls.supports_mm_prefix = supports_mm_prefix
    setattr(cls, GENESIS_G4_79_MARKER_ATTR, True)
    return True


def revert_mm_prefix_support(cls: type) -> None:
    """Undo :func:`inject_mm_prefix_support` (test / rollback helper)."""
    if not _cls_is_patched(cls):
        return
    orig = getattr(cls, _ORIG_ATTR, None)
    if orig is None:
        # Came from a base class — drop our override entirely.
        if "supports_mm_prefix" in cls.__dict__:
            delattr(cls, "supports_mm_prefix")
    else:
        cls.supports_mm_prefix = orig
    setattr(cls, GENESIS_G4_79_MARKER_ATTR, False)


def apply() -> tuple[str, str]:
    """Apply G4_79. Never raises."""
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("G4_79")
    log_decision("G4_79", decision, reason)
    if not decision:
        return "skipped", reason

    try:
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionBackend,
        )
    except ImportError as e:
        # Fail LOUD when enabled: an operator who opted in must not
        # believe the unblock is active while the import target moved.
        log.warning(
            "[G4_79] TurboQuantAttentionBackend import failed (%s) — "
            "mm_prefix unblock NOT installed; Gemma 4 MM + TQ boot will "
            "still be rejected by the validity gate.", e,
        )
        return "failed", f"TurboQuantAttentionBackend import failed: {e}"

    if not inject_mm_prefix_support(TurboQuantAttentionBackend):
        return "applied", "idempotent (Genesis override already installed)"

    log.info(
        "[G4_79] TurboQuantAttentionBackend.supports_mm_prefix -> True "
        "installed. Gemma 4 MM models pass the mm_prefix validity gate; "
        "vision/audio towers run eager and never reach TQ. If boot still "
        "rejects with 'kv_cache_dtype not supported', see the module "
        "docstring dtype-half recipe (G4_31 selection-time extension or "
        "G4_32 fallback)."
    )
    return "applied", (
        "supports_mm_prefix=True injected on TurboQuantAttentionBackend "
        "(Gemma 4 MM + TQ validity unblock, mm half)"
    )


def is_applied() -> bool:
    try:
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionBackend,
        )
    except ImportError:
        return False
    return _cls_is_patched(TurboQuantAttentionBackend)
