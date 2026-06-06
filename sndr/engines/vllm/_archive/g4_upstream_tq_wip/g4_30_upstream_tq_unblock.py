# SPDX-License-Identifier: Apache-2.0
"""G4_30 — unblock upstream TurboQuant backend for multimodal Gemma 4.

================================================================
WHY THIS EXISTS
================================================================

vllm dev371+ ships a built-in ``TurboQuantAttentionBackend`` that
delivers REAL KV-cache memory savings (3.8× compression at +2.71% PPL
for ``turboquant_4bit_nc``). The implementation is text-only:

  * ``supports_mm_prefix()`` inherits ``False`` from the base class.
  * Gemma 4 is multimodal (``is_mm_prefix_lm = True``).
  * Backend selection rejects with::

        Reason: ['partial multimodal token full attention not supported']

For pure **text** inference (no image inputs), the multimodal-prefix
attention path is never invoked at runtime — the math is identical
whether the model is "multimodal-capable" or not. The check at boot
time is precautionary; we can safely opt-in by overriding
``supports_mm_prefix`` to True.

================================================================
WHAT THIS DOES
================================================================

Monkey-patches ``TurboQuantAttentionBackend`` (and its known sibling
backends if present) so that ``supports_mm_prefix()`` returns True.
This allows vllm's backend selector to accept TURBOQUANT for Gemma 4
boot. Activated by env flag ``GENESIS_ENABLE_G4_30_TQ_UNBLOCK=1``.

================================================================
RISKS
================================================================

If a request includes IMAGE tokens (Gemma 4 multimodal input), the
TurboQuant attention forward path may produce incorrect attention
scores on the image-text boundary — the original validator was
designed to prevent exactly this. Operators using G4_30 must
guarantee text-only workloads at the API layer.

The compress/decompress kernels themselves are dtype-agnostic; the
correctness concern is purely about the cross-modal attention mask
handling that the multimodal-prefix path expects.

================================================================
ENV FLAG
================================================================

``GENESIS_ENABLE_G4_30_TQ_UNBLOCK=1`` enables the override.
Disabled by default — operators must opt-in explicitly.

================================================================
COMPOSITION
================================================================

* Composes with the launch flags ``--attention-backend TURBOQUANT``
  + ``--kv-cache-dtype turboquant_<preset>``.
* Replaces the prior Genesis G4_19/19b/19c shadow round-trip (which
  delivered quality at the cost of no real memory savings). Operators
  using G4_30 should turn G4_19c OFF.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.gemma4.g4_30_tq_unblock")

GENESIS_G4_30_MARKER = (
    "Genesis G4_30 upstream TurboQuant backend multimodal-unblock v1 "
    "(allows --attention-backend TURBOQUANT on Gemma 4 text-only inference)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_30_TQ_UNBLOCK"
_APPLIED = False
_ORIGINAL_SUPPORTS_MM_PREFIX = None


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Override TurboQuantAttentionBackend.supports_mm_prefix → True."""
    global _APPLIED, _ORIGINAL_SUPPORTS_MM_PREFIX

    if not _env_enabled():
        return "skipped", (
            f"G4_30 disabled (set {_ENV_ENABLE}=1 to allow the upstream "
            "TurboQuant backend on Gemma 4 multimodal — TEXT-ONLY inference)"
        )

    if _APPLIED:
        return "applied", "G4_30 already installed (idempotent)"

    try:
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionBackend,
        )
    except ImportError as e:
        return "skipped", (
            f"vllm TurboQuant backend not importable: {e}; "
            "G4_30 is no-op on this pin"
        )

    # supports_mm_prefix may be a classmethod (per base class) — bind override
    original = getattr(TurboQuantAttentionBackend, "supports_mm_prefix", None)
    if original is None:
        return "skipped", (
            "TurboQuantAttentionBackend has no supports_mm_prefix attr; "
            "G4_30 is no-op on this pin (API may have changed)"
        )

    if getattr(original, "_genesis_g4_30_wrapped", False):
        _APPLIED = True
        return "applied", "G4_30 already wrapped (idempotent)"

    _ORIGINAL_SUPPORTS_MM_PREFIX = original

    # IMPORTANT: ``classmethod`` objects make their internal attributes
    # readonly (mappingproxy-style). Setting marker attributes on the
    # classmethod descriptor raises "AttributeError: readonly attribute".
    # Set markers on the INNER function before wrapping in classmethod.
    def _genesis_supports_mm_prefix_inner(cls) -> bool:
        return True

    _genesis_supports_mm_prefix_inner._genesis_g4_30_wrapped = True
    _genesis_supports_mm_prefix_inner.__wrapped__ = original

    TurboQuantAttentionBackend.supports_mm_prefix = classmethod(
        _genesis_supports_mm_prefix_inner
    )
    _APPLIED = True

    log.info(
        "[G4_30] installed: TurboQuantAttentionBackend.supports_mm_prefix "
        "now returns True — Gemma 4 boot with --attention-backend "
        "TURBOQUANT will pass validation. WARNING: image input paths may "
        "be miscomputed; text-only inference only."
    )
    return "applied", (
        "G4_30 installed: TurboQuant backend now accepts multimodal "
        "model_config (supports_mm_prefix → True). Use text-only inputs "
        "until upstream adds proper multimodal-prefix support."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    """Restore upstream supports_mm_prefix. Returns True on success."""
    global _APPLIED, _ORIGINAL_SUPPORTS_MM_PREFIX
    if not _APPLIED or _ORIGINAL_SUPPORTS_MM_PREFIX is None:
        return False
    try:
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionBackend,
        )
        TurboQuantAttentionBackend.supports_mm_prefix = (
            _ORIGINAL_SUPPORTS_MM_PREFIX
        )
        _APPLIED = False
        return True
    except ImportError:
        return False


__all__ = ["GENESIS_G4_30_MARKER", "apply", "is_applied", "revert"]
