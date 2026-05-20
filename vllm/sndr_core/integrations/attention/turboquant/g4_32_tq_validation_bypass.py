# SPDX-License-Identifier: Apache-2.0
"""G4_32 — bypass TurboQuantAttentionBackend.validate_configuration.

================================================================
PROBLEM
================================================================

Upstream vllm dev371 ships ``TurboQuantAttentionBackend`` with a
``validate_configuration`` that rejects Gemma 4 boot. Specifically::

    ValueError: Selected backend AttentionBackendEnum.TURBOQUANT is not
    valid for this configuration. Reason: ['kv_cache_dtype not supported']

Even though our ``cache_config.cache_dtype = "turboquant_4bit_nc"`` and
TQ's ``supports_kv_cache_dtype()`` accepts ``turboquant_*`` strings,
SOMEWHERE in the validation path the dtype gets coerced (or another
constraint fires) such that the backend rejects.

The user directive: stop debugging which validator fires — just
**bypass the validator** and let the actual TQ runtime code run. If
the runtime works on the real flow, the validator's restriction was
unnecessary or over-cautious for our specific setup.

================================================================
FIX
================================================================

Monkey-patch ``TurboQuantAttentionBackend.validate_configuration`` to
ALWAYS return an empty list (no invalid reasons). This forces vllm's
backend selector to accept TURBOQUANT regardless of any per-condition
check failure. The actual TQ compress/decompress kernels run at
runtime; if they're incompatible with the real config we'll see a
clearer runtime error (which is actionable), not a defensive boot
abort (which is a black box).

================================================================
RISK
================================================================

If the validator was correctly catching a real incompatibility, the
runtime path will crash later with the actual underlying issue. That's
still better than a black-box "not supported" because the runtime
trace identifies the exact call site.

For our use case (Gemma 4 31B AWQ + turboquant_4bit_nc + MTP K=4 +
2× A5000), we have empirical reason to believe TQ runtime should work
on text-only inference. G4_30 (multimodal unblock) addresses the
modal compatibility; this patch addresses the dtype/spec_config
compatibility.

================================================================
ENV FLAG
================================================================

``GENESIS_ENABLE_G4_32_TQ_VALIDATION_BYPASS=1`` — opt-in. Default OFF.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.gemma4.g4_32_tq_validation_bypass")

GENESIS_G4_32_MARKER = (
    "Genesis G4_32 TurboQuant validate_configuration bypass v1 "
    "(returns empty invalid_reasons — accept backend unconditionally)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_32_TQ_VALIDATION_BYPASS"
_APPLIED = False
_ORIGINAL = None


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Install validate_configuration bypass on TurboQuantAttentionBackend."""
    global _APPLIED, _ORIGINAL

    if not _env_enabled():
        return "skipped", (
            f"G4_32 disabled (set {_ENV_ENABLE}=1 to bypass upstream "
            "TurboQuant backend's validate_configuration)"
        )

    if _APPLIED:
        return "applied", "G4_32 already installed (idempotent)"

    try:
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionBackend,
        )
    except ImportError as e:
        return "skipped", (
            f"vllm TurboQuant backend not importable: {e}; "
            "G4_32 is no-op on this pin"
        )

    original = TurboQuantAttentionBackend.validate_configuration
    if getattr(original, "_genesis_g4_32_wrapped", False):
        _APPLIED = True
        return "applied", "G4_32 already wrapped (idempotent)"

    _ORIGINAL = original

    def _bypass_validate_configuration_inner(cls, **kwargs):
        """Always-pass validator. Logs the args once for forensics."""
        if not getattr(_bypass_validate_configuration_inner, "_logged", False):
            _bypass_validate_configuration_inner._logged = True
            sanitized = {
                k: (str(v)[:80] if v is not None else None)
                for k, v in kwargs.items()
            }
            log.warning(
                "[G4_32] BYPASS validate_configuration — kwargs=%s",
                sanitized,
            )
        return []  # no invalid reasons

    _bypass_validate_configuration_inner._genesis_g4_32_wrapped = True
    _bypass_validate_configuration_inner.__wrapped__ = original

    TurboQuantAttentionBackend.validate_configuration = classmethod(
        _bypass_validate_configuration_inner
    )
    _APPLIED = True

    log.info(
        "[G4_32] installed: TurboQuant validate_configuration always "
        "returns []. Backend will be selected unconditionally."
    )
    return "applied", (
        "G4_32 installed: TurboQuant backend's validate_configuration "
        "now always returns no errors. Runtime crashes (if any) will "
        "now show the actual incompatibility instead of the black-box "
        "validator rejection."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED, _ORIGINAL
    if not _APPLIED or _ORIGINAL is None:
        return False
    try:
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionBackend,
        )
        TurboQuantAttentionBackend.validate_configuration = _ORIGINAL
        _APPLIED = False
        return True
    except ImportError:
        return False


__all__ = ["GENESIS_G4_32_MARKER", "apply", "is_applied", "revert"]
