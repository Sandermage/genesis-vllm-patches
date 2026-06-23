# SPDX-License-Identifier: Apache-2.0
"""G4_60L — install ``supports_mm_prefix=True`` on stock TurboQuantAttentionBackend.

================================================================
PROBLEM
================================================================

vllm pin ``0.20.2rc1.dev371+gbf610c2f5`` ships a
``TurboQuantAttentionBackend`` class that does NOT override the base-class
``supports_mm_prefix`` classmethod. It therefore inherits the
default ``False`` from
``vllm/v1/attention/backend.py:215-218``.

For NON-multimodal models (most Qwen) this is fine. For Gemma 4
31B AWQ it breaks engine init because Gemma 4 is loaded via
``Gemma4ForMultimodalLM`` regardless of the
``--language-model-only`` flag (which only skips vision / audio
tower weights, NOT the MM wrapper class itself, NOT the
``model_arch_config.is_mm_prefix_lm = True`` attribute).

The MM-prefix-LM flag propagates to ``Attention.__init__``:

    self.use_mm_prefix = (
        model_config is not None and model_config.is_mm_prefix_lm
    )

Then ``Attention.__init__`` calls ``get_attn_backend(...,
use_mm_prefix=self.use_mm_prefix, ...)``. The base validator
``vllm/v1/attention/backend.py:271-298`` adds
"partial multimodal token full attention not supported" to the
invalid_reasons list whenever ``use_mm_prefix == True`` AND the
selected backend's ``supports_mm_prefix()`` returns ``False``.

With Phase 7.G4.31B.K4-BACKEND-FIX (commit ``4096645b``) now
emitting ``--attention-backend TURBOQUANT`` from
``profile.backend_plan.target_default``, vllm hard-fails at engine
init with the bare ``ValueError`` (no fallback search runs when an
explicit backend was requested).

The PR #42637 overlay file
``sndr/engines/vllm/patches/attention/turboquant/overlays/pr42637/
turboquant_attn.py`` (lines 221-223) adds the missing override
explicitly. The β'-A hand-launcher bind-mounts that file at
container start so the overlay class wins. The V2 compose path
emits no bind-mount, so stock vllm's class lacking the override
is what reaches runtime.

================================================================
FIX
================================================================

At apply() time, monkey-patch the stock class:

    TurboQuantAttentionBackend.supports_mm_prefix = classmethod(
        lambda cls: True,
    )

This is byte-equivalent to the overlay's behavior for the
``supports_mm_prefix`` check. No other overlay differences are
addressed by this patch — the second observed rejection reason
"kv_cache_dtype not supported" is a separate skip-layer issue
(global ``--attention-backend TURBOQUANT`` colliding with
per-layer skip-layer ``kv_cache_dtype='auto'`` resolution) that is
handled by G4_69_SKIP_LAYERS_NATIVE_BACKEND, OR — if that proves
insufficient — by a follow-up Phase 7.G4.31B.K4-SKIP-LAYER-
BACKEND-FIX.

================================================================
SCOPE
================================================================

Affects ONLY processes that import ``TurboQuantAttentionBackend`` from
stock vllm. Idempotent: if the overlay file is already bind-
mounted (β'-A hand-launcher path) the class already has
``supports_mm_prefix=True`` and this patch no-ops.

Qwen 3.5 / 3.6 TQ presets are unaffected — Qwen architectures
don't set ``is_mm_prefix_lm=True``, so the validator never checks
``supports_mm_prefix``.

26B-A4B presets are unaffected — they use ``kv_cache_dtype=auto``
and never reach the TURBOQUANT validator at all.

================================================================
LIFECYCLE
================================================================

  vllm_version_range: "<0.21"

When PR #42637 lands upstream and the Genesis pin bumps past 0.21,
the stock class will already have ``supports_mm_prefix=True``
natively. The idempotency check in this patch's ``apply()`` will
make it a no-op, and the patch can be retired in the next
audit-driven cleanup.

================================================================
ENV FLAG
================================================================

``GENESIS_ENABLE_G4_60L_TQ_BACKEND_MM_PREFIX=1`` enables the
override. Default OFF; opted in via the
``gemma4-31b-tq-mtp-structured-k4`` profile's ``patches_delta.enable``
block.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.turboquant.g4_60l")

GENESIS_G4_60L_MARKER = (
    "Genesis G4_60L — supports_mm_prefix=True override on stock "
    "TurboQuantAttentionBackend (PR42637 partial backport, monkey-patch only)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_60L_TQ_BACKEND_MM_PREFIX"
_APPLIED = False
_OVERLAY_ALREADY_HAS_OVERRIDE = False


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def is_applied() -> bool:
    return _APPLIED


def apply() -> tuple[str, str]:
    """Install supports_mm_prefix=True on stock TurboQuantAttentionBackend.

    Returns:
      ("skipped", msg) — env gate off, vllm class missing, or already
                         overridden by overlay bind-mount.
      ("applied", msg) — override installed (or re-attempted and
                         confirmed installed).
    """
    global _APPLIED, _OVERLAY_ALREADY_HAS_OVERRIDE

    if not _env_enabled():
        return "skipped", (
            f"G4_60L disabled "
            f"(set {_ENV_ENABLE}=1 to enable the supports_mm_prefix "
            "override on stock TurboQuantAttentionBackend)"
        )

    if _APPLIED:
        return "applied", "G4_60L already installed (idempotent)"

    try:
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionBackend,
        )
    except ImportError as e:
        return "skipped", (
            f"vllm TurboQuantAttentionBackend not importable: {e}"
        )

    # Idempotency: if the PR42637 overlay file is bind-mounted, the
    # class already has supports_mm_prefix returning True natively.
    # In that case the monkey-patch is a no-op — record the
    # observation and mark applied so dispatch reporting is consistent.
    try:
        natively_true = TurboQuantAttentionBackend.supports_mm_prefix() is True
    except Exception:  # noqa: BLE001
        natively_true = False

    if natively_true:
        _OVERLAY_ALREADY_HAS_OVERRIDE = True
        _APPLIED = True
        return "applied", (
            "G4_60L no-op: TurboQuantAttentionBackend.supports_mm_prefix=True "
            "already present (overlay bind-mount or prior install)"
        )

    TurboQuantAttentionBackend.supports_mm_prefix = classmethod(lambda cls: True)
    _APPLIED = True
    log.info(
        "[G4_60L] installed: TurboQuantAttentionBackend.supports_mm_prefix=True "
        "(stock class did not override; overlay bind-mount not active)"
    )
    return "applied", (
        "G4_60L installed: supports_mm_prefix=True override on stock "
        "TurboQuantAttentionBackend (closes 'partial multimodal token full "
        "attention not supported' rejection for Gemma 4 31B AWQ + "
        "TURBOQUANT backend on the V2 compose path)"
    )


__all__ = [
    "GENESIS_G4_60L_MARKER",
    "apply",
    "is_applied",
]
