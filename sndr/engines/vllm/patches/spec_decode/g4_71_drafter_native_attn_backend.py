# SPDX-License-Identifier: Apache-2.0
"""G4_71 — Force native (FlashAttn) attention backend for Gemma 4 MTP drafter.

================================================================
PROBLEM
================================================================

PN260 trace localized the K>=2 cudaErrorIllegalAddress to drafter's
own attention layers (draft_model.layers.0..3) calling
`triton_turboquant_decode_attention` with a native FlashAttn-shaped
KV cache (5-dim bf16) instead of the TurboQuant packed 4-dim uint8
shape the kernel expects.

The mismatch arises because:

  1. `--attention-backend TURBOQUANT` is set globally.
  2. Drafter's `Attention.__init__` reads `cache_config.cache_dtype`
     = "turboquant_4bit_nc" → its `self.kv_cache_dtype` stays
     "turboquant_4bit_nc".
  3. Drafter's `get_attn_backend()` therefore returns
     `TurboQuantAttentionBackend` → impl = TurboQuantAttentionImpl.
  4. G4_69 does NOT reroute drafter because G4_69 only triggers on
     `kv_cache_dtype == "auto"` (or any non-TQ string); drafter's
     dtype is the TQ-prefix.
  5. But downstream, drafter's KVCacheTensor is bound to a native
     5-dim bf16 tensor (path is not yet fully traced — possibly
     spec_decode worker rebinding, possibly per-layer allocator giving
     drafter a native spec via PN259c if drafter's spec class is
     `FullAttentionSpec` somehow).

PN261-A guard converts the resulting CUDA illegal address into a
Python RuntimeError with layer + call_site info. PN261-C (this patch)
addresses the root cause: drafter must never have a TurboQuant impl
when its physical KV cache is native.

================================================================
FIX
================================================================

Wrap `vllm.model_executor.layers.attention.attention.Attention.__init__`.
Before delegating to the original, inspect `kwargs.get("prefix")`. If
the prefix starts with the configured drafter prefix (default:
"draft_model."), substitute `kwargs["attn_backend"]` with the resolved
FlashAttention v2 backend class. The original init then uses this
explicit backend and never calls `get_attn_backend()`, so the cached
TurboQuant result for the same shape signature does not pollute
drafter's dispatch.

We deliberately do NOT mutate `kv_cache_dtype` — drafter's spec
(via `get_kv_cache_spec`) is already returning native FullAttentionSpec
in the observed trace, so leaving the dtype alone is harmless. If
future investigation reveals drafter's spec is TQ when we wanted
native, this patch can additionally force `kv_cache_dtype="auto"` for
drafter prefixes (mirroring the skip-layer mechanism).

================================================================
INTERACTION
================================================================

  * G4_31 still preserves turboquant_* dtype against AWQ overrides
    for target — unaffected.
  * G4_69 still reroutes target's skip-listed layers (kv_cache_dtype
    = "auto") to FlashAttn — unaffected (drafter is now bypassing
    that path entirely via explicit attn_backend kwarg).
  * G4_60G's spec dispatch is unchanged.
  * PN259c split allocator continues to enforce no-cross-layout
    aliasing.
  * PN260 trace + PN261-A assert continue to guard against future
    regressions of this specific mismatch.

================================================================
ENV FLAG
================================================================

  GENESIS_ENABLE_G4_71_DRAFTER_NATIVE_BACKEND=1   (opt-in)

Optional override:
  GENESIS_G4_71_DRAFTER_PREFIX=draft_model.       (default)

When unset: original behavior (drafter Attention initialized with
TurboQuant backend, crashes on first decode under K+1 verify >=2).

================================================================
ACCEPTANCE GATES
================================================================

Per user 2026-05-19 PN261 directive:

  Gate 0: MTP OFF + G4_71 — no regression
  Gate 1: MTP K=2 + G4_71 — no cudaErrorIllegalAddress (and no PN261-A
          RuntimeError either, because drafter is no longer TQ-impl)
  Gate 2: MTP K=4 + G4_71 — no crash
  Gate 3: PN248 acceptance — accepted_per_req > 0 (if 0, return to
          H8a skip-list / KV-sharing investigation)

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.spec_decode.g4_71_drafter_native_attn_backend")

GENESIS_G4_71_MARKER = (
    "Genesis G4_71 Force FlashAttn attention backend for Gemma 4 MTP "
    "drafter layers (prefix 'draft_model.' by default)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_71_DRAFTER_NATIVE_BACKEND"
_ENV_PREFIX = "GENESIS_G4_71_DRAFTER_PREFIX"
_APPLIED = False
_ORIGINAL_INIT = None
_REROUTE_COUNT = [0]


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _drafter_prefix() -> str:
    return os.environ.get(_ENV_PREFIX, "draft_model.").strip()


def apply() -> tuple[str, str]:
    """Install Attention.__init__ wrap that routes drafter to FlashAttn."""
    global _APPLIED, _ORIGINAL_INIT

    if not _env_enabled():
        return "skipped", (
            f"G4_71 disabled (set {_ENV_ENABLE}=1 to force drafter "
            "Attention layers onto FlashAttn backend)"
        )

    if _APPLIED:
        return "applied", "G4_71 already installed (idempotent)"

    try:
        from vllm.model_executor.layers.attention.attention import Attention
    except ImportError as e:
        return "skipped", (
            f"vllm.model_executor.layers.attention.attention not importable: {e}"
        )

    original = Attention.__init__
    if getattr(original, "_genesis_g4_71_wrapped", False):
        _APPLIED = True
        return "applied", "Attention.__init__ already wrapped (idempotent)"
    _ORIGINAL_INIT = original

    drafter_prefix = _drafter_prefix()

    def _wrapped_init(self, *args, **kwargs):
        """Force FlashAttn backend for drafter Attention layers.

        Side effects on ``self``:
          * ``self._genesis_g4_71_is_drafter = True`` — read by G4_72 in
            ``get_kv_cache_spec`` to override TQ spec with native spec.
          * ``self._genesis_g4_71_drafter_prefix = prefix`` — kept for
            diagnostic logs.

        Marker is stamped BOTH before and after delegating to the
        original ``Attention.__init__``:
          * Before — guards against the (unlikely) case where the
            original init internally calls ``self.get_kv_cache_spec``
            during construction; G4_72 then sees the marker.
          * After — defensive re-stamp in case original init re-assigned
            ``self.__dict__`` (Python instance attributes survive
            standard ``__init__`` patterns, but the cost is one setattr).
        """
        prefix = kwargs.get("prefix", "") or ""
        is_drafter = (
            isinstance(prefix, str)
            and prefix.startswith(drafter_prefix)
        )
        # Pre-init marker stamp (belt) — safe because Python lets us
        # attach attributes to a freshly-allocated instance.
        if is_drafter:
            try:
                self._genesis_g4_71_is_drafter = True
                self._genesis_g4_71_drafter_prefix = prefix
            except Exception:
                # Slotted Attention class would reject this; G4_72 then
                # falls back to its own prefix re-check.
                pass

        if is_drafter and kwargs.get("attn_backend") is None:
            try:
                # Resolve FlashAttention v2 backend on demand. We pick
                # FLASH_ATTN as it is the canonical native attention
                # backend on Ampere/Ada with bf16+head_size 256/512 and
                # matches what G4_69's auto-priority dispatcher would
                # have chosen for skip-listed target layers.
                from vllm.v1.attention.backends.registry import (
                    AttentionBackendEnum,
                )
                flash_backend_cls = AttentionBackendEnum.FLASH_ATTN.get_class()
                kwargs["attn_backend"] = flash_backend_cls
                _REROUTE_COUNT[0] += 1
                if _REROUTE_COUNT[0] <= 6:
                    log.warning(
                        "[G4_71] drafter Attention init detected "
                        "(prefix=%r) — forcing attn_backend=FlashAttn "
                        "to prevent TurboQuant kernel/cache mismatch. "
                        "(call #%d)",
                        prefix,
                        _REROUTE_COUNT[0],
                    )
                elif _REROUTE_COUNT[0] == 7:
                    log.warning(
                        "[G4_71] further drafter-reroute logs suppressed "
                        "(count > 6)"
                    )
            except Exception as _e:
                log.warning(
                    "[G4_71] failed to resolve FlashAttn backend for "
                    "drafter prefix %r: %s — falling through to "
                    "original behavior (will likely crash at first "
                    "decode under MTP K>=2).",
                    prefix,
                    _e,
                )

        result = original(self, *args, **kwargs)

        # Post-init re-stamp (suspenders) — defensive in case the
        # original __init__ replaced self.__dict__ via some inheritance
        # hook. Pre-init stamp above is the primary guarantee.
        if is_drafter:
            try:
                self._genesis_g4_71_is_drafter = True
                self._genesis_g4_71_drafter_prefix = prefix
            except Exception:
                pass

        return result

    _wrapped_init._genesis_g4_71_wrapped = True  # type: ignore[attr-defined]
    Attention.__init__ = _wrapped_init  # type: ignore[method-assign]
    _APPLIED = True

    log.info(
        "[G4_71] installed: Attention.__init__ now forces FlashAttn "
        "backend when prefix starts with %r.",
        drafter_prefix,
    )
    return "applied", (
        f"G4_71 installed: drafter Attention init detection on prefix "
        f"{drafter_prefix!r}; substitutes attn_backend=FlashAttn for "
        f"those layers to prevent TurboQuant impl on native KV cache."
    )


def is_applied() -> bool:
    return _APPLIED


def reroute_count() -> int:
    return _REROUTE_COUNT[0]


def revert() -> bool:
    global _APPLIED, _ORIGINAL_INIT
    if not _APPLIED or _ORIGINAL_INIT is None:
        return False
    try:
        from vllm.model_executor.layers.attention.attention import Attention
        Attention.__init__ = _ORIGINAL_INIT  # type: ignore[method-assign]
    except ImportError:
        return False
    _APPLIED = False
    _ORIGINAL_INIT = None
    return True


__all__ = [
    "GENESIS_G4_71_MARKER",
    "apply",
    "is_applied",
    "reroute_count",
    "revert",
]
