# SPDX-License-Identifier: Apache-2.0
"""G4_71b — Per-layer drafter backend force: route head_size=256 to TRITON_ATTN.

================================================================
WHY
================================================================

β control + PN271b (2026-05-20) proved that the canonical Gemma 4 +
TQ + MTP launcher has a kernel-vs-storage contract mismatch on
drafter[0..2]:

  --attention-backend TURBOQUANT  ←  global engine config
  G4_69 + skip-list [58,59]       ←  target[58/59] forced native bf16
  drafter[0..2]                   ←  inherit global TQ backend
  target[58]                      ←  native bf16 Triton storage

Drafter's TurboQuantAttentionImpl reads native bf16 bytes from
target's bound cache as TQ-packed → garbage attention → acceptance=0.

PN271b's safety guard now denies this configuration non-overridably
(KERNEL_STORAGE_DTYPE_MISMATCH). To run β′-A (the corrected control
experiment with physical kv_sharing on layers 0..2), the launcher
must produce a contract that the guard accepts as EXACT_COPY:

  drafter[0..2] ←  Triton NHD native bf16
  target[58]    ←  Triton NHD native bf16
  kv_sharing ON

This patch is the drafter-side half of that contract.

================================================================
WHAT IT DOES
================================================================

Wraps ``Attention.__init__``. When the layer is a drafter
(prefix starts with ``draft_model.``) AND ``head_size == 256``
(sliding drafter layers 0..2), override:

  kwargs["attn_backend"] = TRITON_ATTN

BEFORE delegating to the original init. Also resets
``self.kv_cache_dtype = 'auto'`` post-init so the drafter's inner
Attention reads bf16 (not TQ) bytes regardless of what the global
``--attention-backend`` flag set.

Drafter layer 3 (head_size=512) is handled by G4_75 (already exists).
Both patches compose cleanly: each owns a disjoint head_size class.

================================================================
WHEN TO ENABLE
================================================================

Only with β′-A research opt-ins:

  GENESIS_ENABLE_G4_71B_DRAFTER_SLIDING_TRITON=1
  GENESIS_ENABLE_G4_71_DRAFTER_NATIVE_BACKEND=0     (legacy FA force OFF)
  GENESIS_ENABLE_G4_76_DISABLE_DRAFTER_KV_SHARING=0 (allow native sharing)
  GENESIS_ENABLE_G4_75_DRAFTER_HEAD512_TRITON=1     (layer 3 to Triton)

This patch is opt-in. Do NOT enable in production. The β′-A launcher
should set GENESIS_ALLOW_SPEC_DECODE_KV_ADAPTER + FUNCTIONAL_UNKNOWN
envs only if PN271b verdict requires them; if PN271b sees
EXACT_COPY for layers 0..2 (the design goal), no override env is
needed for those pairs.

================================================================
ENV FLAGS
================================================================

  GENESIS_ENABLE_G4_71B_DRAFTER_SLIDING_TRITON=1   (master opt-in)
  GENESIS_G4_71B_DRAFTER_PREFIX=draft_model.       (override prefix)
  GENESIS_G4_71B_HEAD_THRESHOLD=256                (head_size that triggers)
  GENESIS_G4_71B_TARGET_BACKEND=TRITON_ATTN

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.spec_decode.g4_71b_drafter_sliding_triton")

GENESIS_G4_71B_MARKER = (
    "Genesis G4_71b Route drafter sliding (head_size=256) layers to "
    "TRITON_ATTN + reset kv_cache_dtype='auto' "
    "(β′-A enabler for clean kv_sharing contract)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_71B_DRAFTER_SLIDING_TRITON"
_ENV_PREFIX = "GENESIS_G4_71B_DRAFTER_PREFIX"
_ENV_HEAD_THRESHOLD = "GENESIS_G4_71B_HEAD_THRESHOLD"
_ENV_TARGET_BACKEND = "GENESIS_G4_71B_TARGET_BACKEND"
_APPLIED = False
_ORIGINAL_INIT = None
_REROUTE_COUNT = [0]


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _drafter_prefix() -> str:
    return os.environ.get(_ENV_PREFIX, "draft_model.").strip()


def _head_threshold() -> int:
    raw = os.environ.get(_ENV_HEAD_THRESHOLD, "256").strip()
    try:
        return int(raw)
    except ValueError:
        return 256


def _target_backend_name() -> str:
    return os.environ.get(_ENV_TARGET_BACKEND, "TRITON_ATTN").strip()


def apply() -> tuple[str, str]:
    global _APPLIED, _ORIGINAL_INIT

    if not _env_enabled():
        return "skipped", (
            f"G4_71b disabled (set {_ENV_ENABLE}=1 to route drafter "
            "sliding layers to TRITON_ATTN; enables β′-A clean contract)"
        )
    if _APPLIED:
        return "applied", "G4_71b already installed (idempotent)"

    log.warning("[G4_71b] apply() entered")

    try:
        from vllm.model_executor.layers.attention.attention import Attention
    except Exception as e:  # noqa: BLE001
        log.warning("[G4_71b] SKIP: Attention not importable: %s", e)
        return "skipped", f"Attention not importable: {e!r}"

    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
    except Exception as e:  # noqa: BLE001
        log.warning(
            "[G4_71b] SKIP: AttentionBackendEnum not importable: %s", e,
        )
        return "skipped", f"AttentionBackendEnum not importable: {e!r}"

    target_backend_name = _target_backend_name()
    try:
        target_backend_cls = AttentionBackendEnum[
            target_backend_name].get_class()
    except Exception as e:  # noqa: BLE001
        log.warning(
            "[G4_71b] SKIP: target backend %r not resolvable: %s",
            target_backend_name, e,
        )
        return "skipped", (
            f"target backend {target_backend_name!r} not resolvable: {e!r}"
        )

    original = Attention.__init__
    if getattr(original, "_genesis_g4_71b_wrapped", False):
        _APPLIED = True
        return "applied", "Attention.__init__ already wrapped by G4_71b"
    _ORIGINAL_INIT = original

    drafter_prefix = _drafter_prefix()
    head_threshold = _head_threshold()
    log.warning(
        "[G4_71b] import phase OK — drafter_prefix=%r head_threshold=%d "
        "target_backend=%s (%s); wrapping Attention.__init__",
        drafter_prefix, head_threshold, target_backend_name,
        target_backend_cls.__name__,
    )

    def _wrapped_init(self, *args, **kwargs):
        prefix = kwargs.get("prefix", "") or ""
        head_size = kwargs.get("head_size", None)
        if head_size is None and len(args) >= 2:
            head_size = args[1]
        is_drafter_sliding = (
            isinstance(prefix, str)
            and prefix.startswith(drafter_prefix)
            and isinstance(head_size, int)
            and head_size == head_threshold
        )

        if is_drafter_sliding:
            # Pre-init marker (belt)
            try:
                self._genesis_g4_71b_drafter_sliding_triton = True
                self._genesis_g4_71b_target_backend = target_backend_name
            except Exception:
                pass
            kwargs["attn_backend"] = target_backend_cls
            _REROUTE_COUNT[0] += 1
            if _REROUTE_COUNT[0] <= 6:
                log.warning(
                    "[G4_71b] drafter head_size==%d (prefix=%r) — "
                    "overriding attn_backend to %s for clean kv_sharing "
                    "contract with target[58] (Triton NHD bf16). "
                    "(call #%d)",
                    head_size, prefix, target_backend_name,
                    _REROUTE_COUNT[0],
                )

        result = original(self, *args, **kwargs)

        if is_drafter_sliding:
            # Post-init re-stamp + reset kv_cache_dtype to native.
            # The global engine config (--attention-backend TURBOQUANT)
            # propagates kv_cache_dtype='turboquant_4bit_nc' to every
            # Attention layer by default. We must reset this for drafter
            # so the Triton impl reads bytes as bf16, not TQ-packed.
            try:
                self._genesis_g4_71b_drafter_sliding_triton = True
                self._genesis_g4_71b_target_backend = target_backend_name
                # Reset the kv_cache_dtype label (defensive)
                if hasattr(self, "kv_cache_dtype"):
                    if str(getattr(self, "kv_cache_dtype",
                                   "")).lower() not in ("auto", ""):
                        log.warning(
                            "[G4_71b] resetting kv_cache_dtype "
                            "from %r to 'auto' for drafter %s",
                            self.kv_cache_dtype, prefix,
                        )
                        self.kv_cache_dtype = "auto"
            except Exception as _e:  # noqa: BLE001
                log.warning(
                    "[G4_71b] post-init stamp/reset failed: %s", _e,
                )

        return result

    _wrapped_init._genesis_g4_71b_wrapped = True  # type: ignore[attr-defined]
    Attention.__init__ = _wrapped_init  # type: ignore[method-assign]
    _APPLIED = True

    log.warning(
        "[G4_71b] INSTALLED: drafter (prefix %r) head_size==%d -> %s; "
        "kv_cache_dtype reset to 'auto' post-init.",
        drafter_prefix, head_threshold, target_backend_name,
    )
    return "applied", (
        f"G4_71b installed: drafter sliding (head={head_threshold}) "
        f"-> {target_backend_name} + kv_cache_dtype='auto' reset"
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
    "GENESIS_G4_71B_MARKER",
    "apply",
    "is_applied",
    "reroute_count",
    "revert",
]
