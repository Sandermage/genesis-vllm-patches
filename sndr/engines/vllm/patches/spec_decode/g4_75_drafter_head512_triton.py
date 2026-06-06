# SPDX-License-Identifier: Apache-2.0
"""G4_75 — Per-layer drafter backend split: route head_size=512 to TRITON_ATTN.

================================================================
PROBLEM (PN264)
================================================================

After G4_71 + G4_72 + G4_73 + G4_74 (with MAX_BLOCKS cap), K=2 boot
reaches "Application startup complete" and PN262 confirms drafter
sliding layers (0..2, head_size=256) work correctly with HND
FlashAttn cache. But first prompt crashes::

    RuntimeError: FlashAttention forward only supports head dimension
    at most 256

Drafter architecture is asymmetric:

  draft_model.layers.0..2:  sliding window, head_size=256, num_kv=8
  draft_model.layers.3:     full attention,  head_size=512, num_kv=2

FlashAttention v2 (the bundled `vllm_flash_attn` in this pin) caps
head_size at 256. Layer 3 fails on the assertion.

Backend capability probe (2026-05-19) results::

  FLASH_ATTN:    head_size cap 256                  (rejects 512)
  TRITON_ATTN:   supports_head_size(h) = (h >= 32)  (ACCEPTS 512)
  FLASHINFER:    supported [64, 128, 256]           (rejects 512)
  FLEX_ATTENTION: no explicit limit                 (untested)
  TORCH_SDPA:    not registered in this pin

TRITON_ATTN is the surgical choice: it supports any head_size >= 32
and is the canonical fallback for non-FlashAttn-compatible shapes on
Ampere/Ada cards.

================================================================
FIX
================================================================

Wrap `Attention.__init__` (similar to G4_71). When the layer is a
drafter and `head_size == 512`, override `kwargs["attn_backend"]` with
``AttentionBackendEnum.TRITON_ATTN.get_class()`` BEFORE delegating to
the original init. G4_71's blanket FlashAttn override is preempted
because G4_75 applies AFTER G4_71 in the boot sequence (it's the
outer wrap, so its kwargs mutation wins).

Also stamps a marker on `self` so G4_74 can skip Triton-routed drafter
layers (Triton uses NHD `(num_blocks, 2, ...)` natively; transposing
it to HND would break Triton's contract)::

    self._genesis_g4_75_drafter_triton = True

================================================================
ENV FLAGS
================================================================

  GENESIS_ENABLE_G4_75_DRAFTER_HEAD512_TRITON=1   (opt-in)
  GENESIS_G4_75_DRAFTER_PREFIX=draft_model.       (override prefix)
  GENESIS_G4_75_DRAFTER_HEAD_THRESHOLD=512        (head_size that
                                                   triggers reroute)
  GENESIS_G4_75_TARGET_BACKEND=TRITON_ATTN        (AttentionBackendEnum
                                                   name to use)

================================================================
ACCEPTANCE GATE
================================================================

  K=2 boot — server up.
  K=2 first prompt — no FlashAttn head_size error; no PN262 unbind
  error; first prompt returns tokens. PN262 trace shows drafter layers
  0..2 with HND shape (2, num_blocks, 16, 8, 256); drafter layer 3 with
  Triton NHD shape (num_blocks, 2, 32, 2, 512) (different impl, no
  PN262 fail-fast because we'll keep PN262_FAIL_FAST=0).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.spec_decode.g4_75_drafter_head512_triton")

GENESIS_G4_75_MARKER = (
    "Genesis G4_75 Route head_size=512 drafter layers to TRITON_ATTN "
    "(PN264 follow-up to G4_71/G4_74 layout fix)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_75_DRAFTER_HEAD512_TRITON"
_ENV_PREFIX = "GENESIS_G4_75_DRAFTER_PREFIX"
_ENV_HEAD_THRESHOLD = "GENESIS_G4_75_DRAFTER_HEAD_THRESHOLD"
_ENV_TARGET_BACKEND = "GENESIS_G4_75_TARGET_BACKEND"
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
    raw = os.environ.get(_ENV_HEAD_THRESHOLD, "512").strip()
    try:
        return int(raw)
    except ValueError:
        return 512


def _target_backend_name() -> str:
    return os.environ.get(_ENV_TARGET_BACKEND, "TRITON_ATTN").strip()


def apply() -> tuple[str, str]:
    """Wrap Attention.__init__ to route drafter head=threshold to Triton."""
    global _APPLIED, _ORIGINAL_INIT

    if not _env_enabled():
        return "skipped", (
            f"G4_75 disabled (set {_ENV_ENABLE}=1 to route drafter "
            "layers with head_size==threshold to TRITON_ATTN, bypassing "
            "FlashAttn's head_size<=256 cap on drafter full-attention layer)"
        )

    if _APPLIED:
        return "applied", "G4_75 already installed (idempotent)"

    log.warning("[G4_75] apply() entered — beginning import phase")

    try:
        from vllm.model_executor.layers.attention.attention import Attention
    except Exception as e:  # noqa: BLE001
        log.warning("[G4_75] SKIP: Attention not importable: %s", e)
        return "skipped", f"Attention not importable: {e!r}"

    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
    except Exception as e:  # noqa: BLE001
        log.warning("[G4_75] SKIP: AttentionBackendEnum not importable: %s", e)
        return "skipped", f"AttentionBackendEnum not importable: {e!r}"

    target_backend_name = _target_backend_name()
    try:
        target_backend_cls = AttentionBackendEnum[target_backend_name].get_class()
    except Exception as e:  # noqa: BLE001
        log.warning(
            "[G4_75] SKIP: target backend %r not resolvable: %s",
            target_backend_name, e,
        )
        return "skipped", (
            f"target backend {target_backend_name!r} not resolvable: {e!r}"
        )

    original = Attention.__init__
    if getattr(original, "_genesis_g4_75_wrapped", False):
        _APPLIED = True
        return "applied", "Attention.__init__ already wrapped by G4_75 (idempotent)"
    _ORIGINAL_INIT = original

    drafter_prefix = _drafter_prefix()
    head_threshold = _head_threshold()
    log.warning(
        "[G4_75] import phase OK — drafter_prefix=%r head_threshold=%d "
        "target_backend=%s (%s); about to wrap Attention.__init__",
        drafter_prefix, head_threshold, target_backend_name,
        target_backend_cls.__name__,
    )

    def _wrapped_init(self, *args, **kwargs):
        """Override drafter head_size==threshold layer backend to Triton.

        Attention.__init__ signature:
            (self, num_heads, head_size, scale, num_kv_heads=None, ...,
             prefix='', ..., attn_backend=None, ...)
        head_size is positional index 1 (after num_heads). Most callers
        use positional for num_heads/head_size/scale and keyword for
        prefix/attn_backend. Read both ways.
        """
        prefix = kwargs.get("prefix", "") or ""
        # head_size: positional index 1, or kwarg
        head_size = kwargs.get("head_size", None)
        if head_size is None and len(args) >= 2:
            head_size = args[1]
        is_drafter_target = (
            isinstance(prefix, str)
            and prefix.startswith(drafter_prefix)
            and isinstance(head_size, int)
            and head_size == head_threshold
        )

        if is_drafter_target:
            # Pre-init marker stamping (belt) so G4_74 sees it even if
            # original __init__ internally invokes get_kv_cache_spec
            # or similar before returning.
            try:
                self._genesis_g4_75_drafter_triton = True
                self._genesis_g4_75_target_backend = target_backend_name
            except Exception:
                pass

            kwargs["attn_backend"] = target_backend_cls
            _REROUTE_COUNT[0] += 1
            if _REROUTE_COUNT[0] <= 6:
                log.warning(
                    "[G4_75] drafter head_size==%d detected "
                    "(prefix=%r) — overriding attn_backend to %s "
                    "(was FlashAttn from G4_71). Triton supports "
                    "head_size>=32 and uses NHD layout natively. "
                    "G4_74 will skip this layer (marker set). (call #%d)",
                    head_size, prefix, target_backend_name,
                    _REROUTE_COUNT[0],
                )
            elif _REROUTE_COUNT[0] == 7:
                log.warning(
                    "[G4_75] further drafter-head-reroute logs suppressed (count > 6)"
                )

        result = original(self, *args, **kwargs)

        # Post-init re-stamp (suspenders).
        if is_drafter_target:
            try:
                self._genesis_g4_75_drafter_triton = True
                self._genesis_g4_75_target_backend = target_backend_name
            except Exception:
                pass

        return result

    _wrapped_init._genesis_g4_75_wrapped = True  # type: ignore[attr-defined]
    Attention.__init__ = _wrapped_init  # type: ignore[method-assign]
    _APPLIED = True

    log.warning(
        "[G4_75] INSTALLED: Attention.__init__ wrapped; drafter prefix "
        "%r layers with head_size==%d will be rerouted to %s backend.",
        drafter_prefix, head_threshold, target_backend_name,
    )
    return "applied", (
        f"G4_75 installed: drafter (prefix {drafter_prefix!r}) "
        f"head_size=={head_threshold} -> {target_backend_name}."
    )


def is_applied() -> bool:
    return _APPLIED


def reroute_count() -> int:
    return _REROUTE_COUNT[0]


def revert() -> bool:
    """Best-effort revert (test isolation only)."""
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
    "GENESIS_G4_75_MARKER",
    "apply",
    "is_applied",
    "reroute_count",
    "revert",
]
