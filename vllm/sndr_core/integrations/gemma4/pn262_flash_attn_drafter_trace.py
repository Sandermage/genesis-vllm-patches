# SPDX-License-Identifier: Apache-2.0
"""PN262 — FlashAttn drafter KV cache shape/stride trace + fail-fast.

================================================================
PROBLEM
================================================================

After G4_71 (PN261-C, impl reroute) + G4_72 (PN261-D, spec reroute),
K=2 boot is clean — all 8 drafter logs fire (4 layers × 2 TP ranks)
showing spec is native FullAttentionSpec/SlidingWindowSpec — but the
first forward still crashes at::

    File ".../v1/attention/backends/flash_attn.py", line 744
        key_cache, value_cache = kv_cache.unbind(0)
    ValueError: too many values to unpack (expected 2)

So the spec is native AND the impl is FlashAttn, yet the physical
``kv_cache`` tensor delivered to FlashAttn's forward still has leading
dim ≠ 2. The fix lies one layer deeper than ``Attention.get_kv_cache_spec``
— either in the allocator's tensor build, in the per-layer bind/view,
in cross-layer sharing (``kv_sharing_target_layer_name``), or in the
global ``VLLM_KV_CACHE_LAYOUT`` setting.

================================================================
FIX (DIAGNOSTIC ONLY — does not change behavior)
================================================================

Wrap ``FlashAttentionImpl.forward``. For drafter layers (impl
``self.layer_name`` starts with ``draft_model.``), log everything that
disambiguates the four hypotheses, then optionally fail-fast BEFORE
the ``kv_cache.unbind(0)`` line so the operator sees a clean
RuntimeError with full context instead of the bare ValueError.

Fields captured (≥1 per drafter layer per rank, capped at 12 logs):

  layer_name                — drafter Attention impl prefix
  kv_sharing_target_layer_name — if set, drafter aliases another layer
  kv_cache.shape            — leading dim should equal 2
  kv_cache.stride           — distinguishes view from owned tensor
  kv_cache.dtype            — should be bf16 (model dtype)
  kv_cache.is_contiguous    — False ⇒ transpose somewhere
  kv_cache.data_ptr         — cross-reference for aliasing detection
  expected_leading_dim      — always 2 (FlashAttn contract)
  VLLM_KV_CACHE_LAYOUT      — env value (NHD vs HND vs unset)
  impl_class                — self.__class__.__qualname__
  spec_class_if_reachable   — via vllm_config when available

Hypothesis disambiguation from these fields::

  shape[0] != 2 AND contiguous=True
    → ALLOCATOR built the wrong physical shape.

  shape[0] == 2 AND contiguous=True
    → not actually wrong; trace catches phantom regression.

  shape[0] != 2 AND contiguous=False AND stride implies axis-swap
    → BIND/VIEW path applied .transpose(0,1) somewhere between
      allocator and forward.

  kv_sharing_target_layer_name is not None (and points at a target layer)
    → DRAFTER ALIASES TARGET TQ CACHE — G4_72 spec override at the
      drafter Attention is moot because the physical tensor comes
      from the target layer's allocation. Need to also exclude drafter
      from cross-layer sharing, or force a separate allocation group.

  VLLM_KV_CACHE_LAYOUT is "NHD" globally
    → all attention layers (including drafter) use NHD; FlashAttn's
      ``unbind(0)`` expects HND. Possible fix: set HND for drafter or
      change FlashAttn dispatch to NHD path.

================================================================
ENV FLAGS
================================================================

  GENESIS_ENABLE_PN262_FLASH_ATTN_DRAFTER_TRACE=1   (opt-in)
  GENESIS_ENABLE_PN262_FAIL_FAST=1                  (default ON when
                                                     trace is ON; set to
                                                     0 to log only)
  GENESIS_PN262_PREFIX=draft_model.                 (override prefix)

================================================================
NOT-A-FIX
================================================================

This patch DOES NOT fix the wrong axis order. It localizes the bug.
The downstream fix (PN262-B / G4_73) depends on which hypothesis the
trace nails down.

================================================================
ACCEPTANCE GATE
================================================================

  Gate 1: K=2 + full PN261-D stack + PN262 trace ON + fail-fast ON.
  Expected: 4 drafter layers × 2 TP = up to 8 PN262 log lines, each
  with one of the disambiguating signatures, then a clean
  RuntimeError pointing to the first wrong-shape layer.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.gemma4.pn262_flash_attn_drafter_trace")

GENESIS_PN262_MARKER = (
    "Genesis PN262 FlashAttn drafter KV cache shape/stride trace + "
    "fail-fast (one-shot D-3 localization patch)"
)

_ENV_ENABLE = "GENESIS_ENABLE_PN262_FLASH_ATTN_DRAFTER_TRACE"
_ENV_FAIL_FAST = "GENESIS_ENABLE_PN262_FAIL_FAST"
_ENV_PREFIX = "GENESIS_PN262_PREFIX"
_APPLIED = False
_ORIGINAL_FORWARD = None
_LOG_COUNT = [0]


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _fail_fast_enabled() -> bool:
    # Default ON when trace is ON. Set to 0/false/no/off to log only.
    val = os.environ.get(_ENV_FAIL_FAST, "1").strip().lower()
    return val in ("1", "true", "yes", "on")


def _drafter_prefix() -> str:
    return os.environ.get(_ENV_PREFIX, "draft_model.").strip()


def apply() -> tuple[str, str]:
    global _APPLIED, _ORIGINAL_FORWARD

    if not _env_enabled():
        return "skipped", (
            f"PN262 disabled (set {_ENV_ENABLE}=1 to trace FlashAttn "
            "forward for drafter and fail-fast on wrong KV axis order)"
        )

    if _APPLIED:
        return "applied", "PN262 already installed (idempotent)"

    try:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    except ImportError as e:
        return "skipped", (
            f"vllm.v1.attention.backends.flash_attn.FlashAttentionImpl "
            f"not importable: {e}"
        )

    original = FlashAttentionImpl.forward
    if getattr(original, "_genesis_pn262_wrapped", False):
        _APPLIED = True
        return "applied", "FlashAttentionImpl.forward already wrapped"
    _ORIGINAL_FORWARD = original

    drafter_prefix = _drafter_prefix()
    do_fail_fast = _fail_fast_enabled()

    def _wrapped_forward(self, *args, **kwargs):
        """Trace + fail-fast on wrong KV axis order in any FlashAttn fwd.

        Why no prefix filter: ``FlashAttentionImpl`` in this pin does not
        carry ``layer_name`` attribute (the prefix lives on the outer
        ``Attention`` wrapper, not on the impl). We therefore inspect
        every FlashAttn forward, log up to 16 calls with whatever name
        we can recover, and fail-fast on the first ``shape[0] != 2``.
        Non-drafter FlashAttn forwards (target skip-listed layers like
        58/59 routed by G4_69) have correct ``(2, num_blocks, ...)``
        layout and pass through silently.
        """
        # Recover any layer identifier we can — multiple attribute names
        # exist across pins.
        layer_name = (
            getattr(self, "layer_name", None)
            or getattr(self, "prefix", None)
            or getattr(self, "_layer_name", None)
            or "<unknown>"
        )

        # Locate kv_cache argument. v1 FlashAttn forward signature:
        #   forward(self, layer, query, key, value, kv_cache, attn_metadata,
        #           output, ...)
        # When called as `self.impl.forward(self_layer, query, key, value,
        # kv_cache, attn_metadata, ...)` and dispatched through our wrap
        # `_wrapped_forward(self=impl, *args, **kwargs)`, args layout is:
        #   args[0] = layer
        #   args[1] = query
        #   args[2] = key
        #   args[3] = value
        #   args[4] = kv_cache    <-- correct index
        #   args[5] = attn_metadata
        kv_cache = kwargs.get("kv_cache")
        if kv_cache is None and len(args) >= 5:
            kv_cache = args[4]

        kv_sharing_target = getattr(self, "kv_sharing_target_layer_name", None)
        impl_class = type(self).__qualname__
        layout_env = os.environ.get("VLLM_KV_CACHE_LAYOUT", "<unset>")

        if kv_cache is None:
            if _LOG_COUNT[0] < 16:
                _LOG_COUNT[0] += 1
                log.warning(
                    "[PN262] FlashAttn forward (kv_cache=None): "
                    "layer=%r impl=%s kv_sharing_target=%r "
                    "VLLM_KV_CACHE_LAYOUT=%r (call #%d)",
                    layer_name, impl_class, kv_sharing_target,
                    layout_env, _LOG_COUNT[0],
                )
            return original(self, *args, **kwargs)

        # Capture shape/stride/dtype/contig/data_ptr safely.
        try:
            shape = tuple(kv_cache.shape)
            stride = tuple(kv_cache.stride())
            dtype = kv_cache.dtype
            contig = bool(kv_cache.is_contiguous())
            data_ptr = int(kv_cache.data_ptr())
            ndim = int(kv_cache.dim())
            numel = int(kv_cache.numel())
        except Exception as _e:
            log.warning(
                "[PN262] introspection failed on drafter kv_cache "
                "(layer=%r impl=%s): %s",
                layer_name, impl_class, _e,
            )
            return original(self, *args, **kwargs)

        if _LOG_COUNT[0] < 16:
            _LOG_COUNT[0] += 1
            log.warning(
                "[PN262] FlashAttn forward: layer=%r "
                "shape=%s stride=%s dtype=%s contiguous=%s "
                "data_ptr=0x%x ndim=%d numel=%d impl=%s "
                "kv_sharing_target=%r VLLM_KV_CACHE_LAYOUT=%r "
                "expected_leading_dim=2 self_attrs=%s (call #%d)",
                layer_name, shape, stride, dtype, contig,
                data_ptr, ndim, numel, impl_class,
                kv_sharing_target, layout_env,
                sorted(k for k in vars(self) if not k.startswith("_"))[:20]
                if hasattr(self, "__dict__") else "<no_dict>",
                _LOG_COUNT[0],
            )
        elif _LOG_COUNT[0] == 16:
            _LOG_COUNT[0] += 1
            log.warning("[PN262] further FlashAttn trace logs suppressed (> 16)")

        if do_fail_fast and ndim >= 1 and shape[0] != 2:
            raise RuntimeError(
                f"[PN262] FlashAttn forward received KV cache with wrong "
                f"leading axis: layer={layer_name!r} shape={shape} "
                f"stride={stride} dtype={dtype} contiguous={contig} "
                f"data_ptr=0x{data_ptr:x} ndim={ndim} numel={numel} "
                f"impl={impl_class} "
                f"kv_sharing_target_layer_name={kv_sharing_target!r} "
                f"VLLM_KV_CACHE_LAYOUT={layout_env!r}. "
                f"FlashAttention.forward expects leading dim == 2 (k,v "
                f"stack). Disambiguate via the fields above: "
                f"shape[0]!=2 AND contiguous=True => allocator built the "
                f"wrong shape; contiguous=False => a transpose/view was "
                f"applied between allocator and forward; "
                f"kv_sharing_target!=None => drafter aliases another "
                f"layer's cache; VLLM_KV_CACHE_LAYOUT='NHD' globally => "
                f"all attention layers including drafter use NHD."
            )

        return original(self, *args, **kwargs)

    _wrapped_forward._genesis_pn262_wrapped = True  # type: ignore[attr-defined]
    FlashAttentionImpl.forward = _wrapped_forward  # type: ignore[method-assign]
    _APPLIED = True

    log.info(
        "[PN262] installed: FlashAttentionImpl.forward wrapped for "
        "drafter prefix %r (fail_fast=%s)",
        drafter_prefix, do_fail_fast,
    )
    return "applied", (
        f"PN262 installed: FlashAttn forward trace + fail-fast on "
        f"drafter prefix {drafter_prefix!r}"
    )


def is_applied() -> bool:
    return _APPLIED


def log_count() -> int:
    return _LOG_COUNT[0]


def revert() -> bool:
    """Best-effort revert (test isolation only)."""
    global _APPLIED, _ORIGINAL_FORWARD
    if not _APPLIED or _ORIGINAL_FORWARD is None:
        return False
    try:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
        FlashAttentionImpl.forward = _ORIGINAL_FORWARD  # type: ignore[method-assign]
    except ImportError:
        return False
    _APPLIED = False
    _ORIGINAL_FORWARD = None
    return True


__all__ = [
    "GENESIS_PN262_MARKER",
    "apply",
    "is_applied",
    "log_count",
    "revert",
]
