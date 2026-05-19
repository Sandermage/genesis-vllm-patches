# SPDX-License-Identifier: Apache-2.0
"""G4_72 — Force native FullAttentionSpec/SlidingWindowSpec for Gemma 4 MTP drafter.

================================================================
PROBLEM (PN261-D)
================================================================

G4_71 routes drafter Attention layers to FlashAttn impl during
``Attention.__init__``. However G4_60g's ``get_kv_cache_spec`` wrap
still treats drafter layers as TurboQuant because their
``self.kv_cache_dtype == "turboquant_4bit_nc"`` (inherited from the
global ``--kv-cache-dtype turboquant_4bit_nc`` engine flag — drafter
is not listed in ``cache_config.kv_cache_dtype_skip_layers``).

Symptom on K=2 after G4_71 alone:

  * PN261-A no longer fires (good — drafter never builds TQ impl).
  * cudaErrorIllegalAddress disappears (good).
  * NEW error at FlashAttention backend's ``forward``::

      ValueError: too many values to unpack (expected 2)
        key_cache, value_cache = kv_cache.unbind(0)
        File "vllm/v1/attention/backends/flash_attn.py:744"

    drafter's allocated cache shape is
    ``(num_blocks, 2, block_size, num_kv_heads, head_dim)`` —
    the leading-axis-2 layout that ``TQFullAttentionSpec`` /
    ``TQSlidingWindowSpec`` configures. FlashAttn expects
    ``(2, num_blocks, block_size, num_kv_heads, head_dim)``.

Root cause: spec/impl mismatch. G4_71 fixed the impl but not the spec.
The KV cache allocator builds the physical tensor from the spec, so
the spec must also be native for drafter.

================================================================
FIX
================================================================

Wrap ``Attention.get_kv_cache_spec`` AFTER G4_60g (or on a vanilla
Attention if G4_60g is disabled). If the Attention instance carries
the marker ``_genesis_g4_71_is_drafter == True`` (set by G4_71's
``__init__`` wrap), return a native ``FullAttentionSpec`` or
``SlidingWindowSpec`` regardless of the TQ-prefixed dtype.

The native spec uses:

  * ``dtype = vllm_config.model_config.dtype`` (bf16 for Gemma 4) —
    NOT ``self.kv_cache_torch_dtype`` (which is ``torch.uint8`` because
    Attention was init under TQ dtype before G4_71 substituted impl).
  * ``kv_quant_mode = get_kv_quant_mode("auto")`` — explicitly no-quant.
  * Sliding window passthrough when ``self.sliding_window is not None``.

For non-drafter layers, delegates to ``_ORIGINAL_GET_KV_CACHE_SPEC``
(which is G4_60g's wrap if applied, else the vanilla upstream method).

================================================================
ORDERING
================================================================

G4_72 must wrap AFTER G4_60g, so calling ``_ORIGINAL_GET_KV_CACHE_SPEC``
on a non-drafter layer routes through G4_60g's TQ-first dispatch.
The plugin apply-all sweep runs sequentially per env flag, so as long
as the ``apply()`` order in ``vllm/sndr_core/__init__.py`` lists G4_60g
before G4_72, this invariant holds.

If G4_72 is enabled but G4_60g is not, fall through to whatever vanilla
``Attention.get_kv_cache_spec`` does for non-drafter — drafter still
gets native specs.

================================================================
INTERACTION
================================================================

  * G4_31 — preserves turboquant_* dtype against AWQ overrides for
    target; unaffected (we override post-init).
  * G4_60g — TQ-first dispatch for non-drafter; preserved as inner wrap.
  * G4_69 — target skip-listed layers; unaffected.
  * G4_71 — drafter FlashAttn impl; sets marker we rely on.
  * PN259c — split allocator now sees uniform native spec for drafter
    group; allocator must not cross-alias drafter native cache with
    target TQ cache (PN259c invariant: no-cross-layout aliasing).
  * PN261-A assert — still in place as safety belt; should not fire
    once both G4_71 and G4_72 are active.

================================================================
ENV FLAG
================================================================

  GENESIS_ENABLE_G4_72_DRAFTER_NATIVE_SPEC=1   (opt-in)

When unset: G4_71-only behavior (impl native but spec TQ → axis-order
mismatch crash on first drafter forward).

================================================================
ACCEPTANCE GATES
================================================================

Per user 2026-05-19 PN261-D directive:

  Gate 0: MTP OFF + G4_72 — no regression
  Gate 1: MTP K=2 + G4_71 + G4_72 — no crash; no PN261-A assert;
          no flash_attn.py unpack error.
  Gate 2: MTP K=4 + G4_71 + G4_72 — no crash.
  Gate 3: PN248 acceptance — accepted_per_req > 0 (if 0, return to
          H8a skip-list / KV-sharing investigation).

Expected drafter cache shape after this patch (trace via PN260
logs added to allocator if needed):

  draft_model.layers.{0..3}: (2, num_blocks, block_size, num_kv_heads, head_dim)

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.gemma4.g4_72_drafter_native_kv_cache_spec")

GENESIS_G4_72_MARKER = (
    "Genesis G4_72 Force native FullAttentionSpec/SlidingWindowSpec for "
    "Gemma 4 MTP drafter layers (companion to G4_71 impl reroute)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_72_DRAFTER_NATIVE_SPEC"
_APPLIED = False
_ORIGINAL_GET_KV_CACHE_SPEC = None
_DRAFTER_SPEC_LOG_COUNT = [0]


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Wrap Attention.get_kv_cache_spec to force native spec for drafter."""
    global _APPLIED, _ORIGINAL_GET_KV_CACHE_SPEC

    if not _env_enabled():
        return "skipped", (
            f"G4_72 disabled (set {_ENV_ENABLE}=1 to force drafter "
            "Attention layers to receive native FullAttentionSpec / "
            "SlidingWindowSpec instead of TQ-flavored spec)"
        )

    if _APPLIED:
        return "applied", "G4_72 already installed (idempotent)"

    try:
        from vllm.model_executor.layers.attention.attention import Attention
    except ImportError as e:
        return "skipped", (
            f"vllm.model_executor.layers.attention.attention not importable: {e}"
        )

    original = Attention.get_kv_cache_spec
    if getattr(original, "_genesis_g4_72_wrapped", False):
        _APPLIED = True
        return "applied", "Attention.get_kv_cache_spec already wrapped (idempotent)"
    _ORIGINAL_GET_KV_CACHE_SPEC = original

    def _wrapped_get_kv_cache_spec(self, vllm_config):
        """Return native spec for drafter, delegate otherwise."""
        is_drafter = getattr(self, "_genesis_g4_71_is_drafter", False)
        if not is_drafter:
            return original(self, vllm_config)

        # Drafter path — synthesize a native spec independent of
        # self.kv_cache_dtype (which is still TQ-prefixed because the
        # global engine flag set it that way).
        import torch

        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            SlidingWindowSpec,
        )

        try:
            from vllm.model_executor.layers.quantization.utils.quant_utils import (
                get_kv_quant_mode,
            )
        except ImportError:
            # Older pin path; if neither is importable, fall back to None.
            try:
                from vllm.model_executor.layers.attention.attention import (
                    get_kv_quant_mode,
                )
            except ImportError:  # pragma: no cover
                def get_kv_quant_mode(_dt):  # type: ignore[no-redef]
                    return None

        block_size = vllm_config.cache_config.block_size

        # Model dtype is the correct native cache dtype for drafter.
        # self.kv_cache_torch_dtype was set under TQ init → torch.uint8,
        # which is wrong for FlashAttn (G4_71 forces FlashAttn impl).
        native_dtype = getattr(vllm_config.model_config, "dtype", None)
        if isinstance(native_dtype, str):
            native_dtype = getattr(torch, native_dtype, torch.bfloat16)
        if native_dtype is None:
            native_dtype = torch.bfloat16

        # Force no-quantization mode for drafter cache.
        quant_mode = get_kv_quant_mode("auto")

        prefix = getattr(self, "_genesis_g4_71_drafter_prefix", "<unknown>")
        _DRAFTER_SPEC_LOG_COUNT[0] += 1
        if _DRAFTER_SPEC_LOG_COUNT[0] <= 12:
            log.warning(
                "[G4_72] drafter get_kv_cache_spec rerouted to native "
                "(prefix=%r, sliding_window=%s, head_size=%s, "
                "head_size_v=%s, num_kv_heads=%s, dtype=%s, "
                "kv_cache_dtype=%r) (call #%d)",
                prefix,
                self.sliding_window,
                self.head_size,
                self.head_size_v,
                self.num_kv_heads,
                native_dtype,
                self.kv_cache_dtype,
                _DRAFTER_SPEC_LOG_COUNT[0],
            )
        elif _DRAFTER_SPEC_LOG_COUNT[0] == 13:
            log.warning(
                "[G4_72] further drafter spec-reroute logs suppressed "
                "(count > 12)"
            )

        if self.sliding_window is not None:
            return SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                head_size_v=self.head_size_v,
                dtype=native_dtype,
                kv_quant_mode=quant_mode,
                sliding_window=self.sliding_window,
            )
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            dtype=native_dtype,
            kv_quant_mode=quant_mode,
        )

    _wrapped_get_kv_cache_spec._genesis_g4_72_wrapped = True  # type: ignore[attr-defined]
    Attention.get_kv_cache_spec = _wrapped_get_kv_cache_spec  # type: ignore[method-assign]
    _APPLIED = True

    log.info(
        "[G4_72] installed: Attention.get_kv_cache_spec now returns native "
        "spec when self._genesis_g4_71_is_drafter is True."
    )
    return "applied", (
        "G4_72 installed: drafter Attention layers (G4_71 marker) now "
        "receive native FullAttentionSpec/SlidingWindowSpec; non-drafter "
        "layers continue through G4_60g (or vanilla) dispatch."
    )


def is_applied() -> bool:
    return _APPLIED


def drafter_spec_reroute_count() -> int:
    return _DRAFTER_SPEC_LOG_COUNT[0]


def revert() -> bool:
    """Best-effort revert (test isolation only)."""
    global _APPLIED, _ORIGINAL_GET_KV_CACHE_SPEC
    if not _APPLIED or _ORIGINAL_GET_KV_CACHE_SPEC is None:
        return False
    try:
        from vllm.model_executor.layers.attention.attention import Attention

        Attention.get_kv_cache_spec = _ORIGINAL_GET_KV_CACHE_SPEC  # type: ignore[method-assign]
    except ImportError:
        return False
    _APPLIED = False
    _ORIGINAL_GET_KV_CACHE_SPEC = None
    return True


__all__ = [
    "GENESIS_G4_72_MARKER",
    "apply",
    "is_applied",
    "drafter_spec_reroute_count",
    "revert",
]
