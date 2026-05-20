# SPDX-License-Identifier: Apache-2.0
"""G4_60g — patch ``Attention.get_kv_cache_spec`` for per-layer TQ dispatch.

================================================================
PROBLEM
================================================================

In vllm pin ``0.20.2rc1.dev371+gbf610c2f5``,
``Attention.get_kv_cache_spec`` (file
``vllm/model_executor/layers/attention/attention.py``) dispatches in this
order::

    if self.sliding_window is not None:
        return SlidingWindowSpec(...)          # <-- WINS for SW layer
    elif self.kv_cache_dtype.startswith("turboquant_"):
        return TQFullAttentionSpec(...)
    else:
        return FullAttentionSpec(...)

The consequence on Gemma 4 with ``--kv-cache-dtype turboquant_*``:

  * Sliding layers (head_dim=256, ~50 of ~64) hit the FIRST branch and
    get a plain ``SlidingWindowSpec`` — vllm sizes their pages by the
    uncompressed ``head_size × dtype`` formula. TurboQuant compression
    is effectively disabled on the sliding tier.

  * Full layers (head_dim=512) hit the second branch and get
    ``TQFullAttentionSpec`` — compression applies there.

  * Mixed-spec KV cache groups (plain SlidingWindowSpec + TQFullAttention
    Spec) cannot share a unified page size; KVCacheManager either fails
    ``unify_kv_cache_spec_page_size`` or collapses both tiers down to
    the uncompressed size.

This is exactly the architectural problem upstream PR #42637 fixes.

================================================================
FIX
================================================================

Replace ``Attention.get_kv_cache_spec`` with a per-layer TQ-first
dispatcher (mirrors PR #42637 lines 580-633)::

    if self.kv_cache_dtype.startswith("turboquant_"):
        # TQ is the primary axis — pick TQ variant matching layer geometry
        if self.sliding_window is not None:
            return TQSlidingWindowSpec(...)    # <-- new (needs G4_60a)
        return TQFullAttentionSpec(...)
    elif self.sliding_window is not None:
        return SlidingWindowSpec(...)
    else:
        return FullAttentionSpec(...)

================================================================
DEPENDENCIES
================================================================

  * **G4_60a** MUST apply first (defines ``TQSlidingWindowSpec``). If
    G4_60a not applied, ``apply()`` returns ``skipped`` with explanation.

  * Recommended companion: **G4_60h** (turboquant/config.py overlay with
    ``slot_size_aligned`` property + skip-layer helpers). Without it,
    ``TurboQuantConfig.from_cache_dtype`` may not return
    ``slot_size_aligned`` field. G4_60g still works — page sizing falls
    through to the parent ``SlidingWindowSpec.real_page_size_bytes``
    when ``tq_slot_size == 0``.

================================================================
SCOPE
================================================================

Active only when ``GENESIS_ENABLE_G4_60G_TQ_DISPATCH=1``. Touches a
single classmethod on a single class. No other vllm code paths change.

For non-turboquant ``cache_dtype`` (auto, fp8, etc.), the wrapped
dispatcher behaves identically to upstream — only the ordering of
the existing branches changes, and the non-TQ branches are preserved
verbatim.

================================================================
RISK
================================================================

Branch-order swap could regress workloads where:
  1. Operator launched with ``--kv-cache-dtype turboquant_*``
  2. AND model has sliding-window layers
  3. AND production was relying on the dev371 fallback to plain
     ``SlidingWindowSpec`` (i.e. silently disabled TQ on SW tier).

If a user observes such a regression, disable via
``GENESIS_ENABLE_G4_60G_TQ_DISPATCH=0``. The Genesis G4_19/19b/19c
wrapper-strategy stack remains the production fallback (proven working
as of 2026-05-17 smoke test).

================================================================
REFERENCES
================================================================

  * Upstream PR: https://github.com/vllm-project/vllm/pull/42637
  * Upstream source (PR #42637 HEAD ``fdeb14981``):
    ``vllm/model_executor/layers/attention/attention.py`` lines 575-633.
  * Dev371 source:
    ``vllm/model_executor/layers/attention/attention.py`` lines 570-610
    (verified 2026-05-17 via docker exec).
  * Companion patches: G4_60a (TQSlidingWindowSpec), G4_60h (TQ config).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.gemma4.g4_60g_attention_dispatch")

GENESIS_G4_60G_MARKER = (
    "Genesis G4_60g Attention.get_kv_cache_spec TQ-first dispatch with "
    "TQSlidingWindowSpec path (PR #42637 cherry-pick)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_60G_TQ_DISPATCH"
_APPLIED = False
_ORIGINAL_GET_KV_CACHE_SPEC = None


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Patch Attention.get_kv_cache_spec for TQ-first per-layer dispatch.

    Returns:
        Tuple ``(status, message)``.
    """
    global _APPLIED, _ORIGINAL_GET_KV_CACHE_SPEC

    if not _env_enabled():
        return "skipped", (
            f"G4_60g disabled (set {_ENV_ENABLE}=1 to enable TQ-first "
            "Attention.get_kv_cache_spec dispatch — PR #42637 cherry-pick)"
        )

    if _APPLIED:
        return "applied", "G4_60g already installed (idempotent)"

    # Verify G4_60a is applied — otherwise TQSlidingWindowSpec import fails.
    try:
        from vllm.v1.kv_cache_interface import (  # noqa: F401
            TQSlidingWindowSpec as _,
        )
    except ImportError:
        return "skipped", (
            "G4_60a prerequisite not applied: TQSlidingWindowSpec not "
            "available on vllm.v1.kv_cache_interface. Enable "
            "GENESIS_ENABLE_G4_60A_TQ_SLIDING_SPEC=1 first."
        )

    try:
        from vllm.model_executor.layers.attention.attention import Attention
        # AttentionType lives in vllm.v1.attention.backend in dev371+
        try:
            from vllm.v1.attention.backend import AttentionType
        except ImportError:
            # Older pin compatibility
            from vllm.attention.backends.abstract import AttentionType
    except ImportError as e:
        return "skipped", (
            f"vllm.model_executor.layers.attention.attention not importable: {e}"
        )

    original = Attention.get_kv_cache_spec
    if getattr(original, "_genesis_g4_60g_wrapped", False):
        _APPLIED = True
        return "applied", (
            "Attention.get_kv_cache_spec already wrapped (idempotent)"
        )
    _ORIGINAL_GET_KV_CACHE_SPEC = original

    def _wrapped_get_kv_cache_spec(self, vllm_config):
        """TQ-first per-layer dispatcher (PR #42637 lines 575-633)."""
        # Late imports to keep cold-import surface minimal — these are
        # exercised at boot, not on every forward.
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            SlidingWindowSpec,
            TQFullAttentionSpec,
            TQSlidingWindowSpec,
        )

        # Block size may get updated after model loading; refresh.
        block_size = vllm_config.cache_config.block_size

        # vllm core invariant: enc-dec/encoder-only attention must not
        # call get_kv_cache_spec. Preserve the assertion.
        assert self.attn_type == AttentionType.DECODER, (
            "Genesis G4_60g: get_kv_cache_spec called on non-DECODER "
            f"attention layer ({self.attn_type=}); upstream invariant "
            "violated."
        )

        # Resolve KV quant mode once — used by non-TQ branches.
        try:
            from vllm.model_executor.layers.quantization.utils.quant_utils import (
                get_kv_quant_mode,
            )
        except ImportError:
            # Older pin path
            from vllm.model_executor.layers.attention.attention import (
                get_kv_quant_mode,
            )

        quant_mode = get_kv_quant_mode(self.kv_cache_dtype)

        # === TQ-first dispatch (PR #42637 ordering) ===
        if self.kv_cache_dtype.startswith("turboquant_"):
            from vllm.model_executor.layers.quantization.turboquant.config import (
                TurboQuantConfig,
            )

            tq_config = TurboQuantConfig.from_cache_dtype(
                self.kv_cache_dtype, self.head_size
            )
            # PR #42637 uses ``slot_size_aligned`` (post-merge field).
            # Older config exposes ``slot_size`` only — graceful fallback.
            tq_slot_size = getattr(
                tq_config, "slot_size_aligned", None
            )
            if tq_slot_size is None:
                tq_slot_size = getattr(tq_config, "slot_size", 0)

            if self.sliding_window is not None:
                return TQSlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=self.num_kv_heads,
                    head_size=self.head_size,
                    head_size_v=self.head_size_v,
                    dtype=self.kv_cache_torch_dtype,
                    tq_slot_size=tq_slot_size,
                    sliding_window=self.sliding_window,
                )
            return TQFullAttentionSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                head_size_v=self.head_size_v,
                dtype=self.kv_cache_torch_dtype,
                tq_slot_size=tq_slot_size,
            )

        # === Non-TQ branches (preserved verbatim from dev371) ===
        if self.sliding_window is not None:
            assert not vllm_config.model_config.use_mla, (
                "MLA is not supported for slidingwindow"
            )
            return SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                head_size_v=self.head_size_v,
                dtype=self.kv_cache_torch_dtype,
                kv_quant_mode=quant_mode,
                sliding_window=self.sliding_window,
            )
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=quant_mode,
        )

    _wrapped_get_kv_cache_spec._genesis_g4_60g_wrapped = True  # type: ignore[attr-defined]
    Attention.get_kv_cache_spec = _wrapped_get_kv_cache_spec  # type: ignore[method-assign]

    _APPLIED = True
    log.info(
        "[G4_60g] Attention.get_kv_cache_spec wrapped: TQ-first dispatch "
        "with TQSlidingWindowSpec branch active."
    )
    return "applied", (
        "G4_60g installed: Attention.get_kv_cache_spec now dispatches "
        "turboquant_* layers to TQ specs (sliding/full) before the SW "
        "branch."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    """Best-effort revert (test isolation only)."""
    global _APPLIED, _ORIGINAL_GET_KV_CACHE_SPEC
    if not _APPLIED or _ORIGINAL_GET_KV_CACHE_SPEC is None:
        return False
    try:
        from vllm.model_executor.layers.attention.attention import Attention

        Attention.get_kv_cache_spec = _ORIGINAL_GET_KV_CACHE_SPEC  # type: ignore[method-assign]
    except Exception:  # noqa: BLE001
        return False
    _APPLIED = False
    _ORIGINAL_GET_KV_CACHE_SPEC = None
    return True


__all__ = [
    "GENESIS_G4_60G_MARKER",
    "apply",
    "is_applied",
    "revert",
]
