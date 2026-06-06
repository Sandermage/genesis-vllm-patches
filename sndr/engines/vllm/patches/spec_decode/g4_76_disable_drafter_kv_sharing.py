# SPDX-License-Identifier: Apache-2.0
"""G4_76 — Disable Gemma4 drafter kv_sharing for independent drafter cache.

================================================================
PROBLEM (PN265)
================================================================

After G4_71/G4_72/G4_73/G4_74-cap/G4_75 unblocked K=2 boot and FIRST
prompt returned tokens, multi-prompt H8-0 sanity probe revealed:

  * Short prompts: gibberish output + 額 token loop
  * Long prompts (14 tokens): CUDA error: an illegal memory access
    was encountered

Root cause is architectural inconsistency:

  1. Gemma4Proposer._setup_gemma4_kv_sharing (gemma4.py:328) sets
     ``attn.kv_sharing_target_layer_name = "model.layers.{N}.self_attn.attn"``
     on every drafter Attention. This tells vllm "drafter shares
     target's KV cache & block_table & slot_mapping".

  2. G4_74 broke the physical alias (transpose+contiguous gave drafter
     an independent HND tensor). G4_74 cap=256 sized that tensor at 256
     blocks. So drafter's cache is small and independent.

  3. BUT ``kv_sharing_target_layer_name`` is still set, so the kv-cache
     manager uses target's slot_mapping for drafter writes. Target's
     block indices go up to 24987 (full target budget) — drafter's
     cache has only 256 entries → drafter write to block 299 (in a
     14-token prompt) goes out-of-bounds → CUDA illegal memory access.

The state is contradictory: physical cache says "drafter is independent"
but the kv_sharing wiring says "drafter is aliased". Either both, or
neither. We pick neither.

================================================================
FIX
================================================================

Wrap ``Gemma4Proposer._setup_gemma4_kv_sharing`` and make it a no-op.
Drafter Attention layers then keep ``kv_sharing_target_layer_name``
at its default (None). vllm's standard kv_cache flow treats them as
fully independent attention layers:

  * Drafter has its own kv_cache_groups entries (via G4_72's native spec
    + G4_71's FlashAttn impl).
  * Drafter has its own block_table allocated from the kv_cache_manager.
  * Drafter's slot_mapping references drafter's own block indices.
  * Drafter writes stay inside drafter's cache → no OOB.

Trade-off: drafter is fully independent. It will have a COLD kv_cache
at request start (no inherited target context). Acceptance will be
0% until G4_77 warm-up is added (run drafter forward over prompt
before MTP propose).

After G4_76 the G4_74 cap can be relaxed or removed — drafter's num_blocks
comes from the kv_cache_manager's budget split. We keep G4_74 in place
as a layout safeguard (it still transposes NHD→HND if needed), but
the cap (GENESIS_G4_74_DRAFTER_MAX_BLOCKS) is no longer required and
can be left at 0 (no cap).

================================================================
ENV FLAGS
================================================================

  GENESIS_ENABLE_G4_76_DISABLE_DRAFTER_KV_SHARING=1   (opt-in)

================================================================
ACCEPTANCE GATE
================================================================

  Gate 1 — K=2 boot + first short prompt: server up, no CUDA illegal
    access.
  Gate 2 — K=2 long prompt (14+ tokens): no OOB, no CUDA illegal
    access. PN262 shows drafter shape (2, num_blocks_drafter, ...)
    HND (G4_74 still active).
  Gate 3 — drafter has cold kv_cache; output likely gibberish AND
    acceptance still 0% — this is EXPECTED for G4_76 alone.
    G4_77 warm-up restores quality.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.spec_decode.g4_76_disable_drafter_kv_sharing")

GENESIS_G4_76_MARKER = (
    "Genesis G4_76 Disable Gemma4Proposer._setup_gemma4_kv_sharing — "
    "drafter becomes fully-independent attention layer (PN265 fix)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_76_DISABLE_DRAFTER_KV_SHARING"
_APPLIED = False
_ORIGINAL_SETUP = None
_NOOP_COUNT = [0]
_BACKEND_COERCE_COUNT = [0]
_ORIGINAL_FA_GET_SHAPE = None
_ORIGINAL_TRITON_GET_SHAPE = None
_ORIGINAL_ATTENTION_INIT = None
_ATTN_DTYPE_OVERRIDE_COUNT = [0]


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Wrap Gemma4Proposer._setup_gemma4_kv_sharing to be a no-op."""
    global _APPLIED, _ORIGINAL_SETUP

    if not _env_enabled():
        return "skipped", (
            f"G4_76 disabled (set {_ENV_ENABLE}=1 to disable "
            "Gemma4Proposer._setup_gemma4_kv_sharing — drafter "
            "becomes fully independent attention with its own "
            "kv_cache, block_table, and slot_mapping)"
        )

    if _APPLIED:
        return "applied", "G4_76 already installed (idempotent)"

    log.warning("[G4_76] apply() entered — beginning import phase")

    try:
        from vllm.v1.spec_decode.gemma4 import Gemma4Proposer
    except Exception as e:  # noqa: BLE001
        log.warning("[G4_76] SKIP: Gemma4Proposer not importable: %s", e)
        return "skipped", f"Gemma4Proposer not importable: {e!r}"

    if not hasattr(Gemma4Proposer, "_setup_gemma4_kv_sharing"):
        log.warning(
            "[G4_76] SKIP: Gemma4Proposer._setup_gemma4_kv_sharing missing "
            "on this pin — gemma4 kv_sharing already disabled or method renamed"
        )
        return "skipped", "Gemma4Proposer._setup_gemma4_kv_sharing missing"

    original = Gemma4Proposer._setup_gemma4_kv_sharing
    if getattr(original, "_genesis_g4_76_wrapped", False):
        _APPLIED = True
        return "applied", "Gemma4Proposer._setup_gemma4_kv_sharing already wrapped"
    _ORIGINAL_SETUP = original

    def _wrapped_setup(self, target_attn_layer_names):
        """No-op replacement for _setup_gemma4_kv_sharing.

        Standard Gemma4 MTP wires each drafter layer's
        kv_sharing_target_layer_name to a target layer (so drafter shares
        target's KV/block_table). G4_76 disables this wiring entirely:
        drafter keeps kv_sharing_target_layer_name=None and is treated
        as an independent attention layer downstream.
        """
        _NOOP_COUNT[0] += 1
        if _NOOP_COUNT[0] <= 4:
            log.warning(
                "[G4_76] _setup_gemma4_kv_sharing no-op (called with "
                "%d target_attn_layer_names; drafter layers will NOT "
                "have kv_sharing_target_layer_name set, becoming fully "
                "independent). (call #%d)",
                len(target_attn_layer_names)
                if target_attn_layer_names is not None else -1,
                _NOOP_COUNT[0],
            )
        elif _NOOP_COUNT[0] == 5:
            log.warning("[G4_76] further no-op logs suppressed (> 4)")
        return None

    _wrapped_setup._genesis_g4_76_wrapped = True  # type: ignore[attr-defined]
    Gemma4Proposer._setup_gemma4_kv_sharing = _wrapped_setup  # type: ignore[method-assign]

    # --- Companion fix (G4_76b): coerce cache_dtype_str for native backends ---
    #
    # After G4_76 disables kv_sharing, drafter is in its own kv_cache_group.
    # When _reshape_kv_cache_tensors runs, it calls:
    #   attn_backend.get_kv_cache_shape(..., cache_dtype_str=
    #       self.cache_config.cache_dtype)
    # The global cache_config.cache_dtype is "turboquant_4bit_nc" (set by
    # --kv-cache-dtype). FlashAttn/Triton backends don't understand the TQ
    # string and raise:
    #   RuntimeError: Unsupported fp8 kv cache data type: turboquant_4bit_nc
    #
    # Drafter's per-layer spec is native (G4_72 → bf16), so the TQ string
    # is meaningless for FA/Triton get_kv_cache_shape. Coerce it to "auto"
    # so the backend uses the spec's native dtype.
    #
    # We coerce only on FlashAttentionBackend and TritonAttentionBackend —
    # TurboQuant backend keeps its TQ string for the target layers.
    global _ORIGINAL_FA_GET_SHAPE, _ORIGINAL_TRITON_GET_SHAPE
    try:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
        if not getattr(FlashAttentionBackend.get_kv_cache_shape,
                       "_genesis_g4_76_wrapped", False):
            _ORIGINAL_FA_GET_SHAPE = FlashAttentionBackend.get_kv_cache_shape

            def _fa_get_shape(
                num_blocks, block_size, num_kv_heads, head_size,
                *, cache_dtype_str="auto", **kwargs,
            ):
                if (isinstance(cache_dtype_str, str)
                        and cache_dtype_str.startswith("turboquant_")):
                    _BACKEND_COERCE_COUNT[0] += 1
                    if _BACKEND_COERCE_COUNT[0] <= 4:
                        log.warning(
                            "[G4_76b] FlashAttentionBackend.get_kv_cache_shape "
                            "called with cache_dtype_str=%r; coercing to "
                            "'auto' (drafter native cache; TQ string is "
                            "meaningless for FlashAttn). (call #%d)",
                            cache_dtype_str, _BACKEND_COERCE_COUNT[0],
                        )
                    cache_dtype_str = "auto"
                return _ORIGINAL_FA_GET_SHAPE(
                    num_blocks, block_size, num_kv_heads, head_size,
                    cache_dtype_str=cache_dtype_str, **kwargs,
                )

            _fa_get_shape._genesis_g4_76_wrapped = True  # type: ignore[attr-defined]
            FlashAttentionBackend.get_kv_cache_shape = staticmethod(
                _fa_get_shape
            )
    except Exception as e:  # noqa: BLE001
        log.warning(
            "[G4_76b] FlashAttentionBackend.get_kv_cache_shape wrap failed: %s",
            e,
        )

    try:
        from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
        if not getattr(TritonAttentionBackend.get_kv_cache_shape,
                       "_genesis_g4_76_wrapped", False):
            _ORIGINAL_TRITON_GET_SHAPE = TritonAttentionBackend.get_kv_cache_shape

            def _triton_get_shape(
                num_blocks, block_size, num_kv_heads, head_size,
                *, cache_dtype_str="auto", **kwargs,
            ):
                if (isinstance(cache_dtype_str, str)
                        and cache_dtype_str.startswith("turboquant_")):
                    _BACKEND_COERCE_COUNT[0] += 1
                    if _BACKEND_COERCE_COUNT[0] <= 4:
                        log.warning(
                            "[G4_76b] TritonAttentionBackend.get_kv_cache_shape "
                            "called with cache_dtype_str=%r; coercing to "
                            "'auto' (drafter native cache; TQ string is "
                            "meaningless for Triton). (call #%d)",
                            cache_dtype_str, _BACKEND_COERCE_COUNT[0],
                        )
                    cache_dtype_str = "auto"
                return _ORIGINAL_TRITON_GET_SHAPE(
                    num_blocks, block_size, num_kv_heads, head_size,
                    cache_dtype_str=cache_dtype_str, **kwargs,
                )

            _triton_get_shape._genesis_g4_76_wrapped = True  # type: ignore[attr-defined]
            TritonAttentionBackend.get_kv_cache_shape = staticmethod(
                _triton_get_shape
            )
    except Exception as e:  # noqa: BLE001
        log.warning(
            "[G4_76b] TritonAttentionBackend.get_kv_cache_shape wrap failed: %s",
            e,
        )

    # --- Companion fix (G4_76c): override Attention.kv_cache_dtype for drafter ---
    #
    # reshape_and_cache_flash (called from FlashAttn impl's KV cache
    # update path) reads kv_cache_dtype from impl/layer attribute. The
    # global is "turboquant_4bit_nc"; the C++ op rejects it with:
    #   RuntimeError: Unsupported fp8 kv cache data type:
    #     turboquant_4bit_nc
    #
    # Wrap Attention.__init__: AFTER original init, for drafter layers,
    # reset self.kv_cache_dtype="auto" and self.impl.kv_cache_dtype="auto"
    # so downstream cache ops use the native path.
    global _ORIGINAL_ATTENTION_INIT
    try:
        from vllm.model_executor.layers.attention.attention import Attention
    except Exception as e:  # noqa: BLE001
        log.warning(
            "[G4_76c] Attention not importable — skipping kv_cache_dtype "
            "override: %s", e,
        )
        Attention = None  # type: ignore[assignment]

    if Attention is not None and not getattr(
        Attention.__init__, "_genesis_g4_76c_wrapped", False
    ):
        _ORIGINAL_ATTENTION_INIT = Attention.__init__
        drafter_prefix_local = "draft_model."

        def _wrapped_attn_init(self, *args, **kwargs):
            result = _ORIGINAL_ATTENTION_INIT(self, *args, **kwargs)
            prefix = kwargs.get("prefix", "") or ""
            if isinstance(prefix, str) and prefix.startswith(drafter_prefix_local):
                try:
                    if isinstance(getattr(self, "kv_cache_dtype", None), str) \
                            and self.kv_cache_dtype.startswith("turboquant_"):
                        old = self.kv_cache_dtype
                        self.kv_cache_dtype = "auto"
                        _ATTN_DTYPE_OVERRIDE_COUNT[0] += 1
                        if _ATTN_DTYPE_OVERRIDE_COUNT[0] <= 8:
                            log.warning(
                                "[G4_76c] drafter Attention(prefix=%r) "
                                "kv_cache_dtype: %r -> 'auto' (count=%d)",
                                prefix, old, _ATTN_DTYPE_OVERRIDE_COUNT[0],
                            )
                except Exception:  # noqa: BLE001
                    pass
                # Also propagate to impl if it has the same attribute.
                try:
                    impl = getattr(self, "impl", None)
                    if impl is not None \
                            and isinstance(getattr(impl, "kv_cache_dtype", None), str) \
                            and impl.kv_cache_dtype.startswith("turboquant_"):
                        impl.kv_cache_dtype = "auto"
                except Exception:  # noqa: BLE001
                    pass
            return result

        _wrapped_attn_init._genesis_g4_76c_wrapped = True  # type: ignore[attr-defined]
        Attention.__init__ = _wrapped_attn_init  # type: ignore[method-assign]

    _APPLIED = True

    log.warning(
        "[G4_76] INSTALLED: Gemma4Proposer._setup_gemma4_kv_sharing wrapped "
        "as no-op + FlashAttn/Triton get_kv_cache_shape wrapped to coerce "
        "cache_dtype_str='turboquant_*' to 'auto' for drafter group + "
        "Attention.__init__ wrapped to reset drafter kv_cache_dtype='auto' "
        "(drafter has native bf16 cache after G4_72)."
    )
    return "applied", (
        "G4_76 installed: drafter kv_sharing disabled — drafter is "
        "fully independent attention with its own kv_cache, block_table, "
        "and slot_mapping."
    )


def is_applied() -> bool:
    return _APPLIED


def noop_count() -> int:
    return _NOOP_COUNT[0]


def revert() -> bool:
    """Best-effort revert (test isolation only)."""
    global _APPLIED, _ORIGINAL_SETUP
    if not _APPLIED or _ORIGINAL_SETUP is None:
        return False
    try:
        from vllm.v1.spec_decode.gemma4 import Gemma4Proposer
        Gemma4Proposer._setup_gemma4_kv_sharing = _ORIGINAL_SETUP  # type: ignore[method-assign]
    except ImportError:
        return False
    _APPLIED = False
    _ORIGINAL_SETUP = None
    return True


__all__ = [
    "GENESIS_G4_76_MARKER",
    "apply",
    "is_applied",
    "noop_count",
    "revert",
]
