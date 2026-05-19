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

log = logging.getLogger("genesis.gemma4.g4_76_disable_drafter_kv_sharing")

GENESIS_G4_76_MARKER = (
    "Genesis G4_76 Disable Gemma4Proposer._setup_gemma4_kv_sharing — "
    "drafter becomes fully-independent attention layer (PN265 fix)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_76_DISABLE_DRAFTER_KV_SHARING"
_APPLIED = False
_ORIGINAL_SETUP = None
_NOOP_COUNT = [0]


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
    _APPLIED = True

    log.warning(
        "[G4_76] INSTALLED: Gemma4Proposer._setup_gemma4_kv_sharing wrapped "
        "as no-op — drafter Attention layers will not have "
        "kv_sharing_target_layer_name set."
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
