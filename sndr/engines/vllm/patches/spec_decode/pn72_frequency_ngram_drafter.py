# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN72 — frequency-based ngram draft post-filter.

Monkey-patches `vllm.v1.spec_decode.ngram_proposer.NgramProposer.propose`
to invoke the original implementation, then run our numpy-only frequency
filter on the returned drafts (rejects drafts whose first token has
< MIN_OBS occurrences in the recent context window).

================================================================
DESIGN: WHY WRAPPER (not text-patching the numba inner function)
================================================================

vllm's ngram drafter uses `@njit(parallel=True)` for the hot loop. Text-
patching numba-decorated source is risky — JIT recompiles, signature
drift can cause silent fallback to slow Python, or crash. Instead we
wrap the public `propose()` method, which is plain Python — safe to
monkey-patch, and our filter runs after numba returns. Latency cost is
~O(window) numpy compare per request, called once per step.

================================================================
DOUBLE PROTECTION (per Sander's requirement)
================================================================

  1. ENV GATE — if `GENESIS_ENABLE_PN72_FREQUENCY_NGRAM_DRAFTER=0` (or
     unset), wrapper short-circuits to identity. Zero overhead.
  2. TRY/EXCEPT — if our filter raises for any reason, log a warning
     and return the original (unfiltered) drafts. NEVER breaks vllm.
  3. IDEMPOTENT — re-applying the patch detects a marker attribute and
     skips, so multiple `apply_all` calls don't double-wrap.

================================================================
COMPOSABILITY
================================================================

P70 (auto-strict-ngram): orthogonal. P70 forces `prompt_lookup_min=8` at
config-time → fewer drafts produced. PN72 then post-filters those drafts
on first-token frequency. Both ON = strictly more conservative. Both OFF
= vanilla vllm. Independent env flags, no conflict.

P77 (adaptive ngram K): orthogonal. P77 adjusts how many speculative
tokens to ask for; PN72 only post-filters the produced drafts. Compose.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("genesis.wiring.pn72_frequency_ngram_drafter")


_MARKER = "_genesis_pn72_wrapped"

# Audit A-04 fix 2026-05-06: module-level handle to the pre-patch
# `NgramProposer.propose` so revert() can actually restore it. None until
# apply() runs successfully; then holds the original method ref.
_ORIGINAL_PROPOSE = None


def should_apply() -> bool:
    """Single source of truth — ask the helper module."""
    try:
        from sndr.engines.vllm.kernels_legacy.ngram_frequency_filter import is_enabled
    except Exception:
        return False
    return is_enabled()


def apply() -> tuple[str, str]:
    """Monkey-patch NgramProposer.propose with frequency post-filter.

    Returns (status, reason) — never raises. Caller (apply_all) interprets
    'applied' / 'skipped' / 'failed'.
    """
    if not should_apply():
        return "skipped", (
            "opt-in: set GENESIS_ENABLE_PN72_FREQUENCY_NGRAM_DRAFTER=1 to "
            "post-filter ngram drafts by first-token frequency in last "
            "GENESIS_PN72_FREQUENCY_WINDOW (default 1024) tokens. Rejects "
            "drafts with count < GENESIS_PN72_MIN_OBSERVATIONS (default 4)."
        )

    try:
        from vllm.v1.spec_decode import ngram_proposer as _np_mod
    except Exception as e:
        return "skipped", (
            f"vllm.v1.spec_decode.ngram_proposer not importable ({e}); "
            f"likely older/newer vllm pin without ngram method"
        )

    NgramProposer = getattr(_np_mod, "NgramProposer", None)
    if NgramProposer is None:
        return "skipped", "NgramProposer class not found in module"

    # Idempotency guard
    if getattr(NgramProposer.propose, _MARKER, False):
        return "applied", (
            "PN72 wrapper already installed (idempotent skip)"
        )

    # Audit A-04 fix 2026-05-06: stash original at module-level so revert()
    # can actually restore it. Previously `original_propose` lived only in
    # the closure of `_patched_propose`, so revert() had no handle and
    # could only clear the marker — leaving the wrap permanently in place.
    global _ORIGINAL_PROPOSE
    if _ORIGINAL_PROPOSE is None:
        _ORIGINAL_PROPOSE = NgramProposer.propose
    original_propose = _ORIGINAL_PROPOSE

    try:
        from sndr.engines.vllm.kernels_legacy.ngram_frequency_filter import (
            filter_with_env, record_filter_call,
        )
    except Exception as e:
        return "failed", f"helper import failed: {e}"

    def _patched_propose(
        self: Any,
        sampled_token_ids: list,
        num_tokens_no_spec: Any,
        token_ids_cpu: Any,
        slot_mappings: Any = None,
    ) -> list[list[int]]:
        """Wrapped propose — call original, then post-filter via helper.

        On ANY filter-side exception, log + return unfiltered drafts.
        Goal: never break vllm, only ever conservatively REJECT.
        """
        # Always call original — that's the contract with vllm
        drafts = original_propose(
            self, sampled_token_ids, num_tokens_no_spec,
            token_ids_cpu, slot_mappings,
        )
        # Defensive: if env flipped to OFF mid-process (unlikely), bypass
        try:
            from sndr.engines.vllm.kernels_legacy.ngram_frequency_filter import is_enabled
            if not is_enabled():
                return drafts
            in_count = sum(1 for d in drafts if d)
            filtered = filter_with_env(
                drafts, num_tokens_no_spec, token_ids_cpu,
            )
            kept = sum(1 for d in filtered if d)
            record_filter_call(in_count, kept, errors=0)
            return filtered
        except Exception as e:
            log.warning(
                "[PN72] filter raised %s — returning unfiltered drafts. "
                "This is the safety fallback; investigate the trace.",
                type(e).__name__,
                exc_info=True,
            )
            try:
                record_filter_call(0, 0, errors=1)
            except Exception:
                pass
            return drafts

    setattr(_patched_propose, _MARKER, True)
    NgramProposer.propose = _patched_propose
    return "applied", (
        "PN72 wrapped NgramProposer.propose — frequency filter active "
        "(MIN_OBS gating + WINDOW tokens). On any error → graceful "
        "fallback to unfiltered drafts (vllm never breaks)."
    )


def is_applied() -> bool:
    """Probe — check whether the marker is on the live class method."""
    try:
        from vllm.v1.spec_decode import ngram_proposer as _np_mod
        NgramProposer = getattr(_np_mod, "NgramProposer", None)
        if NgramProposer is None:
            return False
        return bool(getattr(NgramProposer.propose, _MARKER, False))
    except Exception:
        return False


def revert() -> bool:
    """Restore `NgramProposer.propose` to the pre-patch original.

    Audit A-04 fix 2026-05-06: previously this only cleared the marker
    (returning False). Now it actually rebinds the class method back to
    the stashed module-level `_ORIGINAL_PROPOSE` so subsequent `apply()`
    calls don't accumulate wrappers.

    Returns True on successful restore, False if not patched / failed.
    """
    global _ORIGINAL_PROPOSE
    try:
        from vllm.v1.spec_decode import ngram_proposer as _np_mod
        NgramProposer = getattr(_np_mod, "NgramProposer", None)
        if NgramProposer is None:
            return False
        # No wrap to revert
        if not getattr(NgramProposer.propose, _MARKER, False):
            return False
        if _ORIGINAL_PROPOSE is None:
            # Wrap installed but original lost — module reload is the only
            # safe path. Caller should restart the engine.
            return False
        NgramProposer.propose = _ORIGINAL_PROPOSE
        # Clear stashed ref so future apply() captures a fresh original
        # (defensive in case revert() then apply() is called without
        # reload).
        _ORIGINAL_PROPOSE = None
        return True
    except Exception:
        return False
