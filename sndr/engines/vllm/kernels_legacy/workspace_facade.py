# SPDX-License-Identifier: Apache-2.0
"""WorkspaceFacade — extracted policy logic for the 4 TurboQuant workspace
patches (P98 / P99 / PN118 / SNDR_WORKSPACE_001).

Phase 6 P3.2 deep refactor — v11.3.0 lands the EXTRACTED, TESTED policy
module. The 4 source patches are UNCHANGED in v11.3.0 (operator can
still run the legacy text-patch path). v12.0.0 will optionally wire the
patches to delegate into this facade — that switch requires rig bench
validation under cudagraph capture, which is deferred to a separate
operator session.

What this facade owns
---------------------

The 4 patches encode FOUR composable policy decisions over the
WorkspaceManager + turboquant_attn hot path:

1. **DECODE_REVERT_VS_MANAGER** (P98) — should this decode call use the
   pre-vllm#40941 `getattr(layer, "_tq_*_buf", None)` path (FAST on
   Ampere small batch) or the upstream WorkspaceManager indirection
   (SLOW on Ampere)? Operator-configurable; default is upstream.

2. **WS_MEMO_CACHE_HIT** (P99) — given `(shapes_and_dtypes, ubatch_id,
   ws_ptr)`, is there a cached list of tensor views that we can return
   without recomputing `_compute_bytes + round_up + accumulate`?

3. **WS_TRY_ACQUIRE_LOCKED** (PN118) — given a LOCKED workspace and a
   request larger than current size, should we return None (graceful
   fallback — caller uses torch.empty) or raise?

4. **WS_GROW_AFTER_LOCK** (SNDR_WORKSPACE_001) — at the lock-growth
   guard site, should we raise AssertionError (upstream default) or
   warn + allow growth (Genesis-original)?

Each is a pure decision function. They take inputs (workspace state,
shapes, env flags) and return a verdict + reason. The actual mutation
of workspace state stays in the patches' text-patch apply() functions
— byte-equivalent to v11.2.0 behavior.

Composition
-----------

The decision functions COMPOSE in the following order on a single
WorkspaceManager.get_simultaneous() call:

  decode_revert_decision  (P98)   → fast path or manager?
  └─ if manager:
       memo_cache_decision (P99)   → cache hit? return cached view.
       └─ if miss:
            try_acquire_decision (PN118)
            ├─ unlocked or fits: proceed to alloc
            └─ locked-undersized: caller fallback
       grow_after_lock_decision (SNDR_WORKSPACE_001)
       └─ raise or warn-grow?

Operators who run all 4 patches enabled get all 4 decisions active
simultaneously — they are operationally orthogonal (each affects a
different code site).

Byte-equivalence contract
--------------------------

Each decision function returns EXACTLY the same boolean / numeric /
None verdict that the equivalent text-patch path produces. Unit tests
encode the verdict matrix for every documented input combination.

When v12.0.0 wires the text patches to delegate here, the patches
become 1-line dispatch calls; the actual decision logic moves to this
module (covered by unit tests rather than text-patch fixtures).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa
Status: v11.3.0 EXTRACTED policy module (not yet wired into patches;
v12.0.0 wire-up deferred to bench window)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from threading import RLock
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import torch  # noqa: F401

log = logging.getLogger("genesis.workspace_facade")


__all__ = [
    "WorkspaceDecision",
    "WorkspaceFacade",
    "P98_ENV_FLAG",
    "P99_ENV_FLAG",
    "PN118_ENV_FLAG",
    "SNDR_WS_001_ENV_FLAG",
]


# Env flags — single source of truth, mirrors registry entries.
#
# v12.0 (2026-06-04): aligned env-flag names with the canonical short form
# stored in ``PATCH_REGISTRY[pid]["env_flag"]``. Previous descriptive
# long-form names were never recognised by the dispatcher and silently
# dropped operator-set overrides.
P98_ENV_FLAG = "GENESIS_ENABLE_P98"
P99_ENV_FLAG = "GENESIS_ENABLE_P99"
PN118_ENV_FLAG = "GENESIS_ENABLE_PN118"
SNDR_WS_001_ENV_FLAG = "GENESIS_ENABLE_SNDR_WORKSPACE_001"


@dataclass(frozen=True)
class WorkspaceDecision:
    """Verdict + reason returned by every facade decision function.

    `verdict` is the action (e.g., 'use_fast_path', 'cache_hit',
    'graceful_fallback', 'warn_and_grow'). `reason` is a short
    operator-facing string. `extra` holds decision-specific payload
    (cached tensor view, fallback tensor, etc.) — None for pure
    verdict-only decisions.
    """
    verdict: str
    reason: str
    extra: Any = None


def _env_on(flag: str) -> bool:
    """Centralized env-flag interpretation. Empty + "0" + missing = off."""
    return os.environ.get(flag, "").strip() in ("1", "true", "True")


class WorkspaceFacade:
    """Stateless policy facade — every method is a pure decision function.

    Per-process state (P99 memo cache, decision counters for
    observability) lives in class-level dicts protected by RLock. The
    facade does NOT own workspace tensors; those stay in
    WorkspaceManager + TurboQuantBufferManager.

    Operator opt-in: each decision is gated by its source patch's env
    flag. With no flags set, every decision returns the upstream-default
    verdict.
    """

    # P99 memoization cache — (shapes_and_dtypes_key, ubatch_id, ws_ptr)
    # → list of tensor views. Bounded by _MEMO_MAX_ENTRIES; oldest
    # evicted FIFO when full.
    _MEMO_CACHE: dict[tuple, list] = {}
    _MEMO_ORDER: list[tuple] = []
    _MEMO_MAX_ENTRIES: int = 256
    _LOCK = RLock()

    # Decision counters — observability surface for `sndr explain
    # workspace_facade`. Reset via clear_for_tests().
    _STATS: dict[str, int] = {
        "decode_revert_fast_path": 0,
        "decode_revert_manager": 0,
        "memo_hit": 0,
        "memo_miss": 0,
        "memo_evict": 0,
        "try_acquire_pass": 0,
        "try_acquire_graceful_fallback": 0,
        "grow_after_lock_warn": 0,
        "grow_after_lock_raise": 0,
    }

    # ─── P98 — decode revert vs manager ───────────────────────────────

    @classmethod
    def decide_decode_path(
        cls,
        layer: Any,
        is_decode: bool = True,
    ) -> WorkspaceDecision:
        """P98 — choose between fast getattr path and WorkspaceManager.

        On `GENESIS_ENABLE_P98=1` and a decode call
        on a layer that has the legacy `_tq_*_buf` attrs, return
        verdict='use_fast_path' + extra=(mid_o, output, lse) buffers.

        Otherwise verdict='use_manager' (caller proceeds with upstream
        WorkspaceManager indirection).

        Byte-equivalent to P98's text-patch insertion (see
        integrations/attention/turboquant/p98_tq_workspace_revert.py
        lines 85-110).
        """
        if not _env_on(P98_ENV_FLAG):
            with cls._LOCK:
                cls._STATS["decode_revert_manager"] += 1
            return WorkspaceDecision(
                verdict="use_manager",
                reason="P98 disabled (default — upstream WorkspaceManager path)",
            )
        if not is_decode:
            with cls._LOCK:
                cls._STATS["decode_revert_manager"] += 1
            return WorkspaceDecision(
                verdict="use_manager",
                reason="non-decode call — P98 only reverts decode path",
            )
        # Try to extract legacy attrs in the order P98 expects them.
        mid_o = getattr(layer, "_tq_mid_o_buf", None) if layer is not None else None
        output = getattr(layer, "_tq_output_buf", None) if layer is not None else None
        lse = getattr(layer, "_tq_lse_buf", None) if layer is not None else None
        if mid_o is None and output is None and lse is None:
            with cls._LOCK:
                cls._STATS["decode_revert_manager"] += 1
            return WorkspaceDecision(
                verdict="use_manager",
                reason=(
                    "P98 enabled but layer lacks legacy _tq_*_buf attrs — "
                    "falling back to WorkspaceManager"
                ),
            )
        with cls._LOCK:
            cls._STATS["decode_revert_fast_path"] += 1
        return WorkspaceDecision(
            verdict="use_fast_path",
            reason="P98 fast path: legacy per-layer buffers found",
            extra=(mid_o, output, lse),
        )

    # ─── P99 — memoization cache ─────────────────────────────────────

    @classmethod
    def lookup_memo(
        cls,
        shapes_and_dtypes_key: tuple,
        ubatch_id: int,
        ws_ptr: int,
    ) -> WorkspaceDecision:
        """P99 — query the get_simultaneous() memoization cache.

        Returns verdict='cache_hit' + extra=cached_views_list when a
        prior call with the same (key, ubatch_id, ws_ptr) cached its
        tensor-views list. Returns verdict='cache_miss' otherwise.

        `ws_ptr` discrimination ensures the cache invalidates on
        workspace re-allocation (a new ws_ptr means a new underlying
        buffer; views from the old ptr are stale).

        Byte-equivalent to P99's text-patch dict lookup.
        """
        if not _env_on(P99_ENV_FLAG):
            return WorkspaceDecision(
                verdict="cache_miss",
                reason="P99 disabled (default — no memoization)",
            )
        cache_key = (shapes_and_dtypes_key, ubatch_id, ws_ptr)
        with cls._LOCK:
            cached = cls._MEMO_CACHE.get(cache_key)
            if cached is None:
                cls._STATS["memo_miss"] += 1
                return WorkspaceDecision(
                    verdict="cache_miss",
                    reason="P99 enabled but cache key not present",
                )
            cls._STATS["memo_hit"] += 1
            # Copy of list — tensors are views, returned by reference.
            return WorkspaceDecision(
                verdict="cache_hit",
                reason="P99 cache hit — returning cached views",
                extra=list(cached),
            )

    @classmethod
    def store_memo(
        cls,
        shapes_and_dtypes_key: tuple,
        ubatch_id: int,
        ws_ptr: int,
        tensor_views: list,
    ) -> WorkspaceDecision:
        """P99 — store a freshly-computed views list into the memo cache.

        FIFO eviction at `_MEMO_MAX_ENTRIES` entries. Returns
        verdict='stored' on insert, 'noop' when P99 disabled.
        """
        if not _env_on(P99_ENV_FLAG):
            return WorkspaceDecision(
                verdict="noop",
                reason="P99 disabled (default — no memoization storage)",
            )
        cache_key = (shapes_and_dtypes_key, ubatch_id, ws_ptr)
        with cls._LOCK:
            if cache_key in cls._MEMO_CACHE:
                # Refresh ordering — pop + reinsert
                try:
                    cls._MEMO_ORDER.remove(cache_key)
                except ValueError:
                    pass
                cls._MEMO_ORDER.append(cache_key)
                cls._MEMO_CACHE[cache_key] = list(tensor_views)
                return WorkspaceDecision(
                    verdict="stored",
                    reason="P99 cache refresh — key already present, views updated",
                )
            # FIFO eviction
            while len(cls._MEMO_CACHE) >= cls._MEMO_MAX_ENTRIES:
                evicted = cls._MEMO_ORDER.pop(0)
                cls._MEMO_CACHE.pop(evicted, None)
                cls._STATS["memo_evict"] += 1
            cls._MEMO_CACHE[cache_key] = list(tensor_views)
            cls._MEMO_ORDER.append(cache_key)
            return WorkspaceDecision(
                verdict="stored",
                reason="P99 cache fresh insert",
            )

    # ─── PN118 — try_acquire graceful fallback ───────────────────────

    @classmethod
    def decide_try_acquire(
        cls,
        is_locked: bool,
        current_size_bytes: int,
        required_total_bytes: int,
    ) -> WorkspaceDecision:
        """PN118 — graceful fallback decision on locked-undersized workspace.

        On `GENESIS_ENABLE_PN118=1`, when
        workspace is LOCKED AND required_total_bytes > current_size_bytes,
        returns verdict='graceful_fallback' (caller uses torch.empty
        instead of growing the locked workspace).

        Otherwise verdict='pass' (caller proceeds with normal acquire
        path — either unlocked or already-sufficient).

        Byte-equivalent to PN118's try_get_simultaneous() body
        (vllm#42551 jasonboukheir backport, integrations/attention/
        turboquant/pn118_tq_workspace_fallback.py lines 140-176).
        """
        if not _env_on(PN118_ENV_FLAG):
            with cls._LOCK:
                cls._STATS["try_acquire_pass"] += 1
            return WorkspaceDecision(
                verdict="pass",
                reason="PN118 disabled — upstream path (may raise on locked-undersized)",
            )
        if not is_locked:
            with cls._LOCK:
                cls._STATS["try_acquire_pass"] += 1
            return WorkspaceDecision(
                verdict="pass",
                reason="PN118 enabled but workspace unlocked — normal acquire",
            )
        if required_total_bytes <= current_size_bytes:
            with cls._LOCK:
                cls._STATS["try_acquire_pass"] += 1
            return WorkspaceDecision(
                verdict="pass",
                reason="PN118 enabled, locked, but request fits — normal acquire",
            )
        with cls._LOCK:
            cls._STATS["try_acquire_graceful_fallback"] += 1
        return WorkspaceDecision(
            verdict="graceful_fallback",
            reason=(
                f"PN118 graceful fallback: workspace locked + undersized "
                f"(need {required_total_bytes}, have {current_size_bytes})"
            ),
        )

    # ─── SNDR_WORKSPACE_001 — grow-after-lock guard ──────────────────

    @classmethod
    def decide_grow_after_lock(
        cls,
        is_locked: bool,
    ) -> WorkspaceDecision:
        """SNDR_WORKSPACE_001 — replace lock-growth AssertionError with
        warn + allow.

        On `GENESIS_ENABLE_SNDR_WORKSPACE_001=1` AND workspace is_locked,
        returns verdict='warn_and_grow' (caller logs WARN + proceeds
        with grow). Otherwise verdict='raise' (upstream default —
        caller raises AssertionError).

        Byte-equivalent to SNDR_WORKSPACE_001's text-patch lock guard
        replacement (integrations/worker/
        sndr_workspace_001_grow_after_lock.py lines 56-78).
        """
        if not _env_on(SNDR_WS_001_ENV_FLAG):
            with cls._LOCK:
                cls._STATS["grow_after_lock_raise"] += 1
            return WorkspaceDecision(
                verdict="raise",
                reason="SNDR_WORKSPACE_001 disabled (upstream — AssertionError on lock growth)",
            )
        if not is_locked:
            # Lock guard doesn't fire on unlocked path; this is a
            # no-decision case (no growth to guard against).
            return WorkspaceDecision(
                verdict="no_guard_needed",
                reason="workspace not locked — no growth guard applicable",
            )
        with cls._LOCK:
            cls._STATS["grow_after_lock_warn"] += 1
        return WorkspaceDecision(
            verdict="warn_and_grow",
            reason="SNDR_WORKSPACE_001 enabled — warn + allow grow instead of AssertionError",
        )

    # ─── Composition helper ──────────────────────────────────────────

    @classmethod
    def decide_get_simultaneous(
        cls,
        is_decode: bool,
        layer: Any,
        shapes_and_dtypes_key: tuple,
        ubatch_id: int,
        ws_ptr: int,
        is_locked: bool,
        current_size_bytes: int,
        required_total_bytes: int,
    ) -> dict:
        """Top-level composition — produces all 4 decisions for a single
        WorkspaceManager.get_simultaneous() call site.

        Returns a dict with all 4 decision objects keyed by patch ID +
        a composite `next_action` summarising what the caller should do.

        Operator can pipe this into observability via
        `sndr explain workspace_facade --trace`.
        """
        p98 = cls.decide_decode_path(layer, is_decode=is_decode)
        # P98 fast path bypasses everything else.
        if p98.verdict == "use_fast_path":
            return {
                "P98": p98,
                "P99": None,
                "PN118": None,
                "SNDR_WORKSPACE_001": None,
                "next_action": "use_fast_path_buffers",
            }
        p99 = cls.lookup_memo(shapes_and_dtypes_key, ubatch_id, ws_ptr)
        if p99.verdict == "cache_hit":
            return {
                "P98": p98,
                "P99": p99,
                "PN118": None,
                "SNDR_WORKSPACE_001": None,
                "next_action": "return_cached_views",
            }
        pn118 = cls.decide_try_acquire(
            is_locked, current_size_bytes, required_total_bytes,
        )
        if pn118.verdict == "graceful_fallback":
            return {
                "P98": p98,
                "P99": p99,
                "PN118": pn118,
                "SNDR_WORKSPACE_001": None,
                "next_action": "caller_uses_torch_empty",
            }
        sndr = cls.decide_grow_after_lock(is_locked)
        next_action = "normal_acquire"
        if sndr.verdict == "warn_and_grow":
            next_action = "log_warn_and_proceed_with_grow"
        elif sndr.verdict == "raise":
            next_action = (
                "raise_assertion_error_if_growth_needed"
                if is_locked else "normal_acquire"
            )
        return {
            "P98": p98,
            "P99": p99,
            "PN118": pn118,
            "SNDR_WORKSPACE_001": sndr,
            "next_action": next_action,
        }

    # ─── Observability + maintenance ─────────────────────────────────

    @classmethod
    def stats(cls) -> dict:
        """Decision counters snapshot."""
        with cls._LOCK:
            return dict(cls._STATS)

    @classmethod
    def memo_size(cls) -> int:
        """Number of entries in the P99 memo cache."""
        with cls._LOCK:
            return len(cls._MEMO_CACHE)

    @classmethod
    def clear_for_tests(cls) -> None:
        """Drop memo cache + reset counters — for test isolation only."""
        with cls._LOCK:
            cls._MEMO_CACHE.clear()
            cls._MEMO_ORDER.clear()
            for k in cls._STATS:
                cls._STATS[k] = 0

    @classmethod
    def summary(cls) -> dict:
        """Operator-facing summary — used by future `sndr explain
        workspace_facade`."""
        with cls._LOCK:
            return {
                "memo_size": len(cls._MEMO_CACHE),
                "memo_max": cls._MEMO_MAX_ENTRIES,
                "stats": dict(cls._STATS),
                "env_flags": {
                    "P98": _env_on(P98_ENV_FLAG),
                    "P99": _env_on(P99_ENV_FLAG),
                    "PN118": _env_on(PN118_ENV_FLAG),
                    "SNDR_WORKSPACE_001": _env_on(SNDR_WS_001_ENV_FLAG),
                },
            }
