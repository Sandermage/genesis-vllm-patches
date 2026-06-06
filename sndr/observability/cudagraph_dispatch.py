# SPDX-License-Identifier: Apache-2.0
"""Sprint 2.6 — CUDA graph dispatch hit-rate observability.

Audit closure 2026-05-08 (noonghunna): Wave 6 PN16 V1 regression was a
CUDA graph dispatch mismatch — prompt mutation pushed requests into
captured-graph "miss" → eager fallback → 5-10× slower per-token decode.
The pathology is invisible to wall_TPS averaging (mixes hit + miss
requests) and shows up only as exploded CV (37% vs 6% baseline).

This module gives operators a direct measurement: per-process counter
of captured-graph hits vs eager fallbacks, with periodic summary logs.

Design points:

  • Process-local counter (no shared state across workers; each vllm
    worker emits its own line — operator aggregates by `grep` / log
    aggregator).
  • Lock-protected for thread safety inside a single worker.
  • Periodic emit (every ``GENESIS_CUDAGRAPH_LOG_EVERY`` requests,
    default 1000) — avoids per-request log spam at decode-time scale.
  • Default OFF — opt-in via ``GENESIS_CUDAGRAPH_DISPATCH_TRACE=1``
    so production with no need for the trace pays zero cost.

Wire-in (Wave 7+ followup, separate patch):

  Genesis text-patches the dispatch site in vllm's gpu_model_runner.py
  to call ``record_dispatch(matched=True/False)`` per scheduled request.
  This module exposes the counter contract; the wire-in is incremental
  and gated by its own env flag.

Until the wire-in lands, ``record_dispatch`` is a callable contract that
operators / future patches use. Calling it from outside Genesis (e.g. a
custom debug script) also works — it's pure Python state, no vllm
runtime dependency.

Author: Sandermage; Sprint 2.6 / 2026-05-09.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("genesis.cudagraph")


# ─── Config ────────────────────────────────────────────────────────────


def _is_trace_enabled() -> bool:
    val = os.environ.get("GENESIS_CUDAGRAPH_DISPATCH_TRACE", "").strip().lower()
    return val in ("1", "true", "yes", "y", "on")


def _log_every() -> int:
    """Emit a summary line every N dispatches. 0 means never auto-emit
    (operator must call ``emit_summary()`` manually). Default 1000."""
    try:
        v = int(os.environ.get("GENESIS_CUDAGRAPH_LOG_EVERY", "1000"))
        return max(0, v)
    except (TypeError, ValueError):
        return 1000


# ─── Counter ──────────────────────────────────────────────────────────


@dataclass
class CudagraphDispatchSummary:
    """Snapshot of dispatch counters at a point in time."""
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate_pct(self) -> Optional[float]:
        if self.total == 0:
            return None
        return round(100.0 * self.hits / self.total, 2)

    @property
    def miss_rate_pct(self) -> Optional[float]:
        if self.total == 0:
            return None
        return round(100.0 * self.misses / self.total, 2)


class _DispatchCounter:
    """Process-local thread-safe counter. Internal — operators use the
    module-level ``record_dispatch`` / ``get_summary`` helpers."""

    def __init__(self, log_every: int = 1000):
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._last_emit = 0
        self._log_every = max(0, log_every)

    def record(self, matched: bool) -> None:
        with self._lock:
            if matched:
                self._hits += 1
            else:
                self._misses += 1
            if self._log_every == 0:
                return
            total = self._hits + self._misses
            if total - self._last_emit >= self._log_every:
                self._last_emit = total
                self._emit_unlocked()

    def snapshot(self) -> CudagraphDispatchSummary:
        with self._lock:
            return CudagraphDispatchSummary(hits=self._hits, misses=self._misses)

    def emit(self) -> None:
        """Emit a summary line now (regardless of log_every threshold)."""
        with self._lock:
            self._emit_unlocked()

    def _emit_unlocked(self) -> None:
        """MUST be called with self._lock held."""
        snap = CudagraphDispatchSummary(hits=self._hits, misses=self._misses)
        if snap.total == 0:
            return
        log.info(
            "[Genesis cudagraph] dispatch hit-rate %s%% — captured-graph "
            "hits %d / %d total, eager fallback %d (%s%%)",
            snap.hit_rate_pct, snap.hits, snap.total,
            snap.misses, snap.miss_rate_pct,
        )

    def reset(self) -> None:
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._last_emit = 0


# ─── Public API ───────────────────────────────────────────────────────


_COUNTER: Optional[_DispatchCounter] = None


def _get_counter() -> _DispatchCounter:
    global _COUNTER
    if _COUNTER is None:
        _COUNTER = _DispatchCounter(log_every=_log_every())
    return _COUNTER


def record_dispatch(matched: bool) -> None:
    """Record one CUDA graph dispatch event.

    Args:
        matched: True if the request was dispatched into a captured
            cudagraph (fast path); False if it fell back to eager
            execution (slow path).

    Default OFF — this is a no-op unless
    ``GENESIS_CUDAGRAPH_DISPATCH_TRACE=1`` so production paths pay
    zero overhead when the trace isn't needed.

    Thread-safe; process-local. Each vllm worker maintains its own
    counter — operators aggregate across workers via log scraping.
    """
    if not _is_trace_enabled():
        return
    _get_counter().record(matched)


def get_summary() -> CudagraphDispatchSummary:
    """Snapshot the current counter — safe to call any time (returns
    zeroed snapshot when trace disabled or no events yet)."""
    if _COUNTER is None:
        return CudagraphDispatchSummary()
    return _COUNTER.snapshot()


def emit_summary() -> None:
    """Emit a summary log line on demand (no-op when no events).

    Useful for end-of-bench reporting or `sndr report` invocation."""
    if _COUNTER is None:
        return
    _COUNTER.emit()


def reset_summary() -> None:
    """Reset counters to zero. Mostly for tests / iterative bench runs."""
    if _COUNTER is None:
        return
    _COUNTER.reset()


# Module-reset hook for tests — wipes the singleton.
def _reset_module_state() -> None:
    global _COUNTER
    _COUNTER = None
