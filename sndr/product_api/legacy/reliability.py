# SPDX-License-Identifier: Apache-2.0
"""Per-host reachability tracking with a circuit-breaker state indicator.

Ports the aggregator's circuit-breaker idea to SNDR's fleet: every reachability
probe is recorded against the host key, from which we derive an uptime %, a
bounded sample history (for a sparkline), and a three-state breaker verdict:

* ``closed``     — last probe OK (or failures below threshold); normal flow.
* ``open``       — ``fail_threshold`` consecutive failures within ``recovery``
                   seconds; the host is treated as down (back off).
* ``half_open``  — the recovery window has elapsed since the last failure; the
                   next probe is the trial that either closes or re-opens it.

Read-only bookkeeping — the tracker never itself contacts a host; callers feed it
probe outcomes and may consult :meth:`allow` to decide whether to back off.
"""
from __future__ import annotations

from typing import Any, Optional

_MAX_SAMPLES = 60
_SPARK = 30


class ReliabilityTracker:
    def __init__(self, *, fail_threshold: int = 3, recovery: float = 60.0) -> None:
        self._hist: dict[str, list[tuple[float, bool]]] = {}
        self._fail_threshold = fail_threshold
        self._recovery = recovery

    def record(self, key: str, ok: bool, *, now: float) -> None:
        hist = self._hist.setdefault(key, [])
        hist.append((now, bool(ok)))
        if len(hist) > _MAX_SAMPLES:
            del hist[: len(hist) - _MAX_SAMPLES]

    def _consecutive_fails(self, hist: list[tuple[float, bool]]) -> int:
        cf = 0
        for _, ok in reversed(hist):
            if ok:
                break
            cf += 1
        return cf

    def _state(self, hist: list[tuple[float, bool]], now: float) -> str:
        cf = self._consecutive_fails(hist)
        if cf < self._fail_threshold:
            return "closed"
        last_ts = hist[-1][0]
        return "open" if (now - last_ts) < self._recovery else "half_open"

    def allow(self, key: str, *, now: float) -> bool:
        """Whether a caller should probe now (False only while the breaker is open)."""
        hist = self._hist.get(key)
        if not hist:
            return True
        return self._state(hist, now) != "open"

    def snapshot(self, key: str, *, now: float) -> Optional[dict[str, Any]]:
        hist = self._hist.get(key)
        if not hist:
            return None
        oks = sum(1 for _, ok in hist if ok)
        last_ok = next((ts for ts, ok in reversed(hist) if ok), None)
        return {
            "uptime_pct": round(oks / len(hist) * 100, 1),
            "checks": len(hist),
            "samples": [1 if ok else 0 for _, ok in hist][-_SPARK:],
            "state": self._state(hist, now),
            "consecutive_fails": self._consecutive_fails(hist),
            "last_ok": last_ok,
        }

    def snapshot_all(self, *, now: float) -> dict[str, Any]:
        return {key: self.snapshot(key, now=now) for key in self._hist}


# Module-level singleton shared by the probe routes + the reliability endpoint.
TRACKER = ReliabilityTracker()
