# SPDX-License-Identifier: Apache-2.0
"""Background health-watch for managed containers → operator alerts.

A light loop snapshots the managed (vLLM/engine) containers, compares each one's
state to the previous tick, and emits an alert on a meaningful transition — the
engine going DOWN (crash / OOM-kill / stop) or coming back. Alerts go through
:mod:`notify` (audit event + Telegram).

The transition detector is a pure function so it's fully unit-tested; the thread
is thin glue. First tick only establishes the baseline (no alerts), so starting
the daemon doesn't page you for already-stopped containers.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional


def _norm(state: str) -> str:
    return "running" if str(state).strip().lower() == "running" else "down"


def detect_transitions(prev: dict[str, str], containers: list[Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Compare the previous name→state map against the current containers.

    Returns ``(alerts, new_state_map)``. An unseen container only seeds the
    baseline (no alert). Returns recovery alerts too, so you get the all-clear."""
    alerts: list[dict[str, Any]] = []
    current: dict[str, str] = {}
    for c in containers:
        name = getattr(c, "name", None) or (c.get("name") if isinstance(c, dict) else None)
        if not name:
            continue
        state = getattr(c, "state", None) if not isinstance(c, dict) else c.get("state", "")
        status = getattr(c, "status", "") if not isinstance(c, dict) else c.get("status", "")
        norm = _norm(state or "")
        current[name] = norm
        old = prev.get(name)
        if old is None:
            continue  # baseline only
        if old == "running" and norm == "down":
            alerts.append({"name": name, "level": "critical",
                           "message": f"🔴 <b>{name}</b> is DOWN — {status or state}"})
        elif old == "down" and norm == "running":
            alerts.append({"name": name, "level": "ok",
                           "message": f"🟢 <b>{name}</b> recovered (running)"})
    return alerts, current


class Watcher:
    """Stateful wrapper over :func:`detect_transitions` (holds the prev snapshot)."""

    def __init__(self) -> None:
        self._prev: dict[str, str] = {}
        self.last_tick: float = 0.0
        self.watched: int = 0

    def tick(self, containers: list[Any]) -> list[dict[str, Any]]:
        alerts, self._prev = detect_transitions(self._prev, containers)
        self.watched = len(containers)
        return alerts


_THREAD: Optional[threading.Thread] = None
_STARTED = False


def start_watch(get_control: Callable[[], Any], *, interval: float = 20.0) -> bool:
    """Start the background watch thread once. Returns True if started.

    ``get_control`` returns a ContainerControl (or raises if unavailable). Each
    tick lists managed containers and dispatches alerts via :mod:`notify`."""
    global _THREAD, _STARTED
    if _STARTED:
        return False
    from . import notify

    def loop() -> None:
        watcher = Watcher()
        while True:
            try:
                if notify.alerts_enabled():
                    control = get_control()
                    containers = control.list_managed()
                    for alert in watcher.tick(containers):
                        notify.notify(alert["message"], kind=alert["level"],
                                      detail={"container": alert["name"]})
                    watcher.last_tick = time.time()
            except Exception:
                pass  # transient (socket gone, host unreachable) — keep looping
            time.sleep(interval)

    _THREAD = threading.Thread(target=loop, name="sndr-container-watch", daemon=True)
    _THREAD.start()
    _STARTED = True
    return True


__all__ = ["detect_transitions", "Watcher", "start_watch"]
