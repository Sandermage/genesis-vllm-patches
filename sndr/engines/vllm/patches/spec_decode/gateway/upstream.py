# SPDX-License-Identifier: Apache-2.0
"""upstream — health-tracked upstream state machine.

Each upstream (``default``, ``structured``) has its own ``UpstreamState``.
A background task polls ``GET /v1/models`` on each upstream every 5
seconds; consecutive results drive a 3-state machine:

    up  -- 1 of last 3 failed --> degraded
    up  -- 3 consec. failed   --> down
    down  -- 3 consec. ok      --> up

The dispatcher consults ``is_routable()``: ``True`` only for ``up``
state. ``degraded`` still routes but emits a warning + metric;
``down`` triggers fallback.

Authored 2026-05-20 (D2a).
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from . import metrics

log = logging.getLogger("genesis.spec_decode.gateway.upstream")


@dataclass
class UpstreamState:
    name: str          # 'default' / 'structured'
    base_url: str      # e.g. 'http://localhost:8101'
    # /health is unauthenticated on vLLM; /v1/models requires
    # `--api-key`-style bearer auth. Probe /health to keep gateway
    # auth-free.
    health_path: str = "/health"
    timeout_s: float = 2.0
    state: str = "down"   # 'up' / 'degraded' / 'down'
    last_check_ts: float = 0.0
    last_error: str = ""
    # rolling history of last 3 check results (True = ok)
    history: deque = field(default_factory=lambda: deque(maxlen=3))

    def is_routable(self) -> bool:
        return self.state in ("up", "degraded")

    def is_up(self) -> bool:
        return self.state == "up"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_url": self.base_url,
            "state": self.state,
            "last_check_ts": self.last_check_ts,
            "last_error": self.last_error,
            "history": list(self.history),
        }


async def _probe_once(state: UpstreamState, client) -> None:
    """One health check tick."""
    state.last_check_ts = time.time()
    try:
        url = state.base_url.rstrip("/") + state.health_path
        r = await client.get(url, timeout=state.timeout_s)
        ok = (r.status_code == 200)
        state.last_error = "" if ok else f"HTTP {r.status_code}"
    except Exception as e:  # noqa: BLE001
        ok = False
        state.last_error = f"{type(e).__name__}: {e}"

    state.history.append(ok)

    # Probe-failure metric: count every non-2xx / exception probe.
    if not ok:
        try:
            metrics.UPSTREAM_PROBE_FAILURES_TOTAL.labels(
                upstream=state.name).inc()
        except Exception:
            pass

    # Transition logic
    old_state = state.state
    hist = list(state.history)
    if len(hist) >= 3 and all(hist[-3:]):
        new_state = "up"
    elif len(hist) >= 3 and not any(hist[-3:]):
        new_state = "down"
    else:
        # mixed: degraded if at least one of last 3 ok; else down
        if len(hist) >= 1 and any(hist[-3:]):
            new_state = "degraded" if not all(hist[-3:]) else "up"
        else:
            new_state = "down"
    state.state = new_state

    # Metrics — both the legacy genesis_upstream_health gauge and the
    # canonical sndr_upstream_health_state are kept in lockstep.
    # Dashboards built on either name continue to work; new dashboards
    # should reference the sndr_ name.
    gauge_val = {"up": 1.0, "degraded": 0.5, "down": 0.0}.get(new_state, 0.0)
    try:
        metrics.UPSTREAM_HEALTH.labels(upstream=state.name).set(gauge_val)
        metrics.UPSTREAM_HEALTH_STATE.labels(
            upstream=state.name).set(gauge_val)
    except Exception:
        pass

    if new_state != old_state:
        log.warning(
            "[gateway.upstream] %s: %s -> %s (last_error=%r history=%s)",
            state.name, old_state, new_state, state.last_error, hist,
        )


async def run_health_loop(states: list[UpstreamState],
                          interval_s: float = 5.0) -> None:
    """Background coroutine. Probes each upstream every interval_s."""
    import httpx
    async with httpx.AsyncClient() as client:
        while True:
            for s in states:
                try:
                    await _probe_once(s, client)
                except Exception as _e:  # noqa: BLE001
                    log.warning("[gateway.upstream] probe error %s: %s",
                                s.name, _e)
            await asyncio.sleep(interval_s)


__all__ = ["UpstreamState", "run_health_loop"]
