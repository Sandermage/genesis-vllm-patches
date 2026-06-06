# SPDX-License-Identifier: Apache-2.0
"""metrics — Prometheus counters/histograms for the dispatcher.

Day-1 metric catalog per the deployment plan section 5. All metrics
are no-ops if ``prometheus_client`` isn't installed — gateway must
still work in environments without the dep.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("genesis.spec_decode.gateway.metrics")


class _NullMetric:
    """Stand-in when prometheus_client is unavailable."""

    def labels(self, **kwargs: Any) -> "_NullMetric":
        return self

    def inc(self, *args: Any, **kwargs: Any) -> None:
        pass

    def observe(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set(self, *args: Any, **kwargs: Any) -> None:
        pass


try:
    from prometheus_client import (  # type: ignore[import-not-found]
        Counter,
        Gauge,
        Histogram,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    REGISTRY = CollectorRegistry()
    _AVAILABLE = True
except ImportError:
    log.warning(
        "[gateway.metrics] prometheus_client not installed; metrics "
        "are no-ops. Install with `pip install prometheus-client`.")
    Counter = Gauge = Histogram = None  # type: ignore[assignment]
    REGISTRY = None  # type: ignore[assignment]
    generate_latest = lambda *a, **k: b""  # type: ignore[assignment]
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"
    _AVAILABLE = False


def _counter(name: str, doc: str, labelnames: tuple[str, ...] = ()) -> Any:
    if not _AVAILABLE:
        return _NullMetric()
    return Counter(name, doc, list(labelnames), registry=REGISTRY)


def _gauge(name: str, doc: str, labelnames: tuple[str, ...] = ()) -> Any:
    if not _AVAILABLE:
        return _NullMetric()
    return Gauge(name, doc, list(labelnames), registry=REGISTRY)


def _histogram(name: str, doc: str, labelnames: tuple[str, ...] = (),
               buckets: tuple[float, ...] = (
                   0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
                   2.5, 5.0, 10.0, 30.0, 60.0)) -> Any:
    if not _AVAILABLE:
        return _NullMetric()
    return Histogram(name, doc, list(labelnames),
                     buckets=list(buckets), registry=REGISTRY)


# ---- Day-1 metric catalog (per plan section 5) --------------------

ROUTED_DEFAULT = _counter(
    "genesis_routed_default_total",
    "Requests forwarded to the default upstream.",
)

ROUTED_STRUCTURED = _counter(
    "genesis_routed_structured_total",
    "Requests forwarded to the structured-k4 upstream.",
)

FALLBACK_TOTAL = _counter(
    "genesis_fallback_total",
    "Fallback events. reason: force_default / structured_down / "
    "no_artifact / router_exception / streaming / upstream_error.",
    labelnames=("reason",),
)

UPSTREAM_ERROR = _counter(
    "genesis_upstream_error_total",
    "Upstream call failures (connection refused, timeout, 5xx).",
    labelnames=("upstream", "kind"),
)

ROUTER_DECISION = _counter(
    "genesis_router_decision_total",
    "Router output. profile = artifact or default. accepted = router "
    "selected the artifact's profile (True) or fell back (False).",
    labelnames=("profile", "accepted"),
)

REQUEST_LATENCY = _histogram(
    "genesis_request_latency_seconds",
    "End-to-end gateway latency (router decision + upstream call).",
    labelnames=("upstream",),
)

FORCE_DEFAULT_ACTIVE = _gauge(
    "genesis_force_default_active",
    "1 when admin force-default override is active.",
)

UPSTREAM_HEALTH = _gauge(
    "genesis_upstream_health",
    "Health state: 1=up, 0.5=degraded, 0=down. "
    "Kept for dashboard back-compat; new dashboards should use "
    "sndr_upstream_health_state.",
    labelnames=("upstream",),
)


# ---- D2c additions (2026-05-20) -----------------------------------
#
# Canonical SNDR_ namespace for new metrics. Existing `genesis_*`
# metric names ARE NOT renamed (dashboards built on D2a/D2b
# expositions continue to work). New observability surfaces use
# `sndr_*` so operators can begin migrating dashboards on their own
# schedule.
#
# Catalog ordering matches the dashboard panel order in
# deploy/dashboards/sndr-gateway-overview.json.

ROUTE_LATENCY = _histogram(
    "sndr_route_latency_seconds",
    "Per-request gateway latency (router decision + upstream "
    "round-trip). Labels: upstream, profile (artifact's profile "
    "or 'default'), stream ('true'/'false').",
    labelnames=("upstream", "profile", "stream"),
)

STREAMING_REQUEST_TOTAL = _counter(
    "sndr_streaming_request_total",
    "Streaming chat-completion requests proxied (stream=true).",
    labelnames=("upstream",),
)

STREAMING_ERROR_TOTAL = _counter(
    "sndr_streaming_error_total",
    "Streaming proxy errors. reason in: "
    "open_failed / mid_stream / client_disconnect.",
    labelnames=("upstream", "reason"),
)

UPSTREAM_PROBE_FAILURES_TOTAL = _counter(
    "sndr_upstream_probe_failures_total",
    "Health-check probes that returned a non-2xx or raised. "
    "Distinct from upstream_error_total which counts real-traffic "
    "request failures.",
    labelnames=("upstream",),
)

UPSTREAM_HEALTH_STATE = _gauge(
    "sndr_upstream_health_state",
    "Canonical-named gauge mirroring genesis_upstream_health. "
    "1=up, 0.5=degraded, 0=down.",
    labelnames=("upstream",),
)


def render() -> tuple[bytes, str]:
    """Return (body, content_type) for /metrics endpoint."""
    if not _AVAILABLE:
        return (b"# prometheus_client not installed\n",
                "text/plain; charset=utf-8")
    return (generate_latest(REGISTRY), CONTENT_TYPE_LATEST)


__all__ = [
    # D2a/D2b (existing, unchanged)
    "ROUTED_DEFAULT", "ROUTED_STRUCTURED", "FALLBACK_TOTAL",
    "UPSTREAM_ERROR", "ROUTER_DECISION", "REQUEST_LATENCY",
    "FORCE_DEFAULT_ACTIVE", "UPSTREAM_HEALTH",
    # D2c (new SNDR-namespaced surfaces)
    "ROUTE_LATENCY", "STREAMING_REQUEST_TOTAL", "STREAMING_ERROR_TOTAL",
    "UPSTREAM_PROBE_FAILURES_TOTAL", "UPSTREAM_HEALTH_STATE",
    "render",
]
