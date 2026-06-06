# SPDX-License-Identifier: Apache-2.0
"""PN282 — Spec-decode acceptance proxy metrics.

Production observability sibling of PN248 (which writes a heavyweight
log trace at /tmp/genesis_pn248_acceptance_trace.log).

Exposes three Prometheus series on the vllm worker's existing /metrics
endpoint (via the default ``prometheus_client.REGISTRY`` that vllm
already serves):

  sndr_spec_decode_accepted_per_call_total{k, profile}   Counter
  sndr_spec_decode_calls_total{profile}                  Counter
  sndr_spec_decode_max_spec_len{profile}                 Gauge

``k`` is the number of draft tokens accepted on a single rejection_sample
call for a single request. With K=4 spec-decode the value range is
{0,1,2,3,4}. Each request in the batch contributes one increment to the
``accepted_per_call_total`` series under its own ``k`` value; the
``calls_total`` series increments once per rejection_sample call (not
per request).

Design constraints:

  * Default OFF — gated by ``SNDR_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC=1``
    (canonical) or legacy ``GENESIS_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC=1``
    (warns once on first read).
  * Lazy init — counter objects are created on first ``record_acceptance``
    call so the registry stays clean when the metric is disabled.
  * ``prometheus_client`` is a hard vllm dep; if the import fails the
    module degrades to no-op (still importable, never raises).
  * ``profile`` label resolved once at first-use from
    ``SNDR_SPEC_DECODE_PROFILE_LABEL`` (default ``"unknown"``); not
    per-request.
  * No file I/O, no thread locks beyond what ``prometheus_client``
    already provides on the counter primitives.
  * No torch import — safe to import in test contexts without GPU.

Coexists with PN248: PN248 wraps ``rejection_sample`` for the log trace,
PN282 wraps the same function for the metric. The two wraps stack
independently; each carries its own idempotency marker.

Author: Sandermage; PN282 / 2026-05-20.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger("genesis.spec_decode_metrics")

_CANONICAL_ENV = "SNDR_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC"
_LEGACY_ENV = "GENESIS_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC"
_PROFILE_ENV = "SNDR_SPEC_DECODE_PROFILE_LABEL"
_DEFAULT_PROFILE = "unknown"

_ACCEPTED_COUNTER = None
_CALLS_COUNTER = None
_MAX_SPEC_LEN_GAUGE = None
_PROFILE_LABEL: Optional[str] = None
_LEGACY_WARNED = False


def _is_enabled() -> bool:
    raw = os.environ.get(_CANONICAL_ENV, "").strip().lower()
    if raw in ("1", "true", "yes", "y", "on"):
        return True
    legacy = os.environ.get(_LEGACY_ENV, "").strip().lower()
    if legacy in ("1", "true", "yes", "y", "on"):
        global _LEGACY_WARNED
        if not _LEGACY_WARNED:
            log.warning(
                "[PN282] %s is deprecated — use %s instead "
                "(legacy alias accepted for 1 release).",
                _LEGACY_ENV, _CANONICAL_ENV,
            )
            _LEGACY_WARNED = True
        return True
    return False


def _profile_label() -> str:
    global _PROFILE_LABEL
    if _PROFILE_LABEL is None:
        _PROFILE_LABEL = os.environ.get(
            _PROFILE_ENV, _DEFAULT_PROFILE,
        ).strip() or _DEFAULT_PROFILE
    return _PROFILE_LABEL


def _try_init_counters() -> bool:
    """Lazy-create the Prometheus primitives on first use.

    Returns True if the metric surface is live; False if
    ``prometheus_client`` is unavailable (degraded mode — record calls
    silently no-op)."""
    global _ACCEPTED_COUNTER, _CALLS_COUNTER, _MAX_SPEC_LEN_GAUGE
    if _ACCEPTED_COUNTER is not None:
        return True
    try:
        from prometheus_client import Counter, Gauge
    except Exception as e:  # noqa: BLE001
        log.warning(
            "[PN282] prometheus_client unavailable (%s) — metric "
            "module degraded to no-op.", type(e).__name__,
        )
        return False
    try:
        _ACCEPTED_COUNTER = Counter(
            "sndr_spec_decode_accepted_per_call_total",
            "Number of rejection_sample request-outcomes bucketed by "
            "accepted draft token count k.",
            labelnames=("k", "profile"),
        )
        _CALLS_COUNTER = Counter(
            "sndr_spec_decode_calls_total",
            "Number of rejection_sample calls observed.",
            labelnames=("profile",),
        )
        _MAX_SPEC_LEN_GAUGE = Gauge(
            "sndr_spec_decode_max_spec_len",
            "Most recently observed max_spec_len passed to "
            "rejection_sample.",
            labelnames=("profile",),
        )
    except Exception as e:  # noqa: BLE001
        # Most likely a duplicate registration on hot-reload — re-fetch
        # from the global registry instead of failing.
        log.warning(
            "[PN282] counter init raised %s: %s — attempting registry "
            "rebind.", type(e).__name__, e,
        )
        try:
            from prometheus_client import REGISTRY
            for collector in list(REGISTRY._collector_to_names):  # type: ignore[attr-defined]
                names = REGISTRY._collector_to_names[collector]  # type: ignore[attr-defined]
                if "sndr_spec_decode_accepted_per_call_total" in names:
                    _ACCEPTED_COUNTER = collector
                if "sndr_spec_decode_calls_total" in names:
                    _CALLS_COUNTER = collector
                if "sndr_spec_decode_max_spec_len" in names:
                    _MAX_SPEC_LEN_GAUGE = collector
        except Exception:  # noqa: BLE001
            return False
        if _ACCEPTED_COUNTER is None:
            return False
    return True


def record_acceptance(
    accepted_per_req: list[int],
    max_spec_len: int,
) -> None:
    """Record one rejection_sample call's outcome.

    Args:
        accepted_per_req: list of length B (batch size) where each entry
            is the count of draft tokens accepted for that request on
            this call. Range per entry: {0, 1, ..., max_spec_len}.
        max_spec_len: K parameter passed to rejection_sample. Used to
            update the max_spec_len gauge for dashboard awareness.

    No-op when ``SNDR_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC`` is unset or
    when ``prometheus_client`` is not importable. Safe to call from any
    thread; counter primitives are internally locked by
    ``prometheus_client``.
    """
    if not _is_enabled():
        return
    if not _try_init_counters():
        return
    profile = _profile_label()
    try:
        for accepted in accepted_per_req:
            _ACCEPTED_COUNTER.labels(  # type: ignore[union-attr]
                k=str(int(accepted)),
                profile=profile,
            ).inc()
        _CALLS_COUNTER.labels(profile=profile).inc()  # type: ignore[union-attr]
        _MAX_SPEC_LEN_GAUGE.labels(profile=profile).set(  # type: ignore[union-attr]
            float(max_spec_len)
        )
    except Exception as e:  # noqa: BLE001
        # Counter primitives must never break the sample loop.
        log.debug(
            "[PN282] record_acceptance suppressed exception: %s",
            e,
        )


def is_enabled() -> bool:
    """Public predicate — true when the operator opted in via env."""
    return _is_enabled()


def get_profile_label() -> str:
    """Public accessor — resolved profile label string (frozen after
    first use)."""
    return _profile_label()


def _reset_module_state() -> None:
    """Test-only: drop counter handles + profile label cache.

    The Prometheus registry retains the collectors across resets (its
    own internal state is process-global). Tests that need a clean
    registry should use a fresh ``CollectorRegistry`` instance instead
    of relying on this hook.
    """
    global _ACCEPTED_COUNTER, _CALLS_COUNTER, _MAX_SPEC_LEN_GAUGE
    global _PROFILE_LABEL, _LEGACY_WARNED
    _ACCEPTED_COUNTER = None
    _CALLS_COUNTER = None
    _MAX_SPEC_LEN_GAUGE = None
    _PROFILE_LABEL = None
    _LEGACY_WARNED = False


__all__ = [
    "record_acceptance",
    "is_enabled",
    "get_profile_label",
]
