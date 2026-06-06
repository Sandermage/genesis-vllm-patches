# SPDX-License-Identifier: Apache-2.0
"""Per-patch apply() timing + memory-delta instrumentation (Wave 7).

Why it exists
─────────────
Wave 6 closure (PN16 V1 regression) showed that text-patches can ship
with hidden behavioral side effects (CUDA graph dispatch mismatch,
MTP draft divergence) that are invisible to apply-time logging because
``apply()`` returns a successful status before the regression manifests.

This module adds a thin instrumentation layer that captures, per-patch:

  • ``elapsed_ms``   — wall time spent inside the patch's apply()
  • ``rss_delta_kb`` — process resident-set delta (host RAM, not VRAM)
  • ``status``       — applied / skipped / failed
  • ``reason``       — short summary string from apply()

Stored in a process-local list (``_METRICS``) for post-mortem analysis
via ``get_apply_metrics()``. Also emitted as one structured log line
per patch under the ``genesis.observability`` logger so operators can
``grep`` boot logs.

The collected metrics enable downstream tooling:
  • ``sndr report bundle`` — embed the metric list in support bundles
  • Regression bench harness (Wave 7 §2.2) — assert no patch's
    elapsed_ms exceeds its baseline by more than X%
  • ``sndr patches metrics`` CLI — table view of slowest applies

Opt-in via ``GENESIS_OBSERVABILITY=1`` to keep the default boot path
allocation-free.

Author: Sandermage(Sander) Barzov Aleksandr; Wave 7 2026-05-09.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

log = logging.getLogger("genesis.observability")


@dataclass
class PatchApplyMetric:
    """Captured metrics for a single patch's apply() call."""
    name: str
    status: str = "unknown"
    reason: str = ""
    elapsed_ms: float = 0.0
    rss_delta_kb: int = 0
    ordinal: int = -1
    extras: dict = field(default_factory=dict)


_METRICS: list[PatchApplyMetric] = []
_NEXT_ORDINAL: int = 0


def _is_enabled() -> bool:
    val = os.environ.get("GENESIS_OBSERVABILITY", "").strip().lower()
    return val in ("1", "true", "yes", "y", "on")


def _read_rss_kb() -> int:
    """Read this process's resident-set size (KB). Returns 0 on
    platforms without ``/proc/self/status`` (mac dev). Kept defensive
    so the instrumentation never crashes the apply loop."""
    try:
        with open("/proc/self/status", "rb") as f:
            for line in f:
                if line.startswith(b"VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except Exception:
        pass
    # mac fallback via resource module — returns RSS in kilobytes on
    # macOS (bytes on linux; ru_maxrss linux semantics differ but we
    # only use this when /proc isn't readable, e.g., during dev tests).
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return 0


@contextmanager
def measure_patch_apply(name: str) -> Iterator[PatchApplyMetric]:
    """Context manager that records ``elapsed_ms`` + ``rss_delta_kb``
    for the wrapped block. Caller is expected to set
    ``metric.status`` / ``metric.reason`` from the apply() result.

    Usage::

        with measure_patch_apply("PN16") as m:
            status, reason = pn16.apply()
            m.status, m.reason = status, reason

    When ``GENESIS_OBSERVABILITY`` is unset, this is a no-op
    pass-through (the metric is still yielded but never stored or
    logged), preserving the default boot path's zero-overhead posture.
    """
    global _NEXT_ORDINAL
    metric = PatchApplyMetric(name=name)

    enabled = _is_enabled()
    if not enabled:
        # Yield a no-op metric so callers don't need to branch.
        yield metric
        return

    rss_before = _read_rss_kb()
    t0 = time.perf_counter_ns()
    try:
        yield metric
    finally:
        elapsed_ns = time.perf_counter_ns() - t0
        rss_after = _read_rss_kb()
        metric.elapsed_ms = elapsed_ns / 1_000_000.0
        metric.rss_delta_kb = max(0, rss_after - rss_before)
        metric.ordinal = _NEXT_ORDINAL
        _NEXT_ORDINAL += 1
        _METRICS.append(metric)
        log.info(
            "[PatchMetrics] %s status=%s elapsed_ms=%.2f rss_delta_kb=%d "
            "ordinal=%d reason=%s",
            metric.name, metric.status, metric.elapsed_ms,
            metric.rss_delta_kb, metric.ordinal, metric.reason[:120],
        )


def get_apply_metrics() -> list[PatchApplyMetric]:
    """Return a copy of the per-patch apply metrics in apply order.

    Returns empty list when observability is disabled OR when no
    patches have been applied yet."""
    return list(_METRICS)


def reset_apply_metrics() -> None:
    """Clear the metrics buffer + reset ordinals (for tests / repeat
    boots within the same process)."""
    global _NEXT_ORDINAL
    _METRICS.clear()
    _NEXT_ORDINAL = 0


def metrics_summary() -> dict:
    """Aggregate summary suitable for embedding in
    ``sndr report bundle`` JSON. Returns empty dict when no metrics."""
    if not _METRICS:
        return {}
    total_elapsed = sum(m.elapsed_ms for m in _METRICS)
    applied = [m for m in _METRICS if m.status == "applied"]
    skipped = [m for m in _METRICS if m.status == "skipped"]
    failed = [m for m in _METRICS if m.status == "failed"]
    by_elapsed = sorted(_METRICS, key=lambda m: m.elapsed_ms, reverse=True)
    return {
        "count": len(_METRICS),
        "applied": len(applied),
        "skipped": len(skipped),
        "failed": len(failed),
        "total_elapsed_ms": round(total_elapsed, 2),
        "slowest_3": [
            {
                "name": m.name,
                "elapsed_ms": round(m.elapsed_ms, 2),
                "status": m.status,
            }
            for m in by_elapsed[:3]
        ],
        "highest_rss_3": [
            {
                "name": m.name,
                "rss_delta_kb": m.rss_delta_kb,
                "status": m.status,
            }
            for m in sorted(
                _METRICS, key=lambda m: m.rss_delta_kb, reverse=True,
            )[:3]
        ],
    }
