# SPDX-License-Identifier: Apache-2.0
"""Pure-API layer for ``sndr patches pn95-status`` — M.6.3.

Reads the PN95 worker-side stats JSON, runs a set of operator-facing
self-diagnosis predicates, and returns a structured :class:`Pn95Report`.
The CLI consumes the report verbatim for both ``--json`` and human
output. Disk-tier statistics are best-effort: if the module that exports
them is unavailable, the field carries ``{"error": "..."}`` exactly as
the pre-M.6.3 CLI rendered it.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Pn95Report:
    """Outcome of a PN95-stats read.

    ``available`` is ``True`` only when the stats JSON was read +
    parsed successfully. ``reason`` carries the operator-facing message
    on the missing-file / parse-error paths; ``parse_error`` lets the
    caller distinguish the two failure classes (CLI maps to rc=1 for
    missing-file vs. rc=2 for parse error).
    """

    available: bool
    reason: str = ""
    parse_error: bool = False
    stats: dict[str, Any] = field(default_factory=dict)
    disk_tier: dict[str, Any] = field(default_factory=dict)
    hints: tuple[dict[str, str], ...] = field(default_factory=tuple)


# Predicates run on the stats dict to surface operator-visible
# anomalies. Each entry: ``(predicate, severity, message)``. Predicates
# may raise ``KeyError`` / ``TypeError`` on incomplete stats; the
# evaluator swallows those and treats the predicate as not-hitting.
PN95_STATUS_HINTS: tuple[tuple[Callable[[dict[str, Any]], bool], str, str], ...] = (
    (
        lambda s: s["ticks_total"] == 0,
        "warn",
        "Zero scheduler ticks recorded. Likely SITE5 anchor missed the "
        "vllm Scheduler.schedule() entry — re-apply via "
        "`python3 -m sndr.apply` after a fresh container boot.",
    ),
    (
        lambda s: s["ticks_pressure_check"] > 0
        and s["ticks_demote_triggered"] == 0
        and s["blocks_demoted_total"] == 0,
        "warn",
        "Pressure checks running but no demotes ever fired. Most common "
        "cause is the multiproc gap: scheduler_tick runs in EngineCore "
        "process whose _PN95_BLOCK_POOL_REFS is empty (pools live in "
        "Worker processes). The fall-through eviction-driven path "
        "(SITE7 demote-on-evict) still works on natural vllm prefix "
        "eviction. To get proactive coverage call "
        "sndr.cache._pn95_runtime.worker_side_proactive_demote "
        "from a Worker-side hook (BlockPool.get_new_blocks or similar).",
    ),
    (
        lambda s: s.get("last_free_mib", -1) >= 0
        and s.get("last_free_mib", 0) < 200,
        "warn",
        "GPU free memory below 200 MiB — kernel scratch allocations "
        "(Marlin GEMM, FlashAttention) are at risk of CUDA OOM. PN95 "
        "manages KV-cache bytes only; this looks like an activation-buffer "
        "budget issue. Lower --gpu-memory-utilization (e.g. 0.92 → 0.88) "
        "or reduce --max-num-batched-tokens.",
    ),
    (
        lambda s: s["blocks_demoted_total"] > 0 and s["prefix_store_entries"] == 0,
        "warn",
        "Demote counter incremented but prefix store is empty — the CPU "
        "slab eviction TTL may be too short, or compression is dropping "
        "entries. Inspect _PN95_PREFIX_STORE in a worker REPL.",
    ),
    (
        lambda s: s["prefix_store_promote_hits"] > 0,
        "ok",
        'Prefix store is actively serving cache hits — multi-turn '
        'workloads are benefiting from CPU offload.',
    ),
)


def _evaluate_hints(stats: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for predicate, severity, msg in PN95_STATUS_HINTS:
        try:
            hit = predicate(stats)
        except (KeyError, TypeError):
            hit = False
        if hit:
            out.append({"severity": severity, "message": msg})
    return out


def _read_disk_tier_stats() -> dict[str, Any]:
    """Best-effort fetch of PN95 disk-tier statistics.

    On any import / call failure (module missing on cold dev hosts,
    disk tier disabled at runtime), the dict carries ``{"error": "..."}``
    so the renderer can decide whether to surface the section.
    """
    try:
        from sndr.cache import _pn95_disk_tier as _dt

        return _dt.disk_tier_stats()
    except Exception as e:
        return {"error": str(e)}


def read_pn95_status(stats_path: str = "/tmp/pn95_stats.json") -> Pn95Report:
    """Read + parse the PN95 worker stats JSON and return a typed report.

    Three terminal outcomes:

      * File missing → ``available=False``, ``parse_error=False``,
        ``reason`` carries the operator hint text.
      * Parse error → ``available=False``, ``parse_error=True``,
        ``reason`` carries ``"parse error: <msg>"``.
      * Success → ``available=True``, ``stats`` / ``disk_tier`` /
        ``hints`` populated.
    """
    if not os.path.isfile(stats_path):
        msg = (
            f"PN95 stats file not found at {stats_path}. "
            "Either PN95 is not enabled in this deployment "
            "(GENESIS_ENABLE_PN95_TIER_AWARE_CACHE=1) or the worker "
            "hasn't dumped stats yet — stats land every "
            "GENESIS_PN95_STATS_INTERVAL ticks (default 100)."
        )
        return Pn95Report(available=False, reason=msg, parse_error=False)
    try:
        with open(stats_path, "r") as fh:
            stats = json.load(fh)
    except (OSError, ValueError) as e:
        return Pn95Report(
            available=False,
            reason=f"parse error: {e}",
            parse_error=True,
        )

    hints = _evaluate_hints(stats)
    disk_tier = _read_disk_tier_stats()
    return Pn95Report(
        available=True,
        stats=stats,
        disk_tier=disk_tier,
        hints=tuple(hints),
    )
