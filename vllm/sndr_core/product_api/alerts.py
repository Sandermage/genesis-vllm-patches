# SPDX-License-Identifier: Apache-2.0
"""In-GUI alert center — threshold rules over the hardware telemetry the daemon
already collects, with a small deduplicating store.

Each evaluation turns the current telemetry snapshot into a flat list of alerts
keyed by a stable ``key`` (host + subject + metric). The :class:`AlertStore`
merges successive evaluations: an already-firing alert keeps its ``first_seen``
(so the GUI can show "for how long"), a condition that clears moves to a bounded
``recent`` list as resolved. Read-only; no mutation of the host.

This is the hardware counterpart to :mod:`container_watch` (which watches engine
containers and pushes UP/DOWN to external notify channels) — here the surface is
the GUI itself.
"""
from __future__ import annotations

from typing import Any, Optional

_LEVEL_ORDER = {"critical": 0, "warn": 1, "info": 2, "ok": 3}

# Thresholds (kept here so a single edit retunes the whole alert surface).
TEMP_CRIT = 87        # °C
TEMP_WARN = 80
VRAM_WARN = 0.97      # fraction of total
DISK_CRIT = 92        # % used
DISK_WARN = 85
RAM_WARN = 95         # % used


def _mk(host: str, subject: str, metric: str, level: str, category: str, title: str, detail: str) -> dict[str, Any]:
    return {
        "key": f"{host}:{subject}:{metric}",
        "level": level,
        "category": category,
        "title": title,
        "detail": detail,
        "host": host,
    }


def _int(v: Any) -> Optional[int]:
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def evaluate_hardware(host: str, telemetry: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn a hardware-telemetry payload into a list of firing alerts."""
    out: list[dict[str, Any]] = []
    for i, g in enumerate(telemetry.get("gpus") or ()):
        name = g.get("name") or f"GPU {i}"
        tag = f"GPU{i}"

        temp = g.get("temp_gpu")
        if isinstance(temp, (int, float)):
            if temp >= TEMP_CRIT:
                out.append(_mk(host, f"gpu{i}", "temp", "critical", "gpu", f"{tag} overheating", f"{name}: {int(temp)} °C"))
            elif temp >= TEMP_WARN:
                out.append(_mk(host, f"gpu{i}", "temp", "warn", "gpu", f"{tag} running hot", f"{name}: {int(temp)} °C"))

        ecc = _int(g.get("ecc_uncorrected"))
        if ecc and ecc > 0:
            out.append(_mk(host, f"gpu{i}", "ecc", "critical", "gpu", f"{tag} uncorrectable ECC errors", f"{name}: {ecc} uncorrected"))

        used, total = g.get("mem_used"), g.get("mem_total")
        if isinstance(used, (int, float)) and isinstance(total, (int, float)) and total > 0:
            frac = used / total
            if frac >= VRAM_WARN:
                out.append(_mk(host, f"gpu{i}", "vram", "warn", "gpu", f"{tag} VRAM near full",
                               f"{name}: {round(frac * 100)}% used"))

    sysd = telemetry.get("system") or {}
    disk = sysd.get("disk")
    if isinstance(disk, dict) and disk.get("used_pct") is not None:
        up = disk["used_pct"]
        mount = disk.get("mount", "/")
        free = disk.get("free_gb")
        detail = f"{mount}: {round(up)}% used" + (f" · {free} GB free" if free is not None else "")
        if up >= DISK_CRIT:
            out.append(_mk(host, "disk", mount, "critical", "disk", "Disk almost full", detail))
        elif up >= DISK_WARN:
            out.append(_mk(host, "disk", mount, "warn", "disk", "Disk filling up", detail))

    rt, ru = sysd.get("ram_total_gb"), sysd.get("ram_used_gb")
    if isinstance(rt, (int, float)) and isinstance(ru, (int, float)) and rt > 0:
        up = ru / rt * 100
        if up >= RAM_WARN:
            out.append(_mk(host, "ram", "used", "warn", "host", "System memory pressure", f"{round(up)}% RAM used"))

    return out


class AlertStore:
    """Deduplicating store: merges successive evaluations, tracks first/last seen."""

    def __init__(self, cap: int = 50) -> None:
        self._active: dict[str, dict[str, Any]] = {}
        self._recent: list[dict[str, Any]] = []
        self._cap = cap

    def update(self, fired: list[dict[str, Any]], *, now: float) -> None:
        seen: set[str] = set()
        for a in fired:
            key = a["key"]
            seen.add(key)
            existing = self._active.get(key)
            if existing is not None:
                existing.update(level=a["level"], title=a["title"], detail=a["detail"], last_seen=now)
            else:
                self._active[key] = {**a, "first_seen": now, "last_seen": now}
        for key in list(self._active):
            if key not in seen:
                resolved = self._active.pop(key)
                resolved["resolved_at"] = now
                self._recent.insert(0, resolved)
        del self._recent[self._cap:]

    def snapshot(self) -> dict[str, Any]:
        active = sorted(self._active.values(), key=lambda a: (_LEVEL_ORDER.get(a["level"], 9), a["first_seen"]))
        counts = {"critical": 0, "warn": 0, "info": 0}
        for a in active:
            counts[a["level"]] = counts.get(a["level"], 0) + 1
        return {"active": active, "recent": self._recent[:20], "counts": counts}


# Module-level singleton — the GUI polls one shared store.
STORE = AlertStore()
