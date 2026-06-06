# SPDX-License-Identifier: Apache-2.0
"""Container inventory service.

Wraps the local Docker daemon's REST socket to enumerate containers,
join each container's metadata with sndr-relevant signals (engine pin,
served model name, apply matrix from logs).

Falls back gracefully when docker is not available — returns an empty
list rather than raising. The GUI shows an info banner in that case.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any

from sndr.product_api.schemas.containers import (
    ContainerDetail,
    ContainerInventoryReport,
    ContainerPort,
    ContainerState,
    ContainerSummary,
)


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _docker_ps_json() -> list[dict[str, Any]]:
    """Run ``docker ps -a --format '{{json .}}'`` and parse one JSON per line."""
    if not _docker_available():
        return []
    try:
        out = subprocess.run(
            ["docker", "ps", "-a", "--no-trunc", "--format", "{{json .}}"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return []
        rows = []
        for line in out.stdout.strip().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows
    except subprocess.TimeoutExpired:
        return []


def _docker_inspect(container_id: str) -> dict[str, Any] | None:
    """Inspect a single container (full metadata)."""
    if not _docker_available():
        return None
    try:
        out = subprocess.run(
            ["docker", "inspect", container_id],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        rows = json.loads(out.stdout)
        return rows[0] if rows else None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, IndexError):
        return None


def _parse_state(raw: str) -> ContainerState:
    raw_l = raw.lower()
    for s in ("running", "paused", "exited", "restarting", "created", "dead"):
        if raw_l.startswith(s):
            return s  # type: ignore[return-value]
    return "unknown"


def _parse_ports(raw: str) -> list[ContainerPort]:
    """Parse docker ps Ports field, e.g. '0.0.0.0:8102->8102/tcp'."""
    if not raw:
        return []
    out = []
    for part in raw.split(", "):
        m = re.search(r"(?:0\.0\.0\.0|\[::\]):(\d+)->(\d+)/(\w+)", part)
        if m:
            try:
                out.append(ContainerPort(
                    host_port=int(m.group(1)),
                    container_port=int(m.group(2)),
                    protocol=m.group(3),  # type: ignore[arg-type]
                ))
            except (ValueError, TypeError):
                continue
    return out


def _parse_iso(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _extract_engine_signals(env_pairs: list[str]) -> dict[str, str | None]:
    """Pull SNDR_ENGINE, SNDR_ENGINE_PIN, served model name from env list."""
    out: dict[str, str | None] = {
        "engine": None,
        "engine_pin": None,
        "served_model": None,
    }
    for entry in env_pairs:
        k, _, v = entry.partition("=")
        if k == "SNDR_ENGINE":
            out["engine"] = v
        elif k == "SNDR_ENGINE_PIN":
            out["engine_pin"] = v
        elif k == "VLLM_SERVED_MODEL_NAME":
            out["served_model"] = v
    return out


def _summarize_apply_log(name: str) -> dict[str, int]:
    """Count applied/skipped/failed/unresolved markers in container logs."""
    if not _docker_available():
        return {}
    try:
        out = subprocess.run(
            ["docker", "logs", name, "--tail", "5000"],
            capture_output=True, text=True, timeout=10,
        )
        text = (out.stdout or "") + (out.stderr or "")
        return {
            "applied": text.count("[Genesis] applied:"),
            "skipped": text.count("[Genesis] skipped:"),
            "failed": text.count("[Genesis] FAILED:"),
            "unresolved": text.count("UNRESOLVED"),
        }
    except subprocess.TimeoutExpired:
        return {}


def list_containers(*, engine: str | None = None) -> list[ContainerSummary]:
    """Return all containers (optionally filtered by engine)."""
    out: list[ContainerSummary] = []
    for row in _docker_ps_json():
        try:
            inspect = _docker_inspect(row.get("ID", "")) or {}
            cfg = inspect.get("Config", {}) or {}
            envs = cfg.get("Env", []) or []
            signals = _extract_engine_signals(envs)

            summary = ContainerSummary(
                name=row.get("Names", "").lstrip("/"),
                container_id=row.get("ID", "")[:12],
                image=row.get("Image", ""),
                image_digest=inspect.get("Image"),
                state=_parse_state(row.get("State", "")),
                status=row.get("Status", ""),
                created_at=_parse_iso(inspect.get("Created", "")),
                started_at=_parse_iso(inspect.get("State", {}).get("StartedAt")),
                finished_at=_parse_iso(inspect.get("State", {}).get("FinishedAt"))
                            if inspect.get("State", {}).get("FinishedAt") not in ("", "0001-01-01T00:00:00Z") else None,
                served_model_name=signals.get("served_model"),
                engine=signals.get("engine"),
                engine_pin=signals.get("engine_pin"),
                ports=_parse_ports(row.get("Ports", "")),
            )
            if engine and summary.engine != engine:
                continue
            out.append(summary)
        except Exception:
            # Defensive: bad container metadata shouldn't kill the listing.
            continue
    return out


def get_container_detail(name: str) -> ContainerDetail | None:
    """Return full detail for one container by name."""
    inspect = _docker_inspect(name)
    if inspect is None:
        return None
    summaries = [s for s in list_containers() if s.name == name]
    if not summaries:
        return None
    summary = summaries[0]
    cfg = inspect.get("Config", {}) or {}
    return ContainerDetail(
        **summary.model_dump(),
        cmd=cfg.get("Cmd") or [],
        env={(k_v.split("=", 1) + [""])[0]: (k_v.split("=", 1) + [""])[1]
              for k_v in cfg.get("Env", []) or []},
        mounts={m.get("Source", ""): m.get("Destination", "")
                for m in inspect.get("Mounts", []) or []},
        labels=cfg.get("Labels", {}) or {},
        sndr_apply_summary=_summarize_apply_log(name),
    )


def inventory_report() -> ContainerInventoryReport:
    """Aggregate counts useful for the GUI summary card."""
    summaries = list_containers()
    by_state: dict[str, int] = {}
    by_engine: dict[str, int] = {}
    for s in summaries:
        by_state[s.state] = by_state.get(s.state, 0) + 1
        if s.engine:
            by_engine[s.engine] = by_engine.get(s.engine, 0) + 1
    return ContainerInventoryReport(
        total=len(summaries),
        by_state=by_state,
        by_engine=by_engine,
    )


__all__ = [
    "get_container_detail",
    "inventory_report",
    "list_containers",
]
