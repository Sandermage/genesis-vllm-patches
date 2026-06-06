# SPDX-License-Identifier: Apache-2.0
"""Operator-local host profile persistence for the GUI.

Stores remote/local host metadata (SSH target, port, notes) as JSON under the
operator-local GUI state dir. No SSH is opened and no command is run — the GUI
shows a copyable tunnel command; execution stays a separate explicit step.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HostProfile:
    id: str
    label: str
    host: str
    transport: str  # "local" | "ssh"
    ssh_target: str
    port: int
    notes: str
    role: str = ""  # "production" | "staging" | "dev" | "" (free-form)
    hardware: str = ""  # e.g. "2x A5000 24GB"
    gpus: int = 0
    engine_port: int = 8000  # vLLM OpenAI port to probe for reachability
    # Engine bearer for a key-protected engine. Like the SSH password, the value
    # is NOT persisted here — it lives encrypted in the secrets store keyed
    # ``apikey:<id>``; the payload exposes only a boolean ``has_api_key``. The
    # field remains for legacy rows (migrated on first read) and transient input.
    api_key: str = ""
    ssh_user: str = ""  # SSH login user (falls back to ssh_target's user@ part)
    ssh_auth: str = "agent"  # "agent" | "key" | "password"
    ssh_key_path: str = ""  # private key path for ssh_auth="key"
    ssh_port: int = 22
    # NB: the SSH password itself is never stored here — it lives encrypted in
    # the secrets store keyed by ``ssh:<id>``. The payload exposes only a
    # boolean ``has_ssh_password``.
    # Discovered hardware summary (set by /hosts/discover) — the single source
    # the Planner reads real VRAM / arch from.
    gpu_vram_mib: int = 0
    gpu_name: str = ""
    gpu_arch: str = ""
    interconnect: str = ""
    tags: tuple[str, ...] = ()


def _state_dir() -> Path:
    from vllm.sndr_core.locations.project_paths import install_root

    return install_root() / "gui"


def _hosts_path() -> Path:
    return _state_dir() / "hosts.json"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:64] or "host"


def _read() -> list[dict[str, Any]]:
    path = _hosts_path()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _write(rows: list[dict[str, Any]]) -> None:
    path = _hosts_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    tmp.replace(path)


def _clean_tags(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        items = [str(part).strip() for part in value]
    else:
        items = []
    seen: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.append(item)
    return tuple(seen[:12])


def _to_profile(row: dict[str, Any]) -> HostProfile:
    return HostProfile(
        id=str(row.get("id", "")),
        label=str(row.get("label", "")),
        host=str(row.get("host", "")),
        transport=str(row.get("transport", "ssh")),
        ssh_target=str(row.get("ssh_target", "")),
        port=int(row.get("port", 8765) or 8765),
        notes=str(row.get("notes", "")),
        role=str(row.get("role", "") or ""),
        hardware=str(row.get("hardware", "") or ""),
        gpus=int(row.get("gpus", 0) or 0),
        engine_port=int(row.get("engine_port", 8000) or 8000),
        api_key=str(row.get("api_key", "") or ""),
        ssh_user=str(row.get("ssh_user", "") or ""),
        ssh_auth=str(row.get("ssh_auth", "agent") or "agent"),
        ssh_key_path=str(row.get("ssh_key_path", "") or ""),
        ssh_port=int(row.get("ssh_port", 22) or 22),
        gpu_vram_mib=int(row.get("gpu_vram_mib", 0) or 0),
        gpu_name=str(row.get("gpu_name", "") or ""),
        gpu_arch=str(row.get("gpu_arch", "") or ""),
        interconnect=str(row.get("interconnect", "") or ""),
        tags=_clean_tags(row.get("tags")),
    )


def list_host_profiles() -> tuple[HostProfile, ...]:
    return tuple(_to_profile(row) for row in _read())


def upsert_host_profile(profile: dict[str, Any]) -> HostProfile:
    label = str(profile.get("label") or "").strip()
    host = str(profile.get("host") or "").strip()
    explicit_id = str(profile.get("id") or "").strip()
    if not (explicit_id or label or host):
        raise ValueError("Host profile needs at least a label or host.")
    profile_id = explicit_id or _slug(label or host)

    record = {
        "id": profile_id,
        "label": label or host or profile_id,
        "host": host or label or profile_id,
        "transport": str(profile.get("transport") or "ssh"),
        "ssh_target": str(profile.get("ssh_target") or ""),
        "port": int(profile.get("port") or 8765),
        "notes": str(profile.get("notes") or ""),
        "role": str(profile.get("role") or ""),
        "hardware": str(profile.get("hardware") or ""),
        "gpus": int(profile.get("gpus") or 0),
        "engine_port": int(profile.get("engine_port") or 8000),
        # NB: api_key is intentionally NOT persisted to disk — the caller routes
        # it into the encrypted secrets store (see http_app host endpoints).
        "ssh_user": str(profile.get("ssh_user") or ""),
        "ssh_auth": str(profile.get("ssh_auth") or "agent"),
        "ssh_key_path": str(profile.get("ssh_key_path") or ""),
        "ssh_port": int(profile.get("ssh_port") or 22),
        "gpu_vram_mib": int(profile.get("gpu_vram_mib") or 0),
        "gpu_name": str(profile.get("gpu_name") or ""),
        "gpu_arch": str(profile.get("gpu_arch") or ""),
        "interconnect": str(profile.get("interconnect") or ""),
        "tags": list(_clean_tags(profile.get("tags"))),
    }
    rows = [row for row in _read() if str(row.get("id")) != profile_id]
    rows.append(record)
    rows.sort(key=lambda row: str(row.get("label", "")).lower())
    _write(rows)
    return _to_profile(record)


def delete_host_profile(profile_id: str) -> bool:
    rows = _read()
    remaining = [row for row in rows if str(row.get("id")) != profile_id]
    if len(remaining) == len(rows):
        return False
    _write(remaining)
    return True


def host_profile_payload(profile: HostProfile) -> dict[str, Any]:
    data = asdict(profile)
    # Never expose the raw engine key in API responses — the HTTP layer adds a
    # boolean ``has_api_key`` instead (mirrors how the SSH password is handled).
    data.pop("api_key", None)
    return data


__all__ = [
    "HostProfile",
    "delete_host_profile",
    "host_profile_payload",
    "list_host_profiles",
    "upsert_host_profile",
]
