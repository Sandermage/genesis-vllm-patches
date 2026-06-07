# SPDX-License-Identifier: Apache-2.0
"""Per-container update-mode preferences (Manual / Semi-automatic / Automatic).

Stored daemon-side (not as docker labels, which can't be changed without
recreating the container) in a small JSON file under SNDR_HOME, keyed by
``<source>/<container>`` so local and per-host containers don't collide.

Modes (semantics mirror the researched Watchtower/Komodo/Portainer models):

* ``manual``    — never auto-pull or recreate. Detection still runs; the
  operator applies updates by hand. This is the safe global DEFAULT, and the
  ONLY mode allowed for critical containers (vLLM engines — pin policy forbids
  implicit image moves).
* ``semi``      — auto-pull the new image so it is local and ready, then notify;
  do NOT recreate. The operator clicks "Apply" when traffic allows (so a warm
  KV cache is never dropped mid-request). Recommended for the daemon sidecar.
* ``auto``      — pull + recreate on the daemon's schedule, health-gated. Only
  permitted for non-critical containers; blocked for engines.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

VALID_MODES = ("manual", "semi", "auto")
DEFAULT_MODE = "manual"

_lock = threading.Lock()


def _store_path() -> Path:
    home = os.environ.get("SNDR_HOME") or os.path.join(Path.home(), ".sndr")
    base = Path(home) / "state"
    base.mkdir(parents=True, exist_ok=True)
    return base / "container_update_prefs.json"


def _key(source: str, container: str) -> str:
    return f"{source}/{container}"


def _load() -> dict[str, dict]:
    try:
        return json.loads(_store_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(data: dict[str, dict]) -> None:
    path = _store_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def normalize_mode(mode: Optional[str]) -> str:
    m = (mode or "").strip().lower()
    return m if m in VALID_MODES else DEFAULT_MODE


def get_mode(source: str, container: str) -> str:
    with _lock:
        entry = _load().get(_key(source, container)) or {}
    return normalize_mode(entry.get("mode"))


def _update(source: str, container: str, **fields) -> dict:
    with _lock:
        data = _load()
        entry = dict(data.get(_key(source, container)) or {})
        entry.update(fields)
        data[_key(source, container)] = entry
        _save(data)
        return entry


def set_mode(source: str, container: str, mode: str, *, is_critical: bool = False) -> dict:
    """Persist the update mode. ``auto`` is refused for critical containers."""
    m = normalize_mode(mode)
    if m == "auto" and is_critical:
        return {"ok": False, "mode": get_mode(source, container),
                "error": "automatic updates are blocked for critical containers (e.g. vLLM engines) — use manual or semi"}
    _update(source, container, mode=m)
    return {"ok": True, "mode": m, "error": None}


def set_previous(source: str, container: str, image_id: str) -> None:
    """Record the image a container ran BEFORE an update, so we can roll back to
    it. Retained until the next successful update overwrites it (the image must
    still exist locally — don't prune between update and a possible rollback)."""
    if image_id:
        _update(source, container, previous_image=image_id)


def get_previous(source: str, container: str) -> Optional[str]:
    with _lock:
        entry = _load().get(_key(source, container)) or {}
    val = entry.get("previous_image")
    return val if isinstance(val, str) and val else None


__all__ = ["VALID_MODES", "DEFAULT_MODE", "get_mode", "set_mode",
           "set_previous", "get_previous", "normalize_mode"]
