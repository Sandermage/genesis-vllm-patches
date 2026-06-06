# SPDX-License-Identifier: Apache-2.0
"""Operator-local patch enablement overrides.

Lets an operator force a patch on/off from the GUI without editing the registry
(patches are code). Overrides persist under ``$SNDR_HOME/patch_overrides.json``
and are **consumed** by the launch plan: each override emits a
``GENESIS_ENABLE_<flag>=1|0`` line into the generated launch env, so the launch
command the operator runs reflects the choice. Strictly validated (no arbitrary
shell/env injection)."""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

_LOCK = threading.RLock()
_PATCH_ID_RE = re.compile(r"^[A-Za-z0-9_]{1,32}$")
_ENV_FLAG_RE = re.compile(r"^GENESIS_[A-Z0-9_]{1,80}$")
VALID_STATES = ("on", "off", "default")


def _path() -> Path:
    home = os.environ.get("SNDR_HOME") or os.environ.get("GENESIS_HOME")
    base = Path(home).expanduser() if home else (Path.home() / ".sndr")
    return base / "patch_overrides.json"


def load() -> dict[str, dict]:
    """Return ``{patch_id: {"state": "on"|"off", "env_flag": str}}``."""
    with _LOCK:
        try:
            data = json.loads(_path().read_text("utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}


def _save(data: dict) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), "utf-8")
    os.replace(tmp, path)


def set_override(patch_id: str, state: str, env_flag: str) -> dict[str, dict]:
    """Force a patch on/off (or clear with ``default``). Returns the new map."""
    if not _PATCH_ID_RE.match(patch_id or ""):
        raise ValueError("Invalid patch id.")
    if state not in VALID_STATES:
        raise ValueError(f"state must be one of {VALID_STATES}.")
    if state != "default" and not _ENV_FLAG_RE.match(env_flag or ""):
        raise ValueError("Invalid env flag.")
    with _LOCK:
        data = load()
        if state == "default":
            data.pop(patch_id, None)
        else:
            data[patch_id] = {"state": state, "env_flag": env_flag}
        _save(data)
        return data


def env_lines() -> list[str]:
    """Render overrides as ``GENESIS_ENABLE_<flag>=1|0`` env lines."""
    lines: list[str] = []
    for entry in load().values():
        flag = entry.get("env_flag")
        if not _ENV_FLAG_RE.match(flag or ""):
            continue
        lines.append(f"{flag}={'1' if entry.get('state') == 'on' else '0'}")
    return sorted(lines)


__all__ = ["load", "set_override", "env_lines", "VALID_STATES"]
