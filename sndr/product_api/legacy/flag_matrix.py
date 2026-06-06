# SPDX-License-Identifier: Apache-2.0
"""Env-flag matrix — the full GENESIS_ENABLE_* catalogue with defaults and,
optionally, the live state on a running engine.

The Patches section lists patches; this is the flag-centric companion an operator
needs to answer "which flags exist, what are their defaults, and which are
actually ON on this engine right now — matching the registry or drifting?".

Pure functions over ``patches.list_patches()`` (the registry) plus an optional
overlay of the live flags read from a container's env (via container_link). Read
only; no mutation.
"""
from __future__ import annotations

from typing import Any, Optional


def _effective_default(row: Any) -> bool:
    """The default that actually applies: production_default wins when set,
    otherwise the registry default_on."""
    prod = getattr(row, "production_default", None)
    if isinstance(prod, bool):
        return prod
    return bool(getattr(row, "default_on", False))


def build_matrix(live_flags: Optional[set[str]] = None) -> dict[str, Any]:
    """Flag rows from the registry, optionally overlaid with live engine state.

    ``live_flags`` is the set of GENESIS_* env var names actually ON in the
    running container (from container_link.live_patches). When provided, each row
    gains ``live_on`` and a ``drift`` verdict:

      in_sync — default-on AND live-on, or default-off AND live-off
      missing — default-on BUT live-off (a silent feature regression)
      extra   — default-off BUT live-on (enabled beyond the registry default)
    """
    from . import patches

    rows: list[dict[str, Any]] = []
    counts = {"total": 0, "default_on": 0, "default_off": 0, "missing": 0, "extra": 0}
    for row in patches.list_patches():
        flag = getattr(row, "env_flag", None)
        if not flag:
            continue
        default = _effective_default(row)
        entry: dict[str, Any] = {
            "env_flag": flag,
            "patch_id": getattr(row, "patch_id", None),
            "title": getattr(row, "title", None),
            "family": getattr(row, "family", None),
            "tier": getattr(row, "tier", None),
            "lifecycle": getattr(row, "lifecycle", None),
            "default_on": default,
        }
        counts["total"] += 1
        counts["default_on" if default else "default_off"] += 1
        if live_flags is not None:
            live_on = flag in live_flags
            entry["live_on"] = live_on
            if default and not live_on:
                entry["drift"] = "missing"
                counts["missing"] += 1
            elif not default and live_on:
                entry["drift"] = "extra"
                counts["extra"] += 1
            else:
                entry["drift"] = "in_sync"
        rows.append(entry)

    rows.sort(key=lambda r: (r["family"] or "", r["env_flag"]))
    return {"flags": rows, "counts": counts, "has_live": live_flags is not None}


def live_flags_from_inspect(inspect: dict[str, Any]) -> set[str]:
    """The set of GENESIS_* flags ON in a container's env (delegates to
    container_link so the parse stays in one place)."""
    from . import container_link

    return {f["flag"] for f in container_link.live_patches(inspect)}
