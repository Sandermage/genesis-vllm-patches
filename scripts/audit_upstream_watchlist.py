#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit `docs/upstream/UPSTREAM_WATCHLIST.yaml` schema + categorise entries.

Etap 5.1 (audit 2026-05-12): previously the watchlist YAML claimed to be
read by `tools/check_upstream_drift.py`, but that script only inspected
text-patch anchors. The watchlist was effectively a docs-only file —
new entries (e.g. `vllm#42102`) were invisible to `make audit-upstream`.

This script closes the gap:

  • Loads the YAML, validates schema (allowed action/status/upstream
    format, required fields, sentinel).
  • Categorises entries:
      PORT_CANDIDATE — action=port AND status=merged
                       (upstream landed, our backport is due)
      RETIRE_CANDIDATE — action=retire (regardless of status —
                         operator must check marker presence)
      WATCH          — status=open OR action in {watch, drift-check}
      DONE           — status=closed AND action not actionable
  • Emits a tabular report (or JSON via --json).
  • Exit codes: 0 = clean / 1 = PORT_CANDIDATE present / 2 = schema error.

Wired into Makefile via `make audit-upstream-watchlist` and folded
into `make audit-upstream(-offline)` aggregates.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
WATCHLIST_PATH = REPO_ROOT / "docs" / "upstream" / "UPSTREAM_WATCHLIST.yaml"

_ALLOWED_STATUSES = {"open", "merged", "closed"}
_ALLOWED_ACTIONS = {
    "port", "watch", "retire", "a-b-test", "drift-check", "cookbook",
}
_UPSTREAM_RE = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?#\d+$")


def _load_yaml() -> dict:
    try:
        import yaml
    except ImportError:
        print("ERROR: pyyaml not installed (`pip install pyyaml`)",
              file=sys.stderr)
        sys.exit(2)
    if not WATCHLIST_PATH.is_file():
        print(f"ERROR: watchlist not found at {WATCHLIST_PATH}",
              file=sys.stderr)
        sys.exit(2)
    with WATCHLIST_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _validate(data: dict) -> list[str]:
    """Return a list of schema errors; empty list = clean."""
    errors: list[str] = []
    if data.get("__sentinel__") != "complete":
        errors.append("missing or wrong `__sentinel__: complete` "
                      "(truncated file?)")
    entries = data.get("watch")
    if not isinstance(entries, list):
        errors.append("root `watch:` must be a list")
        return errors
    seen_ids: set[str] = set()
    for idx, e in enumerate(entries):
        if not isinstance(e, dict):
            errors.append(f"entry #{idx}: must be a mapping")
            continue
        upstream = e.get("upstream", "")
        if not isinstance(upstream, str) or not _UPSTREAM_RE.match(upstream):
            errors.append(
                f"entry #{idx}: upstream={upstream!r} must match "
                "`[owner/repo]#<number>`"
            )
        if upstream in seen_ids:
            errors.append(f"entry #{idx}: duplicate upstream {upstream!r}")
        seen_ids.add(upstream)
        if e.get("status") not in _ALLOWED_STATUSES:
            errors.append(
                f"{upstream!r}: status={e.get('status')!r} must be one of "
                f"{sorted(_ALLOWED_STATUSES)}"
            )
        if e.get("action") not in _ALLOWED_ACTIONS:
            errors.append(
                f"{upstream!r}: action={e.get('action')!r} must be one of "
                f"{sorted(_ALLOWED_ACTIONS)}"
            )
        if not e.get("since"):
            errors.append(f"{upstream!r}: missing `since` (ISO date)")
    return errors


def _categorise(entry: dict) -> str:
    status = entry.get("status")
    action = entry.get("action")
    if action == "port" and status == "merged":
        return "PORT_CANDIDATE"
    if action == "retire":
        return "RETIRE_CANDIDATE"
    if status == "closed" and action not in ("port", "retire"):
        return "DONE"
    return "WATCH"


def _render_text(rows: list[dict]) -> str:
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    out: list[str] = []
    out.append("Upstream watchlist audit")
    out.append("=" * 60)
    for cat in ("PORT_CANDIDATE", "RETIRE_CANDIDATE", "WATCH", "DONE"):
        items = by_cat.get(cat, [])
        if not items:
            continue
        out.append(f"\n[{cat}] {len(items)} entries")
        for r in items:
            patches = ", ".join(r.get("local_patches") or []) or "—"
            out.append(
                f"  • {r['upstream']:<28} status={r['status']:<7}"
                f" action={r['action']:<14} patches=[{patches}]"
            )
    out.append("")
    out.append(f"Total: {len(rows)} entries")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON")
    p.add_argument("--skip-network", action="store_true",
                    help="Currently a no-op (script is offline-only); "
                         "kept for symmetry with audit_upstream_status.py")
    args = p.parse_args(argv)
    _ = args.skip_network  # silence linter

    data = _load_yaml()
    errors = _validate(data)
    if errors:
        if args.json:
            print(json.dumps({"schema_errors": errors}, indent=2))
        else:
            print("Schema errors:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    for e in data["watch"]:
        rows.append({
            "upstream": e.get("upstream"),
            "status": e.get("status"),
            "action": e.get("action"),
            "since": e.get("since"),
            "local_patches": e.get("local_patches") or [],
            "category": _categorise(e),
        })

    if args.json:
        # YAML auto-parses `since: 2026-05-12` as a datetime.date object;
        # `default=str` round-trips it as an ISO string so the output
        # stays JSON-friendly without forcing string parsing in the YAML.
        print(json.dumps(
            {"entries": rows, "schema_errors": []},
            indent=2, sort_keys=False, default=str,
        ))
    else:
        print(_render_text(rows))

    has_port_candidate = any(r["category"] == "PORT_CANDIDATE" for r in rows)
    return 1 if has_port_candidate else 0


if __name__ == "__main__":
    sys.exit(main())
