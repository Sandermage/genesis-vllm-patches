#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Env-flag consumer audit — Phase 10.5 D-extension (2026-06-01).

Verifies every non-retired ``env_flag`` declared in PATCH_REGISTRY is
**consumed** by at least one ``is_enabled``-style call site (or a
``Flags.<NAME>`` reference, or a literal env-name reference) in the
``sndr/`` Python tree (v12 layout; previously ``vllm/sndr_core/``).

A registered patch whose env_flag is read nowhere is a **dead patch**:
the dispatcher loads the entry and the boot log claims it as
"available", but no runtime code path checks the flag, so the patch
is silently off forever. This rule catches that drift at PR time
instead of on a future operator's debug session.

What "consumed" means
---------------------

For env_flag ``GENESIS_ENABLE_PXX``, the audit considers any of
these forms a valid consumer:

  * Literal string ``GENESIS_ENABLE_PXX`` — direct ``os.environ.get``
    call or comment in source.
  * ``Flags.PXX`` attribute reference — canonical ``is_enabled(Flags.X)``
    pattern.
  * ``Flags.<stripped tail>`` — when the env_flag is something like
    ``SNDR_ENABLE_PN283_PROC_BRIDGE``, the corresponding Flags attribute
    is ``Flags.PN283_PROC_BRIDGE``.

The 10 canonical prefixes (ENABLE / DISABLE / LEGACY / ALLOW / INFO
× {SNDR, GENESIS}) are stripped before the tail-match.

Scope
-----

  * Searches every ``.py`` file under ``sndr/`` except those inside
    ``_retired/`` or ``_archive/`` (retired patches by definition
    should not have live consumers).
  * Includes ``apply/_per_patch_dispatch.py`` (legacy register table)
    and ``cli/`` (operator-facing surfaces that gate behaviour).
  * Skips ``__pycache__`` and any non-Python file.

Out-of-scope
------------

  * ``test/`` consumers — flags referenced ONLY in tests are still
    orphans at runtime. The audit looks at runtime source only.
  * YAML / markdown references — that's what ``audit_config_keys``
    + ``check_doc_sync`` cover from the producer side.

Exit code
---------

  0 — every non-retired env_flag has at least one consumer.
  1 — one or more orphans found (CI gate fires).
  2 — internal error (registry or filesystem).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOT = REPO_ROOT / "sndr"

_CANONICAL_PREFIXES = (
    "GENESIS_ENABLE_", "SNDR_ENABLE_",
    "GENESIS_DISABLE_", "SNDR_DISABLE_",
    "GENESIS_LEGACY_", "SNDR_LEGACY_",
    "GENESIS_ALLOW_", "SNDR_ALLOW_",
    "GENESIS_INFO_", "SNDR_INFO_",
)


def _strip_prefix(flag: str) -> str:
    for p in _CANONICAL_PREFIXES:
        if flag.startswith(p):
            return flag[len(p):]
    return flag


def _load_source_corpus() -> str:
    """Concatenate every non-retired .py file under sndr/ into one
    big string. Cheap enough for our tree (~2 MB) and lets a single
    regex search cover every consumer site."""
    if not SCAN_ROOT.is_dir():
        return ""
    parts: list[str] = []
    for path in SCAN_ROOT.rglob("*.py"):
        # Skip retired wiring (_retired/ and the v12 _archive/) +
        # __pycache__.
        if "_retired" in path.parts:
            continue
        if "_archive" in path.parts:
            continue
        if "__pycache__" in path.parts:
            continue
        try:
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(parts)


def audit() -> list[dict]:
    """Return list of orphan findings. Empty list means clean."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from sndr.dispatcher.registry import PATCH_REGISTRY
    finally:
        sys.path.pop(0)

    corpus = _load_source_corpus()
    if not corpus:
        return [{
            "patch_id": "(audit)",
            "env_flag": "(corpus)",
            "lifecycle": "(audit)",
            "reason": f"could not read any .py files under {SCAN_ROOT}",
        }]

    orphans: list[dict] = []
    for pid, meta in PATCH_REGISTRY.items():
        if not isinstance(meta, dict):
            continue
        if meta.get("lifecycle") == "retired":
            continue
        flag = meta.get("env_flag")
        if not flag:
            continue
        tail = _strip_prefix(flag)
        # Three valid consumer patterns.
        patterns = (
            re.escape(flag),
            r"Flags\." + re.escape(tail),
            r"Flags\." + re.escape(pid),
        )
        if not any(re.search(pat, corpus) for pat in patterns):
            orphans.append({
                "patch_id": pid,
                "env_flag": flag,
                "lifecycle": meta.get("lifecycle", "?"),
                "reason": (
                    f"no consumer found: tried literal {flag!r}, "
                    f"Flags.{tail}, Flags.{pid} — no match anywhere "
                    f"under sndr/ (excluding _retired/ and _archive/). "
                    f"Either add a runtime read site OR mark the "
                    f"entry as lifecycle='retired'."
                ),
            })
    return orphans


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true",
                    help="emit JSON payload instead of human-readable summary")
    args = ap.parse_args()

    orphans = audit()

    if args.json:
        print(json.dumps({
            "orphan_count": len(orphans),
            "orphans": orphans,
            "passed": not orphans,
        }, indent=2, sort_keys=True))
    else:
        from sndr.dispatcher.registry import PATCH_REGISTRY
        total = sum(
            1 for meta in PATCH_REGISTRY.values()
            if isinstance(meta, dict)
            and meta.get("env_flag")
            and meta.get("lifecycle") != "retired"
        )
        print(f"audit-env-flag-consumers: {total} active env_flags scanned")
        print("─" * 70)
        if orphans:
            print(f"  ✗ ORPHAN ({len(orphans)}):")
            for o in orphans:
                print(f"      {o['patch_id']} ({o['lifecycle']}): {o['env_flag']}")
        else:
            print(f"  ✓ all {total} active env_flags have at least one consumer")

    return 0 if not orphans else 1


if __name__ == "__main__":
    sys.exit(main())
