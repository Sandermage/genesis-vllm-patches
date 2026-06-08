# SPDX-License-Identifier: Apache-2.0
"""`sndr findings` — external findings pipeline CLI.

Implements EXTERNAL_FINDINGS_PIPELINE_2026-05-12_RU.md §3 surface:

  sndr findings list [--status <s>] [--due-for-review] [--json]
  sndr findings add --source ... --url ... --category ... --title ... [...]
  sndr findings update <id> --status ... [--notes ...] [--reviewed]
  sndr findings validate [--json]

Findings live under `docs/_internal/external_findings/<id>.yaml`. The
CLI is intentionally minimal — operator power-features (e.g. opening
PRs from a finding) layer on top later.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Optional

from . import _io


__all__ = [
    "add_argparser",
    "run_list",
    "run_add",
    "run_update",
    "run_validate",
]


def add_argparser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "findings",
        help="External findings pipeline — vLLM PRs / club issues / paper refs.",
        description=(
            "Structured tracking for upstream observations. Each finding "
            "is a self-contained YAML under "
            "`docs/_internal/external_findings/<id>.yaml`."
        ),
    )
    sub = p.add_subparsers(dest="findings_cmd", required=True)

    p_l = sub.add_parser("list", help="Enumerate findings (optionally filtered).")
    p_l.add_argument("--status", default=None,
                     help="Filter to findings whose status matches.")
    p_l.add_argument("--due-for-review", action="store_true",
                     help="Only findings past their review cadence.")
    p_l.add_argument("--root", default=None,
                     help="Override findings directory.")
    p_l.add_argument("--json", action="store_true")
    p_l.set_defaults(func=run_list)

    p_a = sub.add_parser("add", help="Create a new finding YAML.")
    p_a.add_argument("--id", required=True, dest="finding_id",
                     help="Finding id (e.g. external-vllm-12345).")
    p_a.add_argument("--source", required=True,
                     help="vllm-pr | vllm-issue | club-3090 | sglang | ...")
    p_a.add_argument("--url", required=True, help="Reference URL.")
    p_a.add_argument("--title", required=True, help="Short title.")
    p_a.add_argument("--category", required=True,
                     help="memory-cache | spec-decode | tool-call | ...")
    p_a.add_argument("--status", default="watch",
                     help="Initial status (default: watch).")
    p_a.add_argument("--risk", default="medium",
                     help="low | medium | high (default: medium).")
    p_a.add_argument("--cadence", default="biweekly", dest="review_cadence",
                     help="weekly | biweekly | on-pin-bump | retired (default: biweekly).")
    p_a.add_argument("--acceptance", default="(to be defined)",
                     help="Acceptance criterion for downstream action.")
    p_a.add_argument("--root", default=None, help="Override findings directory.")
    p_a.set_defaults(func=run_add)

    p_u = sub.add_parser("update",
                         help="Mutate a finding (status transition, notes, mark reviewed).")
    p_u.add_argument("finding_id", help="Finding id to update.")
    p_u.add_argument("--status", default=None,
                     help="New status (state-machine validated).")
    p_u.add_argument("--note", default=None, action="append", dest="notes",
                     help="Append a note (repeatable).")
    p_u.add_argument("--reviewed", action="store_true",
                     help="Update last_reviewed to today.")
    p_u.add_argument("--root", default=None, help="Override findings directory.")
    p_u.set_defaults(func=run_update)

    p_v = sub.add_parser("validate",
                         help="Run schema + cross-finding rules over the directory.")
    p_v.add_argument("--root", default=None, help="Override findings directory.")
    p_v.add_argument("--json", action="store_true")
    p_v.set_defaults(func=run_validate)


# ─── Helpers ───────────────────────────────────────────────────────────


def _resolve_root(opts: argparse.Namespace) -> Path:
    if opts.root:
        return Path(opts.root).expanduser().resolve()
    from sndr.findings.registry import DEFAULT_FINDINGS_DIR
    return DEFAULT_FINDINGS_DIR


def _find_path_by_id(root: Path, finding_id: str) -> Optional[Path]:
    from sndr.findings.registry import discover_findings
    for path, f in discover_findings(root):
        if f.id == finding_id:
            return path
    return None


def _finding_summary(f) -> dict:
    return {
        "id": f.id,
        "title": f.title,
        "source": f.source,
        "url": f.url,
        "category": f.category,
        "status": f.status,
        "risk": f.risk,
        "review_cadence": f.review_cadence,
        "last_reviewed": f.last_reviewed,
        "target": list(f.target),
    }


# ─── list ──────────────────────────────────────────────────────────────


def run_list(opts: argparse.Namespace) -> int:
    from sndr.findings import discover_findings, is_due_for_review

    root = _resolve_root(opts)
    findings = [f for _path, f in discover_findings(root)]
    if opts.status:
        findings = [f for f in findings if f.status == opts.status]
    if opts.due_for_review:
        findings = [f for f in findings if is_due_for_review(f)]

    if opts.json:
        print(json.dumps(
            {"findings": [_finding_summary(f) for f in findings],
             "count": len(findings),
             "root": str(root)},
            indent=2, sort_keys=True,
        ))
        return 0

    print(f"sndr findings list — {root}")
    print("─" * 70)
    if not findings:
        print("  (no findings match the filters)")
        return 0
    for f in findings:
        stale_marker = " (REVIEW DUE)" if is_due_for_review(f) else ""
        print(f"  {f.id}")
        print(f"      {f.title}  [{f.source}]")
        print(f"      status={f.status}  risk={f.risk}  "
              f"cadence={f.review_cadence}{stale_marker}")
    print()
    print(f"  Total: {len(findings)}")
    return 0


# ─── add ───────────────────────────────────────────────────────────────


def run_add(opts: argparse.Namespace) -> int:
    from sndr.findings.schema import (
        VALID_CADENCES, VALID_CATEGORIES, VALID_RISKS, VALID_SOURCES,
        VALID_STATUSES,
    )

    # Up-front vocabulary checks so we don't write an invalid YAML.
    pairs = {
        "source": (opts.source, VALID_SOURCES),
        "category": (opts.category, VALID_CATEGORIES),
        "status": (opts.status, VALID_STATUSES),
        "risk": (opts.risk, VALID_RISKS),
        "cadence": (opts.review_cadence, VALID_CADENCES),
    }
    for name, (val, allowed) in pairs.items():
        if val not in allowed:
            _io.warn(f"--{name}={val!r} not in {sorted(allowed)}")
            return 2

    root = _resolve_root(opts)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{opts.finding_id}.yaml"
    if target.exists():
        _io.warn(f"finding {opts.finding_id!r} already exists at {target}; "
                 f"use `sndr findings update` to mutate it")
        return 2

    today = date.today().isoformat()
    yaml_text = f"""# SPDX-License-Identifier: Apache-2.0
# Generated by `sndr findings add` on {today}.

schema_version: 1
id: {opts.finding_id}
source: {opts.source}
url: {opts.url}
title: "{opts.title}"
discovered_at: '{today}'

category: {opts.category}
status: {opts.status}
risk: {opts.risk}

# Downstream actions (extend as the finding matures).
target: []

# Acceptance criterion (what must hold before we act).
acceptance: |
  {opts.acceptance}

notes: []

# Lifecycle
last_reviewed: '{today}'
review_cadence: {opts.review_cadence}
"""
    target.write_text(yaml_text, encoding="utf-8")
    print(f"sndr findings add — wrote {target}")
    print("  Run `sndr findings validate` to confirm shape.")
    return 0


# ─── update ────────────────────────────────────────────────────────────


def run_update(opts: argparse.Namespace) -> int:
    from sndr.findings import (
        is_valid_transition, load_finding,
    )

    root = _resolve_root(opts)
    path = _find_path_by_id(root, opts.finding_id)
    if path is None:
        _io.warn(f"finding {opts.finding_id!r} not found under {root}")
        return 2

    f = load_finding(path)

    # Edit YAML text directly to preserve operator comments. We can't
    # round-trip-edit through PyYAML without losing them, so we do a
    # targeted regex substitution per field.
    import re
    text = path.read_text(encoding="utf-8")

    changes = []
    if opts.status and opts.status != f.status:
        if not is_valid_transition(f.status, opts.status):
            from sndr.findings.schema import ALLOWED_TRANSITIONS
            allowed = sorted(ALLOWED_TRANSITIONS.get(f.status, frozenset()))
            _io.warn(
                f"illegal transition {f.status!r} → {opts.status!r}; "
                f"allowed from {f.status!r}: {allowed}"
            )
            return 2
        text = re.sub(r"^status:\s*\S+",
                      f"status: {opts.status}", text, count=1,
                      flags=re.MULTILINE)
        changes.append(f"status {f.status} → {opts.status}")

    if opts.notes:
        # Append new notes to the `notes:` list. Replace `notes: []`
        # with a multi-line list, or extend an existing list.
        existing = f.notes
        merged = list(existing) + list(opts.notes)
        rendered = "notes:\n" + "\n".join(f'  - "{n}"' for n in merged)
        text = re.sub(r"^notes:.*?(?=\n[^\s-])",
                      rendered, text, count=1,
                      flags=re.MULTILINE | re.DOTALL)
        changes.append(f"+{len(opts.notes)} note(s)")

    if opts.reviewed:
        today = date.today().isoformat()
        text = re.sub(r"^last_reviewed:.*$",
                      f"last_reviewed: '{today}'", text, count=1,
                      flags=re.MULTILINE)
        changes.append(f"last_reviewed → {today}")

    if not changes:
        print(f"sndr findings update — nothing to change for {opts.finding_id!r}")
        return 0

    path.write_text(text, encoding="utf-8")
    print(f"sndr findings update — {opts.finding_id}")
    for c in changes:
        print(f"  • {c}")
    return 0


# ─── validate ──────────────────────────────────────────────────────────


def run_validate(opts: argparse.Namespace) -> int:
    from sndr.findings import validate_directory

    root = _resolve_root(opts)
    result = validate_directory(root)

    if opts.json:
        payload = {
            "root": str(root),
            "findings": [_finding_summary(f) for f in result.findings],
            "issues": [
                {"rule": i.rule, "severity": i.severity, "message": i.message}
                for i in result.issues
            ],
            "errors": len(result.errors),
            "warnings": len(result.warnings),
            "passed": result.passed,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if result.passed else 1

    print(f"sndr findings validate — {root}")
    print("─" * 70)
    print(f"  findings: {len(result.findings)}")
    print(f"  errors:   {len(result.errors)}")
    print(f"  warnings: {len(result.warnings)}")
    print()
    if result.issues:
        for sev in ("error", "warning"):
            rows = [i for i in result.issues if i.severity == sev]
            if not rows:
                continue
            sym = "✗" if sev == "error" else "⚠"
            print(f"  {sym} {sev.upper()} ({len(rows)}):")
            for i in rows:
                print(f"    [{i.rule}] {i.message}")
            print()
    if result.passed:
        print("  ✓ findings registry passes validation")
    else:
        print(f"  ✗ findings registry FAILED ({len(result.errors)} errors)")
    return 0 if result.passed else 1
