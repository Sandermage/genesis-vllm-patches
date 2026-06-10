#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""audit_external_findings.py — CI gate on external findings tracker
(§9.A.7, AUDIT-CLOSURE.4 / A7-EXTERNAL-FINDINGS-AUDIT.1, 2026-05-27).

Thin wrapper around
``vllm.sndr_core.findings.validator.validate_directory()``. No logic
duplication; the wrapper only adds the audit-script CLI surface
(exit codes, ``--json``, ``--strict-warnings``) to match the pattern
of other ``scripts/audit_*.py`` gates.

Scope is fixed by A7-EXTERNAL-FINDINGS-TRACKER.R (2026-05-27): the
tracker format ships at ``vllm/sndr_core/findings/schema.py``; the
live operator tracker dir is ``sndr_private/planning/external_findings/``
(gitignored). On public clones / CI checkouts the dir is absent and
the audit safely no-ops (0 findings, exit 0).

Default mode
────────────

  * No network.
  * Resolve tracker dir via the same precedence the CLI uses
    (``GENESIS_FINDINGS_DIR`` env > auto-discovery > legacy fallback).
  * Run schema + cross-finding rules (F-1 / F-2 / F-3 / F-4).
  * Errors → exit 1. Warnings → informational (exit 0).

``--strict-warnings``
─────────────────────

  Promote F-4 staleness warnings to errors. Tighter gate for
  operator-driven release preflight.

``--with-network``
──────────────────

  RESERVED. URL reachability check is gated behind a separate
  operator-approval ``.R`` (mirrors A.9 ``--allow-ssh`` pattern).
  Invocation today returns exit 2 with the gate explanation.

Exit codes
──────────

  0 — every finding validates (or tracker dir absent on CI)
  1 — at least one error, OR ≥1 warning under ``--strict-warnings``
  2 — usage error / ``--with-network`` requested without operator approval
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent

# Ensure repo root on path so ``vllm.sndr_core.*`` resolves when run as
# ``python3 scripts/audit_external_findings.py``.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve_dir(override: Optional[str]) -> Path:
    """Pick tracker dir: explicit override > findings.registry default."""
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = REPO_ROOT / p
        return p.resolve()
    from sndr.findings.registry import DEFAULT_FINDINGS_DIR
    return DEFAULT_FINDINGS_DIR


def _run_validator(root: Path):
    """Defer to the canonical validator."""
    from sndr.findings.validator import validate_directory
    return validate_directory(root)


def _result_to_dict(result, root: Path) -> dict:
    return {
        "findings_dir": str(root),
        "exists": root.is_dir(),
        "finding_count": len(result.findings),
        "errors": [
            {"rule": i.rule, "severity": i.severity, "message": i.message}
            for i in result.errors
        ],
        "warnings": [
            {"rule": i.rule, "severity": i.severity, "message": i.message}
            for i in result.warnings
        ],
        "passed_schema": result.passed,
    }


def _render_text(payload: dict, strict_warnings: bool) -> str:
    lines: list[str] = []
    lines.append("audit-external-findings: tracker validation")
    lines.append("─" * 70)
    lines.append(f"  findings dir: {payload['findings_dir']}")
    lines.append(f"  exists:       {payload['exists']}")
    lines.append(f"  findings:     {payload['finding_count']}")
    lines.append(f"  errors:       {len(payload['errors'])}")
    lines.append(f"  warnings:     {len(payload['warnings'])}"
                 + ("  (strict: counted as errors)" if strict_warnings else ""))
    lines.append("")
    for e in payload["errors"][:30]:
        lines.append(f"  ✗ [{e['rule']}] {e['message']}")
    for w in payload["warnings"][:30]:
        sym = "✗" if strict_warnings else "⚠"
        lines.append(f"  {sym} [{w['rule']}] {w['message']}")
    if not payload["exists"]:
        lines.append("  · tracker dir absent — clean no-op (CI/public checkout)")
    elif not payload["errors"] and not payload["warnings"]:
        lines.append("  ✓ tracker passes validation")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--findings-dir", default=None,
        help="override default tracker dir (default: GENESIS_FINDINGS_DIR or "
             "auto-discovery)",
    )
    ap.add_argument(
        "--json", action="store_true",
        help="emit machine-readable JSON",
    )
    ap.add_argument(
        "--strict-warnings", action="store_true",
        help="promote F-4 staleness warnings to errors (operator-driven "
             "release preflight)",
    )
    ap.add_argument(
        "--with-network", action="store_true",
        help="(RESERVED) URL reachability check — requires separate "
             "operator-approval `.R` phase; today returns exit 2",
    )
    args = ap.parse_args()

    if args.with_network:
        print(
            "audit-external-findings: --with-network is reserved for a "
            "future operator-approval phase (mirrors A.9 --allow-ssh "
            "double opt-in). Today the audit only does offline schema + "
            "cross-finding validation. Exit 2.",
            file=sys.stderr,
        )
        return 2

    root = _resolve_dir(args.findings_dir)

    try:
        result = _run_validator(root)
    except Exception as e:
        print(f"audit-external-findings: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 2

    payload = _result_to_dict(result, root)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_render_text(payload, args.strict_warnings))

    if payload["errors"]:
        return 1
    if args.strict_warnings and payload["warnings"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
