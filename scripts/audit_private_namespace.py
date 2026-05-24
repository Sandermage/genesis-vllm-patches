#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P0.1 M.7 — Audit private-namespace placement (hard rule #27).

Three forbidden patterns + one positive allowance, all enforced
on the source tree (NOT on a built wheel — that's the M.2 contract).

  Forbidden:
    1. `vllm/**/sndr_private` directory path anywhere under `vllm/`.
       Single private namespace is `vllm/sndr_engine/` via license
       gate; `sndr_private` is a maintainer-archive concept and must
       never appear as a Python package under the public `vllm/`
       namespace.

    2. Packaged `sndr_private` — verified separately by
       `tests/unit/test_wheel_contents.py::test_no_sndr_private_anywhere_in_wheel`
       which builds and inspects the wheel. This script flags the
       absence of that test as a regression signal.

  Allowed:
    3. Repo-root `sndr_private/` directory — maintainer-private
       archive (planning, runs, research, abandoned WIP). Must be
       gitignored.

Exit codes:
  0 — clean (no violations)
  1 — at least one violation found
  2 — usage / IO error

CLI:
  python3 scripts/audit_private_namespace.py
  python3 scripts/audit_private_namespace.py --json
  python3 scripts/audit_private_namespace.py --strict  # treat warnings as errors

Wired into:
  - Makefile (`audit-private-namespace` target)
  - .pre-commit-config.yaml (hook fires on vllm/ + scripts/ changes)
  - tests/unit/scripts/test_audit_private_namespace.py (TDD)

History: introduced 2026-05-24 after P0.PROJECT-STRUCTURE.R+ found
17 files under `vllm/sndr_core/sndr_private/` shipping in the wheel
— architectural error per hard rule #27. P0.1 M.3a-d relocated them
to `sndr_private/archived/` (top-level, gitignored) and
`vllm/sndr_core/integrations/_retired/g4_upstream_tq_wip/` (proper
public namespace for retiring patches).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent


def _check_no_sndr_private_under_vllm() -> list[str]:
    """Rule 1: forbid `vllm/**/sndr_private` directory anywhere."""
    vllm_root = REPO_ROOT / "vllm"
    if not vllm_root.exists():
        return []
    violations: list[str] = []
    for path in vllm_root.rglob("sndr_private"):
        # Path must be a directory (file named "sndr_private" would
        # be unusual; either way, flag it).
        rel = path.relative_to(REPO_ROOT)
        violations.append(str(rel))
    return violations


def _check_wheel_contract_test_exists() -> Optional[str]:
    """Rule 2 cross-check: wheel-contract test must exist (it asserts
    `sndr_private` absence from built wheel). If the test goes away,
    we lose the build-time guard."""
    test_file = (
        REPO_ROOT / "tests" / "unit" / "test_wheel_contents.py"
    )
    if not test_file.exists():
        return (
            f"missing wheel-contract test: {test_file.relative_to(REPO_ROOT)} "
            f"— M.2 regression guard against `sndr_private` leaking into wheel"
        )
    text = test_file.read_text(encoding="utf-8", errors="ignore")
    if "test_no_sndr_private_anywhere_in_wheel" not in text:
        return (
            f"wheel-contract test exists but lacks "
            f"`test_no_sndr_private_anywhere_in_wheel` — regression of M.2 "
            f"sndr_private absence invariant"
        )
    return None


def _check_top_level_sndr_private_gitignored() -> Optional[str]:
    """Rule 3 positive: repo-root `sndr_private/` is the ONLY allowed
    location for the namespace. Verify it's gitignored to keep
    maintainer content out of public commits."""
    gitignore = REPO_ROOT / ".gitignore"
    if not gitignore.exists():
        return (
            ".gitignore missing — cannot verify `sndr_private/` "
            "is excluded from commits"
        )
    text = gitignore.read_text(encoding="utf-8", errors="ignore")
    # Accept any line that ignores sndr_private at repo root. Common
    # forms: `sndr_private/`, `/sndr_private/`, `sndr_private`, etc.
    patterns = ("sndr_private/", "/sndr_private/", "sndr_private")
    if not any(p in text for p in patterns):
        return (
            "`sndr_private` not listed in .gitignore — maintainer "
            "archive may leak to commits / public wheels"
        )
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ else "",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json", action="store_true",
        help="machine-readable JSON output",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="treat warnings (e.g. missing test guard) as errors",
    )
    args = parser.parse_args()

    rule1_violations = _check_no_sndr_private_under_vllm()
    rule2_warning = _check_wheel_contract_test_exists()
    rule3_warning = _check_top_level_sndr_private_gitignored()

    has_errors = bool(rule1_violations)
    has_warnings = bool(rule2_warning) or bool(rule3_warning)

    if args.json:
        payload = {
            "rule_1_no_vllm_sndr_private_namespace": {
                "violations": rule1_violations,
                "ok": not rule1_violations,
            },
            "rule_2_wheel_contract_test_exists": {
                "warning": rule2_warning,
                "ok": rule2_warning is None,
            },
            "rule_3_top_level_sndr_private_gitignored": {
                "warning": rule3_warning,
                "ok": rule3_warning is None,
            },
            "exit_code": (
                1 if has_errors or (args.strict and has_warnings) else 0
            ),
        }
        print(json.dumps(payload, indent=2))
    else:
        print("audit-private-namespace: hard rule #27 enforcement")
        print("─" * 70)
        if rule1_violations:
            print(f"  ❌ Rule 1 — `vllm/**/sndr_private` forbidden "
                  f"({len(rule1_violations)} violation(s)):")
            for v in rule1_violations:
                print(f"      {v}")
        else:
            print("  ✓ Rule 1 — no `sndr_private` namespace under `vllm/`")

        if rule2_warning:
            print(f"  ⚠ Rule 2 — wheel-contract test guard:")
            print(f"      {rule2_warning}")
        else:
            print("  ✓ Rule 2 — wheel-contract test guard present "
                  "(tests/unit/test_wheel_contents.py)")

        if rule3_warning:
            print(f"  ⚠ Rule 3 — top-level `sndr_private/` gitignore:")
            print(f"      {rule3_warning}")
        else:
            print("  ✓ Rule 3 — `sndr_private` listed in .gitignore "
                  "(maintainer archive protected)")

        print()
        if has_errors:
            print("  ✗ FAIL — hard rule #27 violated")
        elif args.strict and has_warnings:
            print("  ✗ FAIL (--strict) — warnings treated as errors")
        else:
            print("  OK — private namespace boundary preserved")

    if has_errors:
        return 1
    if args.strict and has_warnings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
