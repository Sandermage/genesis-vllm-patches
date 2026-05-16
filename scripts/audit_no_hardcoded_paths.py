#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§4.2 / §6.10 — `make audit-no-hardcoded-paths` — operator path drift detector.

Active config files (V1 + V2 model_configs, compose YAMLs) must use
`${env_var}` placeholders for host paths, never hardcoded operator
filesystem paths like `/home/<user>/...` or `/Users/<user>/...`.

Hardcoded paths in active config:
  • break portability — works on operator's homelab, fails everywhere else
  • leak operator identity into shared/public artefacts (§6.10 boundary)
  • silently survive code review because they look "normal"

This gate scans the active config scope and rejects any non-comment
line containing a hardcoded user path pattern.

Allowlist:
  • lines starting with `#` (comments — operator may document paths)
  • lines under `_archive/` dirs (historical reference, not active config)
  • specific files in the EXEMPT_FILES list (e.g. test-deployment
    compose files that are operator-host-specific by design — the
    exemption requires an explicit allow-comment in the file header)

Exit codes:
  0 — no hardcoded paths in active config
  1 — at least one violation
  2 — internal error
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


# ─── Scope ─────────────────────────────────────────────────────────────
#
# Files to scan. Patterns are relative to REPO_ROOT. We focus on active
# config (V1/V2 model configs + compose); active scripts have their own
# operator-path conventions handled by other gates.

SCAN_GLOBS: tuple[str, ...] = (
    "vllm/sndr_core/model_configs/builtin/**/*.yaml",
    "compose/*.yml",
    "compose/*.yaml",
)


# Exempt specific files that are operator-host-specific by design.
# Adding to this list requires the file to explicitly document WHY in
# its header comment block.
#
# Public compose templates under `compose/` are now generated from V2
# presets via `sndr compose render <alias>` and use portable
# `/REPLACE_ME/*` placeholders — no per-file exemption needed.
# Operator-host-specific composes (the previous E26 entry
# `docker-compose.test-v11.yml`) moved to the private maintainer tree
# `sndr_private/compose/` in the 2026-05-16 privacy consolidation.
EXEMPT_FILES: frozenset[str] = frozenset()


# ─── Detection rules ──────────────────────────────────────────────────


# Pattern: `/home/<user>/` or `/Users/<user>/` where <user> is a
# plausible username. We don't match `/home` or `/Users` alone (that
# would match `/home/.bashrc` paths inside containers too).
#
# Allow `/Users/{models,users,public}` and similar generic dir names —
# those aren't operator-specific.
_PATH_RE = re.compile(
    r"(?P<root>/home/|/Users/)(?P<user>[a-zA-Z][a-zA-Z0-9_-]{1,30})/"
)
# Generic non-user mount roots we don't flag (well-known dirs).
_GENERIC_USERS: frozenset[str] = frozenset({
    "models", "shared", "public", "data", "vol", "Shared", "Public",
})


@dataclass
class PathViolation:
    file: Path
    line_no: int
    column: int
    matched: str
    line: str


@dataclass
class FileScanResult:
    path: Path
    violations: list[PathViolation] = field(default_factory=list)
    exempt: bool = False
    skipped: bool = False  # not in scope (no matches found vs. exempt)

    @property
    def passed(self) -> bool:
        return self.exempt or not self.violations


def _is_in_archive(path: Path) -> bool:
    return any(p.name == "_archive" for p in path.parents)


def _line_is_comment(line: str) -> bool:
    return line.lstrip().startswith("#")


def _scan_one_file(path: Path) -> FileScanResult:
    rel = path.relative_to(REPO_ROOT) if REPO_ROOT in path.parents else path
    if str(rel) in EXEMPT_FILES:
        return FileScanResult(path=path, exempt=True)

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return FileScanResult(path=path, skipped=True)

    violations: list[PathViolation] = []
    for ln_no, line in enumerate(text.splitlines(), start=1):
        if _line_is_comment(line):
            continue
        for m in _PATH_RE.finditer(line):
            user = m.group("user")
            if user in _GENERIC_USERS:
                continue
            violations.append(PathViolation(
                file=path,
                line_no=ln_no,
                column=m.start() + 1,
                matched=m.group(0),
                line=line.rstrip(),
            ))
    return FileScanResult(path=path, violations=violations)


def audit_no_hardcoded_paths(
    scan_globs: tuple[str, ...] = SCAN_GLOBS,
) -> list[FileScanResult]:
    results: list[FileScanResult] = []
    seen: set[Path] = set()
    for pat in scan_globs:
        for p in sorted(REPO_ROOT.glob(pat)):
            if p in seen or _is_in_archive(p) or not p.is_file():
                continue
            seen.add(p)
            results.append(_scan_one_file(p))
    return results


# ─── Renderers ────────────────────────────────────────────────────────


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _render_text(results: list[FileScanResult]) -> str:
    lines = []
    lines.append(f"audit-no-hardcoded-paths: {len(results)} file(s) scanned")
    lines.append("─" * 70)

    clean = [r for r in results if r.passed and not r.exempt]
    exempt = [r for r in results if r.exempt]
    violating = [r for r in results if not r.passed]

    for r in violating:
        lines.append(f"  ✗ {_rel(r.path)}: {len(r.violations)} violation(s)")
        for v in r.violations[:5]:
            lines.append(
                f"      L{v.line_no}:{v.column}  {v.matched!r}  "
                f"→ {v.line.strip()[:80]}"
            )
        if len(r.violations) > 5:
            lines.append(f"      ... ({len(r.violations) - 5} more)")

    lines.append("─" * 70)
    lines.append(
        f"  {len(clean)} clean / {len(exempt)} exempt / "
        f"{len(violating)} with violations"
    )
    total = sum(len(r.violations) for r in results)
    if total:
        lines.append("")
        lines.append(
            f"  ✗ {total} hardcoded path(s) found. "
            "Replace with `${env_var}` placeholders or add to EXEMPT_FILES "
            "with a header-comment justification."
        )
    return "\n".join(lines)


def _render_json(results: list[FileScanResult]) -> str:
    return json.dumps({
        "total_files": len(results),
        "clean": sum(1 for r in results if r.passed and not r.exempt),
        "exempt": sum(1 for r in results if r.exempt),
        "violating": sum(1 for r in results if not r.passed),
        "total_violations": sum(len(r.violations) for r in results),
        "files": [
            {
                "path": _rel(r.path),
                "exempt": r.exempt,
                "passed": r.passed,
                "violations": [
                    {
                        "line": v.line_no,
                        "column": v.column,
                        "matched": v.matched,
                        "context": v.line.strip()[:120],
                    }
                    for v in r.violations
                ],
            }
            for r in results
        ],
    }, indent=2, sort_keys=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true",
                    help="Machine-readable JSON.")
    args = ap.parse_args()

    results = audit_no_hardcoded_paths()
    if args.json:
        print(_render_json(results))
    else:
        print(_render_text(results))

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
