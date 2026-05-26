#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""audit_plan_supersession.py — planning supersession consistency
(§9.A.5, AUDIT-CLOSURE.1.A.5, 2026-05-26).

Verifies that explicit "superseded by FILENAME.md" / "Supersedes:
FILENAME.md" references in planning docs (``sndr_private/planning/``)
resolve to real files. Future-proofs against typos as operators adopt
the explicit supersession convention.

Scope
─────

Default scan target is ``sndr_private/planning/`` (gitignored,
operator-private). The audit DOES NOT touch the public tracked tree,
DOES NOT push to public origin, DOES NOT interact with rig/docker.

Why narrow to filename targets
─────────────────────────────

Current planning corpus uses **narrative** supersession ("superseded by
PIN.R", "superseded by Phase 5-7 work") that doesn't carry a verifiable
target. The audit deliberately ignores narrative supersession — false
positives on legitimate prose would drown signal. Only filename-target
references (``X.md``) get verified.

Rules
─────

R-SUP-1 — explicit filename references. For each match of:

  * ``Superseded by [PATH/]FILENAME.md`` (anywhere in body)
  * ``Supersedes: [PATH/]FILENAME.md``
  * ``superseded_by: [PATH/]FILENAME.md`` (YAML-style frontmatter or
    body annotation)
  * ``Superseded by `` followed by an .md token in body prose

…the referenced ``.md`` file must exist either in the same directory
as the source doc, in any parent dir up to ``sndr_private/planning/``,
or as an absolute path under the scan root.

R-SUP-2 — superseded-without-target. If a doc declares ``status:
superseded`` (frontmatter-style YAML line), it MUST carry a
``superseded_by:`` reference. Bare "I am superseded" without a
successor is operator hygiene loss.

Inline marker ``<!-- audit-plan-supersession: allow -->`` on the same
line waives a finding (historical quote, intentionally-broken example).

Exit codes
──────────

  0 — every filename-target supersession reference resolves; every
      status:superseded entry has a target
  1 — at least one violation
  2 — internal error / scan dir missing

Modes
─────

  python3 scripts/audit_plan_supersession.py            # human-readable
  python3 scripts/audit_plan_supersession.py --json     # machine
  python3 scripts/audit_plan_supersession.py --scan-root PATH

Wiring
──────

This audit is standalone (``make audit-plan-supersession``), **not**
in the ``make gates`` aggregate. It scans gitignored private trees,
so CI / pre-commit cannot reach the corpus. Operator runs it during
planning hygiene phases.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent

# Default scan root for planning docs (gitignored, operator-private).
DEFAULT_SCAN_ROOT = REPO_ROOT / "sndr_private" / "planning"


# ─── Patterns ─────────────────────────────────────────────────────────────


# Match an .md token after "superseded by" / "Supersedes:" / similar.
# We extract just the filename (possibly with path); strip backticks/
# brackets/parens that wrap it in markdown prose.
_FILENAME_TOKEN = r"`?([\w./\-]+\.md)`?"

# R-SUP-1 inline body patterns — any of these on a line trigger verification.
_BODY_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (
        re.compile(
            r"\*\*Superseded\s+by\*\*[:\s]+" + _FILENAME_TOKEN,
            re.IGNORECASE,
        ),
        "Superseded-by header reference",
    ),
    (
        re.compile(
            r"\*\*Supersedes\*\*[:\s]+" + _FILENAME_TOKEN,
            re.IGNORECASE,
        ),
        "Supersedes header reference",
    ),
    (
        re.compile(
            r"^\s*Superseded\s+by[:\s]+" + _FILENAME_TOKEN,
            re.IGNORECASE | re.MULTILINE,
        ),
        "Superseded-by line reference",
    ),
    (
        re.compile(
            r"^\s*Supersedes[:\s]+" + _FILENAME_TOKEN,
            re.IGNORECASE | re.MULTILINE,
        ),
        "Supersedes line reference",
    ),
    (
        re.compile(
            r"^\s*superseded_by:\s+" + _FILENAME_TOKEN,
            re.MULTILINE,
        ),
        "superseded_by YAML reference",
    ),
)

# R-SUP-2 — declared "I am superseded" without target.
_STATUS_SUPERSEDED_RE = re.compile(
    r"^\s*status:\s*superseded\b",
    re.IGNORECASE | re.MULTILINE,
)
_SUPERSEDED_BY_PRESENT_RE = re.compile(
    r"superseded_by:\s*\S+|Superseded\s+by[:\s]+\S+\.md",
    re.IGNORECASE,
)


# Inline allow marker — same line as the offending pattern waives it.
INLINE_ALLOW_MARKER = "<!-- audit-plan-supersession: allow -->"


# ─── Finding ──────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Finding:
    source: str
    line: int
    rule: str
    detail: str

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


# ─── Resolution helpers ───────────────────────────────────────────────────


def _resolve_target(
    target: str,
    source_md: Path,
    *,
    scan_root: Path,
) -> Optional[Path]:
    """Try to resolve ``target`` against several anchors:

      1. Absolute path (if target starts with ``/``)
      2. Relative to source_md's directory
      3. Same basename anywhere under scan_root (deepest match first)

    Returns the resolved Path if found, else None.
    """
    if target.startswith("/"):
        p = Path(target)
        return p if p.exists() else None

    # Try relative-to-source.
    rel = (source_md.parent / target).resolve()
    if rel.exists():
        return rel

    # Try basename-anywhere-under-scan-root.
    basename = Path(target).name
    matches = sorted(
        scan_root.rglob(basename),
        key=lambda p: -len(str(p)),  # deepest path first
    )
    if matches:
        return matches[0]

    return None


def _line_has_allow_marker(text: str, line_number: int) -> bool:
    lines = text.splitlines()
    if 0 < line_number <= len(lines):
        return INLINE_ALLOW_MARKER in lines[line_number - 1]
    return False


# ─── Audit core ───────────────────────────────────────────────────────────


def audit_planning_tree(
    *,
    scan_root: Optional[Path] = None,
) -> list[Finding]:
    """Walk ``scan_root`` for .md files and apply R-SUP-1 + R-SUP-2."""
    scan_root = (scan_root or DEFAULT_SCAN_ROOT).resolve()
    if not scan_root.is_dir():
        return []

    findings: list[Finding] = []

    for md in sorted(scan_root.rglob("*.md")):
        try:
            text = md.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = md.relative_to(scan_root.parent)

        # R-SUP-1: every filename-target match must resolve.
        seen_lines: set[tuple[int, str]] = set()
        for pat, label in _BODY_PATTERNS:
            for m in pat.finditer(text):
                target = m.group(1)
                # Line number from start position.
                lineno = text.count("\n", 0, m.start()) + 1
                key = (lineno, target)
                if key in seen_lines:
                    continue
                seen_lines.add(key)
                if _line_has_allow_marker(text, lineno):
                    continue
                resolved = _resolve_target(
                    target, md, scan_root=scan_root,
                )
                if resolved is None:
                    findings.append(Finding(
                        source=str(rel),
                        line=lineno,
                        rule="R-SUP-1",
                        detail=(
                            f"{label}: target {target!r} does not "
                            f"resolve to any .md under {scan_root.name}/"
                        ),
                    ))

        # R-SUP-2: "status: superseded" without "superseded_by:" target.
        for m in _STATUS_SUPERSEDED_RE.finditer(text):
            lineno = text.count("\n", 0, m.start()) + 1
            if _line_has_allow_marker(text, lineno):
                continue
            # Does the doc carry ANY supersession-by reference anywhere?
            if _SUPERSEDED_BY_PRESENT_RE.search(text):
                continue
            findings.append(Finding(
                source=str(rel),
                line=lineno,
                rule="R-SUP-2",
                detail=(
                    "status: superseded declared but no "
                    "`superseded_by:` / `Superseded by FILENAME.md` "
                    "target found anywhere in the doc"
                ),
            ))

    return findings


# ─── Render ───────────────────────────────────────────────────────────────


def _render_text(findings: list[Finding], scan_root: Path) -> str:
    lines: list[str] = []
    lines.append("audit-plan-supersession: planning hygiene")
    lines.append("─" * 70)
    lines.append(f"  scan root:   {scan_root}")
    lines.append(f"  findings:    {len(findings)}")
    by_rule: dict[str, int] = {}
    for f in findings:
        by_rule[f.rule] = by_rule.get(f.rule, 0) + 1
    for rule, count in sorted(by_rule.items()):
        lines.append(f"    {rule:12s} {count}")
    lines.append("")
    for f in findings[:50]:
        lines.append(f"  ✗ {f.source}:{f.line}  [{f.rule}]")
        lines.append(f"      {f.detail}")
    if len(findings) > 50:
        lines.append(f"  … ({len(findings) - 50} more)")
    if not findings:
        lines.append("  ✓ Every supersession claim resolves cleanly")
    else:
        lines.append("")
        lines.append(
            "  ✗ Fix: update the supersession target OR add the inline\n"
            f"      marker for an intentional historical citation:\n"
            f"      {INLINE_ALLOW_MARKER}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON")
    ap.add_argument(
        "--scan-root", default=str(DEFAULT_SCAN_ROOT),
        help="override default scan root (planning tree)",
    )
    args = ap.parse_args()

    scan_root = Path(args.scan_root).resolve()

    try:
        findings = audit_planning_tree(scan_root=scan_root)
    except Exception as e:
        print(f"audit-plan-supersession: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({
            "scan_root": str(scan_root),
            "findings": [f.as_dict() for f in findings],
            "count": len(findings),
        }, indent=2, sort_keys=True))
    else:
        print(_render_text(findings, scan_root))

    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
