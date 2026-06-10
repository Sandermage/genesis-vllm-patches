#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""audit_repo_garbage.py — preventive repo-garbage audit
(§9.A.2, AUDIT-CLOSURE.2, 2026-05-27).

Catches the kind of accidental commits that leak into a repo across
long sessions: merge-conflict leftovers (``*.orig`` / ``*.rej``),
filenames with illegal characters (quotes / brackets / control chars),
macOS metadata (``.DS_Store``), editor backup files (``*~``), and a
short list of "random temp output" filenames operators sometimes
leave behind.

The audit was filed in master plan v6.1 §9.A.2 as **URGENT** because
the recon pass surfaced literal ``?? \", f.type)...`` orphan filenames
at the repo root. Those were already cleaned by the time this audit
shipped; the audit role is **preventive** — keep the tracked tree
clean going forward.

Scope
─────

Default mode scans:

  1. Every file in ``git ls-files`` for forbidden filename patterns
     (merge leftovers, illegal chars, .DS_Store, editor backups,
     temp-output names).
  2. Untracked-but-visible files in critical zones (``docs/``,
     ``scripts/``, ``sndr/``) for the same patterns —
     catches accidental drag-and-drop into a critical surface
     before the operator runs ``git add``.

Expected-untracked allowlist (never flagged):

  * ``.claude/`` — Claude Code session artifacts (operator-local)
  * ``CLAUDE.md`` — user-level operator instructions (gitignored)
  * ``sndr_private/`` — operator's private planning tree (gitignored)

Exit codes
──────────

  0 — no garbage detected
  1 — at least one finding
  2 — internal error / git not available

Modes
─────

  python3 scripts/audit_repo_garbage.py             # human-readable
  python3 scripts/audit_repo_garbage.py --json      # machine-readable
  python3 scripts/audit_repo_garbage.py --tracked-only  # skip untracked-zone scan
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


# ─── Forbidden filename rules ─────────────────────────────────────────────


# Each rule: (regex on basename, category label).
_FILENAME_RULES: tuple[tuple[re.Pattern, str], ...] = (
    # Merge-conflict leftovers.
    (re.compile(r"\.(orig|rej)$"), "merge-leftover"),
    # macOS metadata.
    (re.compile(r"^\.DS_Store$"), "macos-metadata"),
    # Editor backup (Emacs / vim / nano).
    (re.compile(r"~$"), "editor-backup"),
    (re.compile(r"^#.*#$"), "emacs-autosave"),
    (re.compile(r"\.swp$|\.swo$"), "vim-swap"),
    # NOTE: ``temp-output`` rule (``out.txt`` / ``tmp.json`` / ``scratch.md``
    # at repo root) is handled separately in ``_classify_filename`` so
    # that ``.github/workflows/test.yml`` and other legitimate deep-path
    # files matching the same basename pattern don't trip it.
    # Stray Python bytecode at non-conventional paths.
    (re.compile(r"\.pyc$"), "stray-pyc"),
)


# Forbidden characters in filenames (shell-hostile + Windows-illegal).
_ILLEGAL_CHARS_RE = re.compile(r'[<>:"|?*\x00-\x1f]')


# ─── Allowlist (never flag) ───────────────────────────────────────────────


# Top-level allowlisted entries — expected untracked-but-present.
_TOP_LEVEL_ALLOWLIST: frozenset[str] = frozenset({
    ".claude",
    "CLAUDE.md",
    "sndr_private",
    # Standard caches/dirs that .gitignore covers but may still show.
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "vllm_sndr_core.egg-info",
})

# Critical zones scanned for untracked-but-present files (in addition
# to tracked-tree scan).
_CRITICAL_ZONES: tuple[str, ...] = (
    "docs/",
    "scripts/",
    "sndr/",            # v12 runtime tree (was vllm/sndr_core/)
    "vllm/sndr_core/",  # historical pre-v12 path — harmless fallback
    "tests/",
)


# ─── Finding ──────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Finding:
    path: str
    category: str
    state: str   # "tracked" | "untracked"

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


# ─── Scope discovery ──────────────────────────────────────────────────────


def _git_ls_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True, text=True, cwd=REPO_ROOT, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git ls-files failed (rc={result.returncode}): {result.stderr}"
        )
    return [line for line in result.stdout.splitlines() if line]


def _git_untracked() -> list[str]:
    """Return paths that are present in working tree but not tracked
    AND not ignored (respects ``.gitignore``)."""
    result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        capture_output=True, text=True, cwd=REPO_ROOT, check=False,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line]


_TEMP_OUTPUT_RE = re.compile(
    r"^(out|output|tmp|temp|debug|junk|scratch)"
    r"\.(txt|log|json|md|yaml|yml)$"
)


def _classify_filename(path: str) -> str | None:
    """Return forbidden-pattern category, or None if filename is clean.

    ``path`` is the repo-relative path; some rules are path-aware.
    """
    name = Path(path).name
    for pat, label in _FILENAME_RULES:
        if pat.search(name):
            return label
    if _ILLEGAL_CHARS_RE.search(name):
        return "illegal-filename-chars"
    # Path-aware rule: "scratchpad" names (``out.txt`` / ``tmp.json`` /
    # ``debug.md`` etc.) only matter at the immediate repo root. The
    # same basename inside ``docs/`` / ``scripts/`` / ``.github/`` is a
    # legitimate file with a domain meaning (e.g. CI workflow ``test.yml``).
    if "/" not in path and _TEMP_OUTPUT_RE.match(name):
        return "temp-output-at-root"
    return None


def _in_top_level_allowlist(path: str) -> bool:
    """Top-level allowlisted dirs/files (e.g. ``.claude/...``,
    ``CLAUDE.md``)."""
    head = path.split("/", 1)[0]
    return head in _TOP_LEVEL_ALLOWLIST


def _in_critical_zone(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in _CRITICAL_ZONES)


# ─── Audit ────────────────────────────────────────────────────────────────


def audit(
    *, scan_untracked: bool = True,
) -> list[Finding]:
    findings: list[Finding] = []

    # 1. Tracked tree.
    for rel in _git_ls_files():
        if _in_top_level_allowlist(rel):
            continue
        category = _classify_filename(rel)
        if category is not None:
            findings.append(Finding(
                path=rel, category=category, state="tracked",
            ))

    # 2. Untracked-but-visible in critical zones.
    if scan_untracked:
        for rel in _git_untracked():
            if _in_top_level_allowlist(rel):
                continue
            if not _in_critical_zone(rel):
                continue
            category = _classify_filename(rel)
            if category is not None:
                findings.append(Finding(
                    path=rel, category=category, state="untracked",
                ))

    return findings


# ─── Render ───────────────────────────────────────────────────────────────


def _render_text(findings: list[Finding]) -> str:
    lines: list[str] = []
    lines.append("audit-repo-garbage: preventive cleanliness check")
    lines.append("─" * 70)
    lines.append(f"  findings: {len(findings)}")
    by_cat: dict[str, int] = {}
    for f in findings:
        by_cat[f.category] = by_cat.get(f.category, 0) + 1
    for cat, count in sorted(by_cat.items()):
        lines.append(f"    {cat:24s} {count}")
    lines.append("")
    for f in findings[:50]:
        lines.append(f"  ✗ {f.path}  [{f.category}, {f.state}]")
    if len(findings) > 50:
        lines.append(f"  … ({len(findings) - 50} more)")
    if not findings:
        lines.append("  ✓ No garbage detected — repo tree clean")
    else:
        lines.append("")
        lines.append(
            "  ✗ Fix: remove the file(s) OR add a justified .gitignore "
            "entry if intentional."
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
        "--tracked-only", action="store_true",
        help="skip untracked-zone scan (faster; tracked tree only)",
    )
    args = ap.parse_args()

    try:
        findings = audit(scan_untracked=not args.tracked_only)
    except RuntimeError as e:
        print(f"audit-repo-garbage: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({
            "findings": [f.as_dict() for f in findings],
            "count": len(findings),
            "scanned_untracked": not args.tracked_only,
        }, indent=2, sort_keys=True))
    else:
        print(_render_text(findings))

    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
