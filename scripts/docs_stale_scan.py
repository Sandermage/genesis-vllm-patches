#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Docs stale-token scanner (supplement §3, audit 2026-05-12).

Public-facing markdown / README / docs must NOT reference retired
namespaces or stale CLI commands. Users following the doc would run
commands that no longer exist and reach for files that no longer live
in the named path.

Forbidden tokens (allowlisted in `docs/_internal/` and `**/_archive/**`):

  • `vllm/_genesis` / `vllm._genesis`  — pre-v11 namespace
  • `vllm/sndr_core/wiring/`           — pre-v10 layout (renamed to integrations)
  • `wiring/patch_`                    — pre-v10 patch file naming
  • `genesis doctor`                   — retired CLI verb (replaced by `sndr doctor`)
  • `genesis verify`                   — retired (use `sndr verify`)
  • `genesis migrate`                  — retired
  • `./scripts/launch.sh`              — retired entrypoint
  • `vllm-server-mtp-test`             — internal container name leaked from dev rig

Exit codes:
  0 — clean.
  1 — forbidden tokens present (each line printed with file:line).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Patterns the scanner must reject in public docs. Each is a literal
# substring search (case-sensitive); the audit-public-paths gate handles
# the regex-y IP/path cases separately.
FORBIDDEN_TOKENS: tuple[str, ...] = (
    "vllm/_genesis",
    "vllm._genesis",
    "vllm/sndr_core/wiring",
    "wiring/patch_",
    "genesis doctor",
    "genesis verify",
    "genesis migrate",
    "./scripts/launch.sh",
    "scripts/launch.sh ",       # trailing space → standalone usage hint
    "vllm-server-mtp-test",
)

# Search scope.
SEARCH_PATHS: tuple[str, ...] = (
    "README.md",
    "docs",
    "scripts/launch/README.md",
)

# Allowlist: paths where references are intentional (archive, internal
# planning docs, migration notes that explain the retirement).
ALLOWLIST_SUBSTRINGS: tuple[str, ...] = (
    "docs/_internal/",
    "/_archive/",
    "docs/archive/",
    # Historical upstream planning + reference docs. They literally
    # describe the v7→v11 migration so the retired tokens are the
    # content. New content should not land here without explicit
    # archive marker; this allowlist captures the existing corpus.
    "docs/upstream/",
    "docs/reference/",
    # Migration appendices that explicitly document the retirement.
    "docs/INSTALL.md",  # Migration appendix referencing _genesis alias
    "docs/CONTRIBUTING.md",  # logger back-compat note
    "docs/BENCHMARK_GUIDE.md",  # symlink back-compat note
    # The scanner itself.
    "scripts/docs_stale_scan.py",
)


def _is_allowlisted(path: Path) -> bool:
    rel = str(path.relative_to(REPO_ROOT)) if path.is_absolute() else str(path)
    return any(sub in rel for sub in ALLOWLIST_SUBSTRINGS)


def _iter_md_files() -> list[Path]:
    out: list[Path] = []
    for s in SEARCH_PATHS:
        p = REPO_ROOT / s
        if p.is_file() and p.suffix == ".md":
            out.append(p)
        elif p.is_dir():
            for f in p.rglob("*.md"):
                if f.is_file():
                    out.append(f)
    return out


def _scan_file(path: Path) -> list[tuple[int, str, str]]:
    """Return [(lineno, token, line)] of forbidden hits."""
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    hits: list[tuple[int, str, str]] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        for token in FORBIDDEN_TOKENS:
            if token in line:
                hits.append((idx, token, line.rstrip()))
                break
    return hits


def main(argv: list[str] | None = None) -> int:
    files = _iter_md_files()
    print(f"=== docs_stale_scan.py ===")
    print(f"scanning {len(files)} markdown files in {', '.join(SEARCH_PATHS)}")
    total = 0
    for f in files:
        if _is_allowlisted(f):
            continue
        for lineno, token, line in _scan_file(f):
            rel = f.relative_to(REPO_ROOT)
            print(f"{rel}:{lineno}: [{token}] {line[:120]}")
            total += 1
    print()
    if total == 0:
        print("✓ docs-stale-scan: clean")
        return 0
    print(f"✗ docs-stale-scan: {total} stale token(s)")
    print()
    print("Fix options:")
    print("  - Update the doc to use the current command/path/namespace.")
    print("  - Move the doc into docs/_internal/ or docs/archive/ if it's historical.")
    print("  - Mark the section as 'ARCHIVED / not current' and request allowlist.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
