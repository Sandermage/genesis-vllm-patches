#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 7 release gate — `make audit-public-docs`.

Implements PROJECT_ROADMAP_V2 §6.10 public/private docs boundary. Catches
private-information leaks that the docs-stale-scan (§6.3 gate #1) doesn't:

  D-1  No public doc links into the private maintainer tree
       (`sndr_private/*` — the consolidated location, plus the legacy
       `docs/_internal/*` namespace kept here for forward protection
       against re-introduction).
  D-2  No RFC-1918 private IPv4 ranges in public docs.
  D-3  No `/home/<user>` or `/Users/<user>` operator paths in public docs.
  D-4  No server-only container names (vllm-server-mtp-test, vllm-pn95-2xa5000-*).
  D-5  No retired CLI verbs (genesis doctor, genesis verify, genesis migrate,
       ./scripts/launch.sh in instructions).
  D-6  No unresolved TODO/FIXME/XXX markers, `<PLACEHOLDER>` slot-tokens,
       or NotImplementedError in public docs. The plain English noun
       "placeholder" (used to describe e.g. PN64) does NOT count — only
       actionable markers do.

Allowlist (intentionally private / historical):
  sndr_private/, _archive/

Exit code:
  0 — clean.
  1 — at least one rule failed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

ALLOWLIST_PREFIXES = (
    "sndr_private/",
    "_archive/",
    # Phase 4.A (2026-05-22): docs/_internal/ is the retired internal-
    # docs path (migrated to sndr_private/planning/ — see .gitignore
    # line 44). The directory is gitignored, but if legacy files exist
    # on a developer laptop from before the migration, they would
    # otherwise be scanned as public docs. Adding the prefix here is
    # belt-and-suspenders alongside the .gitignore entry: defensive
    # against re-introduction even if the gitignore policy ever drifts.
    "docs/_internal/",
)


def _is_public_doc(rel: Path) -> bool:
    """Public means: under docs/ but NOT under any allowlist prefix,
    AND a markdown file (the gate's scope)."""
    s = rel.as_posix()
    if not s.startswith("docs/") and s != "README.md":
        return False
    if any(s.startswith(p) for p in ALLOWLIST_PREFIXES):
        return False
    return rel.suffix == ".md"


def _gather_public_doc_files() -> list[Path]:
    out = []
    for fp in REPO_ROOT.rglob("*.md"):
        rel = fp.relative_to(REPO_ROOT)
        if _is_public_doc(rel):
            out.append(fp)
    # README.md at repo root counts as public.
    return sorted(out)


def _grep(pattern: re.Pattern, files: list[Path]) -> list[str]:
    hits = []
    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = fp.relative_to(REPO_ROOT).as_posix()
        for i, line in enumerate(text.splitlines(), 1):
            if "audit-public-docs: allow" in line:
                continue
            if pattern.search(line):
                hits.append(f"{rel}:{i}: {line.strip()[:120]}")
    return hits


def check_d1_no_internal_links(files: list[Path]) -> list[str]:
    """D-1: public docs must not reference the private maintainer tree
    (`sndr_private/` is the canonical location post-consolidation;
    `docs/_internal/` is the retired legacy path, kept in the regex
    so a regression cannot silently re-introduce it)."""
    return _grep(re.compile(r"sndr_private/|docs/_internal"), files)


def check_d2_no_private_ips(files: list[Path]) -> list[str]:
    """D-2: no RFC-1918 private IPv4."""
    pat = re.compile(
        r"\b(?:10|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b"
    )
    return _grep(pat, files)


def check_d3_no_operator_paths(files: list[Path]) -> list[str]:
    """D-3: no operator home paths."""
    return _grep(re.compile(r"/(?:home|Users)/sander"), files)


def check_d4_no_server_container_names(files: list[Path]) -> list[str]:
    """D-4: server-only container names."""
    pat = re.compile(r"vllm-server-mtp-test|vllm-pn95-2xa5000")
    return _grep(pat, files)


def check_d5_no_retired_verbs(files: list[Path]) -> list[str]:
    """D-5: retired CLI surface."""
    pat = re.compile(
        r"\bgenesis (?:doctor|verify|migrate)\b|\./scripts/launch\.sh"
    )
    return _grep(pat, files)


def check_d6_no_unresolved_todos(files: list[Path]) -> list[str]:
    """D-6: actionable TODO/FIXME/XXX markers, slot-tokens, NotImplementedError.

    The plain English noun "placeholder" used to describe e.g. PN64 is
    legitimate prose and does not block this gate; only actionable
    markers are caught.

    NotImplementedError mentioned as a backticked code identifier (e.g.
    in a patch description that mentions removing an upstream raise) is
    legitimate prose, not an unresolved marker. Lines where every match
    is inside backticks are not reported.
    """
    pat = re.compile(
        r"TODO\([^)]*\)"
        r"|\bFIXME\b"
        r"|\bXXX\b"
        r"|<\s*PLACEHOLDER\s*>"
        r"|\bNotImplementedError\b"
    )
    hits: list[str] = []
    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = fp.relative_to(REPO_ROOT).as_posix()
        for i, line in enumerate(text.splitlines(), 1):
            if "audit-public-docs: allow" in line:
                continue
            matches = list(pat.finditer(line))
            if not matches:
                continue
            # If every match is inside backticks, treat as identifier
            # reference and skip — only unbacktickeed markers are real.
            bare = False
            for m in matches:
                before = line[:m.start()]
                after = line[m.end():]
                if before.count("`") % 2 == 0 and after.count("`") % 2 == 0:
                    bare = True
                    break
            if bare:
                hits.append(f"{rel}:{i}: {line.strip()[:120]}")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    files = _gather_public_doc_files()

    checks = {
        "D-1 no _internal links": check_d1_no_internal_links(files),
        "D-2 no private IPs": check_d2_no_private_ips(files),
        "D-3 no operator paths": check_d3_no_operator_paths(files),
        "D-4 no server container names": check_d4_no_server_container_names(files),
        "D-5 no retired CLI verbs": check_d5_no_retired_verbs(files),
        "D-6 no unresolved TODOs/placeholders": check_d6_no_unresolved_todos(files),
    }
    total = sum(len(v) for v in checks.values())

    if args.json:
        print(json.dumps(
            {"checks": checks, "total_failures": total,
             "public_docs_scanned": len(files)},
            indent=2, sort_keys=True,
        ))
    else:
        print(f"audit-public-docs: {len(files)} public doc files scanned")
        print("─" * 70)
        for check, hits in checks.items():
            if hits:
                print(f"  ✗ {check}: {len(hits)} hit(s)")
                for h in hits[:5]:
                    print(f"      {h}")
                if len(hits) > 5:
                    print(f"      ... ({len(hits) - 5} more)")
            else:
                print(f"  ✓ {check}: clean")
        print()
        if total:
            print(f"  FAIL — {total} total violation(s)")
        else:
            print("  OK — public docs pass boundary check")

    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
