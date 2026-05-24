#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""check_doc_sync.py — verify patch counts in docs match PATCH_REGISTRY.

Background: 2026-05-11 audit found patch counts of 132 / 123 / 128 / 130 / 37
scattered across README.md / PATCHES.md / INSTALL.md / BENCHMARKS.md / MODELS.md.
Reality (PATCH_REGISTRY) = 134. This is recurring drift class.

This script reads PATCH_REGISTRY count from registry.py source (regex parse
of dict literal — no torch/vllm import required) and grep'es docs for
patch-count claims, reporting mismatches.

Usage:
    python3 scripts/check_doc_sync.py              # report mode, exit 0/1
    python3 scripts/check_doc_sync.py --strict     # exit 1 if any mismatch
    python3 scripts/check_doc_sync.py --json       # machine-readable output

Exit codes:
    0  — all docs in sync (or --strict not set)
    1  — mismatches found in --strict mode
    2  — registry.py not parseable or doc files missing
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "vllm" / "sndr_core" / "dispatcher" / "registry.py"

# Doc files to check. Each entry: (path_relative_to_repo, [(regex, expected_capture_group_index)])
DOC_PATTERNS: dict[Path, list[tuple[str, int]]] = {
    REPO_ROOT / "README.md": [
        (r"patches-(\d+)-green", 1),                        # badge
        (r"\*\*(\d+) community patches\*\*", 1),            # narrative
        (r"applies (\d+) small, surgical changes", 1),      # narrative
        (r"category \((\d+) entries\)", 1),                 # category section
        (r"Of (\d+) registry entries", 1),                  # narrative
        (r"## 📦 (\d+) patches by category", 1),            # section heading
        (r"\| \*\*TOTAL\*\* \| \*\*(\d+)\*\*", 1),          # total row
        (r"All (\d+) patches table", 1),                    # table claim
        (r"^# (\d+) patches across (\d+) categories", 1),   # rare line
        (r"registry-(\d+)%20patches-green", 1),             # patches badge img URL
    ],
    REPO_ROOT / "docs" / "PATCHES.md": [
        (r"Total PATCH_REGISTRY entries:\*\* (\d+)", 1),    # header bullet
        (r"\| Total PATCH_REGISTRY entries \| \*\*(\d+)\*\*", 1),  # table
        (r"Tier=community \(Apache 2\.0, sndr_core\) \| \*\*(\d+)\*\*", 1),
    ],
    REPO_ROOT / "docs" / "INSTALL.md": [
        (r"# (\d+) community patches", 1),                  # tree comment
    ],
    REPO_ROOT / "docs" / "MODELS.md": [
        (r"Genesis maintains (\d+) vLLM runtime patches", 1),
        (r"Genesis patch lock-in \((\d+) runtime patches", 1),
        (r"Genesis patch lock-in \((\d+) entries in `PATCH_REGISTRY`", 1),
    ],
    REPO_ROOT / "docs" / "BENCHMARKS.md": [
        # Current-state claims (historical Wave snapshots are excluded by
        # specific phrasing — "snapshot", "ago", "pre-v11" markers).
        (r"Genesis `v12\.0\.0` — (\d+) PATCH_REGISTRY entries", 1),
        (r"Patches:\s+(\d+) total → ~\d+ APPLY \| ~\d+ SKIP", 1),
        (r"smaller \(\d+ entries vs (\d+) today\)", 1),
    ],
    # CONFIG-HYGIENE.docs-reconcile.1.GATE-EXTEND (2026-05-24): coverage
    # extended to the 5 files that previously drifted silently.
    REPO_ROOT / "docs" / "FAQ.md": [
        (r"\*\*(\d+) entries\*\* — \d+ full-implementation", 1),
        (r"About \d+ of (\d+) entries are marked `default_on=True`", 1),
    ],
    REPO_ROOT / "docs" / "CONFIGURATION.md": [
        (r"Genesis v12\.0\.0 — registry has \*\*(\d+) entries\*\*", 1),
    ],
    REPO_ROOT / "docs" / "QUICKSTART.md": [
        (r"Genesis `v12\.0\.0` \((\d+) PATCH_REGISTRY entries\)", 1),
    ],
    REPO_ROOT / "docs" / "RELEASE_POLICY.md": [
        (r"operators re-bench all (\d+) entries", 1),
        (r"Currently 0/(\d+) entries carry", 1),
        (r"`require-static`, (\d+)/(\d+) covered out-of-the-box", 1),
        (r"~\d+/(\d+) \(\d+\.\d+%\) in `bench_with_baseline`", 1),
    ],
}


# CONFIG-HYGIENE.docs-reconcile.1.GATE-EXTEND (2026-05-24):
# Transition allowlist — sites that are KNOWN to claim a stale registry
# count at a transition commit. Each entry is
# `(relative_path, line_number, found_value)`. Strict mode treats these
# as PENDING (informational warning) rather than ERROR.
#
# EMPTIED by CONFIG-HYGIENE.docs-reconcile.1.MECHANICAL (2026-05-24).
# Kept here (as an empty frozenset) so the scaffolding remains
# discoverable for the NEXT counter-bump cycle (e.g., when the registry
# advances 227 → 228 and a new round of doc fixes is staged); operators
# can add transition entries here without redesigning the gate.
_TRANSITION_ALLOWLIST: frozenset[tuple[str, int, int]] = frozenset()


def count_registry_entries(registry_path: Path) -> int:
    """Count top-level dict entries in PATCH_REGISTRY = {...}.

    Uses regex parse — no Python import (so script is torch-less).
    Counts `"PATCH_ID": {` keys at indent depth 1 (4 spaces).
    """
    if not registry_path.is_file():
        raise FileNotFoundError(f"Registry not found: {registry_path}")
    text = registry_path.read_text()
    # Find PATCH_REGISTRY = { ... } block (multi-line)
    m = re.search(r"PATCH_REGISTRY\s*[:=]\s*[a-zA-Z\[\]\s,]*\{", text)
    if not m:
        # Try simpler: just count `"PXX": {` style top-level entries
        # at exact 4-space indent which is canonical for dict literal at module top
        pass
    # Count exact 4-space-indented `"KEY": {` entries.
    # Keys may contain hyphens (e.g. `PN40-classifier`) so include `-`.
    matches = re.findall(r"^    \"([A-Za-z0-9_\-]+)\":\s*\{", text, flags=re.M)
    return len(set(matches))


def check_doc(doc_path: Path, expected_count: int, patterns: list[tuple[str, int]]) -> list[dict]:
    """Return list of mismatches in this doc.

    Each mismatch dict carries `transition_pending=True` when the
    `(rel_path, line, found_value)` tuple is in `_TRANSITION_ALLOWLIST`
    — signalling that the stale value is known and scheduled for fix
    by `CONFIG-HYGIENE.docs-reconcile.1.MECHANICAL`. Strict mode treats
    those as PENDING (does not fail) and any unallowlisted mismatch as
    ERROR (fails).
    """
    mismatches = []
    if not doc_path.is_file():
        return [{"doc": str(doc_path), "error": "file not found"}]
    text = doc_path.read_text()
    rel = doc_path.relative_to(REPO_ROOT).as_posix()
    for pattern, group_idx in patterns:
        for m in re.finditer(pattern, text, flags=re.M | re.S):
            value = int(m.group(group_idx))
            if value != expected_count:
                line_no = text[:m.start()].count("\n") + 1
                transition_pending = (rel, line_no, value) in _TRANSITION_ALLOWLIST
                mismatches.append({
                    "doc": rel,
                    "line": line_no,
                    "found": value,
                    "expected": expected_count,
                    "pattern": pattern,
                    "match_text": m.group(0)[:80],
                    "transition_pending": transition_pending,
                })
    return mismatches


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true", help="exit 1 on any mismatch")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    try:
        expected = count_registry_entries(REGISTRY_PATH)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    all_mismatches: list[dict] = []
    for doc_path, patterns in DOC_PATTERNS.items():
        all_mismatches.extend(check_doc(doc_path, expected, patterns))

    pending = [mm for mm in all_mismatches if mm.get("transition_pending")]
    errors = [
        mm for mm in all_mismatches
        if not mm.get("transition_pending") and "error" not in mm
    ]
    file_errors = [mm for mm in all_mismatches if "error" in mm]

    if args.json:
        result = {
            "expected_registry_count": expected,
            "mismatches": all_mismatches,
            "transition_pending": pending,
            "errors": errors + file_errors,
            "status": "FAIL" if errors or file_errors else "OK",
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"PATCH_REGISTRY count: {expected} (source: {REGISTRY_PATH.relative_to(REPO_ROOT)})")
        print()
        if not all_mismatches:
            print(f"✓ All checked docs claim {expected} patches consistently.")
        else:
            if errors or file_errors:
                print(f"✗ {len(errors) + len(file_errors)} unallowlisted mismatch(es):")
                for mm in file_errors:
                    print(f"  {mm['doc']}: {mm['error']}")
                for mm in errors:
                    print(f"  {mm['doc']}:{mm['line']}  found={mm['found']}  expected={mm['expected']}")
                    print(f"    match: {mm['match_text']}")
            if pending:
                print(
                    f"\n⚠ {len(pending)} known-stale site(s) PENDING fix by "
                    "CONFIG-HYGIENE.docs-reconcile.1.MECHANICAL:"
                )
                for mm in pending:
                    print(
                        f"  {mm['doc']}:{mm['line']}  found={mm['found']}  expected={mm['expected']}"
                    )
            if not errors and not file_errors:
                print(
                    f"\n✓ No unallowlisted drift; {len(pending)} site(s) pending mechanical fix."
                )

    if (errors or file_errors) and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
