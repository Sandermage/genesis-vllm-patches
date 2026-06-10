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
# v12.x moved the registry to sndr/dispatcher/registry.py; the old
# vllm/sndr_core path is now a re-export shim with no literal to count.
REGISTRY_PATH = REPO_ROOT / "sndr" / "dispatcher" / "registry.py"
if not REGISTRY_PATH.is_file():
    REGISTRY_PATH = REPO_ROOT / "sndr" / "dispatcher" / "registry.py"

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
    # Phase 10.5 extension (2026-06-01): USAGE.md previously drifted
    # silently (231 → 236 stale by 5 entries) because the file wasn't
    # in this allowlist. Patterns target the three count-claim sites:
    # the "Stack as of …" header bullet, the "applies N small surgical
    # changes" prose intro, and the "## 5. Patches …" section opener.
    REPO_ROOT / "docs" / "USAGE.md": [
        (r"Genesis `v12\.0\.0` \((\d+) PATCH_REGISTRY entries\)", 1),
        (r"applies (\d+) small surgical", 1),
        (r"The patch registry is the heart of Genesis\. (\d+) entries live in", 1),
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


def compute_registry_stats(registry_path: Path) -> dict[str, int]:
    """Per-field counts parsed from registry.py source (torch-less).

    Splits the file at each top-level entry boundary, then extracts
    `lifecycle` / `implementation_status` / `apply_module` from each
    block. Used by `_check_patches_md_stats_table()` to validate the
    hand-maintained Quick-stats table in docs/PATCHES.md against
    current ground truth.

    Returns dict with: total, lifecycle.<value>, impl.<value>,
    apply_module_set, apply_module_none.
    """
    text = registry_path.read_text()
    # Anchor splits at top-level entries: `    "PATCH_ID": {`
    anchors = list(re.finditer(r"^    \"([A-Za-z0-9_\-]+)\":\s*\{", text, flags=re.M))
    blocks: list[tuple[str, str]] = []
    for i, m in enumerate(anchors):
        start = m.end()
        end = anchors[i + 1].start() if i + 1 < len(anchors) else len(text)
        blocks.append((m.group(1), text[start:end]))

    from collections import Counter
    life = Counter()
    impl = Counter()
    apply_set = 0
    apply_none = 0
    for _pid, body in blocks:
        # lifecycle: "value"
        m_life = re.search(r"\"lifecycle\":\s*\"([A-Za-z_]+)\"", body)
        life[m_life.group(1) if m_life else "experimental"] += 1
        m_impl = re.search(r"\"implementation_status\":\s*\"([A-Za-z_]+)\"", body)
        if m_impl:
            impl[m_impl.group(1)] += 1
        m_am = re.search(r"\"apply_module\":\s*([^,\n]+)", body)
        if m_am and m_am.group(1).strip() not in ("None", "None,"):
            apply_set += 1
        else:
            apply_none += 1

    stats = {"total": len(blocks), "apply_module_set": apply_set,
             "apply_module_none": apply_none}
    for k, v in life.items():
        stats[f"lifecycle.{k}"] = v
    for k, v in impl.items():
        stats[f"impl.{k}"] = v
    return stats


# Patterns for the hand-maintained "Quick stats" table at the top of
# docs/PATCHES.md (the one that ~2 weeks of registry growth stale-
# drifted before catch). Each entry maps a regex (captures the
# claimed integer in group 1) to a callable that, given `stats` from
# `compute_registry_stats()`, returns the expected ground truth.
_PATCHES_MD_STATS_PATTERNS: list[tuple[str, str]] = [
    (r"\| Lifecycle=experimental \| (\d+) \|", "lifecycle.experimental"),
    (r"\| Lifecycle=legacy \(pre-dispatcher\) \| (\d+) \|", "lifecycle.legacy"),
    (r"\| Lifecycle=retired \| (\d+) \|", "lifecycle.retired"),
    (r"\| Lifecycle=research \| (\d+) \|", "lifecycle.research"),
    (r"\| Lifecycle=coordinator \| (\d+) ", "lifecycle.coordinator"),
    (r"\| Implementation status=full \| (\d+) \|", "impl.full"),
    (r"Apply-loop coverage \(apply_module set\) \| (\d+) / (\d+) ", "apply_module_set"),
]


# Per-doc impl_status breakdown patterns. Each entry is
# (relative_path, regex_with_6_capture_groups, label).
# Group order MUST match _IMPL_BREAKDOWN_KEYS:
#   (total, full, marker_only, retired, partial, placeholder).
#
# FAQ.md prose:  "**N entries** - X full-implementation + Y marker-only
#                + Z retired + W partial + V placeholder"
# CONFIGURATION.md prose: "registry has **N entries** (X full-
#                implementation + Y marker_only + Z retired + W partial
#                + V placeholder)"
# Drift caught 2026-06-01 — both forms had become stale (174/17/7
# vs 177/20/8) because earlier check_doc_sync only checked total count.
# Group order MUST match the doc prose order (below). v12.x docs added the
# `experimental` impl_status so the six categories now sum to total.
_IMPL_BREAKDOWN_KEYS = [
    "total", "impl.full", "impl.experimental", "impl.marker_only",
    "impl.partial", "impl.retired", "impl.placeholder",
]
_IMPL_BREAKDOWN_SOURCES: list[tuple[str, str]] = [
    (
        "docs/FAQ.md",
        # "**252 entries**: 192 full-implementation, 26 experimental,
        #  20 marker-only, 8 partial, 4 retired, 2 placeholder"
        r"\*\*(\d+) entries\*\*: (\d+) full-implementation,\s*(\d+) experimental,\s*"
        r"(\d+) marker-only,\s*(\d+) partial,\s*(\d+) retired,\s*(\d+) placeholder",
    ),
    (
        "docs/CONFIGURATION.md",
        # Same prose inside a markdown blockquote, so line breaks carry a
        # `> ` continuation marker — [\s>]* admits it between fields.
        r"registry has \*\*(\d+) entries\*\*: (\d+) full-implementation,[\s>]*"
        r"(\d+) experimental,[\s>]*(\d+) marker_only,[\s>]*(\d+) partial,[\s>]*"
        r"(\d+) retired,[\s>]*(\d+) placeholder",
    ),
]


def _check_impl_breakdowns(stats: dict[str, int]) -> list[dict]:
    """Verify the impl_status breakdown prose across each registered doc.

    Each source emits up to 6 mismatch records (one per bucket whose
    claimed value drifts from registry ground truth). A source whose
    prose form has been rewritten (regex no longer matches) emits a
    single 'pattern not found' record so the gate flags it loudly
    instead of silently ignoring the section.
    """
    mismatches: list[dict] = []
    for rel, pattern in _IMPL_BREAKDOWN_SOURCES:
        path = REPO_ROOT / rel
        if not path.is_file():
            mismatches.append({"doc": rel, "error": "file not found"})
            continue
        text = path.read_text()
        m = re.search(pattern, text)
        if not m:
            mismatches.append({
                "doc": rel, "line": 0, "found": None,
                "expected": stats["total"], "pattern": pattern,
                "match_text": "(prose form not found — wording changed?)",
                "transition_pending": False,
            })
            continue
        line_no = text[:m.start()].count("\n") + 1
        for i, key in enumerate(_IMPL_BREAKDOWN_KEYS):
            found = int(m.group(i + 1))
            expected = stats.get(key, 0)
            if found != expected:
                mismatches.append({
                    "doc": rel, "line": line_no, "found": found,
                    "expected": expected,
                    "pattern": f"{pattern} [{key}]",
                    "match_text": m.group(0)[:80],
                    "transition_pending": False,
                })
    return mismatches


def _check_patches_md_stats_table(stats: dict[str, int]) -> list[dict]:
    """Verify the Quick-stats table in docs/PATCHES.md matches stats.

    Reports each row that drifts (e.g. table claims `experimental=157`
    but registry has 162). The Apply-loop coverage row has two
    captures (numerator + denominator); both are checked.
    """
    path = REPO_ROOT / "docs" / "PATCHES.md"
    if not path.is_file():
        return [{"doc": str(path), "error": "file not found"}]
    text = path.read_text()
    rel = path.relative_to(REPO_ROOT).as_posix()
    mismatches: list[dict] = []
    for pattern, key in _PATCHES_MD_STATS_PATTERNS:
        m = re.search(pattern, text)
        if not m:
            mismatches.append({
                "doc": rel, "line": 0, "found": None,
                "expected": stats.get(key, 0), "pattern": pattern,
                "match_text": "(row not found — table structure changed?)",
                "transition_pending": False,
            })
            continue
        line_no = text[:m.start()].count("\n") + 1
        # apply_module pattern captures (numerator, denominator)
        if key == "apply_module_set":
            num = int(m.group(1))
            den = int(m.group(2))
            if num != stats["apply_module_set"]:
                mismatches.append({
                    "doc": rel, "line": line_no, "found": num,
                    "expected": stats["apply_module_set"],
                    "pattern": pattern, "match_text": m.group(0)[:80],
                    "transition_pending": False,
                })
            if den != stats["total"]:
                mismatches.append({
                    "doc": rel, "line": line_no, "found": den,
                    "expected": stats["total"],
                    "pattern": pattern + " [denominator]",
                    "match_text": m.group(0)[:80],
                    "transition_pending": False,
                })
            continue
        found = int(m.group(1))
        expected = stats.get(key, 0)
        if found != expected:
            mismatches.append({
                "doc": rel, "line": line_no, "found": found,
                "expected": expected, "pattern": pattern,
                "match_text": m.group(0)[:80],
                "transition_pending": False,
            })
    return mismatches


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

    # Phase 10.5 extension (2026-06-01): also validate the Quick-stats
    # table in docs/PATCHES.md (lifecycle / impl_status / apply_module
    # coverage rows) against per-field counts parsed from registry.
    try:
        stats = compute_registry_stats(REGISTRY_PATH)
        all_mismatches.extend(_check_patches_md_stats_table(stats))
        all_mismatches.extend(_check_impl_breakdowns(stats))
    except Exception as e:
        print(f"WARN: stats-table check skipped: {e}", file=sys.stderr)

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
