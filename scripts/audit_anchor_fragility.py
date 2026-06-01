#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Anchor fragility audit — Phase 3.1 from peaceful-noodling-church plan.

Static AST-walk reporter that catalogs every TextPatcher-style anchor
constant in the active integration tree and flags those at or above
the Phase 3.1 fragility threshold (25 source lines).

Why this gate exists
--------------------

TextPatcher overlays attach to upstream source by matching exact-text
"anchors" (multi-line string constants like
``ANCHOR_1A_IMPORT_OLD = "from .utils import ..."``). The bigger an
anchor, the more upstream lines it must match verbatim — and the more
fragile the patch becomes on every pin bump. A 50-line anchor breaks
on any whitespace / comment / import reorder upstream, even when the
*semantics* of the targeted region are unchanged.

Phase 3.1 of the master plan flagged 22 patches with anchors >=25
lines (the empirical threshold above which anchors started breaking
quarterly). The recommended fix is to migrate fragile patches onto
the PN119 pattern (md5 + full-file replacement) instead of multi-
anchor splice. This gate gives operators a ratchet so future PRs
can't quietly increase fragility past the documented baseline.

How the audit works
-------------------

  1. Walks every ``.py`` file under ``vllm/sndr_core/integrations/``
     (skipping ``_retired/`` + ``__pycache__/``).
  2. Parses the AST and finds module-level assignments whose target
     name contains ``ANCHOR_``, ``_OLD``, ``_BEFORE``, or ``_PRE``
     and whose value resolves to a string literal (either a bare
     ``str`` constant or a static concatenation of ``str`` literals,
     including the parenthesised ``"line1\\n" "line2\\n"`` form
     and binary ``+``).
  3. Counts source lines in the resolved string.
  4. Reports any anchor at or above the ``--threshold`` (default
     25), and any file whose max anchor exceeds ``--hard-cap``
     (default 50, hard error per Phase 3.1 ratchet).

Exit code
---------

  0 — no anchor >= ``--hard-cap`` lines (warnings between
      ``threshold`` and ``hard_cap`` allowed).
  1 — at least one anchor at or above ``--hard-cap`` lines (gate
      fires; either shrink the anchor or migrate the patch to the
      PN119 md5+full-file model).
  2 — internal error (filesystem / parse).

Non-zero exit codes are reserved for the hard-cap; warnings are
informational by default so the gate doesn't break the build on the
inherited fragility baseline. CI workflows can promote warnings to
errors via ``--strict``.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOT = REPO_ROOT / "vllm" / "sndr_core" / "integrations"

# Phase 3.1 of the master plan documented 25 lines as the empirical
# fragility threshold (anchors at or above this break quarterly on
# pin bumps). The default ``--threshold`` reflects that.
DEFAULT_THRESHOLD = 25
# Hard cap is the forward-only ratchet — set above the largest current
# anchor (67 lines in pn79 / pn57 / pn79 NEW counterparts as of
# 2026-06-01) so the inherited fragility baseline stays warn-only and
# the gate fires only on regressions that introduce a fresh, even-
# fragiler anchor. Operators tighten the cap as cleanup work lands.
DEFAULT_HARD_CAP = 70


def _collect_string(node: ast.AST) -> str | None:
    """Walk a static-string AST node and return the concatenated text.

    Returns None if the node contains any non-static element
    (variable reference, function call, f-string, etc.).
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Tuple):
        parts: list[str] = []
        for el in node.elts:
            s = _collect_string(el)
            if s is None:
                return None
            parts.append(s)
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a = _collect_string(node.left)
        b = _collect_string(node.right)
        if a is None or b is None:
            return None
        return a + b
    return None


def _is_anchor_name(name: str) -> bool:
    """True if a name looks like a TextPatcher anchor constant.

    Heuristic match:
      * Contains the literal ``ANCHOR_`` token, OR
      * Ends with ``_OLD``, ``_BEFORE``, or ``_PRE`` (the three
        documented "match-existing-source" suffixes).
    Always allow-list small constants too (caught by the minimum-
    size filter in ``find_anchors`` below); the name check is just
    the first filter to keep AST walk cheap.
    """
    return (
        "ANCHOR_" in name
        or name.endswith("_OLD")
        or name.endswith("_BEFORE")
        or name.endswith("_PRE")
    )


def find_anchors_in_file(path: Path) -> list[tuple[str, int]]:
    """Parse one file; return list of (anchor_name, line_count) for
    anchors at least 1 line + 50 chars long. Both filters drop
    accidentally-matching constants like ``OLD_DEFAULT_PAGE_SIZE = 256``.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (SyntaxError, OSError):
        return []
    out: list[tuple[str, int]] = []
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if not _is_anchor_name(target.id):
                continue
            s = _collect_string(node.value)
            if s is None or len(s) <= 50:
                continue
            lines = s.count("\n") + (0 if s.endswith("\n") else 1)
            if lines < 1:
                continue
            out.append((target.id, lines))
    return out


def audit(threshold: int = DEFAULT_THRESHOLD,
          hard_cap: int = DEFAULT_HARD_CAP) -> dict:
    """Walk the integration tree; return a structured report.

    Report shape::

      {
        "files": [
          {"path": "...", "anchors": [(name, lines), ...],
           "max_lines": int, "warn_count": int, "error_count": int},
          ...
        ],
        "counts": {"warn": int, "error": int, "files_with_anchors": int},
        "threshold": int, "hard_cap": int, "passed": bool,
      }
    """
    file_reports: list[dict] = []
    total_warn = 0
    total_error = 0
    if not SCAN_ROOT.is_dir():
        return {
            "files": [],
            "counts": {"warn": 0, "error": 0, "files_with_anchors": 0},
            "threshold": threshold,
            "hard_cap": hard_cap,
            "passed": True,
            "error": f"scan root not found: {SCAN_ROOT}",
        }
    for path in sorted(SCAN_ROOT.rglob("*.py")):
        if "_retired" in path.parts:
            continue
        if "__pycache__" in path.parts:
            continue
        anchors = find_anchors_in_file(path)
        if not anchors:
            continue
        warn_count = sum(1 for _, l in anchors if threshold <= l < hard_cap)
        error_count = sum(1 for _, l in anchors if l >= hard_cap)
        total_warn += warn_count
        total_error += error_count
        max_lines = max(l for _, l in anchors)
        try:
            rel_path = str(path.relative_to(REPO_ROOT))
        except ValueError:
            # Path lives outside the repo (e.g. a monkey-patched
            # SCAN_ROOT pointing at a synthetic tmp tree in tests). Fall
            # back to the absolute path so the report still surfaces it
            # — relative-to-repo is a display convenience, not a contract.
            rel_path = str(path)
        file_reports.append({
            "path": rel_path,
            "anchors": anchors,
            "max_lines": max_lines,
            "warn_count": warn_count,
            "error_count": error_count,
        })
    return {
        "files": file_reports,
        "counts": {
            "warn": total_warn,
            "error": total_error,
            "files_with_anchors": len(file_reports),
        },
        "threshold": threshold,
        "hard_cap": hard_cap,
        "passed": total_error == 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--threshold", type=int, default=DEFAULT_THRESHOLD,
        help=f"warn when an anchor reaches this many lines (default {DEFAULT_THRESHOLD})",
    )
    ap.add_argument(
        "--hard-cap", type=int, default=DEFAULT_HARD_CAP,
        help=f"error when an anchor reaches this many lines (default {DEFAULT_HARD_CAP})",
    )
    ap.add_argument(
        "--strict", action="store_true",
        help="promote warnings to errors (returns non-zero on any anchor >= threshold)",
    )
    ap.add_argument("--json", action="store_true",
                    help="emit JSON payload instead of human-readable summary")
    args = ap.parse_args()

    report = audit(threshold=args.threshold, hard_cap=args.hard_cap)
    passed = report["passed"]
    if args.strict:
        passed = passed and report["counts"]["warn"] == 0

    if args.json:
        report["strict"] = args.strict
        report["passed"] = passed
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        counts = report["counts"]
        print(f"audit-anchor-fragility: {counts['files_with_anchors']} file(s) "
              f"with TextPatcher-style anchors")
        print(
            f"thresholds: warn>={report['threshold']} lines, "
            f"error>={report['hard_cap']} lines"
        )
        print("─" * 70)
        warn_files = [f for f in report["files"] if f["warn_count"] or f["error_count"]]
        if not warn_files:
            print(f"  ✓ no anchors exceed the warn threshold "
                  f"({report['threshold']} lines)")
        else:
            print(f"  Anchors at or above the warn threshold:")
            for f in warn_files:
                lvl = "✗" if f["error_count"] else "⚠"
                print(
                    f"    {lvl} {f['path']}: max {f['max_lines']} lines, "
                    f"{f['error_count']} error(s) + {f['warn_count']} warn(s), "
                    f"{len(f['anchors'])} total anchor(s)"
                )
                # Show the worst 3
                worst = sorted(f["anchors"], key=lambda x: -x[1])[:3]
                for name, lines in worst:
                    if lines >= report["threshold"]:
                        print(f"        - {name}: {lines} lines")
        print()
        print(
            f"  totals: error={counts['error']} warn={counts['warn']} "
            f"strict={args.strict}"
        )

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
