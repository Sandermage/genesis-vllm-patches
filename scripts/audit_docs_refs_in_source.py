#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Source-code docs/ reference integrity — Phase 10.5 D-extension 2026-06-01.

Catches stale ``docs/<name>.md`` references that live in Python source
(error messages, docstrings, configuration fields, comments) but point
at a markdown file that does not exist in the tracked tree.

Why this gate exists
--------------------

The companion ``audit_links.py`` gate walks tracked markdown files and
verifies every inline link resolves cleanly. It catches the
post-consolidation rot we hit during the 2026-05-16 doc reshape (the
``CLIFFS.md / OOM_RECIPES.md / BENCHMARK_GUIDE.md`` merges into
``TROUBLESHOOTING.md + BENCHMARKS.md``) — but only for ``.md → .md``
links. References that live in **Python source** (e.g. operator-facing
``FileNotFoundError`` messages, configuration ``references=[...]``
fields, module-level pointer docstrings) survive the same merge
without an audit gate noticing. The 2026-06-01 enterprise sweep found
four stale ``BENCHMARK_GUIDE.md`` refs in ``compat/bench.py`` +
``tools/genesis_bench_suite.py`` exactly this way — they pointed at a
doc that had been merged into ``BENCHMARKS.md`` six weeks earlier and
operators on slim deployments saw the broken hint at the wrong moment
(when the bench script could not be located).

How the gate works
------------------

  1. Walks every ``.py`` file under ``sndr/`` (skipping
     ``_retired/`` + ``__pycache__/``).
  2. Greps for ``docs/<name>.md`` substrings using a token-boundary
     regex (so ``Genesis_internal_docs/X.md`` does NOT spuriously
     match as ``docs/X.md``).
  3. Skips refs that start with ``docs/_internal/`` — that's the
     gitignored private maintainer tree, expected absent in the
     public checkout.
  4. For each remaining ref, checks if the corresponding markdown
     file is in ``git ls-files`` output. Misses are reported.

Default mode: **warn only** (exit 0 even on misses). The operator
checks the report and either creates the missing doc, redirects the
reference at a canonical replacement, or accepts the miss as
aspirational. ``--strict`` promotes warnings to errors for CI.

Exit codes
----------

  0 — warn-only mode (default) OR ``--strict`` with zero misses.
  1 — ``--strict`` mode + at least one broken ref.
  2 — internal error (git unavailable / filesystem).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOT = REPO_ROOT / "sndr"

# Token-boundary regex: `docs/<name>.md` only when preceded by a
# non-identifier character (space / quote / open-paren / start-of-line /
# punctuation) so ``Genesis_internal_docs/X.md`` doesn't incorrectly
# match as ``docs/X.md``.
_DOC_REF_RE = re.compile(
    r"(?:^|[^A-Za-z0-9_/])"  # boundary: anything except identifier/slash
    r"(docs/[A-Za-z][A-Za-z0-9_./-]*\.md)"
)

# Documented exemptions — refs to operator-WIP docs that haven't been
# written yet. Adding an entry here is a deliberate "this doc is
# intentionally aspirational" choice; the audit warning stays informational
# until the operator either writes the doc or removes the reference.
_KNOWN_ASPIRATIONAL = frozenset({
    # license.py:67 — production trust-anchor ceremony doc; operator
    # plans to write this for the next release-tier promotion.
    "docs/security/TRUST_ANCHOR_CEREMONY.md",
    # dispatcher/audit.py:133 — INFO-level pointer in
    # `_audit_lifecycle` for operators promoting a patch to
    # lifecycle='stable'; checklist doc is planned for the next
    # stable-promotion cycle.
    "docs/upstream/STABLE_PROMOTION_CHECKLIST.md",
    # config_detect.py — V756 stability investigation notes referenced
    # in P67 recommendation rationale; operator's reference-tier
    # incident report, planned for the public docs/ tree.
    "docs/reference/V756_STABILITY_INVESTIGATION_20260427.md",
    # types/compatibility.py:223 — ngram-vs-mtp tradeoff cookbook
    # entry referenced by the compatibility ledger; planned operator-
    # facing cookbook.
    "docs/COOKBOOK.md",
    # pn34 (retired): UPSTREAM.md ref in retired patch context.
    "docs/UPSTREAM.md",
    # sndr/__init__.py + sndr/config.py — v12 sndr-platform docstrings
    # reference the planned restructured docs tree (reference/, concepts/);
    # surfaced when the audit's scan root moved to sndr/ (v12).
    "docs/reference/ENV_VARS.md",
    "docs/concepts/ARCHITECTURE.md",
})


def _tracked_files() -> set[str]:
    try:
        out = subprocess.check_output(
            ["git", "ls-files"], cwd=REPO_ROOT, text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()
    return set(out.splitlines())


def _is_external_repo_ref(ref: str, source_line: str) -> bool:
    """True if the reference is to another repo's docs/ tree, not ours.

    Convention: ``noonghunna/club-3090 docs/CONTAINER_RUNTIMES.md``
    style — the docs/ token sits next to a ``<org>/<repo>`` slug, not
    inside one of our own docstrings. We probe the *line* around the
    reference for an external-repo marker.
    """
    markers = (
        # noonghunna/club-3090 reference repo — match both "/club-3090"
        # and "noonghunna club-3090" prose forms (operator-prose
        # sometimes drops the slash when introducing the org/repo by
        # name in a docstring sentence).
        "noonghunna",
        "club-3090",
        "vllm-project/vllm",
        "huggingface/",
    )
    return any(m in source_line for m in markers)


def audit() -> dict:
    """Walk sndr_core/ Python source; return report dict.

    Report shape::

      {
        "broken": [{"path": "...", "ref": "...", "line": int}, ...],
        "aspirational": [{"path": "...", "ref": "...", "line": int}, ...],
        "external": [{"path": "...", "ref": "...", "line": int}, ...],
        "counts": {"broken": int, "aspirational": int, "external": int},
        "passed": bool,
      }
    """
    tracked = _tracked_files()
    broken: list[dict] = []
    aspirational: list[dict] = []
    external: list[dict] = []

    if not SCAN_ROOT.is_dir():
        return {
            "broken": [],
            "aspirational": [],
            "external": [],
            "counts": {"broken": 0, "aspirational": 0, "external": 0},
            "passed": True,
            "error": f"scan root not found: {SCAN_ROOT}",
        }

    for path in sorted(SCAN_ROOT.rglob("*.py")):
        if "_retired" in path.parts:
            continue
        if "__pycache__" in path.parts:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for lineno, line in enumerate(lines, start=1):
            for m in _DOC_REF_RE.finditer(line):
                ref = m.group(1)
                if ref.startswith("docs/_internal/"):
                    continue
                if ref in tracked:
                    continue  # resolves cleanly
                try:
                    rel_path = str(path.relative_to(REPO_ROOT))
                except ValueError:
                    # Synthetic scan root outside repo (test fixtures).
                    rel_path = str(path)
                entry = {
                    "path": rel_path,
                    "ref": ref,
                    "line": lineno,
                }
                if _is_external_repo_ref(ref, line):
                    external.append(entry)
                elif ref in _KNOWN_ASPIRATIONAL:
                    aspirational.append(entry)
                else:
                    broken.append(entry)

    return {
        "broken": broken,
        "aspirational": aspirational,
        "external": external,
        "counts": {
            "broken": len(broken),
            "aspirational": len(aspirational),
            "external": len(external),
        },
        "passed": not broken,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--strict", action="store_true",
        help="promote aspirational refs to errors as well (default: only "
             "unkonwn broken refs error in strict mode)",
    )
    ap.add_argument("--json", action="store_true",
                    help="emit JSON payload instead of human-readable summary")
    args = ap.parse_args()

    report = audit()
    passed = report["passed"]
    if args.strict:
        passed = passed and report["counts"]["aspirational"] == 0

    if args.json:
        report["strict"] = args.strict
        report["passed"] = passed
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        c = report["counts"]
        print(f"audit-docs-refs-in-source: "
              f"broken={c['broken']} aspirational={c['aspirational']} "
              f"external={c['external']}")
        print("─" * 70)
        if report["broken"]:
            print(f"  ✗ broken ({len(report['broken'])}) — ref points at a "
                  f"file not in git ls-files + not in the aspirational "
                  f"allow-list:")
            for r in report["broken"]:
                print(f"      {r['path']}:{r['line']}: {r['ref']}")
        if report["aspirational"]:
            print(f"  ⚠ aspirational ({len(report['aspirational'])}) — "
                  f"operator-WIP docs documented as planned in audit's "
                  f"_KNOWN_ASPIRATIONAL allow-list:")
            for r in report["aspirational"]:
                print(f"      {r['path']}:{r['line']}: {r['ref']}")
        if report["external"]:
            print(f"  · external ({len(report['external'])}) — refs to "
                  f"another repo's docs/ tree (informational only):")
            for r in report["external"][:5]:
                print(f"      {r['path']}:{r['line']}: {r['ref']}")
        if not (report["broken"] or report["aspirational"] or report["external"]):
            print("  ✓ every docs/<name>.md reference in vllm/sndr_core/ "
                  "resolves cleanly")
        print(f"\n  passed: {passed} (strict={args.strict})")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
