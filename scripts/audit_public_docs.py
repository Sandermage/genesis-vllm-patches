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
  D-7  No stale-as-current Genesis version anchors. Specific phrasings
       like "(canonical, v11.0.0+)", "Repository layout (v11.0.0)", or
       "v7.5x stack tested" present a non-current version as if it
       were the current PROD baseline. Pure historical attribution
       (CHANGELOG entries, CREDITS attributions, "Removed in v11.0.0")
       is intentionally allowed.
  D-8  No stale-as-current vLLM pin anchors in operator-facing text.
       Specific phrasings like "currently `0.20.1rc1.dev16+gXXXX`",
       "pip install vllm==0.20.1rc1.dev16+gXXXX", or "Not in nightly
       image as of dev93+gXXXX" claim a pre-current pin as the active
       baseline. CHANGELOG history sections and CREDITS attribution
       are exempt.

Allowlist (intentionally private / historical):
  sndr_private/, _archive/

Transition allowlists (D-7, D-8): see `_D7_TRANSITION_ALLOWLIST` and
`_D8_TRANSITION_ALLOWLIST`. Both MUST be emptied by
`CONFIG-HYGIENE.docs-reconcile.1.MECHANICAL`.

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
    # v12 (2026-06-05+): maintainer session journals / specs / ops
    # playbooks written by the superpowers workflow. They are working
    # engineering logs (rig IPs, operator paths, SSH transcripts are
    # their subject matter), i.e. the successor of the docs/_internal/
    # session-log class — NOT operator-facing public documentation.
    # The D-1..D-8 boundary rules do not apply inside this tree.
    "docs/superpowers/",
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


# P0.1 M.4/M.6 (2026-05-24): two public docs define the three-zone
# namespace policy and therefore must mention `sndr_private/` by
# name — that's their entire purpose. Exempt them from D-1 only.
# All other public docs continue to be blocked from referencing the
# private tree. Per-file exemption (not regex relaxation) preserves
# D-1's strength for every other doc.
_D1_DEFINITIONAL_EXEMPT = (
    "docs/CORE_ENGINE_BOUNDARY.md",
    "docs/LICENSE_POLICY.md",
)


def check_d1_no_internal_links(files: list[Path]) -> list[str]:
    """D-1: public docs must not reference the private maintainer tree
    (`sndr_private/` is the canonical location post-consolidation;
    `docs/_internal/` is the retired legacy path, kept in the regex
    so a regression cannot silently re-introduce it).

    Exempt: the two policy docs that DEFINE the three-zone boundary
    (`CORE_ENGINE_BOUNDARY.md`, `LICENSE_POLICY.md`) must name the
    private path to do their job."""
    filtered = [
        fp for fp in files
        if fp.relative_to(REPO_ROOT).as_posix() not in _D1_DEFINITIONAL_EXEMPT
    ]
    return _grep(re.compile(r"sndr_private/|docs/_internal"), filtered)


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


# CONFIG-HYGIENE.docs-reconcile.1.GATE-EXTEND (2026-05-24):
# D-7 / D-8 transition allowlists. Each entry is `(rel_path, line_no)`
# and suppresses the corresponding finding for that exact site at a
# transition commit. EMPTIED by
# CONFIG-HYGIENE.docs-reconcile.1.MECHANICAL (2026-05-24).
# Scaffolding retained (as empty frozensets) for the next pin-bump or
# version-bump cycle: operators can add entries here without
# redesigning the gate or touching pattern logic.
_D7_TRANSITION_ALLOWLIST: frozenset[tuple[str, int]] = frozenset()

_D8_TRANSITION_ALLOWLIST: frozenset[tuple[str, int]] = frozenset()

# Files where stale version / pin tokens are intentionally historical
# (engineering log, attribution log) — exempt at file level from D-7/D-8.
_D7_D8_FILE_EXEMPT = (
    "CHANGELOG.md",
    "docs/CREDITS.md",
)

# Permanent line-level exemption for D-7/D-8: lines whose entire purpose
# is historical attribution and where the stale token is intentional.
# Distinct from the transition allowlists — these entries are NOT
# expected to be cleared by `CONFIG-HYGIENE.docs-reconcile.1.MECHANICAL`.
_D7_PERMANENT_EXEMPT: frozenset[tuple[str, int]] = frozenset()

_D8_PERMANENT_EXEMPT: frozenset[tuple[str, int]] = frozenset({
    # "> Previous v7.59 baseline (2026-04-28): vLLM dev212+g8cd174fa3 era —"
    # Explicit "Previous" prefix — historical baseline marker.
    ("docs/CONFIGURATION.md", 44),
})

# The current canonical pin SHA. The D-8 stale-pin pattern uses `dev1\d+`
# to catch pre-current dev1xx-era pins presented as current; the prior
# canonical pin (dev148+gb4c80ec0f) also matched `dev1\d+`, so its SHA used
# to be excluded via negative lookahead. The current canonical pin is now
# dev424+g3f5a1e173 (ratified 2026-06-25) — it does NOT match `dev1\d+`, so
# the exclusion is a no-op for the live pin, but the constant is kept
# pointing at the current SHA per convention. Update this on each pin bump.
_D8_CURRENT_PIN_SHA = "g3f5a1e173"


def _grep_with_allowlist(
    pattern: re.Pattern,
    files: list[Path],
    allowlist: frozenset[tuple[str, int]],
    file_exempt: tuple[str, ...] = (),
) -> list[str]:
    """Like `_grep` but applies a `(rel_path, line_no)` allowlist + a
    file-level exempt list. Used by D-7 and D-8 to suppress known
    transition-state sites without weakening pattern strength."""
    hits = []
    for fp in files:
        rel = fp.relative_to(REPO_ROOT).as_posix()
        if rel in file_exempt:
            continue
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if "audit-public-docs: allow" in line:
                continue
            if (rel, i) in allowlist:
                continue
            if pattern.search(line):
                hits.append(f"{rel}:{i}: {line.strip()[:120]}")
    return hits


def check_d7_no_stale_version_as_current(files: list[Path]) -> list[str]:
    """D-7: stale-as-current Genesis version anchors.

    Specifically flags phrasings that claim a non-current version
    (v7.5x or v11.0.x) as if it were the active PROD baseline:

      - "(canonical, v11.0.0+)"
      - "Repository layout (v11.0.0)"
      - "v7.5x stack tested"
      - "install.sh ... --pin v11.0"

    Historical references ("Removed in v11.0.0", "renamed in v11.0.0",
    "pre-v11 scripts") are not matched — they describe past events
    accurately. CHANGELOG.md and docs/CREDITS.md are file-level exempt.

    Transition allowlist `_D7_TRANSITION_ALLOWLIST` suppresses the four
    known stale-as-current sites in docs/INSTALL.md that
    `CONFIG-HYGIENE.docs-reconcile.1.MECHANICAL` will fix.
    """
    pat = re.compile(
        r"\(canonical,\s*v(?:7\.\d+|11\.\d)"
        r"|Repository layout\s+\(v(?:7\.\d+|11\.\d)"
        r"|\bv7\.\d+\s+stack\s+tested\b"
        r"|--pin\s+v11\.\d"
    )
    return _grep_with_allowlist(
        pat,
        files,
        _D7_TRANSITION_ALLOWLIST | _D7_PERMANENT_EXEMPT,
        _D7_D8_FILE_EXEMPT,
    )


def check_d8_no_stale_pin_as_current(files: list[Path]) -> list[str]:
    r"""D-8: stale-as-current vLLM pin anchors.

    Flags pre-current pins (dev16, dev93, dev209, dev212) presented in
    current-state phrasings:

      - "currently `0.20.1rc1.dev16+gXXXX`"
      - "pip install --pre vllm==0.20.1rc1.dev16+gXXXX"
      - "# vllm 0.20.1rc1.dev16+gXXXX" (comment)
      - "Not in nightly image as of dev93+gXXXX" (active-claim about
        current upstream-merge state)
      - "vLLM dev212+g..." in hardware "Primary tested" claim

    The current canonical pin is `0.23.1rc1.dev148+gb4c80ec0f` (per
    docs/USAGE.md, docs/QUICKSTART.md, docs/BENCHMARKS.md). Its SHA is
    excluded from the stale-pin pattern via `_D8_CURRENT_PIN_SHA` — it is
    the active pin, not a stale one (it would otherwise match `dev1\d+`).
    Pre-current pins mentioned in historical context (CHANGELOG.md,
    docs/CREDITS.md, or BENCHMARKS.md "Wave 7 / v7.72 (dev9) snapshot") are
    intentionally allowed via file-level exempt OR by being phrased without
    current-state markers.

    Transition allowlist `_D8_TRANSITION_ALLOWLIST` suppresses the
    known stale-as-current sites in docs/INSTALL.md and docs/PATCHES.md
    that `CONFIG-HYGIENE.docs-reconcile.1.MECHANICAL` will fix.
    """
    # Exclude the current canonical pin SHA: dev148 matches `dev1\d+`, but
    # it is the active pin (not stale), so a negative lookahead lets the
    # legitimate current-state references through while still catching
    # genuinely stale dev1xx-era pins.
    nf = rf"(?!\d+\.\d+\.\d+rc\d+\.dev1\d+\+{re.escape(_D8_CURRENT_PIN_SHA)})"
    pat = re.compile(
        rf"currently\s+`?{nf}\d+\.\d+\.\d+rc\d+\.dev1\d+\+g"
        rf"|vllm=={nf}\d+\.\d+\.\d+rc\d+\.dev1\d+\+g"
        rf"|^#\s*vllm\s+{nf}\d+\.\d+\.\d+rc\d+\.dev1\d+\+g"
        r"|Not in nightly image as of dev\d+\+g"
        r"|vLLM\s+dev21[0-9]\+g[a-f0-9]+"
    )
    return _grep_with_allowlist(
        pat,
        files,
        _D8_TRANSITION_ALLOWLIST | _D8_PERMANENT_EXEMPT,
        _D7_D8_FILE_EXEMPT,
    )


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
        "D-7 no stale-version-as-current": check_d7_no_stale_version_as_current(files),
        "D-8 no stale-pin-as-current": check_d8_no_stale_pin_as_current(files),
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
