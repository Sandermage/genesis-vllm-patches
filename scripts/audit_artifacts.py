#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 7 release gate — `make audit-artifacts`.

Implements PROJECT_ROADMAP_V2 §6.11 artifact storage policy. Each artefact
class has a defined location, tracked/untracked rule, and release inclusion
flag. Drift = release blocked.

Checks:

  A-1  evidence ledger exists at docs/_internal/ROADMAP_EVIDENCE_LEDGER_*.md
  A-2  evidence/patch_proof/ — if present, contains only *.json + _waivers/
  A-3  release/ — when public-release flag set, must contain SBOM + constraints
  A-4  ~/.sndr/bench-results/ — informational only (not in git)
  A-5  No bench-result JSON under git-tracked paths (~/.sndr/* belongs outside repo)
  A-6  ROLLBACK_PLAYBOOK.md present at docs/ROLLBACK_PLAYBOOK.md

Exit code:
  0 — all checks pass.
  1 — at least one check failed.
  2 — internal error.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    )
    return [ln for ln in result.stdout.splitlines() if ln]


def check_evidence_ledger_present() -> list[str]:
    """A-1: evidence ledger MD must exist when a maintainer planning
    tree is checked out alongside the public repo.

    The ledger lives outside the public source tree by design — public
    clones don't carry it, and the gate is a no-op on those checkouts.
    Maintainers point the gate at their planning dir via the
    ``GENESIS_PLANNING_DIR`` env var; if it's unset and the legacy
    ``docs/_internal/`` path is also absent, the gate skips silently.
    When a planning dir IS visible to the gate, at least one
    ``ROADMAP_EVIDENCE_LEDGER_*.md`` must live in it."""
    candidates: list[Path] = []
    env_dir = os.environ.get("GENESIS_PLANNING_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(REPO_ROOT / "docs" / "_internal")  # legacy fallback
    if not any(b.is_dir() for b in candidates):
        return []
    for base in candidates:
        if base.is_dir() and list(base.glob("ROADMAP_EVIDENCE_LEDGER_*.md")):
            return []
    return [
        "A-1: maintainer planning tree visible but contains no "
        "ROADMAP_EVIDENCE_LEDGER_*.md"
    ]


def check_patch_proof_layout() -> list[str]:
    """A-2: if evidence/patch_proof/ exists, it contains *.json + _waivers/
    only. `.gitkeep` is exempt — it preserves the otherwise-gitignored
    directory in a fresh clone so operators can run `sndr patches prove
    --all` into it without first having to `mkdir -p`."""
    issues = []
    pp = REPO_ROOT / "evidence" / "patch_proof"
    if not pp.is_dir():
        return []  # Optional — release tier creates it; dev may not have it.
    for child in pp.iterdir():
        if child.is_dir():
            if child.name != "_waivers":
                issues.append(
                    f"A-2: unexpected directory in evidence/patch_proof/: {child.name}"
                )
        elif child.is_file():
            if child.name == ".gitkeep":
                continue
            if child.suffix != ".json":
                issues.append(
                    f"A-2: non-JSON file in evidence/patch_proof/: {child.name}"
                )
    return issues


def check_release_artefacts_present(public_release: bool) -> list[str]:
    """A-3: release tier requires SBOM + constraints under release/."""
    if not public_release:
        return []
    issues = []
    for name in ("SBOM.spdx.json", "constraints.txt"):
        if not (REPO_ROOT / "release" / name).exists():
            issues.append(f"A-3: missing release artefact: release/{name}")
    return issues


def check_no_bench_results_tracked(files: list[str]) -> list[str]:
    """A-5: bench results belong to ~/.sndr/bench-results/, never git."""
    bad = []
    for f in files:
        if "bench-results" in f and f.endswith(".json"):
            bad.append(f"A-5: bench-result JSON tracked in git: {f}")
    return bad


def check_rollback_playbook_present() -> list[str]:
    """A-6: rollback playbook must be reachable from public docs.

    Phase 2.3 deliverable originally lived at docs/ROLLBACK_PLAYBOOK.md.
    After the 2026-05-16 docs consolidation, the rollback procedures
    were merged into the broader docs/TROUBLESHOOTING.md (sections
    'Rollback playbook' + the named R-001..R-008 procedures). The
    gate now passes when EITHER path is present + contains the R-001
    procedure anchor that identifies the canonical content.
    """
    legacy = REPO_ROOT / "docs" / "ROLLBACK_PLAYBOOK.md"
    consolidated = REPO_ROOT / "docs" / "TROUBLESHOOTING.md"
    for path in (legacy, consolidated):
        if path.is_file() and "R-001" in path.read_text(encoding="utf-8"):
            return []
    return [
        "A-6: rollback playbook missing — expected docs/ROLLBACK_PLAYBOOK.md "
        "OR docs/TROUBLESHOOTING.md containing the R-001..R-008 procedures"
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--public-release", action="store_true",
                    help="Strict mode (A-3 release artefacts check).")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    try:
        files = _git_tracked_files()
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"git ls-files failed: {e}\n")
        return 2

    checks = {
        "A-1 evidence ledger": check_evidence_ledger_present(),
        "A-2 patch proof layout": check_patch_proof_layout(),
        "A-3 release artefacts": check_release_artefacts_present(args.public_release),
        "A-5 bench-results not tracked": check_no_bench_results_tracked(files),
        "A-6 rollback playbook": check_rollback_playbook_present(),
    }
    total_failures = sum(len(v) for v in checks.values())

    if args.json:
        print(json.dumps(
            {"checks": checks, "total_failures": total_failures,
             "files_scanned": len(files), "public_release": args.public_release},
            indent=2, sort_keys=True,
        ))
    else:
        print(f"audit-artifacts: {len(files)} tracked files scanned"
              + (" (public-release mode)" if args.public_release else ""))
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
        if total_failures:
            print(f"  FAIL — {total_failures} total violation(s)")
        else:
            print("  OK — artefact storage policy passes")
    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
