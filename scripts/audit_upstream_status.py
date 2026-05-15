#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Upstream PR status audit for iron-rule-#11 compliance.

Walks PATCH_REGISTRY, extracts every patch with `upstream_pr: N`,
queries GitHub for that PR's merge state, cross-references with the
patch's lifecycle / vllm_version_range / superseded_by fields, and
surfaces:

  - NEWLY-MERGED:    upstream merged BUT our lifecycle != "retired"
                     (iron-rule-#11 deep-diff queue)
  - STALE-RETIRED:   lifecycle == "retired" but upstream still OPEN
                     (premature retire? — investigate)
  - WATCH:           upstream still OPEN, our patch active (normal)
  - SUPERSEDED-OK:   already retired with provenance (no action)
  - NO-PR:           patch has no upstream_pr (Genesis-original — no
                     action needed)

Why this exists
---------------
Iron rule #11 (Sander 2026-05-11) requires deep-diff verification on
every pin bump. Previously the boot-time wiring drift detector was the
only signal that a patch had been superseded — that's reactive. This
script is proactive: catches retire-eligible patches before the next
bump, and flags premature retires before they hide regressions.

Usage
-----
  python3 scripts/audit_upstream_status.py
      Full report, table + counts.

  python3 scripts/audit_upstream_status.py --json
      Machine-readable output for CI / dashboards.

  python3 scripts/audit_upstream_status.py --filter newly-merged
      Only show the actionable queue.

  python3 scripts/audit_upstream_status.py --skip-network
      Use cached results only (for offline / fast-fail CI).

Requirements
------------
- `gh` CLI authenticated (uses gh api → GitHub REST v3)
- python3 stdlib only

CI hook (suggested)
-------------------
Add to `.github/workflows/upstream_audit.yml`:

  schedule:
    - cron: '0 7 * * 1'  # Monday 07:00 UTC weekly
  jobs:
    audit:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - run: python3 scripts/audit_upstream_status.py --filter newly-merged

When the report surfaces NEWLY-MERGED entries, run the iron-rule-#11
workflow: deep-diff our patch vs upstream code → retire OR update.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent
    / "vllm" / "sndr_core" / "dispatcher" / "registry.py"
)


@dataclass
class PatchAuditRow:
    patch_id: str
    upstream_pr: int
    pr_title: str
    pr_state: str  # "open" / "closed" / "merged" / "error"
    pr_merged_at: Optional[str]
    lifecycle: Optional[str]
    has_superseded_by: bool
    has_vllm_version_range: bool
    category: str  # NEWLY-MERGED / STALE-RETIRED / WATCH / SUPERSEDED-OK / ERROR


# ─── Registry parsing ──────────────────────────────────────────────────────


def _load_registry_entries() -> dict[str, str]:
    text = REGISTRY_PATH.read_text()
    entries: dict[str, str] = {}
    for m in re.finditer(
        r'    "(\w+)":\s*\{(.*?)^    \},', text, flags=re.M | re.S
    ):
        entries[m.group(1)] = m.group(2)
    return entries


def _extract_upstream_pr(body: str) -> Optional[int]:
    m = re.search(r'"upstream_pr"\s*:\s*(\d+)', body)
    return int(m.group(1)) if m else None


def _extract_lifecycle(body: str) -> Optional[str]:
    m = re.search(r'"lifecycle"\s*:\s*"([^"]+)"', body)
    return m.group(1) if m else None


def _has_field(body: str, field: str) -> bool:
    return bool(re.search(rf'"{field}"\s*:', body))


def _enables_upstream_feature(body: str) -> bool:
    """Check `enables_upstream_feature: True` registry-driven waiver.

    Used for patches that ACTIVATE/wrap an upstream feature rather than
    BACKPORT a fix. Audit excludes them from NEWLY-MERGED categorization.
    """
    return bool(re.search(
        r'"enables_upstream_feature"\s*:\s*True', body))


# ─── GitHub API (via gh CLI) ───────────────────────────────────────────────


def _query_pr(pr_number: int) -> dict:
    """Return {state, merged_at, title, kind: 'pr'|'issue'} or {error: '...'}.

    Tries `pulls/N` first. If 404, falls back to `issues/N` — some
    Genesis patches' `upstream_pr` field references the issue (bug
    report) rather than the fix PR. Returning `kind="issue"` lets
    callers categorize differently (issues don't have merge semantics).
    """
    try:
        out = subprocess.run(
            ["gh", "api", f"repos/vllm-project/vllm/pulls/{pr_number}",
             "--jq", '{state, merged_at, title}'],
            capture_output=True, text=True, timeout=15,
        )
    except FileNotFoundError:
        return {"error": "gh CLI not found — install + authenticate"}
    except subprocess.TimeoutExpired:
        return {"error": f"gh timeout for #{pr_number}"}

    if out.returncode == 0:
        try:
            data = json.loads(out.stdout)
            data["kind"] = "pr"
            return data
        except json.JSONDecodeError as e:
            return {"error": f"json parse: {e}"}

    # 404 → try issues endpoint
    if "Not Found" in (out.stderr or ""):
        try:
            issue_out = subprocess.run(
                ["gh", "api", f"repos/vllm-project/vllm/issues/{pr_number}",
                 "--jq", '{state, title}'],
                capture_output=True, text=True, timeout=15,
            )
            if issue_out.returncode == 0:
                data = json.loads(issue_out.stdout)
                data["kind"] = "issue"
                data["merged_at"] = None  # issues don't merge
                return data
        except Exception:
            pass

    return {"error": (out.stderr or "").strip()[:120]
            or f"gh exit={out.returncode}"}


# ─── Internal-supersession waiver ──────────────────────────────────────────
# Patches whose `upstream_pr` field references a PR/issue that's NOT the
# actual source of supersession. Typical: we retired internally (our own
# patch evolution superseded an earlier one) but the upstream_pr field
# still points to the original tracking item. Add a one-line reason.
_INTERNAL_SUPERSESSION_WAIVER = {
    # P61's LAST-occurrence approach was superseded by our own P12 v2
    # FIRST-occurrence (v7.62.5, 2026-04-XX). Upstream PR #40783 remains
    # OPEN — that's normal, we don't depend on it landing.
    "P61": "internal: P12 v2 FIRST-occurrence (v7.62.5)",
}


# ─── Intentional-inverse waiver ────────────────────────────────────────────
# Patches that DELIBERATELY oppose / revert a merged upstream PR because
# upstream's design regressed performance on our hardware. These keep
# lifecycle="experimental" (or research) — they are NOT retire candidates
# despite upstream being merged. The credit/notes must explain why
# upstream's approach was rejected.
_INTENTIONAL_INVERSE_WAIVER = {
    # P98 reverts upstream #40941 (WorkspaceManager indirection) because
    # current_workspace_manager().get_simultaneous() Python lookup × N
    # layers × per-step caused 17% TPS regression (200→167) on PROD
    # Ampere TQ small-batch single-stream workloads. Documented in P98
    # credit field as "DELIBERATE INVERSE". Keep active until either:
    # (a) upstream's WorkspaceManager design improves perf on our HW, OR
    # (b) we upgrade away from Ampere TQ small-batch profile.
    "P98": "intentional revert of #40941 (WorkspaceManager 17% TPS regression on Ampere TQ)",
}


# Patches whose `upstream_pr` references a GitHub ISSUE (bug report) not
# a PR. These don't have a merge state — categorize as ISSUE-REF.
# Audit script handles via _query_pr fallback to issues endpoint.


# ─── Categorization ────────────────────────────────────────────────────────


def categorize(row_data: dict) -> str:
    """Decide which audit bucket a patch goes in."""
    pr = row_data["pr"]
    pid = row_data["pid"]
    if "error" in pr:
        return "ERROR"

    kind = pr.get("kind", "pr")
    state = pr.get("state")
    merged_at = pr.get("merged_at")
    is_merged = kind == "pr" and state == "closed" and bool(merged_at)
    lifecycle = row_data["lifecycle"]

    if kind == "issue":
        # Issues don't have merge semantics. Categorize based on issue
        # state + our lifecycle.
        if state == "closed":
            return "ISSUE-CLOSED"  # bug fixed upstream — likely actionable
        return "ISSUE-OPEN"

    if is_merged:
        if lifecycle == "retired":
            return "SUPERSEDED-OK"
        if pid in _INTENTIONAL_INVERSE_WAIVER:
            return "INTENTIONAL-INVERSE"  # waived: kept on purpose
        if row_data.get("enables_upstream_feature"):
            return "ENABLES-UPSTREAM"  # waived: convenience activator
        return "NEWLY-MERGED"  # action queue

    # PR still open
    if lifecycle == "retired":
        if pid in _INTERNAL_SUPERSESSION_WAIVER:
            return "RETIRED-INTERNAL"  # waived: internal supersession
        return "STALE-RETIRED"  # weird state — premature retire?
    return "WATCH"  # normal: upstream open, our patch active


# ─── Main audit ────────────────────────────────────────────────────────────


def run_audit(skip_network: bool = False) -> list[PatchAuditRow]:
    entries = _load_registry_entries()
    rows: list[PatchAuditRow] = []

    candidates: list[tuple[str, str, int]] = []
    for pid, body in entries.items():
        pr = _extract_upstream_pr(body)
        if pr is not None:
            candidates.append((pid, body, pr))

    print(
        f"# Auditing {len(candidates)}/{len(entries)} patches with "
        f"`upstream_pr`...",
        file=sys.stderr,
    )

    for i, (pid, body, pr) in enumerate(candidates):
        lifecycle = _extract_lifecycle(body)
        has_sb = _has_field(body, "superseded_by")
        has_vvr = _has_field(body, "vllm_version_range")

        if skip_network:
            pr_info = {"state": "unknown", "merged_at": None,
                       "title": "(network skipped)"}
        else:
            pr_info = _query_pr(pr)
            if i and i % 10 == 0:
                print(f"# ...{i}/{len(candidates)}",
                      file=sys.stderr)
                time.sleep(0.2)  # gentle rate-limit

        category = categorize({
            "pr": pr_info, "lifecycle": lifecycle, "pid": pid,
            "enables_upstream_feature": _enables_upstream_feature(body),
        })

        rows.append(PatchAuditRow(
            patch_id=pid,
            upstream_pr=pr,
            pr_title=pr_info.get("title", "")[:80]
                if "error" not in pr_info else f"ERROR: {pr_info['error']}",
            pr_state=pr_info.get("state", "error"),
            pr_merged_at=pr_info.get("merged_at"),
            lifecycle=lifecycle,
            has_superseded_by=has_sb,
            has_vllm_version_range=has_vvr,
            category=category,
        ))

    return rows


# ─── Output formatters ─────────────────────────────────────────────────────


_CATEGORY_PRIORITY = {
    "NEWLY-MERGED": 0,        # action required
    "STALE-RETIRED": 1,       # investigate — retired locally but upstream open
    "ISSUE-CLOSED": 2,        # upstream issue resolved — check our patch state
    "ERROR": 3,
    "ISSUE-OPEN": 4,          # issue tracked, watching
    "WATCH": 5,
    "INTENTIONAL-INVERSE": 6, # waived — kept on purpose vs merged upstream
    "ENABLES-UPSTREAM": 7,    # waived — convenience activator of upstream feature
    "RETIRED-INTERNAL": 8,    # waived — internal supersession
    "SUPERSEDED-OK": 9,
}


def _print_table(rows: list[PatchAuditRow]) -> None:
    rows_sorted = sorted(
        rows,
        key=lambda r: (_CATEGORY_PRIORITY.get(r.category, 9), r.patch_id),
    )

    counts: dict[str, int] = {}
    for r in rows_sorted:
        counts[r.category] = counts.get(r.category, 0) + 1

    print()
    print("=" * 100)
    print(f"  Upstream PR audit ({len(rows_sorted)} patches with upstream_pr)")
    print("=" * 100)

    for category in [
        "NEWLY-MERGED", "STALE-RETIRED", "ISSUE-CLOSED",
        "ERROR", "ISSUE-OPEN", "WATCH",
        "INTENTIONAL-INVERSE", "ENABLES-UPSTREAM",
        "RETIRED-INTERNAL", "SUPERSEDED-OK",
    ]:
        rows_in_cat = [r for r in rows_sorted if r.category == category]
        if not rows_in_cat:
            continue
        print()
        print(f"── {category} ({len(rows_in_cat)}) " + "─" * 70)

        if category == "NEWLY-MERGED":
            print("  Action: deep-diff our patch vs upstream → retire OR update")
        elif category == "STALE-RETIRED":
            print("  Action: investigate — our patch retired but upstream OPEN")
        elif category == "ISSUE-CLOSED":
            print("  Action: upstream bug fixed — check whether our patch is now redundant")
        elif category == "ERROR":
            print("  Action: check gh authentication / network / PR access")

        for r in rows_in_cat:
            merged = (r.pr_merged_at or "")[:10] if r.pr_merged_at else "(not merged)"
            lc = r.lifecycle or "?"
            prov = "✓prov" if (r.has_superseded_by and r.has_vllm_version_range) else "  -  "
            print(
                f"  {r.patch_id:6}  PR #{r.upstream_pr:6}  "
                f"{r.pr_state:6}  {merged:10}  lc={lc:13}  {prov}  "
                f"{r.pr_title}"
            )

    print()
    print("=" * 100)
    print(
        "  Summary: "
        + "  ".join(f"{cat}={counts.get(cat, 0)}"
                    for cat in ["NEWLY-MERGED", "STALE-RETIRED",
                                "ISSUE-CLOSED", "ERROR",
                                "ISSUE-OPEN", "WATCH",
                                "INTENTIONAL-INVERSE",
                                "ENABLES-UPSTREAM",
                                "RETIRED-INTERNAL", "SUPERSEDED-OK"])
    )
    print("=" * 100)

    if counts.get("NEWLY-MERGED", 0) > 0:
        print()
        print(
            "ACTION REQUIRED: {} newly-merged patch(es). Run iron-rule-#11 "
            "deep-diff for each — see CONTRIBUTING.md Pin-bump playbook."
            .format(counts["NEWLY-MERGED"])
        )


def _print_json(rows: list[PatchAuditRow]) -> None:
    print(json.dumps([asdict(r) for r in rows], indent=2))


# ─── CLI ───────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--json", action="store_true",
                   help="JSON output for machine consumption")
    p.add_argument(
        "--filter", choices=[
            "newly-merged", "stale-retired", "issue-closed",
            "watch", "issue-open", "intentional-inverse",
            "enables-upstream", "retired-internal",
            "superseded-ok", "error",
        ],
        help="Only show one category (e.g. for CI failure gating)",
    )
    p.add_argument("--skip-network", action="store_true",
                   help="Skip gh API calls (offline / fast-fail CI)")
    p.add_argument("--fail-on-newly-merged", action="store_true",
                   help="Exit 1 if any NEWLY-MERGED patches found (CI gate)")
    args = p.parse_args()

    rows = run_audit(skip_network=args.skip_network)

    if args.filter:
        bucket = args.filter.upper().replace("-", "-")  # cosmetic
        target = bucket  # already uppercase from .upper() above
        rows = [r for r in rows if r.category == target]

    if args.json:
        _print_json(rows)
    else:
        _print_table(rows)

    if args.fail_on_newly_merged:
        if any(r.category == "NEWLY-MERGED" for r in rows):
            print(
                "\n✗ Exit 1: NEWLY-MERGED patches found "
                "(--fail-on-newly-merged active)",
                file=sys.stderr,
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
