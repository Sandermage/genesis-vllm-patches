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
    / "sndr" / "dispatcher" / "registry.py"
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
    # Phase 5.1.A (2026-05-22) — registry-driven relationship hint
    # (one of VALID_UPSTREAM_PR_RELATIONSHIPS, or "backport" if absent).
    upstream_pr_relationship: str = "backport"


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
    # v11.3.0 BUG #13 fix: also accept URL-string form
    # `"upstream_pr": "https://github.com/vllm-project/vllm/pull/<N>"`
    # — 4 G4_* entries use this form and were previously invisible to
    # the upstream audit gate. Issue URLs (no PR number) deliberately
    # stay None — see normalize_upstream_pr in dispatcher/spec.py.
    m = re.search(r'"upstream_pr"\s*:\s*(\d+)', body)
    if m:
        return int(m.group(1))
    m = re.search(
        r'"upstream_pr"\s*:\s*"[^"]*github\.com/vllm-project/vllm/'
        r'pull/(\d+)[^"]*"',
        body,
    )
    if m:
        return int(m.group(1))
    return None


def _extract_lifecycle(body: str) -> Optional[str]:
    m = re.search(r'"lifecycle"\s*:\s*"([^"]+)"', body)
    return m.group(1) if m else None


def _has_field(body: str, field: str) -> bool:
    return bool(re.search(rf'"{field}"\s*:', body))


def _extract_upstream_pr_relationship(body: str) -> Optional[str]:
    """Read explicit `upstream_pr_relationship` from a registry entry body.

    Phase 5.1.A (2026-05-22): operator-set relationship hint that routes
    the patch to one of the audit buckets without needing per-patch
    hardcoded waivers in this script. Returns None when the field is
    absent — the registry validator now flags that as an ERROR for any
    entry with an integer `upstream_pr` (5.1.C escalation), so a None
    result here means either the entry has no upstream_pr at all, or
    the registry is broken (which the validator catches separately).
    """
    m = re.search(r'"upstream_pr_relationship"\s*:\s*"([^"]+)"', body)
    return m.group(1) if m else None


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


# Phase 5.1.C cleanup (2026-05-22):
#   The per-patch hardcoded waiver dicts that lived here
#   (_INTENTIONAL_INVERSE_WAIVER for P98; _INTERNAL_SUPERSESSION_WAIVER
#   for P61) were removed once every patch with an integer upstream_pr
#   gained an explicit `upstream_pr_relationship` field (Phase 5.1.B).
#   Adding a new waiver no longer requires editing this script —
#   set `upstream_pr_relationship` on the registry entry instead.
#
# Patches whose `upstream_pr` references a GitHub ISSUE (bug report) not
# a PR. These don't have a merge state — categorize as ISSUE-REF.
# Audit script handles via _query_pr fallback to issues endpoint.


# ─── Retire-eligibility taxonomy (Phase 5.1.D, 2026-05-23) ────────────────


_PURE_UPSTREAM_RELATIONSHIPS = frozenset({"backport"})
"""Relationship values that permit *status-based* retire scoring.

A patch with `upstream_pr_relationship in _PURE_UPSTREAM_RELATIONSHIPS`
AND upstream PR merged AND lifecycle active is a `RETIRE-CANDIDATE`.
Iron-rule-#11 deep-diff is still REQUIRED before the actual retire —
the relationship being pure only means status alone is sufficient to put
the patch in the deep-diff queue.

Every OTHER relationship value signals that upstream coverage is NOT a
straight superset of Genesis runtime logic — the patch may guard a
regression, invert upstream behavior, live at a different layer, or
provide an env-gated convenience over upstream. For those, status alone
is NOT a retire signal. They classify as `NEEDS-DEEP-PARITY` and the
default verdict is KEEP unless a separate parity proof exists.

This constant codifies the discipline operator established after the
2026-05-23 sidecleanup audit, where 5 patches with merged upstream PRs
were initially proposed for retire by an agent reading PR dates only —
all 5 turned out to be non-pure relationships (intentional_inverse,
enables_upstream, defensive_overlay, related_not_superseding) that
would have caused regressions if retired blindly.

================================================================
EXTENSION PROTOCOL — adding a new `upstream_pr_relationship` value
================================================================

When a future patch class needs a relationship value not in the current
six, follow this protocol so the audit + retire logic stays coherent:

1. Pick a short, unambiguous identifier in `snake_case` (e.g.
   `redundant`, `pre_emptive`). The value must describe the SEMANTIC
   relationship between Genesis patch and the cited upstream PR, not
   the operational outcome.

2. Decide *status-based retire eligibility*:
   - If a merged upstream PR alone is sufficient signal that Genesis
     patch is safe to retire (subject to iron-rule-#11 deep-diff) →
     ADD the new value to `_PURE_UPSTREAM_RELATIONSHIPS` here.
   - If upstream PR merge does NOT imply Genesis safety → LEAVE
     `_PURE_UPSTREAM_RELATIONSHIPS` alone. The new value will
     automatically route to `NEEDS-DEEP-PARITY` via the default branch.

3. Register the value in `vllm/sndr_core/dispatcher/spec.py` →
   `VALID_UPSTREAM_PR_RELATIONSHIPS` tuple with a one-line trailing
   comment describing when to use it.

4. Update `retire_eligibility()` below if the new value needs custom
   bucket routing distinct from the default `NEEDS-DEEP-PARITY` /
   `RETIRE-CANDIDATE` / `ACTIVE` / `ALREADY-RETIRED` / `UNKNOWN`
   verdicts. Most additions will NOT need this — the default split
   on `in _PURE_UPSTREAM_RELATIONSHIPS` is sufficient.

5. Update `categorize()` bucket label if the new value warrants its own
   audit-output bucket (e.g. so reviewers see at-a-glance which patches
   carry it). Otherwise it folds into `WATCH` / `NEEDS-DEEP-PARITY`.

6. Add a test case in `tests/unit/dispatcher/test_retire_eligibility.py`
   (or equivalent) that pins the new value's bucket routing. The test
   suite locks both `_PURE_UPSTREAM_RELATIONSHIPS` membership and the
   resulting `retire_eligibility()` return.

7. Document the rationale in a `sndr_private/planning/audits/PHASE_*_
   RELATIONSHIP_*_RU.md` design note: when this value should be used,
   what registry-credit pattern accompanies it, what deep-parity proof
   discharges its NEEDS-DEEP-PARITY verdict (if applicable).

8. Use the new value in at least one registry entry, with `credit`
   field explicitly explaining why this value (not a simpler
   alternative) applies to that patch. This anchors the policy to a
   concrete example.

The audit script itself should NOT need a code change for steps 3-5
in the common case — the `_PURE_UPSTREAM_RELATIONSHIPS` membership
check is the single decision point. Steps 4-5 only apply when the new
value carries semantic state that the existing 5-bucket taxonomy
cannot express.
"""


def retire_eligibility(row_data: dict) -> str:
    """Return the canonical retire verdict for a (PR-state, lifecycle,
    relationship) triple.

    Returns ONE of:

      "RETIRE-CANDIDATE"   — pure-`backport` upstream PR merged AND our
                             patch lifecycle is active. Iron-rule-#11
                             deep-diff REQUIRED before any actual retire.
                             This is the only verdict that authorizes
                             putting a patch in the retire queue.
      "NEEDS-DEEP-PARITY"  — upstream PR merged AND our patch's relationship
                             is one of the non-pure classes
                             (counter_regression, intentional_inverse,
                             enables_upstream, defensive_overlay,
                             related_not_superseding). PR-status alone is
                             INSUFFICIENT to retire. Default verdict KEEP
                             unless a separate deep parity audit proves
                             equivalence.
      "ACTIVE"             — upstream PR still open AND our patch
                             lifecycle is active. No retire action.
      "ALREADY-RETIRED"    — our patch lifecycle is "retired". Action
                             already complete.
      "UNKNOWN"            — issue reference, PR query error, or other
                             state that cannot be auto-classified.

    Implementation: derives the verdict from `categorize()`'s bucket so
    the two stay in lockstep. Adding a new audit bucket requires updating
    this mapping; the test suite locks both surfaces.

    Lifecycle override: when our patch is already `retired`, the verdict
    is ALWAYS ALREADY-RETIRED regardless of bucket. The bucket is then a
    purely informational record (e.g. `RELATED-NOT-SUPERSEDING` on an
    open PR documents why the retire happened without that PR). No deep-
    parity work is meaningful for a retired patch — it isn't applying.

    Operator contract: PIN.R recon scripts / agents MUST consult this
    function (or equivalent registry-aware filter) before producing a
    "retire-NOW" list. Reading PR merge dates without consulting
    `upstream_pr_relationship` produces false positives — see
    docs/_internal/PIN_R_SIDECLEANUP_DEEP_PARITY_AUDIT for the empirical
    proof.
    """
    if row_data.get("lifecycle") == "retired":
        return "ALREADY-RETIRED"
    bucket = categorize(row_data)
    if bucket == "NEWLY-MERGED":
        return "RETIRE-CANDIDATE"
    if bucket in {
        "COUNTER-REGRESSION",
        "INTENTIONAL-INVERSE",
        "ENABLES-UPSTREAM",
        "DEFENSIVE-OVERLAY",
        "RELATED-NOT-SUPERSEDING",
    }:
        return "NEEDS-DEEP-PARITY"
    if bucket in {"SUPERSEDED-OK", "STALE-RETIRED"}:
        return "ALREADY-RETIRED"
    if bucket == "WATCH":
        return "ACTIVE"
    return "UNKNOWN"  # ERROR / ISSUE-OPEN / ISSUE-CLOSED


# ─── Categorization ────────────────────────────────────────────────────────


def categorize(row_data: dict) -> str:
    """Decide which audit bucket a patch goes in.

    Phase 5.1.C (2026-05-22) — routing is now driven entirely by the
    registry's `upstream_pr_relationship` field:

      1. PR error → ERROR
      2. Reference is an issue → ISSUE-OPEN / ISSUE-CLOSED
      3. PR merged + lifecycle=retired → SUPERSEDED-OK
      4. PR merged + explicit `upstream_pr_relationship` →
         COUNTER-REGRESSION / INTENTIONAL-INVERSE / ENABLES-UPSTREAM /
         DEFENSIVE-OVERLAY / RELATED-NOT-SUPERSEDING
      5. PR merged + relationship is `backport` (or unset for an
         entry without integer upstream_pr — shouldn't reach here in
         practice) → NEWLY-MERGED
      6. PR still open + lifecycle=retired + relationship is
         `related_not_superseding` → RELATED-NOT-SUPERSEDING
      7. PR still open + lifecycle=retired → STALE-RETIRED
      8. PR still open + lifecycle active → WATCH

    Hardcoded waiver dicts and the legacy `enables_upstream_feature`
    boolean were retired in 5.1.C — to add a new waiver, set
    `upstream_pr_relationship` on the registry entry.
    """
    pr = row_data["pr"]
    if "error" in pr:
        return "ERROR"

    kind = pr.get("kind", "pr")
    state = pr.get("state")
    merged_at = pr.get("merged_at")
    is_merged = kind == "pr" and state == "closed" and bool(merged_at)
    lifecycle = row_data["lifecycle"]
    relationship = row_data.get("upstream_pr_relationship")

    if kind == "issue":
        # Issues don't have merge semantics. Categorize based on issue
        # state + our lifecycle.
        if state == "closed":
            return "ISSUE-CLOSED"  # bug fixed upstream — likely actionable
        return "ISSUE-OPEN"

    if is_merged:
        if lifecycle == "retired":
            return "SUPERSEDED-OK"
        if relationship == "counter_regression":
            return "COUNTER-REGRESSION"
        if relationship == "intentional_inverse":
            return "INTENTIONAL-INVERSE"
        if relationship == "enables_upstream":
            return "ENABLES-UPSTREAM"
        if relationship == "defensive_overlay":
            return "DEFENSIVE-OVERLAY"
        if relationship == "related_not_superseding":
            return "RELATED-NOT-SUPERSEDING"
        return "NEWLY-MERGED"  # action queue

    # PR still open
    if lifecycle == "retired":
        if relationship == "related_not_superseding":
            return "RELATED-NOT-SUPERSEDING"
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

        relationship_explicit = _extract_upstream_pr_relationship(body)
        relationship_for_output = relationship_explicit or "backport"

        category = categorize({
            "pr": pr_info, "lifecycle": lifecycle, "pid": pid,
            "upstream_pr_relationship": relationship_explicit,
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
            upstream_pr_relationship=relationship_for_output,
        ))

    return rows


# ─── Output formatters ─────────────────────────────────────────────────────


_CATEGORY_PRIORITY = {
    "NEWLY-MERGED": 0,          # action required
    "STALE-RETIRED": 1,         # investigate — retired locally but upstream open
    "ISSUE-CLOSED": 2,          # upstream issue resolved — check our patch state
    "ERROR": 3,
    "ISSUE-OPEN": 4,            # issue tracked, watching
    "WATCH": 5,
    # Phase 5.1.A (2026-05-22) + 5.1.C cleanup: explicit-relationship
    # waiver buckets. The RETIRED-INTERNAL bucket from 5.1.A was retired
    # in 5.1.C — P61 (its only consumer) routes via the explicit
    # `upstream_pr_relationship: "related_not_superseding"` field now.
    "COUNTER-REGRESSION": 6,    # waived — Genesis corrects a regression in the cited PR
    "INTENTIONAL-INVERSE": 7,   # waived — kept on purpose vs merged upstream
    "DEFENSIVE-OVERLAY": 8,     # waived — lower-layer defensive guard alongside upstream
    "RELATED-NOT-SUPERSEDING": 9,  # waived — different layer; coverage doesn't overlap
    "ENABLES-UPSTREAM": 10,     # waived — convenience activator of upstream feature
    "SUPERSEDED-OK": 11,
}

# Categories shown in the table output (mirrors _CATEGORY_PRIORITY order).
# Kept as a separate constant so the table iterates deterministically.
_CATEGORY_DISPLAY_ORDER = [
    "NEWLY-MERGED", "STALE-RETIRED", "ISSUE-CLOSED",
    "ERROR", "ISSUE-OPEN", "WATCH",
    "COUNTER-REGRESSION", "INTENTIONAL-INVERSE",
    "DEFENSIVE-OVERLAY", "RELATED-NOT-SUPERSEDING",
    "ENABLES-UPSTREAM", "SUPERSEDED-OK",
]


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

    for category in _CATEGORY_DISPLAY_ORDER:
        rows_in_cat = [r for r in rows_sorted if r.category == category]
        if not rows_in_cat:
            continue
        print()
        print(f"── {category} ({len(rows_in_cat)}) " + "─" * 70)

        if category == "NEWLY-MERGED":
            print("  Verdict: RETIRE-CANDIDATE — iron-rule-#11 deep-diff REQUIRED before retire")
        elif category == "STALE-RETIRED":
            print("  Action: investigate — our patch retired but upstream OPEN")
        elif category == "ISSUE-CLOSED":
            print("  Action: upstream bug fixed — check whether our patch is now redundant")
        elif category == "ERROR":
            print("  Action: check gh authentication / network / PR access")
        elif category == "COUNTER-REGRESSION":
            print("  Verdict: NEEDS-DEEP-PARITY — Genesis corrects a regression in the cited PR; default KEEP")
        elif category == "INTENTIONAL-INVERSE":
            print("  Verdict: NEEDS-DEEP-PARITY — Genesis deliberately reverses the cited PR; default KEEP")
        elif category == "ENABLES-UPSTREAM":
            print("  Verdict: NEEDS-DEEP-PARITY — Genesis is an env-gated convenience over upstream; default KEEP")
        elif category == "DEFENSIVE-OVERLAY":
            print("  Verdict: NEEDS-DEEP-PARITY — defensive guard at a different layer; default KEEP")
        elif category == "RELATED-NOT-SUPERSEDING":
            print("  Verdict: NEEDS-DEEP-PARITY — lives at a different layer; coverage doesn't overlap; default KEEP")

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
                    for cat in _CATEGORY_DISPLAY_ORDER)
    )
    print("=" * 100)

    if counts.get("NEWLY-MERGED", 0) > 0:
        print()
        print(
            "ACTION REQUIRED: {} RETIRE-CANDIDATE patch(es) (pure backports "
            "with merged upstream). Run iron-rule-#11 deep-diff for EACH "
            "before any retire — see CONTRIBUTING.md Pin-bump playbook. "
            "Patches in other waiver buckets are NEEDS-DEEP-PARITY (default "
            "KEEP); do NOT add them to a retire list based on PR status alone."
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
            "watch", "issue-open",
            "counter-regression", "intentional-inverse",
            "defensive-overlay", "related-not-superseding",
            "enables-upstream", "superseded-ok", "error",
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
