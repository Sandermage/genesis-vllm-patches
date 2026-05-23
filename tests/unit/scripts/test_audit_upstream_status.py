# SPDX-License-Identifier: Apache-2.0
"""Phase 5.1.A + 5.1.C (2026-05-22) — `audit_upstream_status.py`
classify() routing.

The audit script's `categorize()` translates a (PR-state, lifecycle,
registry-driven relationship) triple into one of the audit buckets.

Phase 5.1.A added three buckets:

  - COUNTER-REGRESSION       — `upstream_pr_relationship: counter_regression`
  - DEFENSIVE-OVERLAY        — `upstream_pr_relationship: defensive_overlay`
  - RELATED-NOT-SUPERSEDING  — `upstream_pr_relationship: related_not_superseding`

Existing buckets that the new field drives:

  - INTENTIONAL-INVERSE      — `upstream_pr_relationship: intentional_inverse`
  - ENABLES-UPSTREAM         — `upstream_pr_relationship: enables_upstream`

Phase 5.1.C removed the legacy fallback chain:

  - The `_INTENTIONAL_INVERSE_WAIVER` dict (P98) is gone.
  - The `_INTERNAL_SUPERSESSION_WAIVER` dict (P61) is gone, along with
    the RETIRED-INTERNAL bucket that only P61 routed through.
  - The `enables_upstream_feature: True` boolean fallback is gone;
    `categorize()` no longer reads that field.

Classification is now driven entirely by `upstream_pr_relationship`
on the registry entry. These tests verify the explicit-field routing
and the absence of legacy fallback paths.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import audit_upstream_status as M  # noqa: E402


def _merged_pr(merged_at="2026-05-01T00:00:00Z"):
    return {
        "kind": "pr",
        "state": "closed",
        "merged_at": merged_at,
        "title": "Some upstream fix",
    }


def _open_pr():
    return {
        "kind": "pr",
        "state": "open",
        "merged_at": None,
        "title": "Some upstream fix (open)",
    }


def _open_issue():
    return {
        "kind": "issue",
        "state": "open",
        "merged_at": None,
        "title": "Bug report",
    }


def _closed_issue():
    return {
        "kind": "issue",
        "state": "closed",
        "merged_at": None,
        "title": "Bug report (closed)",
    }


# ─── Explicit-relationship routing ────────────────────────────────────────


class TestExplicitRelationshipRouting:

    def test_counter_regression_routes_to_new_bucket(self):
        cat = M.categorize({
            "pr": _merged_pr(),
            "pid": "P_FAKE",
            "lifecycle": "stable",
            "upstream_pr_relationship": "counter_regression",
        })
        assert cat == "COUNTER-REGRESSION"

    def test_defensive_overlay_routes_to_new_bucket(self):
        cat = M.categorize({
            "pr": _merged_pr(),
            "pid": "P_FAKE",
            "lifecycle": "stable",
            "upstream_pr_relationship": "defensive_overlay",
        })
        assert cat == "DEFENSIVE-OVERLAY"

    def test_related_not_superseding_routes_to_new_bucket(self):
        cat = M.categorize({
            "pr": _merged_pr(),
            "pid": "P_FAKE",
            "lifecycle": "stable",
            "upstream_pr_relationship": "related_not_superseding",
        })
        assert cat == "RELATED-NOT-SUPERSEDING"

    def test_intentional_inverse_routes_via_explicit_field(self):
        """Post-5.1.C: routing is via the registry field, not the
        hardcoded P98 waiver dict (now deleted). Verifying for a
        non-P98 PID confirms there's no patch-id dependency left."""
        cat = M.categorize({
            "pr": _merged_pr(),
            "pid": "P_FAKE",
            "lifecycle": "experimental",
            "upstream_pr_relationship": "intentional_inverse",
        })
        assert cat == "INTENTIONAL-INVERSE"

    def test_enables_upstream_routes_via_explicit_field(self):
        """Post-5.1.C: `enables_upstream_feature: True` boolean is no
        longer read. The explicit field is the only path."""
        cat = M.categorize({
            "pr": _merged_pr(),
            "pid": "P_FAKE",
            "lifecycle": "stable",
            "upstream_pr_relationship": "enables_upstream",
        })
        assert cat == "ENABLES-UPSTREAM"

    def test_backport_routes_to_newly_merged_when_active(self):
        """`backport` is the default; merged upstream + still-active
        local patch is the action queue."""
        cat = M.categorize({
            "pr": _merged_pr(),
            "pid": "P_FAKE",
            "lifecycle": "stable",
            "upstream_pr_relationship": "backport",
        })
        assert cat == "NEWLY-MERGED"

    def test_missing_relationship_treated_as_backport(self):
        """A None / absent relationship value (e.g. when the audit
        script can't extract the field from a malformed entry) still
        falls through merged-bucket enum checks to NEWLY-MERGED."""
        cat = M.categorize({
            "pr": _merged_pr(),
            "pid": "P_FAKE",
            "lifecycle": "stable",
            "upstream_pr_relationship": None,
        })
        assert cat == "NEWLY-MERGED"


# ─── Legacy fallbacks removed — regression tests ──────────────────────────


class TestLegacyFallbacksRemoved:
    """Phase 5.1.C: the legacy boolean and waiver dicts were removed.
    These tests confirm the routing relies only on the explicit field."""

    def test_legacy_boolean_no_longer_routes_to_enables_upstream(self):
        """An `enables_upstream_feature` key in row_data must be a
        no-op now (was the legacy boolean fallback in 5.1.A). Without
        the explicit field the patch routes to NEWLY-MERGED."""
        cat = M.categorize({
            "pr": _merged_pr(),
            "pid": "P_FAKE",
            "lifecycle": "stable",
            "upstream_pr_relationship": None,
            "enables_upstream_feature": True,  # ignored now
        })
        assert cat == "NEWLY-MERGED"

    def test_p98_without_explicit_field_no_longer_inverse(self):
        """The hardcoded `_INTENTIONAL_INVERSE_WAIVER = {"P98": ...}`
        dict was deleted in 5.1.C. Without the explicit field, P98
        routes to NEWLY-MERGED just like any other patch."""
        cat = M.categorize({
            "pr": _merged_pr(),
            "pid": "P98",
            "lifecycle": "experimental",
            "upstream_pr_relationship": None,
        })
        assert cat == "NEWLY-MERGED"

    def test_p61_without_explicit_field_routes_to_stale_retired(self):
        """The `_INTERNAL_SUPERSESSION_WAIVER = {"P61": ...}` dict and
        the RETIRED-INTERNAL bucket are both gone. Without the explicit
        field, P61 + retired lifecycle + open upstream routes to
        STALE-RETIRED (the standard "investigate" bucket)."""
        cat = M.categorize({
            "pr": _open_pr(),
            "pid": "P61",
            "lifecycle": "retired",
            "upstream_pr_relationship": None,
        })
        assert cat == "STALE-RETIRED"

    def test_retired_internal_bucket_is_gone(self):
        assert "RETIRED-INTERNAL" not in M._CATEGORY_PRIORITY
        assert "RETIRED-INTERNAL" not in M._CATEGORY_DISPLAY_ORDER

    def test_waiver_dicts_are_gone(self):
        assert not hasattr(M, "_INTERNAL_SUPERSESSION_WAIVER")
        assert not hasattr(M, "_INTENTIONAL_INVERSE_WAIVER")

    def test_enables_upstream_feature_helper_is_gone(self):
        assert not hasattr(M, "_enables_upstream_feature")


# ─── Live-registry: 5.1.B special patches resolve correctly ───────────────


class TestLiveRegistrySpecialPatchExtraction:
    """End-to-end: extract the relationship from the live registry text
    for each of the 8 special-classification patches and confirm the
    field is present + correctly valued. Guards both against the
    extractor regex breaking and against an operator silently dropping
    the field on one of these patches."""

    EXPECTED_SPECIAL = {
        "PN116": "counter_regression",
        "P98":   "intentional_inverse",
        "P75":   "enables_upstream",
        "P99":   "enables_upstream",
        "PN51":  "defensive_overlay",
        "PN90":  "related_not_superseding",
        "PN24":  "related_not_superseding",
        "P61":   "related_not_superseding",
    }

    def test_each_special_patch_extracts_expected_relationship(self):
        entries = M._load_registry_entries()
        for pid, expected in self.EXPECTED_SPECIAL.items():
            assert pid in entries, (
                f"{pid}: not in live PATCH_REGISTRY anymore"
            )
            got = M._extract_upstream_pr_relationship(entries[pid])
            assert got == expected, (
                f"{pid}: registry value drifted; expected {expected!r}, "
                f"got {got!r}"
            )


# ─── Lifecycle precedence (retire still wins regardless of hint) ──────────


class TestLifecyclePrecedence:

    def test_retired_lifecycle_wins_over_relationship_when_merged(self):
        """If lifecycle is `retired`, the patch is already retired —
        SUPERSEDED-OK regardless of any relationship hint."""
        cat = M.categorize({
            "pr": _merged_pr(),
            "pid": "P_FAKE",
            "lifecycle": "retired",
            "upstream_pr_relationship": "counter_regression",
        })
        assert cat == "SUPERSEDED-OK"

    def test_retired_lifecycle_with_open_pr_and_related_hint(self):
        """`related_not_superseding` on an OPEN PR + retired lifecycle
        still routes to RELATED-NOT-SUPERSEDING (kept as informational
        record that this retire wasn't caused by the cited PR)."""
        cat = M.categorize({
            "pr": _open_pr(),
            "pid": "P_FAKE",
            "lifecycle": "retired",
            "upstream_pr_relationship": "related_not_superseding",
        })
        assert cat == "RELATED-NOT-SUPERSEDING"


# ─── Existing-bucket regression (must not break) ──────────────────────────


class TestExistingBucketsUnchanged:

    def test_open_pr_active_local_is_watch(self):
        cat = M.categorize({
            "pr": _open_pr(),
            "pid": "P_FAKE",
            "lifecycle": "stable",
            "upstream_pr_relationship": None,
        })
        assert cat == "WATCH"

    def test_open_pr_retired_local_is_stale_retired(self):
        cat = M.categorize({
            "pr": _open_pr(),
            "pid": "P_FAKE",
            "lifecycle": "retired",
            "upstream_pr_relationship": None,
        })
        assert cat == "STALE-RETIRED"

    def test_open_issue_routes_to_issue_open(self):
        cat = M.categorize({
            "pr": _open_issue(),
            "pid": "P_FAKE",
            "lifecycle": "stable",
            "upstream_pr_relationship": None,
        })
        assert cat == "ISSUE-OPEN"

    def test_closed_issue_routes_to_issue_closed(self):
        cat = M.categorize({
            "pr": _closed_issue(),
            "pid": "P_FAKE",
            "lifecycle": "stable",
            "upstream_pr_relationship": None,
        })
        assert cat == "ISSUE-CLOSED"

    def test_pr_error_routes_to_error(self):
        cat = M.categorize({
            "pr": {"error": "boom"},
            "pid": "P_FAKE",
            "lifecycle": "stable",
            "upstream_pr_relationship": None,
        })
        assert cat == "ERROR"


# ─── Extractor + display-order metadata ───────────────────────────────────


class TestRegistryExtractor:

    def test_extracts_explicit_value(self):
        body = '"upstream_pr_relationship": "counter_regression",'
        assert M._extract_upstream_pr_relationship(body) == "counter_regression"

    def test_returns_none_when_absent(self):
        body = '"upstream_pr": 12345,'
        assert M._extract_upstream_pr_relationship(body) is None

    def test_extractor_ignores_quoted_substrings(self):
        body = (
            '"credit": "counter_regression note in prose",\n'
            '"upstream_pr_relationship": "defensive_overlay",\n'
        )
        assert M._extract_upstream_pr_relationship(body) == "defensive_overlay"


class TestDisplayOrderMetadata:

    def test_new_buckets_have_priority_entries(self):
        for bucket in (
            "COUNTER-REGRESSION",
            "DEFENSIVE-OVERLAY",
            "RELATED-NOT-SUPERSEDING",
        ):
            assert bucket in M._CATEGORY_PRIORITY, (
                f"new bucket {bucket} missing from _CATEGORY_PRIORITY"
            )
            assert bucket in M._CATEGORY_DISPLAY_ORDER, (
                f"new bucket {bucket} missing from _CATEGORY_DISPLAY_ORDER"
            )

    def test_action_queue_is_first(self):
        """NEWLY-MERGED has priority 0 — table output must list it
        first so operators see the action queue at the top."""
        assert M._CATEGORY_PRIORITY["NEWLY-MERGED"] == 0
        assert M._CATEGORY_DISPLAY_ORDER[0] == "NEWLY-MERGED"

    def test_priority_and_display_order_agree_on_membership(self):
        assert set(M._CATEGORY_PRIORITY) == set(M._CATEGORY_DISPLAY_ORDER)


# ─── Phase 5.1.D (2026-05-23): retire_eligibility() public API ────────────


class TestPureUpstreamRelationshipsConstant:
    """The constant codifies which relationship values authorize
    status-based retire scoring. Any change must be deliberate; tests
    lock the set."""

    def test_constant_exists_and_is_frozen(self):
        assert hasattr(M, "_PURE_UPSTREAM_RELATIONSHIPS")
        assert isinstance(M._PURE_UPSTREAM_RELATIONSHIPS, frozenset)

    def test_only_backport_is_pure(self):
        """If a new pure relationship is added (e.g. `redundant`), this
        test must be updated explicitly — it's the operator-facing
        contract, not an implementation detail."""
        assert M._PURE_UPSTREAM_RELATIONSHIPS == frozenset({"backport"})

    def test_non_pure_relationships_excluded(self):
        for impure in (
            "counter_regression",
            "intentional_inverse",
            "enables_upstream",
            "defensive_overlay",
            "related_not_superseding",
        ):
            assert impure not in M._PURE_UPSTREAM_RELATIONSHIPS


class TestRetireEligibility:
    """The `retire_eligibility()` function is the canonical API PIN.R
    recon / sidecleanup tooling MUST consult before producing any
    retire-candidate list. Status-based retire on non-pure relationships
    is the empirically-confirmed false-positive class (2026-05-23
    sidecleanup audit caught all 5 cases below before any commit).
    """

    def test_pure_backport_merged_active_is_retire_candidate(self):
        verdict = M.retire_eligibility({
            "pr": _merged_pr(),
            "pid": "P_FAKE_BACKPORT",
            "lifecycle": "stable",
            "upstream_pr_relationship": "backport",
        })
        assert verdict == "RETIRE-CANDIDATE"

    def test_missing_relationship_treated_as_pure_backport(self):
        verdict = M.retire_eligibility({
            "pr": _merged_pr(),
            "pid": "P_FAKE",
            "lifecycle": "experimental",
            "upstream_pr_relationship": None,
        })
        assert verdict == "RETIRE-CANDIDATE"

    # ─ The 5 false-positive lock-in cases (PIN.R sidecleanup 2026-05-23) ─

    def test_enables_upstream_is_needs_deep_parity_not_retire(self):
        """P75, P99 case. Genesis is an env-gated convenience over
        upstream; status-based retire would silently lose Genesis
        defaults / fallbacks (e.g. ImportError graceful fallback)."""
        verdict = M.retire_eligibility({
            "pr": _merged_pr(),
            "pid": "P_LIKE_P75",
            "lifecycle": "experimental",
            "upstream_pr_relationship": "enables_upstream",
        })
        assert verdict == "NEEDS-DEEP-PARITY"

    def test_intentional_inverse_is_needs_deep_parity_not_retire(self):
        """P98 case. Genesis deliberately REVERSES the cited PR's
        behavior for our hardware shape. Status-based retire would
        re-introduce the regression the patch was created to invert."""
        verdict = M.retire_eligibility({
            "pr": _merged_pr(),
            "pid": "P_LIKE_P98",
            "lifecycle": "experimental",
            "upstream_pr_relationship": "intentional_inverse",
        })
        assert verdict == "NEEDS-DEEP-PARITY"

    def test_defensive_overlay_is_needs_deep_parity_not_retire(self):
        """PN51 case. Genesis lives at a lower layer as defensive guard
        alongside upstream's primary fix; the two are orthogonal, not
        substitutes. Status-based retire would remove the defensive
        layer."""
        verdict = M.retire_eligibility({
            "pr": _merged_pr(),
            "pid": "P_LIKE_PN51",
            "lifecycle": "experimental",
            "upstream_pr_relationship": "defensive_overlay",
        })
        assert verdict == "NEEDS-DEEP-PARITY"

    def test_related_not_superseding_is_needs_deep_parity_not_retire(self):
        """PN90 / PN24 case. Genesis lives at a different layer with
        coverage that does NOT overlap the cited PR; the relationship
        is informational, not supersession. Status-based retire would
        misread the relationship."""
        verdict = M.retire_eligibility({
            "pr": _merged_pr(),
            "pid": "P_LIKE_PN90",
            "lifecycle": "experimental",
            "upstream_pr_relationship": "related_not_superseding",
        })
        assert verdict == "NEEDS-DEEP-PARITY"

    def test_counter_regression_is_needs_deep_parity_not_retire(self):
        """PN116 case. Genesis corrects a regression INTRODUCED by the
        cited PR. The PR being merged is precisely the condition that
        makes the Genesis fix necessary, not the condition for retire."""
        verdict = M.retire_eligibility({
            "pr": _merged_pr(),
            "pid": "P_LIKE_PN116",
            "lifecycle": "experimental",
            "upstream_pr_relationship": "counter_regression",
        })
        assert verdict == "NEEDS-DEEP-PARITY"

    # ─ Lifecycle / PR-state precedence ─

    def test_retired_lifecycle_returns_already_retired(self):
        verdict = M.retire_eligibility({
            "pr": _merged_pr(),
            "pid": "P_FAKE",
            "lifecycle": "retired",
            "upstream_pr_relationship": "backport",
        })
        assert verdict == "ALREADY-RETIRED"

    def test_retired_with_open_upstream_returns_already_retired(self):
        verdict = M.retire_eligibility({
            "pr": _open_pr(),
            "pid": "P_FAKE",
            "lifecycle": "retired",
            "upstream_pr_relationship": "backport",
        })
        assert verdict == "ALREADY-RETIRED"

    def test_open_pr_active_local_is_active(self):
        verdict = M.retire_eligibility({
            "pr": _open_pr(),
            "pid": "P_FAKE",
            "lifecycle": "stable",
            "upstream_pr_relationship": "backport",
        })
        assert verdict == "ACTIVE"

    def test_pr_error_is_unknown(self):
        verdict = M.retire_eligibility({
            "pr": {"error": "boom"},
            "pid": "P_FAKE",
            "lifecycle": "stable",
            "upstream_pr_relationship": "backport",
        })
        assert verdict == "UNKNOWN"

    def test_issue_reference_is_unknown(self):
        """Issues don't have merge semantics — retire-by-status is
        nonsensical. retire_eligibility must NOT classify them as
        RETIRE-CANDIDATE."""
        for issue_state in (_open_issue(), _closed_issue()):
            verdict = M.retire_eligibility({
                "pr": issue_state,
                "pid": "P_FAKE",
                "lifecycle": "stable",
                "upstream_pr_relationship": "backport",
            })
            assert verdict == "UNKNOWN", (
                f"issue {issue_state['state']!r} was classified as "
                f"{verdict!r}; must be UNKNOWN to keep retire-by-status off"
            )


class TestLiveRegistryFalsePositiveLock:
    """End-to-end safety net: walk the same 5 patches that the
    2026-05-23 sidecleanup audit caught as false-positives and confirm
    `retire_eligibility()` returns NEEDS-DEEP-PARITY for each, NOT
    RETIRE-CANDIDATE. If a future operator silently changes one of these
    patches' relationship to `backport`, this test fails loudly.
    """

    # PR merge state pinned for determinism (mirrors PIN.R recon 2026-05-23).
    # We don't query GitHub in this test — we simulate the merged-PR state
    # that the live recon observed for each.
    SIDECLEANUP_FALSE_POSITIVES = {
        "P75":   "enables_upstream",
        "P99":   "enables_upstream",
        "P98":   "intentional_inverse",
        "PN51":  "defensive_overlay",
        "PN90":  "related_not_superseding",
    }

    def test_each_false_positive_resolves_to_needs_deep_parity(self):
        entries = M._load_registry_entries()
        for pid, expected_rel in self.SIDECLEANUP_FALSE_POSITIVES.items():
            assert pid in entries, (
                f"{pid}: not in live PATCH_REGISTRY anymore — "
                f"sidecleanup false-positive lock is now stale"
            )
            actual_rel = M._extract_upstream_pr_relationship(entries[pid])
            assert actual_rel == expected_rel, (
                f"{pid}: relationship drifted from {expected_rel!r} "
                f"to {actual_rel!r}; the false-positive lock depends "
                f"on the registry field staying non-pure"
            )
            lifecycle = M._extract_lifecycle(entries[pid])
            verdict = M.retire_eligibility({
                "pr": _merged_pr(),
                "pid": pid,
                "lifecycle": lifecycle,
                "upstream_pr_relationship": actual_rel,
            })
            assert verdict == "NEEDS-DEEP-PARITY", (
                f"{pid}: retire_eligibility returned {verdict!r}; "
                f"must be NEEDS-DEEP-PARITY for non-pure relationship "
                f"{actual_rel!r} regardless of upstream PR merge state"
            )
