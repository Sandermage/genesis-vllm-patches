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
