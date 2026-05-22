# SPDX-License-Identifier: Apache-2.0
"""Phase 5.1.A (2026-05-22) — `audit_upstream_status.py` classify() routing.

The audit script's `categorize()` translates a (PR-state, lifecycle,
registry-driven relationship hint) triple into one of the audit buckets.
Phase 5.1.A added three buckets:

  - COUNTER-REGRESSION       — `upstream_pr_relationship: counter_regression`
  - DEFENSIVE-OVERLAY        — `upstream_pr_relationship: defensive_overlay`
  - RELATED-NOT-SUPERSEDING  — `upstream_pr_relationship: related_not_superseding`

Existing buckets that the new field can drive:

  - INTENTIONAL-INVERSE      — explicit hint (preferred over hardcoded waiver)
  - ENABLES-UPSTREAM         — explicit hint (preferred over legacy boolean)

The hardcoded waiver dicts and the legacy `enables_upstream_feature: True`
boolean stay as fallback during the 5.1.A → 5.1.C migration window. These
tests verify both routing layers (explicit-field-first, fallback-second).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

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


# ─── Phase 5.1.A new buckets via explicit relationship field ──────────────


class TestExplicitRelationshipRouting:

    def test_counter_regression_routes_to_new_bucket(self):
        cat = M.categorize({
            "pr": _merged_pr(),
            "pid": "P_FAKE",
            "lifecycle": "stable",
            "upstream_pr_relationship": "counter_regression",
            "enables_upstream_feature": False,
        })
        assert cat == "COUNTER-REGRESSION"

    def test_defensive_overlay_routes_to_new_bucket(self):
        cat = M.categorize({
            "pr": _merged_pr(),
            "pid": "P_FAKE",
            "lifecycle": "stable",
            "upstream_pr_relationship": "defensive_overlay",
            "enables_upstream_feature": False,
        })
        assert cat == "DEFENSIVE-OVERLAY"

    def test_related_not_superseding_routes_to_new_bucket(self):
        cat = M.categorize({
            "pr": _merged_pr(),
            "pid": "P_FAKE",
            "lifecycle": "stable",
            "upstream_pr_relationship": "related_not_superseding",
            "enables_upstream_feature": False,
        })
        assert cat == "RELATED-NOT-SUPERSEDING"

    def test_intentional_inverse_routes_via_explicit_hint(self):
        """Explicit hint preferred over the hardcoded P98 waiver dict."""
        cat = M.categorize({
            "pr": _merged_pr(),
            "pid": "P_FAKE",  # not in _INTENTIONAL_INVERSE_WAIVER
            "lifecycle": "experimental",
            "upstream_pr_relationship": "intentional_inverse",
            "enables_upstream_feature": False,
        })
        assert cat == "INTENTIONAL-INVERSE"

    def test_enables_upstream_routes_via_explicit_hint(self):
        """Explicit hint preferred over `enables_upstream_feature: True`."""
        cat = M.categorize({
            "pr": _merged_pr(),
            "pid": "P_FAKE",
            "lifecycle": "stable",
            "upstream_pr_relationship": "enables_upstream",
            "enables_upstream_feature": False,
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
            "enables_upstream_feature": False,
        })
        assert cat == "NEWLY-MERGED"


# ─── Implicit-backport / back-compat fallbacks ────────────────────────────


class TestBackCompatFallbacks:

    def test_missing_relationship_treated_as_backport(self):
        """During Phase 5.1.A migration window: absent field == backport.
        Merged + stable lifecycle == NEWLY-MERGED action queue."""
        cat = M.categorize({
            "pr": _merged_pr(),
            "pid": "P_FAKE",
            "lifecycle": "stable",
            "upstream_pr_relationship": None,
            "enables_upstream_feature": False,
        })
        assert cat == "NEWLY-MERGED"

    def test_legacy_boolean_fallback_still_works(self):
        """Phase 5.1.A keeps the legacy boolean working until 5.1.C
        migrates P75/P99 to explicit field syntax."""
        cat = M.categorize({
            "pr": _merged_pr(),
            "pid": "P75",
            "lifecycle": "stable",
            "upstream_pr_relationship": None,
            "enables_upstream_feature": True,
        })
        assert cat == "ENABLES-UPSTREAM"

    def test_hardcoded_intentional_inverse_waiver_still_works(self):
        """P98 waiver dict survives until 5.1.C cleanup."""
        cat = M.categorize({
            "pr": _merged_pr(),
            "pid": "P98",
            "lifecycle": "experimental",
            "upstream_pr_relationship": None,
            "enables_upstream_feature": False,
        })
        assert cat == "INTENTIONAL-INVERSE"

    def test_hardcoded_internal_supersession_waiver_still_works(self):
        """P61 waiver dict survives until 5.1.C cleanup."""
        cat = M.categorize({
            "pr": _open_pr(),
            "pid": "P61",
            "lifecycle": "retired",
            "upstream_pr_relationship": None,
            "enables_upstream_feature": False,
        })
        assert cat == "RETIRED-INTERNAL"


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
            "enables_upstream_feature": False,
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
            "enables_upstream_feature": False,
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
            "enables_upstream_feature": False,
        })
        assert cat == "WATCH"

    def test_open_pr_retired_local_is_stale_retired(self):
        cat = M.categorize({
            "pr": _open_pr(),
            "pid": "P_FAKE",
            "lifecycle": "retired",
            "upstream_pr_relationship": None,
            "enables_upstream_feature": False,
        })
        assert cat == "STALE-RETIRED"

    def test_open_issue_routes_to_issue_open(self):
        cat = M.categorize({
            "pr": _open_issue(),
            "pid": "P_FAKE",
            "lifecycle": "stable",
            "upstream_pr_relationship": None,
            "enables_upstream_feature": False,
        })
        assert cat == "ISSUE-OPEN"

    def test_closed_issue_routes_to_issue_closed(self):
        cat = M.categorize({
            "pr": _closed_issue(),
            "pid": "P_FAKE",
            "lifecycle": "stable",
            "upstream_pr_relationship": None,
            "enables_upstream_feature": False,
        })
        assert cat == "ISSUE-CLOSED"

    def test_pr_error_routes_to_error(self):
        cat = M.categorize({
            "pr": {"error": "boom"},
            "pid": "P_FAKE",
            "lifecycle": "stable",
            "upstream_pr_relationship": None,
            "enables_upstream_feature": False,
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
