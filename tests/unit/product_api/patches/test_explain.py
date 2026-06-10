# SPDX-License-Identifier: Apache-2.0
"""Tests for ``product_api.patches.explain`` — M.6.1."""
from __future__ import annotations

from sndr.product_api.legacy.patches import explain
from sndr.product_api.legacy.patches.types import ExplainView


class TestResolvePatchId:
    def test_exact_match(self):
        assert explain.resolve_patch_id("P67") == "P67"

    def test_case_insensitive(self):
        # Real registry uses upper-case canonical keys.
        assert explain.resolve_patch_id("p67") == "P67"

    def test_unknown_returns_none(self):
        assert explain.resolve_patch_id("PXXXX_NOT_REAL") is None


class TestSuggestCandidates:
    def test_returns_prefix_matches(self):
        cands = explain.suggest_candidates("PN")
        # Plenty of PN-prefix keys exist; cap at 8.
        assert len(cands) <= 8
        assert all(c.startswith("PN") for c in cands)

    def test_unknown_prefix_empty(self):
        cands = explain.suggest_candidates("ZZ999")
        assert cands == []


class TestExplainPatch:
    def test_known_returns_view(self):
        view = explain.explain_patch("P67")
        assert isinstance(view, ExplainView)
        assert view.patch_id == "P67"
        assert view.meta is not None
        assert view.spec.patch_id == "P67"

    def test_case_insensitive_canonical(self):
        view = explain.explain_patch("p67")
        assert view is not None
        assert view.patch_id == "P67"  # canonical case preserved

    def test_unknown_returns_none(self):
        assert explain.explain_patch("not-a-real-id") is None

    def test_live_decision_either_value_or_error(self):
        """``live_decision`` is a (bool, str) tuple when the dispatcher
        probe succeeded, ``None`` otherwise — in which case
        ``live_decision_error`` carries the exception class name."""
        view = explain.explain_patch("P67")
        assert view is not None
        if view.live_decision is None:
            assert isinstance(view.live_decision_error, str)
        else:
            applied, reason = view.live_decision
            assert isinstance(applied, bool)
            assert isinstance(reason, str)
