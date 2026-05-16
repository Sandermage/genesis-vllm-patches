# SPDX-License-Identifier: Apache-2.0
"""Tests for `vllm.sndr_core.dispatcher.registry_metadata` overlay.

production_default is derived from both implementation_status and
test_status. A `lifecycle=stable` entry without tests resolves to
`review_required`, not the unconditional `eligible` it used to
report. The `_production_default_for(impl_status, test_status)`
helper centralises that mapping so the matrix is unit-testable in
isolation from the live registry.

Coverage:
  - impl_status x test_status -> production_default matrix
  - EXPLICIT_OVERRIDES bypass (audited overrides win)
  - Lifecycle fallback when implementation_status absent
  - Live registry: at least one entry resolves to review_required
    so the overlay isn't silently returning eligible everywhere.
"""
from __future__ import annotations

import pytest

from vllm.sndr_core.dispatcher.registry_metadata import (
    EXPLICIT_OVERRIDES,
    _LIFECYCLE_TO_IMPL,
    _production_default_for,
    derive_metadata,
)


# ─── _production_default_for matrix ─────────────────────────────────────


class TestProductionDefaultMatrix:
    """Pure cells of the (impl_status, test_status) -> production_default
    function. Decoupled from the real registry so changes in YAML data
    don't break the matrix."""

    @pytest.mark.parametrize("impl,test,expected", [
        # Blocked statuses — outcome is independent of test_status.
        ("partial",     "unit", "blocked"),
        ("partial",     "none", "blocked"),
        ("placeholder", "unit", "blocked"),
        ("placeholder", "none", "blocked"),
        ("retired",     "unit", "blocked"),
        ("retired",     "none", "blocked"),
        # research lifecycle gates the patch behind an explicit research
        # flag regardless of how well it is tested.
        ("research", "unit", "research_only"),
        ("research", "none", "research_only"),
        # Otherwise-ok statuses with no tests need explicit review.
        ("full",        "none", "review_required"),
        ("live",        "none", "review_required"),
        ("scaffold",    "none", "review_required"),
        ("coordinator", "none", "review_required"),
        # Tested ok statuses are immediately eligible.
        ("full",        "unit",        "eligible"),
        ("full",        "integration", "eligible"),
        ("full",        "bench",       "eligible"),
        ("live",        "unit",        "eligible"),
        ("scaffold",    "unit",        "eligible"),
        ("coordinator", "unit",        "eligible"),
    ])
    def test_matrix(self, impl, test, expected):
        assert _production_default_for(impl, test) == expected


# ─── derive_metadata flow ───────────────────────────────────────────────


class TestDeriveMetadata:
    def test_explicit_override_wins(self):
        """EXPLICIT_OVERRIDES bypasses the derive pipeline entirely."""
        # PN95 is overridden to partial/blocked.
        d = derive_metadata("PN95", {"lifecycle": "stable", "family": "kv_cache"})
        assert d["implementation_status"] == "partial"
        assert d["production_default"] == "blocked"

    def test_explicit_status_in_registry_uses_helper(self):
        """When registry sets implementation_status explicitly, the
        production_default still flows through `_production_default_for`."""
        d = derive_metadata(
            "ZNEW1",
            {"implementation_status": "full", "family": "spec_decode"},
        )
        assert d["implementation_status"] == "full"
        # No tests/ files for ZNEW1 -> test_status=none -> review_required.
        assert d["test_status"] == "none"
        assert d["production_default"] == "review_required"

    def test_lifecycle_fallback_uses_helper(self):
        """Lifecycle-based fallback (no explicit impl_status) also
        routes through the helper."""
        d = derive_metadata(
            "ZNEW2",
            {"lifecycle": "stable", "family": "spec_decode"},
        )
        assert d["implementation_status"] == "full"
        assert d["production_default"] == "review_required"

    def test_lifecycle_retired_blocked(self):
        d = derive_metadata("ZNEW3", {"lifecycle": "retired"})
        assert d["implementation_status"] == "retired"
        assert d["production_default"] == "blocked"

    def test_lifecycle_research_research_only(self):
        d = derive_metadata("ZNEW4", {"lifecycle": "research"})
        assert d["implementation_status"] == "research"
        assert d["production_default"] == "research_only"

    def test_unknown_lifecycle_falls_to_live(self):
        d = derive_metadata("ZNEW5", {"lifecycle": "totally-unknown"})
        assert d["implementation_status"] == "live"
        assert d["production_default"] == "review_required"


# ─── _LIFECYCLE_TO_IMPL canonical mapping ───────────────────────────────


class TestLifecycleMapping:
    def test_all_known_lifecycles_mapped(self):
        """Every lifecycle state used in the codebase must be in
        `_LIFECYCLE_TO_IMPL` (or it falls through to the 'live' default)."""
        for lc in ("retired", "deprecated", "research", "stable",
                   "coordinator", "legacy"):
            assert lc in _LIFECYCLE_TO_IMPL

    def test_stable_maps_to_full(self):
        assert _LIFECYCLE_TO_IMPL["stable"] == "full"

    def test_deprecated_maps_to_retired(self):
        assert _LIFECYCLE_TO_IMPL["deprecated"] == "retired"


# ─── Real registry sanity ───────────────────────────────────────────────


class TestRealRegistry:
    """Smoke check against the live registry — `eligible` and
    `review_required` should both appear in production_default
    output. If everything is eligible, the helper is silently
    bypassed somewhere."""

    def test_real_registry_has_review_required_entries(self):
        """At least one patch must resolve to `review_required`,
        otherwise either test coverage is total (unlikely) or the
        helper is bypassed."""
        from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
        review_required = []
        for pid, meta in PATCH_REGISTRY.items():
            if not isinstance(meta, dict):
                continue
            d = derive_metadata(pid, meta)
            if d["production_default"] == "review_required":
                review_required.append(pid)
        assert len(review_required) > 0, (
            "review_required count = 0 — either every patch has tests "
            "(unlikely) or the helper is bypassed."
        )

    def test_overrides_not_review_required(self):
        """Every EXPLICIT_OVERRIDES entry must declare a curated
        production_default (eligible / blocked / research_only) —
        never `review_required`, which is the auto-derived state."""
        for pid, override in EXPLICIT_OVERRIDES.items():
            assert override["production_default"] != "review_required", (
                f"EXPLICIT_OVERRIDES[{pid}] should have audited "
                f"production_default, not 'review_required'"
            )
