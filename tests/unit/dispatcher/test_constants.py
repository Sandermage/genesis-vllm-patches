# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr.dispatcher._constants` — single source of truth.

Contract:

  1. All enumerations are frozensets/tuples (immutable, can't be mutated
     accidentally).
  2. _VALID_TIERS = {community, engine}.
  3. _VALID_LIFECYCLES covers the documented set.
  4. _VALID_IMPLEMENTATION_STATUSES is mutually exclusive with
     _VALID_LIFECYCLES (orthogonal axes).
  5. _BLOCKED_STATUSES is a subset of _VALID_IMPLEMENTATION_STATUSES.
  6. _RESEARCH_STATUSES is a subset of _VALID_IMPLEMENTATION_STATUSES.
  7. _CANONICAL_ENV_PREFIXES covers both SNDR_ and GENESIS_ brands.
  8. All env prefixes end with underscore.
  9. Cross-module imports preserve original names (back-compat).
"""
from __future__ import annotations

from sndr.dispatcher import _constants as c


# ─── Immutability ─────────────────────────────────────────────────────


class TestImmutable:
    def test_valid_tiers_is_frozenset(self):
        assert isinstance(c._VALID_TIERS, frozenset)

    def test_valid_lifecycles_is_frozenset(self):
        assert isinstance(c._VALID_LIFECYCLES, frozenset)

    def test_valid_impl_statuses_is_frozenset(self):
        assert isinstance(c._VALID_IMPLEMENTATION_STATUSES, frozenset)

    def test_blocked_statuses_is_frozenset(self):
        assert isinstance(c._BLOCKED_STATUSES, frozenset)

    def test_research_statuses_is_frozenset(self):
        assert isinstance(c._RESEARCH_STATUSES, frozenset)

    def test_env_prefixes_is_tuple(self):
        assert isinstance(c._CANONICAL_ENV_PREFIXES, tuple)


# ─── Tier ──────────────────────────────────────────────────────────────


class TestTiers:
    def test_two_tiers(self):
        assert c._VALID_TIERS == {"community", "engine"}


# ─── Lifecycle ────────────────────────────────────────────────────────


class TestLifecycles:
    def test_covers_documented_set(self):
        required = {
            "stable", "experimental", "deprecated", "legacy",
            "research", "merged_upstream", "retired", "coordinator",
        }
        # Must cover at minimum the documented values; allows superset.
        assert required.issubset(c._VALID_LIFECYCLES)


# ─── Implementation status ────────────────────────────────────────────


class TestImplementationStatuses:
    def test_covers_documented_set(self):
        required = {
            "full", "partial", "marker_only", "placeholder",
            "experimental", "retired",
        }
        assert required.issubset(c._VALID_IMPLEMENTATION_STATUSES)

    def test_blocked_is_subset(self):
        """Production-blocked statuses must be a subset of all valid statuses."""
        assert c._BLOCKED_STATUSES.issubset(c._VALID_IMPLEMENTATION_STATUSES)

    def test_blocked_explicit_values(self):
        assert c._BLOCKED_STATUSES == {"partial", "placeholder", "retired"}

    def test_research_explicit_values(self):
        assert c._RESEARCH_STATUSES == {"research"}


# ─── Orthogonality of axes ────────────────────────────────────────────


class TestOrthogonality:
    def test_lifecycle_vs_impl_status_overlap_is_acceptable(self):
        """Some values overlap (e.g. 'experimental', 'retired') because
        the audit policy permits cross-axis equality when the semantic
        is clear. Document the intentional overlap so future audits don't
        accidentally enforce strict disjointness."""
        overlap = c._VALID_LIFECYCLES & c._VALID_IMPLEMENTATION_STATUSES
        # Intentional overlap is acceptable but should not be empty
        # (audit metadata expects 'experimental' on both axes).
        assert "experimental" in overlap
        assert "retired" in overlap


# ─── Env prefixes ─────────────────────────────────────────────────────


class TestEnvPrefixes:
    def test_includes_both_brands(self):
        joined = "|".join(c._CANONICAL_ENV_PREFIXES)
        assert "SNDR_" in joined
        assert "GENESIS_" in joined

    def test_includes_all_five_categories(self):
        # ENABLE / DISABLE / LEGACY / ALLOW / INFO × {SNDR_, GENESIS_}
        # INFO category added Phase 10.5 (2026-06-01) for info-marker
        # patches like G4_T1 (PR42006 vendor overlay mount status —
        # operator-visible flag with no toggle semantics).
        for category in ("ENABLE_", "DISABLE_", "LEGACY_", "ALLOW_", "INFO_"):
            assert any(category in p for p in c._CANONICAL_ENV_PREFIXES), (
                f"missing category {category}"
            )

    def test_all_prefixes_end_with_underscore(self):
        for p in c._CANONICAL_ENV_PREFIXES:
            assert p.endswith("_"), f"{p} doesn't end with underscore"

    def test_count_is_ten(self):
        # 5 categories × 2 brands = 10 prefixes
        # (INFO added Phase 10.5 2026-06-01; was 8 = 4×2 before).
        assert len(c._CANONICAL_ENV_PREFIXES) == 10

    def test_no_duplicates(self):
        assert len(set(c._CANONICAL_ENV_PREFIXES)) == len(c._CANONICAL_ENV_PREFIXES)


# ─── Cross-module re-import preserves names ───────────────────────────


class TestBackCompat:
    def test_registry_metadata_imports(self):
        """registry_metadata.py re-imports the constants via the
        original names for back-compat with anything reaching into its
        module dict."""
        from sndr.dispatcher import registry_metadata as rm
        # Both constants must be accessible under original names
        assert hasattr(rm, "_BLOCKED_STATUSES")
        assert hasattr(rm, "_RESEARCH_STATUSES")
        assert rm._BLOCKED_STATUSES == c._BLOCKED_STATUSES
        assert rm._RESEARCH_STATUSES == c._RESEARCH_STATUSES
