# SPDX-License-Identifier: Apache-2.0
"""Tests for ``product_api.patches.listing`` — M.6.1.

Pure-data layer behind ``sndr patches list``. Mirrors the legacy
``TestFilters`` / ``TestList`` coverage from ``test_patches_cli.py`` so
the contract stays enforced when CLI back-compat shims are removed in
M.6.4.
"""
from __future__ import annotations

from sndr.product_api.legacy.patches import listing
from sndr.product_api.legacy.patches.types import PatchRow


class TestMatchesFilters:
    def test_tier_filter(self):
        from sndr.dispatcher.spec import iter_patch_specs

        for s in iter_patch_specs():
            assert listing.matches_filters(s, tier=s.tier) is True
            assert listing.matches_filters(s, tier="__never__") is False

    def test_default_on(self):
        from sndr.dispatcher.spec import iter_patch_specs

        for s in iter_patch_specs():
            assert listing.matches_filters(s, default_on=True) is bool(s.default_on)
            assert listing.matches_filters(s, default_on=False) is (not s.default_on)

    def test_has_upstream(self):
        from sndr.dispatcher.spec import iter_patch_specs

        for s in iter_patch_specs():
            assert listing.matches_filters(s, has_upstream=True) is bool(s.upstream_pr)

    def test_family_substring_match(self):
        from sndr.dispatcher.spec import iter_patch_specs

        # Every spec matches a substring of its own family (sanity).
        for s in iter_patch_specs():
            if s.family:
                assert listing.matches_filters(s, family=s.family[:1]) is True


class TestSpecToRow:
    def test_returns_patchrow(self):
        from sndr.dispatcher import PATCH_REGISTRY
        from sndr.dispatcher.spec import patch_spec_for

        meta = PATCH_REGISTRY["P67"]
        row = listing.spec_to_row(patch_spec_for("P67", meta))
        assert isinstance(row, PatchRow)
        assert row.patch_id == "P67"
        assert row.production_default in {"applied", "marker", "opt-in", "blocked"}

    def test_dict_form_back_compat(self):
        """``spec_to_row_dict`` returns the legacy dict shape callers
        relied on before M.6.1 — keys match ``PatchRow`` fields."""
        from sndr.dispatcher import PATCH_REGISTRY
        from sndr.dispatcher.spec import patch_spec_for

        meta = PATCH_REGISTRY["P67"]
        d = listing.spec_to_row_dict(patch_spec_for("P67", meta))
        for key in (
            "patch_id", "tier", "lifecycle", "family",
            "default_on", "production_default",
            "implementation_status", "env_flag",
            "upstream_pr", "title", "apply_module",
        ):
            assert key in d


class TestListPatches:
    def test_unfiltered_count_matches_iter_patch_specs(self):
        from sndr.dispatcher.spec import iter_patch_specs

        rows = listing.list_patches()
        assert len(rows) == sum(1 for _ in iter_patch_specs())
        # All entries are PatchRow.
        assert all(isinstance(r, PatchRow) for r in rows)

    def test_sorted_by_patch_id(self):
        rows = listing.list_patches()
        ids = [r.patch_id for r in rows]
        assert ids == sorted(ids)

    def test_tier_filter_subset(self):
        all_rows = listing.list_patches()
        community_rows = listing.list_patches(tier="community")
        engine_rows = listing.list_patches(tier="engine")
        assert len(community_rows) + len(engine_rows) <= len(all_rows)

    def test_has_upstream_filter(self):
        rows = listing.list_patches(has_upstream=True)
        assert all(r.upstream_pr is not None for r in rows)
