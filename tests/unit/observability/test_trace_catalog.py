# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr.observability.trace_catalog` — §6.H6
of the unified development plan.

Locks the surface of TRACE_CATALOG so future patch additions register
themselves through a single append rather than ad-hoc grep.
"""
from __future__ import annotations

import pytest

from sndr.observability.trace_catalog import (
    TRACE_CATALOG,
    TRACE_CATEGORIES,
    TraceSpec,
    find_by_id,
    find_by_patch,
    iter_by_category,
)


# ─── TRACE_CATALOG contract ───────────────────────────────────────────


class TestCatalogContract:
    def test_catalog_is_non_empty(self):
        assert len(TRACE_CATALOG) >= 1

    def test_every_entry_is_traceSpec(self):
        for s in TRACE_CATALOG:
            assert isinstance(s, TraceSpec)

    def test_ids_are_unique(self):
        ids = [s.id for s in TRACE_CATALOG]
        assert len(ids) == len(set(ids)), (
            f"duplicate trace id(s): {sorted(set(x for x in ids if ids.count(x) > 1))}"
        )

    def test_paths_are_unique(self):
        paths = [s.container_path for s in TRACE_CATALOG]
        assert len(paths) == len(set(paths)), (
            "duplicate container_path entries"
        )

    def test_all_paths_under_tmp_genesis_prefix(self):
        """The H7 collect verb scans /tmp/genesis_*. Any entry outside
        that prefix would silently never get picked up."""
        for s in TRACE_CATALOG:
            assert s.container_path.startswith("/tmp/genesis_"), (
                f"{s.id!r} container_path {s.container_path!r} does "
                "not begin with /tmp/genesis_ — H7 scanner would skip it"
            )

    def test_categories_are_valid_enum(self):
        for s in TRACE_CATALOG:
            assert s.category in TRACE_CATEGORIES, (
                f"{s.id!r} has invalid category {s.category!r} — must "
                f"be in TRACE_CATEGORIES ({TRACE_CATEGORIES})"
            )

    def test_enable_env_present_or_marked_always(self):
        """An entry has either an env flag OR `enable_env=None` to mean
        'always written'. Empty string is invalid."""
        for s in TRACE_CATALOG:
            if s.enable_env is not None:
                assert isinstance(s.enable_env, str) and s.enable_env, (
                    f"{s.id!r} has falsy enable_env {s.enable_env!r}"
                )

    def test_description_is_non_empty_string(self):
        for s in TRACE_CATALOG:
            assert isinstance(s.description, str) and s.description.strip()

    def test_at_least_one_always_on_trace(self):
        """The launcher writes /tmp/genesis_boot.log unconditionally —
        the catalog must reflect that as enable_env=None so the H6 CLI
        doesn't tell the operator to set an env flag that doesn't exist."""
        always_on = [s for s in TRACE_CATALOG if s.enable_env is None]
        assert len(always_on) >= 1


# ─── find_by_id ──────────────────────────────────────────────────────


class TestFindById:
    def test_returns_spec_for_known_id(self):
        spec = find_by_id("boot")
        assert spec is not None
        assert spec.id == "boot"
        assert spec.category == "boot"

    def test_returns_none_for_unknown(self):
        assert find_by_id("nope_not_a_trace") is None

    def test_returns_none_for_empty(self):
        assert find_by_id("") is None


# ─── find_by_patch ───────────────────────────────────────────────────


class TestFindByPatch:
    def test_returns_empty_for_unknown_patch(self):
        assert find_by_patch("PN9999999") == ()

    def test_returns_singleton_for_single_emitter(self):
        out = find_by_patch("PN248")
        assert len(out) == 1
        assert out[0].id == "pn248_acceptance"

    def test_returns_all_for_multi_emitter(self):
        """PN258 emits both `pn258_oracle` (txt) and `pn258_oracle_trace`
        (log). The helper must return BOTH so the H7 collect verb
        doesn't miss one."""
        out = find_by_patch("PN258")
        assert len(out) == 2
        ids = {s.id for s in out}
        assert {"pn258_oracle", "pn258_oracle_trace"} == ids


# ─── iter_by_category ────────────────────────────────────────────────


class TestIterByCategory:
    def test_returns_all_canonical_categories(self):
        out = iter_by_category()
        assert tuple(out.keys()) == TRACE_CATEGORIES

    def test_each_category_holds_only_matching_specs(self):
        out = iter_by_category()
        for cat, specs in out.items():
            for s in specs:
                assert s.category == cat, (
                    f"spec {s.id!r} in category {cat!r} bucket but "
                    f"its own .category is {s.category!r}"
                )

    def test_no_spec_dropped(self):
        out = iter_by_category()
        total = sum(len(v) for v in out.values())
        assert total == len(TRACE_CATALOG), (
            "iter_by_category dropped some specs"
        )
