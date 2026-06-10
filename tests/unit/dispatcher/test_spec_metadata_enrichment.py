# SPDX-License-Identifier: Apache-2.0
"""S5 / P1-2 audit closure (2026-05-08 noonghunna): registry metadata
enrichment.

Audit finding: 132/132 specs missing ``implementation_status`` and
``category`` fields. Closure: PatchSpec now carries three new fields
(``category``, ``implementation_status``, ``source``) with intelligent
inference defaults so existing entries don't need bulk edits.

Tests cover:

  • Inference helpers (family→category, lifecycle→status, upstream_pr→source)
  • Explicit registry override wins over inference
  • Real PATCH_REGISTRY: every entry resolves to a valid enum value
  • Backward compat: existing PatchSpec consumers unaffected
"""
from __future__ import annotations

import pytest

from sndr.dispatcher.spec import (
    VALID_CATEGORIES,
    VALID_IMPLEMENTATION_STATUSES,
    VALID_SOURCES,
    infer_category,
    infer_implementation_status,
    infer_source,
    iter_patch_specs,
    patch_spec_for,
)


# ─── Inference helpers ─────────────────────────────────────────────────


class TestInferCategory:
    @pytest.mark.parametrize(
        "family,expected",
        [
            ("attention.gdn", "gdn"),
            ("attention.turboquant", "quantization"),
            ("spec_decode", "spec_decode"),
            ("spec-decode", "spec_decode"),
            ("structured_output", "structured_output"),
            ("tool_parsing", "tool_parsing"),
            ("reasoning", "reasoning"),
            ("moe", "moe"),
            ("kernels", "kernel"),
            ("kv_cache", "kv_cache"),
            ("middleware", "observability"),
            ("multimodal", "memory"),
            # Family prefix fallback
            ("attention", "attention"),
            ("attention.unknown_subfamily", "attention"),
        ],
    )
    def test_known_family_mappings(self, family, expected):
        assert infer_category(family) == expected

    def test_unknown_family_returns_uncategorized(self):
        assert infer_category("totally_unknown_family") == "uncategorized"

    def test_empty_family(self):
        assert infer_category("") == "uncategorized"
        assert infer_category(None) == "uncategorized"  # type: ignore[arg-type]

    def test_all_returns_in_valid_categories(self):
        """Every value the helper returns must be in the canonical enum."""
        for fam in [
            "attention.gdn", "spec_decode", "moe", "memory", "totally-bogus",
        ]:
            assert infer_category(fam) in VALID_CATEGORIES


class TestInferImplementationStatus:
    def test_explicit_field_wins(self):
        meta = {"implementation_status": "blocked", "lifecycle": "retired"}
        assert infer_implementation_status(meta) == "blocked"

    def test_retired_lifecycle_maps_to_retired(self):
        meta = {"lifecycle": "retired"}
        assert infer_implementation_status(meta) == "retired"

    def test_deprecated_lifecycle_maps_to_retired(self):
        meta = {"lifecycle": "deprecated"}
        assert infer_implementation_status(meta) == "retired"

    def test_research_lifecycle_maps_to_research(self):
        meta = {"lifecycle": "research"}
        assert infer_implementation_status(meta) == "research"

    def test_default_to_live(self):
        """Without patch_id and without explicit `implementation_status` — fallback to lifecycle-based.
        `stable` → `full` (production-grade default, audit P1-2 closure 2026-05-12).
        Without lifecycle at all — `live` (generic fallback).
        """
        assert infer_implementation_status({"lifecycle": "stable"}) == "full"
        assert infer_implementation_status({}) == "live"

    def test_returns_in_valid_enum(self):
        for status in [
            infer_implementation_status({"lifecycle": "retired"}),
            infer_implementation_status({"lifecycle": "research"}),
            infer_implementation_status({}),
            infer_implementation_status({"implementation_status": "upstream_merged"}),
        ]:
            assert status in VALID_IMPLEMENTATION_STATUSES


class TestInferSource:
    def test_explicit_field_wins(self):
        meta = {"source": "club_3090_adapted", "upstream_pr": 12345}
        assert infer_source(meta) == "club_3090_adapted"

    def test_upstream_pr_int_maps_to_vllm_pr_backport(self):
        meta = {"upstream_pr": 40269}
        assert infer_source(meta) == "vllm_pr_backport"

    def test_related_upstream_prs_maps_to_vllm_pr_backport(self):
        meta = {"related_upstream_prs": [40807, 40269]}
        assert infer_source(meta) == "vllm_pr_backport"

    def test_club_3090_in_credit(self):
        meta = {"credit": "noonghunna club-3090#51 finding"}
        assert infer_source(meta) == "club_3090_adapted"

    def test_sglang_in_credit(self):
        meta = {"credit": "Backport from SGLang #21019"}
        assert infer_source(meta) == "cross_engine_research"

    def test_llama_cpp_in_credit(self):
        meta = {"credit": "Adapted from llama.cpp ngram cache heuristic"}
        assert infer_source(meta) == "cross_engine_research"

    def test_default_genesis_original(self):
        meta = {"credit": "Genesis-original 2026-04-29"}
        assert infer_source(meta) == "genesis_original"

    def test_no_credit_no_pr(self):
        assert infer_source({}) == "genesis_original"

    def test_returns_in_valid_enum(self):
        cases = [
            {"upstream_pr": 1},
            {"credit": "club-3090"},
            {"credit": "SGLang research"},
            {},
        ]
        for meta in cases:
            assert infer_source(meta) in VALID_SOURCES


# ─── PatchSpec integration ─────────────────────────────────────────────


class TestPatchSpecEnrichment:
    def test_spec_has_category_field(self):
        meta = {
            "title": "test",
            "family": "spec_decode",
            "tier": "community",
            "lifecycle": "stable",
        }
        spec = patch_spec_for("PTEST", meta, apply_module_map={})
        assert spec.category == "spec_decode"

    def test_spec_has_implementation_status_field(self):
        meta = {
            "title": "retired test",
            "family": "memory",
            "tier": "community",
            "lifecycle": "retired",
        }
        spec = patch_spec_for("PRETIRED", meta, apply_module_map={})
        assert spec.implementation_status == "retired"

    def test_spec_has_source_field(self):
        meta = {
            "title": "backport test",
            "family": "spec_decode",
            "tier": "community",
            "lifecycle": "stable",
            "upstream_pr": 12345,
        }
        spec = patch_spec_for("PBP", meta, apply_module_map={})
        assert spec.source == "vllm_pr_backport"

    def test_explicit_category_overrides_family_inference(self):
        """Operator can override inferred category by setting it explicitly."""
        meta = {
            "title": "x",
            "family": "spec_decode",  # would infer "spec_decode"
            "tier": "community",
            "lifecycle": "stable",
            "category": "research",  # explicit override
        }
        spec = patch_spec_for("POVERRIDE", meta, apply_module_map={})
        assert spec.category == "research"


# ─── Real registry: every entry has valid enrichment ──────────────────


class TestRealRegistryEnrichment:
    def test_every_spec_has_valid_category(self):
        invalid = [
            (s.patch_id, s.category)
            for s in iter_patch_specs()
            if s.category not in VALID_CATEGORIES
        ]
        assert invalid == [], (
            f"specs with invalid category: {invalid}. "
            f"Add the family→category mapping to spec.py "
            "_FAMILY_TO_CATEGORY OR set explicit `category` on the entry."
        )

    def test_every_spec_has_valid_status(self):
        invalid = [
            (s.patch_id, s.implementation_status)
            for s in iter_patch_specs()
            if s.implementation_status not in VALID_IMPLEMENTATION_STATUSES
        ]
        assert invalid == [], (
            f"specs with invalid implementation_status: {invalid}"
        )

    def test_every_spec_has_valid_source(self):
        invalid = [
            (s.patch_id, s.source)
            for s in iter_patch_specs()
            if s.source not in VALID_SOURCES
        ]
        assert invalid == [], (
            f"specs with invalid source: {invalid}"
        )

    def test_uncategorized_count_under_threshold(self):
        """Most specs should resolve to a real category, not uncategorized.
        When this threshold breaks, add a mapping to _FAMILY_TO_CATEGORY."""
        uncategorized = [
            s.patch_id for s in iter_patch_specs()
            if s.category == "uncategorized"
        ]
        # Allow a small grace band — set hard threshold low and grow when
        # legitimate
        assert len(uncategorized) < 30, (
            f"too many uncategorized patches ({len(uncategorized)}): "
            f"first few = {uncategorized[:5]}"
        )

    def test_distribution_summary(self):
        """Smoke: print distribution so operators see at a glance.

        Not a strict assertion — just ensures the spec loop runs without
        error against the real registry."""
        import collections
        cat_count = collections.Counter(s.category for s in iter_patch_specs())
        status_count = collections.Counter(
            s.implementation_status for s in iter_patch_specs()
        )
        source_count = collections.Counter(s.source for s in iter_patch_specs())
        # Sanity: at least one of each enum should appear in the real registry
        assert cat_count.most_common(1), "no specs?"
        assert status_count.most_common(1)
        assert source_count.most_common(1)
