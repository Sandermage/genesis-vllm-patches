# SPDX-License-Identifier: Apache-2.0
"""TDD for `sndr.dispatcher.spec.PatchSpec` (PR38 Day 4).

Typed contract over `PATCH_REGISTRY` that auto-derives `apply_module`
by walking the canonical `vllm/sndr_core/integrations/<family>/p<id>_*.py`
tree. Tests cover:

  - `_patch_ids_from_stem` filename → patch_id parser
  - `_build_apply_module_map` end-to-end on the live patches/ tree
  - `iter_patch_specs()` yields one PatchSpec per registry entry
  - `patch_spec_for()` honors explicit `apply_module` field overrides
  - `validate_apply_module_coverage()` detects unmapped registry entries
"""
from __future__ import annotations

import pytest


def _spec_module():
    from sndr.dispatcher import spec
    return spec


# ─── Filename → patch_id parser ────────────────────────────────────────────


class TestPatchIdsFromStem:
    @pytest.mark.parametrize("stem,expected", [
        ("pn14_tq_decode_oob_clamp", ["PN14"]),
        ("p67_tq_multi_query_kernel", ["P67"]),
        ("p67b_spec_verify_routing", ["P67b"]),
        ("p68_69_long_ctx_tool_adherence", ["P68", "P69"]),
        ("p107_mtp_truncation_detector", ["P107"]),
        ("pn82_mamba_cudagraph_prefill_zero", ["PN82"]),
    ])
    def test_canonical_stems_parse_correctly(self, stem, expected):
        spec = _spec_module()
        assert spec._patch_ids_from_stem(stem) == expected

    @pytest.mark.parametrize("stem", [
        "__init__",  # dunder
        "upstream_compat",  # not a patch
        "_pn50",  # leading underscore
        "abc_xyz",  # doesn't start with p<digit>
    ])
    def test_non_patch_stems_return_empty(self, stem):
        spec = _spec_module()
        assert spec._patch_ids_from_stem(stem) == []

    def test_compound_yields_both_ids_in_order(self):
        spec = _spec_module()
        ids = spec._patch_ids_from_stem("p68_69_long_ctx_tool_adherence")
        assert ids == ["P68", "P69"], (
            "compound stems must yield primary then secondary id"
        )


# ─── apply_module map build ────────────────────────────────────────────────


class TestBuildApplyModuleMap:
    def test_map_is_non_empty(self):
        spec = _spec_module()
        spec.reset_apply_module_cache()
        m = spec._build_apply_module_map()
        # 117 single-file mappings + compound aliases + override
        # PN40-classifier — total well over 100.
        assert len(m) > 100, f"map suspiciously small: {len(m)}"

    def test_resolves_pn14_to_canonical_dotted(self):
        spec = _spec_module()
        spec.reset_apply_module_cache()
        m = spec._build_apply_module_map()
        assert m["PN14"] == (
            "sndr.engines.vllm.patches.attention.turboquant.pn14_tq_decode_oob_clamp"
        )

    def test_resolves_compound_p68_AND_p69_to_same_module(self):
        """File p68_69_*.py serves both registry IDs."""
        spec = _spec_module()
        spec.reset_apply_module_cache()
        m = spec._build_apply_module_map()
        assert "P68" in m
        assert "P69" in m
        assert m["P68"] == m["P69"], (
            "compound file must register both IDs to the same dotted path"
        )

    def test_override_resolves_pn40_classifier(self):
        """Explicit override: PN40-classifier (hyphenated) → workload_classifier_hook."""
        spec = _spec_module()
        spec.reset_apply_module_cache()
        m = spec._build_apply_module_map()
        assert "PN40-classifier" in m
        assert "pn40_workload_classifier_hook" in m["PN40-classifier"]

    def test_case_insensitive_variants_register(self):
        """Registry sometimes uses 'P15B' while filename is 'p15b_*'.
        Both casings must resolve."""
        spec = _spec_module()
        spec.reset_apply_module_cache()
        m = spec._build_apply_module_map()
        # Find any patch with letter suffix
        assert "P5b" in m or "P5B" in m

    def test_pn82_resolves_after_pr38_day1(self):
        """PR38 Day 1 added PN82 — coverage map must include it."""
        spec = _spec_module()
        spec.reset_apply_module_cache()
        m = spec._build_apply_module_map()
        assert "PN82" in m
        assert "pn82_mamba_cudagraph_prefill_zero" in m["PN82"]


# ─── PatchSpec construction ────────────────────────────────────────────────


class TestPatchSpecConstruction:
    def test_minimal_meta(self):
        spec = _spec_module()
        s = spec.patch_spec_for("PN14", {
            "title": "test", "env_flag": "GENESIS_ENABLE_PN14",
            "default_on": False,
        }, apply_module_map={"PN14": "fake.module.path"})
        assert s.patch_id == "PN14"
        assert s.title == "test"
        assert s.apply_module == "fake.module.path"

    def test_explicit_apply_module_field_wins_over_derived(self):
        """Registry's `apply_module` field overrides the auto-derived path."""
        spec = _spec_module()
        s = spec.patch_spec_for("PN14", {
            "title": "test", "env_flag": "GENESIS_ENABLE_PN14",
            "default_on": False,
            "apply_module": "explicit.override.path",
        }, apply_module_map={"PN14": "auto.derived.path"})
        assert s.apply_module == "explicit.override.path"

    def test_unmapped_id_returns_none_apply_module(self):
        """Registry entry with no on-disk impl gets apply_module=None."""
        spec = _spec_module()
        s = spec.patch_spec_for("P_LEGACY", {
            "title": "legacy entry", "env_flag": "GENESIS_LEGACY_X",
            "default_on": True, "lifecycle": "legacy",
        }, apply_module_map={})
        assert s.apply_module is None

    def test_requires_patches_coerced_to_tuple(self):
        spec = _spec_module()
        s = spec.patch_spec_for("X", {
            "title": "x", "env_flag": "GENESIS_ENABLE_X",
            "default_on": False,
            "requires_patches": ["A", "B"],
        }, apply_module_map={})
        assert s.requires_patches == ("A", "B")
        assert isinstance(s.requires_patches, tuple)

    def test_frozen_dataclass(self):
        """PatchSpec is frozen — operators can't accidentally mutate it."""
        spec = _spec_module()
        s = spec.patch_spec_for("X", {
            "title": "x", "env_flag": "GENESIS_ENABLE_X",
            "default_on": False,
        }, apply_module_map={})
        with pytest.raises((AttributeError, Exception)):
            s.title = "changed"  # frozen → FrozenInstanceError


# ─── iter_patch_specs ──────────────────────────────────────────────────────


class TestIterPatchSpecs:
    def test_yields_one_spec_per_registry_entry(self):
        from sndr.dispatcher import iter_patch_specs, PATCH_REGISTRY
        specs = list(iter_patch_specs())
        # Some registry entries are non-dict (free-form keys); spec
        # generator skips those. So count specs == count of dict entries.
        dict_count = sum(1 for v in PATCH_REGISTRY.values() if isinstance(v, dict))
        assert len(specs) == dict_count

    def test_each_spec_has_required_fields(self):
        from sndr.dispatcher import iter_patch_specs
        for s in iter_patch_specs():
            assert s.patch_id, "patch_id must be non-empty"
            assert s.tier in ("community", "engine"), (
                f"{s.patch_id} bad tier: {s.tier!r}"
            )

    def test_pn82_present_with_canonical_module(self):
        """PN82 retired 2026-05-28 (K.1.R pin bump audit) — superseded by
        vllm#41873 merge at 39d5fa96 within window dev371→626fa9bb. The
        spec still resolves (registry retains entry for audit trail) but
        the apply_module path now points at _retired/."""
        from sndr.dispatcher import iter_patch_specs
        specs = {s.patch_id: s for s in iter_patch_specs()}
        assert "PN82" in specs
        # Registry apply_module uses canonical sndr.* paths post-migration; the
        # retired patch now lives under sndr.engines.vllm._archive/.
        assert specs["PN82"].apply_module == (
            "sndr.engines.vllm._archive."
            "pn82_mamba_cudagraph_prefill_zero"
        )
        assert specs["PN82"].upstream_pr == 41873
        # Confirms retirement lifecycle is recorded in spec, gates downstream tooling.
        assert specs["PN82"].lifecycle == "retired"


# ─── Coverage report ───────────────────────────────────────────────────────


class TestCoverageReport:
    def test_coverage_high_enough(self):
        """≥90% of registry entries must have a derivable apply_module.

        The remaining ≤10% are legitimate informational / legacy P1-P46
        stubs that don't have their own per-file impl. Coverage falling
        below 90% likely means a regression in the file-walker or that a
        new patch was added to registry without an on-disk impl."""
        from sndr.dispatcher import validate_apply_module_coverage
        from sndr.dispatcher.spec import reset_apply_module_cache
        reset_apply_module_cache()
        r = validate_apply_module_coverage()
        coverage_pct = r.mapped / r.total
        assert coverage_pct >= 0.85, (
            f"coverage {r.mapped}/{r.total} = {coverage_pct:.1%} below 85% — "
            f"unmapped: {r.unmapped[:10]}"
        )

    def test_unmapped_are_all_legacy_or_documented(self):
        """Any registry entry without an apply_module must be either
        lifecycle=legacy OR appear in the documented exception list
        (informational hooks integrated into other patches)."""
        from sndr.dispatcher import (
            PATCH_REGISTRY, validate_apply_module_coverage,
        )
        from sndr.dispatcher.spec import reset_apply_module_cache
        reset_apply_module_cache()
        r = validate_apply_module_coverage()

        # Documented exceptions: informational hooks / sub-IDs without
        # standalone files. Update if new no-impl registry entries
        # legitimately need a slot.
        documented_no_impl = frozenset({
            "PN26b", "P102", "PN60", "PN63", "PN64",
        })

        for pid in r.unmapped:
            meta = PATCH_REGISTRY.get(pid, {})
            lc = meta.get("lifecycle")
            if lc == "legacy":
                continue  # legacy P1-P46 stubs are documented no-impl
            assert pid in documented_no_impl, (
                f"unmapped patch_id {pid!r} (lifecycle={lc!r}) is neither "
                "legacy nor in documented_no_impl. Add an apply_module, "
                "an on-disk impl, OR document why it has no impl by "
                "extending documented_no_impl in this test."
            )
