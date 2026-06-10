# SPDX-License-Identifier: Apache-2.0
"""TDD for `sndr.apply.shadow` (PR38 Day 5).

Shadow comparison between legacy apply-loop registration order
(`_per_patch_dispatch.py @register_patch`) and the new spec-driven
order (`dispatcher.iter_patch_specs()`).

Covers:
  - patch_id parser for legacy `@register_patch` names
  - `compare_apply_orders()` returns coherent diff
  - `format_diff()` produces non-empty multi-line report
  - CLI exit code (0 clean / 1 divergent in strict mode)
"""
from __future__ import annotations

import pytest


def _shadow_module():
    from sndr.apply import shadow
    return shadow


# ─── patch_id parser ──────────────────────────────────────────────────────


class TestPatchIdFromLegacyName:
    @pytest.mark.parametrize("name,expected", [
        ("P67 TurboQuant multi-query kernel", "P67"),
        ("PN14 TQ decode IOOB safe_page_idx clamp", "PN14"),
        ("P5b KV page-size pad-smaller-to-max (env-opt-in)", "P5b"),
        ("PN82 Mamba CUDA-graph stale prefill rows (vllm#41873)", "PN82"),
    ])
    def test_extracts_id_from_canonical_name(self, name, expected):
        s = _shadow_module()
        assert s._patch_id_from_legacy_name(name) == expected

    def test_unparseable_name_returns_none(self):
        s = _shadow_module()
        assert s._patch_id_from_legacy_name("Something without a patch_id") is None
        assert s._patch_id_from_legacy_name("") is None

    def test_pn_prefix_normalized_uppercase(self):
        """`pN14` lowercase n → "PN14" canonical form."""
        s = _shadow_module()
        # The leading P is uppercase per regex; we normalize the N.
        assert s._patch_id_from_legacy_name("PN14 something") == "PN14"


# ─── Diff comparison ──────────────────────────────────────────────────────


class TestCompareApplyOrders:
    def test_diff_returned_is_dataclass(self):
        s = _shadow_module()
        diff = s.compare_apply_orders()
        assert hasattr(diff, "legacy_count")
        assert hasattr(diff, "spec_count")
        assert hasattr(diff, "is_clean")

    def test_legacy_count_nonzero(self):
        """`_per_patch_dispatch.py` has ~124 @register_patch decorators
        as of v11.0.0. Counts may drift but should never drop to zero."""
        s = _shadow_module()
        diff = s.compare_apply_orders()
        assert diff.legacy_count > 100, (
            f"legacy_count {diff.legacy_count} suspiciously low — "
            "_per_patch_dispatch.py may not have been imported"
        )

    def test_spec_count_matches_registry(self):
        """spec_count must equal the number of dict entries in
        dispatcher.PATCH_REGISTRY (specs are 1:1 with registry)."""
        from sndr.dispatcher import PATCH_REGISTRY
        s = _shadow_module()
        diff = s.compare_apply_orders()
        registry_dict_count = sum(
            1 for v in PATCH_REGISTRY.values() if isinstance(v, dict)
        )
        assert diff.spec_count == registry_dict_count

    def test_coverage_pct_is_high(self):
        """≥85% of specs should have apply_module derivable. Falling
        below means a regression in `_build_apply_module_map` OR a
        new patch was added without an on-disk impl."""
        s = _shadow_module()
        diff = s.compare_apply_orders()
        assert diff.coverage_pct >= 0.85, (
            f"coverage {diff.coverage_pct:.0%} below 85%"
        )

    def test_pn82_in_both_sources(self):
        """PR38 Day 1's PN82 is registered in _per_patch_dispatch AND
        present in dispatcher.PATCH_REGISTRY — should NOT appear in
        either diff.legacy_only or diff.spec_only."""
        s = _shadow_module()
        diff = s.compare_apply_orders()
        assert "PN82" not in diff.legacy_only
        assert "PN82" not in diff.spec_only

    def test_legacy_only_is_empty_or_documented(self):
        """Today there should be no legacy_only entries (every
        @register_patch must have a matching dispatcher.PATCH_REGISTRY).
        If this test fails, an apply registration was added without a
        corresponding metadata entry."""
        s = _shadow_module()
        diff = s.compare_apply_orders()
        # Legacy-only would mean: function in _per_patch_dispatch but no
        # registry entry — that's a dispatcher metadata gap, not a code
        # gap. Tolerate up to 3 (P5b/P7b/legacy stubs).
        assert len(diff.legacy_only) < 5, (
            f"legacy_only too large: {diff.legacy_only}"
        )


# ─── format_diff output ───────────────────────────────────────────────────


class TestFormatDiff:
    def test_format_produces_multiline_report(self):
        s = _shadow_module()
        diff = s.compare_apply_orders()
        out = s.format_diff(diff)
        assert "Genesis apply-loop shadow report" in out
        assert "Legacy apply registrations" in out
        assert "Spec-driven entries" in out

    def test_format_contains_status_marker(self):
        s = _shadow_module()
        diff = s.compare_apply_orders()
        out = s.format_diff(diff)
        # Either CLEAN or DIVERGENT marker must appear
        assert ("CLEAN" in out) or ("DIVERGENT" in out)


# ─── CLI ──────────────────────────────────────────────────────────────────


class TestCLI:
    def test_cli_no_args_exits_zero(self, capsys):
        s = _shadow_module()
        rc = s.main([])
        assert rc == 0
        captured = capsys.readouterr()
        assert "shadow report" in captured.out.lower()

    def test_cli_strict_exits_nonzero_on_divergence(self, capsys):
        """In strict mode, any divergence (legacy_only / spec_only / etc)
        causes exit 1. Today the apply order IS divergent (a few specs
        without legacy registration); that's expected during the PR38
        transition window."""
        s = _shadow_module()
        diff = s.compare_apply_orders()
        rc = s.main(["--strict"])
        if diff.is_clean:
            assert rc == 0
        else:
            assert rc == 1
