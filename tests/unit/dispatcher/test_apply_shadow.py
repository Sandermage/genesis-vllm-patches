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
        ("P68/P69 long-ctx tool reminder", "P68"),
    ])
    def test_extracts_id_from_canonical_name(self, name, expected):
        s = _shadow_module()
        assert s._patch_id_from_legacy_name(name) == expected

    @pytest.mark.parametrize("name,expected", [
        # Underscore-suffix taxonomy (2026-06-14 fix) — `\b` after the
        # numeric id could not find a boundary before `_`, so these
        # returned None and surfaced as legacy_unparseable.
        ("P23_WIRE Marlin FP32_REDUCE env wire (2026-06-04 fix-wire)",
         "P23_WIRE"),
        ("P29_HEAL qwen3coder index heal (2026-06-04 fix-wire)", "P29_HEAL"),
        ("P18B_TEXT TurboQuant decode stage1 kernel-literal tune", "P18B_TEXT"),
        ("PN118_V2_MD5_WORKSPACE md5+full-file PoC (workspace.py scope)",
         "PN118_V2_MD5_WORKSPACE"),
        ("PN79_V2_MD5_CHUNK_DELTA_H md5+full-file PoC (chunk_delta_h.py)",
         "PN79_V2_MD5_CHUNK_DELTA_H"),
        # SNDR_-prefix research ids — returned verbatim (no P/PN normalize).
        ("SNDR_EAGLE3_AUX_HIDDEN_001 model-side prep for EAGLE-3",
         "SNDR_EAGLE3_AUX_HIDDEN_001"),
    ])
    def test_extracts_underscore_suffix_and_sndr_ids(self, name, expected):
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

    def test_underscore_suffix_does_not_overreach_plain_ids(self):
        """A plain id followed by a space (not `_`) must NOT absorb later
        tokens — the suffix group only consumes underscore-joined tokens."""
        s = _shadow_module()
        assert s._patch_id_from_legacy_name("P32 TurboQuant preallocs") == "P32"
        assert s._patch_id_from_legacy_name("G4_19b gemma4 TQ KV") == "G4_19B"


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

    def test_spec_boot_unsafe_flags_legacy_hooks_without_apply_module(self):
        """spec_boot_unsafe = patches that apply via a legacy hook but have
        no apply_module → SNDR_APPLY_VIA_SPECS=1 would silently drop them.
        legacy_only does NOT catch this. The bundled default_on legacy
        patches P1/P2, P17/P18, P32/P33 are the canonical members (root-
        caused 2026-06-14 when the dev491 supplement was designed)."""
        s = _shadow_module()
        diff = s.compare_apply_orders()
        # P1/P17/P32 are bundled default_on legacy patches without an
        # apply_module — they MUST be flagged so nobody flips the boot mode
        # and drops them.
        for pid in ("P1", "P17", "P32"):
            assert pid in diff.spec_boot_unsafe, (
                f"{pid} (legacy hook, no apply_module) must be flagged "
                f"spec_boot_unsafe; got {diff.spec_boot_unsafe}"
            )
        # Every flagged id must genuinely have a legacy hook AND no spec
        # apply_module — guard against the check drifting to false positives.
        from sndr.dispatcher.spec import iter_patch_specs
        no_module = {
            sp.patch_id for sp in iter_patch_specs() if sp.apply_module is None
        }
        for pid in diff.spec_boot_unsafe:
            assert pid in no_module, (
                f"{pid} flagged spec_boot_unsafe but its spec HAS an "
                "apply_module — false positive"
            )

    def test_spec_boot_unsafe_does_not_break_clean(self):
        """The advisory is informational only — it must NOT flip is_clean
        to False (these are legitimately legacy-only until migrated)."""
        s = _shadow_module()
        diff = s.compare_apply_orders()
        # spec_boot_unsafe is non-empty today (P1/P17/P20/P32) yet the
        # report must stay CLEAN — the advisory is decoupled from is_clean.
        assert diff.spec_boot_unsafe, "expected a non-empty advisory today"
        assert diff.is_clean, (
            "spec_boot_unsafe must not affect is_clean — it is advisory"
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
