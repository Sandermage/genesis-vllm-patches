# SPDX-License-Identifier: Apache-2.0
"""TDD for PN32 v3 — GDN _forward_core chunked-prefill re-anchor + PN79 composition.

2026-06-11 re-anchor (pin 0.22.1rc1.dev259+g303916e93, preflight residual
triage plan section 1b): the v2 anchor died with the upstream gdn/ split
and the mixed-batch decode peel-off rework (#44700). The prefill branch is
now the `# 2.3` block of `QwenGatedDeltaNetAttention._forward_core` at
pristine `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`
lines 1503-1532.

Composition hazard (verifier-mandated): PN79 sub-patch 3C
(`pn79_inplace_ssm_state.py` ANCHOR_3C_PREFILL_INPLACE_OLD, PROD-applied)
anchors on the IDENTICAL pristine block (lines 1509-1532). PN32 v3
therefore carries TWO anchor variants with required-at-least-one
semantics (both `required=False`; the TextPatcher kernel returns SKIPPED
`no_applicable_sub_patches` when every sub-patch misses):

  - pristine-shaped variant — matches an untouched pristine file
    (PN79 disabled);
  - post-PN79-shaped variant — matches the file AFTER PN79 3C applied
    (anchor assembled from PN79's own ANCHOR_3C_PREFILL_INPLACE_NEW
    constant, chain convention per PN365-imports-PN50 precedent).

Apply-order dependency: PN79 (boot dispatch ~line 2782) runs BEFORE PN32
(~line 4270). The reverse order breaks PN79 (its required 3C anchor would
no longer match) — asserted textually below.

These tests verify, textually on the committed pristine fixture
(`tests/legacy/pristine_fixtures/qwen_gdn_linear_attn.py`, md5-identical
to /private/tmp/candidate_pin_current on the extraction host):
  1. pristine variant matches pristine exactly once; post-PN79 variant
     matches zero times (mutual exclusion, direction 1)
  2. after applying PN79's 3C replacement to a pristine copy, the
     post-PN79 variant matches exactly once and the pristine variant
     matches zero times (mutual exclusion, direction 2)
  3. end-to-end TextPatcher apply on tmp copies (both shapes) — APPLIED
     with exactly the expected variant, and the result compiles
  4. replacement contract: single-seq detection via
     prefill_query_start_loc.shape[0] == 2, persist into
     ssm_state[prefill_state_indices], cutedsl bypass guard (CustomOp
     forward_cutedsl asserts chunk_indices/chunk_offsets not None at
     pristine lines 398-400 — the chunked path passes None for both)
  5. self-collision invariants: drift markers disjoint from emitted text
  6. PN32-before-PN79 ordering breakage is documented (PN79's 3C anchor
     dies on a PN32-patched file)
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
PRISTINE_FIXTURE = (
    REPO_ROOT / "tests" / "legacy" / "pristine_fixtures" / "qwen_gdn_linear_attn.py"
)


def _pn32():
    from sndr.engines.vllm.patches.attention.gdn import pn32_gdn_chunked_prefill as M
    return M


def _pn79():
    from sndr.engines.vllm.patches.attention.gdn import pn79_inplace_ssm_state as M
    return M


def _pristine() -> str:
    if not PRISTINE_FIXTURE.is_file():
        pytest.skip(f"pristine fixture not found: {PRISTINE_FIXTURE}")
    return PRISTINE_FIXTURE.read_text(encoding="utf-8")


def _post_pn79() -> str:
    """Pristine source with PN79's 3C replacement applied (textual)."""
    pn79 = _pn79()
    src = _pristine()
    assert src.count(pn79.ANCHOR_3C_PREFILL_INPLACE_OLD) == 1, (
        "PN79 3C OLD anchor no longer unique in the pristine fixture — "
        "fixture or PN79 drifted; PN32 composition must be re-verified"
    )
    return src.replace(
        pn79.ANCHOR_3C_PREFILL_INPLACE_OLD,
        pn79.ANCHOR_3C_PREFILL_INPLACE_NEW,
        1,
    )


# ─────────────────────────────────────────────────────────────────────
# 1+2. Anchor variant matching — required-at-least-one, mutual exclusion
# ─────────────────────────────────────────────────────────────────────


class TestAnchorVariants:

    def test_pristine_variant_matches_pristine_exactly_once(self):
        M = _pn32()
        assert _pristine().count(M.PN32_ANCHOR) == 1

    def test_post_pn79_variant_absent_from_pristine(self):
        M = _pn32()
        assert _pristine().count(M.PN32_ANCHOR_POST_PN79) == 0

    def test_post_pn79_variant_matches_post_pn79_exactly_once(self):
        M = _pn32()
        assert _post_pn79().count(M.PN32_ANCHOR_POST_PN79) == 1

    def test_pristine_variant_absent_from_post_pn79(self):
        M = _pn32()
        assert _post_pn79().count(M.PN32_ANCHOR) == 0

    def test_pristine_anchor_equals_header_plus_pn79_3c_old(self):
        """Cross-check: PN32's quoted pristine block must be byte-identical
        to the `# 2.3` header + PN79's 3C OLD anchor — both patches anchor
        on the same upstream code, so a drift in one constant must fail
        loudly here."""
        M, pn79 = _pn32(), _pn79()
        assert M.PN32_ANCHOR == (
            M.PN32_BLOCK_HEADER + pn79.ANCHOR_3C_PREFILL_INPLACE_OLD
        )

    def test_post_pn79_anchor_built_from_pn79_constant(self):
        """Chain convention: the post-PN79 anchor must be derived from
        PN79's ANCHOR_3C_PREFILL_INPLACE_NEW constant so the two modules
        cannot silently diverge."""
        M, pn79 = _pn32(), _pn79()
        assert M.PN32_ANCHOR_POST_PN79 == (
            M.PN32_BLOCK_HEADER + pn79.ANCHOR_3C_PREFILL_INPLACE_NEW
        )

    def test_both_variants_required_false(self):
        """Required-at-least-one semantics: both sub-patches must be
        required=False so the kernel soft-skips the non-matching variant
        and SKIPs (no_applicable_sub_patches) only when BOTH miss."""
        M = _pn32()
        from sndr.kernel import TextPatcher

        patcher = TextPatcher(
            patch_name="pn32-v3-variant-semantics-probe",
            target_file="/nonexistent",
            marker=M.GENESIS_PN32_MARKER,
            sub_patches=M.build_sub_patches(),
        )
        assert len(patcher.sub_patches) == 2
        assert all(not sp.required for sp in patcher.sub_patches)


# ─────────────────────────────────────────────────────────────────────
# 3. End-to-end TextPatcher apply on tmp copies of both shapes
# ─────────────────────────────────────────────────────────────────────


def _apply_pn32_to(tmp_path, content: str):
    M = _pn32()
    from sndr.kernel import TextPatcher

    target = tmp_path / "qwen_gdn_linear_attn.py"
    target.write_text(content, encoding="utf-8")
    patcher = TextPatcher(
        patch_name="pn32-v3-e2e-probe",
        target_file=str(target),
        marker=M.GENESIS_PN32_MARKER,
        sub_patches=M.build_sub_patches(),
        upstream_drift_markers=list(M.PN32_UPSTREAM_DRIFT_MARKERS),
    )
    result, failure = patcher.apply()
    return result, failure, patcher, target


class TestEndToEndApply:

    def test_applies_on_pristine_via_pristine_variant(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        from sndr.kernel import TextPatchResult

        result, failure, patcher, target = _apply_pn32_to(tmp_path, _pristine())
        assert result == TextPatchResult.APPLIED, failure
        assert patcher.applied_sub_patches == [
            "pN32_v3_forward_core_chunked_prefill_pristine"
        ]
        # Patched file must still be valid Python.
        compile(target.read_text(encoding="utf-8"), str(target), "exec")

    def test_applies_on_post_pn79_via_post_variant(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        from sndr.kernel import TextPatchResult

        result, failure, patcher, target = _apply_pn32_to(tmp_path, _post_pn79())
        assert result == TextPatchResult.APPLIED, failure
        assert patcher.applied_sub_patches == [
            "pN32_v3_forward_core_chunked_prefill_post_pn79"
        ]
        compile(target.read_text(encoding="utf-8"), str(target), "exec")

    def test_skips_when_both_variants_miss(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        from sndr.kernel import TextPatchResult

        result, failure, _, _ = _apply_pn32_to(
            tmp_path, "def unrelated():\n    return 0\n"
        )
        assert result == TextPatchResult.SKIPPED
        assert failure is not None
        assert failure.reason == "no_applicable_sub_patches"

    def test_idempotent_on_second_apply(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        M = _pn32()
        from sndr.kernel import TextPatcher, TextPatchResult

        result, _, _, target = _apply_pn32_to(tmp_path, _pristine())
        assert result == TextPatchResult.APPLIED
        second = TextPatcher(
            patch_name="pn32-v3-idempotency-probe",
            target_file=str(target),
            marker=M.GENESIS_PN32_MARKER,
            sub_patches=M.build_sub_patches(),
        )
        result2, _ = second.apply()
        assert result2 == TextPatchResult.IDEMPOTENT


# ─────────────────────────────────────────────────────────────────────
# 4. Replacement contract (plan section 1b)
# ─────────────────────────────────────────────────────────────────────


class TestReplacementContract:

    def test_single_seq_detection_via_prefill_query_start_loc(self):
        """Plan: single-seq detection via prefill_query_start_loc.shape[0]
        == 2 (the v2 non_spec_query_start_loc no longer carries the
        prefill cu_seqlens on this pin — the builder peels decodes off)."""
        M = _pn32()
        for repl in (M.PN32_REPLACEMENT, M.PN32_REPLACEMENT_POST_PN79):
            assert "prefill_query_start_loc.shape[0] == 2" in repl
            assert "non_spec_query_start_loc.shape[0]" not in repl

    def test_persists_into_prefill_state_indices(self):
        """Plan: persist into ssm_state[prefill_state_indices] (v2 wrote
        to ssm_state[non_spec_state_indices_tensor] — stale on this pin)."""
        M = _pn32()
        persist = "ssm_state[prefill_state_indices] = last_recurrent_state.to("
        for repl in (M.PN32_REPLACEMENT, M.PN32_REPLACEMENT_POST_PN79):
            assert persist in repl
            assert "ssm_state[non_spec_state_indices_tensor]" not in repl

    def test_cutedsl_bypass_guard_present(self):
        """Plan (verified at pristine lines 398-400): forward_cutedsl
        asserts chunk_indices/chunk_offsets are not None; the chunked
        path passes None for both, so cutedsl must bypass to original."""
        M = _pn32()
        for repl in (M.PN32_REPLACEMENT, M.PN32_REPLACEMENT_POST_PN79):
            assert "self.gdn_prefill_backend != 'cutedsl'" in repl
            assert "chunk_indices=None," in repl
            assert "chunk_offsets=None," in repl

    def test_chunk_local_cu_seqlens_dtype_from_prefill_query_start_loc(self):
        M = _pn32()
        for repl in (M.PN32_REPLACEMENT, M.PN32_REPLACEMENT_POST_PN79):
            assert "dtype=attn_metadata.prefill_query_start_loc.dtype," in repl

    def test_post_pn79_replacement_preserves_pn79_inplace_path(self):
        """The non-chunked branch of the post-PN79 replacement must keep
        PN79's backend-gated in-place logic (re-indented under else:)."""
        M = _pn32()
        repl = M.PN32_REPLACEMENT_POST_PN79
        assert '_pn79_inplace = self.gdn_prefill_backend == "triton"' in repl
        assert '"ssm_state_indices": prefill_state_indices,' in repl
        assert "if not _pn79_inplace:" in repl

    def test_pristine_replacement_has_no_pn79_text(self):
        M = _pn32()
        assert "[Genesis PN79" not in M.PN32_REPLACEMENT
        assert "_pn79_" not in M.PN32_REPLACEMENT


# ─────────────────────────────────────────────────────────────────────
# 5. Self-collision invariants (preflight plan section 6 lint contract)
# ─────────────────────────────────────────────────────────────────────


class TestSelfCollision:

    def test_drift_markers_disjoint_from_emitted_text(self):
        M = _pn32()
        marker_line = f"# [Genesis wiring marker: {M.GENESIS_PN32_MARKER}]\n"
        for dm in M.PN32_UPSTREAM_DRIFT_MARKERS:
            assert dm not in M.PN32_REPLACEMENT
            assert dm not in M.PN32_REPLACEMENT_POST_PN79
            assert dm not in marker_line

    def test_drift_markers_absent_from_pristine(self):
        M = _pn32()
        src = _pristine()
        for dm in M.PN32_UPSTREAM_DRIFT_MARKERS:
            assert dm not in src

    def test_replacements_do_not_resurrect_either_anchor(self):
        """Sequential-apply safety: neither replacement may contain either
        anchor as a substring, or the sibling variant would double-apply."""
        M = _pn32()
        for repl in (M.PN32_REPLACEMENT, M.PN32_REPLACEMENT_POST_PN79):
            assert M.PN32_ANCHOR not in repl
            assert M.PN32_ANCHOR_POST_PN79 not in repl


# ─────────────────────────────────────────────────────────────────────
# 6. Ordering constraint documentation (PN79 must run first)
# ─────────────────────────────────────────────────────────────────────


class TestOrderingConstraint:

    def test_pn32_before_pn79_breaks_pn79_anchor(self, tmp_path, monkeypatch):
        """The REVERSE order (PN32 first) kills PN79's required 3C anchor —
        this is WHY the boot dispatch order (PN79 before PN32) is a hard
        constraint and not a style choice."""
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        pn79 = _pn79()
        from sndr.kernel import TextPatchResult

        result, _, _, target = _apply_pn32_to(tmp_path, _pristine())
        assert result == TextPatchResult.APPLIED
        post_pn32 = target.read_text(encoding="utf-8")
        assert post_pn32.count(pn79.ANCHOR_3C_PREFILL_INPLACE_OLD) == 0

    def test_module_documents_apply_order_dependency(self):
        import inspect

        M = _pn32()
        src = inspect.getsource(M)
        assert "PN79" in src
        # Ordering must be stated explicitly for the registry owner.
        assert "before PN32" in src or "PN79 first" in src
