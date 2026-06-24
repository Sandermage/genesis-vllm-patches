# SPDX-License-Identifier: Apache-2.0
"""TDD for P85 both-sites fix — re-anchor + PN346 composition (plan section 5).

2026-06-11 (pin 0.22.1rc1.dev259+g303916e93, preflight residual triage
action plan section 5):

Site 1 — upstream widened ``MambaManager.cache_blocks`` to the new
``retention_interval`` keyword signature (pristine
``v1/core/single_type_kv_cache_manager.py`` lines 1211-1229). The old
single-line-signature anchor matches zero times on this pin; P85's
required Site 1 sub-patch failed on every boot. Fix: re-derive
P85_SITE1_OLD/NEW byte-exact from pristine, forwarding
``retention_interval`` through ``super().cache_blocks``.

Site 2 — composition hazard (verifier-mandated, the only ``agree:
false`` row of the triage): sibling PN346 (effectively default-ON —
only ``GENESIS_DISABLE_PN346`` is honored; boot-dispatched at
``_per_patch_dispatch.py`` line ~5416, BEFORE P85 at ~5945) rewrites a
byte-identical 4-line subsequence inside P85_SITE2_OLD, inserting 12
lines mid-anchor. The pristine-shaped Site 2 anchor therefore fails on
every real (post-PN346) boot even though it byte-matches the pristine
file. P85 v2 carries TWO Site 2 anchor variants with
required-at-least-one semantics (both ``required=False``; the
TextPatcher kernel soft-skips the variant whose anchor is absent), per
the P18B / PN32-on-PN79 chain convention:

  - pristine-shaped variant (``P85_SITE2_OLD``) — matches an untouched
    pristine file (PN346 disabled via GENESIS_DISABLE_PN346=1);
  - post-PN346-shaped variant (``P85_SITE2_OLD_POST_PN346``) — matches
    the file AFTER PN346 applied; assembled textually from PN346's own
    PN346_ANCHOR_OLD/NEW constants (chain convention, PN32-imports-PN79
    precedent) so the two modules cannot silently diverge. Its
    replacement carries PN346's ``drop_eagle_block`` guard in the
    coarse fallback.

Because Site 1 is ``required=True``, the kernel's all-miss SKIP cannot
fire for a Site-2-only drift; ``apply()`` adds an explicit pre-gate
(``site2_anchor_present``) returning a structured skip BEFORE any write
when neither Site 2 variant matches.

2026-06-24 (pin bump -> 0.23.1rc1.dev301+g04c2a8dea): Site 1 re-anchored
again. Upstream rewrote the ``MambaManager.cache_blocks`` loop body —
the old ``if block.is_null: continue / assert block.block_hash is not
None`` pair folded into ``if block.is_null or block.block_hash is None:
continue`` plus a sparse-retention comment. ``P85_SITE1_OLD/NEW`` and the
committed fixture's MambaManager.cache_blocks body were updated byte-
exact to dev301 (em-dash preserved). Site 2 is unchanged (still matches).

These tests verify, textually on the committed pristine fixture
(``tests/legacy/pristine_fixtures/single_type_kv_cache_manager.py``,
md5 93fe087f893767edb2049647ed335c20 — dev301 Site 1 body):
  1. Site 1 anchor matches pristine exactly once and carries the
     retention_interval signature + forward
  2. Site 2 pristine variant matches pristine exactly once; post-PN346
     variant matches zero times (mutual exclusion, direction 1)
  3. after applying PN346's replacement to a pristine copy textually,
     the post-PN346 variant matches exactly once and the pristine
     variant matches zero times (mutual exclusion, direction 2)
  4. chain convention: post-PN346 constants are derived from PN346's
     constants; the post-PN346 replacement carries the drop_eagle_block
     guard, the pristine replacement does not
  5. end-to-end TextPatcher apply on tmp copies (both shapes) —
     APPLIED with exactly the expected variant, result compiles
  6. site2_anchor_present pre-gate: True on both real shapes, False
     when Site 2 drifted away
  7. both apply orders compose: PN346-then-P85 (boot order) AND
     P85-then-PN346 (PN346's anchor survives inside P85's coarse
     fallback exactly once)
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
PRISTINE_FIXTURE = (
    REPO_ROOT
    / "tests"
    / "legacy"
    / "pristine_fixtures"
    / "single_type_kv_cache_manager.py"
)


def _p85():
    from sndr.engines.vllm.patches.kv_cache import (
        p85_hybrid_fine_shadow_prefix_cache as M,
    )
    return M


def _pn346():
    from sndr.engines.vllm.patches.kv_cache import (
        pn346_mamba_mtp_apc_boundary as M,
    )
    return M


def _pristine() -> str:
    if not PRISTINE_FIXTURE.is_file():
        pytest.skip(f"pristine fixture not found: {PRISTINE_FIXTURE}")
    return PRISTINE_FIXTURE.read_text(encoding="utf-8")


def _post_pn346() -> str:
    """Pristine source with PN346's replacement applied (textual)."""
    pn346 = _pn346()
    src = _pristine()
    assert src.count(pn346.PN346_ANCHOR_OLD) == 1, (
        "PN346 anchor no longer unique in the pristine fixture — fixture "
        "or PN346 drifted; P85 Site 2 composition must be re-verified"
    )
    return src.replace(pn346.PN346_ANCHOR_OLD, pn346.PN346_ANCHOR_NEW, 1)


# ─────────────────────────────────────────────────────────────────────
# 1. Site 1 — retention_interval re-anchor
# ─────────────────────────────────────────────────────────────────────


class TestSite1RetentionInterval:

    def test_site1_old_matches_pristine_exactly_once(self):
        M = _p85()
        assert _pristine().count(M.P85_SITE1_OLD) == 1

    def test_site1_old_carries_retention_interval_signature(self):
        M = _p85()
        assert "retention_interval: int | None = None,\n" in M.P85_SITE1_OLD
        assert (
            "super().cache_blocks(request, num_tokens, "
            "retention_interval=retention_interval)" in M.P85_SITE1_OLD
        )

    def test_site1_new_preserves_retention_interval_forward(self):
        """The replacement must keep the upstream signature AND the
        retention_interval forward intact — P85 only appends the shadow
        registration block."""
        M = _p85()
        assert "retention_interval: int | None = None,\n" in M.P85_SITE1_NEW
        assert (
            "super().cache_blocks(request, num_tokens, "
            "retention_interval=retention_interval)" in M.P85_SITE1_NEW
        )
        # Shadow block still present and unchanged in spirit.
        assert "[Genesis P85] Shadow fine-grained hash entries" in M.P85_SITE1_NEW
        assert "self.block_pool.cached_block_hash_to_block.insert(" in M.P85_SITE1_NEW

    def test_site1_new_starts_with_site1_old_body(self):
        """Append-only contract: NEW = OLD body + shadow block + the
        trailing new_step_starts def. Guards against accidental
        upstream-body drift inside the replacement."""
        M = _p85()
        tail = "\n    def new_step_starts(self) -> None:\n"
        assert M.P85_SITE1_OLD.endswith(tail)
        assert M.P85_SITE1_NEW.endswith(tail)
        old_body = M.P85_SITE1_OLD[: -len(tail)]
        assert M.P85_SITE1_NEW.startswith(old_body)


# ─────────────────────────────────────────────────────────────────────
# 2+3+4. Site 2 — dual anchor variants, mutual exclusion, chain build
# ─────────────────────────────────────────────────────────────────────


class TestSite2AnchorVariants:

    def test_pristine_variant_matches_pristine_exactly_once(self):
        M = _p85()
        assert _pristine().count(M.P85_SITE2_OLD) == 1

    def test_post_pn346_variant_absent_from_pristine(self):
        M = _p85()
        assert _pristine().count(M.P85_SITE2_OLD_POST_PN346) == 0

    def test_post_pn346_variant_matches_post_pn346_exactly_once(self):
        M = _p85()
        assert _post_pn346().count(M.P85_SITE2_OLD_POST_PN346) == 1

    def test_pristine_variant_absent_from_post_pn346(self):
        M = _p85()
        assert _post_pn346().count(M.P85_SITE2_OLD) == 0

    def test_post_pn346_old_built_from_pn346_constants(self):
        """Chain convention (PN32-imports-PN79 precedent): the post-PN346
        anchor must be P85_SITE2_OLD with PN346's OLD→NEW splice applied,
        derived from PN346's own constants."""
        M, pn346 = _p85(), _pn346()
        assert M.P85_SITE2_OLD.count(pn346.PN346_ANCHOR_OLD) == 1
        assert M.P85_SITE2_OLD_POST_PN346 == M.P85_SITE2_OLD.replace(
            pn346.PN346_ANCHOR_OLD, pn346.PN346_ANCHOR_NEW, 1
        )

    def test_post_pn346_new_built_from_pn346_constants(self):
        M, pn346 = _p85(), _pn346()
        assert M.P85_SITE2_NEW.count(pn346.PN346_ANCHOR_OLD) == 1
        assert M.P85_SITE2_NEW_POST_PN346 == M.P85_SITE2_NEW.replace(
            pn346.PN346_ANCHOR_OLD, pn346.PN346_ANCHOR_NEW, 1
        )

    def test_post_pn346_replacement_carries_drop_eagle_guard(self):
        """The coarse fallback of the post-PN346 replacement must carry
        PN346's drop_eagle_block boundary guard (verifier resolution
        (a): the fix must not undo PN346's accuracy fix)."""
        M = _p85()
        assert (
            "if drop_eagle_block and max_num_blocks > 0:\n"
            in M.P85_SITE2_NEW_POST_PN346
        )
        assert "max_num_blocks -= 1\n" in M.P85_SITE2_NEW_POST_PN346
        assert "[Genesis PN346" in M.P85_SITE2_NEW_POST_PN346

    def test_pristine_replacement_has_no_pn346_text(self):
        M = _p85()
        assert "[Genesis PN346" not in M.P85_SITE2_NEW
        assert "drop_eagle_block and max_num_blocks" not in M.P85_SITE2_NEW

    def test_variant_required_semantics(self):
        """Site 1 required=True; both Site 2 variants required=False
        (required-at-least-one — kernel soft-skips the non-matching
        variant)."""
        M = _p85()
        subs = {sp.name: sp for sp in M.build_sub_patches()}
        assert len(subs) == 3
        assert subs["p85_mamba_cache_blocks_shadow"].required is True
        assert (
            subs["p85_mamba_find_longest_cache_hit_fine_pristine"].required
            is False
        )
        assert (
            subs["p85_mamba_find_longest_cache_hit_fine_post_pn346"].required
            is False
        )


# ─────────────────────────────────────────────────────────────────────
# 5. End-to-end TextPatcher apply on tmp copies of both shapes
# ─────────────────────────────────────────────────────────────────────


def _apply_p85_to(tmp_path, content: str):
    M = _p85()
    from sndr.kernel import TextPatcher

    target = tmp_path / "single_type_kv_cache_manager.py"
    target.write_text(content, encoding="utf-8")
    patcher = TextPatcher(
        patch_name="p85-dual-anchor-e2e-probe",
        target_file=str(target),
        marker=M.GENESIS_P85_MARKER,
        sub_patches=M.build_sub_patches(),
    )
    result, failure = patcher.apply()
    return result, failure, patcher, target


class TestEndToEndApply:

    def test_applies_on_pristine_via_pristine_variant(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        from sndr.kernel import TextPatchResult

        result, failure, patcher, target = _apply_p85_to(tmp_path, _pristine())
        assert result == TextPatchResult.APPLIED, failure
        assert patcher.applied_sub_patches == [
            "p85_mamba_cache_blocks_shadow",
            "p85_mamba_find_longest_cache_hit_fine_pristine",
        ]
        compile(target.read_text(encoding="utf-8"), str(target), "exec")

    def test_applies_on_post_pn346_via_post_variant(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        from sndr.kernel import TextPatchResult

        result, failure, patcher, target = _apply_p85_to(tmp_path, _post_pn346())
        assert result == TextPatchResult.APPLIED, failure
        assert patcher.applied_sub_patches == [
            "p85_mamba_cache_blocks_shadow",
            "p85_mamba_find_longest_cache_hit_fine_post_pn346",
        ]
        compile(target.read_text(encoding="utf-8"), str(target), "exec")

    def test_idempotent_on_second_apply(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        M = _p85()
        from sndr.kernel import TextPatcher, TextPatchResult

        result, _, _, target = _apply_p85_to(tmp_path, _pristine())
        assert result == TextPatchResult.APPLIED
        second = TextPatcher(
            patch_name="p85-dual-anchor-idempotency-probe",
            target_file=str(target),
            marker=M.GENESIS_P85_MARKER,
            sub_patches=M.build_sub_patches(),
        )
        result2, _ = second.apply()
        assert result2 == TextPatchResult.IDEMPOTENT


# ─────────────────────────────────────────────────────────────────────
# 6. Site 2 pre-gate (required-at-least-one belt for the wrapper)
# ─────────────────────────────────────────────────────────────────────


class TestSite2PreGate:

    def test_present_on_pristine(self):
        M = _p85()
        assert M.site2_anchor_present(_pristine()) is True

    def test_present_on_post_pn346(self):
        M = _p85()
        assert M.site2_anchor_present(_post_pn346()) is True

    def test_absent_when_site2_drifted(self):
        """Site 1 intact but Site 2 gone in both shapes → pre-gate False
        (apply() must skip BEFORE writing; Site-1-only half-apply would
        be a store-side no-op that hides the drift behind the marker)."""
        M = _p85()
        drifted = _pristine().replace(M.P85_SITE2_OLD, "", 1)
        assert M.P85_SITE1_OLD in drifted  # Site 1 still present
        assert M.site2_anchor_present(drifted) is False


# ─────────────────────────────────────────────────────────────────────
# 7. Apply-order composition (PN346 boot-dispatches BEFORE P85)
# ─────────────────────────────────────────────────────────────────────


class TestOrderingComposition:

    def test_pn346_anchor_survives_p85_pristine_apply_exactly_once(
        self, tmp_path, monkeypatch
    ):
        """Reverse order (P85 first, PN346 disabled-then-enabled later)
        also composes: P85's pristine-variant replacement re-emits the
        4-line coarse fallback, so PN346's anchor still matches exactly
        once inside it."""
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        pn346 = _pn346()
        from sndr.kernel import TextPatchResult

        result, _, _, target = _apply_p85_to(tmp_path, _pristine())
        assert result == TextPatchResult.APPLIED
        post_p85 = target.read_text(encoding="utf-8")
        assert post_p85.count(pn346.PN346_ANCHOR_OLD) == 1

    def test_boot_order_pn346_then_p85(self, tmp_path, monkeypatch):
        """The real boot order: PN346 applied first (textually), then
        P85 — post-PN346 variant must fire and the result compiles."""
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        from sndr.kernel import TextPatchResult

        result, failure, patcher, target = _apply_p85_to(tmp_path, _post_pn346())
        assert result == TextPatchResult.APPLIED, failure
        assert (
            "p85_mamba_find_longest_cache_hit_fine_post_pn346"
            in patcher.applied_sub_patches
        )
        compile(target.read_text(encoding="utf-8"), str(target), "exec")

    def test_module_documents_pn346_composition(self):
        import inspect

        M = _p85()
        src = inspect.getsource(M)
        assert "PN346" in src
