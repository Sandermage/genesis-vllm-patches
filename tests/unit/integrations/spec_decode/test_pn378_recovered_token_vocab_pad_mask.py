# SPDX-License-Identifier: Apache-2.0
"""PN378 — recovered-token vocab-pad -inf mask (vendor of vllm#45060, kernel half).

Contract pinned here (TDD, written before the implementation):
  1. Patcher carries ONE required sub-patch: the vocab-padding -inf
     mask inserted between the score product and the tl.max reduction
     of ``sample_recovered_tokens_kernel``.
  2. KERNEL HALF ONLY: the module never touches scheduler.py — the
     scheduler half of #45060 (the ``assert generated_token_ids``)
     is deliberately NOT vendored; PN133 v2 carries a log.error on
     that condition instead (roadmap chunk-3 Theme A).
  3. Deliberate spelling divergence for drift-marker hygiene: our mask
     uses ``float("-inf")`` while upstream #45060 writes
     ``-float("inf")`` — semantically identical constants, so the PR's
     exact structural line stays usable as a drift marker without
     colliding with our own emitted text (lint_drift_markers contract).
  4. apply() on the pin-form (g303916e93) kernel installs the mask
     between ``score = prob * inv_q`` and the ``tl.max`` reduction,
     keeps the in-vocab init (``recovered_id = 0``), and the result
     still compiles.
  5. Second apply() is idempotent (marker short-circuit).
  6. apply() on #45060's merged form self-skips via drift markers
     (reason: upstream_merged) without touching the file.
  7. Drift markers do not collide with PN378's own replacement text or
     its Layer-6 marker line (tools/lint_drift_markers.py contract)
     AND at least one marker is an exact substring of the merged form.
  8. Anchors are unique and drift markers absent in the pristine pin
     tree (opportunistic — skipped when the pin tree is not present).
  9. The module documents the live exposure (Qwen vocab 151936 %
     BLOCK_SIZE 8192 != 0), the PN133 coordination, and references the
     registry env flag GENESIS_ENABLE_PN378_VOCAB_PAD_MASK.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

# The lint/preflight tools disable the Layer-0 file cache the same way;
# unit tests patch fresh tmp files, so the cache must never satisfy
# apply() from a previous run's state.
os.environ.setdefault("GENESIS_NO_PATCH_CACHE", "1")

from sndr.engines.vllm.patches.spec_decode import (  # noqa: E402
    pn378_recovered_token_vocab_pad_mask as m,
)

# ── Fake targets ─────────────────────────────────────────────────────
# Pin-form (g303916e93): the recovered-token kernel tail in
# v1/sample/rejection_sampler.py — byte-faithful copy of the anchor
# region (lines 860-932 of the pristine pin file).

PIN_SAMPLER = (
    "# fake v1/sample/rejection_sampler.py (pin g303916e93 form)\n"
    "@triton.jit\n"
    "def sample_recovered_tokens_kernel(\n"
    "    output_token_ids_ptr,  # [num_tokens]\n"
    "    cu_num_draft_tokens_ptr,  # [batch_size]\n"
    "    draft_token_ids_ptr,  # [num_tokens]\n"
    "    draft_probs_ptr,  # [num_tokens, vocab_size] or None\n"
    "    target_probs_ptr,  # [num_tokens, vocab_size]\n"
    "    inv_q_ptr,  # [batch_size, vocab_size]\n"
    "    vocab_size,\n"
    "    BLOCK_SIZE: tl.constexpr,\n"
    "    NO_DRAFT_PROBS: tl.constexpr,\n"
    "    USE_FP64_GUMBEL: tl.constexpr,\n"
    "):\n"
    "    req_idx = tl.program_id(0)\n"
    "    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr"
    " + req_idx - 1)\n"
    "    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)\n"
    "    num_draft_tokens = end_idx - start_idx\n"
    "\n"
    "    # Early exit for out-of-range positions.\n"
    "    pos = tl.program_id(1)\n"
    "    if pos >= num_draft_tokens:\n"
    "        return\n"
    "\n"
    "    token_idx = start_idx + pos\n"
    "\n"
    "    if NO_DRAFT_PROBS:\n"
    "        draft_token_id = tl.load(draft_token_ids_ptr + token_idx)\n"
    "\n"
    "    if USE_FP64_GUMBEL:\n"
    '        max_val = tl.full((), float("-inf"), tl.float64)\n'
    "    else:\n"
    '        max_val = tl.full((), float("-inf"), tl.float32)\n'
    "    recovered_id = 0\n"
    "    for v in range(0, vocab_size, BLOCK_SIZE):\n"
    "        vocab_offset = v + tl.arange(0, BLOCK_SIZE)\n"
    "        vocab_mask = vocab_offset < vocab_size\n"
    "\n"
    "        if NO_DRAFT_PROBS:\n"
    "            prob = tl.load(\n"
    "                target_probs_ptr + token_idx * vocab_size + vocab_offset,\n"
    "                mask=(vocab_mask & (vocab_offset != draft_token_id)),\n"
    "                other=0.0,\n"
    "            )\n"
    "        else:\n"
    "            draft_prob = tl.load(\n"
    "                draft_probs_ptr + token_idx * vocab_size + vocab_offset,\n"
    "                mask=vocab_mask,\n"
    "                other=0.0,\n"
    "            )\n"
    "            target_prob = tl.load(\n"
    "                target_probs_ptr + token_idx * vocab_size + vocab_offset,\n"
    "                mask=vocab_mask,\n"
    "                other=0.0,\n"
    "            )\n"
    "            prob = tl.maximum(target_prob - draft_prob, 0.0)\n"
    "            # NOTE(woosuk): We don't need `prob = prob / tl.sum(prob)`"
    " here because\n"
    "            # `tl.argmax` will select the maximum value.\n"
    "\n"
    "        inv_q = tl.load(\n"
    "            inv_q_ptr + req_idx * vocab_size + vocab_offset,\n"
    "            mask=vocab_mask,\n"
    "            other=0.0,\n"
    "        )\n"
    "\n"
    "        # Local tile reduction\n"
    "        score = prob * inv_q\n"
    "        local_max, local_id = tl.max(score, axis=0, return_indices=True)\n"
    "\n"
    "        if local_max > max_val:\n"
    "            max_val = local_max\n"
    "            recovered_id = v + local_id\n"
    "\n"
    "    tl.store(output_token_ids_ptr + token_idx, recovered_id)\n"
)

# #45060 merged form (what rejection_sampler.py looks like AFTER the
# upstream PR lands its kernel hunk) — PN378 must self-skip on this.
# Exact hunk text from `gh pr diff 45060` (2026-06-11): the comment is
# extended in place and the tl.where mask line is added after the
# score product. Upstream spells the constant `-float("inf")`.

MERGED_SAMPLER = PIN_SAMPLER.replace(
    "        # Local tile reduction\n"
    "        score = prob * inv_q\n"
    "        local_max, local_id = tl.max(score, axis=0, return_indices=True)\n",
    "        # Local tile reduction. Mask padding (``vocab_offset >= "
    "vocab_size``,\n"
    "        # score ``0.0`` via ``other=0.0``) to ``-inf``. Otherwise, "
    "with all-NaN\n"
    "        # ``target_probs`` those zeros win the NaN-propagating "
    "``tl.max`` and\n"
    "        # yield an out-of-vocab ``recovered_id == vocab_size``. "
    "Masking keeps\n"
    "        # ``recovered_id`` at its in-vocab init (0); healthy runs "
    "are unaffected\n"
    "        # since real scores are ``>= 0 > -inf``.\n"
    "        score = prob * inv_q\n"
    '        score = tl.where(vocab_mask, score, -float("inf"))\n'
    "        local_max, local_id = tl.max(score, axis=0, return_indices=True)\n",
).replace(
    "(pin g303916e93 form)", "(post-vllm#45060 merged form)"
)

PIN_TREE = Path("/private/tmp/candidate_pin_current/vllm/v1/sample")

OUR_MASK_LINE = 'score = tl.where(vocab_mask, score, float("-inf"))'
UPSTREAM_MASK_LINE = 'score = tl.where(vocab_mask, score, -float("inf"))'


# ── Helpers ──────────────────────────────────────────────────────────


def _install_fake(tmp_path, monkeypatch, sampler_text):
    target = tmp_path / "rejection_sampler.py"
    target.write_text(sampler_text, encoding="utf-8")
    monkeypatch.setattr(m, "resolve_vllm_file", lambda rel: str(target))
    # apply() is dispatcher-gated (opt-in env flag, registry-driven) —
    # force the gate open for unit tests of the patch mechanics.
    import sndr.dispatcher as dispatcher
    monkeypatch.setattr(
        dispatcher, "should_apply", lambda pid: (True, "test override")
    )
    return target


# ── Patcher shape ────────────────────────────────────────────────────


class TestPatcherShape:
    def test_patcher_has_single_required_mask_sub(self, tmp_path, monkeypatch):
        _install_fake(tmp_path, monkeypatch, PIN_SAMPLER)
        patcher = m._make_patcher()
        assert patcher is not None
        by_name = {sp.name: sp for sp in patcher.sub_patches}
        assert set(by_name) == {"pn378_vocab_pad_mask"}
        assert by_name["pn378_vocab_pad_mask"].required is True

    def test_mask_spelling_diverges_from_upstream(self):
        """Drift-marker hygiene divergence: our mask spells the constant
        ``float("-inf")`` (upstream writes ``-float("inf")``) so the
        PR's exact structural line never appears in our own output."""
        assert OUR_MASK_LINE in m.PN378_MASK_NEW
        assert UPSTREAM_MASK_LINE not in m.PN378_MASK_NEW

    def test_kernel_half_only_no_scheduler_text(self):
        """#45060's scheduler half is NOT vendored — PN133 v2 owns that
        site (log.error, not assert). The module must not carry the
        assert or target scheduler.py."""
        src = Path(m.__file__).read_text(encoding="utf-8")
        assert "assert generated_token_ids" not in m.PN378_MASK_NEW
        assert m._TARGET_REL == "v1/sample/rejection_sampler.py"
        assert "core/sched/scheduler.py" not in src

    def test_patcher_none_when_target_missing(self, monkeypatch):
        monkeypatch.setattr(m, "resolve_vllm_file", lambda rel: None)
        assert m._make_patcher() is None

    def test_module_documents_exposure_and_pn133_coordination(self):
        """Docstring must carry: the upstream PR, the live exposure
        (Qwen vocab 151936 vs BLOCK_SIZE 8192), and the PN133
        coordination (PN133 repairs accounting; PN378 removes the
        out-of-vocab source)."""
        doc = m.__doc__ or ""
        assert "45060" in doc
        assert "151936" in doc
        assert "8192" in doc
        assert "PN133" in doc

    def test_module_references_registry_env_flag(self):
        src = Path(m.__file__).read_text(encoding="utf-8")
        assert "GENESIS_ENABLE_PN378_VOCAB_PAD_MASK" in src


# ── Apply semantics ──────────────────────────────────────────────────


class TestApply:
    def test_apply_pin_form_installs_mask(self, tmp_path, monkeypatch):
        target = _install_fake(tmp_path, monkeypatch, PIN_SAMPLER)
        status, reason = m.apply()
        assert status == "applied", reason

        out = target.read_text(encoding="utf-8")
        # Mask line installed exactly once, our spelling only.
        assert out.count(OUR_MASK_LINE) == 1
        assert UPSTREAM_MASK_LINE not in out
        # Ordering: score product -> mask -> reduction.
        assert (
            out.index("score = prob * inv_q")
            < out.index(OUR_MASK_LINE)
            < out.index("local_max, local_id = tl.max(")
        )
        # The in-vocab init the mask relies on is untouched.
        assert "    recovered_id = 0\n" in out
        # File still compiles after the splice.
        compile(out, str(target), "exec")

    def test_second_apply_is_idempotent(self, tmp_path, monkeypatch):
        _install_fake(tmp_path, monkeypatch, PIN_SAMPLER)
        first_status, first_reason = m.apply()
        assert first_status == "applied", first_reason
        # Canonical result_to_wiring_status contract: IDEMPOTENT maps to
        # ("skipped", "...: already applied (marker present)").
        second_status, second_reason = m.apply()
        assert second_status == "skipped"
        assert "already applied" in second_reason

    def test_self_skips_on_45060_merged_form(self, tmp_path, monkeypatch):
        target = _install_fake(tmp_path, monkeypatch, MERGED_SAMPLER)
        status, reason = m.apply()
        assert status == "skipped"
        assert "upstream_merged" in reason
        # Self-skip must not modify the merged file.
        assert target.read_text(encoding="utf-8") == MERGED_SAMPLER

    def test_apply_skips_when_gate_closed(self, tmp_path, monkeypatch):
        """Opt-in patch (default_on=False): with the dispatcher gate
        closed (env flag unset / registry says no), apply() must skip
        without touching the target."""
        target = tmp_path / "rejection_sampler.py"
        target.write_text(PIN_SAMPLER, encoding="utf-8")
        monkeypatch.setattr(m, "resolve_vllm_file", lambda rel: str(target))
        import sndr.dispatcher as dispatcher
        monkeypatch.setattr(
            dispatcher, "should_apply", lambda pid: (False, "opt-in: env unset")
        )
        monkeypatch.delenv("GENESIS_ENABLE_PN378_VOCAB_PAD_MASK", raising=False)
        status, _reason = m.apply()
        assert status == "skipped"
        assert target.read_text(encoding="utf-8") == PIN_SAMPLER

    def test_apply_skips_when_target_missing(self, monkeypatch):
        monkeypatch.setattr(m, "resolve_vllm_file", lambda rel: None)
        import sndr.dispatcher as dispatcher
        monkeypatch.setattr(
            dispatcher, "should_apply", lambda pid: (True, "test override")
        )
        status, _reason = m.apply()
        assert status == "skipped"


# ── Lint contract (tools/lint_drift_markers.py) ──────────────────────


class TestDriftMarkerSelfCollision:
    def test_markers_not_substring_of_own_emitted_text(
        self, tmp_path, monkeypatch
    ):
        _install_fake(tmp_path, monkeypatch, PIN_SAMPLER)
        patcher = m._make_patcher()
        marker_line = f"# [Genesis wiring marker: {patcher.marker}]\n"
        assert patcher.upstream_drift_markers, "drift markers must exist"
        for dm in patcher.upstream_drift_markers:
            if dm.startswith("[Genesis"):
                continue  # defended convention — exempt
            for sp in patcher.sub_patches:
                assert dm not in sp.replacement, (
                    f"drift marker {dm!r} collides with {sp.name} "
                    "replacement — would false-fire Layer 3 (PN369 class)"
                )
            assert dm not in marker_line

    def test_markers_match_45060_merged_form(self, tmp_path, monkeypatch):
        """Markers must actually fire on the real merged form —
        the PN367-v1 regression was markers that could never match."""
        _install_fake(tmp_path, monkeypatch, PIN_SAMPLER)
        patcher = m._make_patcher()
        assert any(
            dm in MERGED_SAMPLER for dm in patcher.upstream_drift_markers
        )


# ── Pristine pin invariants (opportunistic) ──────────────────────────


@pytest.mark.skipif(
    not (PIN_TREE / "rejection_sampler.py").is_file(),
    reason="pristine pin tree not present on this machine",
)
class TestAnchorsAgainstPristinePin:
    def test_anchor_unique_replacement_and_markers_absent(self):
        src = (PIN_TREE / "rejection_sampler.py").read_text(encoding="utf-8")
        assert src.count(m.PN378_MASK_OLD) == 1
        assert m.PN378_MASK_NEW not in src
        assert OUR_MASK_LINE not in src
        for dm in m._DRIFT_MARKERS:
            assert dm not in src

    def test_fixture_anchor_region_byte_matches_pristine(self):
        """The fake pin-form fixture must carry the EXACT anchor bytes
        from the pristine tree, so apply-tests exercise real anchors."""
        src = (PIN_TREE / "rejection_sampler.py").read_text(encoding="utf-8")
        assert m.PN378_MASK_OLD in PIN_SAMPLER
        assert m.PN378_MASK_OLD in src

    def test_live_exposure_block_size_8192_in_pin_wrapper(self):
        """The wrapper hardcodes BLOCK_SIZE = 8192; Qwen vocab 151936 %
        8192 == 4480 != 0, so the final tile carries padding lanes and
        the #45060 bug class is LIVE on our stack. If upstream changes
        the block size, re-derive the exposure analysis."""
        src = (PIN_TREE / "rejection_sampler.py").read_text(encoding="utf-8")
        assert "BLOCK_SIZE = 8192" in src
        assert 151936 % 8192 != 0
