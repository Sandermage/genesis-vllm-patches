# SPDX-License-Identifier: Apache-2.0
"""PN388 — mamba-block-aligned intermediate prefill split (vendor of vllm#45477).

Contract pinned here (TDD, written before the implementation):

  1. Patcher carries the dual-anchor variant set for the one rewrite site
     (``Scheduler._mamba_block_aligned_split`` in v1/core/sched/scheduler.py):
       * a PRISTINE-shaped variant (pin g303916e93, P34 disabled), and
       * a POST-P34-shaped variant (the same block after the effectively
         always-on P34 zero-collapse guard has expanded the first branch).
     Both are ``required=False`` with required-at-least-one semantics
     (mirrors the P85-on-PN346 dual-anchor convention); exactly one
     matches any given file by construction.

  2. The replacement is the PR #45477 flat form (round_down on the chunk
     END, not the chunk LENGTH) WITHOUT the PR's Marconi
     ``num_uncached_common_prefix_tokens`` tail — that parameter does not
     exist in our pin's function signature (documented Genesis divergence,
     iron rule #10).

  3. The fix removes the LIVE prefix-cache poison: every NON-FINAL prefill
     chunk ends on a mamba block boundary; a budget-fragmented first chunk
     defers (``num_new_tokens == 0``) instead of ending mid-block. The
     final unaligned prompt tail is still allowed.

  4. apply() on either pin form (pristine or post-P34) installs the rewrite,
     keeps the trailing ``return num_new_tokens``, and the file still
     compiles. Second apply() is idempotent (marker short-circuit).

  5. apply() on #45477's merged form self-skips via drift markers
     (reason: upstream_merged) without touching the file.

  6. Drift markers do not collide with PN388's own replacement text or its
     Layer-6 marker line (tools/lint_drift_markers.py self-collision
     contract) AND at least one marker is an exact substring of the merged
     form.

  7. Coexistence with P34: PN388 anchors on the POST-P34 shape (P34
     boot-dispatches first via requires_patches), and the module references
     the registry env flag GENESIS_ENABLE_PN388_MAMBA_BLOCK_ALIGNED_SPLIT.

  8. Behavioural oracle: a pure-Python re-implementation of the patched
     arithmetic, run through a fragmenting-budget prefill loop on the exact
     PROD shape (prompt 2002, mamba block 1600, eagle on), produces only
     block-aligned non-final chunk ends — while the unpatched pin+P34 form
     produces unaligned non-final ends (the poison).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

# Unit tests patch fresh tmp files; the Layer-0 file cache must never
# satisfy apply() from a previous run's state (same as the lint tools).
os.environ.setdefault("GENESIS_NO_PATCH_CACHE", "1")

from sndr.engines.vllm.patches.scheduler import (  # noqa: E402
    pn388_mamba_block_aligned_prefill_split as m,
)

# ── Fake targets ─────────────────────────────────────────────────────
# Pristine pin-form (g303916e93): the _mamba_block_aligned_split body —
# byte-faithful copy of the anchor region (pin lines 293-338).

PIN_SCHEDULER = (
    "# fake v1/core/sched/scheduler.py (pin g303916e93 form)\n"
    "class Scheduler:\n"
    "    def _mamba_block_aligned_split(\n"
    "        self,\n"
    "        request,\n"
    "        num_new_tokens,\n"
    "        num_new_local_computed_tokens=0,\n"
    "        num_external_computed_tokens=0,\n"
    "    ):\n"
    "        num_computed_tokens = (\n"
    "            request.num_computed_tokens\n"
    "            + num_new_local_computed_tokens\n"
    "            + num_external_computed_tokens\n"
    "        )\n"
    "        # Perform block-aligned splitting at prefill phase, including:\n"
    "        # * non-resumed requests: num_computed_tokens < num_prompt_tokens + 0\n"
    "        # * resumed requests: num_computed_tokens < (\n"
    "        #                       num_prompt_tokens + num_output_tokens\n"
    "        #                     )\n"
    "        # NOTE: Use `request.num_tokens - 1` to bypass normal decoding.\n"
    "        if num_computed_tokens < max(request.num_prompt_tokens, request.num_tokens - 1):\n"
    "            # To enable block-aligned caching of the Mamba state, `num_new_tokens`\n"
    "            # must be a multiple of `block_size`.\n"
    "            # As an exception, if `num_new_tokens` is less than `block_size`, the\n"
    "            # state is simply not cached, requiring no special handling.\n"
    "            # Additionally, when Eagle mode is enabled, FullAttn prunes the last\n"
    "            # matching block. To prevent this from causing a Mamba cache miss, the\n"
    "            # last chunk must be not smaller than `block_size`.\n"
    "            block_size = self.cache_config.block_size\n"
    "            last_cache_position = request.num_tokens - request.num_tokens % block_size\n"
    "            # eagle prune\n"
    "            if self.use_eagle:\n"
    "                last_cache_position = max(last_cache_position - block_size, 0)\n"
    "            num_computed_tokens_after_sched = num_computed_tokens + num_new_tokens\n"
    "            if num_computed_tokens_after_sched < last_cache_position:\n"
    "                # align to block_size\n"
    "                num_new_tokens = num_new_tokens // block_size * block_size\n"
    "            elif (\n"
    "                num_computed_tokens\n"
    "                < last_cache_position\n"
    "                < num_computed_tokens_after_sched\n"
    "            ):\n"
    "                # force to cache the last chunk\n"
    "                num_new_tokens = last_cache_position - num_computed_tokens\n"
    "            else:\n"
    "                # prefill the last few tokens\n"
    "                pass\n"
    "        return num_new_tokens\n"
)

# Post-P34 form: identical except P34's zero-collapse guard has expanded
# the first (``num_computed_tokens_after_sched < last_cache_position``)
# branch. This is what the file looks like on every real hybrid boot,
# because P34 is effectively always-on and dispatches before PN388.
_P34_OLD = (
    "            if num_computed_tokens_after_sched < last_cache_position:\n"
    "                # align to block_size\n"
    "                num_new_tokens = num_new_tokens // block_size * block_size"
)
_P34_NEW = (
    "            if num_computed_tokens_after_sched < last_cache_position:\n"
    "                # align to block_size\n"
    "                # [Genesis P34] Zero-collapse deadlock guard (upstream PR #40757).\n"
    "                # When two adjacent multimodal inputs can't fit in the encoder\n"
    "                # cache simultaneously, the gap can be < block_size; aligning\n"
    "                # down then collapses to 0 and the scheduler spins forever.\n"
    "                # Keep the sub-block value when alignment would zero-out —\n"
    "                # Mamba state is still maintained by preprocess_mamba via\n"
    "                # mamba_state_idx (\"simply not cached\" exception applies).\n"
    "                aligned = num_new_tokens // block_size * block_size\n"
    "                if aligned > 0:\n"
    "                    num_new_tokens = aligned"
)
POST_P34_SCHEDULER = PIN_SCHEDULER.replace(_P34_OLD, _P34_NEW).replace(
    "(pin g303916e93 form)", "(post-P34 form)"
)

# #45477 merged form (what scheduler.py looks like AFTER the upstream PR
# lands its scheduler hunk) — PN388 must self-skip on this. The structural
# line below is taken verbatim from `gh pr diff 45477`.
MERGED_SCHEDULER = (
    "# fake v1/core/sched/scheduler.py (post-vllm#45477 merged form)\n"
    "class Scheduler:\n"
    "    def _mamba_block_aligned_split(self, request, num_new_tokens):\n"
    "        prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)\n"
    "        if num_computed_tokens >= prefill_end:\n"
    "            return num_new_tokens\n"
    "        block_size = self.cache_config.block_size\n"
    "        last_cache_position = round_down(request.num_tokens, block_size)\n"
    "        if self.use_eagle:\n"
    "            last_cache_position = max(last_cache_position - block_size, 0)\n"
    "        chunk_end = num_computed_tokens + num_new_tokens\n"
    "        if num_computed_tokens < last_cache_position:\n"
    "            chunk_end = min(round_down(chunk_end, block_size), last_cache_position)\n"
    "        elif chunk_end < prefill_end:\n"
    "            chunk_end = round_down(chunk_end, block_size)\n"
    "        num_new_tokens = max(chunk_end - num_computed_tokens, 0)\n"
    "        return num_new_tokens\n"
)

PIN_TREE = Path("/private/tmp/candidate_pin_current/vllm/v1/core/sched")

MAMBA_BLOCK = 1600
PROMPT_LEN = 2002


# ── Helpers ──────────────────────────────────────────────────────────


def _install_fake(tmp_path, monkeypatch, scheduler_text):
    target = tmp_path / "scheduler.py"
    target.write_text(scheduler_text, encoding="utf-8")
    monkeypatch.setattr(m, "resolve_vllm_file", lambda rel: str(target))
    # apply() is dispatcher-gated (opt-in env flag) — force the gate open
    # for unit tests of the patch mechanics.
    import sndr.dispatcher as dispatcher

    monkeypatch.setattr(
        dispatcher, "should_apply", lambda pid: (True, "test override")
    )
    return target


# Pure-Python oracle of the PATCHED arithmetic (mirrors the replacement
# the patch installs). Used to prove the alignment invariant end-to-end.
def _patched_split(num_computed, num_new, num_prompt, num_tokens, use_eagle, bs):
    prefill_end = max(num_prompt, num_tokens - 1)
    if num_computed >= prefill_end:
        return num_new
    last = (num_tokens // bs) * bs
    if use_eagle:
        last = max(last - bs, 0)
    chunk_end = num_computed + num_new
    if num_computed < last:
        chunk_end = min((chunk_end // bs) * bs, last)
    elif chunk_end < prefill_end:
        chunk_end = (chunk_end // bs) * bs
    return max(chunk_end - num_computed, 0)


# Pure-Python oracle of the UNPATCHED pin+P34 arithmetic (the poison).
def _pin_p34_split(num_computed, num_new, num_prompt, num_tokens, use_eagle, bs):
    if num_computed < max(num_prompt, num_tokens - 1):
        last = num_tokens - num_tokens % bs
        if use_eagle:
            last = max(last - bs, 0)
        after = num_computed + num_new
        if after < last:
            aligned = num_new // bs * bs
            if aligned > 0:
                num_new = aligned
        elif num_computed < last < after:
            num_new = last - num_computed
    return num_new


def _run_prefill(split, prompt_len, budgets, use_eagle, bs):
    computed, ends, step = 0, [], 0
    while computed < prompt_len and step < 10000:
        nn = split(
            computed, min(prompt_len - computed, budgets[step % len(budgets)]),
            prompt_len, prompt_len, use_eagle, bs,
        )
        step += 1
        if nn == 0:
            continue  # scheduler defers num_new_tokens == 0 (handled upstream)
        computed += nn
        ends.append(computed)
    return ends


# ── Patcher shape ────────────────────────────────────────────────────


class TestPatcherShape:
    def test_patcher_has_dual_required_at_least_one_variants(
        self, tmp_path, monkeypatch
    ):
        _install_fake(tmp_path, monkeypatch, POST_P34_SCHEDULER)
        patcher = m._make_patcher()
        assert patcher is not None
        by_name = {sp.name: sp for sp in patcher.sub_patches}
        assert set(by_name) == {
            "pn388_split_pristine",
            "pn388_split_post_p34",
        }
        # Required-at-least-one: both variants are soft (required=False) so
        # the kernel cannot abort when the file is in the other shape.
        assert by_name["pn388_split_pristine"].required is False
        assert by_name["pn388_split_post_p34"].required is False

    def test_variants_are_mutually_exclusive_by_construction(self):
        """The post-P34 anchor carries P34's `[Genesis P34` comment lines
        and breaks the contiguous pristine first-branch run, so exactly one
        anchor matches any given file."""
        assert m.PN388_PRISTINE_OLD in PIN_SCHEDULER
        assert m.PN388_PRISTINE_OLD not in POST_P34_SCHEDULER
        assert m.PN388_POST_P34_OLD in POST_P34_SCHEDULER
        assert m.PN388_POST_P34_OLD not in PIN_SCHEDULER

    def test_replacement_omits_marconi_param(self):
        """Genesis divergence (iron rule #10): the PR's Marconi
        `num_uncached_common_prefix_tokens` tail is dropped — that
        parameter is absent from our pin's function signature."""
        assert "num_uncached_common_prefix_tokens" not in m.PN388_PRISTINE_NEW
        assert "num_uncached_common_prefix_tokens" not in m.PN388_POST_P34_NEW

    def test_replacement_rounds_chunk_end_not_length(self):
        """The fix rounds the chunk END, not the chunk LENGTH (the root
        cause). Both variants must install the chunk_end arithmetic."""
        for repl in (m.PN388_PRISTINE_NEW, m.PN388_POST_P34_NEW):
            assert "chunk_end = num_computed_tokens + num_new_tokens" in repl
            assert "num_new_tokens = max(chunk_end - num_computed_tokens, 0)" in repl

    def test_patcher_none_when_target_missing(self, monkeypatch):
        monkeypatch.setattr(m, "resolve_vllm_file", lambda rel: None)
        assert m._make_patcher() is None

    def test_module_documents_exposure_pn346_p85_and_async(self):
        """Docstring must carry: the upstream PR, the PROD exposure
        (Qwen3.6 GDN+Mamba + MTP K=3 + APC, mamba block, prompt 2002), the
        PN346 + P85 composition, P34 coexistence, and the async-ON A/B
        caveat (the PR validated --no-async-scheduling)."""
        doc = m.__doc__ or ""
        assert "45477" in doc
        assert "PN346" in doc
        assert "P85" in doc
        assert "P34" in doc
        assert "async" in doc.lower()
        assert "1600" in doc  # mamba block size at the PROD shape

    def test_module_references_registry_env_flag(self):
        src = Path(m.__file__).read_text(encoding="utf-8")
        assert "GENESIS_ENABLE_PN388_MAMBA_BLOCK_ALIGNED_SPLIT" in src


# ── Apply semantics ──────────────────────────────────────────────────


class TestApply:
    @pytest.mark.parametrize(
        "fixture", ["pristine", "post_p34"]
    )
    def test_apply_installs_rewrite(self, tmp_path, monkeypatch, fixture):
        text = PIN_SCHEDULER if fixture == "pristine" else POST_P34_SCHEDULER
        target = _install_fake(tmp_path, monkeypatch, text)
        status, reason = m.apply()
        assert status == "applied", reason

        out = target.read_text(encoding="utf-8")
        # The flat rewrite arithmetic is installed exactly once.
        assert out.count("chunk_end = num_computed_tokens + num_new_tokens") == 1
        # The trailing return is preserved.
        assert "        return num_new_tokens\n" in out
        # The old four-way branch is gone.
        assert "num_computed_tokens_after_sched" not in out
        # The marker banner is present.
        assert m.GENESIS_PN388_MARKER in out
        # File still compiles after the splice.
        compile(out, str(target), "exec")

    def test_second_apply_is_idempotent(self, tmp_path, monkeypatch):
        _install_fake(tmp_path, monkeypatch, POST_P34_SCHEDULER)
        first_status, first_reason = m.apply()
        assert first_status == "applied", first_reason
        second_status, second_reason = m.apply()
        assert second_status == "skipped"
        assert "already applied" in second_reason

    def test_self_skips_on_45477_merged_form(self, tmp_path, monkeypatch):
        target = _install_fake(tmp_path, monkeypatch, MERGED_SCHEDULER)
        status, reason = m.apply()
        assert status == "skipped"
        assert "upstream_merged" in reason
        assert target.read_text(encoding="utf-8") == MERGED_SCHEDULER

    def test_apply_skips_when_no_variant_matches(self, tmp_path, monkeypatch):
        """Required-at-least-one belt: if neither anchor variant matches
        (drifted file), apply() skips without writing.

        The 2026-06-17 redesign NARROWED both anchors to the inner four-way
        branch (``num_computed_tokens_after_sched = ...`` through
        ``else: pass``), so a drift on the OUTER ``if num_computed_tokens <
        max(...)`` guard no longer breaks the anchor — a variant would still
        match and apply() would (correctly) apply. To exercise the belt the
        drift must mutate the ACTUAL anchor text. We perturb the inner
        branch's first line, which is the shared opening line of BOTH
        ``PN388_PRISTINE_OLD`` and ``PN388_POST_P34_OLD``, so neither variant
        matches. The line stays valid Python, so this isolates anchor drift
        (not a syntax/compile failure)."""
        anchor_line = (
            "            num_computed_tokens_after_sched = "
            "num_computed_tokens + num_new_tokens\n"
        )
        # Sanity: the line we mutate really is the opening line of both
        # narrowed anchor variants — otherwise the drift would be stale again.
        assert anchor_line in m.PN388_PRISTINE_OLD
        assert anchor_line in m.PN388_POST_P34_OLD
        assert anchor_line in PIN_SCHEDULER
        drifted = PIN_SCHEDULER.replace(
            anchor_line,
            "            num_computed_tokens_after_sched = "
            "num_computed_tokens + num_new_tokens + 0  # drift\n",
        )
        # The mutated anchor breaks BOTH variants — neither shape matches.
        assert m.PN388_PRISTINE_OLD not in drifted
        assert m.PN388_POST_P34_OLD not in drifted
        target = _install_fake(tmp_path, monkeypatch, drifted)
        status, _reason = m.apply()
        assert status in ("skipped", "failed")
        assert target.read_text(encoding="utf-8") == drifted

    def test_apply_skips_when_gate_closed(self, tmp_path, monkeypatch):
        """Opt-in patch (default_on=False): with the dispatcher gate
        closed, apply() must skip without touching the target."""
        target = tmp_path / "scheduler.py"
        target.write_text(POST_P34_SCHEDULER, encoding="utf-8")
        monkeypatch.setattr(m, "resolve_vllm_file", lambda rel: str(target))
        import sndr.dispatcher as dispatcher

        monkeypatch.setattr(
            dispatcher, "should_apply", lambda pid: (False, "opt-in: env unset")
        )
        monkeypatch.delenv(
            "GENESIS_ENABLE_PN388_MAMBA_BLOCK_ALIGNED_SPLIT", raising=False
        )
        status, _reason = m.apply()
        assert status == "skipped"
        assert target.read_text(encoding="utf-8") == POST_P34_SCHEDULER

    def test_apply_skips_when_target_missing(self, monkeypatch):
        monkeypatch.setattr(m, "resolve_vllm_file", lambda rel: None)
        import sndr.dispatcher as dispatcher

        monkeypatch.setattr(
            dispatcher, "should_apply", lambda pid: (True, "test override")
        )
        status, _reason = m.apply()
        assert status == "skipped"


# ── Drift marker self-collision (tools/lint_drift_markers.py) ─────────


class TestDriftMarkerSelfCollision:
    def test_markers_not_substring_of_own_emitted_text(
        self, tmp_path, monkeypatch
    ):
        _install_fake(tmp_path, monkeypatch, POST_P34_SCHEDULER)
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

    def test_markers_match_45477_merged_form(self, tmp_path, monkeypatch):
        _install_fake(tmp_path, monkeypatch, POST_P34_SCHEDULER)
        patcher = m._make_patcher()
        assert any(
            dm in MERGED_SCHEDULER for dm in patcher.upstream_drift_markers
        )


# ── Behavioural correctness (the actual bug) ─────────────────────────


class TestPoisonInvariant:
    def test_patched_form_keeps_nonfinal_chunk_ends_aligned(self):
        """PROD shape (prompt 2002, mamba block 1600, eagle on) under a
        fragmenting concurrent budget: the patched split produces ONLY
        block-aligned non-final chunk ends (no poison)."""
        budgets = [364, 700, MAMBA_BLOCK + 13, 5000]
        ends = _run_prefill(_patched_split, PROMPT_LEN, budgets, True, MAMBA_BLOCK)
        assert ends[-1] == PROMPT_LEN, f"prefill stalled: {ends}"
        unaligned = [e for e in ends[:-1] if e % MAMBA_BLOCK != 0]
        assert not unaligned, f"non-final chunk ends not block-aligned: {unaligned}"

    def test_unpatched_pin_p34_form_poisons(self):
        """Regression witness: the CURRENT pin+P34 form produces unaligned
        non-final chunk ends — exactly the prefix-cache poison PN388
        removes. (If this ever stops failing the pin already has the fix.)"""
        budgets = [364, 700, MAMBA_BLOCK + 13, 5000]
        ends = _run_prefill(_pin_p34_split, PROMPT_LEN, budgets, True, MAMBA_BLOCK)
        unaligned = [e for e in ends[:-1] if e % MAMBA_BLOCK != 0]
        assert unaligned, (
            "expected the unpatched pin+P34 form to poison (unaligned "
            "non-final ends); none found — pin may already be fixed"
        )

    def test_unaligned_external_start_realigns(self):
        """Unaligned externally-computed start (KV connector) must re-align
        chunk ends — rounding the chunk END (not LENGTH) recovers boundary
        alignment on the first scheduled chunk (PR #45477 commit 2)."""
        prompt_len = 5 * MAMBA_BLOCK + 7
        for num_external in (1, 368, MAMBA_BLOCK - 1, MAMBA_BLOCK + 17):
            computed, ends, step = num_external, [], 0
            budgets = [364, 700, MAMBA_BLOCK + 13, prompt_len]
            while computed < prompt_len and step < 200:
                nn = _patched_split(
                    computed,
                    min(prompt_len - computed, budgets[step % len(budgets)]),
                    prompt_len, prompt_len, True, MAMBA_BLOCK,
                )
                step += 1
                if nn == 0:
                    continue
                computed += nn
                ends.append(computed)
            assert computed == prompt_len, f"stalled at {computed}/{prompt_len}"
            unaligned = [e for e in ends[:-1] if e % MAMBA_BLOCK != 0]
            assert not unaligned, (
                f"external={num_external}: unaligned non-final ends {unaligned}"
            )


# ── Pristine pin invariants (opportunistic) ──────────────────────────


@pytest.mark.skipif(
    not (PIN_TREE / "scheduler.py").is_file(),
    reason="pristine pin tree not present on this machine",
)
class TestAnchorsAgainstPristinePin:
    def test_pristine_anchor_unique_and_replacement_absent(self):
        src = (PIN_TREE / "scheduler.py").read_text(encoding="utf-8")
        assert src.count(m.PN388_PRISTINE_OLD) == 1
        assert m.PN388_PRISTINE_NEW not in src
        for dm in m._DRIFT_MARKERS:
            assert dm not in src

    def test_post_p34_anchor_assembled_from_p34_constants(self):
        """The post-P34 anchor must be exactly the pristine anchor with
        P34's documented transform applied — so it matches the real boot
        file once P34 has run."""
        src = (PIN_TREE / "scheduler.py").read_text(encoding="utf-8")
        assert _P34_OLD in src  # P34's own anchor is present in pristine
        assembled = m.PN388_PRISTINE_OLD.replace(_P34_OLD, _P34_NEW)
        assert assembled == m.PN388_POST_P34_OLD
