# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PN369 — relaxed acceptance, torch / Triton runtime group.

Companion to `test_pn369_relaxed_acceptance.py` (torch-less group). This
file imports torch at module level and is auto-skipped on torch-less
hosts by the tests/conftest.py AST scan; run it inside the vLLM
container (or any torch-capable host):

  python3 -m pytest \
      tests/unit/integrations/spec_decode/test_pn369_relaxed_acceptance_torch.py -v

Covers:
  - vectorized relaxed_ok mask vs a hand-computed loop reference
    (random fixtures + hand fixtures + delta=0 near-strict degenerate)
  - env-gated `compute_relaxed_ok_mask` entry point (None when off)
  - block-verify PyTorch reference TAIL EXTENSION: no extension /
    extend by 1 / extend by 2 / gap stops the walk / extend to full ->
    bonus / greedy never extends
  - relaxed-all-zeros == relaxed-None bit-equivalence (degenerate OFF)
  - call_block_verify_sample precondition contract fix (accepts the
    upstream [batch_size] cumsum layout; rejects [batch_size + 1])
  - Triton kernel: default-args OFF equivalence (old call sites remain
    source- and bit-compatible), tail-extension cases, Triton-PyTorch
    parity on the same fixtures (CUDA only)
"""
from __future__ import annotations

import pytest
import torch

from sndr.engines.vllm.kernels_legacy.block_verify_sampler import (
    PLACEHOLDER_TOKEN_ID,
    call_block_verify_sample,
    compute_relaxed_ok_mask,
    rejection_random_sample_block_verify_pytorch,
    relaxed_ok_mask,
)
from sndr.engines.vllm.kernels_legacy import block_verify_sampler as bvs

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _mask_loop_reference(target_probs, draft_token_ids, topk, delta):
    """Hand-computed loop reference for the relaxed_ok mask."""
    num_tokens, vocab = target_probs.shape
    k = max(1, min(topk, vocab))
    out = torch.zeros(num_tokens, dtype=torch.int32)
    for i in range(num_tokens):
        row = target_probs[i]
        tok = int(draft_token_ids[i].item())
        top_vals, top_ids = row.topk(k)
        in_topk = tok in top_ids.tolist()
        within = float(row[tok].item()) >= float(top_vals[0].item()) - delta
        out[i] = 1 if (in_topk and within) else 0
    return out


# ════════════════════════════════════════════════════════════════════════
# relaxed_ok mask
# ════════════════════════════════════════════════════════════════════════


class TestRelaxedMask:
    def test_mask_matches_loop_reference_random(self):
        torch.manual_seed(7)
        for topk, delta in [(1, 0.0), (4, 0.2), (8, 0.5), (32, 1.0)]:
            target = torch.randn(16, 64).softmax(-1)
            drafts = torch.randint(0, 64, (16,), dtype=torch.int32)
            got = relaxed_ok_mask(target, drafts, topk, delta)
            want = _mask_loop_reference(target, drafts, topk, delta)
            assert torch.equal(got.cpu(), want), (
                f"topk={topk} delta={delta}: {got.tolist()} != {want.tolist()}"
            )

    def test_mask_hand_fixture(self):
        """Hand-built fixture covering all quadrant outcomes."""
        target = torch.tensor([
            # top1=0.50(tok0); tok1=0.35 in top4, 0.50-0.35 <= 0.2 -> PASS
            [0.50, 0.35, 0.10, 0.05],
            # top1=0.50(tok0); tok2=0.10 in top4 but 0.50-0.10 > 0.2 -> FAIL
            [0.50, 0.35, 0.10, 0.05],
            # tok0 IS top1 -> trivially in window -> PASS
            [0.50, 0.35, 0.10, 0.05],
        ])
        drafts = torch.tensor([1, 2, 0], dtype=torch.int32)
        got = relaxed_ok_mask(target, drafts, 4, 0.2)
        assert got.tolist() == [1, 0, 1]
        # topk=2: tok2=0.10 not in top2 -> FAIL even though delta=1.0
        got2 = relaxed_ok_mask(target[1:2], drafts[1:2], 2, 1.0)
        assert got2.tolist() == [0]

    def test_mask_delta_zero_near_strict(self):
        """delta=0 degenerates to ties-with-top1 only (near-strict)."""
        target = torch.tensor([
            [0.40, 0.40, 0.15, 0.05],    # tok1 ties top1 -> PASS
            [0.50, 0.49, 0.005, 0.005],  # tok1 below top1 -> FAIL
        ])
        drafts = torch.tensor([1, 1], dtype=torch.int32)
        got = relaxed_ok_mask(target, drafts, 4, 0.0)
        assert got.tolist() == [1, 0]

    def test_mask_topk_clamped_to_vocab(self):
        target = torch.tensor([[0.7, 0.2, 0.1]])
        drafts = torch.tensor([2], dtype=torch.int32)
        # topk=32 > vocab=3 must not raise; delta=1.0 -> everything passes
        got = relaxed_ok_mask(target, drafts, 32, 1.0)
        assert got.tolist() == [1]

    def test_mask_dtype_and_contiguity(self):
        target = torch.randn(8, 32).softmax(-1)
        drafts = torch.randint(0, 32, (8,), dtype=torch.int32)
        got = relaxed_ok_mask(target, drafts, 4, 0.2)
        assert got.dtype == torch.int32
        assert got.is_contiguous()
        assert got.shape == drafts.shape

    def test_compute_mask_env_gated(self, monkeypatch):
        target = torch.randn(4, 16).softmax(-1)
        drafts = torch.randint(0, 16, (4,), dtype=torch.int32)

        monkeypatch.delenv("GENESIS_ENABLE_PN369_RELAXED_ACCEPTANCE", raising=False)
        bvs._pn369_reset_config_cache()
        assert compute_relaxed_ok_mask(target, drafts) is None

        monkeypatch.setenv("GENESIS_ENABLE_PN369_RELAXED_ACCEPTANCE", "1")
        monkeypatch.setenv("GENESIS_PN369_RELAXED_TOPK", "4")
        monkeypatch.setenv("GENESIS_PN369_RELAXED_DELTA", "0.2")
        bvs._pn369_reset_config_cache()
        got = compute_relaxed_ok_mask(target, drafts)
        assert got is not None
        want = _mask_loop_reference(target, drafts, 4, 0.2)
        assert torch.equal(got.cpu(), want)
        bvs._pn369_reset_config_cache()


# ════════════════════════════════════════════════════════════════════════
# block-verify tail extension (PyTorch reference, CPU)
# ════════════════════════════════════════════════════════════════════════
#
# Shared fixture (single request, gamma=3): one-hot drafts with
# target=0.5 at the drafted token and 0.5 at the next token.
#   ratio_k = 0.5      -> p_prefix = [1, .5, .25, .125]
#   h_block = [1/3, 1/7, 0.125]
# So shared u = 0.99 rejects ALL positions (block rule accepts 0) and
# shared u = 0.20 accepts EXACTLY position 0 (0.2 <= 1/3, > 1/7, > .125).


def _tail_fixture(u_shared: float, device="cpu"):
    max_spec_len = 3
    vocab_size = 4
    out = torch.full((1, max_spec_len + 1), PLACEHOLDER_TOKEN_ID,
                     dtype=torch.int32, device=device)
    cu = torch.tensor([3], dtype=torch.int32, device=device)
    drafts = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    draft_probs = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], device=device)
    target_probs = torch.tensor([
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.5, 0.5, 0.0],
        [0.0, 0.0, 0.5, 0.5],
    ], device=device)
    bonus = torch.tensor([[42]], dtype=torch.int32, device=device)
    recovered = torch.tensor([97, 98, 99], dtype=torch.int32, device=device)
    uniform = torch.tensor([u_shared, 0.5, 0.5], device=device)
    is_greedy = torch.tensor([False], device=device)
    return (out, cu, drafts, draft_probs, target_probs, bonus, recovered,
            uniform, is_greedy, max_spec_len, vocab_size)


def _run_pt_reference(fixture, relaxed_ok):
    rejection_random_sample_block_verify_pytorch(*fixture, relaxed_ok=relaxed_ok)
    return fixture[0]


class TestBlockVerifyTailExtension:
    def test_no_extension_zero_mask_equals_none(self):
        """relaxed_ok all-zeros must be bit-identical to relaxed_ok=None
        (degenerate OFF equivalence)."""
        out_none = _run_pt_reference(_tail_fixture(0.99), None)
        out_zeros = _run_pt_reference(
            _tail_fixture(0.99), torch.zeros(3, dtype=torch.int32)
        )
        assert torch.equal(out_none, out_zeros)
        # Block rule rejected everything -> recovered at position 0.
        assert out_none[0].tolist() == [97, -1, -1, -1]

    def test_extend_by_one_from_block_accept_one(self):
        """Block rule accepts exactly position 0; relaxed window passes at
        the next position only -> accepted_len 1 -> 2, recovered at 2."""
        relaxed = torch.tensor([0, 1, 0], dtype=torch.int32)
        out = _run_pt_reference(_tail_fixture(0.20), relaxed)
        assert out[0].tolist() == [0, 1, 99, -1]

    def test_extend_by_two_from_block_accept_zero(self):
        """Block rule accepts nothing; relaxed passes at positions 0,1 but
        not 2 -> accepted_len 0 -> 2, recovered at the NEW first-rejected
        position (2)."""
        relaxed = torch.tensor([1, 1, 0], dtype=torch.int32)
        out = _run_pt_reference(_tail_fixture(0.99), relaxed)
        assert out[0].tolist() == [0, 1, 99, -1]

    def test_extension_stops_at_first_gap(self):
        """Contiguity: a gap at position 1 blocks position 2 even though
        the window passes there."""
        relaxed = torch.tensor([1, 0, 1], dtype=torch.int32)
        out = _run_pt_reference(_tail_fixture(0.99), relaxed)
        assert out[0].tolist() == [0, 98, -1, -1]

    def test_extend_to_full_emits_bonus(self):
        """Extension reaching the full draft length must append the bonus
        token (full-accept semantics preserved)."""
        relaxed = torch.tensor([1, 1, 1], dtype=torch.int32)
        out = _run_pt_reference(_tail_fixture(0.99), relaxed)
        assert out[0].tolist() == [0, 1, 2, 42]

    def test_greedy_rows_never_extend(self):
        fixture = list(_tail_fixture(0.99))
        fixture[8] = torch.tensor([True])  # is_greedy
        relaxed = torch.tensor([1, 1, 1], dtype=torch.int32)
        out = _run_pt_reference(tuple(fixture), relaxed)
        assert out[0].tolist() == [-1, -1, -1, -1]

    def test_call_entry_point_accepts_upstream_cu_layout(self):
        """Regression test for the A4 precondition contract fix: the
        upstream caller passes cu_num_draft_tokens as np.cumsum WITHOUT a
        leading zero, i.e. shape [batch_size]. The old check required
        [batch_size + 1] and made EVERY call raise ValueError (permanent
        silent fallback to the per-token path)."""
        (out, cu, drafts, draft_probs, target_probs, bonus, _recovered,
         uniform, is_greedy, max_spec_len, vocab_size) = _tail_fixture(0.99)
        result = call_block_verify_sample(
            output_token_ids=out,
            cu_num_draft_tokens=cu,            # [batch_size] = [1]
            draft_token_ids=drafts,
            draft_probs=draft_probs,
            target_probs=target_probs,
            bonus_token_ids=bonus,
            uniform_probs=uniform,
            is_greedy=is_greedy,
            num_draft_tokens=[3],
            generators={},
            max_spec_len=max_spec_len,
            vocab_size=vocab_size,
            use_pytorch=True,
        )
        # u=0.99 rejects all -> recovered token written at position 0.
        assert result[0, 0].item() != PLACEHOLDER_TOKEN_ID
        assert result[0, 1].item() == PLACEHOLDER_TOKEN_ID

    def test_call_entry_point_rejects_wrong_cu_layout(self):
        (out, _cu, drafts, draft_probs, target_probs, bonus, _recovered,
         uniform, is_greedy, max_spec_len, vocab_size) = _tail_fixture(0.99)
        with pytest.raises(ValueError, match="must equal batch_size"):
            call_block_verify_sample(
                output_token_ids=out,
                cu_num_draft_tokens=torch.tensor([0, 3], dtype=torch.int32),
                draft_token_ids=drafts,
                draft_probs=draft_probs,
                target_probs=target_probs,
                bonus_token_ids=bonus,
                uniform_probs=uniform,
                is_greedy=is_greedy,
                num_draft_tokens=[3],
                generators={},
                max_spec_len=max_spec_len,
                vocab_size=vocab_size,
                use_pytorch=True,
            )

    def test_call_entry_point_rejects_bad_relaxed_shape(self):
        (out, cu, drafts, draft_probs, target_probs, bonus, _recovered,
         uniform, is_greedy, max_spec_len, vocab_size) = _tail_fixture(0.99)
        with pytest.raises(ValueError, match="relaxed_ok shape"):
            call_block_verify_sample(
                output_token_ids=out,
                cu_num_draft_tokens=cu,
                draft_token_ids=drafts,
                draft_probs=draft_probs,
                target_probs=target_probs,
                bonus_token_ids=bonus,
                uniform_probs=uniform,
                is_greedy=is_greedy,
                num_draft_tokens=[3],
                generators={},
                max_spec_len=max_spec_len,
                vocab_size=vocab_size,
                use_pytorch=True,
                relaxed_ok=torch.zeros(5, dtype=torch.int32),
            )

    def test_call_entry_point_with_relaxed_full_extension(self):
        """End-to-end through the public entry point: full relaxed
        extension emits drafts + bonus."""
        (out, cu, drafts, draft_probs, target_probs, bonus, _recovered,
         uniform, is_greedy, max_spec_len, vocab_size) = _tail_fixture(0.99)
        result = call_block_verify_sample(
            output_token_ids=out,
            cu_num_draft_tokens=cu,
            draft_token_ids=drafts,
            draft_probs=draft_probs,
            target_probs=target_probs,
            bonus_token_ids=bonus,
            uniform_probs=uniform,
            is_greedy=is_greedy,
            num_draft_tokens=[3],
            generators={},
            max_spec_len=max_spec_len,
            vocab_size=vocab_size,
            use_pytorch=True,
            relaxed_ok=torch.ones(3, dtype=torch.int32),
        )
        assert result[0].tolist() == [0, 1, 2, 42]


# ════════════════════════════════════════════════════════════════════════
# Triton kernel (CUDA)
# ════════════════════════════════════════════════════════════════════════


@requires_cuda
class TestTritonTailExtension:
    def _run_triton(self, fixture, relaxed_ok):
        from sndr.engines.vllm.kernels_legacy.block_verify_sampler import (
            _BLOCK_VERIFY_VOCAB_BLOCK,
            rejection_random_sample_block_verify_kernel,
        )
        (out, cu, drafts, draft_probs, target_probs, bonus, recovered,
         uniform, is_greedy, max_spec_len, vocab_size) = fixture
        rejection_random_sample_block_verify_kernel[(1,)](
            out, cu, drafts, draft_probs, target_probs, bonus, recovered,
            uniform, is_greedy, max_spec_len, vocab_size,
            BLOCK_SIZE=_BLOCK_VERIFY_VOCAB_BLOCK,
            relaxed_ok_ptr=relaxed_ok,
            RELAXED_MODE=relaxed_ok is not None,
        )
        return out

    def test_kernel_default_args_off_equivalence(self):
        """Calling the kernel WITHOUT the new args (old positional form)
        must still work (parameter defaults) and equal an explicit
        RELAXED_MODE=False run — proves existing call sites stay source-
        and bit-compatible."""
        from sndr.engines.vllm.kernels_legacy.block_verify_sampler import (
            _BLOCK_VERIFY_VOCAB_BLOCK,
            rejection_random_sample_block_verify_kernel,
        )
        fix_old = _tail_fixture(0.99, device="cuda")
        (out, cu, drafts, draft_probs, target_probs, bonus, recovered,
         uniform, is_greedy, max_spec_len, vocab_size) = fix_old
        rejection_random_sample_block_verify_kernel[(1,)](
            out, cu, drafts, draft_probs, target_probs, bonus, recovered,
            uniform, is_greedy, max_spec_len, vocab_size,
            BLOCK_SIZE=_BLOCK_VERIFY_VOCAB_BLOCK,
        )
        out_explicit = self._run_triton(_tail_fixture(0.99, device="cuda"), None)
        assert torch.equal(out, out_explicit)

    def test_triton_zero_mask_equals_off(self):
        out_off = self._run_triton(_tail_fixture(0.99, device="cuda"), None)
        out_zeros = self._run_triton(
            _tail_fixture(0.99, device="cuda"),
            torch.zeros(3, dtype=torch.int32, device="cuda"),
        )
        assert torch.equal(out_off, out_zeros)

    @pytest.mark.parametrize("u,relaxed,expected", [
        (0.99, [0, 0, 0], [97, -1, -1, -1]),   # no extension
        (0.20, [0, 1, 0], [0, 1, 99, -1]),     # block 1 -> extend by 1
        (0.99, [1, 1, 0], [0, 1, 99, -1]),     # block 0 -> extend by 2
        (0.99, [1, 0, 1], [0, 98, -1, -1]),    # gap stops the walk
        (0.99, [1, 1, 1], [0, 1, 2, 42]),      # full extension -> bonus
    ])
    def test_triton_tail_extension_cases(self, u, relaxed, expected):
        out = self._run_triton(
            _tail_fixture(u, device="cuda"),
            torch.tensor(relaxed, dtype=torch.int32, device="cuda"),
        )
        assert out[0].tolist() == expected

    @pytest.mark.parametrize("u,relaxed", [
        (0.99, [0, 0, 0]),
        (0.20, [0, 1, 0]),
        (0.99, [1, 1, 0]),
        (0.99, [1, 0, 1]),
        (0.99, [1, 1, 1]),
    ])
    def test_triton_pytorch_parity_tail_extension(self, u, relaxed):
        relaxed_cpu = torch.tensor(relaxed, dtype=torch.int32)
        out_pt = _run_pt_reference(_tail_fixture(u), relaxed_cpu)
        out_tri = self._run_triton(
            _tail_fixture(u, device="cuda"), relaxed_cpu.cuda()
        )
        assert torch.equal(out_pt, out_tri.cpu()), (
            f"u={u} relaxed={relaxed}: pt={out_pt.tolist()} "
            f"tri={out_tri.cpu().tolist()}"
        )
