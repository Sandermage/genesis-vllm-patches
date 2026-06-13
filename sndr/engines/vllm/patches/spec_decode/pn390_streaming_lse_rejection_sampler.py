# SPDX-License-Identifier: Apache-2.0
"""PN390 — streaming-LSE rejection sampler (vendor of vllm#45369).

Upstream optimization (vllm#45369, OPEN/DRAFT as of 2026-06-13, studied
via ``gh pr view`` + ``gh pr diff`` 2026-06-13): the non-greedy
speculative rejection sampler currently materializes the full softmax
probability tensor

    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)

a ``[num_draft_tokens_total, vocab_size]`` fp32 buffer. The two Triton
kernels that consume it (``rejection_random_sample_kernel`` for the
accept/reject test, ``sample_recovered_tokens_kernel`` for the recovery
draw) read individual probabilities out of that buffer. The PR removes
the materialized tensor: it computes ONE logsumexp value per row with a
fused one-pass Triton producer (``compute_target_lse`` /
``target_lse_kernel``) and reconstructs the probability where it is
actually used as ``exp(logit - logsumexp(row))`` — the exact algebraic
identity ``softmax(logits)[i] == exp(logits[i] - logsumexp(logits))``.

LIVE EXPOSURE on our stack: both PROD models run MTP K=3 non-greedy
rejection sampling on every accepted/rejected draft, so this path fires
every decode step. The transient ``target_probs`` buffer at vocab 151936
is ``num_draft_tokens_total * 151936 * 4`` bytes — 3.6 MB at a single
draft row, 14.6 MB at a 24-row burst (K=3, batch 8). On the byte-bound
A5000 pair (PCIe, no NVLink) the full-vocab softmax WRITE + the kernel
READS of that buffer dominate the sampler mechanism. The PR's A100
sweep shows -8..-11% mechanism latency on the WITH-draft-probs rows
(the heavy ``else`` arm — exactly the arm P71/PN90/P82 keep live since
draft_probs is present under MTP), and larger reclaim is expected on
A5000 where the path is byte-bound rather than compute-bound. The
reclaimed 3.6-14.6 MB transient also returns cudagraph capture headroom.

Cross-patch composition (verified against pristine pin g303916e93):
  * PN378 (``pn378_recovered_token_vocab_pad_mask``) vendors the OTHER
    half of #45060 in the SAME file: it masks the vocab-padding lanes of
    ``sample_recovered_tokens_kernel``'s final tile to ``-inf`` with a
    ``tl.where(vocab_mask, score, ...)`` line on the SCORE product. PN390
    instead rewrites the PROB LOADS in that kernel
    (``other=0.0`` → ``other=float("-inf")`` on the target-logit loads,
    plus an ``exp(logit - lse)`` reconstruction). The two edits are
    LINE-ORTHOGONAL — PN378 splices between ``score = prob * inv_q`` and
    the reduction; PN390 rewrites the prob-load lines higher in the
    loop body. They compose cleanly when both are enabled, and PN390's
    own ``other=float("-inf")`` flip is the natural complement to PN378
    (a fully-masked tile now carries ``-inf`` logits whose ``exp`` is
    ``0`` — the recovery argmax cannot pick a padding lane).
  * PN369 (``pn369_relaxed_acceptance``) reads target probs on the TORCH
    side, NOT inside these kernels, so it is NOT touched by this kernel
    rewrite. If PN369 is ever promoted to consume the kernel path it must
    LSE-reconstruct (``exp(logit - lse)``) or recompute a local softmax —
    documented here so the dependency is not lost.
  * P71 (block-verify) and PN90 read the same rejection-sampler probs;
    #45369 reconstructs them ULP-identically, so acceptance decisions are
    unchanged outside the knife-edge ratio==uniform tie (the PR's parity
    test bounds this).

SELF-COLLISION HYGIENE (lint_drift_markers contract, roadmap chunk-3
Theme A "CRITICAL"): #45369 flips a kernel load ``other=0.0 → -inf`` —
the SAME -inf form as PN378's #45060 drift marker. The lint checks each
patcher in isolation (a drift marker must not be a substring of THAT
patcher's own emitted replacement text), so PN390's markers must be
disjoint from PN390's own output. We therefore do NOT use the
``other=float("-inf")`` flip or the vendored symbol names
(``compute_target_lse`` / ``target_lse_kernel``) as drift markers — we
emit those, so they would self-collide. Instead the drift markers are
two upstream-form lines we DELIBERATELY spell differently (documented
divergence per iron rule #10):
  * upstream's body constant ``BLOCK_SIZE: tl.constexpr = 8192`` — we
    name ours ``GENESIS_PN390_LSE_BLOCK_SIZE`` so upstream's exact line
    never appears in our emitted text; and
  * upstream's LSE-kernel store ``tl.store(target_lse_ptr + row, m +
    tl.log(s))`` — we factor the same value through a named intermediate
    (``lse = m + tl.log(s)``) so upstream's one-liner is absent from our
    output.
Both markers are exact substrings of #45369's merged form and are absent
from the pristine pin tree (g303916e93: both count 0, byte-verified).

Activation: opt-in via ``GENESIS_ENABLE_PN390_STREAMING_LSE_SAMPLER=1``
(default OFF — pending the A/B BLOCK_SIZE sweep on the server:
{8192/16384/...} and num_warps, which is too high at the 6-24 active
rows of single-stream MTP decode). The kernel rewrite is correctness-
ULP-stable but the block-size tuning is hardware-specific; PROD must not
adopt it before the bench cycle confirms the A5000 win. Self-skips when
#45369 lands upstream: drift markers below are exact substrings of the
PR's form.

Author backport: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Vendor target: vllm-project/vllm#45369 (OPEN/DRAFT as of 2026-06-13).
"""
from __future__ import annotations

import logging

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, result_to_wiring_status

log = logging.getLogger("genesis.wiring.pn390_streaming_lse_rejection_sampler")

GENESIS_PN390_MARKER = (
    "Genesis PN390 streaming-LSE rejection sampler "
    "(vendor of vllm#45369) v1"
)

_TARGET_REL = "v1/sample/rejection_sampler.py"

# Drift markers — exact substrings of #45369's merged form, taken from
# `gh pr diff 45369` on 2026-06-13. Absent in the pristine pin tree
# (g303916e93: both count 0, byte-verified) and deliberately NOT
# substrings of our own emitted text: we name our body constant
# `GENESIS_PN390_LSE_BLOCK_SIZE` (never the bare `BLOCK_SIZE: tl.constexpr
# = 8192`) and factor the store through a named `lse` intermediate
# (never the bare one-liner) — lint_drift_markers self-collision contract.
_DRIFT_MARKERS = (
    # Upstream's body block-size constant (we diverge: Genesis-named const).
    "    BLOCK_SIZE: tl.constexpr = 8192\n",
    # Upstream's fused-LSE store one-liner (we diverge: named intermediate).
    "    tl.store(target_lse_ptr + row, m + tl.log(s))\n",
)


# ── Sub-patch 1 (required): the import — add HAS_TRITON ───────────────
# `compute_target_lse` falls back to torch.logsumexp when triton is
# unavailable (CPU / no-triton build), matching the upstream guard.

PN390_IMPORT_OLD = "from vllm.triton_utils import tl, triton\n"
PN390_IMPORT_NEW = (
    "from vllm.triton_utils import HAS_TRITON, tl, triton  "
    "# [Genesis PN390 vendor of vllm#45369] HAS_TRITON gates the\n"
    "# streaming-LSE Triton producer; torch.logsumexp is the no-triton "
    "fallback.\n"
)


# ── Sub-patch 2 (required): rejection_sample body — drop the softmax ──
# Replace the materialized full-vocab `target_probs` with the per-row
# `target_lse`. Anchor: the softmax block in `rejection_sample`
# (count==1 byte-verified vs pristine g303916e93).

PN390_BODY_OLD = (
    "    # Compute probability distribution from target logits.\n"
    "    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)\n"
    "    assert target_probs.is_contiguous()\n"
)
PN390_BODY_NEW = (
    "    # [Genesis PN390 vendor of vllm#45369] Compute only the softmax\n"
    "    # normalizer (per-row logsumexp) instead of materializing the full\n"
    "    # [num_tokens, vocab_size] fp32 target_probs buffer. The kernels\n"
    "    # reconstruct each probability they need as exp(logit - lse) — the\n"
    "    # exact identity softmax(x)[i] == exp(x[i] - logsumexp(x)). This\n"
    "    # reclaims the transient probs tensor (3.6-14.6 MB at vocab 151936\n"
    "    # under MTP K=3) and the HBM traffic of writing/reading it.\n"
    "    # Genesis-named constant (never the bare `BLOCK_SIZE: tl.constexpr\n"
    "    # = 8192`) so the upstream form stays usable as a drift marker.\n"
    "    GENESIS_PN390_LSE_BLOCK_SIZE = 8192\n"
    "    target_lse = compute_target_lse(\n"
    "        target_logits, vocab_size, GENESIS_PN390_LSE_BLOCK_SIZE\n"
    "    )\n"
    "    assert target_lse.is_contiguous()\n"
)


# ── Sub-patch 3 (required): sample_recovered_tokens call site ─────────
# The `rejection_sample` body calls `sample_recovered_tokens(...,
# target_probs, ...)`; pass `target_logits, target_lse` instead.

PN390_SRT_CALL_OLD = (
    "        draft_token_ids,\n"
    "        draft_probs,\n"
    "        target_probs,\n"
    "        sampling_metadata,\n"
    "        device,\n"
    "        use_fp64_gumbel,\n"
    "    )\n"
)
PN390_SRT_CALL_NEW = (
    "        draft_token_ids,\n"
    "        draft_probs,\n"
    "        # [Genesis PN390 vendor of vllm#45369] logits + per-row LSE\n"
    "        # replace the materialized target_probs at the call site.\n"
    "        target_logits,\n"
    "        target_lse,\n"
    "        sampling_metadata,\n"
    "        device,\n"
    "        use_fp64_gumbel,\n"
    "    )\n"
)


# ── Sub-patch 4 (required): rejection_random_sample_kernel call site ──
# Same swap at the random-sample kernel launch.

PN390_RRS_CALL_OLD = (
    "        draft_token_ids,\n"
    "        draft_probs,\n"
    "        target_probs,\n"
    "        bonus_token_ids,\n"
    "        recovered_token_ids,\n"
    "        uniform_probs,\n"
)
PN390_RRS_CALL_NEW = (
    "        draft_token_ids,\n"
    "        draft_probs,\n"
    "        # [Genesis PN390 vendor of vllm#45369] logits + per-row LSE.\n"
    "        target_logits,\n"
    "        target_lse,\n"
    "        bonus_token_ids,\n"
    "        recovered_token_ids,\n"
    "        uniform_probs,\n"
)


# ── Sub-patch 5 (required): sample_recovered_tokens signature ─────────
# Wrapper takes `target_logits` + `target_lse` instead of `target_probs`;
# read vocab_size from the logits tensor.

PN390_SRT_SIG_OLD = (
    "    # [num_tokens, vocab_size]\n"
    "    target_probs: torch.Tensor,\n"
    "    sampling_metadata: SamplingMetadata,\n"
    "    device: torch.device,\n"
    "    use_fp64_gumbel: bool = False,\n"
    ") -> torch.Tensor:\n"
    "    # NOTE(woosuk): Create only one distribution for each request.\n"
    "    batch_size = len(num_draft_tokens)\n"
    "    vocab_size = target_probs.shape[-1]\n"
)
PN390_SRT_SIG_NEW = (
    "    # [num_tokens, vocab_size]\n"
    "    # [Genesis PN390 vendor of vllm#45369] target_logits + per-row LSE\n"
    "    # replace the materialized target_probs.\n"
    "    target_logits: torch.Tensor,\n"
    "    # [num_tokens]\n"
    "    target_lse: torch.Tensor,\n"
    "    sampling_metadata: SamplingMetadata,\n"
    "    device: torch.device,\n"
    "    use_fp64_gumbel: bool = False,\n"
    ") -> torch.Tensor:\n"
    "    # NOTE(woosuk): Create only one distribution for each request.\n"
    "    batch_size = len(num_draft_tokens)\n"
    "    vocab_size = target_logits.shape[-1]\n"
)


# ── Sub-patch 6 (required): sample_recovered_tokens_kernel launch arg ─
# Inside the wrapper, the kernel launch passes `target_probs` — swap to
# `target_logits, target_lse`.

PN390_SRT_KCALL_OLD = (
    "        draft_token_ids,\n"
    "        draft_probs,\n"
    "        target_probs,\n"
    "        inv_q,\n"
    "        vocab_size,\n"
    "        BLOCK_SIZE,\n"
)
PN390_SRT_KCALL_NEW = (
    "        draft_token_ids,\n"
    "        draft_probs,\n"
    "        # [Genesis PN390 vendor of vllm#45369] logits + per-row LSE.\n"
    "        target_logits,\n"
    "        target_lse,\n"
    "        inv_q,\n"
    "        vocab_size,\n"
    "        BLOCK_SIZE,\n"
)


# ── Sub-patch 7 (required): inject compute_target_lse + producer kernel
# Pure additions — the fused one-pass online-LSE producer and its Triton
# kernel. Inserted between `sample_recovered_tokens` and the greedy
# kernel (the same site #45369 uses). NOTE the deliberate spelling
# divergence in the store (named `lse` intermediate, never the upstream
# one-liner) so the drift marker stays disjoint from our emitted text.

PN390_INJECT_OLD = (
    "    return recovered_token_ids\n"
    "\n"
    "\n"
    "# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.\n"
    '@triton.jit(do_not_specialize=["max_spec_len"])\n'
    "def rejection_greedy_sample_kernel(\n"
)
PN390_INJECT_NEW = (
    "    return recovered_token_ids\n"
    "\n"
    "\n"
    "# [Genesis PN390 vendor of vllm#45369] Fused one-pass logsumexp\n"
    "# producer. Replaces the full-vocab softmax materialization: kernels\n"
    "# reconstruct probabilities on the fly as exp(logit - lse).\n"
    "def compute_target_lse(\n"
    "    target_logits: torch.Tensor,\n"
    "    vocab_size: int,\n"
    "    block_size: int,\n"
    ") -> torch.Tensor:\n"
    '    """Compute per-row logsumexp without materializing target probs."""\n'
    "    if target_logits.is_cuda and HAS_TRITON:\n"
    "        target_lse = torch.empty(\n"
    "            (target_logits.shape[0],),\n"
    "            dtype=torch.float32,\n"
    "            device=target_logits.device,\n"
    "        )\n"
    "        target_lse_kernel[(target_logits.shape[0],)](\n"
    "            target_logits,\n"
    "            target_lse,\n"
    "            vocab_size,\n"
    "            BLOCK_SIZE=block_size,\n"
    "            num_warps=8,\n"
    "        )\n"
    "        return target_lse\n"
    "    # No-triton / CPU fallback: torch's logsumexp is exact enough.\n"
    "    return torch.logsumexp(target_logits, dim=-1)\n"
    "\n"
    "\n"
    "@triton.jit\n"
    "def target_lse_kernel(\n"
    "    target_logits_ptr,  # [num_tokens, vocab_size]\n"
    "    target_lse_ptr,  # [num_tokens]\n"
    "    vocab_size,\n"
    "    BLOCK_SIZE: tl.constexpr,\n"
    "):\n"
    "    row = tl.program_id(0)\n"
    "\n"
    '    m = tl.full((), float("-inf"), tl.float32)\n'
    "    s = tl.full((), 0.0, tl.float32)\n"
    "    for v in range(0, vocab_size, BLOCK_SIZE):\n"
    "        vocab_offset = v + tl.arange(0, BLOCK_SIZE)\n"
    "        vocab_mask = vocab_offset < vocab_size\n"
    "        logits = tl.load(\n"
    "            target_logits_ptr + row * vocab_size + vocab_offset,\n"
    "            mask=vocab_mask,\n"
    '            other=float("-inf"),\n'
    "        ).to(tl.float32)\n"
    "        tile_max = tl.max(logits, axis=0)\n"
    "        new_m = tl.maximum(m, tile_max)\n"
    "        # If all visited tiles are -inf so far, new_m is -inf. Avoid\n"
    "        # forming -inf - -inf in either branch; masked contributions\n"
    "        # should be zero (handles fully-masked top-k/top-p vocab tiles).\n"
    '        finite_new_m = tl.where(new_m > float("-inf"), new_m, 0.0)\n'
    "        old_scale = tl.where(\n"
    '            m > float("-inf"),\n'
    "            tl.exp2((m - finite_new_m) * 1.4426950408889634),\n"
    "            0.0,\n"
    "        )\n"
    "        tile_contrib = tl.where(\n"
    '            logits > float("-inf"),\n'
    "            tl.exp2((logits - finite_new_m) * 1.4426950408889634),\n"
    "            0.0,\n"
    "        )\n"
    "        s = s * old_scale + tl.sum(tile_contrib, axis=0)\n"
    "        m = new_m\n"
    "\n"
    "    # [Genesis PN390] Factor the final logsumexp through a named\n"
    "    # intermediate rather than upstream's fused store one-liner, so\n"
    "    # that upstream line stays usable as a drift marker disjoint from\n"
    "    # our own emitted text (lint_drift_markers self-collision contract).\n"
    "    lse = m + tl.log(s)\n"
    "    tl.store(target_lse_ptr + row, lse)\n"
    "\n"
    "\n"
    "# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.\n"
    '@triton.jit(do_not_specialize=["max_spec_len"])\n'
    "def rejection_greedy_sample_kernel(\n"
)


# ── Sub-patch 8 (required): rejection_random_sample_kernel signature ──

PN390_RRS_SIG_OLD = (
    "    draft_probs_ptr,  # [num_tokens, vocab_size] or None\n"
    "    target_probs_ptr,  # [num_tokens, vocab_size]\n"
    "    bonus_token_ids_ptr,  # [batch_size]\n"
)
PN390_RRS_SIG_NEW = (
    "    draft_probs_ptr,  # [num_tokens, vocab_size] or None\n"
    "    # [Genesis PN390 vendor of vllm#45369] logits + per-row LSE ptrs.\n"
    "    target_logits_ptr,  # [num_tokens, vocab_size]\n"
    "    target_lse_ptr,  # [num_tokens]\n"
    "    bonus_token_ids_ptr,  # [batch_size]\n"
)


# ── Sub-patch 9 (required): rejection_random_sample_kernel prob load ──
# Reconstruct the accept-test target prob as exp(logit - lse).

PN390_RRS_LOAD_OLD = (
    "                target_prob = tl.load(\n"
    "                    target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id\n"
    "                )\n"
)
PN390_RRS_LOAD_NEW = (
    "                # [Genesis PN390 vendor of vllm#45369] Reconstruct the\n"
    "                # target prob as exp(logit - lse) instead of reading the\n"
    "                # materialized softmax buffer (ULP-identical to softmax).\n"
    "                target_logit = tl.load(\n"
    "                    target_logits_ptr + (start_idx + pos) * vocab_size + draft_token_id\n"
    "                )\n"
    "                target_lse = tl.load(target_lse_ptr + start_idx + pos)\n"
    "                target_prob = tl.exp(target_logit - target_lse)\n"
)


# ── Sub-patch 10 (required): sample_recovered_tokens_kernel signature ─

PN390_SRTK_SIG_OLD = (
    "    draft_probs_ptr,  # [num_tokens, vocab_size] or None\n"
    "    target_probs_ptr,  # [num_tokens, vocab_size]\n"
    "    inv_q_ptr,  # [batch_size, vocab_size]\n"
)
PN390_SRTK_SIG_NEW = (
    "    draft_probs_ptr,  # [num_tokens, vocab_size] or None\n"
    "    # [Genesis PN390 vendor of vllm#45369] logits + per-row LSE ptrs.\n"
    "    target_logits_ptr,  # [num_tokens, vocab_size]\n"
    "    target_lse_ptr,  # [num_tokens]\n"
    "    inv_q_ptr,  # [batch_size, vocab_size]\n"
)


# ── Sub-patch 11 (required): srtk per-row LSE preload ─────────────────
# Load the row's LSE once before the vocab-tile loop. Anchor: the
# token_idx init + the max_val branch (count==1).

PN390_SRTK_PRELOAD_OLD = (
    "    if NO_DRAFT_PROBS:\n"
    "        draft_token_id = tl.load(draft_token_ids_ptr + token_idx)\n"
    "\n"
    "    if USE_FP64_GUMBEL:\n"
)
PN390_SRTK_PRELOAD_NEW = (
    "    if NO_DRAFT_PROBS:\n"
    "        draft_token_id = tl.load(draft_token_ids_ptr + token_idx)\n"
    "\n"
    "    # [Genesis PN390 vendor of vllm#45369] Per-row LSE, loaded once;\n"
    "    # probabilities below are reconstructed as exp(logit - lse).\n"
    "    target_lse = tl.load(target_lse_ptr + token_idx)\n"
    "    if USE_FP64_GUMBEL:\n"
)


# ── Sub-patch 12 (required): srtk NO_DRAFT_PROBS prob load ────────────
# Mask padding lanes to -inf (their exp is 0) and reconstruct prob.

PN390_SRTK_NODRAFT_OLD = (
    "        if NO_DRAFT_PROBS:\n"
    "            prob = tl.load(\n"
    "                target_probs_ptr + token_idx * vocab_size + vocab_offset,\n"
    "                mask=(vocab_mask & (vocab_offset != draft_token_id)),\n"
    "                other=0.0,\n"
    "            )\n"
)
PN390_SRTK_NODRAFT_NEW = (
    "        if NO_DRAFT_PROBS:\n"
    "            # [Genesis PN390 vendor of vllm#45369] Masked/padding lanes\n"
    "            # load -inf so their reconstructed prob exp(-inf - lse) is 0.\n"
    "            target_logit = tl.load(\n"
    "                target_logits_ptr + token_idx * vocab_size + vocab_offset,\n"
    "                mask=(vocab_mask & (vocab_offset != draft_token_id)),\n"
    '                other=float("-inf"),\n'
    "            )\n"
    "            prob = tl.exp(target_logit - target_lse)\n"
)


# ── Sub-patch 13 (required): srtk with-draft-probs load (the HEAVY arm)
# The `else` arm — draft_probs present, LIVE under MTP K=3 + P71/PN90.

PN390_SRTK_DRAFT_OLD = (
    "            target_prob = tl.load(\n"
    "                target_probs_ptr + token_idx * vocab_size + vocab_offset,\n"
    "                mask=vocab_mask,\n"
    "                other=0.0,\n"
    "            )\n"
    "            prob = tl.maximum(target_prob - draft_prob, 0.0)\n"
)
PN390_SRTK_DRAFT_NEW = (
    "            # [Genesis PN390 vendor of vllm#45369] Heavy arm (draft\n"
    "            # probs present — LIVE under MTP K=3 + P71/PN90). Masked\n"
    "            # lanes load -inf so exp(-inf - lse) == 0.\n"
    "            target_logit = tl.load(\n"
    "                target_logits_ptr + token_idx * vocab_size + vocab_offset,\n"
    "                mask=vocab_mask,\n"
    '                other=float("-inf"),\n'
    "            )\n"
    "            target_prob = tl.exp(target_logit - target_lse)\n"
    "            prob = tl.maximum(target_prob - draft_prob, 0.0)\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_TARGET_REL)
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN390 v1/sample/rejection_sampler.py — streaming-LSE "
            "rejection sampler (vendor of vllm#45369)"
        ),
        target_file=str(target),
        marker=GENESIS_PN390_MARKER,
        sub_patches=[
            TextPatch(
                name="pn390_import_has_triton",
                anchor=PN390_IMPORT_OLD,
                replacement=PN390_IMPORT_NEW,
                required=True,
            ),
            TextPatch(
                name="pn390_body_drop_softmax",
                anchor=PN390_BODY_OLD,
                replacement=PN390_BODY_NEW,
                required=True,
            ),
            TextPatch(
                name="pn390_srt_call_site",
                anchor=PN390_SRT_CALL_OLD,
                replacement=PN390_SRT_CALL_NEW,
                required=True,
            ),
            TextPatch(
                name="pn390_rrs_call_site",
                anchor=PN390_RRS_CALL_OLD,
                replacement=PN390_RRS_CALL_NEW,
                required=True,
            ),
            TextPatch(
                name="pn390_srt_signature",
                anchor=PN390_SRT_SIG_OLD,
                replacement=PN390_SRT_SIG_NEW,
                required=True,
            ),
            TextPatch(
                name="pn390_srt_kernel_launch",
                anchor=PN390_SRT_KCALL_OLD,
                replacement=PN390_SRT_KCALL_NEW,
                required=True,
            ),
            TextPatch(
                name="pn390_inject_lse_producer",
                anchor=PN390_INJECT_OLD,
                replacement=PN390_INJECT_NEW,
                required=True,
            ),
            TextPatch(
                name="pn390_rrs_kernel_signature",
                anchor=PN390_RRS_SIG_OLD,
                replacement=PN390_RRS_SIG_NEW,
                required=True,
            ),
            TextPatch(
                name="pn390_rrs_kernel_load",
                anchor=PN390_RRS_LOAD_OLD,
                replacement=PN390_RRS_LOAD_NEW,
                required=True,
            ),
            TextPatch(
                name="pn390_srtk_signature",
                anchor=PN390_SRTK_SIG_OLD,
                replacement=PN390_SRTK_SIG_NEW,
                required=True,
            ),
            TextPatch(
                name="pn390_srtk_lse_preload",
                anchor=PN390_SRTK_PRELOAD_OLD,
                replacement=PN390_SRTK_PRELOAD_NEW,
                required=True,
            ),
            TextPatch(
                name="pn390_srtk_nodraft_load",
                anchor=PN390_SRTK_NODRAFT_OLD,
                replacement=PN390_SRTK_NODRAFT_NEW,
                required=True,
            ),
            TextPatch(
                name="pn390_srtk_draft_load",
                anchor=PN390_SRTK_DRAFT_OLD,
                replacement=PN390_SRTK_DRAFT_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=list(_DRIFT_MARKERS),
    )


def apply() -> tuple[str, str]:
    """Apply PN390 — streaming-LSE rejection sampler. Never raises.

    Opt-in: gated through the dispatcher on
    ``GENESIS_ENABLE_PN390_STREAMING_LSE_SAMPLER`` (default_on=False in
    the registry — pending the A/B BLOCK_SIZE + num_warps bench sweep on
    the server before PROD adoption).
    """
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN390")
    log_decision("PN390", decision, reason)
    if not decision:
        return "skipped", reason

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", f"PN390: target file {_TARGET_REL} not found"

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "PN390 applied: the non-greedy rejection sampler no longer "
            "materializes the full-vocab target_probs softmax buffer. "
            "compute_target_lse produces one logsumexp per row and the "
            "rejection / recovery kernels reconstruct each probability as "
            "exp(logit - lse) (ULP-identical to softmax). Reclaims the "
            "3.6-14.6 MB transient at vocab 151936 under MTP K=3 and its "
            "HBM traffic. Line-orthogonal to PN378 (which masks the score "
            "product in the same kernel). vllm#45369."
        ),
        patch_name=patcher.patch_name,
    )


def is_applied() -> bool:
    """Return True iff our marker is present in the target file."""
    patcher = _make_patcher()
    if patcher is None:
        return False
    try:
        with open(patcher.target_file, encoding="utf-8") as f:
            return patcher.marker in f.read()
    except (OSError, UnicodeDecodeError):
        return False
