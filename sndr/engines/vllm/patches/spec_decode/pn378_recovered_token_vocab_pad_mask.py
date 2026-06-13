# SPDX-License-Identifier: Apache-2.0
"""PN378 — recovered-token vocab-pad -inf mask (vendor of vllm#45060, kernel half).

Upstream bug class (vllm#45060, root cause of #26372 / #33729 / #42722):
``sample_recovered_tokens_kernel`` in ``v1/sample/rejection_sampler.py``
tiles the vocab in ``BLOCK_SIZE`` chunks; the final tile's padding lanes
(``vocab_offset >= vocab_size``) load with ``other=0.0`` so their score
is ``0.0``. When the target model emits NaN logits, ``target_probs`` is
all-NaN and every REAL lane scores NaN — the NaN-propagating ``tl.max``
reduction then lets the contiguous zero run at the end of the last tile
win, and the kernel returns ``recovered_id == vocab_size``: an
out-of-vocabulary token id. ``RejectionSampler.parse_output`` keeps only
ids ``< vocab_size``, so that row collapses to an empty list — the
"empty spec-decode output" that crashed a Prometheus counter (#26372,
worked around by #33729) and tail-stalled DeepSeek V3.2 sync MTP
(#42722, the PR our PN133 vendors).

LIVE EXPOSURE on our stack: the wrapper hardcodes ``BLOCK_SIZE = 8192``
and Qwen3.6's vocab is 151936; ``151936 % 8192 == 4480 != 0``, so the
final tile carries 3712 padding lanes on every recovered-token sample of
both PROD models (35B FP8 MTP K=3, 27B int4 MTP K=3). A persistent-NaN
forward therefore livelocks the request: PN133 only repairs the
*accounting* (drafts counted as rejected, request stays schedulable) but
the row still commits zero tokens step after step — the hole this patch
closes at the source.

Vendor of OPEN PR vllm#45060 (studied via ``gh pr view`` + ``gh pr
diff`` 2026-06-11), KERNEL HALF ONLY: mask the padding lanes to ``-inf``
before the tile reduction. On all-NaN rows ``local_max`` is NaN, the
``local_max > max_val`` compare stays False, and ``recovered_id`` keeps
its in-vocab init (0). Healthy rows are unaffected: real scores are
``>= 0 > -inf`` and the argmax is byte-identical (upstream's existing
kernel parity test passed unchanged on the PR).

The SCHEDULER HALF of #45060 (``assert generated_token_ids``) is
deliberately NOT vendored — per the roadmap (chunk-3 Theme A,
2026-06-11) PN133 is the safer half: it keeps the request schedulable
and its v2 adds a ``logger.error`` observability arm on the exact
invariant the upstream assert enforces. PROD must degrade loudly, not
crash the engine core. COMPOSITION: PN378 (this patch) removes the
out-of-vocab source; PN133 v2 keeps accounting correct + logs if the
condition somehow still fires. Different files, zero overlap.

Genesis divergence — spelling only (documented per iron rule #10): our
mask line spells the constant ``float("-inf")`` where upstream #45060
writes ``-float("inf")``. The two are the same IEEE-754 value folded at
Triton trace time; the divergence exists so the PR's exact structural
line stays usable as an upstream drift marker without ever matching our
own emitted text (tools/lint_drift_markers.py self-collision contract).

Rebind analysis (verified against pristine pin g303916e93): the kernel
is a module-level ``@triton.jit`` function referenced ONLY by the
``sample_recovered_tokens`` wrapper in the SAME file; the GPU runner
imports ``RejectionSampler`` from this module, so a source-level text
patch applied before import needs NO runtime rebind. The CPU worker
(``cpu_model_runner.py``) rebinds the kernel to the SGL CPU shim — out
of scope (we run CUDA). The sibling
``v1/worker/gpu/spec_decode/rejection_sampler.py`` does not contain
this kernel (byte-verified, count 0).

Activation: opt-in via ``GENESIS_ENABLE_PN378_VOCAB_PAD_MASK=1``
(default OFF until the planned bench cycle — roadmap: land #45005
(PN372) + #45060 (this) in the same cycle to harden the full MTP K=3
failure chain). Self-skips when #45060 lands upstream: drift markers
below are exact substrings of the PR's form.

Author backport: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Vendor target: vllm-project/vllm#45060 (OPEN as of 2026-06-11).
"""
from __future__ import annotations

import logging

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, result_to_wiring_status

log = logging.getLogger("genesis.wiring.pn378_recovered_token_vocab_pad_mask")

GENESIS_PN378_MARKER = (
    "Genesis PN378 recovered-token vocab-pad mask "
    "(vendor of vllm#45060 kernel half) v1"
)

_TARGET_REL = "v1/sample/rejection_sampler.py"

# Drift markers — exact substrings of #45060's form, taken from
# `gh pr diff 45060` on 2026-06-11. Absent in the pristine pin tree
# (g303916e93: both count 0, byte-verified) and deliberately NOT
# substrings of our own replacement text: our mask line spells
# `float("-inf")` (never `-float("inf")`) and our comment block is
# original wording (lint_drift_markers self-collision contract).
_DRIFT_MARKERS = (
    # The PR's structural mask line (upstream constant spelling).
    '        score = tl.where(vocab_mask, score, -float("inf"))\n',
    # The PR's extended tile-reduction comment head.
    "        # Local tile reduction. Mask padding (``vocab_offset >= "
    "vocab_size``,\n",
)

# ── Sub-patch (required): the padding mask ───────────────────────────
# Anchor: the local tile reduction of sample_recovered_tokens_kernel.
# Unique in the file (count==1 byte-verified against
# /private/tmp/candidate_pin_current/vllm at pin g303916e93; the
# `score = prob * inv_q` product appears nowhere else).

PN378_MASK_OLD = (
    "        # Local tile reduction\n"
    "        score = prob * inv_q\n"
    "        local_max, local_id = tl.max(score, axis=0, return_indices=True)\n"
)

PN378_MASK_NEW = (
    "        # Local tile reduction\n"
    "        # [Genesis PN378 vendor of vllm#45060, kernel half] Padding\n"
    "        # lanes of the final vocab tile (vocab_offset >= vocab_size)\n"
    "        # load with other=0.0. With all-NaN target_probs the\n"
    "        # NaN-propagating tl.max otherwise lets that zero-score\n"
    "        # padding run win the reduction and the kernel returns\n"
    "        # recovered_id == vocab_size — an out-of-vocab id that\n"
    "        # RejectionSampler.parse_output drops, collapsing the row to\n"
    "        # [] and livelocking the request (Genesis PN133 only repairs\n"
    "        # the accounting; vocab 151936 % BLOCK_SIZE 8192 != 0 keeps\n"
    "        # the bug live on Qwen). Mask padding lanes to -inf:\n"
    "        # recovered_id keeps its in-vocab init (0) on NaN rows;\n"
    "        # healthy rows are unaffected (real scores >= 0 > -inf).\n"
    "        score = prob * inv_q\n"
    '        score = tl.where(vocab_mask, score, float("-inf"))\n'
    "        local_max, local_id = tl.max(score, axis=0, return_indices=True)\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_TARGET_REL)
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN378 v1/sample/rejection_sampler.py — recovered-token "
            "vocab-pad -inf mask (vendor of vllm#45060, kernel half)"
        ),
        target_file=str(target),
        marker=GENESIS_PN378_MARKER,
        sub_patches=[
            TextPatch(
                name="pn378_vocab_pad_mask",
                anchor=PN378_MASK_OLD,
                replacement=PN378_MASK_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=list(_DRIFT_MARKERS),
    )


def apply() -> tuple[str, str]:
    """Apply PN378 — recovered-token vocab-pad -inf mask. Never raises.

    Opt-in: gated through the dispatcher on
    ``GENESIS_ENABLE_PN378_VOCAB_PAD_MASK`` (default_on=False in the
    registry — pending the PN372+PN378 MTP-hardening bench cycle).
    """
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN378")
    log_decision("PN378", decision, reason)
    if not decision:
        return "skipped", reason

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", f"PN378: target file {_TARGET_REL} not found"

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "PN378 applied: sample_recovered_tokens_kernel now masks the "
            "vocab padding lanes of the final tile to -inf before the "
            "tl.max reduction, so an all-NaN target_probs row can never "
            "return an out-of-vocab recovered id (vllm#45060 kernel "
            "half). Closes the persistent-NaN livelock hole PN133 leaves "
            "open on Qwen vocab 151936 (% 8192 != 0). Scheduler half NOT "
            "taken — PN133 v2 logs the invariant instead."
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
