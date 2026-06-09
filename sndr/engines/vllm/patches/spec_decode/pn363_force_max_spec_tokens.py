# SPDX-License-Identifier: Apache-2.0
"""PN363 — vendor of OPEN PR vllm#43114 force_max_spec_tokens for FULL CG.

Pad SuffixDecodingProposer drafts to ``num_speculative_tokens`` with
``eos_token_id`` so spec-decode batches hit FULL CUDA graph dispatch
instead of falling back to PIECEWISE.

================================================================
WHAT THIS DOES (verbatim per upstream PR)
================================================================

Today, ``SuffixDecodingProposer.propose()`` returns lists of varying
length (1, 2, 3, ... tokens per request). The target-model
``GPUModelRunner`` then builds ``num_scheduled_tokens`` per request as
``1 + len(drafts)``. Because requests in the same batch have different
draft counts, ``get_uniform_token_count`` in
``vllm/v1/worker/gpu/cudagraph_utils.py`` returns ``None``
(``num_tokens != max_query_len * num_reqs``), and the dispatcher
downgrades from ``CUDAGraphMode.FULL`` to ``CUDAGraphMode.PIECEWISE``.

PR author's measurement on a Qwen3-0.6B suffix-decoding run shows
~88% of decode steps run as PIECEWISE under default behaviour:

    | Unpadded | Padded | Runtime    | Count |
    | 1        | 4      | PIECEWISE  | 173   |
    | 2        | 4      | PIECEWISE  | 138   |
    | 3        | 4      | PIECEWISE  |  25   |
    | 4        | 4      | FULL       |  15   |

After this patch:

    | 4        | 4      | FULL       | 1200  |
    | 9        | 16     | PIECEWISE  |    3  |

Author measured -15% average ITL at 8-concurrency in PROD on
MiniMaxM2 (TP8+EP, Ascend hardware).

The padding token is the model's ``eos_token_id``. The rejection
sampler in **greedy mode** rejects it deterministically (drafted !=
target argmax in 99.999% of cases), so the request still advances by
the same number of accepted tokens — only the graph dispatch mode
changes.

================================================================
APPLICABILITY TO GENESIS PROD (iron-rule #11 — adapt-then-ship)
================================================================

**Direct applicability**: ``method='suffix'`` ONLY. Genesis PROD
launches ``method='mtp'`` (Qwen3.6-35B-A3B + native MTP K=3).
SuffixDecodingProposer is **NOT** instantiated for our 35B PROD
container, so PN363 by itself is a **NO-OP at our PROD critical
path**.

**Why we ship it anyway**:

  1. **Audit clarity** — the operator referenced PR#43114 in the
     Recovery Lever audit as a candidate. Shipping the mechanical
     backport (default OFF) documents that we read it, applied the
     fix to the right file, and verified the upstream code is
     reachable. Removes it from the "open candidates" pile.
  2. **A/B reuse** — Suffix Decoding is one of the bench A/B levers
     for chat workloads (P75 ``GENESIS_ENABLE_P75``). If/when we A/B
     suffix on 35B, PN363 ensures the comparison runs on the FULL CG
     path (apples-to-apples vs MTP FULL CG path).
  3. **Defensive carrying** — if a future config switches a non-PROD
     bench to suffix on Lorbus 27B INT4, this fix is already in the
     overlay tree, gated behind ``GENESIS_ENABLE_PN363=1``.

**MTP adaptation — DEFERRED to PN364 (design note)**:

The structurally analogous bug for MTP K=3 is *not* in the proposer
(MTP's ``AutoRegressiveSpeculator.propose`` already produces a fixed
``[num_reqs, K]`` tensor — see
``vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py:118``
where ``self.draft_tokens`` is preallocated as
``[max_num_reqs, num_speculative_steps]``). For MTP, the
``uniform_token_count == None`` regression originates on the
*target* side: when the scheduler's
``scheduled_spec_decode_tokens[req_id]`` dict-key is missing for
some requests (post-prefill, post-abort, post-trim during
chunked-prefill), those requests get ``num_scheduled_tokens=1`` while
others get ``num_scheduled_tokens=1+K=4``. Mixed batch ⇒
``get_uniform_token_count`` returns ``None`` ⇒ ``FULL → PIECEWISE``.

**Why a direct suffix-style "pad with eos" port to MTP is UNSAFE**:

  * Greedy rejection: ``accept iff drafted == target_argmax``.
    Padding with ``eos_token_id`` is safe — it gets rejected
    deterministically (target_argmax ≠ eos in 99.999% of decoding
    positions), and the request advances by the same number of
    accepted tokens as without padding.
  * **Probabilistic rejection** (Leviathan-Kalman-Mehta 2023,
    ``min(1, target_p / draft_p)``): the draft probability of the
    pad-eos token at that position is **0** (the MTP draft did not
    propose it). Division by zero ⇒ implementation-defined behaviour
    (NaN / clamp-to-1 / always-accept) ⇒ silently accepts an
    eos-token we didn't want. Genesis PROD launches with
    ``draft_sample_method: probabilistic`` (see PN361 credit), so
    direct port would be a **correctness regression**, not a perf
    win.

**Correct MTP design (PN364, future work)**:

  * Pad ``scheduled_spec_decode_tokens[req_id]`` to K-length, BUT
    also pad the corresponding ``draft_probs`` rows with a one-hot
    at the pad-token-id (so probabilistic rejection sees
    ``draft_p == 1.0`` for the pad slot — guaranteed rejection
    iff ``target_p[eos] < 1.0`` which is virtually always true at
    decode positions). This requires editing
    ``GPUModelRunner._get_spec_decode_draft_probs`` to inject the
    one-hot fake rows for missing-K requests. ~40 LOC, needs two
    composing edits (scheduler + runner), plus a unit test
    verifying the rejection-sampler output is invariant to padding.
    Deferred until operator confirms suffix path is the wrong A/B
    lever and dedicates the design budget.

**Composition matrix**:

  * PN340 / PN341 (MTP decode-bubble GDN attn / GPU-runner): different
    files. PN363 = suffix file only. No anchor overlap.
  * PN357 (Eagle3/DFlash remapped greedy speedup): different file
    (eagle3_llama / dflash_qwen3). No overlap.
  * PN361 (fail-closed missing draft probs): different file
    (gpu_model_runner.py). No overlap.
  * PN133 (MTP scheduler empty-output): different anchor in
    ``scheduler.py``. No overlap.
  * PN348 (qwen3 MTP backbone dedup): different file. No overlap.
  * G_DYNAMIC_K_MTP (Genesis-original adaptive K): patches
    ``DraftModelProposer`` (MTP path), not suffix. Orthogonal.
  * P75 (suffix-decoding enable): PN363 EXTENDS P75 by giving its
    proposer the new ``force_max_spec_tokens`` knob. Composes
    cleanly (P75 enables the suffix code path; PN363 upgrades it).

================================================================
Implementation strategy
================================================================

Two text-patches on the same file
``vllm/v1/spec_decode/suffix_decoding.py``:

  (a) ``pn363_init`` — inject ``force_max_spec_tokens`` /
      ``_pad_token_id`` / ``_pad_template`` initialisation at the
      end of ``__init__``. Anchor: the ``self.suffix_cache = ...``
      construction (unique in the file).

  (b) ``pn363_propose`` — replace the trailing
      ``draft_token_ids.append(draft.token_ids)`` line with the
      pad-aware variant. Anchor: ``min_token_prob=self.min_token_prob,
      )`` + the ``draft_token_ids.append(draft.token_ids)`` line
      below it (joint anchor — unique in the file).

Both anchors are stable against the PR's own diff (verified against
the upstream pre-image at /tmp/pr43114.diff).

Authorship and provenance
=========================

  * Author: Sander (Sandermage / Aleksandr Barzov), Odessa, Ukraine.
  * Vendor target: vllm-project/vllm#43114 (OPEN as of 2026-06-09,
    author Csrayz, last touched 2026-06-01).
  * Upstream pre-image SHA1 (vllm pin 0.22.1rc1.dev259+g303916e93):
    verified that ``self.num_speculative_tokens = config.num_speculative_tokens``
    and ``draft_token_ids.append(draft.token_ids)`` are present in
    the live container's ``suffix_decoding.py``.
  * Reads: ``/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/suffix_decoding.py``
    (live container, 2026-06-09).
  * Comparison: PR diff at /tmp/pr43114.diff (4 files, +248/-4).
  * Verification commands shipped at end of this docstring.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

log = logging.getLogger("genesis.wiring.pn363_force_max_spec_tokens")

GENESIS_PN363_MARKER = (
    "Genesis PN363 vendor of vllm#43114 (force_max_spec_tokens) v1"
)

_TARGET_REL = "v1/spec_decode/suffix_decoding.py"


# ── Sub-patch (a): __init__ — add force_max_spec_tokens plumbing ──
#
# Anchor: the unique 3-line `self.suffix_cache = SuffixDecodingCache(...)`
# construction block. Verified unique in pin
# 0.22.1rc1.dev259+g303916e93 via grep on the live container.
PN363_INIT_OLD = (
    "        self.suffix_cache = SuffixDecodingCache(\n"
    "            max_tree_depth=config.suffix_decoding_max_tree_depth,\n"
    "            max_cached_requests=config.suffix_decoding_max_cached_requests,\n"
    "        )\n"
)
PN363_INIT_NEW = (
    "        self.suffix_cache = SuffixDecodingCache(\n"
    "            max_tree_depth=config.suffix_decoding_max_tree_depth,\n"
    "            max_cached_requests=config.suffix_decoding_max_cached_requests,\n"
    "        )\n"
    "\n"
    "        # [Genesis PN363 vendor of vllm#43114] force_max_spec_tokens plumbing.\n"
    "        # When enabled in SpeculativeConfig, propose() pads short draft\n"
    "        # lists to num_speculative_tokens with eos_token_id so target-side\n"
    "        # get_uniform_token_count() can return a non-None value and the\n"
    "        # dispatcher selects CUDAGraphMode.FULL instead of PIECEWISE.\n"
    "        # The pad token is rejected by the GREEDY rejection sampler\n"
    "        # (drafted != target_argmax at the pad slot), so the request\n"
    "        # advances by the same number of accepted tokens.\n"
    "        # NOT SAFE for probabilistic rejection (see PN363 docstring) —\n"
    "        # config.force_max_spec_tokens default is False to enforce opt-in.\n"
    "        self.force_max_spec_tokens = bool(\n"
    "            getattr(config, \"force_max_spec_tokens\", False)\n"
    "        )\n"
    "        self._pad_token_id = -1\n"
    "        self._pad_template: list[int] = []\n"
    "        if self.force_max_spec_tokens:\n"
    "            try:\n"
    "                from vllm.tokenizers import cached_tokenizer_from_config  # noqa: E501\n"
    "                _tok = cached_tokenizer_from_config(vllm_config.model_config)\n"
    "            except Exception:  # noqa: BLE001\n"
    "                _tok = None\n"
    "            if _tok is None or getattr(_tok, \"eos_token_id\", None) is None:\n"
    "                # Defensive: disable force_max if tokenizer / eos not available.\n"
    "                self.force_max_spec_tokens = False\n"
    "            else:\n"
    "                self._pad_token_id = int(_tok.eos_token_id)\n"
    "                self._pad_template = [self._pad_token_id] * self.num_speculative_tokens\n"
)


# ── Sub-patch (b): propose() — replace bare append with pad-aware append ──
#
# Anchor includes the closing `)` of speculate() and the bare append
# line so it is unique in the file (the file has exactly one call to
# self.suffix_cache.speculate followed by a `draft_token_ids.append`).
PN363_PROPOSE_OLD = (
    "            draft = self.suffix_cache.speculate(\n"
    "                req_id,\n"
    "                pattern,\n"
    "                max_spec_tokens=min(\n"
    "                    self.num_speculative_tokens, self.max_model_len - num_tokens - 1\n"
    "                ),\n"
    "                max_spec_factor=self.max_spec_factor,\n"
    "                min_token_prob=self.min_token_prob,\n"
    "            )\n"
    "\n"
    "            draft_token_ids.append(draft.token_ids)\n"
)
PN363_PROPOSE_NEW = (
    "            # [Genesis PN363 vendor of vllm#43114] capture max_spec_tokens\n"
    "            # for the per-request pad decision below.\n"
    "            _pn363_max_spec_tokens = min(\n"
    "                self.num_speculative_tokens, self.max_model_len - num_tokens - 1\n"
    "            )\n"
    "            draft = self.suffix_cache.speculate(\n"
    "                req_id,\n"
    "                pattern,\n"
    "                max_spec_tokens=_pn363_max_spec_tokens,\n"
    "                max_spec_factor=self.max_spec_factor,\n"
    "                min_token_prob=self.min_token_prob,\n"
    "            )\n"
    "\n"
    "            # [Genesis PN363 vendor of vllm#43114] pad to num_speculative_tokens\n"
    "            # so all decode-mode requests have identical num_scheduled_tokens\n"
    "            # on the target side → uniform batch shape → FULL CUDA graph.\n"
    "            # Empty lists (partial prefills, max-len reached) are NEVER padded.\n"
    "            if (\n"
    "                self.force_max_spec_tokens\n"
    "                and _pn363_max_spec_tokens >= self.num_speculative_tokens\n"
    "                and draft.token_ids\n"
    "            ):\n"
    "                _pn363_draft = list(draft.token_ids)\n"
    "                _pn363_draft.extend(\n"
    "                    self._pad_template[len(_pn363_draft):]\n"
    "                )\n"
    "                draft_token_ids.append(_pn363_draft)\n"
    "            else:\n"
    "                draft_token_ids.append(draft.token_ids)\n"
)


def _env_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN363", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Apply PN363 force_max_spec_tokens plumbing to SuffixDecodingProposer.

    Returns
    -------
    (status, detail) where status ∈ {"applied", "skipped", "failed"}.
    """
    if _env_disabled():
        return "skipped", "PN363 disabled via GENESIS_DISABLE_PN363=1"

    target = resolve_vllm_file(_TARGET_REL)
    if target is None:
        return "skipped", f"PN363: target file {_TARGET_REL} not found"

    patcher = TextPatcher(
        patch_name="PN363 suffix_decoding.py — force_max_spec_tokens (vllm#43114)",
        target_file=str(target),
        marker=GENESIS_PN363_MARKER,
        sub_patches=[
            TextPatch(
                name="pn363_init_force_max_plumbing",
                anchor=PN363_INIT_OLD,
                replacement=PN363_INIT_NEW,
                required=True,
            ),
            TextPatch(
                name="pn363_propose_pad_short_drafts",
                anchor=PN363_PROPOSE_OLD,
                replacement=PN363_PROPOSE_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN363",
        ],
    )

    try:
        result, failure = patcher.apply()
    except Exception as e:  # noqa: BLE001
        return "failed", f"PN363 apply raised {e!r}"

    if result == TextPatchResult.FAILED:
        return "failed", f"PN363 FAILED — {failure.reason if failure else 'unknown'}"
    if result == TextPatchResult.SKIPPED:
        return "skipped", f"PN363 skipped — {failure.reason if failure else 'unknown'}"
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "PN363 idempotent (already applied)"

    return "applied", (
        "PN363 applied: SuffixDecodingProposer now honours "
        "force_max_spec_tokens in SpeculativeConfig. When enabled AND "
        "tokenizer.eos_token_id is available, short draft lists are "
        "padded so target-side batches stay uniform → FULL CUDA graph "
        "dispatch instead of PIECEWISE. Vendor of OPEN PR vllm#43114. "
        "Genesis PROD MTP path UNAFFECTED (suffix proposer not "
        "instantiated). Composes with P75 (suffix-decoding enable)."
    )


def is_applied() -> bool:
    """Return True iff the PN363 marker is present in the target file."""
    target = resolve_vllm_file(_TARGET_REL)
    if target is None:
        return False
    try:
        return GENESIS_PN363_MARKER in Path(str(target)).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
