# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN90 — Probabilistic draft rejection (vllm#40269 backport).

Wave 3.1 (audit closure 2026-05-09 / production roadmap §16.1).

================================================================
WHAT THIS PATCH DOES
================================================================

vLLM's spec-decode rejection sampler (`v1/sample/rejection_sampler.py`)
takes `draft_probs` as one of its arguments — the probability
distribution the drafter assigned to each speculated token. With
`draft_probs` populated, the verifier can use the **probabilistic**
acceptance rule:

    accept_prob = min(1, target_prob / draft_prob)

instead of the simpler argmax-or-bonus rule used when `draft_probs`
is None. The probabilistic rule is mathematically aligned with the
rejection-sampling theory and accepts more tokens on average — net
+0.5-2% acceptance rate on typical workloads.

**The bug in dev93**: `gpu_model_runner.py:3416` passes literal
`None` for `draft_probs`. The MTP/Eagle/DFlash drafters in
`llm_base_proposer.py` discard logits inside `_greedy_sample()`
(line ~414) — they call `compute_logits(hidden_states).argmax(...)`,
losing the probability distribution.

================================================================
PN90's FIX
================================================================

Three text-patches to `llm_base_proposer.py`:

1. **`_greedy_sample` wrapper** (anchor at line 414-418):
   When `GENESIS_ENABLE_PN90_PROBABILISTIC_DRAFT=1`, also softmax
   the logits and append to `self._pn90_step_probs_buf` before
   returning argmax. No-op when env unset.

2. **`propose()` entry** (anchor at line 439):
   Reset `self._pn90_step_probs_buf = []` at the start of each
   propose call so per-batch buffers don't leak.

3. **`propose()` exit** (anchor at line 633):
   After `torch.stack(draft_token_ids_list, dim=1)`, also stack
   the per-step probs and reshape to the rejection_sampler's
   expected `[total_draft_tokens, vocab_size]` 2D layout.
   Result stored as `self._pn90_draft_probs` for the runner.

One text-patch to `gpu_model_runner.py`:

4. **rejection_sampler call site** (anchor at line ~3414):
   Read `getattr(self.drafter, "_pn90_draft_probs", None)` instead
   of literal `None`. Backwards-compatible: when env is unset, the
   drafter never sets the attribute, getattr returns None → behavior
   is identical to upstream.

================================================================
SAFETY MODEL
================================================================

- Default OFF. Opt in via `GENESIS_ENABLE_PN90_PROBABILISTIC_DRAFT=1`.
- When OFF: `_greedy_sample` skips softmax (zero overhead), buffer
  stays empty, `_pn90_draft_probs` is None → rejection_sampler gets
  None just like upstream → bit-identical behavior.
- When ON: ~per-token softmax cost (~100µs on A5000 for vocab=248320).
  K iterations × max_num_seqs => ~1ms extra latency per spec-decode
  cycle. Worth it if acceptance gain > 1%.
- Compatible with: MTP, Eagle, Eagle3, DFlash (all use llm_base_proposer).
- NOT applicable: ngram method (uses different proposer that doesn't
  go through `_greedy_sample`).
- Falls back gracefully when `use_local_argmax_reduction=True`
  (line 415 of _greedy_sample): no logits in that branch, so probs
  buffer stays empty → no `_pn90_draft_probs` → rejection_sampler
  gets None.

================================================================
EXPECTED OUTCOMES
================================================================

Best case: +1-2% TPS via better acceptance on borderline cases
where the standard rule would reject but probabilistic rule
accepts.

Worst case: +0.5% acceptance gain offset by ~1ms softmax cost
per step → net within noise. Defensive enable.

Status: opt-in via env, default OFF until per-PROD A/B bench.

Author: Sandermage(Sander)-Barzov Aleksandr.
Backport reference: vllm#40269 (OPEN as of 2026-05-09).
"""
from __future__ import annotations

import logging
import os

from vllm.sndr_core.detection.guards import resolve_vllm_file, vllm_install_root
from vllm.sndr_core.core import (
    MultiFilePatchTransaction,
    TextPatch,
    TextPatcher,
)

log = logging.getLogger("genesis.wiring.pn90_probabilistic_draft_rejection")

PN90_MARKER_PROPOSER = (
    "Genesis PN90 probabilistic draft rejection (vllm#40269) v1.0_v11.3.0_hotpath — proposer"
)
PN90_MARKER_RUNNER = (
    "Genesis PN90 probabilistic draft rejection (vllm#40269) v1.0_v11.3.0_hotpath — runner"
)

# Drift markers — when upstream lands its own probabilistic-draft
# propagation, these symbols would appear and PN90 must self-retire.
#
# 2026-05-15 update: vllm#40269 merged 2026-05-14 with a different
# implementation than this backport assumed. Upstream lands the feature as:
#   - `self._enable_probabilistic_draft_probs` (gate, set from
#      speculative_config.draft_sample_method == "probabilistic")
#   - `self._last_draft_probs` (captured tensor)
#   - `take_last_draft_probs()` accessor (returns probs to gpu_model_runner)
# We add these as drift markers so PN90 auto-skips on pins that contain
# the upstream feature (operator gets the native path; we don't double-
# patch).
_PROPOSER_DRIFT_MARKERS = (
    "_pn90_step_probs_buf",                  # our marker
    "_pn90_draft_probs",                     # our marker
    # Upstream-native symbols (vllm#40269 merged 2026-05-14):
    "_enable_probabilistic_draft_probs",
    "_last_draft_probs",
    "take_last_draft_probs",
    # Older speculative naming we considered before merge:
    "draft_token_probs",
    "draft_logprobs",
    "self.drafter._draft_probs",
)
_RUNNER_DRIFT_MARKERS = (
    "_pn90_draft_probs",                     # our marker
    # Upstream-side: if line 3416 stops passing literal None, PN90 retires.
    "draft_probs=getattr",
    "draft_probs=self.drafter",
    # Upstream-native (vllm#40269): runner calls .take_last_draft_probs()
    "take_last_draft_probs",
)


# ─── Sub-patch 1: _greedy_sample wrapper ─────────────────────────────────


PN90_GREEDY_SAMPLE_OLD = (
    "    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:\n"
    '        """Greedy-sample draft tokens from hidden states."""\n'
    "        if self.use_local_argmax_reduction:\n"
    "            return self.model.get_top_tokens(hidden_states)\n"
    "        return self.model.compute_logits(hidden_states).argmax(dim=-1)\n"
)
PN90_GREEDY_SAMPLE_NEW = (
    "    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:\n"
    '        """Greedy-sample draft tokens from hidden states (Genesis PN90 v2)."""\n'
    "        if self.use_local_argmax_reduction:\n"
    "            return self.model.get_top_tokens(hidden_states)\n"
    "        # [Genesis PN90 v2 — hot-path optimized] env state is boot-fixed;\n"
    "        # cache the enabled flag in this module's globals so per-draft-step\n"
    "        # env-var lookup + .strip().lower() + 4-tuple membership check\n"
    "        # (~250ns) is eliminated. Spec-decode K=3-4 → 3-4 calls per request.\n"
    "        _pn90_enabled = globals().get('_GENESIS_PN90_enabled')\n"
    "        if _pn90_enabled is None:\n"
    "            import os as _pn90_os\n"
    "            _pn90_enabled = _pn90_os.environ.get(\n"
    "                'GENESIS_ENABLE_PN90_PROBABILISTIC_DRAFT', ''\n"
    "            ).strip().lower() in ('1', 'true', 'yes', 'on')\n"
    "            globals()['_GENESIS_PN90_enabled'] = _pn90_enabled\n"
    "        _pn90_logits = self.model.compute_logits(hidden_states)\n"
    "        if _pn90_enabled:\n"
    "            if not hasattr(self, '_pn90_step_probs_buf'):\n"
    "                self._pn90_step_probs_buf = []\n"
    "            self._pn90_step_probs_buf.append(\n"
    "                _pn90_logits.softmax(dim=-1, dtype=torch.float32)\n"
    "            )\n"
    "        return _pn90_logits.argmax(dim=-1)\n"
)


# ─── Sub-patch 2: propose() entry — reset buffer ─────────────────────────


PN90_PROPOSE_ENTRY_OLD = (
    "    ) -> torch.Tensor:\n"
    "        batch_size = common_attn_metadata.batch_size()\n"
)
PN90_PROPOSE_ENTRY_NEW = (
    "    ) -> torch.Tensor:\n"
    "        # [Genesis PN90 vllm#40269] reset per-call probs buffer so\n"
    "        # buffers from prior batches don't leak into the next.\n"
    "        self._pn90_step_probs_buf = []\n"
    "        self._pn90_draft_probs = None\n"
    "        batch_size = common_attn_metadata.batch_size()\n"
)


# ─── Sub-patch 3: propose() exit — stack + reshape probs ─────────────────


PN90_PROPOSE_EXIT_OLD = (
    "        # [batch_size, num_speculative_tokens]\n"
    "        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)\n"
    "        return draft_token_ids\n"
)
PN90_PROPOSE_EXIT_NEW = (
    "        # [batch_size, num_speculative_tokens]\n"
    "        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)\n"
    "        # [Genesis PN90 vllm#40269] stack per-step probs into the\n"
    "        # rejection_sampler-expected layout: [total_draft_tokens, vocab].\n"
    "        # Order: req0_tok0, req0_tok1, ..., req0_tokK, req1_tok0, ...\n"
    "        # which aligns with cu_num_draft_tokens flattening of token_ids.\n"
    "        if self._pn90_step_probs_buf:\n"
    "            _pn90_stacked = torch.stack(self._pn90_step_probs_buf, dim=1)\n"
    "            _pn90_vocab = _pn90_stacked.shape[-1]\n"
    "            self._pn90_draft_probs = (\n"
    "                _pn90_stacked.contiguous().view(-1, _pn90_vocab)\n"
    "            )\n"
    "        return draft_token_ids\n"
)


# ─── Sub-patch 4: gpu_model_runner.py — feed draft_probs to rejection_sampler ───


PN90_RUNNER_OLD = (
    "        sampler_output = self.rejection_sampler(\n"
    "            spec_decode_metadata,\n"
    "            None,  # draft_probs\n"
    "            logits,\n"
    "            sampling_metadata,\n"
    "        )\n"
    "        return sampler_output\n"
)
PN90_RUNNER_NEW = (
    "        # [Genesis PN90 vllm#40269] propagate draft_probs from drafter\n"
    "        # when GENESIS_ENABLE_PN90_PROBABILISTIC_DRAFT=1. Returns None\n"
    "        # when env unset, drafter is ngram, or use_local_argmax was hit.\n"
    "        _pn90_draft_probs = (\n"
    "            getattr(self.drafter, '_pn90_draft_probs', None)\n"
    "            if hasattr(self, 'drafter') and self.drafter is not None\n"
    "            else None\n"
    "        )\n"
    "        sampler_output = self.rejection_sampler(\n"
    "            spec_decode_metadata,\n"
    "            _pn90_draft_probs,\n"
    "            logits,\n"
    "            sampling_metadata,\n"
    "        )\n"
    "        return sampler_output\n"
)


# ─── Patcher factories ───────────────────────────────────────────────────


def _make_patcher_proposer() -> TextPatcher | None:
    target = resolve_vllm_file("v1/spec_decode/llm_base_proposer.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN90 v1/spec_decode/llm_base_proposer.py — probabilistic draft "
            "rejection (vllm#40269)"
        ),
        target_file=str(target),
        marker=PN90_MARKER_PROPOSER,
        sub_patches=[
            TextPatch(
                name="pn90_greedy_sample_capture_probs",
                anchor=PN90_GREEDY_SAMPLE_OLD,
                replacement=PN90_GREEDY_SAMPLE_NEW,
                required=True,
            ),
            TextPatch(
                name="pn90_propose_reset_buf",
                anchor=PN90_PROPOSE_ENTRY_OLD,
                replacement=PN90_PROPOSE_ENTRY_NEW,
                required=True,
            ),
            TextPatch(
                name="pn90_propose_stack_probs",
                anchor=PN90_PROPOSE_EXIT_OLD,
                replacement=PN90_PROPOSE_EXIT_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=list(_PROPOSER_DRIFT_MARKERS),
        patch_id="PN90",
    )


def _make_patcher_runner() -> TextPatcher | None:
    target = resolve_vllm_file("v1/worker/gpu_model_runner.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN90 v1/worker/gpu_model_runner.py — feed draft_probs to "
            "rejection_sampler (vllm#40269)"
        ),
        target_file=str(target),
        marker=PN90_MARKER_RUNNER,
        sub_patches=[
            TextPatch(
                name="pn90_runner_feed_draft_probs",
                anchor=PN90_RUNNER_OLD,
                replacement=PN90_RUNNER_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=list(_RUNNER_DRIFT_MARKERS),
        patch_id="PN90",
    )


# ─── apply() — atomic via MultiFilePatchTransaction ──────────────────────


def apply() -> tuple[str, str]:
    """Apply PN90 atomically across the proposer + runner files.

    Both patches must apply or none — partial application would leave
    a non-functional state where the proposer caches probs but the
    runner doesn't read them (or vice versa).
    """
    from vllm.sndr_core.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN90")
    log_decision("PN90", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    # Pre-flight: detect if upstream already shipped this. If
    # `gpu_model_runner.py` no longer contains the literal `None,
    # # draft_probs` site, upstream restructured rejection_sampler
    # invocation → PN90 retires.
    runner_target = resolve_vllm_file("v1/worker/gpu_model_runner.py")
    if runner_target is not None and os.path.isfile(runner_target):
        try:
            with open(runner_target, encoding="utf-8") as f:
                runner_content = f.read()
            if "None,  # draft_probs" not in runner_content:
                if PN90_MARKER_RUNNER not in runner_content:
                    log.info(
                        "[PN90] gpu_model_runner.py no longer has the literal "
                        "`None,  # draft_probs` line — upstream may have "
                        "merged a different draft_probs propagation; "
                        "PN90 self-retires."
                    )
                    return (
                        "skipped",
                        "upstream_merged — `None,  # draft_probs` literal "
                        "absent from gpu_model_runner.py (upstream "
                        "restructured rejection_sampler call site)",
                    )
        except OSError:
            pass

    proposer_patcher = _make_patcher_proposer()
    runner_patcher = _make_patcher_runner()
    if proposer_patcher is None:
        return "skipped", "v1/spec_decode/llm_base_proposer.py not resolvable"
    if runner_patcher is None:
        return "skipped", "v1/worker/gpu_model_runner.py not resolvable"

    txn = MultiFilePatchTransaction(
        [proposer_patcher, runner_patcher],
        name="PN90",
    )
    return txn.apply_or_skip()
