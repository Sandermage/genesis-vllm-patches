# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N22 — Local argmax for TP draft (vllm#39419 backport).

RETIRED 2026-06-17 (lifecycle=retired) — superseded_by vllm#39419 (LocalArgmaxMixin),
merged at bd2d83ff, an ancestor of the 0.23.1 pin; the mixin is native on the
deployed pin (live-verified). applies_to capped <0.23.1rc1.dev101; the env flag in
prod YAMLs self-skips (default_off + drift marker).

Backport of [vllm#39419](https://github.com/vllm-project/vllm/pull/39419)
(EanWang; MERGED upstream 2026-06-10T07:59Z — after our current pin
0.22.1rc1.dev259+g303916e93). Adds `get_top_tokens()` plumbing methods
to Qwen3 + Qwen3-DFlash + Qwen3_5MTP model classes, enabling
vocab-parallel argmax without all-gathering full logits across TP ranks.

v2 extension (2026-06-10, dead-binding audit fix): the original v7.65
backport covered qwen3.py + qwen3_dflash.py only — but the live 35B MTP
drafter is class `Qwen3_5MTP` in `qwen3_5_mtp.py` (imports from
qwen3_5.py, NOT qwen3.py), so the proposer's local-argmax path never
engaged on 35B PROD: vanilla full-vocab all-gather argmax ran every
draft step over PCIe (TP=2, vocab 151k). The merged PR refactors the
method into `LocalArgmaxMixin` (interfaces.py) and mixes it into
Qwen3_5MTP; we backport the method body directly onto the class (text
patch can't safely rewrite class bases) with identical semantics,
including the draft_id_to_target_id (D2T) remap parity.

Merged-PR proposer delta vs our pin: logging-only (the D2T
warning/else-info block in `_maybe_share_lm_head` collapses to a single
info log because the mixin now handles D2T instead of bypassing). Our
pin already has the behavioral pieces: `use_local_argmax_reduction`
config (config/speculative.py:134, default False), the `_greedy_sample`
gate calling `self.model.get_top_tokens(hidden_states)`, and the
init-time ValueError when the draft model lacks the method. No
proposer-side vendoring needed.

================================================================
WHY THIS IS NEEDED
================================================================

Old draft path: each TP shard computes its slice of logits[batch,
vocab/tp_size], then all-gather assembles global_logits[batch, vocab],
then argmax. Communication = O(batch × vocab) per draft step. For
PARD-Qwen3-0.6B vocab=40960, batch=32, fp16 → 32×40960×2 = 2.5 MB per
step over PCIe Gen4 — ~1ms latency on dual 4090/A5000.

New draft path: each rank computes local argmax → (max_value, local_index).
Gather only pairs O(batch × 2 × tp_size) = ~1 KB. Reduce: global argmax
= argmax(max_values across ranks), global_index = local_index +
rank × shard_size. Bit-exact equivalent on identical shards.

Empirical (PR author): +9.4% to +30.6% throughput on TP=2 + draft model
(Qwen3-8B + DFlash, max-num-seqs=1, max-num-batched-tokens=8192).

================================================================
SCOPE
================================================================

Genesis backport covers our three production model classes:
- `qwen3.py` — main 27B / 35B-A3B
- `qwen3_dflash.py` — DFlash drafter
- `qwen3_5_mtp.py` — Qwen3_5MTP, the LIVE 35B MTP drafter (v2, 2026-06-10)

Llama (`llama.py`), Eagle3 (`llama_eagle3.py`) and DeepSeek
(`deepseek_eagle3.py`) parts of upstream PR are NOT backported here —
Genesis does not run those models in production. If a user needs them,
they can copy this pattern.

`LogitsProcessor.get_top_tokens()` itself already exists in our pin
(verified at line 106 of `vllm/model_executor/layers/logits_processor.py`
on dev259: local shard argmax -> all-gather (value, index) pairs ->
global argmax; padding masked via shard_indices.num_org_vocab_padding).
The PR is pure plumbing — wiring model classes through to that method.

Qwen3_5MTP lm_head note: with PN348 (shared backbone embed+lm_head) the
class allocates `PPMissingLayer()`, but the proposer's
`_maybe_share_lm_head` ALWAYS rebinds `self.model.lm_head` to the target
model's vocab-sharded ParallelLMHead for MTP models — the same object
`compute_logits()` uses — so `quant_method.apply` + `shard_indices` are
valid on every TP rank and local-argmax shard semantics match the
full-logits path bit-exactly.

================================================================
SAFETY MODEL
================================================================

- env: `GENESIS_ENABLE_PN22_LOCAL_ARGMAX_TP=1`
- default OFF; opt-in.
- Idempotent (marker check)
- Falls through cleanly if anchor missed (SKIPPED, not crash).
- NO callsite swap: this patch only adds the methods. The proposer
  in `vllm/v1/spec_decode/llm_base_proposer.py` only calls
  `get_top_tokens()` when `use_local_argmax_reduction: true` is set in
  `--speculative-config`. ORDER MATTERS: the model method must exist
  BEFORE flipping that config flag, otherwise the proposer raises
  ValueError at init ("does not implement get_top_tokens()").
- Auto-no-op once the pin includes the vllm#39419 merge: drift marker
  "LocalArgmaxMixin" on all three patchers (post-merge files carry the
  mixin in the class bases, NOT a literal `def get_top_tokens(`; a
  plain-method override would drop the mixin's D2T remap, so we must
  self-skip rather than re-apply).

Author: backport for Genesis from EanWang's vllm#39419 (merged).
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.pn22_local_argmax_tp")

GENESIS_PN22_MARKER = "Genesis PN22 local argmax TP for spec-decode draft v7.65"

# ─── Sub-patch 1: qwen3.py ──────────────────────────────────────────
PN22_QWEN3_ANCHOR = (
    "    def compute_logits(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "    ) -> torch.Tensor | None:\n"
    "        logits = self.logits_processor(self.lm_head, hidden_states)\n"
    "        return logits\n"
)

PN22_QWEN3_REPLACEMENT = (
    "    def compute_logits(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "    ) -> torch.Tensor | None:\n"
    "        logits = self.logits_processor(self.lm_head, hidden_states)\n"
    "        return logits\n"
    "\n"
    "    def get_top_tokens(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "    ) -> torch.Tensor:\n"
    "        # [Genesis PN22] vllm#39419 backport — vocab-parallel argmax,\n"
    "        # avoids all-gather of full logits across TP ranks.\n"
    "        # Wins +9-30% TPS on TP>=2 with draft models (DFlash/MTP).\n"
    "        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)\n"
)

# ─── Sub-patch 2: qwen3_dflash.py (DFlash draft) ─────────────────────
PN22_DFLASH_ANCHOR = (
    "        logits_new[:, targets] = logits\n"
    "        return logits_new\n"
    "\n"
    "    def precompute_and_store_context_kv(\n"
)

PN22_DFLASH_REPLACEMENT = (
    "        logits_new[:, targets] = logits\n"
    "        return logits_new\n"
    "\n"
    "    def get_top_tokens(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "    ) -> torch.Tensor:\n"
    "        # [Genesis PN22] vllm#39419 backport — vocab-parallel argmax for DFlash\n"
    "        # draft. Falls back to full logits when draft_id_to_target_id remap\n"
    "        # is active (draft predicts over draft_vocab_size, target expects\n"
    "        # target vocab ids — remap can't be done on local indices).\n"
    "        if self.draft_id_to_target_id is not None:\n"
    "            return self.compute_logits(hidden_states).argmax(dim=-1)\n"
    "        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)\n"
    "\n"
    "    def precompute_and_store_context_kv(\n"
)

# ─── Sub-patch 3: qwen3_5_mtp.py (Qwen3_5MTP — live 35B MTP drafter) ──
# v2 extension 2026-06-10. Anchor verified on live pin
# 0.22.1rc1.dev259+g303916e93 (container vllm-qwen3.6-35b-balanced-k3):
# count=1, get_top_tokens absent, LocalArgmaxMixin absent.
PN22_QWEN3_5_MTP_ANCHOR = (
    "    def compute_logits(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "        spec_step_idx: int = 0,\n"
    "    ) -> torch.Tensor | None:\n"
    "        return self.logits_processor(self.lm_head, hidden_states)\n"
)

PN22_QWEN3_5_MTP_REPLACEMENT = (
    "    def compute_logits(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "        spec_step_idx: int = 0,\n"
    "    ) -> torch.Tensor | None:\n"
    "        return self.logits_processor(self.lm_head, hidden_states)\n"
    "\n"
    "    def get_top_tokens(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "    ) -> torch.Tensor:\n"
    "        # [Genesis PN22] vllm#39419 backport (MERGED upstream 2026-06-10\n"
    "        # as LocalArgmaxMixin) — vocab-parallel argmax for the Qwen3_5MTP\n"
    "        # drafter. Each TP rank takes a local argmax over its lm_head\n"
    "        # shard; only (value, index) pairs are all-gathered:\n"
    "        # O(batch * 2 * tp_size) vs O(batch * vocab_size) full-logits\n"
    "        # all-gather per draft step. self.lm_head is the target model's\n"
    "        # vocab-sharded ParallelLMHead (rebound by the proposer's\n"
    "        # _maybe_share_lm_head for MTP models), so shard semantics match\n"
    "        # compute_logits() exactly.\n"
    "        top = self.logits_processor.get_top_tokens(self.lm_head, hidden_states)\n"
    "        # D2T parity with upstream LocalArgmaxMixin: Qwen3_5MTP carries\n"
    "        # no draft_id_to_target_id today; the guard keeps results correct\n"
    "        # if a pruned-vocab drafter variant ever adds the remap table.\n"
    "        d2t = getattr(self, \"draft_id_to_target_id\", None)\n"
    "        if d2t is not None:\n"
    "            top = top + d2t[top]\n"
    "        return top\n"
)


def build_qwen3_5_mtp_patcher(target_file: str) -> TextPatcher:
    """Build the qwen3_5_mtp.py TextPatcher (factored out for tests).

    Drift markers are scoped to qwen3_5_mtp.py content only:
    - "LocalArgmaxMixin": post-merge pins mix the method in via the class
      bases (no literal method definition in the file) — self-skip so the
      plain-method override never shadows the mixin's D2T handling.
    - "def get_top_tokens(": guards a future upstream shape that defines
      the method directly in the file.
    """
    return TextPatcher(
        patch_name="PN22 qwen3_5_mtp.py — get_top_tokens (vllm#39419 merged)",
        target_file=target_file,
        marker=GENESIS_PN22_MARKER + " (qwen3_5_mtp)",
        sub_patches=[
            TextPatch(
                name="pn22_qwen3_5_mtp_get_top_tokens",
                anchor=PN22_QWEN3_5_MTP_ANCHOR,
                replacement=PN22_QWEN3_5_MTP_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "LocalArgmaxMixin",
            "def get_top_tokens(",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN22 — vocab-parallel argmax plumbing (vllm#39419)."""
    from sndr.dispatcher import should_apply, log_decision
    decision, reason = should_apply("PN22")
    log_decision("PN22", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    # Apply qwen3.py first (always present). Then qwen3_dflash.py (DFlash optional).
    qwen3 = resolve_vllm_file("model_executor/models/qwen3.py")
    if qwen3 is None or not os.path.isfile(str(qwen3)):
        return "skipped", "qwen3.py not found"

    patcher_qwen3 = TextPatcher(
        patch_name="PN22 qwen3.py — get_top_tokens (vllm#39419)",
        target_file=str(qwen3),
        marker=GENESIS_PN22_MARKER,
        sub_patches=[
            TextPatch(
                name="pn22_qwen3_get_top_tokens",
                anchor=PN22_QWEN3_ANCHOR,
                replacement=PN22_QWEN3_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN22]",
            "def get_top_tokens(",
            # Post-merge pins (vllm#39419 merged 2026-06-10) provide the
            # method via LocalArgmaxMixin in the class bases — a plain
            # method override would drop the mixin's D2T remap. Self-skip.
            "LocalArgmaxMixin",
        ],
    )
    # Audit G-POST-03 fix 2026-05-05 (genesis_post_fix_rescan_audit):
    # SKIPPED was being masked as final "applied" — surface it honestly.
    r1, f1 = patcher_qwen3.apply()
    if r1 == TextPatchResult.SKIPPED:
        _r = f1.reason if f1 else "anchor drift / not eligible"
        _d = f" ({f1.detail})" if (f1 and f1.detail) else ""
        return "skipped", f"qwen3.py: {_r}{_d}"
    if r1 == TextPatchResult.FAILED:
        return "failed", f"qwen3.py: {f1.reason if f1 else 'unknown'}"

    # Now qwen3_dflash.py (only present if DFlash supported)
    dflash = resolve_vllm_file("model_executor/models/qwen3_dflash.py")
    if dflash is not None and os.path.isfile(str(dflash)):
        patcher_dflash = TextPatcher(
            patch_name="PN22 qwen3_dflash.py — get_top_tokens (vllm#39419)",
            target_file=str(dflash),
            marker=GENESIS_PN22_MARKER + " (dflash)",
            sub_patches=[
                TextPatch(
                    name="pn22_dflash_get_top_tokens",
                    anchor=PN22_DFLASH_ANCHOR,
                    replacement=PN22_DFLASH_REPLACEMENT,
                    required=True,
                ),
            ],
            upstream_drift_markers=[
                "[Genesis PN22]",
                "def get_top_tokens(",
            ],
        )
        r2, f2 = patcher_dflash.apply()
        if r2 == TextPatchResult.SKIPPED:
            _r = f2.reason if f2 else "anchor drift / not eligible"
            _d = f" ({f2.detail})" if (f2 and f2.detail) else ""
            return "skipped", (
                f"qwen3_dflash.py: {_r}{_d} (qwen3.py applied, but DFlash "
                "subpatch skipped — re-apply needed for matching pair)"
            )
        if r2 == TextPatchResult.FAILED:
            return "failed", f"qwen3_dflash.py: {f2.reason if f2 else 'unknown'}"

    # Now qwen3_5_mtp.py (Qwen3_5MTP — the LIVE 35B MTP drafter; v2
    # extension 2026-06-10). Older pins predate the Qwen3.5/3.6 family
    # and lack the file — skip silently there.
    mtp = resolve_vllm_file("model_executor/models/qwen3_5_mtp.py")
    if mtp is not None and os.path.isfile(str(mtp)):
        patcher_mtp = build_qwen3_5_mtp_patcher(str(mtp))
        r3, f3 = patcher_mtp.apply()
        if r3 == TextPatchResult.SKIPPED:
            _r = f3.reason if f3 else "anchor drift / not eligible"
            _d = f" ({f3.detail})" if (f3 and f3.detail) else ""
            return "skipped", (
                f"qwen3_5_mtp.py: {_r}{_d} (qwen3.py applied, but the "
                "Qwen3_5MTP subpatch skipped — the 35B MTP drafter would "
                "lack get_top_tokens(); do NOT enable "
                "use_local_argmax_reduction until resolved)"
            )
        if r3 == TextPatchResult.FAILED:
            return "failed", f"qwen3_5_mtp.py: {f3.reason if f3 else 'unknown'}"

    return "applied", (
        "PN22 applied: get_top_tokens() added to qwen3.py + qwen3_dflash.py "
        "+ qwen3_5_mtp.py (vllm#39419 backport, merged upstream 2026-06-10). "
        "Enables vocab-parallel argmax in spec-decode draft path once "
        "use_local_argmax_reduction is set in --speculative-config; "
        "+9-30% TPS on TP>=2 per PR author."
    )


def is_applied() -> bool:
    qwen3 = resolve_vllm_file("model_executor/models/qwen3.py")
    if qwen3 is None:
        return False
    try:
        with open(str(qwen3)) as f:
            if GENESIS_PN22_MARKER not in f.read():
                return False
    except OSError:
        return False
    # v2: when the pin ships qwen3_5_mtp.py, the Qwen3_5MTP subpatch
    # marker must be present too — otherwise the 35B drafter binding is
    # still dead and reporting "applied" would mask it.
    mtp = resolve_vllm_file("model_executor/models/qwen3_5_mtp.py")
    if mtp is None or not os.path.isfile(str(mtp)):
        return True
    try:
        with open(str(mtp)) as f:
            return (GENESIS_PN22_MARKER + " (qwen3_5_mtp)") in f.read()
    except OSError:
        return False
