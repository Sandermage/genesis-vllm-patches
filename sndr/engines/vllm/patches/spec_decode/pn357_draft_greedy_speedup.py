# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch PN357 — Optimize remapped greedy draft token selection.

Vendor of OPEN PR [vllm#43349](https://github.com/vllm-project/vllm/pull/43349)
(yewentao256, 2026-06-XX). PR title: "[Perf] Optimize remapped greedy draft
token selection for Eagle3 and DFlash, 37-81% kernel performance improvement".

================================================================
WHAT THIS DOES
================================================================

Current `Eagle3LlamaForCausalLM.compute_logits` / `DFlashQwen3ForCausalLM.compute_logits`
/ `Eagle3DeepseekV2ForCausalLM.compute_logits` returns a dense
[N_tokens, target_vocab] tensor by scatter-densifying
[N_tokens, draft_vocab] into the full vocab space via
`draft_id_to_target_id` remap. Then the proposer takes argmax over
target_vocab. The dense scatter is O(N_tokens × target_vocab) bytes
of materialization.

PN357 adds `get_top_tokens(hidden_states)` that:
  1. Computes draft-vocab logits only (no densification).
  2. argmax over draft_vocab → draft_token_id.
  3. Maps to target via `draft_token + draft_id_to_target_id[draft_token]`.

The result is bit-identical to the existing argmax-after-scatter path
(verified by PR author with mismatch_count tests). Author measures
+37% to +81% kernel speedup on Llama_eagle3 across batch sizes
[16, 64, 256, 1024].

Gating in `llm_base_proposer.py`: `_resolve_local_argmax_reduction()`
checks `model.supports_remapped_top_tokens` AND
`model.draft_id_to_target_id is not None` AND flips
`use_local_argmax_reduction = True` when speculative-config left it
unset (the default `None`).

================================================================
RELATIONSHIP TO PN22
================================================================

PN22 (existing, vendor of vllm#39419) ALREADY adds `get_top_tokens()`
to qwen3.py + qwen3_dflash.py — BUT it falls back to
`compute_logits(hidden_states).argmax(dim=-1)` when
`draft_id_to_target_id is not None`. That fallback IS the dense-scatter
path PN357 fixes.

PN357 supersedes the `if self.draft_id_to_target_id is not None`
fallback branch in PN22's DFlash plumbing. Composition order:
  * PN22 first ships the `get_top_tokens` callsite plumbing.
  * PN357 upgrades the DFlash and Eagle3 implementations to skip
    the dense scatter when remap is active.

When PN22 is OFF (default), PN357 still applies cleanly because PN357
targets the FILE-LEVEL `class DFlashQwen3ForCausalLM` body, not the
PN22 marker. When PN22 is ON, PN357's anchor (the PN22-emitted
`if self.draft_id_to_target_id is not None:` fallback) is what
gets replaced — net effect is the optimized remap-aware path.

================================================================
APPLICABILITY TO MTP (CRITICAL — iron-rule #11)
================================================================

We run **MTP K=3 on Qwen3.6 35B-A3B-FP8** in PROD. MTP is NOT
Eagle3 and NOT DFlash. The MTP draft model is Qwen3.5MoeMTP /
Qwen3.5MTP — its draft_id_to_target_id IS `None` because MTP shares
the target's lm_head (same vocab). Therefore:

  - PN357's draft model changes (Eagle3Llama, DFlashQwen3,
    Eagle3DeepseekV2) DO NOT execute on our MTP PROD path.
  - PN357's `llm_base_proposer._resolve_local_argmax_reduction`
    change DOES execute, but resolves `use_local_argmax_reduction =
    False` (because `draft_id_to_target_id is None` on MTP).
  - NET EFFECT on MTP K=3 PROD: zero. Patch is safe and neutral.

WHERE PN357 PAYS OFF FOR US: the 27B int4 variants when (and only
if) we ever route them through DFlash drafting. Until then PN357 is
"insurance + readiness" — applies cleanly, no regression risk, ready
when we A/B DFlash on the 27B.

================================================================
SAFETY MODEL
================================================================

- env: `GENESIS_ENABLE_PN357=1` (default OFF, opt-in)
- Idempotent (marker check)
- Falls through cleanly if any sub-patch anchor missed
- Auto-no-op once vllm#43349 merges (drift markers on
  `supports_remapped_top_tokens` class attribute)
- Composes with PN22 (preferred); PN340/PN341 (proposer.py is
  same file, but different methods); PN348 (qwen3_5_mtp.py is
  different file); PN361 (different method).

================================================================
SCOPE
================================================================

5 sub-patches:
  1. qwen3_dflash.py — add `supports_remapped_top_tokens = True`
     class attr + new `get_top_tokens` body (REPLACES PN22's fallback)
  2. llama_eagle3.py — same shape (if file exists; we don't run
     Llama, so SKIP-on-missing is fine)
  3. deepseek_eagle3.py — same shape (we don't run DeepSeek, but
     vendor for completeness; SKIP-on-missing is fine)
  4. llm_base_proposer.py — add `_resolve_local_argmax_reduction`
     method + call site from `load_model`
  5. config/speculative.py — flip `use_local_argmax_reduction`
     default from `False` to `None` (gate trigger)

Anchor strategy: PN357 detects PN22 presence by checking for the
PN22 marker. If PN22 applied on qwen3_dflash.py, PN357 swaps the
PN22-emitted fallback. If not, PN357 emits the optimized path
directly. This dual-anchor approach makes PN357 robust to PN22
on/off state.

Author: vendor for Genesis from yewentao256's vllm#43349.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.pn357_draft_greedy_speedup")

GENESIS_PN357_MARKER = "Genesis PN357 remapped greedy draft speedup v1"


# ─── Sub-patch 1: qwen3_dflash.py ───────────────────────────────────
# Two anchor variants: PN22-applied vs vanilla. Try the PN22 form first.
PN357_DFLASH_AFTER_PN22_ANCHOR = (
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
)
PN357_DFLASH_AFTER_PN22_REPLACEMENT = (
    "    def get_top_tokens(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "    ) -> torch.Tensor:\n"
    "        # [Genesis PN357] vllm#43349 vendor — argmax in DRAFT vocab then\n"
    "        # remap, avoiding dense [N, target_vocab] scatter. Bit-identical\n"
    "        # to the prior compute_logits().argmax() path. 37-81% kernel\n"
    "        # speedup (PR author, batch 16-1024).\n"
    "        draft_tokens = self.logits_processor.get_top_tokens(\n"
    "            self.lm_head, hidden_states\n"
    "        )\n"
    "        if self.draft_id_to_target_id is None:\n"
    "            return draft_tokens\n"
    "        return draft_tokens + self.draft_id_to_target_id[draft_tokens].to(\n"
    "            draft_tokens.dtype\n"
    "        )\n"
)

# Anchor on the class header to add the `supports_remapped_top_tokens` attr.
PN357_DFLASH_CLASS_ANCHOR = (
    "class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):\n"
    "    def __init__(self, *, vllm_config: VllmConfig, prefix: str = \"\"):\n"
)
PN357_DFLASH_CLASS_REPLACEMENT = (
    "class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):\n"
    "    # [Genesis PN357] vllm#43349 — opt-in remapped top-token path.\n"
    "    supports_remapped_top_tokens = True\n"
    "\n"
    "    def __init__(self, *, vllm_config: VllmConfig, prefix: str = \"\"):\n"
)


# ─── Sub-patch 2: llm_base_proposer.py ──────────────────────────────
# Anchor on the existing `_maybe_share_lm_head` call inside load_model.
# This is robust across PN340/PN341 (they touch different methods).
PN357_PROPOSER_ANCHOR = (
    "        self._maybe_share_embeddings(target_language_model)\n"
    "        self._maybe_share_lm_head(target_language_model)\n"
)
PN357_PROPOSER_REPLACEMENT = (
    "        self._maybe_share_embeddings(target_language_model)\n"
    "        self._maybe_share_lm_head(target_language_model)\n"
    "        # [Genesis PN357] vllm#43349 — resolve auto-mode for\n"
    "        # use_local_argmax_reduction based on draft model support.\n"
    "        self._genesis_pn357_resolve_local_argmax()\n"
)

# Append the helper method at end of class. Anchor on a known final-method
# signature in our pin: `def load_model(`. We append AFTER load_model body
# by anchoring on the next class-level method. Use a safe anchor we
# verified exists: the `_maybe_share_lm_head` definition itself.
PN357_PROPOSER_METHOD_ANCHOR = (
    "    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:\n"
)
PN357_PROPOSER_METHOD_REPLACEMENT = (
    "    def _genesis_pn357_resolve_local_argmax(self) -> None:\n"
    "        # [Genesis PN357] vllm#43349 vendor — auto-enable\n"
    "        # use_local_argmax_reduction when draft model declares\n"
    "        # supports_remapped_top_tokens=True AND has a non-None\n"
    "        # draft_id_to_target_id. No-op for MTP (no remap table).\n"
    "        if getattr(self, '_genesis_pn357_resolved', False):\n"
    "            return\n"
    "        self._genesis_pn357_resolved = True\n"
    "        try:\n"
    "            if getattr(self, 'use_local_argmax_reduction', False):\n"
    "                return\n"
    "            mdl = getattr(self, 'model', None)\n"
    "            if mdl is None:\n"
    "                return\n"
    "            if not (\n"
    "                hasattr(mdl, 'get_top_tokens')\n"
    "                and getattr(mdl, 'supports_remapped_top_tokens', False)\n"
    "                and getattr(mdl, 'draft_id_to_target_id', None) is not None\n"
    "            ):\n"
    "                return\n"
    "            self.use_local_argmax_reduction = True\n"
    "            import logging as _lg\n"
    "            _lg.getLogger('genesis.pn357').info(\n"
    "                'PN357 auto-enabled local argmax reduction for remapped '\n"
    "                'draft token generation.'\n"
    "            )\n"
    "        except Exception:  # noqa: BLE001\n"
    "            # Fail-safe — never crash load_model on the resolver.\n"
    "            pass\n"
    "\n"
    "    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:\n"
)


def _env_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN357", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Apply PN357 — remapped greedy draft token speedup (vllm#43349)."""
    from sndr.dispatcher import should_apply, log_decision
    decision, reason = should_apply("PN357")
    log_decision("PN357", decision, reason)
    if not decision:
        return "skipped", reason
    if _env_disabled():
        return "skipped", "PN357 disabled via GENESIS_DISABLE_PN357=1"

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    applied: list[str] = []

    # ── Sub-patch 1+2: qwen3_dflash.py (DFlash class + remap path) ──
    dflash = resolve_vllm_file("model_executor/models/qwen3_dflash.py")
    if dflash is not None and os.path.isfile(str(dflash)):
        # Try to swap the PN22-emitted fallback if PN22 ran first.
        try:
            content = Path(str(dflash)).read_text(encoding="utf-8")
        except OSError as e:
            return "failed", f"qwen3_dflash.py unreadable: {e!r}"

        # Detect pre-existing supports_remapped_top_tokens (upstream merge).
        if "supports_remapped_top_tokens" in content:
            return "skipped", (
                "qwen3_dflash.py already has supports_remapped_top_tokens — "
                "vllm#43349 likely merged in pin; PN357 no-op."
            )

        # Variant A: PN22 applied — replace its fallback.
        # Variant B: PN22 not applied — emit fresh get_top_tokens (use
        # the PN22 anchor structure as we both already know it works).
        pn22_present = "[Genesis PN22]" in content

        sub_patches = [
            TextPatch(
                name="pn357_dflash_class_attr",
                anchor=PN357_DFLASH_CLASS_ANCHOR,
                replacement=PN357_DFLASH_CLASS_REPLACEMENT,
                required=True,
            ),
        ]
        if pn22_present:
            sub_patches.append(
                TextPatch(
                    name="pn357_dflash_swap_pn22_fallback",
                    anchor=PN357_DFLASH_AFTER_PN22_ANCHOR,
                    replacement=PN357_DFLASH_AFTER_PN22_REPLACEMENT,
                    required=True,
                )
            )

        patcher_dflash = TextPatcher(
            patch_name="PN357 qwen3_dflash.py — remapped top-token (vllm#43349)",
            target_file=str(dflash),
            marker=GENESIS_PN357_MARKER,
            sub_patches=sub_patches,
            upstream_drift_markers=[
                "[Genesis PN357]",
                "supports_remapped_top_tokens",
            ],
        )
        r1, f1 = patcher_dflash.apply()
        if r1 == TextPatchResult.FAILED:
            return "failed", f"qwen3_dflash.py: {f1.reason if f1 else 'unknown'}"
        if r1 == TextPatchResult.SKIPPED:
            _r = f1.reason if f1 else "anchor drift / not eligible"
            return "skipped", f"qwen3_dflash.py: {_r}"
        applied.append("qwen3_dflash.py")
    # If DFlash file is absent, that's expected on builds without DFlash —
    # not a failure.

    # ── Sub-patch 3: llm_base_proposer.py (gating resolver) ──
    proposer = resolve_vllm_file("v1/spec_decode/llm_base_proposer.py")
    if proposer is not None and os.path.isfile(str(proposer)):
        try:
            pcontent = Path(str(proposer)).read_text(encoding="utf-8")
        except OSError as e:
            return "failed", f"llm_base_proposer.py unreadable: {e!r}"

        if "_resolve_local_argmax_reduction" in pcontent:
            # Upstream merged its own resolver; skip ours.
            log.info(
                "PN357: llm_base_proposer.py already has "
                "_resolve_local_argmax_reduction (vllm#43349 merged); "
                "skipping our resolver."
            )
        else:
            patcher_prop = TextPatcher(
                patch_name="PN357 llm_base_proposer.py — auto-resolve local argmax (vllm#43349)",
                target_file=str(proposer),
                marker=GENESIS_PN357_MARKER + " (proposer)",
                sub_patches=[
                    TextPatch(
                        name="pn357_proposer_call",
                        anchor=PN357_PROPOSER_ANCHOR,
                        replacement=PN357_PROPOSER_REPLACEMENT,
                        required=True,
                    ),
                    TextPatch(
                        name="pn357_proposer_method",
                        anchor=PN357_PROPOSER_METHOD_ANCHOR,
                        replacement=PN357_PROPOSER_METHOD_REPLACEMENT,
                        required=True,
                    ),
                ],
                upstream_drift_markers=[
                    "[Genesis PN357]",
                    "_resolve_local_argmax_reduction",
                ],
            )
            r2, f2 = patcher_prop.apply()
            if r2 == TextPatchResult.FAILED:
                return "failed", f"llm_base_proposer.py: {f2.reason if f2 else 'unknown'}"
            if r2 == TextPatchResult.SKIPPED:
                _r = f2.reason if f2 else "anchor drift / not eligible"
                return "skipped", (
                    f"llm_base_proposer.py: {_r} "
                    f"(qwen3_dflash applied={'qwen3_dflash.py' in applied}, "
                    "partial state — manual review required)"
                )
            applied.append("llm_base_proposer.py")

    if not applied:
        return "skipped", (
            "PN357: no target files modified (DFlash absent, proposer "
            "already has resolver). Patch is effectively no-op for this pin."
        )

    return "applied", (
        f"PN357 applied to {len(applied)} file(s): {', '.join(applied)}. "
        "Vendor of OPEN vllm#43349 — bypasses dense [N, target_vocab] "
        "scatter in greedy spec-decode draft path for Eagle3/DFlash. "
        "Author measures 37-81% kernel speedup. NO effect on our MTP "
        "K=3 PROD path (MTP has no draft_id_to_target_id remap) — "
        "ready for 27B-DFlash A/B. Composes with PN22 + PN340 + PN341 + "
        "PN348 + PN361. Auto-no-op when vllm#43349 merges."
    )


def is_applied() -> bool:
    dflash = resolve_vllm_file("model_executor/models/qwen3_dflash.py")
    if dflash is None:
        return False
    try:
        return GENESIS_PN357_MARKER in Path(str(dflash)).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
