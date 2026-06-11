# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN50 — SGLang #21019 GDN projection fusion backport.

Replaces the unfused split/reshape/cat/.contiguous() chain in the
Qwen3.5/3.6 contiguous projection branch of `gdn_linear_attn.py`
with the Genesis-ported Triton kernel `pn50_gdn_fused_proj`.

Affects only the `gqa_interleaved_layout=False` branch (Qwen3.5/3.6
contiguous-loaded weights). Qwen3-Next (interleaved layout) and the LoRA
path (`hasattr(in_proj_qkv)`) are unaffected.

Anchor stability
----------------
Anchor is the entire 9-line `else:` block of the Qwen3.5 branch in
`mamba/gdn_linear_attn.py`. Verified against pristine upstream + live
container — both match (see test_pn50_*.py).

Models affected (per Genesis 7-config matrix):
  * 27B Lorbus INT4 (TQ k8v4, FP8 short, FP8 long, NGRAM, DFlash) — APPLIES
  * 35B FP8 (PROD, DFlash) — DOES NOT APPLY (Qwen3MoE has no GDN layers)

Default OFF until live A/B prod-validates +TPS gain on at least one
27B config without numerical regression.

Author: Sandermage (Sander) Barzov Aleksandr backport.
Original Triton kernel: Yuan Luo (@yuan-luo), SGLang PR #21019.
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.pn50_gdn_fused_proj")

GENESIS_PN50_MARKER = "Genesis PN50 GDN fused proj v7.66 (SGLang#21019 backport)"


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN50_GDN_FUSED_PROJ", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# Upstream anchor — Qwen3.5/3.6 contiguous-projection branch.
#
# Note 2026-05-15: as of vllm dev371 (nightly bf610c2f) the `forward()`
# method uses 8/12-space indentation (one less nesting level than the
# original 12/16-space block this patch was written for). The earlier
# indentation was likely from an outdated upstream snapshot. The
# Qwen3.5 contiguous branch only matches in `forward()` — `forward_cpu()`
# has the same comment but lacks the trailing `.contiguous()` calls, so
# our anchor remains unique (single match in the file).
#
# 2026-06-09 re-anchor: PROD pin 0.22.1rc1.dev259+g303916e93 ships
# `qwen_gdn_linear_attn.py` (file rename via vllm#41126) with
# `b, a = self.split_ba(ba)` instead of `b, a = ba.chunk(2, dim=-1)`.
# Verified via container grep at lines 1040-1047. forward_cpu() at
# line 1149-1154 still uses the older `ba.chunk(2, dim=-1)` shape
# AND lacks the trailing `.contiguous()` pair — so anchoring on the
# full nine-line block including `b = b.contiguous()` /
# `a = a.contiguous()` keeps the match unique to forward() only.
ANCHOR_OLD = (
    "        else:\n"
    "            # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order\n"
    "            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size\n"
    "            z_size = self.value_dim // self.tp_size\n"
    "            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)\n"
    "            z = z.reshape(z.size(0), -1, self.head_v_dim)\n"
    "            b, a = self.split_ba(ba)\n"
    "            b = b.contiguous()\n"
    "            a = a.contiguous()"
)

ANCHOR_NEW = (
    "        else:\n"
    "            # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order\n"
    "            # [Genesis PN50 SGLang#21019] fused Triton kernel for\n"
    "            # split/reshape/cat/.contiguous(); replaces 5-6 launches +\n"
    "            # 2 explicit copies. Wrapper falls through to original\n"
    "            # PyTorch chain on any constraint violation (non-contig,\n"
    "            # non-pow2 head_dim, kernel failure, etc.) — strict no-regression.\n"
    "            from sndr.engines.vllm.kernels_legacy.pn50_gdn_fused_proj import (\n"
    "                fused_qkvzba_split_reshape_cat_contiguous as _pn50_fused,\n"
    "            )\n"
    "            _pn50_num_heads_qk = (self.key_dim // self.head_k_dim) // self.tp_size\n"
    "            _pn50_num_heads_v = (self.value_dim // self.head_v_dim) // self.tp_size\n"
    "            mixed_qkv, z, b, a = _pn50_fused(\n"
    "                mixed_qkvz, ba,\n"
    "                num_heads_qk=_pn50_num_heads_qk,\n"
    "                num_heads_v=_pn50_num_heads_v,\n"
    "                head_qk=self.head_k_dim,\n"
    "                head_v=self.head_v_dim,\n"
    "            )"
)


def _make_patcher() -> TextPatcher | None:
    # K.1.R.R.4 (2026-05-29): #41126 split mamba/gdn_linear_attn.py.
    # See K.1.R.R.1 / K.1.R.R.2A for the same fallback pattern on
    # P28 / P46 / P60 / P60b / PN11.
    target = (
        resolve_vllm_file("model_executor/layers/mamba/gdn_linear_attn.py")
        or resolve_vllm_file(
            "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
        )
    )
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN50 GDN fused proj (SGLang#21019)",
        target_file=str(target),
        marker=GENESIS_PN50_MARKER,
        sub_patches=[
            TextPatch(
                name="pn50_gdn_fused_proj",
                anchor=ANCHOR_OLD,
                replacement=ANCHOR_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            # Self-collision lint (triage plan §6 2026-06-11): former entry
            # "fused_qkvzba_split_reshape_cat_contiguous" is the SGLang
            # kernel name our own replacement calls — false
            # "upstream_merged" skip on residue. Residue coverage moves to
            # the sanctioned banner prefix below; a real upstream
            # equivalent-fusion merge is caught by required-anchor
            # mismatch (Layer 5) + pin-bump preflight deep-diff.
            "[Genesis PN50",
        ],
    )


def apply() -> tuple[str, str]:
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN50")
    log_decision("PN50", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "gdn_linear_attn.py not found"

    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return (
            "applied",
            "PN50 applied: GDN proj fusion active in Qwen3.5/3.6 contiguous "
            "branch; wrapper falls through to PyTorch on constraint violation",
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied (idempotent)"
    if result == TextPatchResult.SKIPPED:
        msg = failure.reason if failure else "anchor not found"
        return (
            "skipped",
            f"{msg} — likely upstream merged an equivalent fusion or "
            "anchor drifted (check gdn_linear_attn.py Qwen3.5 branch)",
        )
    return "failed", failure.reason if failure else "unknown failure"
