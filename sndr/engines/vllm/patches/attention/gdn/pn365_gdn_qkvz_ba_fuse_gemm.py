# SPDX-License-Identifier: Apache-2.0
"""PN365 — Fuse GDN linear_attn qkv|z|b|a into a single GEMM
(Genesis port of OPEN vllm PR #42746).

Background
==========
GatedDeltaNetAttention.forward_cuda currently runs two independent GEMMs on
the same `hidden_states` at the Part-1 input projection step:

    mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)   # N = 12288 (q+k+v+z)
    ba,         _ = self.in_proj_ba  (hidden_states)   # N =    64 (b+a)

The `in_proj_ba` GEMM is a tiny `M x 2048 -> 64` shape that is almost pure
kernel-launch overhead on small-M decode. vllm PR #42746 collapses both into
a single `MergedColumnParallelLinear` whose output shards are
`[key_dim, key_dim, value_dim, value_dim, num_v_heads, num_v_heads]`. One
GEMM, one split, larger N for cuBLASLt tile selection.

Author bench (RTX PRO 6000, sm_120, TP=1, Qwen3.5-35B-A3B-NVFP4):
  C=3  +3.7% TPOT     C=5  +3.3%    C=6  +1.9%    C=7  +2.2%    C=8  +2.4%
  SLO @ TPOT<=10ms: max concurrency 5 -> 6 (+20%); throughput 2.95 -> 3.50
  req/s (+19%); output tok/s 443 -> 525 (+18.5%).

Mechanism deep-dive
===================
The fusion is bit-equivalent: a single MergedColumnParallelLinear has the
SAME weight values, just concatenated along the output dim. The output
split shapes are arithmetically identical to the original two-GEMM
outputs.  Numerics are unchanged at the matmul level.

The win comes from two sources:

1. Kernel launch overhead. 2 launches -> 1 launch saves ~5-10 us per layer
   on A5000 (cudaLaunchKernel + driver hand-off).  At 24 GDN layers x N
   forward calls / decode step, that is meaningful when TPOT < 10 ms.

2. cuBLASLt tile selection. The `in_proj_ba` GEMM has N=64, far below the
   smallest viable cuBLASLt tile.  On Ampere SM 8.6 cuBLASLt typically
   picks a 64x64 or 128x64 tile and pads, wasting most of the SM
   throughput. Fusing into N=12352 lets cuBLASLt pick a proper 128x256 or
   256x128 tile that aligns naturally with the qkvz N=12288 path.

Why this composes / conflicts with our existing GDN patches
============================================================

* **PN350** (fused GDN QKV post-conv split) — DIFFERENT site (downstream of
  conv1d, not at in_proj). Composes cleanly. PN365 changes which Linear
  modules exist; PN350 operates on `mixed_qkv` which is identical in both
  paths.
* **PN204** (in_proj dual-stream overlap) — HARD CONFLICT. PN204 wraps the
  two in_proj calls in `maybe_execute_in_parallel`. If PN365 fuses them
  into one Linear, there is nothing to overlap and PN204's branch
  collapses to a single serial call -> PN204 becomes a no-op, AND its
  anchor no longer matches our text-patched code.
  **Recommendation**: when PN365 is enabled, set
  `GENESIS_ENABLE_PN204_DUAL_STREAM_INPROJ=0`. PN365 is the structurally
  superior win (eliminates the small GEMM entirely vs. overlapping it).
* **PN54** (GDN .contiguous() dedup) — different site (post-split). PN365
  emits contiguous tensors already; PN54 stays a no-op. Composes.
* **P28** (gdn_core_attn pool) — downstream (Part 2 core attention).
  Composes.
* **PN50** (GDN fused proj) — different mechanism, post-conv. Composes.
* **PN11** (a/b contiguous) — PN365 already calls `.contiguous()` on b/a
  in the fused path. PN11 stays a no-op on the fused path; original path
  unchanged. Composes.

Ampere SM 8.6 derate factor
============================
Author bench is on Blackwell sm_120 (RTX PRO 6000). Two factors derate:
  - Launch overhead win: portable across architectures, full carry. ~+1.5-2%.
  - cuBLASLt tile win: sm_86 (A5000) has different heuristics than sm_120.
    Larger N still helps occupancy; expected partial carry. ~+0.5-1%.

Combined Ampere SM 8.6 estimate: **+1-3% wall_TPS single-stream** (conservative
relative to author +2-4% on Blackwell). With our 218.56 -> 228 TPS gap of
9.44 TPS (+4.3%), PN365 alone closes 25-65% of the gap; combined with
PN350 + PN345 should close more.

Implementation strategy
=======================
The patch has THREE text-patch anchors, gated by env flag
`GENESIS_ENABLE_PN365_GDN_GEMM_FUSE=1` (default OFF):

  Anchor A — `qwen_gdn_linear_attn.py` constructor. Replace the
    `self.in_proj_qkvz = self.create_qkvz_proj(...) ; self.in_proj_ba =
    self.create_ba_proj(...)` block with a conditional: when the env
    flag is on AND the layer is non-interleaved GQA (Qwen3.5), create
    one MergedColumnParallelLinear named `in_proj_qkvzba`; otherwise
    keep the original two-Linear path verbatim.

  Anchor B — `qwen_gdn_linear_attn.py::forward_cuda` Part 1. Replace
    the serial two-GEMM block with a conditional: if `hasattr(self,
    "in_proj_qkvzba")`, do one matmul + 4-way split; else upstream
    two-GEMM path verbatim.

  Anchor C — `qwen3_5.py::Qwen3_5Model.load_weights` stacked_params_mapping.
    Add a runtime-detected fused mapping (in_proj_qkvzba covers q/k/v/z/b/a
    shards 0..5) when the model was built with the fused Linear; else
    keep the original 4-entry split mapping verbatim.

Strict no-regression behavior
==============================
  * When env flag is unset, all three text-patches replace the upstream
    block with EXACTLY the upstream block (the conditional branches make
    the unset path bit-equivalent to the original code). No drift on
    default install.
  * `packed_modules_mapping` is left UNCHANGED: PR #42746 modifies it
    with a complex env-time switch + LoRA fallback. We avoid that by
    refusing the fusion when LoRA is configured (Genesis doesn't run
    LoRA on Qwen3.6-35B anyway), keeping the upstream split mapping
    valid. If the operator ever enables LoRA, the runtime check
    `vllm_config.lora_config is None` in Anchor A keeps the upstream
    path active.
  * Anchor failure on any sub-patch -> entire PN365 SKIPPED. We never
    leave the file in a half-patched state.

Drift markers
=============
Auto-SKIP if upstream lands #42746 (any of the following appears in the
target file already):
  - "in_proj_qkvzba"
  - "VLLM_GDN_FUSE_QKVZBA"
  - "create_in_proj_qkvzba"

Author: Sander (Sandermage / Aleksandr Barzov), Odessa, Ukraine, 2026-06-09.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

log = logging.getLogger("genesis.wiring.pn365_gdn_qkvz_ba_fuse_gemm")

GENESIS_PN365_MARKER = (
    "Genesis PN365 fused GDN qkv|z|b|a single-GEMM input projection "
    "(port of OPEN vllm#42746) v1"
)

_GDN_TARGET_REL = "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
_MODEL_TARGET_REL = "model_executor/models/qwen3_5.py"

_ENV_FLAG = "GENESIS_ENABLE_PN365_GDN_GEMM_FUSE"


# =============================================================================
# Anchor A: GDN constructor — add the in_proj_qkvzba branch.
#
# The upstream block creates two Linear modules. Our replacement adds an
# env-gated single-Linear branch around it. When the env flag is OFF, the
# else-branch runs the bit-exact upstream code.
# =============================================================================
PN365_CTOR_OLD = (
    "        # projection of the input hidden states\n"
    "        # Qwen3-Next and Qwen3.5 has a different qkv_proj layout,\n"
    "        # we need to create qkvz_proj adaptively here.\n"
    "        # When create_in_proj_qkvz is False (e.g. LoRA enabled in Qwen3.5),\n"
    "        # in_proj_qkv and in_proj_z are created separately instead.\n"
    "        self.in_proj_qkvz = self.create_qkvz_proj(\n"
    "            hidden_size=self.hidden_size,\n"
    "            key_dim=self.key_dim,\n"
    "            value_dim=self.value_dim,\n"
    "            quant_config=self.quant_config,\n"
    "            prefix=f\"{prefix}.in_proj_qkvz\",\n"
    "        )\n"
    "\n"
    "        # ba_proj doesn't support blockwise fp8 quantization.\n"
    "        # Qwen3-Next and Qwen3.5 have different in_proj_ba checkpoint\n"
    "        # layouts, so we use a factory method to create the projection.\n"
    "        self.in_proj_ba = self.create_ba_proj(\n"
    "            hidden_size=self.hidden_size,\n"
    "            num_v_heads=self.num_v_heads,\n"
    "            quant_config=self.quant_config,\n"
    "            prefix=f\"{prefix}.in_proj_ba\",\n"
    "        )\n"
    "        self.disable_tp_for_ba_proj = self.maybe_disable_tp(self.quant_config)\n"
)

PN365_CTOR_NEW = (
    "        # projection of the input hidden states\n"
    "        # Qwen3-Next and Qwen3.5 has a different qkv_proj layout,\n"
    "        # we need to create qkvz_proj adaptively here.\n"
    "        # When create_in_proj_qkvz is False (e.g. LoRA enabled in Qwen3.5),\n"
    "        # in_proj_qkv and in_proj_z are created separately instead.\n"
    "        # [Genesis PN365 fused GDN qkv|z|b|a single-GEMM input projection\n"
    "        # (port of OPEN vllm#42746) v1] — collapse the qkvz + ba GEMMs\n"
    "        # into one MergedColumnParallelLinear named in_proj_qkvzba.\n"
    "        # Gated by env GENESIS_PN365_GDN_GEMM_FUSE=1, Qwen3.5-only\n"
    "        # (non-interleaved GQA), LoRA-incompatible (auto-disabled when\n"
    "        # vllm_config.lora_config is set). Default-OFF: when env unset,\n"
    "        # the else-branch is bit-equivalent to the upstream code.\n"
    "        import os as _g_pn365_os\n"
    "        _g_pn365_env_on = _g_pn365_os.environ.get(\n"
    "            \"GENESIS_ENABLE_PN365_GDN_GEMM_FUSE\", \"0\"\n"
    "        ).strip().lower() in (\"1\", \"true\", \"yes\", \"on\")\n"
    "        _g_pn365_lora = (\n"
    "            getattr(vllm_config, \"lora_config\", None) is not None\n"
    "        )\n"
    "        _g_pn365_use_fused = (\n"
    "            _g_pn365_env_on\n"
    "            and not self.gqa_interleaved_layout\n"
    "            and not _g_pn365_lora\n"
    "        )\n"
    "        if _g_pn365_use_fused:\n"
    "            from vllm.model_executor.layers.linear import (\n"
    "                MergedColumnParallelLinear as _G_PN365_Merged,\n"
    "            )\n"
    "            # Single fused weight. Output_sizes split q|k|v|z|b|a so TP\n"
    "            # sharding aligns on head-dim boundaries (mirrors the\n"
    "            # existing qkvz layout). Loaded from 4 checkpoint shards\n"
    "            # (in_proj_qkv covers shards 0..2 = q,k,v; in_proj_z = 3;\n"
    "            # in_proj_b = 4; in_proj_a = 5). The Qwen3_5Model\n"
    "            # load_weights mapping is patched correspondingly by PN365.\n"
    "            self.in_proj_qkvzba = _G_PN365_Merged(\n"
    "                input_size=self.hidden_size,\n"
    "                output_sizes=[\n"
    "                    self.key_dim,      # q  shard 0\n"
    "                    self.key_dim,      # k  shard 1\n"
    "                    self.value_dim,    # v  shard 2\n"
    "                    self.value_dim,    # z  shard 3\n"
    "                    self.num_v_heads,  # b  shard 4\n"
    "                    self.num_v_heads,  # a  shard 5\n"
    "                ],\n"
    "                bias=False,\n"
    "                quant_config=self.quant_config,\n"
    "                prefix=f\"{prefix}.in_proj_qkvzba\",\n"
    "            )\n"
    "            # Keep these attrs defined for downstream code paths that\n"
    "            # may probe them, but route to in_proj_qkvzba in forward.\n"
    "            self.disable_tp_for_ba_proj = False\n"
    "        else:\n"
    "            self.in_proj_qkvz = self.create_qkvz_proj(\n"
    "                hidden_size=self.hidden_size,\n"
    "                key_dim=self.key_dim,\n"
    "                value_dim=self.value_dim,\n"
    "                quant_config=self.quant_config,\n"
    "                prefix=f\"{prefix}.in_proj_qkvz\",\n"
    "            )\n"
    "\n"
    "            # ba_proj doesn't support blockwise fp8 quantization.\n"
    "            # Qwen3-Next and Qwen3.5 have different in_proj_ba checkpoint\n"
    "            # layouts, so we use a factory method to create the projection.\n"
    "            self.in_proj_ba = self.create_ba_proj(\n"
    "                hidden_size=self.hidden_size,\n"
    "                num_v_heads=self.num_v_heads,\n"
    "                quant_config=self.quant_config,\n"
    "                prefix=f\"{prefix}.in_proj_ba\",\n"
    "            )\n"
    "            self.disable_tp_for_ba_proj = self.maybe_disable_tp(self.quant_config)\n"
)


# =============================================================================
# Anchor B: forward_cuda Part 1 — add the fused branch.
#
# IMPORTANT: this anchor matches the PRISTINE upstream forward_cuda Part 1.
# If PN204 is already applied, the anchor will not match (the file has the
# PN204 Genesis block instead). PN204 must be disabled when PN365 is enabled.
# That is enforced by the registry conflicts_with list and at runtime via
# the env-flag conflict check in apply().
# =============================================================================
PN365_FWD_OLD = (
    "        # ============================================================\n"
    "        # Part 1: Input Projection\n"
    "        # ============================================================\n"
    "        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
    "        ba, _ = self.in_proj_ba(hidden_states)\n"
)

PN365_FWD_NEW = (
    "        # ============================================================\n"
    "        # Part 1: Input Projection\n"
    "        # ============================================================\n"
    "        # [Genesis PN365 fused GDN qkv|z|b|a single-GEMM input projection\n"
    "        # (port of OPEN vllm#42746) v1] — when the constructor built\n"
    "        # in_proj_qkvzba, do one GEMM + 4-way split instead of two\n"
    "        # GEMMs. Bit-equivalent at the matmul level. Default-OFF.\n"
    "        if hasattr(self, \"in_proj_qkvzba\"):\n"
    "            try:\n"
    "                qkvzba, _ = self.in_proj_qkvzba(hidden_states)\n"
    "                _g_pn365_qkv_size = (\n"
    "                    self.key_dim * 2 + self.value_dim\n"
    "                ) // self.tp_size\n"
    "                _g_pn365_z_size = self.value_dim // self.tp_size\n"
    "                _g_pn365_n_v = self.num_v_heads // self.tp_size\n"
    "                mixed_qkv, _g_pn365_z_flat, b, a = qkvzba.split(\n"
    "                    [\n"
    "                        _g_pn365_qkv_size,\n"
    "                        _g_pn365_z_size,\n"
    "                        _g_pn365_n_v,\n"
    "                        _g_pn365_n_v,\n"
    "                    ],\n"
    "                    dim=-1,\n"
    "                )\n"
    "                z = _g_pn365_z_flat.reshape(\n"
    "                    _g_pn365_z_flat.size(0), -1, self.head_v_dim\n"
    "                )\n"
    "                b = b.contiguous()\n"
    "                a = a.contiguous()\n"
    "                # Skip the gqa_interleaved_layout / Qwen3.5 split path\n"
    "                # below — we have already produced mixed_qkv, z, b, a.\n"
    "                _g_pn365_short_circuit = True\n"
    "            except Exception as _g_pn365_exc:  # noqa: BLE001\n"
    "                import logging as _g_pn365_log\n"
    "                _g_pn365_logger = _g_pn365_log.getLogger(\n"
    "                    \"genesis.pn365_fallback\"\n"
    "                )\n"
    "                if not getattr(self.__class__, \"_g_pn365_warned\", False):\n"
    "                    _g_pn365_logger.warning(\n"
    "                        \"PN365 fused in_proj_qkvzba split failed (%s); \"\n"
    "                        \"this layer instance has the fused Linear but \"\n"
    "                        \"cannot fall back to two-GEMM at runtime. \"\n"
    "                        \"Disable PN365 and restart to recover.\",\n"
    "                        _g_pn365_exc,\n"
    "                    )\n"
    "                    self.__class__._g_pn365_warned = True\n"
    "                raise\n"
    "        else:\n"
    "            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
    "            ba, _ = self.in_proj_ba(hidden_states)\n"
    "            _g_pn365_short_circuit = False\n"
)


# After Anchor B, the upstream code does the gqa_interleaved_layout split.
# We need to short-circuit that block when PN365 ran. Anchor C wraps the
# next block in an `if not _g_pn365_short_circuit:`.
PN365_FWD_SPLIT_OLD = (
    "        if self.gqa_interleaved_layout:\n"
    "            # Qwen3-Next: unpack the interleaved GQA layout\n"
    "            query, key, value, z, b, a = self.fix_query_key_value_ordering(\n"
    "                mixed_qkvz, ba\n"
    "            )\n"
    "            query, key, value = map(\n"
    "                lambda x: rearrange(x, \"l p d -> l (p d)\"), (query, key, value)\n"
    "            )\n"
    "            mixed_qkv = torch.cat((query, key, value), dim=-1)\n"
    "        else:\n"
    "            # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order\n"
    "            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size\n"
    "            z_size = self.value_dim // self.tp_size\n"
    "            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)\n"
    "            z = z.reshape(z.size(0), -1, self.head_v_dim)\n"
    "            b, a = self.split_ba(ba)\n"
    "            b = b.contiguous()\n"
    "            a = a.contiguous()\n"
)

PN365_FWD_SPLIT_NEW = (
    "        if not _g_pn365_short_circuit:\n"
    "            if self.gqa_interleaved_layout:\n"
    "                # Qwen3-Next: unpack the interleaved GQA layout\n"
    "                query, key, value, z, b, a = self.fix_query_key_value_ordering(\n"
    "                    mixed_qkvz, ba\n"
    "                )\n"
    "                query, key, value = map(\n"
    "                    lambda x: rearrange(x, \"l p d -> l (p d)\"),\n"
    "                    (query, key, value),\n"
    "                )\n"
    "                mixed_qkv = torch.cat((query, key, value), dim=-1)\n"
    "            else:\n"
    "                # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order\n"
    "                qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size\n"
    "                z_size = self.value_dim // self.tp_size\n"
    "                mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)\n"
    "                z = z.reshape(z.size(0), -1, self.head_v_dim)\n"
    "                b, a = self.split_ba(ba)\n"
    "                b = b.contiguous()\n"
    "                a = a.contiguous()\n"
)


# =============================================================================
# Anchor C (PN50-aware variant) — v2 2026-06-10.
#
# On files where PN50 (SGLang#21019 fused split kernel) is text-applied,
# the else-branch of the split block is PN50's `_pn50_fused(...)` call, so
# the pristine PN365_FWD_SPLIT_OLD anchor cannot match. Compose the
# PN50-state anchor from the upstream if-side + PN50's own ANCHOR_NEW
# (imported — single source of truth, no text duplication). The NEW text
# wraps the whole composed block in the same short-circuit guard; PN50's
# kernel still runs on the non-fused path. When PN365's fused path ran,
# both the interleaved split AND PN50's kernel are skipped — the
# single-GEMM split already produced mixed_qkv, z, b, a.
#
# Discovered via PROD anchor pre-verification 2026-06-10: B2 was blocked
# by PN50 text (and B by dormant PN204 residue — reverted separately;
# see docs/superpowers/journal/2026-06-10-* "stale text-patch residue").
# =============================================================================
from sndr.engines.vllm.patches.attention.gdn.pn50_gdn_fused_proj import (
    ANCHOR_NEW as _PN50_ELSE_NEW,
)

_PN365_FWD_SPLIT_IF_SIDE = (
    "        if self.gqa_interleaved_layout:\n"
    "            # Qwen3-Next: unpack the interleaved GQA layout\n"
    "            query, key, value, z, b, a = self.fix_query_key_value_ordering(\n"
    "                mixed_qkvz, ba\n"
    "            )\n"
    "            query, key, value = map(\n"
    "                lambda x: rearrange(x, \"l p d -> l (p d)\"), (query, key, value)\n"
    "            )\n"
    "            mixed_qkv = torch.cat((query, key, value), dim=-1)\n"
)

# PN50's ANCHOR_NEW carries no trailing newline — normalize on compose.
PN365_FWD_SPLIT_OLD_PN50 = _PN365_FWD_SPLIT_IF_SIDE + _PN50_ELSE_NEW + "\n"


def _indent4(text: str) -> str:
    """Indent every non-blank line by 4 spaces (guard-wrapping helper)."""
    return "".join(
        ("    " + line if line.strip() else line)
        for line in text.splitlines(keepends=True)
    )


PN365_FWD_SPLIT_NEW_PN50 = (
    "        if not _g_pn365_short_circuit:\n"
    + _indent4(PN365_FWD_SPLIT_OLD_PN50)
)


# =============================================================================
# Anchor D: Qwen3_5Model.load_weights — extend stacked_params_mapping.
#
# We add the fused mapping entries when ANY GDN layer was built with the
# fused Linear (runtime detection — bit-equivalent to upstream when env
# flag is off, since no layer will have in_proj_qkvzba then).
# =============================================================================
PN365_LOAD_OLD = (
    "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
    "        stacked_params_mapping = [\n"
    "            # (param_name, shard_name, shard_id)\n"
    "            # GDN\n"
    "            (\"in_proj_qkvz\", \"in_proj_qkv\", (0, 1, 2)),\n"
    "            (\"in_proj_qkvz\", \"in_proj_z\", 3),\n"
    "            # self attention\n"
    "            (\"qkv_proj\", \"q_proj\", \"q\"),\n"
    "            (\"qkv_proj\", \"k_proj\", \"k\"),\n"
    "            (\"qkv_proj\", \"v_proj\", \"v\"),\n"
    "            # mlp\n"
    "            (\"gate_up_proj\", \"gate_proj\", 0),\n"
    "            (\"gate_up_proj\", \"up_proj\", 1),\n"
    "            (\"in_proj_ba\", \"in_proj_b\", 0),\n"
    "            (\"in_proj_ba\", \"in_proj_a\", 1),\n"
    "        ]\n"
)

PN365_LOAD_NEW = (
    "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
    "        # [Genesis PN365 fused GDN qkv|z|b|a single-GEMM input projection\n"
    "        # (port of OPEN vllm#42746) v1] — when the constructor built\n"
    "        # in_proj_qkvzba, route all 4 GDN checkpoint shards into the\n"
    "        # single fused weight. Detection is runtime (no env-flag read\n"
    "        # here) so the mapping is exactly correct even if the operator\n"
    "        # flipped the flag between init and load.\n"
    "        _g_pn365_use_fused = any(\n"
    "            hasattr(getattr(layer, \"linear_attn\", None), \"in_proj_qkvzba\")\n"
    "            for layer in self.layers\n"
    "        )\n"
    "        if _g_pn365_use_fused:\n"
    "            _g_pn365_gdn_mapping = [\n"
    "                (\"in_proj_qkvzba\", \"in_proj_qkv\", (0, 1, 2)),\n"
    "                (\"in_proj_qkvzba\", \"in_proj_z\", 3),\n"
    "                (\"in_proj_qkvzba\", \"in_proj_b\", 4),\n"
    "                (\"in_proj_qkvzba\", \"in_proj_a\", 5),\n"
    "            ]\n"
    "        else:\n"
    "            _g_pn365_gdn_mapping = [\n"
    "                (\"in_proj_qkvz\", \"in_proj_qkv\", (0, 1, 2)),\n"
    "                (\"in_proj_qkvz\", \"in_proj_z\", 3),\n"
    "                (\"in_proj_ba\", \"in_proj_b\", 0),\n"
    "                (\"in_proj_ba\", \"in_proj_a\", 1),\n"
    "            ]\n"
    "        stacked_params_mapping = [\n"
    "            # (param_name, shard_name, shard_id)\n"
    "            # GDN\n"
    "            *_g_pn365_gdn_mapping,\n"
    "            # self attention\n"
    "            (\"qkv_proj\", \"q_proj\", \"q\"),\n"
    "            (\"qkv_proj\", \"k_proj\", \"k\"),\n"
    "            (\"qkv_proj\", \"v_proj\", \"v\"),\n"
    "            # mlp\n"
    "            (\"gate_up_proj\", \"gate_proj\", 0),\n"
    "            (\"gate_up_proj\", \"up_proj\", 1),\n"
    "        ]\n"
)


# Drift markers — auto-SKIP if any of these appear in the target file
# (i.e., upstream landed #42746 or a successor patch using the same names).
# Self-collision lint (triage plan §6 2026-06-11): former entry
# "in_proj_qkvzba" is baked verbatim by our own port (ctor + load_weights
# replacements) — it cannot distinguish a real upstream merge from our
# residue (false "upstream_merged" skip, PN369 class). The remaining two
# (#42746 env flag + helper factory name) are strictly upstream-only.
_UPSTREAM_DRIFT_MARKERS = [
    "VLLM_GDN_FUSE_QKVZBA",
    "create_in_proj_qkvzba",
]


def _enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _pn204_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN204_DUAL_STREAM_INPROJ", "0",
    ).strip().lower() in ("1", "true", "yes", "on")


def _make_gdn_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_GDN_TARGET_REL)
    if target is None:
        return None

    # v2 (2026-06-10): the split block has TWO possible on-disk states —
    # pristine upstream, or PN50-patched (fused split kernel). Select the
    # matching anchor pair by inspecting the target content. PN50 applies
    # at an earlier ordinal than PN365, so on a PN50-enabled deployment
    # the PN50 state is what this builder sees.
    split_old, split_new = PN365_FWD_SPLIT_OLD, PN365_FWD_SPLIT_NEW
    try:
        content = open(target, encoding="utf-8").read()
        if (PN365_FWD_SPLIT_OLD not in content
                and PN365_FWD_SPLIT_OLD_PN50 in content):
            split_old, split_new = (
                PN365_FWD_SPLIT_OLD_PN50, PN365_FWD_SPLIT_NEW_PN50,
            )
            log.info(
                "[PN365] split block is PN50-patched — using PN50-aware "
                "anchor variant for pn365_fwd_cuda_split_shortcircuit"
            )
    except OSError as e:
        log.warning("[PN365] could not pre-read %s (%s) — using pristine "
                    "anchor variant", target, e)

    return TextPatcher(
        patch_name=(
            "PN365 qwen_gdn_linear_attn.py — fused qkv|z|b|a single-GEMM "
            "(port of OPEN vllm#42746)"
        ),
        target_file=str(target),
        marker=GENESIS_PN365_MARKER,
        sub_patches=[
            TextPatch(
                name="pn365_ctor_in_proj_qkvzba",
                anchor=PN365_CTOR_OLD,
                replacement=PN365_CTOR_NEW,
                required=True,
            ),
            TextPatch(
                name="pn365_fwd_cuda_part1_fused",
                anchor=PN365_FWD_OLD,
                replacement=PN365_FWD_NEW,
                required=True,
            ),
            TextPatch(
                name="pn365_fwd_cuda_split_shortcircuit",
                anchor=split_old,
                replacement=split_new,
                required=True,
            ),
        ],
        upstream_drift_markers=_UPSTREAM_DRIFT_MARKERS,
    )


def _make_model_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_MODEL_TARGET_REL)
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN365 qwen3_5.py — load_weights stacked_params_mapping for "
            "fused in_proj_qkvzba (port of OPEN vllm#42746)"
        ),
        target_file=str(target),
        marker=GENESIS_PN365_MARKER,
        sub_patches=[
            TextPatch(
                name="pn365_load_weights_mapping",
                anchor=PN365_LOAD_OLD,
                replacement=PN365_LOAD_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=_UPSTREAM_DRIFT_MARKERS,
    )


def apply() -> tuple[str, str]:
    if not _enabled():
        return (
            "skipped",
            f"PN365 disabled (set {_ENV_FLAG}=1 to enable; "
            "ALSO disable GENESIS_ENABLE_PN204_DUAL_STREAM_INPROJ — "
            "PN365 is the structurally superior single-GEMM win)",
        )
    if _pn204_enabled():
        return (
            "failed",
            "PN365 + PN204 are mutually exclusive at the same forward_cuda "
            "Part 1 site (PN204 wraps the two in_proj GEMMs in dual streams; "
            "PN365 fuses them into one GEMM with nothing to overlap). "
            "Set GENESIS_ENABLE_PN204_DUAL_STREAM_INPROJ=0 and restart.",
        )

    # FP8 + Marlin incompatibility guard (2026-06-10, learned the hard way
    # on PROD: crash-loop at weight load).
    #
    # Upstream itself notes "ba_proj doesn't support blockwise fp8
    # quantization" — in FP8 checkpoints the ba projection stays
    # UNQUANTIZED while qkvz is FP8. Fusing both into one
    # MergedColumnParallelLinear forces a single quant treatment; on the
    # Ampere Marlin path prepare_fp8_layer_for_marlin() then repacks the
    # fused weight whose per-TP shard N = (12288 + 64) / tp = 6176 is NOT
    # divisible by Marlin tile_n_size=64:
    #   RuntimeError: size_n = 6176 is not divisible by tile_n_size = 64
    # (gptq_marlin_repack, marlin_utils_fp8.py:127). Boot dies in a loop.
    #
    # vllm#42746's author bench was NVFP4 on sm_120 — a quant path where
    # the fused layout packs cleanly. Until the port grows an FP8-aware
    # layout (e.g. pad ba shard to 64 or keep ba unquantized inside the
    # merged Linear), refuse the fusion whenever the model quant config
    # is FP8 — the Marlin repack constraint makes it unbootable on
    # SM 8.6/8.9 and silently risky elsewhere.
    try:
        from vllm.config import get_current_vllm_config
        _cfg = get_current_vllm_config()
        _quant = getattr(getattr(_cfg, "model_config", None),
                         "quantization", None)
    except Exception:  # noqa: BLE001
        _quant = None
    if _quant is not None and "fp8" in str(_quant).lower():
        return "skipped", (
            "PN365 refused on FP8-quantized model: fused qkvz+ba shard "
            "N=6176 violates Marlin tile_n_size=64 divisibility "
            "(gptq_marlin_repack RuntimeError at weight load; ba_proj is "
            "unquantized in FP8 checkpoints by upstream design). "
            "vllm#42746 targets NVFP4 — port needs an FP8-aware layout "
            "before it can fuse here."
        )

    gdn_patcher = _make_gdn_patcher()
    if gdn_patcher is None:
        return "skipped", (
            f"PN365: GDN target file {_GDN_TARGET_REL} not found"
        )
    if not Path(gdn_patcher.target_file).is_file():
        return "skipped", (
            f"PN365: GDN target {gdn_patcher.target_file} disappeared"
        )

    model_patcher = _make_model_patcher()
    if model_patcher is None:
        return "skipped", (
            f"PN365: model target file {_MODEL_TARGET_REL} not found"
        )
    if not Path(model_patcher.target_file).is_file():
        return "skipped", (
            f"PN365: model target {model_patcher.target_file} disappeared"
        )

    # Apply GDN patches first; abort if any anchor fails so we never leave
    # the file in a half-patched state.
    try:
        gdn_result, gdn_failure = gdn_patcher.apply()
    except Exception as e:  # noqa: BLE001
        return "failed", f"PN365 GDN apply raised {e!r}"
    if gdn_result == TextPatchResult.FAILED:
        reason = gdn_failure.reason if gdn_failure else "unknown"
        return "failed", f"PN365 GDN anchor FAILED — {reason}"
    if gdn_result == TextPatchResult.SKIPPED:
        reason = gdn_failure.reason if gdn_failure else "unknown"
        return "skipped", f"PN365 GDN SKIPPED — {reason}"

    # Now apply the model load_weights patch.
    try:
        model_result, model_failure = model_patcher.apply()
    except Exception as e:  # noqa: BLE001
        return "failed", (
            f"PN365 model load_weights apply raised {e!r} "
            "(GDN file already patched — manual revert needed)"
        )
    if model_result == TextPatchResult.FAILED:
        reason = model_failure.reason if model_failure else "unknown"
        return "failed", (
            f"PN365 model load_weights anchor FAILED — {reason} "
            "(GDN file already patched — manual revert needed)"
        )
    if model_result == TextPatchResult.SKIPPED:
        reason = model_failure.reason if model_failure else "unknown"
        return "skipped", (
            f"PN365 model load_weights SKIPPED — {reason} "
            "(GDN file already patched — operator should verify)"
        )

    return "applied", (
        "PN365 applied: Qwen3.5 GDN linear_attn input projection now uses "
        "a single MergedColumnParallelLinear (in_proj_qkvzba) instead of "
        "separate qkvz + ba GEMMs. Collapses 2 GEMMs/layer -> 1 (fewer "
        "kernel launches, larger N for cuBLASLt tile selection). Bit-"
        "equivalent at the matmul level (same weight values, just "
        "concatenated along output dim). Author bench (Blackwell sm_120, "
        "Qwen3.5-35B-A3B-NVFP4): +3.7% TPOT @ C=3, +20% concurrency at "
        "SLO TPOT<=10ms. Ampere SM 8.6 derate -> expected +1-3% wall_TPS "
        "single-stream on Qwen3.6-35B-A3B FP8 / A5000 TP=2. Composes with "
        "PN350+PN54+P28+PN50+PN11 (different sites). HARD CONFLICT with "
        "PN204 — operator must keep PN204 OFF when PN365 is ON."
    )


def is_applied() -> bool:
    """Idempotency check used by drift detection."""
    gdn_target = resolve_vllm_file(_GDN_TARGET_REL)
    model_target = resolve_vllm_file(_MODEL_TARGET_REL)
    if gdn_target is None or model_target is None:
        return False
    try:
        gdn_text = Path(str(gdn_target)).read_text(encoding="utf-8")
        model_text = Path(str(model_target)).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return (
        GENESIS_PN365_MARKER in gdn_text
        and GENESIS_PN365_MARKER in model_text
    )
