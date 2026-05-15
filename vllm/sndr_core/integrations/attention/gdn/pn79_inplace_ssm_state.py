# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN79 — in-place SSM state for GDN chunk prefill (vllm#41824 backport).

================================================================
What it does
================================================================

Backport of vllm-project/vllm#41824 (Kermit-C, OPEN as of 2026-05-07).
Eliminates per-decode-step gather (`ssm_state[indices].contiguous()`) and
scatter (`ssm_state[indices] = final_state`) copies in the GDN chunk
prefill path, by passing `ssm_state_indices` directly to the Triton kernel.
Kernel uses `IS_CONTINUOUS_BATCHING` constexpr to read/write the global
SSM cache pool in-place via pointer arithmetic.

Author claims 4.5–36 GiB cumulative fp32 traffic eliminated per multi-turn
session (Qwen3.5-0.8B → Qwen3.6-27B scale).

================================================================
Architecture decision: Variant 1 (clean #41824 port) vs Variant 2 (bundle PN59)
================================================================

Choice: **Variant 1** — clean port of #41824, no PN59 streaming-dispatch
bundling. Reasoning (full justification in `docs/_internal/research/`):

1. **Empirical: PN59 streaming dead code on real workload.** Verified
   2026-05-06 evening on 27B+TQ k8v4: streaming-GDN dispatcher bypasses
   on every chunked-prefill chunk (T=64 ≤ threshold=1024) and every
   multi-seq batch. Streaming `_streaming_path` invocations: ZERO under
   our serving pattern.

2. **Occam's razor**: bundling PN59 mechanism into PN79 = adding code
   that empirically never executes. Anti-pattern.

3. **Maintenance burden**: clean port = direct upstream sync when
   #41824 merges. Bundled variant = permanent Genesis divergence.

4. **Decoupling preserves option value**: PN59 stays as separate
   `lifecycle: deprecated` patch. If future workload needs streaming
   (e.g., `--no-chunked-prefill` mode), re-enable PN59 separately —
   not coupled to PN79's apply state.

PN59 + PN54 → INTENDED for `lifecycle: deprecated` / `retired` after
multi-turn evidence proves PN79 win. As of 2026-05-07 both remain
`lifecycle: stable` in dispatcher.PATCH_REGISTRY — migration deferred
until Stage 4 evidence (Cliff 2 multi-turn reproducer + memory traffic
profiler). conflicts_with [PN59, PN54] in PN79 registry entry enforces
mutual exclusion at apply time regardless of lifecycle.

================================================================
Three sub-patches, atomic via MultiFilePatchTransaction
================================================================

Sub-1: `vllm/model_executor/layers/fla/ops/chunk.py`
   1A: import line — drop `input_guard` import
   1B: chunk_gated_delta_rule_fwd signature — add ssm_state_indices, has_initial_state
   1C: chunk_gated_delta_rule_fwd internal call to fwd_h — pass new kwargs
   1D: ChunkGatedDeltaRuleFunction.forward — major rewrite (drop @input_guard,
       manual contiguous, accelerator.device_index context)
   1E: chunk_gated_delta_rule (high-level API) signature + apply call

Sub-2: `vllm/model_executor/layers/fla/ops/chunk_delta_h.py`
   2A: kernel @triton.jit autotune key + constexpr param block
   2B: kernel main flow if-USE_INITIAL_STATE branch
   2C: kernel epilogue if-STORE_FINAL_STATE branch
   2D: Python wrapper chunk_gated_delta_rule_fwd_h signature + body

Sub-3: `vllm/model_executor/layers/mamba/gdn_linear_attn.py`
   3A: GdnLinearAttnFlashInfer.forward_cuda signature + body
   3B: GdnLinearAttnNativeFla.forward_native signature + passthrough
   3C: _forward_core gather/scatter elimination (THE WIN SITE)

================================================================
Atomic apply via MultiFilePatchTransaction
================================================================

All three patchers in single transaction:
- Phase 1 (dry-run): all anchors verified across all 3 files
- Phase 2 (commit): apply each patcher in order
- Rollback on any failure: in-memory snapshots restore

Either ALL THREE files patched OR NONE. Operator never sees half-patched
state which would crash on first GDN forward call.

================================================================
Drift detection
================================================================

Patches auto-SKIP if upstream landed equivalent fix (drift markers):
  - "ssm_state_indices"        — added by PR #41824 OR FLA equivalent
  - "IS_CONTINUOUS_BATCHING"   — Triton kernel constexpr (kernel patch)
  - "HAS_INITIAL_STATE_MASK"   — Triton kernel constexpr (kernel patch)

Configured per-patcher. PN59 wiring already lists "ssm_state_indices" as
its drift marker (added 2026-05-06).

================================================================
Conflict gating
================================================================

PATCH_REGISTRY entry has:
  conflicts_with: ["PN59", "PN54"]

dispatcher.should_apply() will SKIP PN79 if either is enabled:
  - PN59: anchor on chunk_gated_delta_rule_fwd body — overlaps with 1B/1C
  - PN54: removes .contiguous() on the same gather line PN79 erases entirely

Operator must explicitly disable PN59/PN54 in YAML config to enable PN79.
27B PROD config recommended: PN59=0, PN79=1 (per empirical "PN59 dead code"
finding).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Backport of: Kermit-C vllm-project/vllm#41824 (OPEN as of 2026-05-07).
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

log = logging.getLogger("genesis.wiring.pn79_inplace_ssm_state")

GENESIS_PN79_MARKER = "Genesis PN79 in-place SSM state (vllm#41824)"


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN79_INPLACE_SSM_STATE", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# ════════════════════════════════════════════════════════════════════════
# Sub-1: chunk.py anchors
# ════════════════════════════════════════════════════════════════════════

# 1A — import line: drop `input_guard` from utils import
ANCHOR_1A_IMPORT_OLD = "from .utils import FLA_CHUNK_SIZE, SUPPRESS_LEVEL, input_guard\n"
ANCHOR_1A_IMPORT_NEW = "from .utils import FLA_CHUNK_SIZE, SUPPRESS_LEVEL\n"

# 1B — chunk_gated_delta_rule_fwd signature: add ssm_state_indices + has_initial_state
ANCHOR_1B_FWD_SIG_OLD = (
    "    cu_seqlens: torch.Tensor | None = None,\n"
    "    chunk_indices: torch.Tensor | None = None,\n"
    "    chunk_offsets: torch.Tensor | None = None,\n"
    "):\n"
    "    g = chunk_local_cumsum(\n"
)
ANCHOR_1B_FWD_SIG_NEW = (
    "    cu_seqlens: torch.Tensor | None = None,\n"
    "    chunk_indices: torch.Tensor | None = None,\n"
    "    chunk_offsets: torch.Tensor | None = None,\n"
    "    ssm_state_indices: torch.Tensor | None = None,\n"
    "    has_initial_state: torch.Tensor | None = None,\n"
    "):\n"
    "    g = chunk_local_cumsum(\n"
)

# 1C — internal call to chunk_gated_delta_rule_fwd_h: pass new kwargs through
ANCHOR_1C_FWD_INTERNAL_OLD = (
    "        cu_seqlens=cu_seqlens,\n"
    "        chunk_indices=chunk_indices,\n"
    "        chunk_offsets=chunk_offsets,\n"
    "    )\n"
    "    o = chunk_fwd_o(\n"
)
ANCHOR_1C_FWD_INTERNAL_NEW = (
    "        cu_seqlens=cu_seqlens,\n"
    "        chunk_indices=chunk_indices,\n"
    "        chunk_offsets=chunk_offsets,\n"
    "        ssm_state_indices=ssm_state_indices,\n"
    "        has_initial_state=has_initial_state,\n"
    "    )\n"
    "    o = chunk_fwd_o(\n"
)

# 1D — ChunkGatedDeltaRuleFunction.forward MAJOR rewrite: drop @input_guard +
# manual contiguous + accelerator.device_index context
ANCHOR_1D_FORWARD_OLD = (
    "    @staticmethod\n"
    "    @input_guard\n"
    "    @torch.amp.custom_fwd(device_type=\"cuda\")\n"
    "    def forward(\n"
    "        ctx,\n"
    "        q: torch.Tensor,\n"
    "        k: torch.Tensor,\n"
    "        v: torch.Tensor,\n"
    "        g: torch.Tensor,\n"
    "        beta: torch.Tensor,\n"
    "        scale: float,\n"
    "        initial_state: torch.Tensor,\n"
    "        output_final_state: bool,\n"
    "        cu_seqlens: torch.Tensor | None = None,\n"
    "        chunk_indices: torch.Tensor | None = None,\n"
    "        chunk_offsets: torch.Tensor | None = None,\n"
    "        use_qk_l2norm_in_kernel: bool = False,\n"
    "    ):\n"
    "        if use_qk_l2norm_in_kernel:\n"
    "            q = l2norm_fwd(q)\n"
    "            k = l2norm_fwd(k)\n"
    "\n"
    "        g, o, A, final_state, w, h, v_new = chunk_gated_delta_rule_fwd(\n"
    "            q=q,\n"
    "            k=k,\n"
    "            v=v,\n"
    "            g=g,\n"
    "            beta=beta,\n"
    "            scale=scale,\n"
    "            initial_state=initial_state,\n"
    "            output_final_state=output_final_state,\n"
    "            cu_seqlens=cu_seqlens,\n"
    "            chunk_indices=chunk_indices,\n"
    "            chunk_offsets=chunk_offsets,\n"
    "        )\n"
    "        ctx.scale = scale\n"
    "        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel\n"
    "        return o.to(q.dtype), final_state\n"
)
ANCHOR_1D_FORWARD_NEW = (
    "    @staticmethod\n"
    "    @torch.amp.custom_fwd(device_type=\"cuda\")\n"
    "    def forward(\n"
    "        ctx,\n"
    "        q: torch.Tensor,\n"
    "        k: torch.Tensor,\n"
    "        v: torch.Tensor,\n"
    "        g: torch.Tensor,\n"
    "        beta: torch.Tensor,\n"
    "        scale: float,\n"
    "        initial_state: torch.Tensor,\n"
    "        output_final_state: bool,\n"
    "        cu_seqlens: torch.Tensor | None = None,\n"
    "        chunk_indices: torch.Tensor | None = None,\n"
    "        chunk_offsets: torch.Tensor | None = None,\n"
    "        use_qk_l2norm_in_kernel: bool = False,\n"
    "        ssm_state_indices: torch.Tensor | None = None,\n"
    "        has_initial_state: torch.Tensor | None = None,\n"
    "    ):\n"
    "        # [Genesis PN79 vllm#41824] Manual contiguity instead of @input_guard.\n"
    "        # Skip .contiguous() on initial_state when ssm_state_indices given:\n"
    "        # kernel handles non-contiguous via strides, contiguity is expensive\n"
    "        # for large SSM cache views.\n"
    "        q = q.contiguous()\n"
    "        k = k.contiguous()\n"
    "        v = v.contiguous()\n"
    "        g = g.contiguous()\n"
    "        beta = beta.contiguous()\n"
    "        cu_seqlens = cu_seqlens.contiguous() if cu_seqlens is not None else None\n"
    "        chunk_indices = (\n"
    "            chunk_indices.contiguous() if chunk_indices is not None else None\n"
    "        )\n"
    "        chunk_offsets = (\n"
    "            chunk_offsets.contiguous() if chunk_offsets is not None else None\n"
    "        )\n"
    "        ssm_state_indices = (\n"
    "            ssm_state_indices.contiguous() if ssm_state_indices is not None else None\n"
    "        )\n"
    "        has_initial_state = (\n"
    "            has_initial_state.contiguous() if has_initial_state is not None else None\n"
    "        )\n"
    "        if ssm_state_indices is None and initial_state is not None:\n"
    "            initial_state = initial_state.contiguous()\n"
    "\n"
    "        with torch.accelerator.device_index(q.device.index):\n"
    "            if use_qk_l2norm_in_kernel:\n"
    "                q = l2norm_fwd(q)\n"
    "                k = l2norm_fwd(k)\n"
    "\n"
    "            g, o, A, final_state, w, h, v_new = chunk_gated_delta_rule_fwd(\n"
    "                q=q,\n"
    "                k=k,\n"
    "                v=v,\n"
    "                g=g,\n"
    "                beta=beta,\n"
    "                scale=scale,\n"
    "                initial_state=initial_state,\n"
    "                output_final_state=output_final_state,\n"
    "                cu_seqlens=cu_seqlens,\n"
    "                chunk_indices=chunk_indices,\n"
    "                chunk_offsets=chunk_offsets,\n"
    "                ssm_state_indices=ssm_state_indices,\n"
    "                has_initial_state=has_initial_state,\n"
    "            )\n"
    "            ctx.scale = scale\n"
    "            ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel\n"
    "            return o.to(q.dtype), final_state\n"
)


# ─── 1E: chunk_gated_delta_rule (high-level API) — 3 sub-anchors ─────
# 1E completes the public-API part of Sub-1: the high-level wrapper
# `chunk_gated_delta_rule` at chunk.py:129 must accept ssm_state_indices /
# has_initial_state AND forward them to ChunkGatedDeltaRuleFunction.apply.
# Without 1E, calls from gdn_linear_attn.forward_native (after 3B applies)
# crash with TypeError because the high-level API doesn't accept the
# kwargs. Three sub-anchors:
#   1E_SIG       — function signature (add 2 kwargs)
#   1E_VAL       — validation block (gate ValueError on ssm_state_indices)
#   1E_APPLY_CALL — ChunkGatedDeltaRuleFunction.apply() trailing args
ANCHOR_1E_SIG_OLD = (
    "    chunk_indices: torch.Tensor | None = None,\n"
    "    chunk_offsets: torch.Tensor | None = None,\n"
    "    use_qk_l2norm_in_kernel: bool = False,\n"
    "):\n"
)
ANCHOR_1E_SIG_NEW = (
    "    chunk_indices: torch.Tensor | None = None,\n"
    "    chunk_offsets: torch.Tensor | None = None,\n"
    "    use_qk_l2norm_in_kernel: bool = False,\n"
    "    ssm_state_indices: torch.Tensor | None = None,\n"
    "    has_initial_state: torch.Tensor | None = None,\n"
    "):\n"
)

# Pristine validation: raises ValueError when N != len(cu_seqlens)-1.
# After PN79 the initial_state is the full ssm_state pool (N_pool entries),
# but ssm_state_indices selects which seq-row to use → N_pool != batch_size,
# but that's fine. Validation must skip when ssm_state_indices given.
ANCHOR_1E_VAL_OLD = (
    "        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:\n"
)
ANCHOR_1E_VAL_NEW = (
    "        if (\n"
    "            initial_state is not None\n"
    "            and ssm_state_indices is None\n"
    "            and initial_state.shape[0] != len(cu_seqlens) - 1\n"
    "        ):\n"
)

ANCHOR_1E_APPLY_CALL_OLD = (
    "        chunk_offsets,\n"
    "        use_qk_l2norm_in_kernel,\n"
    "    )\n"
    "    return o, final_state\n"
)
ANCHOR_1E_APPLY_CALL_NEW = (
    "        chunk_offsets,\n"
    "        use_qk_l2norm_in_kernel,\n"
    "        ssm_state_indices,\n"
    "        has_initial_state,\n"
    "    )\n"
    "    return o, final_state\n"
)


# ════════════════════════════════════════════════════════════════════════
# Sub-3: gdn_linear_attn.py anchors (THE WIN SITE — gather/scatter elimination)
# ════════════════════════════════════════════════════════════════════════
#
# This is the core value proposition. Removes 4 lines (assert + gather + assert
# + zero-fill) before the kernel call AND 3 lines (Init cache scatter back)
# after. Replaces with single in-place call via ssm_state + indices.
#
# Sub-3 is INDEPENDENT of Sub-2 (kernel) only at call-site syntax level. At
# runtime, ssm_state_indices=indices kwarg flows to chunk_gated_delta_rule_fwd
# (via Sub-1) which forwards to chunk_gated_delta_rule_fwd_h (via Sub-2). If
# Sub-2 not applied → kernel TypeErrors on unknown kwarg → MultiFilePatchTransaction
# ensures all-or-nothing apply.

# ─── 3A: GatedDeltaRule.forward_cuda — FlashInfer fallback ───────────
# FlashInfer kernel does NOT support in-place; this branch keeps the gather/
# scatter copy when ssm_state_indices given but routes through fi_chunk_*
# (which can't read in-place). Triton path (Sub-2) gets the real win.
ANCHOR_3A_FORWARD_CUDA_OLD = (
    "        chunk_indices: torch.Tensor | None = None,\n"
    "        chunk_offsets: torch.Tensor | None = None,\n"
    "        use_qk_l2norm_in_kernel: bool = True,\n"
    "    ):\n"
    "        return fi_chunk_gated_delta_rule(\n"
    "            q=q,\n"
    "            k=k,\n"
    "            v=v,\n"
    "            g=g,\n"
    "            beta=beta,\n"
    "            initial_state=initial_state,\n"
    "            output_final_state=output_final_state,\n"
    "            cu_seqlens=cu_seqlens,\n"
    "            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,\n"
    "        )\n"
)
ANCHOR_3A_FORWARD_CUDA_NEW = (
    "        chunk_indices: torch.Tensor | None = None,\n"
    "        chunk_offsets: torch.Tensor | None = None,\n"
    "        use_qk_l2norm_in_kernel: bool = True,\n"
    "        ssm_state_indices: torch.Tensor | None = None,\n"
    "        has_initial_state: torch.Tensor | None = None,\n"
    "    ):\n"
    "        if ssm_state_indices is not None:\n"
    "            assert has_initial_state is not None\n"
    "            gathered_initial = initial_state[ssm_state_indices].contiguous()\n"
    "            gathered_initial[~has_initial_state, ...] = 0\n"
    "            o, final_state = fi_chunk_gated_delta_rule(\n"
    "                q=q,\n"
    "                k=k,\n"
    "                v=v,\n"
    "                g=g,\n"
    "                beta=beta,\n"
    "                initial_state=gathered_initial,\n"
    "                output_final_state=output_final_state,\n"
    "                cu_seqlens=cu_seqlens,\n"
    "                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,\n"
    "            )\n"
    "            if output_final_state:\n"
    "                initial_state[ssm_state_indices] = final_state.to(initial_state.dtype)\n"
    "            return o, final_state\n"
    "        return fi_chunk_gated_delta_rule(\n"
    "            q=q,\n"
    "            k=k,\n"
    "            v=v,\n"
    "            g=g,\n"
    "            beta=beta,\n"
    "            initial_state=initial_state,\n"
    "            output_final_state=output_final_state,\n"
    "            cu_seqlens=cu_seqlens,\n"
    "            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,\n"
    "        )\n"
)

# ─── 3B: GatedDeltaRule.forward_native — passthrough to fla_chunk_* ──
# Native FLA path: chunk_gated_delta_rule (the high-level API in chunk.py)
# now accepts ssm_state_indices/has_initial_state via Sub-1 (anchor 1B/1C).
# 3B just forwards the kwargs through.
ANCHOR_3B_FORWARD_NATIVE_OLD = (
    "        chunk_indices: torch.Tensor | None = None,\n"
    "        chunk_offsets: torch.Tensor | None = None,\n"
    "        use_qk_l2norm_in_kernel: bool = True,\n"
    "    ):\n"
    "        return fla_chunk_gated_delta_rule(\n"
    "            q=q,\n"
    "            k=k,\n"
    "            v=v,\n"
    "            g=g,\n"
    "            beta=beta,\n"
    "            initial_state=initial_state,\n"
    "            output_final_state=output_final_state,\n"
    "            cu_seqlens=cu_seqlens,\n"
    "            chunk_indices=chunk_indices,\n"
    "            chunk_offsets=chunk_offsets,\n"
    "            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,\n"
    "        )\n"
)
ANCHOR_3B_FORWARD_NATIVE_NEW = (
    "        chunk_indices: torch.Tensor | None = None,\n"
    "        chunk_offsets: torch.Tensor | None = None,\n"
    "        use_qk_l2norm_in_kernel: bool = True,\n"
    "        ssm_state_indices: torch.Tensor | None = None,\n"
    "        has_initial_state: torch.Tensor | None = None,\n"
    "    ):\n"
    "        return fla_chunk_gated_delta_rule(\n"
    "            q=q,\n"
    "            k=k,\n"
    "            v=v,\n"
    "            g=g,\n"
    "            beta=beta,\n"
    "            initial_state=initial_state,\n"
    "            output_final_state=output_final_state,\n"
    "            cu_seqlens=cu_seqlens,\n"
    "            chunk_indices=chunk_indices,\n"
    "            chunk_offsets=chunk_offsets,\n"
    "            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,\n"
    "            ssm_state_indices=ssm_state_indices,\n"
    "            has_initial_state=has_initial_state,\n"
    "        )\n"
)

# ─── 3C: _forward_core gather/scatter elimination (THE WIN SITE) ─────
# ─── Sub-4: olmo_hybrid.py — gather/scatter elimination (analogous to 3C) ─
# Olmo-Hybrid uses the SAME hybrid GDN linear-attn pattern as Qwen3.6 27B
# Lorbus, but call site uses `chunk_gated_delta_rule(...)` (free function
# from vllm/model_executor/layers/fla/ops/chunk.py) instead of
# `self.chunk_gated_delta_rule` (instance method on GdnLinearAttn class).
# Pristine has NO `assert` lines (they exist only in gdn_linear_attn.py 3C).
# Single anchor covers full prefill block.
#
# Note: this anchor never fires on Genesis fleet (no Olmo models in
# model_configs/), but completes Variant 1 port for community users
# building Genesis docker image with Olmo support pkg.
ANCHOR_4A_OLMO_FORWARD_CORE_OLD = (
    "        if attn_metadata.num_prefills > 0:\n"
    "            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()\n"
    "            initial_state[~has_initial_state, ...] = 0\n"
    "            (\n"
    "                core_attn_out_non_spec,\n"
    "                last_recurrent_state,\n"
    "            ) = chunk_gated_delta_rule(\n"
    "                q=query_non_spec,\n"
    "                k=key_non_spec,\n"
    "                v=value_non_spec,\n"
    "                g=g_non_spec,\n"
    "                beta=beta_non_spec,\n"
    "                initial_state=initial_state,\n"
    "                output_final_state=True,\n"
    "                cu_seqlens=non_spec_query_start_loc,\n"
    "                use_qk_l2norm_in_kernel=True,\n"
    "            )\n"
    "            ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(\n"
    "                ssm_state.dtype\n"
    "            )\n"
)
ANCHOR_4A_OLMO_FORWARD_CORE_NEW = (
    "        # [Genesis PN79 vllm#41824] Olmo-Hybrid gather/scatter elimination —\n"
    "        # parallel to Sub-3C in gdn_linear_attn.py. ssm_state passed in-place\n"
    "        # to kernel via ssm_state_indices/has_initial_state kwargs.\n"
    "        if attn_metadata.num_prefills > 0:\n"
    "            (\n"
    "                core_attn_out_non_spec,\n"
    "                last_recurrent_state,\n"
    "            ) = chunk_gated_delta_rule(\n"
    "                q=query_non_spec,\n"
    "                k=key_non_spec,\n"
    "                v=value_non_spec,\n"
    "                g=g_non_spec,\n"
    "                beta=beta_non_spec,\n"
    "                initial_state=ssm_state,\n"
    "                output_final_state=True,\n"
    "                cu_seqlens=non_spec_query_start_loc,\n"
    "                use_qk_l2norm_in_kernel=True,\n"
    "                ssm_state_indices=non_spec_state_indices_tensor,\n"
    "                has_initial_state=has_initial_state,\n"
    "            )\n"
)


ANCHOR_3C_GATHER_SCATTER_OLD = (
    "        # 2.2: Process the remaining part\n"
    "        if attn_metadata.num_prefills > 0:\n"
    "            assert non_spec_state_indices_tensor is not None\n"
    "            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()  # type: ignore[index]\n"
    "            assert has_initial_state is not None\n"
    "            initial_state[~has_initial_state, ...] = 0  # type: ignore[operator]\n"
    "            (\n"
    "                core_attn_out_non_spec,\n"
    "                last_recurrent_state,\n"
    "            ) = self.chunk_gated_delta_rule(\n"
    "                q=query_non_spec,\n"
    "                k=key_non_spec,\n"
    "                v=value_non_spec,\n"
    "                g=g_non_spec,\n"
    "                beta=beta_non_spec,\n"
    "                initial_state=initial_state,\n"
    "                output_final_state=True,\n"
    "                cu_seqlens=non_spec_query_start_loc,\n"
    "                chunk_indices=attn_metadata.chunk_indices,\n"
    "                chunk_offsets=attn_metadata.chunk_offsets,\n"
    "                use_qk_l2norm_in_kernel=False,\n"
    "            )\n"
    "            # Init cache\n"
    "            ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(\n"
    "                ssm_state.dtype\n"
    "            )\n"
)
ANCHOR_3C_GATHER_SCATTER_NEW = (
    "        # 2.2: Process the remaining part\n"
    "        # [Genesis PN79 vllm#41824] gather/scatter elimination —\n"
    "        # ssm_state passed directly to kernel; ssm_state_indices/has_initial_state\n"
    "        # forwarded to Triton kernel which reads/writes in-place via\n"
    "        # IS_CONTINUOUS_BATCHING constexpr branch. Saves per-decode-step\n"
    "        # gather (ssm_state[indices].contiguous()) + scatter (ssm_state[indices] = ...).\n"
    "        if attn_metadata.num_prefills > 0:\n"
    "            (\n"
    "                core_attn_out_non_spec,\n"
    "                last_recurrent_state,\n"
    "            ) = self.chunk_gated_delta_rule(\n"
    "                q=query_non_spec,\n"
    "                k=key_non_spec,\n"
    "                v=value_non_spec,\n"
    "                g=g_non_spec,\n"
    "                beta=beta_non_spec,\n"
    "                initial_state=ssm_state,\n"
    "                output_final_state=True,\n"
    "                cu_seqlens=non_spec_query_start_loc,\n"
    "                chunk_indices=attn_metadata.chunk_indices,\n"
    "                chunk_offsets=attn_metadata.chunk_offsets,\n"
    "                use_qk_l2norm_in_kernel=False,\n"
    "                ssm_state_indices=non_spec_state_indices_tensor,\n"
    "                has_initial_state=has_initial_state,\n"
    "            )\n"
)


# ════════════════════════════════════════════════════════════════════════
# Sub-2: chunk_delta_h.py — Triton kernel changes (HIGHEST RISK)
# ════════════════════════════════════════════════════════════════════════
#
# 7 anchor points (verified against pristine vllm 0.20.2rc1.dev9+g01d4d1ad3
# and PR #41824 diff fetched 2026-05-07):
#   2A: @triton.heuristics dict — add IS_CONTINUOUS_BATCHING + HAS_INITIAL_STATE_MASK
#   2B: kernel @triton.jit signature — add params + 4 strides + 2 constexpr flags
#   2C: kernel main flow USE_INITIAL_STATE — should_load + IS_CONTINUOUS_BATCHING
#   2D: kernel epilogue STORE_FINAL_STATE — IS_CONTINUOUS_BATCHING ht offset
#   2E: chunk_gated_delta_rule_fwd_h Python wrapper signature
#   2F: chunk_gated_delta_rule_fwd_h Python wrapper body — strides if/else
#   2G: chunk_gated_delta_rule_fwd_h Python wrapper kernel call kwargs
#
# Triton DSL care: whitespace/indent verified char-for-char from pristine.
# P67 (turboquant_attn.py) confirmed conflict-free — different file entirely.

# ─── 2A: heuristics dict ──────────────────────────────────────────────
ANCHOR_2A_HEURISTICS_OLD = (
    '        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,\n'
    '        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,\n'
    '        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,\n'
    '    }\n'
    ')\n'
)
ANCHOR_2A_HEURISTICS_NEW = (
    '        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,\n'
    '        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,\n'
    '        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,\n'
    '        "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,\n'
    '        "HAS_INITIAL_STATE_MASK": lambda args: args["has_initial_state"] is not None,\n'
    '    }\n'
    ')\n'
)

# ─── 2B: kernel @triton.jit signature ─────────────────────────────────
ANCHOR_2B_KERNEL_SIG_OLD = (
    "    cu_seqlens,\n"
    "    chunk_offsets,\n"
    "    T,\n"
    "    H: tl.constexpr,\n"
    "    Hg: tl.constexpr,\n"
    "    K: tl.constexpr,\n"
    "    V: tl.constexpr,\n"
    "    BT: tl.constexpr,\n"
    "    BV: tl.constexpr,\n"
    "    USE_G: tl.constexpr,\n"
    "    USE_GK: tl.constexpr,\n"
    "    USE_INITIAL_STATE: tl.constexpr,\n"
    "    STORE_FINAL_STATE: tl.constexpr,\n"
    "    SAVE_NEW_VALUE: tl.constexpr,\n"
    "    IS_VARLEN: tl.constexpr,\n"
    "):\n"
)
ANCHOR_2B_KERNEL_SIG_NEW = (
    "    cu_seqlens,\n"
    "    chunk_offsets,\n"
    "    ssm_state_indices,\n"
    "    has_initial_state,\n"
    "    T,\n"
    "    H: tl.constexpr,\n"
    "    Hg: tl.constexpr,\n"
    "    K: tl.constexpr,\n"
    "    V: tl.constexpr,\n"
    "    BT: tl.constexpr,\n"
    "    BV: tl.constexpr,\n"
    "    stride_init_state_token: tl.constexpr,\n"
    "    stride_final_state_token: tl.constexpr,\n"
    "    stride_indices_seq: tl.constexpr,\n"
    "    stride_has_initial_state: tl.constexpr,\n"
    "    USE_G: tl.constexpr,\n"
    "    USE_GK: tl.constexpr,\n"
    "    USE_INITIAL_STATE: tl.constexpr,\n"
    "    STORE_FINAL_STATE: tl.constexpr,\n"
    "    SAVE_NEW_VALUE: tl.constexpr,\n"
    "    IS_VARLEN: tl.constexpr,\n"
    "    IS_CONTINUOUS_BATCHING: tl.constexpr,\n"
    "    HAS_INITIAL_STATE_MASK: tl.constexpr,\n"
    "):\n"
)

# ─── 2C: kernel main flow — USE_INITIAL_STATE rewrite ─────────────────
ANCHOR_2C_KERNEL_MAIN_OLD = (
    "    if USE_INITIAL_STATE:\n"
    "        h0 = h0 + i_nh * V * K\n"
    "    if STORE_FINAL_STATE:\n"
    "        ht = ht + i_nh * V * K\n"
    "\n"
    "    # load initial state\n"
    "    if USE_INITIAL_STATE:\n"
    "        p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))\n"
    "        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)\n"
    "        if K > 64:\n"
    "            p_h0_2 = tl.make_block_ptr(\n"
    "                h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)\n"
    "            )\n"
    "            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)\n"
    "        if K > 128:\n"
    "            p_h0_3 = tl.make_block_ptr(\n"
    "                h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)\n"
    "            )\n"
    "            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)\n"
    "        if K > 192:\n"
    "            p_h0_4 = tl.make_block_ptr(\n"
    "                h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)\n"
    "            )\n"
    "            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)\n"
)
ANCHOR_2C_KERNEL_MAIN_NEW = (
    "    if USE_INITIAL_STATE:\n"
    "        should_load = True\n"
    "        if IS_CONTINUOUS_BATCHING:\n"
    "            state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(\n"
    "                tl.int64\n"
    "            )\n"
    "            if HAS_INITIAL_STATE_MASK:\n"
    "                has_init = tl.load(has_initial_state + i_n * stride_has_initial_state)\n"
    "                if has_init:\n"
    "                    h0 = h0 + state_idx * stride_init_state_token + i_h * V * K\n"
    "                else:\n"
    "                    should_load = False\n"
    "            else:\n"
    "                h0 = h0 + state_idx * stride_init_state_token + i_h * V * K\n"
    "        else:\n"
    "            h0 = h0 + i_nh * V * K\n"
    "        if should_load:\n"
    "            p_h0_1 = tl.make_block_ptr(\n"
    "                h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)\n"
    "            )\n"
    "            b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)\n"
    "            if K > 64:\n"
    "                p_h0_2 = tl.make_block_ptr(\n"
    "                    h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)\n"
    "                )\n"
    "                b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)\n"
    "            if K > 128:\n"
    "                p_h0_3 = tl.make_block_ptr(\n"
    "                    h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)\n"
    "                )\n"
    "                b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)\n"
    "            if K > 192:\n"
    "                p_h0_4 = tl.make_block_ptr(\n"
    "                    h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)\n"
    "                )\n"
    "                b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)\n"
)

# ─── 2D: kernel epilogue STORE_FINAL_STATE ───────────────────────────
ANCHOR_2D_KERNEL_EPILOGUE_OLD = (
    "    # epilogue\n"
    "    if STORE_FINAL_STATE:\n"
    "        p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))\n"
)
ANCHOR_2D_KERNEL_EPILOGUE_NEW = (
    "    # epilogue\n"
    "    if STORE_FINAL_STATE:\n"
    "        if IS_CONTINUOUS_BATCHING:\n"
    "            state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(\n"
    "                tl.int64\n"
    "            )\n"
    "            ht = ht + state_idx * stride_final_state_token + i_h * V * K\n"
    "        else:\n"
    "            ht = ht + i_nh * V * K\n"
    "        p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))\n"
)

# ─── 2E: Python wrapper chunk_gated_delta_rule_fwd_h signature ────────
ANCHOR_2E_WRAPPER_SIG_OLD = (
    "    cu_seqlens: torch.Tensor | None = None,\n"
    "    chunk_indices: torch.Tensor | None = None,\n"
    "    chunk_offsets: torch.Tensor | None = None,\n"
    ") -> tuple[torch.Tensor, torch.Tensor]:\n"
)
ANCHOR_2E_WRAPPER_SIG_NEW = (
    "    cu_seqlens: torch.Tensor | None = None,\n"
    "    chunk_indices: torch.Tensor | None = None,\n"
    "    chunk_offsets: torch.Tensor | None = None,\n"
    "    ssm_state_indices: torch.Tensor | None = None,\n"
    "    has_initial_state: torch.Tensor | None = None,\n"
    ") -> tuple[torch.Tensor, torch.Tensor]:\n"
)

# ─── 2F: Python wrapper body — strides if/else + final_state alias ────
ANCHOR_2F_WRAPPER_BODY_OLD = (
    '    assert K <= 256, "current kernel does not support head dimension larger than 256."\n'
    "\n"
    "    h = k.new_empty(B, NT, H, V, K)\n"
    "    final_state = (\n"
    "        k.new_empty(N, H, V, K, dtype=torch.float32) if output_final_state else None\n"
    "    )\n"
    "\n"
    "    v_new = torch.empty_like(u) if save_new_value else None\n"
)
ANCHOR_2F_WRAPPER_BODY_NEW = (
    '    assert K <= 256, "current kernel does not support head dimension larger than 256."\n'
    "\n"
    "    if ssm_state_indices is not None:\n"
    "        stride_indices_seq = ssm_state_indices.stride(0)\n"
    "        stride_init_state_token = initial_state.stride(0)\n"
    "        stride_final_state_token = initial_state.stride(0)\n"
    "        final_state = initial_state if output_final_state else None\n"
    "        stride_has_initial_state = (\n"
    "            has_initial_state.stride(0) if has_initial_state is not None else 1\n"
    "        )\n"
    "    else:\n"
    "        stride_indices_seq = 1\n"
    "        stride_init_state_token = 1\n"
    "        stride_final_state_token = 1\n"
    "        stride_has_initial_state = 1\n"
    "        final_state = (\n"
    "            k.new_empty(N, H, V, K, dtype=torch.float32) if output_final_state else None\n"
    "        )\n"
    "\n"
    "    h = k.new_empty(B, NT, H, V, K)\n"
    "\n"
    "    v_new = torch.empty_like(u) if save_new_value else None\n"
)

# ─── 2G: Python wrapper kernel-call kwargs ────────────────────────────
ANCHOR_2G_WRAPPER_KERNEL_CALL_OLD = (
    "        cu_seqlens=cu_seqlens,\n"
    "        chunk_offsets=chunk_offsets,\n"
    "        T=T,\n"
    "        H=H,\n"
    "        Hg=Hg,\n"
    "        K=K,\n"
    "        V=V,\n"
    "        BT=BT,\n"
    "    )\n"
)
ANCHOR_2G_WRAPPER_KERNEL_CALL_NEW = (
    "        cu_seqlens=cu_seqlens,\n"
    "        chunk_offsets=chunk_offsets,\n"
    "        ssm_state_indices=ssm_state_indices,\n"
    "        has_initial_state=has_initial_state,\n"
    "        T=T,\n"
    "        H=H,\n"
    "        Hg=Hg,\n"
    "        K=K,\n"
    "        V=V,\n"
    "        BT=BT,\n"
    "        stride_init_state_token=stride_init_state_token,\n"
    "        stride_final_state_token=stride_final_state_token,\n"
    "        stride_indices_seq=stride_indices_seq,\n"
    "        stride_has_initial_state=stride_has_initial_state,\n"
    "    )\n"
)


# ════════════════════════════════════════════════════════════════════════
# Patcher construction
# ════════════════════════════════════════════════════════════════════════


def _make_chunk_patcher() -> TextPatcher | None:
    """Sub-1: chunk.py — orchestrator + ChunkGatedDeltaRuleFunction.forward."""
    target = resolve_vllm_file("model_executor/layers/fla/ops/chunk.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN79 Sub-1 chunk.py (orchestrator + forward)",
        target_file=str(target),
        marker=GENESIS_PN79_MARKER,
        patch_id="PN79.Sub-1",
        sub_patches=[
            TextPatch(
                name="1A_drop_input_guard_import",
                anchor=ANCHOR_1A_IMPORT_OLD,
                replacement=ANCHOR_1A_IMPORT_NEW,
                required=True,
            ),
            TextPatch(
                name="1B_fwd_signature_add_ssm_state_indices",
                anchor=ANCHOR_1B_FWD_SIG_OLD,
                replacement=ANCHOR_1B_FWD_SIG_NEW,
                required=True,
            ),
            TextPatch(
                name="1C_fwd_internal_call_pass_kwargs",
                anchor=ANCHOR_1C_FWD_INTERNAL_OLD,
                replacement=ANCHOR_1C_FWD_INTERNAL_NEW,
                required=True,
            ),
            TextPatch(
                name="1D_forward_rewrite_drop_input_guard_add_contiguous",
                anchor=ANCHOR_1D_FORWARD_OLD,
                replacement=ANCHOR_1D_FORWARD_NEW,
                required=True,
            ),
            TextPatch(
                name="1E_high_level_api_signature",
                anchor=ANCHOR_1E_SIG_OLD,
                replacement=ANCHOR_1E_SIG_NEW,
                required=True,
            ),
            TextPatch(
                name="1E_high_level_api_validation_skip_on_ssm_indices",
                anchor=ANCHOR_1E_VAL_OLD,
                replacement=ANCHOR_1E_VAL_NEW,
                required=True,
            ),
            TextPatch(
                name="1E_high_level_api_apply_call_trailing_args",
                anchor=ANCHOR_1E_APPLY_CALL_OLD,
                replacement=ANCHOR_1E_APPLY_CALL_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "ssm_state_indices",
            "has_initial_state",
            "torch.accelerator.device_index",
        ],
    )


def _make_gdn_linear_attn_patcher() -> TextPatcher | None:
    """Sub-3: gdn_linear_attn.py — gather/scatter elimination at call site.

    Drift detection: the bare token `ssm_state_indices=non_spec_state_indices_tensor`
    is too generic — pristine already uses it in the decode-path and spec-path
    branches (lines ~1024 and ~1106). For drift detection on the prefill
    branch (the one we patch), we rely on the fact that upstream merge of
    PR #41824 would REMOVE the `.contiguous()` gather call from the prefill
    block. If pristine no longer matches our OLD anchor (because upstream
    removed the gather), TextPatcher reports "required anchor not found" —
    which surfaces operationally as a clean "missing anchor" skip.

    No upstream_drift_markers configured: pristine pollution makes any
    marker we'd pick fire on a never-patched file.
    """
    target = resolve_vllm_file("model_executor/layers/mamba/gdn_linear_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN79 Sub-3 gdn_linear_attn.py (forward_cuda + forward_native + gather/scatter elim)",
        target_file=str(target),
        marker=GENESIS_PN79_MARKER,
        patch_id="PN79.Sub-3",
        sub_patches=[
            TextPatch(
                name="3A_forward_cuda_add_kwargs_and_fi_fallback",
                anchor=ANCHOR_3A_FORWARD_CUDA_OLD,
                replacement=ANCHOR_3A_FORWARD_CUDA_NEW,
                required=True,
            ),
            TextPatch(
                name="3B_forward_native_add_kwargs_passthrough",
                anchor=ANCHOR_3B_FORWARD_NATIVE_OLD,
                replacement=ANCHOR_3B_FORWARD_NATIVE_NEW,
                required=True,
            ),
            TextPatch(
                name="3C_forward_core_remove_gather_scatter",
                anchor=ANCHOR_3C_GATHER_SCATTER_OLD,
                replacement=ANCHOR_3C_GATHER_SCATTER_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[],
    )


def _make_olmo_hybrid_patcher() -> TextPatcher | None:
    """Sub-4: olmo_hybrid.py — gather/scatter elimination (analogous to 3C).

    Returns None gracefully when olmo_hybrid.py is not present in the
    target vllm install. This is the COMMON case on Genesis fleet
    (no Olmo models in model_configs/) — patch is structural completeness
    for community users.

    When file IS present, applies single 4A anchor. Atomic with the
    rest of PN79 transaction: if 1A-3C succeed but 4A fails, all 4
    files roll back via MultiFilePatchTransaction.
    """
    target = resolve_vllm_file("model_executor/models/olmo_hybrid.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN79 Sub-4 olmo_hybrid.py (gather/scatter elim)",
        target_file=str(target),
        marker=GENESIS_PN79_MARKER,
        patch_id="PN79.Sub-4",
        sub_patches=[
            TextPatch(
                name="4A_olmo_forward_core_remove_gather_scatter",
                anchor=ANCHOR_4A_OLMO_FORWARD_CORE_OLD,
                replacement=ANCHOR_4A_OLMO_FORWARD_CORE_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[],
    )


def _make_chunk_delta_h_patcher() -> TextPatcher | None:
    """Sub-2: chunk_delta_h.py — Triton kernel + Python wrapper.

    7 anchor sub-patches applied in top-to-bottom file order:
      2A heuristics dict, 2B kernel signature, 2C kernel main flow,
      2D kernel epilogue, 2E wrapper signature, 2F wrapper body strides,
      2G wrapper kernel-call kwargs.

    Order matters: 2C removes `if STORE_FINAL_STATE: ht = ht + i_nh * V * K`
    from the pre-load region, after which 2D's anchor (`# epilogue\\n
    if STORE_FINAL_STATE:`) is the unique remaining match in the file.
    """
    target = resolve_vllm_file("model_executor/layers/fla/ops/chunk_delta_h.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN79 Sub-2 chunk_delta_h.py (Triton kernel + wrapper)",
        target_file=str(target),
        marker=GENESIS_PN79_MARKER,
        patch_id="PN79.Sub-2",
        sub_patches=[
            TextPatch(
                name="2A_heuristics_add_ContinuousBatching_and_InitialStateMask",
                anchor=ANCHOR_2A_HEURISTICS_OLD,
                replacement=ANCHOR_2A_HEURISTICS_NEW,
                required=True,
            ),
            TextPatch(
                name="2B_kernel_sig_add_params_strides_constexpr",
                anchor=ANCHOR_2B_KERNEL_SIG_OLD,
                replacement=ANCHOR_2B_KERNEL_SIG_NEW,
                required=True,
            ),
            TextPatch(
                name="2C_kernel_main_flow_should_load_branch",
                anchor=ANCHOR_2C_KERNEL_MAIN_OLD,
                replacement=ANCHOR_2C_KERNEL_MAIN_NEW,
                required=True,
            ),
            TextPatch(
                name="2D_kernel_epilogue_ht_offset_branch",
                anchor=ANCHOR_2D_KERNEL_EPILOGUE_OLD,
                replacement=ANCHOR_2D_KERNEL_EPILOGUE_NEW,
                required=True,
            ),
            TextPatch(
                name="2E_wrapper_sig_add_indices_and_mask",
                anchor=ANCHOR_2E_WRAPPER_SIG_OLD,
                replacement=ANCHOR_2E_WRAPPER_SIG_NEW,
                required=True,
            ),
            TextPatch(
                name="2F_wrapper_body_strides_and_final_state_alias",
                anchor=ANCHOR_2F_WRAPPER_BODY_OLD,
                replacement=ANCHOR_2F_WRAPPER_BODY_NEW,
                required=True,
            ),
            TextPatch(
                name="2G_wrapper_kernel_call_pass_strides",
                anchor=ANCHOR_2G_WRAPPER_KERNEL_CALL_OLD,
                replacement=ANCHOR_2G_WRAPPER_KERNEL_CALL_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "IS_CONTINUOUS_BATCHING",
            "HAS_INITIAL_STATE_MASK",
            "stride_init_state_token",
        ],
    )


# ════════════════════════════════════════════════════════════════════════
# apply()
# ════════════════════════════════════════════════════════════════════════


def apply() -> tuple[str, str]:
    """Apply PN79 atomically across 3 files via MultiFilePatchTransaction.

    Current state 2026-05-07:
      - Sub-1 (chunk.py) — 7 anchors READY (1A/1B/1C/1D/1E_SIG/1E_VAL/1E_APPLY_CALL)
      - Sub-2 (chunk_delta_h.py) — 7 anchors READY (2A through 2G)
      - Sub-3 (gdn_linear_attn.py) — 3 anchors READY (3A/3B/3C)
      - Sub-4 (olmo_hybrid.py) — 1 anchor READY (4A) — applies only if file
        present (Genesis fleet has no Olmo; community completeness).

    Atomic up-to-18-anchor commit (17 always + 1 conditional). If ANY anchor fails dry-run (anchor not found
    OR drift marker present already → upstream merged), entire transaction
    rolls back and reports skipped reason. Operator never sees half-patched
    state which would crash boot (orchestrator passing ssm_state_indices
    to a kernel that doesn't accept it).

    Sub-3 sub-patch sequencing matters: 3A and 3B both edit pristine pattern
    `chunk_indices: ... = None,\\n        chunk_offsets: ... = None,\\n
    use_qk_l2norm_in_kernel: bool = True,\\n    ):\\n        return X(`
    where X differs (fi_chunk_gated_delta_rule vs fla_chunk_gated_delta_rule).
    They're applied serially against unique returns; PN79 marker insertion
    blocks 3B from re-firing on already-patched 3A region.
    """
    from vllm.sndr_core.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN79")
    log_decision("PN79", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    chunk_patcher = _make_chunk_patcher()
    chunk_delta_h_patcher = _make_chunk_delta_h_patcher()
    gdn_attn_patcher = _make_gdn_linear_attn_patcher()
    olmo_hybrid_patcher = _make_olmo_hybrid_patcher()  # may be None on builds without Olmo

    # Atomic up-to-4-file transaction. olmo_hybrid is None on most Genesis
    # builds (no Olmo models) — MultiFilePatchTransaction tolerates None entries
    # by skipping them gracefully (single None == structural absence, not failure).
    patchers = [chunk_patcher, chunk_delta_h_patcher, gdn_attn_patcher]
    if olmo_hybrid_patcher is not None:
        patchers.append(olmo_hybrid_patcher)
    txn = MultiFilePatchTransaction(patchers, name="PN79")
    return txn.apply_or_skip()


# ════════════════════════════════════════════════════════════════════════
# Build-time manifest registration (P2.1 Site Map)
# ════════════════════════════════════════════════════════════════════════
#
# `register_for_manifest()` is called by scripts/build_anchor_manifest.py
# at BUILD TIME (not runtime) to enroll PN79's patchers into the Site Map
# anchor offset manifest. It constructs the same TextPatcher objects but
# pointed at PRISTINE FIXTURE paths under tests/pristine_fixtures/ so it
# works without a vllm install (Mac dev / CI gate).
#
# Runtime apply() path (above) is unaffected — it still uses
# resolve_vllm_file() to find the live vllm install. The two paths are
# orthogonal.


def _make_patcher_for_fixture(name: str, fixture_path, sub_patches,
                              patch_id=None, drift_markers=()) -> "TextPatcher":
    """Build a TextPatcher targeting a pristine fixture file (build mode)."""
    return TextPatcher(
        patch_name=name,
        target_file=str(fixture_path),
        marker=GENESIS_PN79_MARKER,
        sub_patches=sub_patches,
        upstream_drift_markers=list(drift_markers),
        patch_id=patch_id,
    )


def register_for_manifest(*, pristine_root) -> None:
    """Register PN79's 4 sub-patchers into the Site Map registry, using
    pristine fixtures from `pristine_root`.

    Called by `scripts/build_anchor_manifest.py`. Idempotent: re-calling
    with the same patchers is a no-op. Different `pristine_root` between
    calls would raise ValueError (different patcher object, same id).

    Args:
        pristine_root: Path to pristine_fixtures/ directory containing
            chunk.py, chunk_delta_h.py, gdn_linear_attn.py, olmo_hybrid.py.
    """
    from vllm.sndr_core.wiring.patcher_registry import register_text_patcher

    chunk_subs = [
        TextPatch(name="1A", anchor=ANCHOR_1A_IMPORT_OLD,
                  replacement=ANCHOR_1A_IMPORT_NEW, required=True),
        TextPatch(name="1B", anchor=ANCHOR_1B_FWD_SIG_OLD,
                  replacement=ANCHOR_1B_FWD_SIG_NEW, required=True),
        TextPatch(name="1C", anchor=ANCHOR_1C_FWD_INTERNAL_OLD,
                  replacement=ANCHOR_1C_FWD_INTERNAL_NEW, required=True),
        TextPatch(name="1D", anchor=ANCHOR_1D_FORWARD_OLD,
                  replacement=ANCHOR_1D_FORWARD_NEW, required=True),
        TextPatch(name="1E_SIG", anchor=ANCHOR_1E_SIG_OLD,
                  replacement=ANCHOR_1E_SIG_NEW, required=True),
        TextPatch(name="1E_VAL", anchor=ANCHOR_1E_VAL_OLD,
                  replacement=ANCHOR_1E_VAL_NEW, required=True),
        TextPatch(name="1E_APPLY_CALL", anchor=ANCHOR_1E_APPLY_CALL_OLD,
                  replacement=ANCHOR_1E_APPLY_CALL_NEW, required=True),
    ]
    kernel_subs = [
        TextPatch(name="2A", anchor=ANCHOR_2A_HEURISTICS_OLD,
                  replacement=ANCHOR_2A_HEURISTICS_NEW, required=True),
        TextPatch(name="2B", anchor=ANCHOR_2B_KERNEL_SIG_OLD,
                  replacement=ANCHOR_2B_KERNEL_SIG_NEW, required=True),
        TextPatch(name="2C", anchor=ANCHOR_2C_KERNEL_MAIN_OLD,
                  replacement=ANCHOR_2C_KERNEL_MAIN_NEW, required=True),
        TextPatch(name="2D", anchor=ANCHOR_2D_KERNEL_EPILOGUE_OLD,
                  replacement=ANCHOR_2D_KERNEL_EPILOGUE_NEW, required=True),
        TextPatch(name="2E", anchor=ANCHOR_2E_WRAPPER_SIG_OLD,
                  replacement=ANCHOR_2E_WRAPPER_SIG_NEW, required=True),
        TextPatch(name="2F", anchor=ANCHOR_2F_WRAPPER_BODY_OLD,
                  replacement=ANCHOR_2F_WRAPPER_BODY_NEW, required=True),
        TextPatch(name="2G", anchor=ANCHOR_2G_WRAPPER_KERNEL_CALL_OLD,
                  replacement=ANCHOR_2G_WRAPPER_KERNEL_CALL_NEW, required=True),
    ]
    gdn_subs = [
        TextPatch(name="3A", anchor=ANCHOR_3A_FORWARD_CUDA_OLD,
                  replacement=ANCHOR_3A_FORWARD_CUDA_NEW, required=True),
        TextPatch(name="3B", anchor=ANCHOR_3B_FORWARD_NATIVE_OLD,
                  replacement=ANCHOR_3B_FORWARD_NATIVE_NEW, required=True),
        TextPatch(name="3C", anchor=ANCHOR_3C_GATHER_SCATTER_OLD,
                  replacement=ANCHOR_3C_GATHER_SCATTER_NEW, required=True),
    ]
    olmo_subs = [
        TextPatch(name="4A", anchor=ANCHOR_4A_OLMO_FORWARD_CORE_OLD,
                  replacement=ANCHOR_4A_OLMO_FORWARD_CORE_NEW, required=True),
    ]

    register_text_patcher(
        "PN79.Sub-1",
        _make_patcher_for_fixture(
            "PN79 Sub-1 chunk.py (build mode)",
            pristine_root / "chunk.py", chunk_subs,
            patch_id="PN79.Sub-1",
            drift_markers=("ssm_state_indices", "has_initial_state",
                           "torch.accelerator.device_index"),
        ),
    )
    register_text_patcher(
        "PN79.Sub-2",
        _make_patcher_for_fixture(
            "PN79 Sub-2 chunk_delta_h.py (build mode)",
            pristine_root / "chunk_delta_h.py", kernel_subs,
            patch_id="PN79.Sub-2",
            drift_markers=("IS_CONTINUOUS_BATCHING", "HAS_INITIAL_STATE_MASK",
                           "stride_init_state_token"),
        ),
    )
    register_text_patcher(
        "PN79.Sub-3",
        _make_patcher_for_fixture(
            "PN79 Sub-3 gdn_linear_attn.py (build mode)",
            pristine_root / "gdn_linear_attn.py", gdn_subs,
            patch_id="PN79.Sub-3",
        ),
    )
    register_text_patcher(
        "PN79.Sub-4",
        _make_patcher_for_fixture(
            "PN79 Sub-4 olmo_hybrid.py (build mode)",
            pristine_root / "olmo_hybrid.py", olmo_subs,
            patch_id="PN79.Sub-4",
        ),
    )
