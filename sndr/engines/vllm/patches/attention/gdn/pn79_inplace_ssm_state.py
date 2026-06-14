# SPDX-License-Identifier: Apache-2.0
"""PN79 — In-place SSM state for GDN chunk prefill (vllm#41824 backport).

STATUS 2026-06-10: PARKED after PROD IMA — kernel-level bug in the
re-anchored in-place path.
====================================================================
K.2 re-anchor applied cleanly (18/18 sub-patches, 4 files atomic) and
short decode traffic was healthy (accuracy spot-check passed), but the
FIRST 8K chunked prefill produced CUDA illegal memory access
(async-reported at rejection_sampler; engine death). The in-place
chunk_delta_h kernel (state_idx rebase / stride math on the
IS_CONTINUOUS_BATCHING branch) is the suspect — exactly the multi-chunk
prefill path the patch modifies. Reverted on PROD (text + env).

Revert lesson (operational): docker start boots with the OLD launcher
env — flip the env BEFORE any start/restart, or the boot apply re-races
the revert (caught twice tonight: PN352, PN79).

To resume: reproduce in isolation with a standalone multi-chunk varlen
test against the pristine kernel (cache_steps > 1, multiple sequences,
ssm_state_indices permuted), compare in-place vs gather/scatter outputs
per chunk, and audit the 2D/2F stride constexprs (stride_init_state_*)
against the actual ssm_state layout [num_blocks, H, V, K] vs the
kernel's assumed [N, H, V, K].

Wiring for PN79 — in-place SSM state for GDN chunk prefill (vllm#41824 backport).

================================================================
What it does
================================================================

Backport of vllm-project/vllm#41824 (Kermit-C, OPEN as of 2026-06-10,
rebased upstream on 2026-06-09 on top of merged #44700). Eliminates the
per-prefill-step gather (`ssm_state[indices]` + zero-fill copy) and
scatter (`ssm_state[indices] = final_state`) copies in the GDN chunk
prefill path, by passing `ssm_state_indices` directly to the Triton
kernel. The kernel uses the `IS_CONTINUOUS_BATCHING` constexpr to
read/write the global SSM cache pool in place via pointer arithmetic;
`HAS_INITIAL_STATE_MASK` skips the initial-state load for sequences
without prior state (replacing the explicit zero-fill).

Author claims 4.5-36 GiB cumulative fp32 traffic eliminated per
multi-turn session (Qwen3.5-0.8B → Qwen3.6-27B scale).

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

4. **Decoupling preserves option value**: PN59 stays as separate patch.
   conflicts_with [PN59, PN54] in the PN79 registry entry enforces
   mutual exclusion at apply time.

================================================================
Four sub-patches, atomic via MultiFilePatchTransaction
================================================================

Sub-1: `vllm/model_executor/layers/fla/ops/chunk.py` (8 anchors)
   1B: chunk_gated_delta_rule_fwd signature — add ssm_state_indices,
       has_initial_state (between chunk_offsets and core_attn_out,
       matching upstream parameter order)
   1C: chunk_gated_delta_rule_fwd internal call to fwd_h — pass kwargs
   1D_DECORATOR: drop @input_guard from ChunkGatedDeltaRuleFunction.forward
       (the import of input_guard in chunk.py is KEPT — smaller diff;
       upstream removes it, the leftover import is harmless at runtime)
   1D_FORWARD_SIG: forward signature + manual-contiguity block (replaces
       @input_guard semantics; skips .contiguous() on initial_state when
       ssm_state_indices given). Upstream's torch.accelerator.device_index
       wrapper is intentionally NOT ported: Genesis TP workers are
       single-device per process, the context is a no-op there, and
       skipping it keeps the body at original indentation (smaller diff).
   1D_FORWARD_CALL: forward's inner chunk_gated_delta_rule_fwd call kwargs
   1E_SIG / 1E_VAL / 1E_APPLY_CALL: high-level chunk_gated_delta_rule API —
       signature, ValueError guard relaxed with `ssm_state_indices is None`,
       .apply() positional args (order MUST mirror forward's signature)

Sub-2: `vllm/model_executor/layers/fla/ops/chunk_delta_h.py` (7 anchors)
   2A: @triton.heuristics dict — IS_CONTINUOUS_BATCHING + HAS_INITIAL_STATE_MASK
   2B: kernel @triton.jit signature — 2 ptr params + 4 stride constexprs
       + 2 constexpr flags (post-#44700 pre-image includes USE_G/USE_GK
       and USE_EXP2)
   2C: kernel main flow USE_INITIAL_STATE — should_load + state_idx rebase
   2D: kernel epilogue STORE_FINAL_STATE — state_idx rebase for ht
   2E: chunk_gated_delta_rule_fwd_h Python wrapper signature
   2F: wrapper body — when ssm_state_indices given, the initial_state pool
       IS the final-state storage (no fp32 new_empty); strides if/else
   2G: wrapper kernel-call kwargs + stride constexprs

Sub-3: `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` (2 anchors)
   3B: ChunkGatedDeltaRule.forward_native — kwargs passthrough to
       fla_chunk_gated_delta_rule (the Triton/FLA backend method)
   3C: QwenGatedDeltaNetAttention._forward_core prefill block — THE WIN
       SITE. Backend-gated: in-place kwargs are passed ONLY when
       `self.gdn_prefill_backend == "triton"` (verified attribute +
       Literal value on live pin g303916e93). flashinfer / cutedsl
       backends keep the upstream gather/scatter verbatim — their
       kernels do not accept the in-place kwargs. This deliberately
       DIVERGES from upstream #41824 (which also rewrites forward_cuda
       with a gather/scatter fallback): gating at the call site is
       strictly safer and leaves forward_cuda / forward_cutedsl untouched.

Sub-4: `vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py` (1 anchor)
   4A: OlmoGatedDeltaNetAttention prefill block — upstream-verbatim
       gather/scatter elimination. Olmo always calls the FLA
       chunk_gated_delta_rule free function (no backend dispatch), so
       no gating is needed. Never fires on Genesis fleet (no Olmo
       models) — structural completeness for community builds.

================================================================
K.2 re-anchor 2026-06-10 — pin 0.22.1rc1.dev259+g303916e93
================================================================

All anchors re-derived for the current pin (upstream commit 2026-06-08,
contains merged #44700 GDN mixed-batch split + core_attn_out buffer
reuse). Every OLD anchor below was verified byte-exact against the LIVE
PROD container files (vllm-qwen3.6-35b-balanced-k3, with PN59 / P103 /
PN106 / PN345 Genesis blocks applied) AND against upstream #41824's
rebased pre-image (gh pr diff 41824, fetched 2026-06-10):

  * Sub-1 chunk.py: pre-image gained `core_attn_out` param threading
    (#44700) — 1B/1C/1D/1E anchors rewritten around it. The old
    monolithic 1D forward rewrite is split into 3 small anchors
    (decorator / signature+contiguity / inner call) so PN59's injected
    streaming block (which sits between the fwd signature and the fwd_h
    call) cannot collide.
  * Sub-2 chunk_delta_h.py: kernel signature gained USE_EXP2 (and the
    pre-existing USE_G/USE_GK); wrapper gained use_exp2. 2B/2E/2G
    rewritten; 2A/2C/2D survive unchanged (verified byte-equivalent).
    2F shrunk to the final_state assignment only so it composes with
    PN106's injected h-pool block directly above it.
  * Sub-3: old target `mamba/gdn_linear_attn.py` was split upstream
    (#41126 era) into per-model files under `mamba/gdn/`. Re-anchored
    to qwen_gdn_linear_attn.py with the backend-gated design above.
    The old 3A forward_cuda anchor is dropped (no longer needed: the
    gate keeps the FlashInfer path upstream-identical).
  * Sub-4: old target `models/olmo_hybrid.py` STILL EXISTS on the new
    pin but no longer contains the linear-attn code (moved to
    mamba/gdn/olmo_gdn_linear_attn.py). Without re-anchoring, the old
    required 4A anchor would dry-run-fail and abort the WHOLE PN79
    transaction. Re-anchored to the new file, upstream-verbatim.

Runtime equivalence contract: with GENESIS_ENABLE_PN79_INPLACE_SSM_STATE
unset the files are untouched. With the patch applied but the new params
None (e.g. non-triton prefill backend, or callers that never pass them),
every code path is behavior-identical to upstream: the manual contiguity
block reproduces @input_guard's .contiguous() semantics (single-device
workers make the dropped device-index context a no-op), the kernel
constexpr branches collapse to the upstream pointer math, and the
wrapper allocates the same fp32 final_state buffer.

================================================================
Drift detection
================================================================

Patches auto-SKIP if upstream landed equivalent fix (drift markers):
  - "ssm_state_indices" / "has_initial_state" /
    "torch.accelerator.device_index"  — Sub-1 chunk.py
  - "IS_CONTINUOUS_BATCHING" / "HAS_INITIAL_STATE_MASK" /
    "stride_init_state_token"         — Sub-2 chunk_delta_h.py

Sub-3 / Sub-4 have NO drift markers: pristine pollution — the decode and
spec-decode branches of both files already use
`ssm_state_indices=...` kwargs for the fused_recurrent kernels.

================================================================
Conflict gating
================================================================

PATCH_REGISTRY entry has conflicts_with: ["PN59", "PN54"].
dispatcher.should_apply() will SKIP PN79 if either is enabled.
P103 coexists safely: its chunked_fwd wrapper bails out to the original
(patched) fwd whenever `ssm_state_indices` is passed (see
p103_fla_cliff2_chunked.py — guard extended 2026-06-10), so Cliff-2
chunking never silently drops in-place state semantics.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Backport of: Kermit-C vllm-project/vllm#41824 (OPEN as of 2026-06-10).
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import (
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

# 1B — chunk_gated_delta_rule_fwd signature: add ssm_state_indices +
# has_initial_state between chunk_offsets and core_attn_out (upstream
# order). Unique to fwd: the other two signature sites in the file have
# `use_qk_l2norm_in_kernel` between chunk_offsets and core_attn_out.
# No body tail in the anchor: on live files PN59's streaming block is
# injected right after `):` — anchor must match both pristine and
# PN59-patched files.
ANCHOR_1B_FWD_SIG_OLD = (
    "    cu_seqlens: torch.Tensor | None = None,\n"
    "    chunk_indices: torch.Tensor | None = None,\n"
    "    chunk_offsets: torch.Tensor | None = None,\n"
    "    core_attn_out: torch.Tensor | None = None,\n"
    "):\n"
)
ANCHOR_1B_FWD_SIG_NEW = (
    "    cu_seqlens: torch.Tensor | None = None,\n"
    "    chunk_indices: torch.Tensor | None = None,\n"
    "    chunk_offsets: torch.Tensor | None = None,\n"
    "    ssm_state_indices: torch.Tensor | None = None,\n"
    "    has_initial_state: torch.Tensor | None = None,\n"
    "    core_attn_out: torch.Tensor | None = None,\n"
    "):\n"
)

# 1C — internal call to chunk_gated_delta_rule_fwd_h: pass new kwargs
# through. The `o = chunk_fwd_o(` tail pins the match to the fwd_h call
# (PN59's streaming-driver call above uses 12-space indentation and a
# different following kwarg).
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

# 1D_DECORATOR — drop @input_guard (the only use in the file; the import
# is kept). @input_guard would force .contiguous() on EVERY tensor arg —
# including the full SSM cache pool passed as initial_state, which is
# exactly the copy this patch eliminates.
ANCHOR_1D_DECORATOR_OLD = (
    "    @staticmethod\n"
    "    @input_guard\n"
    "    @torch.amp.custom_fwd(device_type=\"cuda\")\n"
)
ANCHOR_1D_DECORATOR_NEW = (
    "    @staticmethod\n"
    "    # [Genesis PN79 vllm#41824] @input_guard dropped — manual contiguity\n"
    "    # inside forward() skips .contiguous() on initial_state when\n"
    "    # ssm_state_indices is given (kernel reads the SSM pool in place).\n"
    "    @torch.amp.custom_fwd(device_type=\"cuda\")\n"
)

# 1D_FORWARD_SIG — ChunkGatedDeltaRuleFunction.forward signature tail +
# manual contiguity block. 8-space parameter indentation is unique to
# forward (the high-level API uses 4-space). Body stays at original
# indentation: upstream's torch.accelerator.device_index wrapper is
# intentionally skipped (single-device TP workers; see module docstring).
ANCHOR_1D_FORWARD_SIG_OLD = (
    "        use_qk_l2norm_in_kernel: bool = False,\n"
    "        core_attn_out: torch.Tensor | None = None,\n"
    "    ):\n"
    "        if use_qk_l2norm_in_kernel:\n"
    "            q = l2norm_fwd(q)\n"
    "            k = l2norm_fwd(k)\n"
)
ANCHOR_1D_FORWARD_SIG_NEW = (
    "        use_qk_l2norm_in_kernel: bool = False,\n"
    "        ssm_state_indices: torch.Tensor | None = None,\n"
    "        has_initial_state: torch.Tensor | None = None,\n"
    "        core_attn_out: torch.Tensor | None = None,\n"
    "    ):\n"
    "        # [Genesis PN79 vllm#41824] Manually ensure contiguity instead of\n"
    "        # using @input_guard. Skip .contiguous() on initial_state when\n"
    "        # ssm_state_indices is provided: the kernel handles non-contiguous\n"
    "        # tensors via strides, and forcing contiguity on a large SSM cache\n"
    "        # view is expensive.\n"
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
    "        if use_qk_l2norm_in_kernel:\n"
    "            q = l2norm_fwd(q)\n"
    "            k = l2norm_fwd(k)\n"
)

# 1D_FORWARD_CALL — forward's inner chunk_gated_delta_rule_fwd call.
# 12-space indentation + `core_attn_out=core_attn_out,` right after
# chunk_offsets is unique to this call (PN59's streaming call has
# `chunk_local_cumsum=` after chunk_offsets; chunk_fwd_o's kwargs are
# 8-space).
ANCHOR_1D_FORWARD_CALL_OLD = (
    "            cu_seqlens=cu_seqlens,\n"
    "            chunk_indices=chunk_indices,\n"
    "            chunk_offsets=chunk_offsets,\n"
    "            core_attn_out=core_attn_out,\n"
    "        )\n"
)
ANCHOR_1D_FORWARD_CALL_NEW = (
    "            cu_seqlens=cu_seqlens,\n"
    "            chunk_indices=chunk_indices,\n"
    "            chunk_offsets=chunk_offsets,\n"
    "            ssm_state_indices=ssm_state_indices,\n"
    "            has_initial_state=has_initial_state,\n"
    "            core_attn_out=core_attn_out,\n"
    "        )\n"
)


# ─── 1E: chunk_gated_delta_rule (high-level API) — 3 sub-anchors ─────
# The public wrapper must accept ssm_state_indices / has_initial_state
# AND forward them to ChunkGatedDeltaRuleFunction.apply as positionals
# in the exact order of forward's signature. Without 1E, calls from
# qwen/olmo GDN layers (Sub-3/Sub-4) crash with TypeError.
ANCHOR_1E_SIG_OLD = (
    "    chunk_offsets: torch.Tensor | None = None,\n"
    "    use_qk_l2norm_in_kernel: bool = False,\n"
    "    core_attn_out: torch.Tensor | None = None,\n"
    "):\n"
    "    r\"\"\"\n"
)
ANCHOR_1E_SIG_NEW = (
    "    chunk_offsets: torch.Tensor | None = None,\n"
    "    use_qk_l2norm_in_kernel: bool = False,\n"
    "    ssm_state_indices: torch.Tensor | None = None,\n"
    "    has_initial_state: torch.Tensor | None = None,\n"
    "    core_attn_out: torch.Tensor | None = None,\n"
    "):\n"
    "    r\"\"\"\n"
)

# Pristine validation raises ValueError when N != len(cu_seqlens)-1.
# With PN79 the initial_state is the full ssm_state pool (N_pool rows);
# ssm_state_indices selects per-sequence rows, so N_pool != batch_size
# is expected. Validation must skip when ssm_state_indices given.
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
    "        use_qk_l2norm_in_kernel,\n"
    "        core_attn_out,\n"
    "    )\n"
    "    return o, final_state\n"
)
ANCHOR_1E_APPLY_CALL_NEW = (
    "        use_qk_l2norm_in_kernel,\n"
    "        ssm_state_indices,\n"
    "        has_initial_state,\n"
    "        core_attn_out,\n"
    "    )\n"
    "    return o, final_state\n"
)


# ════════════════════════════════════════════════════════════════════════
# Sub-2: chunk_delta_h.py — Triton kernel changes (HIGHEST RISK)
# ════════════════════════════════════════════════════════════════════════
#
# 7 anchor points (verified against live pin g303916e93 and PR #41824
# rebased diff fetched 2026-06-10):
#   2A: @triton.heuristics dict — add IS_CONTINUOUS_BATCHING + HAS_INITIAL_STATE_MASK
#   2B: kernel @triton.jit signature — add params + 4 strides + 2 constexpr flags
#   2C: kernel main flow USE_INITIAL_STATE — should_load + IS_CONTINUOUS_BATCHING
#   2D: kernel epilogue STORE_FINAL_STATE — IS_CONTINUOUS_BATCHING ht offset
#   2E: chunk_gated_delta_rule_fwd_h Python wrapper signature
#   2F: chunk_gated_delta_rule_fwd_h Python wrapper body — strides if/else
#   2G: chunk_gated_delta_rule_fwd_h Python wrapper kernel call kwargs
#
# Triton DSL care: whitespace/indent verified char-for-char from live pin.
# PN345's autotune pruner lives in the @triton.autotune decorator BETWEEN
# 2A and 2B — no overlap with either anchor. PN106's h-pool block sits
# directly ABOVE 2F's final_state anchor — 2F deliberately excludes the
# `h = ...` allocation so both patches compose.

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
# Post-#44700 pre-image: USE_G/USE_GK present, USE_EXP2 terminal.
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
    "    USE_EXP2: tl.constexpr,\n"
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
    "    USE_EXP2: tl.constexpr,\n"
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
    "    use_exp2: bool = False,\n"
    ") -> tuple[torch.Tensor, torch.Tensor]:\n"
)
ANCHOR_2E_WRAPPER_SIG_NEW = (
    "    cu_seqlens: torch.Tensor | None = None,\n"
    "    chunk_indices: torch.Tensor | None = None,\n"
    "    chunk_offsets: torch.Tensor | None = None,\n"
    "    ssm_state_indices: torch.Tensor | None = None,\n"
    "    has_initial_state: torch.Tensor | None = None,\n"
    "    use_exp2: bool = False,\n"
    ") -> tuple[torch.Tensor, torch.Tensor]:\n"
)

# ─── 2F: Python wrapper body — strides if/else + final_state alias ────
# Anchor covers ONLY the final_state assignment (it sits directly below
# PN106's injected h-pool block on live files; on pristine files it sits
# below the plain `h = k.new_empty(...)` line — both compose).
ANCHOR_2F_WRAPPER_BODY_OLD = (
    "    final_state = (\n"
    "        k.new_empty(N, H, V, K, dtype=torch.float32) if output_final_state else None\n"
    "    )\n"
)
ANCHOR_2F_WRAPPER_BODY_NEW = (
    "    # [Genesis PN79 vllm#41824] In-place SSM state: when ssm_state_indices\n"
    "    # is provided, the caller passes the global SSM state pool as\n"
    "    # initial_state; the kernel reads and writes it in place, so the pool\n"
    "    # itself doubles as final-state storage (no fp32 gather/scatter copy).\n"
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
    "        USE_EXP2=use_exp2,\n"
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
    "        USE_EXP2=use_exp2,\n"
    "    )\n"
)


# ════════════════════════════════════════════════════════════════════════
# Sub-3: qwen_gdn_linear_attn.py anchors (THE WIN SITE)
# ════════════════════════════════════════════════════════════════════════

# ─── 3B: ChunkGatedDeltaRule.forward_native — kwargs passthrough ──────
# The Triton/FLA backend method (active_backend == "triton" binds
# self._forward_method = self.forward_native). Full def block as a
# single anchor: `return fla_chunk_gated_delta_rule(` makes it unique
# (forward_cuda returns fi_*, forward_cutedsl calls the cutedsl op).
ANCHOR_3B_FORWARD_NATIVE_OLD = (
    "    def forward_native(\n"
    "        self,\n"
    "        q: torch.Tensor,\n"
    "        k: torch.Tensor,\n"
    "        v: torch.Tensor,\n"
    "        g: torch.Tensor,\n"
    "        beta: torch.Tensor,\n"
    "        initial_state: torch.Tensor,\n"
    "        output_final_state: bool,\n"
    "        cu_seqlens: torch.Tensor | None = None,\n"
    "        chunk_indices: torch.Tensor | None = None,\n"
    "        chunk_offsets: torch.Tensor | None = None,\n"
    "        use_qk_l2norm_in_kernel: bool = True,\n"
    "        core_attn_out: torch.Tensor | None = None,\n"
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
    "            core_attn_out=core_attn_out,\n"
    "        )\n"
)
ANCHOR_3B_FORWARD_NATIVE_NEW = (
    "    def forward_native(\n"
    "        self,\n"
    "        q: torch.Tensor,\n"
    "        k: torch.Tensor,\n"
    "        v: torch.Tensor,\n"
    "        g: torch.Tensor,\n"
    "        beta: torch.Tensor,\n"
    "        initial_state: torch.Tensor,\n"
    "        output_final_state: bool,\n"
    "        cu_seqlens: torch.Tensor | None = None,\n"
    "        chunk_indices: torch.Tensor | None = None,\n"
    "        chunk_offsets: torch.Tensor | None = None,\n"
    "        use_qk_l2norm_in_kernel: bool = True,\n"
    "        ssm_state_indices: torch.Tensor | None = None,\n"
    "        has_initial_state: torch.Tensor | None = None,\n"
    "        core_attn_out: torch.Tensor | None = None,\n"
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
    "            core_attn_out=core_attn_out,\n"
    "        )\n"
)

# ─── 3C: _forward_core prefill block — backend-gated in-place state ───
# Replaces the gather (`ssm_state[indices]` + zero-fill) and scatter
# (`ssm_state[indices] = final`) ONLY for the Triton/FLA backend.
# flashinfer / cutedsl keep the upstream path verbatim (their kernels
# do not accept the in-place kwargs — passing them would TypeError).
# The empty `**_pn79_kwargs` expansion on the non-triton path is a
# no-op, keeping the call bit-identical to upstream.
ANCHOR_3C_PREFILL_INPLACE_OLD = (
    "            prefill_state_indices = attn_metadata.prefill_state_indices\n"
    "            prefill_has_initial_state = attn_metadata.prefill_has_initial_state\n"
    "            assert prefill_state_indices is not None\n"
    "            assert prefill_has_initial_state is not None\n"
    "            initial_state = ssm_state[prefill_state_indices]\n"
    "            initial_state[~prefill_has_initial_state, ...] = 0\n"
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
    "                cu_seqlens=attn_metadata.prefill_query_start_loc,\n"
    "                chunk_indices=attn_metadata.chunk_indices,\n"
    "                chunk_offsets=attn_metadata.chunk_offsets,\n"
    "                use_qk_l2norm_in_kernel=False,\n"
    "            )\n"
    "            # Init cache\n"
    "            ssm_state[prefill_state_indices] = last_recurrent_state.to(ssm_state.dtype)\n"
)
ANCHOR_3C_PREFILL_INPLACE_NEW = (
    "            prefill_state_indices = attn_metadata.prefill_state_indices\n"
    "            prefill_has_initial_state = attn_metadata.prefill_has_initial_state\n"
    "            assert prefill_state_indices is not None\n"
    "            assert prefill_has_initial_state is not None\n"
    "            # [Genesis PN79 vllm#41824] In-place SSM state for the Triton/FLA\n"
    "            # prefill backend: pass the global state pool plus\n"
    "            # ssm_state_indices / has_initial_state so the chunk kernel reads\n"
    "            # and writes per-sequence states in place (IS_CONTINUOUS_BATCHING).\n"
    "            # Other prefill backends (flashinfer, cutedsl) keep the upstream\n"
    "            # gather/scatter path: their kernels do not accept these kwargs.\n"
    "            _pn79_inplace = self.gdn_prefill_backend == \"triton\"\n"
    "            if _pn79_inplace:\n"
    "                initial_state = ssm_state\n"
    "                _pn79_kwargs = {\n"
    "                    \"ssm_state_indices\": prefill_state_indices,\n"
    "                    \"has_initial_state\": prefill_has_initial_state,\n"
    "                }\n"
    "            else:\n"
    "                initial_state = ssm_state[prefill_state_indices]\n"
    "                initial_state[~prefill_has_initial_state, ...] = 0\n"
    "                _pn79_kwargs = {}\n"
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
    "                cu_seqlens=attn_metadata.prefill_query_start_loc,\n"
    "                chunk_indices=attn_metadata.chunk_indices,\n"
    "                chunk_offsets=attn_metadata.chunk_offsets,\n"
    "                use_qk_l2norm_in_kernel=False,\n"
    "                **_pn79_kwargs,\n"
    "            )\n"
    "            # Init cache\n"
    "            # [Genesis PN79] The Triton kernel already stored the final state\n"
    "            # in place via ssm_state_indices; skip the scatter copy then.\n"
    "            if not _pn79_inplace:\n"
    "                ssm_state[prefill_state_indices] = last_recurrent_state.to(\n"
    "                    ssm_state.dtype\n"
    "                )\n"
)


# ════════════════════════════════════════════════════════════════════════
# Sub-4: olmo_gdn_linear_attn.py — gather/scatter elimination
# ════════════════════════════════════════════════════════════════════════
#
# Olmo-Hybrid's GDN layer always calls the FLA chunk_gated_delta_rule
# free function (imported from vllm.model_executor.layers.fla.ops — the
# file Sub-1 patches), with no backend dispatch. Upstream-verbatim port
# of #41824's olmo_gdn_linear_attn.py hunk; no gating needed.
#
# Never fires on Genesis fleet (no Olmo models in model_configs/) —
# structural completeness for community builds.
ANCHOR_4A_OLMO_PREFILL_OLD = (
    "        if attn_metadata.num_prefills > 0:\n"
    "            assert non_spec_state_indices_tensor is not None\n"
    "            assert has_initial_state is not None\n"
    "            assert non_spec_query_start_loc is not None\n"
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
ANCHOR_4A_OLMO_PREFILL_NEW = (
    "        if attn_metadata.num_prefills > 0:\n"
    "            assert non_spec_state_indices_tensor is not None\n"
    "            assert has_initial_state is not None\n"
    "            assert non_spec_query_start_loc is not None\n"
    "            # [Genesis PN79 vllm#41824] Olmo-Hybrid always uses the Triton/FLA\n"
    "            # chunk kernel for prefill — gather/scatter eliminated\n"
    "            # unconditionally, mirroring upstream #41824.\n"
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
                name="1D_drop_input_guard_decorator",
                anchor=ANCHOR_1D_DECORATOR_OLD,
                replacement=ANCHOR_1D_DECORATOR_NEW,
                required=True,
            ),
            TextPatch(
                name="1D_forward_sig_add_params_manual_contiguity",
                anchor=ANCHOR_1D_FORWARD_SIG_OLD,
                replacement=ANCHOR_1D_FORWARD_SIG_NEW,
                required=True,
            ),
            TextPatch(
                name="1D_forward_inner_call_pass_kwargs",
                anchor=ANCHOR_1D_FORWARD_CALL_OLD,
                replacement=ANCHOR_1D_FORWARD_CALL_NEW,
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
            # Self-collision lint (triage plan §6 2026-06-11): former
            # entries "ssm_state_indices" / "has_initial_state" are baked
            # verbatim by our own vllm#41824 port (signature + kwargs
            # sub-patches) — they cannot distinguish a real upstream merge
            # from our residue (false "upstream_merged" skip, PN369 class).
            # A real #41824 merge is caught by the strictly-upstream-only
            # marker below + required-anchor mismatch (Layer 5).
            "torch.accelerator.device_index",
        ],
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
        # Self-collision lint (triage plan §6 2026-06-11): former entries
        # "IS_CONTINUOUS_BATCHING" / "HAS_INITIAL_STATE_MASK" /
        # "stride_init_state_token" are baked verbatim by our own kernel
        # sub-patches (2A-2G) — they cannot distinguish a real upstream
        # merge from our residue (false "upstream_merged" skip, PN369
        # class). Real-merge detection is delegated to required-anchor
        # mismatch (Layer 5) + pin-bump preflight deep-diff.
        upstream_drift_markers=[],
    )


def _make_qwen_gdn_patcher() -> TextPatcher | None:
    """Sub-3: qwen_gdn_linear_attn.py — backend-gated in-place state.

    K.2 re-anchor (2026-06-10): targets the per-model file introduced by
    the upstream gdn/ split. No fallback to the retired monolithic
    `mamba/gdn_linear_attn.py` — its anchors are structurally gone.

    No upstream_drift_markers: pristine pollution — the decode-path and
    spec-path branches already use `ssm_state_indices=...` kwargs for
    fused_recurrent kernels. If upstream merges #41824, the 3C OLD
    anchor (gather line `initial_state = ssm_state[prefill_state_indices]`)
    disappears and the patch surfaces a clean "missing anchor" skip.
    """
    target = resolve_vllm_file(
        "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
    )
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN79 Sub-3 qwen_gdn_linear_attn.py "
            "(forward_native passthrough + gated gather/scatter elim)"
        ),
        target_file=str(target),
        marker=GENESIS_PN79_MARKER,
        patch_id="PN79.Sub-3",
        sub_patches=[
            TextPatch(
                name="3B_forward_native_add_kwargs_passthrough",
                anchor=ANCHOR_3B_FORWARD_NATIVE_OLD,
                replacement=ANCHOR_3B_FORWARD_NATIVE_NEW,
                required=True,
            ),
            TextPatch(
                name="3C_prefill_backend_gated_inplace_state",
                anchor=ANCHOR_3C_PREFILL_INPLACE_OLD,
                replacement=ANCHOR_3C_PREFILL_INPLACE_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[],
    )


def _make_olmo_gdn_patcher() -> TextPatcher | None:
    """Sub-4: olmo_gdn_linear_attn.py — gather/scatter elimination.

    Returns None gracefully when the file is not present in the target
    vllm install. When present, applies the single 4A anchor. Atomic
    with the rest of the PN79 transaction.

    K.2 re-anchor (2026-06-10): the old target `models/olmo_hybrid.py`
    STILL EXISTS on pin g303916e93 but no longer contains the GDN
    linear-attn code — keeping the old patcher would dry-run-fail its
    required anchor and abort the whole transaction.
    """
    target = resolve_vllm_file(
        "model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py"
    )
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN79 Sub-4 olmo_gdn_linear_attn.py (gather/scatter elim)",
        target_file=str(target),
        marker=GENESIS_PN79_MARKER,
        patch_id="PN79.Sub-4",
        sub_patches=[
            TextPatch(
                name="4A_olmo_prefill_remove_gather_scatter",
                anchor=ANCHOR_4A_OLMO_PREFILL_OLD,
                replacement=ANCHOR_4A_OLMO_PREFILL_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[],
    )


# ════════════════════════════════════════════════════════════════════════
# apply()
# ════════════════════════════════════════════════════════════════════════


def apply() -> tuple[str, str]:
    """Apply PN79 atomically across 4 files via MultiFilePatchTransaction.

    State after K.2 re-anchor 2026-06-10 (pin g303916e93, post-#44700):
      - Sub-1 (chunk.py) — 8 anchors (1B/1C/1D_DECORATOR/1D_FORWARD_SIG/
        1D_FORWARD_CALL/1E_SIG/1E_VAL/1E_APPLY_CALL)
      - Sub-2 (chunk_delta_h.py) — 7 anchors (2A through 2G)
      - Sub-3 (qwen_gdn_linear_attn.py) — 2 anchors (3B/3C, backend-gated)
      - Sub-4 (olmo_gdn_linear_attn.py) — 1 anchor (4A) — applies only if
        file present (Genesis fleet has no Olmo; community completeness).

    Atomic up-to-18-anchor commit (17 always + 1 conditional). If ANY
    anchor fails dry-run (anchor not found OR drift marker present →
    upstream merged), the entire transaction rolls back and reports the
    skipped reason. Operator never sees half-patched state which would
    crash boot (call sites passing ssm_state_indices to a kernel that
    doesn't accept it).
    """
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN79")
    log_decision("PN79", decision, reason)
    if not decision:
        return "skipped", reason

    # PARKED hard-refusal (deep-audit 2026-06-14 #4). PN79 is parked after a
    # reproduced PROD CUDA illegal-memory-access on the FIRST 8K chunked
    # prefill (the in-place chunk_delta_h kernel on the IS_CONTINUOUS_BATCHING
    # branch — see the STATUS note at the top of this module). The 18 required
    # anchors apply cleanly, so a single GENESIS_ENABLE_PN79_INPLACE_SSM_STATE=1
    # in a launcher would silently re-introduce the documented crash. Refuse to
    # apply unless the operator ALSO sets an explicit, un-guessable override
    # that names the risk — intended only for the isolation-repro workflow in
    # the docstring, never a production launcher.
    force_reenable = os.environ.get(
        "GENESIS_PN79_FORCE_REENABLE_DESPITE_PROD_IMA", ""
    ).strip().lower() in ("1", "true", "yes", "on")
    if not force_reenable:
        return "skipped", (
            "PN79 PARKED after PROD IMA (CUDA illegal memory access on the "
            "first 8K chunked prefill; in-place chunk_delta_h kernel suspect). "
            "Refusing to apply the 18 known-crashing anchors despite "
            "GENESIS_ENABLE_PN79_INPLACE_SSM_STATE=1. Set "
            "GENESIS_PN79_FORCE_REENABLE_DESPITE_PROD_IMA=1 ONLY for the "
            "isolation repro (see module docstring); never in a prod launcher."
        )

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    chunk_patcher = _make_chunk_patcher()
    chunk_delta_h_patcher = _make_chunk_delta_h_patcher()
    qwen_gdn_patcher = _make_qwen_gdn_patcher()
    olmo_gdn_patcher = _make_olmo_gdn_patcher()  # None on builds without Olmo

    # Atomic up-to-4-file transaction. olmo_gdn is None on builds that
    # don't ship the Olmo GDN layer — MultiFilePatchTransaction tolerates
    # None entries by skipping them gracefully (structural absence, not
    # failure).
    patchers = [chunk_patcher, chunk_delta_h_patcher, qwen_gdn_patcher]
    if olmo_gdn_patcher is not None:
        patchers.append(olmo_gdn_patcher)
    txn = MultiFilePatchTransaction(patchers, name="PN79")
    return txn.apply_or_skip()


# ════════════════════════════════════════════════════════════════════════
# Build-time manifest registration (P2.1 Site Map)
# ════════════════════════════════════════════════════════════════════════
#
# `register_for_manifest()` is called by scripts/build_anchor_manifest.py
# at BUILD TIME (not runtime) to enroll PN79's patchers into the Site Map
# anchor offset manifest. It constructs the same TextPatcher objects but
# pointed at PRISTINE FIXTURE paths under tests/legacy/pristine_fixtures/
# so it works without a vllm install (Mac dev / CI gate).
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
            chunk.py, chunk_delta_h.py, qwen_gdn_linear_attn.py,
            olmo_gdn_linear_attn.py (extracted at pin g303916e93).
    """
    from sndr.engines.vllm.wiring.patcher_registry import register_text_patcher

    chunk_subs = [
        TextPatch(name="1B", anchor=ANCHOR_1B_FWD_SIG_OLD,
                  replacement=ANCHOR_1B_FWD_SIG_NEW, required=True),
        TextPatch(name="1C", anchor=ANCHOR_1C_FWD_INTERNAL_OLD,
                  replacement=ANCHOR_1C_FWD_INTERNAL_NEW, required=True),
        TextPatch(name="1D_DECORATOR", anchor=ANCHOR_1D_DECORATOR_OLD,
                  replacement=ANCHOR_1D_DECORATOR_NEW, required=True),
        TextPatch(name="1D_FORWARD_SIG", anchor=ANCHOR_1D_FORWARD_SIG_OLD,
                  replacement=ANCHOR_1D_FORWARD_SIG_NEW, required=True),
        TextPatch(name="1D_FORWARD_CALL", anchor=ANCHOR_1D_FORWARD_CALL_OLD,
                  replacement=ANCHOR_1D_FORWARD_CALL_NEW, required=True),
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
    qwen_subs = [
        TextPatch(name="3B", anchor=ANCHOR_3B_FORWARD_NATIVE_OLD,
                  replacement=ANCHOR_3B_FORWARD_NATIVE_NEW, required=True),
        TextPatch(name="3C", anchor=ANCHOR_3C_PREFILL_INPLACE_OLD,
                  replacement=ANCHOR_3C_PREFILL_INPLACE_NEW, required=True),
    ]
    olmo_subs = [
        TextPatch(name="4A", anchor=ANCHOR_4A_OLMO_PREFILL_OLD,
                  replacement=ANCHOR_4A_OLMO_PREFILL_NEW, required=True),
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
            "PN79 Sub-3 qwen_gdn_linear_attn.py (build mode)",
            pristine_root / "qwen_gdn_linear_attn.py", qwen_subs,
            patch_id="PN79.Sub-3",
        ),
    )
    register_text_patcher(
        "PN79.Sub-4",
        _make_patcher_for_fixture(
            "PN79 Sub-4 olmo_gdn_linear_attn.py (build mode)",
            pristine_root / "olmo_gdn_linear_attn.py", olmo_subs,
            patch_id="PN79.Sub-4",
        ),
    )
