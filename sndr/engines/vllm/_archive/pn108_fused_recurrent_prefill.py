# SPDX-License-Identifier: Apache-2.0
"""PN108 — DEPRECATED experiment, kept for documentation.

OUTCOME (2026-05-14, 1× A5000 24 GB, Qwen3.6-27B-INT4-AutoRound):
Attempted to dispatch long-prefill GDN forward to
`fla.ops.fused_recurrent.fused_recurrent_gated_delta_rule` to eliminate
the `h(B, NT, H, V, K)` chunk-state buffer that triggers Cliff 2. The
attempt failed for two distinct reasons that are inherent to fla's
fused_recurrent API design, not a fixable bug:

  1. `inplace_final_state=True` mode requires `ssm_state_indices` to
     be passed; the Triton kernel unconditionally dereferences it in
     the INPLACE branch (`tl.load(ssm_state_indices + i_n * ...)`),
     causing `AttributeError("'NoneType' object has no attribute
     'type'")` at compile time when None is passed.
  2. `inplace_final_state=False` mode allocates `final_state =
     q.new_empty(T, HV, V, K, dtype=initial_state.dtype)` — a
     per-token state buffer. At T=29K, Qwen3.6-27B shape, fp32
     accumulator, that is ~12 GB per call. Cannot fit on 24 GB card.

fla's `fused_recurrent_gated_delta_rule` is designed for the vLLM-style
**continuous-batching decode loop** (multi-request packed state pool
indexed by `ssm_state_indices`, optionally with `num_accepted_tokens`
for speculative-decode accept/reject). Repurposing it for prefill on a
single sequence outside that framing requires patching the kernel
itself — out of scope and superseded by PN59.

REPLACEMENT: **PN59** (streaming-GDN orchestrator). After 2026-05-14
anchor fix for upstream nightly dcacdf9a (added `core_attn_out` param
to `chunk_gated_delta_rule_fwd`), PN59 wraps the existing chunked
kernel iteratively, threading state through a window-by-window loop
on the SAME `chunk_gated_delta_rule_fwd` orchestrator. This achieves
the same memory profile (no full-T h-tensor materialisation) without
needing a different kernel.

This file is retained as a tombstone for two reasons:
  - the design analysis above documents why a "switch to recurrent
    kernel" approach is structurally infeasible with fla as-shipped;
  - the registry entry stays so dispatcher logs make it clear that
    PN108 is permanently disabled, not "missing".

Default OFF. Enabling `GENESIS_ENABLE_PN108_FUSED_RECURRENT_PREFILL=1`
will engage the broken code path and crash at first long-prefill
request. Do not enable.

---

Original design notes follow (preserved verbatim) for historical
reference; the implementation below is the broken first attempt.

PN108 — long-prefill dispatch to fla.fused_recurrent_gated_delta_rule.

Problem
-------
At long contexts (~50K+ tokens) on a single 24 GB GPU, Qwen3-Next prefill
hits a memory ceiling that is NOT primarily a KV-pool problem. The two
documented "cliffs" (noonghunna/qwen36-27b-single-3090 README):

  * Cliff 1 — TurboQuant marlin attention scratch on tool-prefill ≥25K
    (~138 MiB allocation right after the marlin GEMM output)
  * Cliff 2 — flash-linear-attention's chunkwise-parallel GDN kernel,
    fla.ops.chunk_delta_h.chunk_gated_delta_rule_fwd_h, allocates
    `h = k.new_empty(B, NT, HV, K, V)` with NT = T_chunk // 64. The
    buffer scales linearly with the per-call sequence length. Even
    with vLLM's outer chunked-prefill splitting prompts into
    max-num-batched-tokens slices, the cumulative steady-state memory
    pressure leaves no room for that 48–140 MiB transient h to land.

PN59 (streaming-GDN orchestrator) and PN32 (outer chunked dispatch) both
target this. PN59 is currently disabled by an upstream signature drift
(`core_attn_out` parameter added to chunk_gated_delta_rule_fwd in
nightly dcacdf9a — anchor mismatch → silent skip). PN32 chunks the outer
call correctly but adds a torch.cat peak (the list-of-chunks + the
concatenated full output coexist for one instant) which the saturated
single-card budget cannot absorb.

PN108 takes a different path: **switch backends**.

Solution
--------
fla also exposes `fused_recurrent_gated_delta_rule` (file:
`vllm/model_executor/layers/fla/ops/fused_recurrent.py`). It is a pure
token-by-token recurrent kernel — NO `h` chunk-state buffer at all:

  Memory profile of fused_recurrent_gated_delta_rule:
    o            = q.new_empty(NK=1, B, T, HV, V)   ← only T-scaled tensor
    final_state  = initial_state                    ← in-place, no new alloc

  Memory profile of chunk_gated_delta_rule:
    h            = k.new_empty(B, T//64, HV, K, V)  ← extra "Cliff 2" tensor
    o            = (B, T, HV, V)
    plus intermediate g_cumsum, A (wy-rep), w, u, v_new for each chunk

For long prefill, fused_recurrent saves ~1.6 GB per GDN forward (h-tensor
elimination at T=100K, Qwen3.6-27B GDN shape). It is compute-bound, not
memory-bound: ~3–8× slower than chunkwise-parallel because internal
parallelism is restricted to (NK × NV × N × HV) thread blocks instead of
also parallelising across chunks.

Trade-off summary on Qwen3.6-27B int4 AutoRound, single A5000 24 GB:
  - chunked path: ~1000 t/s prefill, OOMs above ~50K on this rig
  - fused_recurrent path: ~150–250 t/s prefill, stable at 150–200K

Output contract is identical for both kernels — `(o[B,T,HV,V],
final_state[N,HV,V,K])` — so the GDN layer's downstream code
(`ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(...)`)
needs no changes.

Dispatch logic
--------------
PN108 patches `_forward_core`'s prefill branch in
`vllm/model_executor/layers/mamba/gdn_linear_attn.py`. When all of the
following hold, the call is rerouted to `fused_recurrent_gated_delta_rule`:

  1. `GENESIS_ENABLE_PN108_FUSED_RECURRENT_PREFILL=1` (opt-in env)
  2. Single-sequence prefill — `cu_seqlens.shape[0] == 2`. Multi-seq
     prefill bypasses to original; thread-safe state propagation across
     packed sequences requires extra plumbing not in scope here.
  3. `query_non_spec.shape[1] > GENESIS_PN108_FUSED_RECURRENT_THRESHOLD`
     (default 32768). Below this, chunkwise-parallel is faster and
     still fits in memory.

Outside that envelope, the original `self.chunk_gated_delta_rule(...)`
runs unchanged.

Numerical equivalence
---------------------
Both kernels implement the same gated-delta-rule recurrence:
  state[t] = decay[t] * state[t-1] + beta[t] * (k[t]^T v[t])
  o[t]     = state[t] @ q[t]

chunk_gated_delta_rule expresses this via a chunkwise-parallel form
(Songlin Yang et al., "Gated Linear Attention"). fused_recurrent_gated_
delta_rule executes the literal recurrence in registers. The math is
identical modulo (a) order-of-operations in the chunkwise WY accumulator
which can produce small fp16 rounding differences, typically <1e-3
relative on each output element after 10K tokens. For our use case
(reasoning + tool calling, not training), the drift is well within
sampling-temperature noise and produces no observable quality change.

Author: Sandermage / Sander Barzov, Odessa.
Created: 2026-05-14 in direct response to PN32/PN59 OOM regression
analysis on single A5000 24 GB at 75K context.
"""
from __future__ import annotations

import logging

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pn108_fused_recurrent_prefill")

GENESIS_PN108_MARKER = (
    "Genesis PN108 fused_recurrent prefill dispatch (Cliff 2 memory-bound fix)"
)


# Anchor — matches upstream gdn_linear_attn.py _forward_core prefill branch.
# Identical to PN32's anchor; PN108 is mutually exclusive with PN32 (one or
# the other patches this same block). The dep-graph in core/conflicts.py
# will assert this; do NOT enable both via env at the same time.
PN108_ANCHOR = (
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

PN108_REPLACEMENT = (
    "        # 2.2: Process the remaining part\n"
    "        if attn_metadata.num_prefills > 0:\n"
    "            assert non_spec_state_indices_tensor is not None\n"
    "            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()  # type: ignore[index]\n"
    "            assert has_initial_state is not None\n"
    "            initial_state[~has_initial_state, ...] = 0  # type: ignore[operator]\n"
    "\n"
    "            # [Genesis PN108 fused_recurrent prefill dispatch]\n"
    "            # For long single-sequence prefill, dispatch to fla's\n"
    "            # fused_recurrent_gated_delta_rule which has no chunk-state\n"
    "            # h buffer (the Cliff-2 OOM trigger). Trade ~3-8x prefill\n"
    "            # speed for unconditional memory safety at any T.\n"
    "            import os as _g_pn108_os\n"
    "            _g_pn108_on = _g_pn108_os.environ.get(\n"
    "                'GENESIS_ENABLE_PN108_FUSED_RECURRENT_PREFILL', ''\n"
    "            ).strip().lower() in ('1', 'true', 'yes', 'on')\n"
    "            try:\n"
    "                _g_pn108_threshold = int(\n"
    "                    _g_pn108_os.environ.get(\n"
    "                        'GENESIS_PN108_FUSED_RECURRENT_THRESHOLD', '32768'\n"
    "                    )\n"
    "                )\n"
    "            except (ValueError, TypeError):\n"
    "                _g_pn108_threshold = 32768\n"
    "            _g_pn108_T = int(query_non_spec.shape[1])\n"
    "            _g_pn108_is_single_seq = (\n"
    "                non_spec_query_start_loc is not None\n"
    "                and non_spec_query_start_loc.shape[0] == 2\n"
    "            )\n"
    "            _g_pn108_use_fused = (\n"
    "                _g_pn108_on\n"
    "                and _g_pn108_is_single_seq\n"
    "                and _g_pn108_T > _g_pn108_threshold\n"
    "            )\n"
    "            if _g_pn108_use_fused:\n"
    "                try:\n"
    "                    from vllm.model_executor.layers.fla.ops.fused_recurrent import (\n"
    "                        fused_recurrent_gated_delta_rule as _g_pn108_fused,\n"
    "                    )\n"
    "                except ImportError:\n"
    "                    _g_pn108_fused = None\n"
    "                if _g_pn108_fused is not None:\n"
    "                    import logging as _g_pn108_log\n"
    "                    _g_pn108_log.getLogger('genesis.pn108').info(\n"
    "                        '[PN108] dispatching long prefill (T=%d > %d) to '\n"
    "                        'fused_recurrent_gated_delta_rule (memory-safe path)',\n"
    "                        _g_pn108_T, _g_pn108_threshold,\n"
    "                    )\n"
    "                    (\n"
    "                        core_attn_out_non_spec,\n"
    "                        last_recurrent_state,\n"
    "                    ) = _g_pn108_fused(\n"
    "                        q=query_non_spec,\n"
    "                        k=key_non_spec,\n"
    "                        v=value_non_spec,\n"
    "                        g=g_non_spec,\n"
    "                        beta=beta_non_spec,\n"
    "                        initial_state=initial_state,\n"
    "                        # inplace=True: write final state back into\n"
    "                        # the initial_state tensor (no per-token alloc\n"
    "                        # of (T, HV, V, K) which would be ~12 GB at\n"
    "                        # T=29K). initial_state is already a fresh\n"
    "                        # .contiguous() copy from ssm_state — safe to\n"
    "                        # mutate.\n"
    "                        inplace_final_state=True,\n"
    "                        cu_seqlens=non_spec_query_start_loc,\n"
    "                        use_qk_l2norm_in_kernel=False,\n"
    "                    )\n"
    "                else:\n"
    "                    (\n"
    "                        core_attn_out_non_spec,\n"
    "                        last_recurrent_state,\n"
    "                    ) = self.chunk_gated_delta_rule(\n"
    "                        q=query_non_spec,\n"
    "                        k=key_non_spec,\n"
    "                        v=value_non_spec,\n"
    "                        g=g_non_spec,\n"
    "                        beta=beta_non_spec,\n"
    "                        initial_state=initial_state,\n"
    "                        output_final_state=True,\n"
    "                        cu_seqlens=non_spec_query_start_loc,\n"
    "                        chunk_indices=attn_metadata.chunk_indices,\n"
    "                        chunk_offsets=attn_metadata.chunk_offsets,\n"
    "                        use_qk_l2norm_in_kernel=False,\n"
    "                    )\n"
    "            else:\n"
    "                (\n"
    "                    core_attn_out_non_spec,\n"
    "                    last_recurrent_state,\n"
    "                ) = self.chunk_gated_delta_rule(\n"
    "                    q=query_non_spec,\n"
    "                    k=key_non_spec,\n"
    "                    v=value_non_spec,\n"
    "                    g=g_non_spec,\n"
    "                    beta=beta_non_spec,\n"
    "                    initial_state=initial_state,\n"
    "                    output_final_state=True,\n"
    "                    cu_seqlens=non_spec_query_start_loc,\n"
    "                    chunk_indices=attn_metadata.chunk_indices,\n"
    "                    chunk_offsets=attn_metadata.chunk_offsets,\n"
    "                    use_qk_l2norm_in_kernel=False,\n"
    "                )\n"
    "            # Init cache\n"
    "            ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(\n"
    "                ssm_state.dtype\n"
    "            )\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/layers/mamba/gdn_linear_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN108 GDN _forward_core fused_recurrent prefill dispatch "
            "(Cliff 2 memory-bound fix)"
        ),
        target_file=str(target),
        marker=GENESIS_PN108_MARKER,
        sub_patches=[
            TextPatch(
                name="pn108_forward_core_fused_recurrent_dispatch",
                anchor=PN108_ANCHOR,
                replacement=PN108_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN108 fused_recurrent prefill dispatch]",
            # PN32 v2 marker — if PN32 patched first, PN108 must SKIP cleanly
            # to avoid corrupting the same prefill branch.
            "[Genesis PN32 v2 v7.69 chunked-prefill]",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN108 — fused_recurrent prefill dispatch.

    Mutually exclusive with PN32 v2 (both target the same _forward_core
    prefill branch). Boot-time dep-graph in core/conflicts.py will
    refuse to apply both at once.
    """
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN108")
    log_decision("PN108", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "gdn_linear_attn.py not resolvable"

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "PN108 applied: GDN long single-sequence prefill (T > "
            "GENESIS_PN108_FUSED_RECURRENT_THRESHOLD, default 32768) "
            "now dispatches to fla's fused_recurrent_gated_delta_rule "
            "(no chunk-state h buffer, memory-bounded at any T). "
            "Slower (~3-8x at long T) but unconditionally safe vs OOM. "
            "Default OFF — opt-in via GENESIS_ENABLE_PN108_FUSED_"
            "RECURRENT_PREFILL=1. Mutually exclusive with PN32."
        ),
        patch_name="PN108 GDN fused_recurrent prefill dispatch",
    )
