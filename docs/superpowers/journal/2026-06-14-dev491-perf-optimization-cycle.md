# dev491 Decode-TPS Optimization Cycle — 2026-06-14

Goal: push every chat-matrix variant > 226 TPS (ideal ≥ 246), focus on memory +
CUDA kernels. Baseline (cleaned dev491, n=5): thinking_off 254.2 / thinking_on
254.3 (✓) · code_gen 218.1 · multi_turn 215.6 · long_gen 187.9 (worst TPOT 5.19ms)
· short_chat 205.5 / tool_call 170.7 (TTFT-bound) · long_ctx 8k/32k 93.8/38.3.

## Method

13-agent Workflow (1.5M tokens): 5 decode-path diagnosis agents (GDN/Mamba
recurrent, TQ attention, MoE Marlin, spec-decode/MTP, memory/allocator) reading
the live dev491 kernel snapshot + our patches, 5 external-research agents
(vLLM/SGLang/FlashInfer/FLA/lmcache/TRT-LLM/papers/forks), 2 patch-impl reviewers,
1 synthesis. Produced a ranked candidate roadmap. Then **A/B each on PROD, one
change at a time, keep only measured wins.**

## Key diagnostic finding (correct + valuable)

Under MTP K=3 (steady-state PROD), the GDN decode path routes EVERY layer through
`fused_sigmoid_gating_delta_rule_update_kernel` (30 of 41 layers — the dominant
decode cost), NOT the optimized packed single-token kernel. Confirmed our model
runs the **V1 model runner** (FP8 + Qwen3Next not in the V2 frozenset, verified
`config/vllm.py:566-572`), so P82/PN390/PN369 acceptance patches are LIVE on
`v1/sample/rejection_sampler.py` (20 Genesis markers present) — not dead.

## A/B RESULTS — the low-risk levers are exhausted or tested-negative

| Lever | Result | Evidence |
|---|---|---|
| **A1** `GENESIS_P67_NUM_KV_SPLITS` 48→24 | **FLAT / revert** | code 217.6 vs 218.1, long_gen 190.9 vs 187.9 (within CV), thinking −1%, long_ctx 8k/32k **−4%/−3%**. Split-KV is not the 35B decode bottleneck (it's the 11 full-attn layers; the 30 GDN layers dominate). The roadmap's #1 quick-win (3 candidates converged) did not materialize. Kept 48 (long-ctx-safe). |
| **A2** P82 accept threshold sweep | **already optimal** | YAML line 160: "0.3 empirically optimum on 35B with MTP K=3" (prior sweep). P82-vs-off is already +12% (captured). Lower threshold = more accepts but biased sampling (quality risk) — and 0.3 is already the swept optimum. Marginal headroom. |
| **A3** exp2 migration on the GDN decode kernel | **tested-negative (rejected without re-testing)** | PN354's own docstring (pn354_gdn_use_exp2.py:30-34): "Decode paths stay natural-base (`fused_recurrent` / `fused_sigmoid_gating`) — **confirmed zero-win there**, and upstream keeps them on exp too." PN354 deliberately ships exp2 for PREFILL only. The workflow re-proposed PN392 (exp2 on this exact kernel) without finding PN354's note — caught by Study→Verify. |
| **PN396** (built+deployed) GDN decode kernel `num_warps` 4→1 | **REGRESSION → reverted** | Built a real bit-exact text-patch (registry + dispatch + anchor), deployed live (marker present, `num_warps=1` confirmed in the kernel), A/B'd chat-matrix n=5: num_warps=1 **lost** — code 209.0 vs 218.1 (−4.2%), thinking 244.6/247.6 vs 254.3/254.2 (−3.8/−2.6%), short_chat −5.5%, tool_call −10.8% (multi_turn flat). The static hypothesis (1 warp → intra-thread BK reduction, no shuffle, matches `fused_recurrent` siblings) was **empirically wrong**: 4 warps win because the extra threads hide per-token q/k/v global-load latency, and this gating+varlen kernel differs from the pure-recurrent siblings. Kept as a documented dead-end (`PN396`, default OFF, "TESTED-NEGATIVE — DO NOT ENABLE" in registry credit + docstring) so the 4-vs-1 question is not re-opened. **The "measure don't assume" rule paid off — static kernel reasoning was refuted by the A/B.** |

## Honest conclusion

The 35B on dev491 is at its **practical single-stream optimum for low-risk tuning**.
The env-level and simple-kernel levers are exhausted (A1 flat) or tested-negative
(A3 = PN354 zero-win) or already-tuned (A2). thinking variants clear 246; the
reasoning-heavy / mid variants (code 218, multi_turn 216, long_gen 188) sit
4–7% below 226, and the chat-matrix CV (~5–7%) exceeds the per-lever headroom of
the remaining micro-optimizations.

## Remaining roadmap (higher-effort, uncertain — a dedicated kernel sprint, NOT one-shot PROD A/Bs)

Validate these with `tools/bench_decode_tpot_clean_ab.py` (decode-only TPOT,
process-isolated arms, Welch's t-test — lower CV than chat-matrix wall_TPS), and
gate behind bit-exact unit tests BEFORE any PROD A/B:

- **B1 — TQ grouped decode `BLOCK_H` 16 → 8** (`triton_turboquant_decode.py:~756`).
  GQA group_size = Hq/Hk = 32/4 = 8, but BLOCK_H=16 → the `tl.dot` computes a full
  [16,*] tile while `mask_h` discards 8 lanes = **~50% wasted tensor-core MACs** on
  the 11 full-attn layers. Bit-exact (masked lanes already discarded). Re-tune
  num_warps 8→4 after. Risk: medium (kernel). Est −0.2..0.4ms TPOT — but A1 showed
  the TQ layers (11/41) move the needle little, so likely marginal.
- **B4 — GDN gating de-duplication** (`fused_sigmoid_gating.py:122-153`, BV cap :208).
  The softplus/b_g/b_beta block depends only on `i_hv` but is recomputed NV=4× per
  token (BV=32, V=128). Raise BV (NV 4→1) OR precompute g_exp/beta to scratch.
  This is the dominant-kernel lever and DIFFERENT from A3's tested-negative exp2
  (it cuts recompute COUNT, not exp base). Est −4..9% on 30 layers = the largest
  standalone number. Risk: medium-high (BV change → register/shared-mem pressure on
  SM8.6 100KB; num_warps co-tune; numerical guard). The most promising untested lever.
- **C2 — global adaptive-K** on the live V1 MTP speculator (drop K=3→2 when batch
  accept low). Re-target `g_dynamic_k_mtp_proposer.py` onto the V1 proposer. High
  effort; recapture CG per K. Research-track.

Rejected (verified non-applicable): PR #44251 SSU configs (we use fla path),
#45295 Marlin pad (correctness), #42311 GDN flatten (eager-only), SGLang
MoE-Align&Sort + uniform-dequant (A100-specific / already have). EAGLE-3 hidden-
state fusion needs a separate draft head (out of MTP scope).

## Aggregate-throughput context

Single-stream >246 on ALL variants is hard on SM8.6 without the B4/C2 kernel work.
The hardware already delivers **689 TPS aggregate @ conc=8** (multi-conc preset) —
the throughput story for multi-tenant load is the concurrency path, not single-stream.

## PROD state

Reverted to the validated baseline (KV_SPLITS=48, P82=0.3, no new kernel patches).
No regression introduced this cycle; the roadmap above is the documented path for a
focused kernel-optimization sprint.
