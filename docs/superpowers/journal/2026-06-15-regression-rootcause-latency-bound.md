# Why decode optimizations regress on 2× A5000, the drifts, the non-regressing levers — 2026-06-15

Deep analysis (9-agent workflow + live measurement) answering: why do patches
regress here, where are the inconsistencies/drifts, and are there solutions that
do NOT regress.

## 1. WHY — the stack is LATENCY-BOUND (measured, not assumed)

Decode signature on the live 35B (nvidia-smi during active generation):
**SM util 90-98%, but power only 76% of cap, memory-controller only 47%**, and
concurrency scales **2.7× to 689 TPS @ conc=8**. SMs are *occupied but stalling* —
issue slots full of warps waiting on latency, with spare power + bandwidth headroom.

Three latency sources, none of which is FLOPs or HBM bytes:
- **L1 — tiny-operand load latency.** Decode M ≈ 8 (max_num_seqs 2 × MTP K+1=4)
  through ~320 expert GEMMs/forward (256 experts / 8 active, moe_intermediate=512)
  → every Marlin M-tile is 8 rows vs the 16-row Ampere tensor-core minimum → derated.
- **L2 — 30 sequential GDN/Mamba2 recurrent layers.** State carries across layers
  and MTP tokens (`fused_sigmoid_gating.py:122` T-loop), reloading q/k/v every step;
  grid ≈ 32-48 CTAs on a 64-SM box → 25-50% SMs idle per GDN layer.
- **L3 — host-staged TP all-reduce.** TP=2 over PHB (PCIe host bridge, no NVLink,
  no working P2P-DMA, `NCCL_P2P_DISABLE=1`). Already software-minimized to 1
  all-reduce/layer; residual is intrinsic PHB round-trip latency.

### THE GENERAL RULE (decision filter for every future patch)

- **ALWAYS HURTS:** anything that **reduces in-flight parallelism / occupancy /
  amortization** on a decode kernel (num_warps↓, KV-splits↓, MTP K↓, CG batch↓,
  FULL→PIECEWISE) — fewer warps to hide L1/L2 → stalls become visible. OR **adds
  work/memory to a tiny-M decode/drafter** for a theoretical accuracy gain.
- **CAN NEVER HELP (don't retry):** **pure compute/precision cuts on decode** —
  fp16-vs-tf32x3, exp2-vs-exp, fewer GEMM passes — decode is never compute- or
  bandwidth-bound (power 76%, mem 47%).
- **CAN HELP (the only winning class):** raise **tokens-in-flight or tokens-per-
  forward without lengthening per-forward latency** — more occupancy (warps↑,
  splits↑ on long-ctx), better amortization (accept↑), tile-to-actual-M,
  cheaper-not-richer accept rules, keeping CG FULL, **removing a serial kernel**
  (fixed-latency tail), and ultimately concurrency/batch (the 2.7× headroom).

Every A/B'd regression this session maps cleanly: PN396 num_warps 4→1 (cut
occupancy, code -4.2%), A1 KV_SPLITS 48→24 (cut grid parallelism — flat where SMs
full, -3..4% on long_ctx where it removed occupancy), MTP K=2 (cut amortization,
-29%), probabilistic draft (added work to tiny-M drafter, -5.9%), P67 fp16 / A3
exp2 (cut compute that wasn't the bottleneck, flat/zero-win).

## 2. NESTYKOVKI / DRIFTS — found, reconciled against the live container, fixed

**Stale-artifact findings — RESOLVED (the session's re-renders already fixed them):**
the workflow flagged D3 (PN368 not live), D4/D5/D7 (untracked env overrides:
PN204=1, TQ_MAX=320000, etc.) from a Jun-13 env dump. Verified against the CURRENT
container: PN368 IS live (2 sub-patches, 5 markers); PN204=0 (balanced, matches
YAML); TQ_MAX=280000 (matches --max-model-len); env == YAML. Not real drifts now.

**Real drifts — FIXED this session:**
- **D1 — P71 ⊥ PN390 latent NameError.** PN390 drops the dense `target_probs`
  buffer (computes only `target_lse`), but the P71 block-verify branch
  (`rejection_sampler.py:518/525`) still references `target_probs` → NameError,
  dormant ONLY because PROD greedy draft gates the branch off (`draft_probs is
  None`). Registry wrongly listed P71 in `PN390.composes_with`. **Fix:** moved P71
  to `PN390.conflicts_with` (it's the PN369 carrier; PN369 was already a conflict).
- **D2 — PN90 enabled but inert landmine.** `GENESIS_ENABLE_PN90_PROBABILISTIC_DRAFT=1`
  with a stale "+2.8% accept, no TPS penalty" comment (a dev93/dev209 result),
  CONTRADICTING the same file's ROLLBACK note (probabilistic = -5.9% TPS/-10%
  accept on dev371+). Inert on dev491 (self-skips) but a landmine: re-activation
  would regress AND unmask D1's NameError. **Fix:** set =0 + corrected the comment.
- **D6 — registry factually wrong: "35B is Qwen3MoE — no GDN layers".** The 35B is
  hybrid_gdn_moe: 30 GDN + 10 full-attn. PN50 (GDN proj fusion) DOES engage on the
  30 GDN layers (head_dim=256 pow2) and was never A/B'd on the 35B. **Fix:**
  corrected the PN50 credit; flagged PN50 for a deliberate 35B A/B.

## 3. NON-REGRESSING LEVERS (by construction — raise occupancy/amortization or remove a serial kernel)

Scoping caveat: the variants below 226 (code, tool_call, multi_turn) are **temp=0 /
greedy**, where P82/PN369 relaxed-accept is **INERT** (greedy argmax has no
threshold). So acceptance tuning cannot move them — only the levers below can.

**Verified healthy (do NOT chase):** MTP draft accept 0.771 (clean per-request,
healthy); P82 0.3 swept-optimal; PN368 live; MoE already 1 grouped GEMM/projection;
block_size_m=8 locked by tiny-M (over-tiling regresses).

**Top candidate — A2: remove the serial topk=8 `moe_sum` reduction.** *(verified)*
`marlin_moe.py:997 ops.moe_sum` → generic strided `at::sum_out` fired 40×/forward
on the decode critical path (the compiled Triton reducer only templates topk∈{2,3,4},
ours is 8). Override `MarlinExperts.moe_sum` → the in-tree `moe_fused_mul_sum_kernel`,
bound to the wrapped-apply stream (the parked PN352 wired the wrong call site +
raced streams — this fixes both). Removing a serial reduction changes neither
parallelism nor occupancy nor batch → cannot reproduce the regression class.
Est −1..3% decode TPOT, helps ALL variants incl. temp=0. Needs a new patch +
numeric-equality gate vs `ops.moe_sum` + CUDA-graph-safety check under PN125 FULL.

**Other non-regressing directions (ranked):**
- A3: `max_num_seqs 2→4` — byte-identical single-stream (still the size-4 FULL graph),
  fills idle SMs at conc≥2 (the 2.7× headroom). NO single-stream win. seqs=3 is
  UNSAFE (max batch 12 has no FULL graph → silent PIECEWISE downgrade).
- B4: `num_warps=8` on `fused_sigmoid_gating` (inverse of PN396 — ADDS latency-hiding
  warps). Uncertain (more shuffle vs more hiding; FLA chose 4). ptxas-gate for spill.
- B2: NCCL small-message tune (`NCCL_ALGO=Tree, NCCL_PROTO=Simple, NCCL_MAX_NCHANNELS=2`)
  — latency cut on the 32 KB all-reduces, works WITHOUT P2P. Off-prod A/B.
- B3: `NUM_KV_SPLITS 48→64` for long-ctx profiles only (inverse of A1).

**Research-track (not a config flip):** P2P-DMA re-enable (live copy 6.6 GB/s =
host-bounce; `can_actually_p2p` never probed — could be a per-layer fixed-latency
cut, but a broken probe HANGS the all-reduce; off-prod only). EAGLE-3 (checkpoint-
blocked). The single shared-trunk MTP layer (`num_nextn_predict_layers=1`) is the
~0.8 accept ceiling — only a structurally-better drafter raises it.

## Bottom line

The 35B is genuinely well-optimized: the regressions are not bugs, they are the
**latency-bound wall** — the SMs are occupied and the only headroom is "more work
in flight," which single-stream cannot supply (hence multi-conc 689@8). The one
remaining single-stream non-regressing build is **A2 (moe_sum serial-reduction
removal)**; everything else either needs concurrency (A3), is uncertain (B4), or is
off-prod research (P2P). Drifts D1/D2/D6 fixed; the system is consistent.
