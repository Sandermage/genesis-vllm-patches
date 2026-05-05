# OOM Hardening Recipes — Single-Card + Multi-Turn Long Context

Curated 2026-05-04 from cross-rig diagnostics on noonghunna/club-3090
(single 3090 24GB rig) + our 2× A5000 PROD. Captures empirical findings
that the audit identified as "OOM on single-card" priority items.

## ⭐ noonghunna/club-3090#22 recipe (Cliff 2b on 24 GB single-card, 60K+ context)

**Symptom**: vLLM 0.20.2 + Genesis v7.72+ on 1× RTX 3090 24 GB, Qwen3.6-27B-int4-AutoRound +
hybrid GDN + chunked-prefill, 60K-token single-shot prompt → `OutOfMemoryError: tried to
allocate 50 MiB, 56 MiB free` at `chunk_o.py:161 o = torch.empty_like(v)`.

**Root cause**: at `gpu_memory_utilization=0.93`, KV pool eats 22.4 GiB, leaving ~1.6 GiB
headroom for activations. FLA `chunk_gated_delta_rule_fwd_h` allocates `(B, NT, H, V, K)`
h-tensor = **1.37 GiB at T=60K** which doesn't fit. PyTorch caching allocator fragments under
repeat 1.37 GiB alloc-free cycles → "50 MiB requested, 56 MiB free" symptom.

**The recipe** (env-only, no code changes — "Level 1" mitigation):

```bash
# 1. Enable Cliff 2b chunking patch (P103 — split T-dim 60K → 4×16K)
export GENESIS_ENABLE_P103=1
export GENESIS_FLA_FWD_H_MAX_T=16384

# 2. Enable PN59 streaming-GDN window-iterative driver
export GENESIS_ENABLE_PN59_STREAMING_GDN=1

# 3. PyTorch allocator hardening (gc 0.6 → 0.85 stops needless GC churn)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.85"

# 4. vllm serve flags — lower gpu_memory_utilization 0.93 → 0.85
vllm serve ... \
    --gpu-memory-utilization 0.85 \         # frees +1.9 GiB activation headroom
    --max-num-batched-tokens 2048 \         # halves Marlin workspace peak
    --max-num-seqs 1 \                      # halves worst-case KV reservation
    ...
```

**Effect**: h-tensor peak 1.37 GiB → **365 MiB** (P103) and headroom 1.6 GiB → **3.5 GiB**.
Real KV usage at 60K single-stream is ~3 GiB out of ~20.5 GiB pool — pool is over-provisioned
for the workload, the 0.05 we give back is "paper" capacity not real.

**Reference launch script**: `scripts/launch/bare_metal_27b_int4_TQ_k8v4_single_card.sh`
already ships with this recipe baked in (v7.72.4+).

**Long-term**: Level 2 (PN59 structural fix — thread chunk_indices/chunk_offsets per window,
wire `GdnScratchPool` into production driver, fix state-chain bug) drops the env-flag
requirement entirely. Tracking in genesis-vllm-patches v7.72.5 sprint.

---

## Quick reference matrix

| Card / VRAM | Workload | Recipe |
|---|---|---|
| 1× 3090 24 GB | Code completion (short ctx ≤ 8K) | TQ k8v4 + util 0.92 + max_num_seqs=4 |
| 1× 3090 24 GB | Long ctx 60-180K + multi-turn | **fp8_e5m2** KV (TQ OOMs); util 0.85; PN35 ON |
| 2× 3080 20 GB (TP=2) | Long ctx > 90K | fp8_e5m2 (TQ k8v4 OOMs at 90K — see club-3090#47) |
| 1× 3090 WSL2 | Any | util 0.85 (vGPU/Xwayland eats ~3.6 GB/card) — see #32 |
| 2× A5000 24 GB (TP=2) | All PROD configs | TQ k8v4 + util 0.90 + MTP K=3 stable |

## Cliff 2b multi-turn OOM (CRITICAL — single-card)

**Symptom**: continuous 5×5 = 25-turn ramp accumulating ~22-25K tokens
OOMs in `chunk_fwd_o → empty_like(v)` after 4-5 hermes turns. ALL six
single-card vLLM composes FAIL continuous soak. Only TP=2 + llama.cpp
survive cleanly.

**Root cause** (per noonghunna codex residency analysis):
- PN12 stays exactly flat at 137 MiB across turns ✓ (Genesis pools clean)
- Growth is in PyTorch caching allocator + vLLM internal state (NOT Genesis)
- Per turn: `total_reserved +1400 MiB`, `total_alloc +590 MiB`,
  `fragmentation +810 MiB`, `free −1402 MiB`
- After 4-5 turns reserved+frag exceed free → OOM in next chunk_fwd_o alloc

**Mitigations** (in priority order):

1. **Lower `--gpu-memory-utilization`** to 0.85 (give allocator headroom)
2. **Drop `--max-model-len`** below the cliff (e.g. 96K instead of 180K)
3. **Use fp8_e5m2 KV** instead of `turboquant_k8v4` (K activation peak < V)
4. **Disable MTP** for high-cliff sessions (MTP K=3 adds ~600 MiB/draft step)
5. **Force `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.6`**
   (already in our 35B PROD/DFlash scripts; add to 27B too)
6. **Restart engine periodically** (every ~50 multi-turn requests in stress)

**Genuine fix** (large work): streaming-GDN refactor per noonghunna Issue #19.
Estimated 2-4 weeks; only true Cliff 2b mitigation that survives continuous
soak.

## TQ k8v4 vs fp8_e5m2 trade-off (single-card focus)

| Property | TQ k8v4 | fp8_e5m2 |
|---|---|---|
| KV memory per token | ~3 bytes (packed) | 1 byte |
| K activation peak | **HIGHER** (per club-3090#47) | lower |
| V activation peak | lower | higher |
| Quality preservation | very high (8-bit K, 4-bit V) | high (lossy fp8) |
| Genesis kernels | P67/P67b/P98/P101 etc | none Genesis-specific |
| Recommended for | TP=2 (24+ GB total), high-quality | single-card tight VRAM |

**Empirical** (club-3090#47, efschu's 2× 3080 20 GB): `turboquant_k8v4` OOMs
at 90K context, but `fp8_e5m2` passes verify-stress 7/7 including 91070-
token recall. Strong evidence for fp8_e5m2 as **safer single-card default**
when VRAM is < 24 GB total.

## WSL2 specific (RossNE99 closure of club-3090#32)

**Headroom**: `--gpu-memory-utilization 0.85` (vs 0.92 native) leaves
~3.6 GB/card slack for vGPU, Xwayland, and any Windows-side display
interleave. Otherwise Worker_TP1 OOMs around model load.

## PN35 (vllm#35975 backport) status

PN35 frees ~64 MiB GPU + ~64 MiB pinned for text-only models — **necessary
but not sufficient** at 0.95 mem-util to close 60K Cliff 2 alone. Pairs with
mem-util drop to 0.93 + the other Cliff 2b mitigations above.

PN35 default-on since v7.68; verify ON via `apply_all` boot log.

## P103 cu_seqlens=[0,T] fix (Issue #18 → fixed v7.71)

Before v7.71: P103 chunked path NEVER engaged on real serving (442/442
invocations bypassed because `cu_seqlens.shape == (2,)` is single-seq
`[0, T]`, not multi-seq). Now correctly recognized as B=1 dense and falls
through to chunked path. **Note**: vLLM's outer chunked-prefill caps T at
`max_num_batched_tokens=4128` (well below P103's `_MAX_T=16384`), so the
chunked path still won't fire on default scripts unless you raise
`max-num-batched-tokens` to ≥ 16384.

## Sources

- club-3090 Issue #19 (cross-rig findings)
- club-3090 Issue #32 (WSL2 OOM, RossNE99)
- club-3090 Issue #41 (GuiPerPT 180K + 0.93 OOM)
- club-3090 Issue #43 (stiggy2k16 OpenClaw failure on long-* configs)
- club-3090 Issue #47 (efschu 2× 3080 20 GB Cliff 2 90K)
- Sandermage/genesis-vllm-patches Issue #18 (P103 gate fix)
- Sandermage/genesis-vllm-patches Issue #19 (streaming-GDN refactor RFE)
