# Deep optimization R&D (dev491, A5000) — caching/MTP/TQ/decode/memory/CUDA

**Date:** 2026-06-16
**Trigger:** user — "study the codebase + model functions, caching, MTP, TQ, decode, memory, CUDA;
develop code to improve speed/latency/quality/memory; finish the memory-optimization implementation."

A 6-angle workflow profiled the **live dev491** stack (not premised — the session's hard lesson) for
genuine, implementable optimizations. Each finding was verified against the installed source + measured.

---

## TL;DR

- **The single-stream stack is genuinely well-optimized — no shippable speed win was found.** The one
  kernel-level candidate (`num_warps=4`) is 24-43% faster *in a microbench* but **within noise
  end-to-end** (207.4 vs 211.8 TPS, both CV ~9%) — attention isn't the A3B-MoE bottleneck.
- **The only real remaining levers are memory/capacity** (G4_83 ~8.4% KV recovery on the 31B; gmu
  0.90→0.93 for 35B concurrency) — they raise context/batch/concurrency, **not** single-stream TPS,
  and both are medium-risk.
- Several "obvious" premises were **refuted by live state** (continuing the session's pattern): the
  "~69% KV waste" was the wrong (non-shipped) config; "nw=8 optimal" was a premise, never measured.

---

## What is ALREADY optimal (verified, do not touch)

- **MTP K=3 + greedy verify** — measured 72.9% acceptance (665/912; per-pos 85.9/73.4/59.5%), mean
  accepted 2.19/3. K=2 broken, K=4 regressed, adaptive-K rejected by 3 benches + version-gated off.
  Greedy correct (probabilistic PN90 = -5.9% TPS). **No per-step GPU→CPU sync** (PN341 applied). K+1
  verify routes through P67 (NUM_KV_SPLITS=48, USE_UPSTREAM=1) under FULL cudagraph. PN390 streaming-LSE
  rejection sampler shipped. **Nothing to pull.**
- **Decode kernel** — BLOCK_H=16 correct (BLOCK_H=8 is 13-16% slower: Ampere tl.dot needs M≥16);
  NUM_KV_SPLITS=48 defensible (splits=16 faster <16K but 48 faster ≥32K, and 35B is 280K-provisioned);
  stage2 reduction launch-bound, no redundant work. BLOCK_KV=16 optimal (32 spills, A/B'd -5.2%).
- **GDN/Mamba prefill** — at the Ampere architectural floor (FlashInfer GDN is SM90-only, CuteDSL
  SM100-only; the FLA Triton path is the only option on SM 8.6; the O(N) scan is the SSM math itself).
  Warm prefill ~192ms (live `request_prefill_time`). Not chunkable/parallelizable further.
- **Buffer pools + one-time caches** — P37/P39a/P46/PN12 pools correctly shared (peak-reducers, no
  double-alloc); G4_61 collapses N per-layer TQ decode scratch into one. Centroid solver (291ms) +
  Hadamard (1ms) computed once/boot via `@lru_cache`/`@functools.cache` (63 hits / 1 miss over 64
  layers), codebook shared across layers. No steady-state recompute.

## The one measured speed candidate — REJECTED end-to-end

`VLLM_TQ_DECODE_NUM_WARPS=4` (P18B forces 8; upstream default is 4; the verify kernel already uses 4).
Microbench: nw=4 is 24-43% faster on the grouped TQ-decode stage1 at the real shape
(B=8/Hq=16/Hk=2/D=128/slot=196/splits=48), **bit-identical** (max_abs_diff=0.0). **End-to-end A/B
(docker cp + restart + canonical bench): 207.4 vs 211.8 TPS — within noise.** The attention kernel is a
small slice of A3B-MoE decode time, so the kernel win doesn't translate. **Not shipped.** (It *might*
help long-context decode where attention dominates more — flagged for a future 8K/32K A/B; not chased
now without a measured win.) Lesson re-applied: microbench ≠ end-to-end; only the canonical bench ships.

## The real remaining levers (memory/capacity, medium-risk)

1. **G4_83 — 31B TQ multi-bucket page allocator.** In the shipped all-TQ 31B config, the 10 global
   layers (head_dim=512, 4 KV heads, TQ page 33152 B) are padded UP to the sliding-layer TQ page
   (head_dim=256, 16 KV heads, 67072 B) — a TQ-to-TQ pad (NOT the bf16 pad I earlier claimed), = 50.6%
   waste on the global tier = **8.43% of the whole KV pool = ~+9.2% more KV blocks** at fixed gmu. Fix:
   route the two TQ page sizes through separate buckets (the existing DeepseekV4-style multi-bucket
   allocator) instead of unifying to the max. Static-derived (the 31B isn't running); confirm via the
   boot-log `num_gpu_blocks` diff. Risk: medium (coexisting physical page sizes). Value: longer
   context / bigger batch on the 31B (a secondary, no-MTP model).
2. **gmu 0.90→0.93 (35B).** +0.03 gmu ≈ +737 MiB/GPU → more KV blocks → higher concurrent throughput
   (no single-stream gain). Risk: medium — GPU0 has only ~2040 MiB free (binding constraint); 0.93 may
   OOM GPU0. Would need a careful boot + OOM check.

## Honest verdict

The Genesis stack is mature and well-tuned (the cumulative result of many prior optimization sessions).
There is **no easy high-value single-stream speed or latency win left** — the kernels, MTP, prefill,
buffers, and caches are at or near optimal, and the one kernel candidate doesn't translate end-to-end.
The available improvements are **memory-capacity** (G4_83, gmu), which are modest (8.4% / concurrency),
medium-risk, and do not move single-stream TPS. G4_83 is the cleanest of these and the one worth
building if more KV headroom on the 31B is wanted.
