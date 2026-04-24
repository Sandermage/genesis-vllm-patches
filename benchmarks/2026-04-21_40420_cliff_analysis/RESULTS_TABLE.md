# Full result tables — 2026-04-21 cliff analysis

## 1. 6-way config matrix (focused sweep 228k → 260k, step 2k, 3 runs each)

Phase 1 = config-only workaround (rows 1–4). Phase 2 = Patch 22 code
fix (rows 5–6).

| # | `gpu_util` | `max_seqs` | P20 | **P22** | KV tokens | Max stable | First crash @ | Crash type | Folder |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.905 | 2 | no | no | 946,656 | 232k | 234k (PARTIAL 1/3) | NCCL all-gather timeout cascade (10 min later) | `raw/01_*` (see Phase 2/3 in preserved master log) |
| 2 | 0.905 | 2 | **yes** | no | 946,656 | 232k | 234k (PARTIAL 1/3) | **Direct** `CUDA OOM: tried 88 MiB, 42.81 MiB free` | `raw/02_patch20_only_234cliff/` |
| 3 | **0.85** | 2 | **yes** | no | 589,584 (−38%) | 260k+ ✅ | *no cliff in tested zone* | — | `raw/03_patch20_util085_seqs2/` |
| 4 | 0.905 | **1** | **yes** | no | 979,872 (+3%) | — | 228k (PARTIAL 1/3) | `CUDA OOM: tried 84 MiB, 78.81 MiB free` | `raw/04_patch20_util0905_seqs1/` |
| 5 | 0.85 | 2 | **yes** | **yes** | 517,616 (−12% from row 3 baseline) | 260k+ ✅ | *no cliff in tested zone* | — | `raw/05_patch22_util085/` |
| **6** 🏆 | **0.905** | 2 | **yes** | **yes** | **877,456** | **260k+** ✅ | *no cliff in tested zone* | — | `raw/06_patch22_util0905_TRUTH/` |

Memory free at phase-1 crashes (per rank): row 2 = 42.81 MiB,
row 4 = 78.81 MiB. PyTorch-allocated at crash: 22.64 GiB in both.
22.64 + free + model buffers ≈ 23.56 GiB (full A5000 capacity).

Row 6 is the breakthrough: **Patch 22 recovers +287k KV tokens**
(877k vs 589k phase-1 workaround) while eliminating the cliff. The
~1 GiB shared dequant buffer now accounted for by vLLM's memory
profiler means KV cache is sized correctly and never triggers the
lazy-allocation OOM at long context.

## 2. Experiment #3 — winning config, speed × context

All 17 points 3/3 PASS. Speed decays smoothly through advertised
`max_model_len=262144`.

| context | TTFT avg | tok/s avg |
|---|---|---|
| 228k | 16.36s | 38.5 |
| 230k | 1.49s | 38.1 |
| 232k | 1.44s | 37.7 |
| 234k | 2.10s | 37.2 |
| 236k | 1.25s | 37.3 |
| 238k | 1.12s | 36.9 |
| 240k | 1.33s | 36.7 |
| 242k | 1.27s | 36.5 |
| 244k | 1.48s | 36.1 |
| 246k | 1.36s | 35.9 |
| 248k | 1.55s | 35.9 |
| 250k | 1.15s | 36.6 |
| 252k | 2.32s | 35.7 |
| 254k | 1.32s | 35.5 |
| 256k | 1.21s | 35.2 |
| 258k | 1.38s | 34.8 |
| 260k | 1.37s | 34.9 |

## 3. Baseline `max_model_len` sweep — short-context speed is flat

9 restarts at different `max_model_len`, `gpu_util=0.905 max_seqs=2`,
no Patch 20. Per-point JSON in
[`raw/01_baseline_sweep_128-256k/`](raw/01_baseline_sweep_128-256k/).

| max_model_len | sanity tok/s | @64 | @256 | @1024 | @2048 | stability 10/10 | stress 6/6 |
|---|---|---|---|---|---|---|---|
| 128k | 141.6 | 144.7 | 143.2 | 141.2 | 139.4 | ✅ | ✅ |
| 148k | 139.9 | 143.0 | 141.2 | 139.3 | 137.8 | ✅ | ✅ |
| 160k | 140.1 | 143.4 | 141.7 | 139.6 | 137.9 | ✅ | ✅ |
| 172k | 142.3 | 143.9 | 142.1 | 140.0 | 138.5 | ✅ | ✅ |
| 188k | 142.2 | 143.9 | 142.2 | 140.3 | 138.8 | ✅ | ✅ |
| 204k | 141.9 | 144.6 | 142.7 | 140.8 | 139.3 | ✅ | ✅ |
| 226k | 141.1 | 143.3 | 141.7 | 139.7 | 138.0 | ✅ | ✅ |
| 245k | 143.4 | 144.6 | 142.8 | 141.0 | 139.2 | ✅ | ✅ |
| 256k | 140.9 | 144.6 | 142.7 | 140.7 | 138.8 | ✅ | ✅ |

**Conclusion:** short-context speed spread ≤ 1% across 2× range of
`max_model_len`. The config doesn't tax short-context generation; the
only tax is the long-context transient ceiling.

## 4. Memory plateau during bench (nvidia-smi post-bench, per GPU)

Same 9-point sweep, snapshot taken *after* each per-point bench
completed. CSVs in
[`raw/01_baseline_sweep_128-256k/*_nvidia-smi.csv`](raw/01_baseline_sweep_128-256k/).

| max_model_len | Memory used | % of 24,564 MiB |
|---|---|---|
| 128k | 22,988 MiB | 93.6% |
| 148k | 23,248 MiB | 94.6% |
| 160k | 23,508 MiB | 95.7% |
| 172k | 23,668 MiB | 96.4% |
| 188k | 23,668 MiB | 96.4% |
| 204k | 23,668 MiB | 96.4% |
| 226k | 23,668 MiB | 96.4% |
| 245k | 23,668 MiB | 96.4% |
| 256k | 23,668 MiB | 96.4% |

Memory plateaus at 172k. Above that, more `max_model_len` doesn't
reserve more KV at `util=0.905` — vLLM's budget is already capped.
Remaining headroom ≈ 900 MiB for all transients combined. Long-context
prefill + rotation + cat intermediates can exceed that above ~230k
without Patch 20, and above ~250k even with Patch 20 at util=0.905.

## 5. Baseline context decay (from the 256k point, `util=0.905`, no P20)

From the 9-point sweep's context-test JSON at `max_model_len=256k`.
Full per-point results across *all* 9 max_model_len values are in
[`aggregated/baseline_aggregate.md`](aggregated/baseline_aggregate.md).

| context | tok/s | TTFT |
|---|---|---|
| 4k | 138.8 | 0.39s |
| 8k | 133.1 | 0.40s |
| 16k | 123.1 | 0.78s |
| 32k | 106.7 | 1.60s |
| 64k | 82.2 | 3.23s |
| 96k | 67.8 | 3.64s |
| 128k | 55.9 | 4.10s |
| 148k | 51.6 | 2.91s |
| 160k | 49.4 | 2.07s |

Smooth 1/log-N decay — classic flash-attn + TurboQuant pattern.
