# Genesis vLLM Patches — Benchmarks

_Latest canonical PROD bench 2026-05-11 — Genesis v11.0.0+wave8 (134 patches), vLLM `0.20.2rc1.dev93+g51f22dcfd`, **2× RTX A5000 24 GB** (Ampere SM_86), driver 580.142, CUDA 13.0.2. Both models served under MTP K=3 spec-decode, TurboQuant k8v4 KV cache, FlashAttention 2, TP=2._

## Latest PROD numbers (Wave 8, 2026-05-11)

| Model | wall_TPS | decode_TPOT | CV% | Tool-call | Method |
|---|---:|---:|---:|:---:|---|
| **Qwen3.6-27B-int4-AutoRound** | **132.28** | **7.31 ms** | 5.29% | 8/8 | `genesis_bench_suite.py --quick --ctx 8k` (5×5×1024) |
| **Qwen3.6-35B-A3B-FP8** (Sprint 1 2026-05-09) | 241.35 | 3.85 ms | 3.02% | 7/7 | same harness |

### Wave 8 Δ vs prior Wave 7 baseline (27B PROD)

| Metric | Wave 7 (2026-05-09) | Wave 8 (2026-05-11) | Δ |
|---|---:|---:|---:|
| wall_TPS | 124.29 | **132.28** | **+6.43%** |
| decode_TPOT_ms | 7.78 | **7.31** | **-6.0%** (faster) |
| TTFT_ms | 108.25 | 100.9 | -6.8% |

**Wave 8 components**: PN90 + PN16 V8 (drift recovery) + P82=1+thr=0.1 (Sprint 1 sweep) + GroupAB additions (P70 / PN12 / PN14 / P94 / P103) + P67_NUM_KV_SPLITS 32→16 + removed retired/broken patches (P61 / P71-broken-on-GQA=6 / P100-Blackwell / PN13 / P83+P85 broken dep).

## Historical reference (v7.72 dev9 snapshot, 2026-05-05)

Preserved for regression-detection. Different stack from current Wave 8 / dev93 above.

| Model | Sustained TPS | CV% | Cold-warm latency | Tool-call clean | Multi-turn 10/10 | VRAM steady-state |
|---|---|---|---|---|---|---|
| **Qwen3.6-35B-A3B-FP8** (MoE) | **192.9 tok/s** | 4.19% | 2.34s | **10/10** | **10/10** survived (avg 1.1s) | 22687 + 21998 = 44685 MiB |
| **Qwen3.6-27B-int4-AutoRound** (Lorbus dense) | **95.6 tok/s** | 4.04% | 4.76s | **10/10** | **10/10** survived (avg 2.3s) | 22753 + 22064 = 44817 MiB |

## How these numbers were captured

```bash
GENESIS_MODEL=qwen3.6-35b-a3b \
  python3 tests/bench/comprehensive_bench.py --turns 10 --skip-needle \
  --out docs/bench_results/35b.md
```

Bench harness: [tests/bench/comprehensive_bench.py](../tests/bench/comprehensive_bench.py).
Six stages — cold-warm latency / sustained TPS / tool-call clean / multi-turn stability / VRAM steady-state / long-context needle.

## Detailed per-model results

### Qwen3.6-35B-A3B-FP8 (MoE)

```
Endpoint:     http://127.0.0.1:8000
Model:        qwen3.6-35b-a3b
Patches ON:   45 / 78 unique (per Genesis structured boot summary)

[1] Cold-warm latency (5×400t, trimmed mean of 5)
    runs:  2.51s, 2.36s, 2.31s, 2.31s, 2.36s
    trimmed mean: 2.34s

[2] Sustained TPS (10 iterations, 400 tokens each)
    iter 1:  189.2 tok/s   iter 6:  192.9 tok/s
    iter 2:  190.3 tok/s   iter 7:  183.9 tok/s
    iter 3:  198.8 tok/s   iter 8:  190.3 tok/s
    iter 4:  203.4 tok/s   iter 9:  208.2 tok/s
    iter 5:  183.9 tok/s   iter10:  188.3 tok/s
    mean:    192.9 tok/s   CV: 4.19%   range: 183.9 – 208.2

[3] Tool-call clean rate (10 different prompts):  10/10 (100%)
    {Berlin, Tokyo, Sydney, New York, London, Paris, Madrid, Moscow, Shanghai, Mumbai}

[4] Multi-turn stability (10-turn soak)
    turn 1:  1.11s   turn 6:  1.14s
    turn 2:  1.06s   turn 7:  1.18s
    turn 3:  1.20s   turn 8:  1.22s
    turn 4:  1.20s   turn 9:  1.21s
    turn 5:  1.05s   turn10:  1.12s
    mean:    1.15s

[5] VRAM steady-state:  GPU0 22687 MiB | GPU1 21998 MiB | total 44685 MiB
```

### Qwen3.6-27B-int4-AutoRound (Lorbus dense + hybrid GDN)

```
Endpoint:     http://127.0.0.1:8000
Model:        qwen3.6-27b
Patches ON:   45 / 78 unique (per Genesis structured boot summary)

[1] Cold-warm latency (5×400t, trimmed mean of 5)
    runs:  4.74s, 4.76s, 4.76s, 4.76s, 5.06s
    trimmed mean: 4.76s

[2] Sustained TPS (10 iterations, 400 tokens each)
    mean:    95.6 tok/s   CV: 4.04%   range: 88.1 – 102.3

[3] Tool-call clean rate (10 different prompts):  10/10 (100%)

[4] Multi-turn stability (10-turn soak)
    turn 1:  2.11s   turn 6:  2.18s
    turn 2:  2.11s   turn 7:  2.29s
    turn 3:  2.05s   turn 8:  2.53s
    turn 4:  2.42s   turn 9:  2.53s
    turn 5:  2.36s   turn10:  2.63s
    mean:    2.32s

[5] VRAM steady-state:  GPU0 22753 MiB | GPU1 22064 MiB | total 44817 MiB
```

## What's enabled in PROD (2026-05-05)

Per Genesis structured boot summary (single block, replaces scattered per-patch INFO lines):

```
══════════════════════════════════════════════════════════════════════════════
Genesis vLLM Patcher — boot summary
══════════════════════════════════════════════════════════════════════════════
  Genesis:  v7.72
  vLLM:     0.20.2rc1.dev9+g01d4d1ad3
  GPU:      2× NVIDIA RTX A5000 (sm_86)
──────────────────────────────────────────────────────────────────────────────
  Patches:  78 total  →  45 APPLY  |  33 SKIP
  By category:
    • compile_safety         APPLY=  4  SKIP=  0
    • hybrid                 APPLY=  1  SKIP=  1
    • kernel                 APPLY=  1  SKIP=  0
    • kernel_safety          APPLY=  1  SKIP=  0
    • memory_savings         APPLY=  2  SKIP=  0
    • model_correctness      APPLY=  1  SKIP=  1
    • perf_hotfix            APPLY= 12  SKIP=  5
    • quantization           APPLY=  1  SKIP=  0
    • request_middleware     APPLY=  1  SKIP=  0
    • spec_decode            APPLY= 13  SKIP= 18
    • stability              APPLY=  1  SKIP=  0
    • structured_output      APPLY=  8  SKIP=  2
══════════════════════════════════════════════════════════════════════════════
```

## Reproduction recipe

1. Pull the dev branch + boot the model:
   ```bash
   git clone https://github.com/Sandermage/genesis-vllm-patches
   cd genesis-vllm-patches
   bash scripts/start_27b_int4_TQ_k8v4.sh   # or start_35b_fp8_PROD.sh
   ```
2. Wait for the structured boot summary line to appear in `docker logs vllm-server-mtp-test`.
3. Run the comprehensive bench:
   ```bash
   pip install --no-deps -e tools/genesis_vllm_plugin
   GENESIS_MODEL=qwen3.6-35b-a3b python3 tests/bench/comprehensive_bench.py
   ```
4. (Optional) Skip the long-context needle ladder for a faster run:
   ```bash
   GENESIS_MODEL=qwen3.6-35b-a3b python3 tests/bench/comprehensive_bench.py --skip-needle
   ```

## Cross-rig validators (call for replication)

Genesis numbers above are 2× RTX A5000 single-rig. Cross-rig validation requested from:

- **noonghunna** (1× RTX 3090, 4× RTX 3090 club-3090) — long-time Cliff 2 + tool-call collaborator
- **apnar** (1× RTX 5090 sm_120 consumer Blackwell) — first sm_120 production rig (club-3090#51 thread)
- **tfriedel** (4× RTX 3090) — vendors Genesis as submodule, runs verify-full.sh against same checkpoints
- **Quentin Machu** (varies, fork commits) — P64 sub-patch E author + bug-class triage
- **MidasMining**, **JartX**, **jhsmith409**, **webcodes-cz** — hardware variety (5090, H20, R6000 Blackwell, 8× A4000)

If you are running Genesis on hardware not listed, please file a benchmark report at `tests/bench/cross_rig_reports/` (PR welcome).
