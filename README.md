# Genesis vLLM Patches

**Runtime patches for [vLLM](https://github.com/vllm-project/vllm) — long-context Qwen3-class inference on consumer NVIDIA Ampere with TurboQuant k8v4 KV cache.**

> **Production status:** **v7.53, 2026-04-27.** Running 24/7 on 2× RTX A5000 with Qwen3.6-35B-A3B-FP8 + MTP K=3 spec-decode + 256K context. Latest sprint shipped **P82 SGLang acceptance OR-clause** (+12% TPS at threshold=0.3, validated on prod with 32/33 quality + 30/30 stability + 0 artifact flags).

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Production](https://img.shields.io/badge/prod-validated-green.svg)](#production-baseline)
[![Patches](https://img.shields.io/badge/patches-59-orange.svg)](PATCHES.md)
[![Throughput](https://img.shields.io/badge/throughput-160--190%20tok%2Fs-brightgreen.svg)](#measured-throughput-table)

---

## Table of contents

1. [Why this fork?](#why-this-fork)
2. [Quick start](#quick-start)
3. [Documentation map](#documentation-map)
4. [Production baseline](#production-baseline)
5. [Measured throughput](#measured-throughput-table)
6. [Workload-specific launch recipes](#workload-specific-launch-recipes)
7. [P67 kernel — the flagship](#p67p67b--the-flagship-genesis-original-kernel)
8. [Tests & validation](#tests--validation)
9. [Long-context verification](#long-context-verification)
10. [What's in the box](#whats-in-the-box)
11. [How patches work](#how-patches-work)
12. [Recent changes](#recent-changes)
13. [Compatibility matrix](#compatibility-matrix)
14. [Architecture](#architecture)
15. [Contributing](#contributing)
16. [Credits](#credits)
17. [Author / License](#author)

---

## Why this fork?

vLLM is a great inference engine, but on **consumer Ampere with FP8 + 4-bit KV cache + speculative decoding + 256K context** several rough edges show up:

- **TurboQuant** (FP8 K + 4-bit V) on Ampere needs explicit BF16→FP8 casts and page-size unification
- **Hybrid models** (GDN, Mamba) hit deadlock guards and dual-stream contention on consumer SMs
- **Spec-decode** (MTP, ngram, EAGLE) has structural acceptance ceilings and async-scheduler races
- **MoE on Ampere** needs Marlin tuning; FP8 block-scaled MM is slow at low M (decode + K+1 verify)
- **Long-context** (>128K) hits prealloc constants, scheduler clamps, and chunked-prefill races
- **Qwen3 thinking + tool-calling** needs reasoning-end timing fixes and multi-tool parsing repairs

Genesis is **59 runtime patches** that fix these classes of issues. Most are backports of in-flight upstream PRs (with author credit); some are Genesis-original (notably **P67** — a Triton multi-query kernel for spec-decode K+1 verify, +32% TPS, and **P82** — SGLang OR-clause acceptance, +12% TPS).

**Patches are opt-in via env flags.** Production launch scripts curate a working set; users can A/B individual patches via `GENESIS_ENABLE_<patch>=0|1`. See [`PATCHES.md`](PATCHES.md) for the full inventory.

---

## Quick start

Two install paths — pick whichever matches your operational style:

### Option A — Docker (recommended for prod)

```bash
# 1. Clone the patcher
git clone https://github.com/Sandermage/genesis-vllm-patches.git
cd genesis-vllm-patches

# 2. Adapt one of the example compose files
cp docker-compose.example.yml docker-compose.yml
# edit model path + GPUs + env flags

# 3. Launch (bind-mounts _genesis/ into vllm/vllm-openai:nightly)
docker compose up -d

# 4. Verify
curl -s http://localhost:8000/v1/models -H "Authorization: Bearer genesis-local"
```

### Option B — Bare-metal (no containers, full control)

For users who want to install Python + vLLM + Genesis directly on the host (no Docker), see [`INSTALL.md` § Bare-metal install](INSTALL.md#bare-metal-install-without-docker). The bare-metal path covers:

- Pre-requisite check (driver / CUDA / Python 3.12 / system libs)
- Dedicated venv + vLLM nightly install (3 strategies: pinned wheel / source / latest)
- Genesis package install via symlink or copy
- Patch enable flags + `apply_all` text-modification of `site-packages/vllm/`
- Production launch script + `systemd` unit
- Update workflow (Genesis update vs vLLM nightly upgrade)
- Bare-metal-specific troubleshooting

### Where to read next

- **[`QUICKSTART.md`](QUICKSTART.md)** — get a working server in 5 minutes
- **[`INSTALL.md`](INSTALL.md)** — full installation walkthrough (Docker AND bare-metal paths)
- **[`CONFIGURATION.md`](CONFIGURATION.md)** — every env var documented (production launch defaults, patch toggles, P67 tuning, diagnostic flags, rollback recipes)

---

## Documentation map

| Doc | When to read it |
|---|---|
| [`README.md`](README.md) | You're here — project overview |
| [`QUICKSTART.md`](QUICKSTART.md) | First-time setup, 5-min walkthrough |
| [`INSTALL.md`](INSTALL.md) | Full install (driver + CUDA + Python + vLLM + Genesis) |
| [`CONFIGURATION.md`](CONFIGURATION.md) | Every env var, defaults, rollback recipes |
| [`PATCHES.md`](PATCHES.md) | All 59 patches × status × env flag × upstream PR × credit |
| [`CREDITS.md`](CREDITS.md) | Comprehensive attribution log (every backport, every contributor) |
| [`MODELS.md`](MODELS.md) | Tested model configurations (Qwen3, Gemma, Llama variants) |
| [`SPONSORS.md`](SPONSORS.md) | Hardware / time sponsors |
| [`vllm/_genesis/CHANGELOG.md`](vllm/_genesis/CHANGELOG.md) | Per-version technical changelog (deep, append-only) |
| [`docs/sprint_reports/`](docs/sprint_reports/) | Time-stamped engineering sprint reports (Russian, audit-trail style) |
| [`docs/upstream/`](docs/upstream/) | Drafts and decisions for upstream vLLM PRs |
| [`docs/reference/`](docs/reference/) | Long-form technical references (memory architecture, bot setup) |

---

## Production baseline

| Component | Version |
|---|---|
| **Genesis patcher** | `v7.53` (P82 deployed at threshold=0.3) |
| **vLLM** | `0.19.2rc1.dev205+g07351e088` (image `vllm/vllm-openai:nightly`) |
| **PyTorch** | `2.11.0+cu130` |
| **Triton** | `3.6.0` |
| **CUDA** | `13.0` |
| **NVIDIA driver** | **`≥ 580.126.09` REQUIRED** ⚠️ (older drivers put PyTorch in 3× slower compat fallback) |
| **GPU** | 2× NVIDIA RTX A5000 (Ampere SM 8.6, 48 GB total VRAM), TP=2 |
| **Model** | Qwen3.6-35B-A3B-FP8 + TurboQuant k8v4 KV cache |
| **OS** | Ubuntu 24.04.4 LTS, kernel 6.8 |
| **Spec-decode** | MTP K=3 + P82 SGLang OR-clause (threshold=0.3) |
| **Context window** | 262,144 tokens (256K) |

Production launch script: [`scripts/launch/start_v747_p82.sh`](scripts/launch/start_v747_p82.sh) (= v743_p81 baseline + P82 t=0.3).

### Measured throughput table

**P82 sweep on prod, 2026-04-27 (3 runs each, mean tok/s):**

| max_tokens | Baseline (P82 OFF) | t=0.2 | t=0.3 ⭐ | t=0.5 |
|---|---|---|---|---|
| 64 | 188.1 | 183.7 (-2.3%) | 185.8 (-1.2%) | 185.8 (-1.2%) |
| 128 | 165.2 | 173.1 (+4.8%) | **187.7 (+13.6%)** | 168.4 (+1.9%) |
| 256 | 149.0 | 157.7 (+5.8%) | **162.4 (+9.0%)** | 158.8 (+6.6%) |
| 512 | 135.4 | 151.7 (+12.0%) | 150.1 (+10.9%) | 138.5 (+2.3%) |
| 1024 | 132.1 | 144.7 (+9.5%) | **144.8 (+9.6%)** | 134.7 (+2.0%) |
| 2048 | 130.1 | 146.5 (+12.6%) | **153.5 (+18.0%)** | 140.5 (+8.0%) |
| **mean (≥128)** | 142.4 | 154.7 (+8.6%) | **159.7 (+12.2%)** | 148.2 (+4.1%) |

**Quality (genesis_quality_harness, 33 tests × 10 categories):** 32/33 PASS at all thresholds (single fail `code_fib` is identical across all runs — model returns valid Y-combinator lambda not the simple solution the harness expects, NOT a P82 regression).

**Stability (30 sequential requests):** 100% success rate at all thresholds, stdev ~12 tok/s (~7% of mean).

**Artifact check (9 probes × 3 thresholds = 27):** zero flags for `<think>` leak, `<|im_start|>` / `<|im_end|>` / `<|endoftext|>` token leakage, raw `<tool_call>` tag in content, triple repetition. Tool-call format clean, JSON-mode clean, multilingual (RU + ZH) preserved.

Full sweep details: [`docs/sprint_reports/SPRINT_REPORT_20260427_phase3_p82_sweep_RU.md`](docs/sprint_reports/SPRINT_REPORT_20260427_phase3_p82_sweep_RU.md).

### Historical perspective (where we came from)

| Metric | v7.13 (driver 570 + per-layer) | v7.48 (driver 580 + shared P38/P40 + P81) | v7.53 (+ P82 t=0.3) |
|---|---|---|---|
| Throughput mean | 130-143 tok/s | 160-190 tok/s (+15-30% vs v7.13) | **187 tok/s @ 128 tok** (+12% vs v7.48) |
| Quality 30-shot | 30/31 PASS | 30/31 PASS | 32/33 PASS (extended harness) |
| Tool-call regression | 2/2 PASS | 2/2 PASS | 2/2 PASS |
| Long-ctx 16K-160K | PASS | PASS | PASS |
| Long-ctx 200K | OOM | **PASS** (153K server tokens) | PASS |
| GMU at which 200K runs | 0.91 (limit) | **0.90** (Sander obligatory range MET) | 0.90 |

---

## Workload-specific launch recipes

Production env flags differ slightly per workload class. Pick the closest match:

### Free-form / general-purpose (most users, recommended)

```bash
# Qwen3.6-35B-A3B-FP8 + MTP K=3 + P82 t=0.3 + TurboQuant k8v4
# Expected: ~187 tok/s @ 128 tok, ~154 tok/s @ 2048 tok
docker run -d --name vllm-server --gpus all -p 8000:8000 \
  -v /path/to/genesis-vllm-patches/vllm/_genesis:/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro \
  -v /path/to/models:/models:ro \
  -e GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1 \
  -e GENESIS_ENABLE_P67B=1 \
  -e GENESIS_ENABLE_P82=1 -e GENESIS_P82_THRESHOLD_SINGLE=0.3 \
  -e GENESIS_ENABLE_P81_FP8_BLOCK_SCALED_M_LE_8=1 \
  -e GENESIS_ENABLE_P70_AUTO_STRICT_NGRAM=1 \
  -e GENESIS_BUFFER_MODE=shared \
  vllm/vllm-openai:nightly -c "
    python3 -m vllm._genesis.patches.apply_all
    exec vllm serve --model /models/Qwen3.6-35B-A3B-FP8 \
      --tensor-parallel-size 2 --gpu-memory-utilization 0.91 \
      --max-model-len 262144 --kv-cache-dtype turboquant_k8v4 \
      --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' \
      --async-scheduling --performance-mode interactivity
  "
```

### Tool-call / agentic-heavy workload

Add suffix decoding (P75, requires `arctic-inference` package):

```bash
# Add to env flags above:
-e GENESIS_ENABLE_P75_SUFFIX_DECODING=1 \
# Replace --speculative-config with method=suffix instead of mtp
# Expected: 99 tok/s mean, peak 175 tok/s on tool-call workload (+32% vs MTP for this class)
```

### Ngram-only deployment (no MTP / no draft model available)

Use adaptive K controller (P77) for free-form workloads where K=3 wastes forward passes:

```bash
# Add to env flags above:
-e GENESIS_ENABLE_P77_ADAPTIVE_NGRAM_K=1 \
# Auto-tunes K based on observed accept rate via EMA + hysteresis
```

For full launch script reference (with all env flags annotated) see [`scripts/launch/start_v747_p82.sh`](scripts/launch/start_v747_p82.sh) and [`CONFIGURATION.md`](CONFIGURATION.md).

---

## P67/P67b — the flagship Genesis-original kernel

P67 is a Triton multi-query attention kernel specifically for spec-decode K+1 verify. It's the highest-EV patch in the Genesis stack and the source of most throughput gains.

### What it does

For MTP K=3 (or any K≥1 spec-decode), the verify step needs attention over K+1 query tokens against shared K/V cache. Upstream vLLM dispatches this through the standard multi-query path, which:

1. Issues K+1 separate `tl.dot` MMAs per KV-tile (one per query token, m=8 each)
2. Loads K/V cache K+1 times (cache misses on every iteration)
3. Under-utilizes Tensor Cores (mma.sync.m16n8k16 wants m≥16; we run m=8 = 50% TC throughput)

P67 fixes #2 by **loading K/V tile ONCE per outer iteration** and looping over K_PLUS_1 internally with `tl.static_range`. P67b enables it on the FULL cudagraph capture path via routing fix.

### Architecture (split-M, the production default since v7.34)

```text
Grid: (batch, num_kv_heads)
Outer loop: KV tiles (BLOCK_KV=32, num_stages=3 pipelined)
  K/V loaded once per tile (cache_modifier=".cg" streaming, scales=".ca" hot)
  Inner static_range over K_PLUS_1=4:
    Q[t] loaded with cache_modifier=".ca"
    S = SCALE * Q[t] @ K_tile  (tf32x3 IEEE precision)
    per-row causal mask
    online softmax update (tl.exp2 + LOG2E)
    acc += P @ V_tile
  Each q_t writes its own row of output [B, K+1, Hq, D]
```

Key opts (v7.50/v7.51):

- `tl.exp2(x * LOG2E)` instead of `tl.exp(x)` — direct hardware ex2, one fewer fp mul per softmax step
- `-FLT_MAX` sentinel instead of `float("-inf")` — avoids `inf*0=NaN` traps with masked-out KV blocks
- `cache_modifier=".cg"` for streaming K/V — saves L1 for Q + scales
- `cache_modifier=".ca"` for Q + scales/zeros + block_table — kept hot in L1
- `tl.range(num_stages=3)` pipelining hint — async-copy/MMA overlap
- Hoisted KV-head-byte invariant out of inner loop

### Empirical (per-version contribution to P67 throughput)

| Version | Δ vs prior | Cumulative vs v7.13 |
|---|---|---|
| v7.13 baseline | — | — |
| v7.22 (sanitized Inf/NaN→0) | +32% (no-spec) | +32% |
| v7.50 (cache_modifier + tl.range) | +10% @ 256 tok | +45% |
| v7.51 (tl.exp2 + -FLT_MAX) | +6.1% stability, +11% long-gen | +55% |
| v7.52 (fused-M opt-in) | -7% mean (kept opt-in for future Blackwell) | (no change to default) |

### Empirically rejected paths (preserved in source as "do not redo")

| Step | What was tried | Result | Why |
|---|---|---|---|
| **D** | Autotune over (BLOCK_KV, NUM_WARPS, NUM_STAGES) | All 5 alternatives -3.8% to -5.8% | Current `(32, 8, stages=3)` IS optimum |
| **E** | num_stages 3→2 (lower register pressure) | -2 to -9% | Dequant-heavy kernel needs deep pipeline for DRAM latency hiding |
| **F** | Manual Q hoist outside KV loop | -5 to -12% | Triton compiler already hoists via `tl.static_range` + `.ca` |
| **3 H** | Fused-M (m=32 single dot, vector online softmax) | -7% throughput on A5000 | Register spill (acc tile [32, 128] fp32 = 16 KB exceeds 64 KB SM register file). **Quality preserved** — kept opt-in for future Blackwell |
| **3 I** | vllm#38786 2D split (KV_SPLITS × TEMP_PAGES=32) | NOT IMPLEMENTED | Wrong kernel family (MLA, head_dim=512); author disowned PR post-#33529 |

Full P67 history: search [`vllm/_genesis/CHANGELOG.md`](vllm/_genesis/CHANGELOG.md) for `P67`. Source: [`vllm/_genesis/kernels/p67_multi_query_kernel.py`](vllm/_genesis/kernels/p67_multi_query_kernel.py) (696 lines, both split-M and fused-M architectures).

---

## Tests & validation

Genesis ships three independent validation tools, all in `scripts/`:

### 1. `genesis_bench_v4.py` — speed + stability bench

Tests: speed (max_tokens 64-2048), context window (4K-160K), context sweep (148K-160K @ 2K step), stability (30 sequential requests), stress (rapid-fire bursts), long-generation (1024/2048).

`v4` supersedes `v3`. Both files are byte-identical in this release; `v3.py` is kept as an alias for downstream scripts that pinned the old name. Migration: any caller of `genesis_bench_v3.py` can switch to `genesis_bench_v4.py` with no flag changes. `v3` will be deleted in Genesis v8.0.

```bash
# Free-form throughput (3 runs per max_tokens):
python3 scripts/genesis_bench_v4.py --host localhost --port 8000 \
  --label run_label --speed-runs 3

# Tool-call quality:
python3 scripts/genesis_bench_v4.py --host localhost --port 8000 \
  --label tool_call_run --skip-speed --skip-context --skip-stability

# Long-context (single 6-digit needle, max_tokens cap to avoid thinking eat-up):
python3 scripts/genesis_bench_v4.py --host localhost --port 8000 \
  --max-context-k 200 --skip-speed
```

### 2. `genesis_quality_harness.py` — regression GO/NO-GO

33 tests across 10 categories: math (8), factual (6), logic (4), code (4), multistep (2), tool_call (2), multilingual RU+ZH (2), coherence (1), JSON (2), long_context needle 10K/30K/80K (3).

```bash
python3 scripts/genesis_quality_harness.py --host localhost --port 8000 \
  --label baseline_run

# With baseline JSON for diff:
python3 scripts/genesis_quality_harness.py --host localhost --port 8000 \
  --label after_change --baseline harness_baseline_*.json
```

Exit codes: 0 = all critical PASS, 1 = at least one critical FAIL, 2 = harness error.

### 3. `genesis_context_sweep.py` — needle recall across context sizes

Builds prompt with needle paragraph at start + filler to hit target N tokens + needle question at end. Verifies the model recalls the canary.

```bash
python3 scripts/genesis_context_sweep.py --host localhost --port 8000 \
  --target-tokens 4000,16000,64000,128000,200000
```

### Container validation

```bash
# Unit tests (CPU-only, no GPU required):
./validate_unit.sh

# Integration test (boots a real container, runs harness end-to-end):
./validate_integration.sh
```

### Verifying P82 / patch state

```bash
# Check apply_all matrix in container logs:
docker logs vllm-server-mtp-test 2>&1 | grep "Genesis Dispatcher"

# Verify model_detect profile:
docker exec vllm-server-mtp-test python3 -m vllm._genesis.model_detect

# Verify applied patches at runtime:
docker exec vllm-server-mtp-test python3 -c "
from vllm._genesis import dispatcher
for pid, m in dispatcher.PATCH_REGISTRY.items():
    print(f'{pid}: {m[\"title\"][:60]}')
"
```

---

## Long-context verification

**P74 chunk-clamp + P72 profile_run cap, batched=8192:**

| Context (tokens) | Result | Latency |
|---|---|---|
| 14,640 | PASS | 2.5s |
| 73,040 | PASS | 9.4s |
| 153,000 | PASS | 18.1s |
| **200,000** | **PASS** ⭐ (was OOM in v7.13) | 24.7s |
| 240,000 | PASS (without OOM) | 31.2s |

Tested with 6-digit needle paragraphs at start + question at end, max_tokens capped at 64 to avoid thinking-mode token eat-up.

---

## What's in the box

```text
genesis-vllm-patches/
├── README.md                  ← you are here
├── QUICKSTART.md              ← 5-minute deploy
├── INSTALL.md                 ← full installation
├── CONFIGURATION.md           ← every env var
├── PATCHES.md                 ← 59 patches × metadata × credits
├── CREDITS.md                 ← attribution log
├── MODELS.md                  ← tested model configs
├── SPONSORS.md                ← hardware sponsors
│
├── vllm/_genesis/             ← THE patch package (drop into vLLM install)
│   ├── __init__.py
│   ├── CHANGELOG.md           ← per-version technical changelog
│   ├── dispatcher.py          ← PATCH_REGISTRY (P56+ rich metadata) + apply gate
│   ├── apply_all.py           ← orchestration entrypoint
│   ├── kernels/               ← Triton kernels (P67 multi-query, etc.)
│   ├── wiring/                ← text-patcher hooks (52 wiring files)
│   ├── patches/               ← @register_patch hooks
│   ├── configs/               ← MoE tuning JSONs (RTX A5000)
│   ├── tests/                 ← pytest unit suite
│   └── ...
│
├── scripts/                   ← bench + quality + diagnostic tools
│   ├── genesis_bench_v3.py    ← speed/context/stability benchmark
│   ├── genesis_quality_harness.py  ← regression GO/NO-GO suite
│   ├── genesis_context_sweep.py    ← needle-recall context sweep
│   ├── launch/                ← reference launch scripts
│   └── ...
│
├── docs/                      ← extended documentation
│   ├── sprint_reports/        ← time-stamped engineering reports
│   ├── reference/             ← long-form technical refs
│   └── upstream/              ← upstream vLLM PR drafts/decisions
│
├── benchmarks/                ← historical bench JSON results
├── reference/                 ← upstream PR diff snapshots (#40792, #40798, etc.)
├── genesis_vllm_plugin/       ← vLLM plugin (auto-loaded via entry point)
│
├── docker-compose.example.yml ← starting point for your deploy
├── docker-compose.<variant>.yml  ← Gemma 4 26B MoE, Qwen3.5 dense, integration test, AWQ, FP16 KV, etc.
├── validate_unit.sh           ← run unit tests
└── validate_integration.sh    ← end-to-end container validation
```

---

## How patches work

Each patch is one of:

1. **Wiring** (`vllm/_genesis/wiring/patch_<id>_*.py`) — text-patches a specific anchor in vLLM source. Idempotent (re-running is a no-op). Drift-detected (if upstream changes the anchor, patch SKIPS gracefully).
2. **Kernel** (`vllm/_genesis/kernels/`) — pure-Python drop-in replacement (a Triton kernel + Python launcher) that's wired into vLLM via a wiring patch.
3. **Library** (`vllm/_genesis/buffer_mode.py`, `prealloc.py`, `model_detect.py`, etc.) — utility module loaded by other patches.

At server start, `python3 -m vllm._genesis.patches.apply_all` runs once, walking every `@register_patch`-decorated function. For each patch the dispatcher checks:

1. Env flag (`GENESIS_ENABLE_<patch>=1`?)
2. Hardware fingerprint (Ampere consumer? data-center? Ada? Hopper? Blackwell?)
3. Model fingerprint (Qwen3 MoE? hybrid? dense? has tool support?)
4. Drift markers in target file (upstream merged a competing fix?)

Decisions are logged in a single matrix:

```text
P82   | APPLY  | SGLang threshold_single OR-clause acceptance | opt-in env (config: neutral) | SGLang team
P81   | APPLY  | fp8 block-scaled MM low-M decode tuning      | opt-in env (config: neutral) | tonyliu312 (vllm#40925)
P78   | SKIP   | TurboQuant .tolist() capture-guard           | opt-in only — set GENESIS_ENABLE_P78_TOLIST_CAPTURE_GUARD=1 to engage | noonghunna
P63   | SKIP   | MTP/Eagle drafter GDN state recovery (deprecated) | opt-in only AND empirically deprecated | Genesis-original (hypothesis disproven 2026-04-25)
...
```

Every decision (APPLY / SKIP / FAIL) is observable in container logs. Adding/removing a patch is a single env flag flip + container `down/up`.

**See [`PATCHES.md`](PATCHES.md) for the full patch inventory by category.**

---

## Recent changes

For the full per-version technical changelog see [`vllm/_genesis/CHANGELOG.md`](vllm/_genesis/CHANGELOG.md). Key milestones:

### v7.53 (2026-04-27) — P82 SGLang acceptance OR-clause [PRODUCTION]

Backported SGLang's `threshold_single` OR-clause from `speculative_sampling.cuh` into vLLM's `RejectionSampler`. Targets the structural ceiling `clean_rate ≈ accept_rate^num_spec` identified in the v7.13 strict-ngram analysis.

Empirical results @ threshold=0.3 (sweep across {0.2, 0.3, 0.5}, prod 2× A5000):

- **+12% mean throughput** on 128-2048 tok generation (up to +18% @ 2048 tok)
- **Quality 32/33** (PASS, identical fail across all thresholds is not P82-induced)
- **Stability 30/30** (100% success rate, stdev 12 tok/s)
- **Zero artifact flags** (no think-leak, no special-token leak, no malformed tool-call, no repetition)

Trade-off: BIASED rule (loses unbiased-sampling guarantee). Acceptable for greedy / low-temperature workloads where bias is in the right direction (target-confident tokens). Not recommended for T≥1.0 creative-writing.

### v7.52 (2026-04-27) — Tier 3 H fused-M kernel as opt-in (REJECTED for prod default)

Re-fused split-M back into a single MMA per KV-tile (m=32, vectorized online softmax with per-row causal mask `q_abs_pos[:, None] >= seq_offset[None, :]`). The v7.27 quality drift was finally fixed — but throughput regressed 7% on A5000 due to register spill (acc tile [32, 128] fp32 = 16 KB exceeds 64 KB SM register file shared across active warps).

Kept opt-in via `GENESIS_P67_USE_FUSED=1` for future Blackwell hardware (larger register file).

### v7.51 (2026-04-27) — P67 softmax precision

`tl.exp(x) → tl.exp2(x * LOG2E)` (one fewer fp mul per softmax step) + `float("-inf") → -FLT_MAX` sentinel (avoids `inf*0=NaN` under masked KV blocks). **+6.1% stability, +11% long-gen.**

### v7.50 (2026-04-27) — P67 cache hints + pipeline hints

`cache_modifier=".ca"/".cg"` for Q/scales/block_table vs streaming K/V + `tl.range()` pipelining hints. **+10% @ 256 tok**, stable across context sweep.

### v7.49 (2026-04-26) — Spec-decode cleanup

Retired P79d (njhill confirmed non-bug); improved P79c with `-1` placeholder discrimination + `prev_step_scheduled_req_ids` gate.

### v7.48 (2026-04-27) — Driver 580 + shared buffer pool + P81

Required driver bump after vLLM nightly went to PyTorch 2.11+cu130. Three changes:

1. **P81 backport of [vllm#40925](https://github.com/vllm-project/vllm/pull/40925)** — `w8a8_triton_block_scaled_mm` low-M decode tuning. **+23% median decode on GB10** per upstream.
2. **`vllm/_genesis/buffer_mode.py`** — env-driven toggle `GENESIS_BUFFER_MODE=shared|per_layer`.
3. **P38/P40 shared-pool fix** — both had per-call `torch.empty` fallback that defeated singleton intent on long-context. Fixed via `GenesisPreallocBuffer.get_or_create()` with single max-size namespace per (Hk, D, dtype, device) — one buffer reused across all 36 attention layers via slicing.

**Result: +15-30% throughput vs v7.13, 200K context now PASSES** (was OOM).

### v7.45 (2026-04-26) — gemini bot review fix

After opening upstream draft PR [vllm#40914](https://github.com/vllm-project/vllm/pull/40914), `gemini-code-assist` flagged that the new routing path wasn't forwarding `mid_o_buf` / `lse_buf` / `buf_holder=layer` cached decode buffers — kernel was allocating fresh tensors every call, defeating the cudagraph replay this PR was supposed to restore. **Bot was right — fix made code both more correct AND faster.**

12-run benchmarks:

| Config | Mean tok/s | std | CV | max |
|---|---|---|---|---|
| v7.40 baseline | 128.3 | 7.3 | 5.7% | 139 |
| v7.42 full-stack (pre-fix) | 127.09 | 8.37 | 6.6% | 140 |
| **v7.45 (with buf-reuse fix)** | **130.68** | **6.59** | **5.0%** | **141** |

### v7.42-v7.43 (2026-04-26) — almost 24-hour push

Closed the spec-decode method comparison loop. **12-run benchmarks per config, mean ± std:**

| Method | Free-form (tok/s) | Tool-call (tok/s) | Quality | CV | Verdict |
|---|---|---|---|---|---|
| **MTP (default)** | **127.0** | (not tested) | 5/5 PASS | 4.7% | **BEST overall, prod default** |
| ngram CPU (numba) | 46.6 | 75 (v7.13 historic) | 5/5 PASS | 4.4% | Fallback |
| ngram_gpu | 43.6 | 45.4 | 4/5 PASS | 3-13% | NO GAIN (V1 stale-data residual; needs PR #40704 V2) |
| **suffix decoding (P75)** | 45.9 | **99.0 mean (max 175!)** | 4/5 PASS | 16-36% | **WIN for tool-call workload (+32%)** |

Plus 4 new patches: P71 (block-verify, Sun 2024), P72 (profile_run cap), P74 (chunk-clamp companion), P77 (adaptive ngram K).

### v7.13 (2026-04-25) — strict-ngram breakthrough

Token-level tracing identified the structural ceiling formula `clean_rate ≈ accept_rate^num_spec`. Config-only fix `prompt_lookup_min=8` → **100% clean (single-query) / 96% (multi-query diverse)**. P82 (this sprint) attacks the residual ceiling with the SGLang OR-clause.

<details>
<summary>Older changelog entries (v7.9-v7.15)</summary>

### v7.15 (2026-04-25 late evening)

P67/P67b TQ multi-query kernel for spec-decode K+1 verify. Genesis-original Triton kernel. **+32% TPS** on no-spec baseline. Wired through P67b (FULL cudagraph routing).

### v7.14 (2026-04-25 evening)

P64 (qwen3coder MTP streaming early-return), P65 (TurboQuant CG downgrade), P66 (cudagraph_capture_sizes spec-decode divisibility filter).

### v7.13 (2026-04-25)

Tool-call clean rate from 20% → 53-70% (2.5-3.5× improvement). P59 + P60 + P60b + P61 + P61b + P62 backports. See [`docs/sprint_reports/REPORT_v7_1_INTEGRATION_20260424_RU.md`](docs/sprint_reports/REPORT_v7_1_INTEGRATION_20260424_RU.md).

### v7.9 (2026-04-24)

Initial v7 architecture: modular `_genesis/` package, dispatcher v2, apply_all orchestration, model_detect / config_detect probes.

</details>

---

## Compatibility matrix

| Hardware | Status | Notes |
|---|---|---|
| RTX A5000 (Ampere SM 8.6, 24 GB) | ✅ **prod-validated** | Primary dev hardware (2×) |
| RTX 3090 (Ampere SM 8.6, 24 GB) | ⚠️ should work | Same SM, slightly different SM count. **Confirmed cross-rig** by [@noonghunna](https://github.com/noonghunna) on [`qwen36-27b-single-3090`](https://github.com/noonghunna/qwen36-27b-single-3090) — strict-ngram fix transfers cross-rig. |
| RTX 4090 (Ada SM 8.9, 24 GB) | ⚠️ should work | Native FP8 (e4m3) — bypasses our e4b15 codepath. Untested. |
| A6000 (Ampere SM 8.6, 48 GB) | ⚠️ should work | More VRAM = bigger context windows. Untested. |
| A100 (Ampere SM 8.0, 40/80 GB) | ⚠️ should work | Data-center FP8 path. P81 less impactful (Marlin pre-tuned). Untested. |
| H100 (Hopper SM 9.0) | ⚠️ partial | Many of our patches are Ampere-specific workarounds; H100 ships native FP8 + better Marlin defaults. |
| ROCm / XPU | ❌ untested | We're CUDA-only. Patches use Triton (potentially portable) but no validation. |

For the model-side compatibility list (Qwen3 variants, Gemma 4, Llama 4, etc.) see [`MODELS.md`](MODELS.md).

### Upstream tracking (as of v7.53, 2026-04-27)

| Patch | Upstream PR status | Drift detection |
|---|---|---|
| P58 (z1ying) | OPEN draft | ✅ marker present, our patch active |
| P59 (ZenoAFfectionate) | OPEN | ✅ |
| P60 / P60b (tdoublep) | OPEN draft | ✅ Phase 1 + Phase 2 both backported |
| P61 / P61b (ExtReMLapin) | OPEN | ✅ |
| P62 (sfbemerk) | OPEN | ✅ |
| P64 (kotori-yan) | OPEN | ✅ |
| P71 (Z. Golpayegani) | OPEN draft + 2 fixes from gemini bot review | ✅ |
| P75 (Snowflake) | **MERGED** in our pin (vllm#25784) | ✅ enabler only — auto-no-op when upstream env present |
| P81 (tonyliu312) | OPEN | ✅ |
| P82 (SGLang algorithm) | not upstream — separate project | ✅ |

When an upstream PR merges, our backport's `upstream_drift_markers` detect it and the patch returns SKIPPED with a clear reason ("upstream may have absorbed this fix"). No code change needed in Genesis — just drop the env flag.

---

## Architecture

```text
vllm/_genesis/
├── __init__.py               # Public API
├── dispatcher.py             # PATCH_REGISTRY + should_apply() gate
├── guards.py                 # Vendor / chip / Python file resolution
├── interface_guard.py        # Genesis interface mismatch detection
├── model_detect.py           # MoE / hybrid / TQ probe (Qwen3 / Gemma / Llama)
├── config_detect.py          # Per-patch config-aware recommendation
├── memory_metrics.py         # Per-pool memory reporting
├── prealloc.py               # GenesisPreallocBuffer framework
├── prealloc_budget.py        # Central token-budget resolver
├── buffer_mode.py            # GENESIS_BUFFER_MODE shared|per_layer toggle
│
├── kernels/                  # Triton kernels + Python launchers
│   ├── p67_multi_query_kernel.py   # FLAGSHIP: spec-decode K+1 verify (+32% TPS)
│   ├── block_verify_sampler.py     # P71 Sun 2024 block-verify
│   ├── tq_continuation_prefill.py  # P22/P23/P26 continuation-prefill
│   ├── tq_decode_tune.py           # P36/P40 decode tuning
│   ├── tq_grouped_decode.py        # P40 grouped decode
│   ├── gdn_core_attn_manager.py    # P28 GDN core_attn buffer
│   ├── gdn_dual_stream.py          # P7 dual-stream parallelism
│   ├── gdn_dual_stream_customop.py # P7b custom_op variant
│   ├── gdn_gating_buffer.py        # P46 GDN gating
│   ├── fla_kkt_buffer.py           # P39a FLA chunk_scaled_dot_kkt
│   ├── moe_intermediate_cache.py   # P37 MoE pool
│   ├── marlin_tuning.py            # P17/P18 Marlin per-SM tuning
│   ├── marlin_fp32_reduce.py       # P23 FP32 reduce
│   ├── router_softmax.py           # P31 fp32 router softmax
│   ├── fp8_dispatcher.py           # P1/P2 FP8 dispatcher
│   ├── page_size_padded.py         # P5/P5b page-size unification
│   ├── block_table_zero.py         # P14 tail zero-fill
│   ├── adaptive_ngram_controller.py# P77 EMA + hysteresis K controller
│   └── dequant_buffer.py           # Shared dequant buffer pool
│
├── wiring/                   # Text-patcher hooks (52 files)
│   ├── text_patch.py         # TextPatcher framework (anchor-based, drift-detected, idempotent)
│   ├── rebind.py             # Attribute-rebind verification registry
│   ├── patch_67_tq_multi_query_kernel.py
│   ├── patch_82_sglang_acceptance_threshold.py     # NEW v7.53
│   ├── patch_81_fp8_block_scaled_m_le_8.py
│   └── ... (49 more, one per patch)
│
├── patches/
│   ├── apply_all.py          # Orchestration entrypoint (run at server start)
│   └── upstream_compat.py    # PR marker registry for drift detection
│
├── middleware/               # ASGI middleware (response cache, etc.)
├── configs/                  # MoE tuning JSONs (RTX A5000 specific)
├── cache/                    # Genesis-local Triton kernel cache
└── tests/                    # pytest unit suite (CPU-only, no GPU required)
```

---

## Contributing

PRs welcome. Bug reports from hardware we don't test (RTX 3090, 4090, A6000, A100, H100, ROCm, XPU) are especially valuable.

### Workflow for adding a new patch

1. **Pick a free patch ID.** Check `apply_all.py` `@register_patch` and `dispatcher.py` `PATCH_REGISTRY`. See [`PATCHES.md`](PATCHES.md) "Adding a new patch" section.
2. **Use a template** — [`patch_71_block_verify.py`](vllm/_genesis/wiring/patch_71_block_verify.py) for branch insertion or [`patch_82_sglang_acceptance_threshold.py`](vllm/_genesis/wiring/patch_82_sglang_acceptance_threshold.py) for in-kernel modification.
3. **Document in CHANGELOG** — add a `vX.YZ` entry to [`vllm/_genesis/CHANGELOG.md`](vllm/_genesis/CHANGELOG.md) explaining the WHY, empirical data, and ship/reject decision.
4. **Cite credits** in the patch docstring + [`CREDITS.md`](CREDITS.md) if backporting from someone else's PR / project.
5. **Static validation:** `python3 -c 'import ast; ast.parse(open("path/to/patch.py").read())'`
6. **Container validation:** `docker compose down && docker compose up -d` (NOT `stop/start` — see [`CONFIGURATION.md`](CONFIGURATION.md) "Container R/W layer note"; stop/start preserves the patched fs and the next monolith run fails on anchor drift).
7. **Empirical validation:** blue/green sweep with `genesis_quality_harness.py` + `genesis_bench_v3.py`. **Ship gate:** ≥30/31 quality + ≥+5% TPS (or whatever the patch targets).

### MoE tuning note

On Ampere with FP8 block quantization, vLLM selects **MARLIN**, not Triton, for MoE. Before tuning check the startup log:

```text
Using MARLIN Fp8 MoE backend out of potential backends: [...]
```

If MARLIN, the only runtime lever is `block_size_m` (P17). We ship a tuned JSON for RTX A5000 at [`vllm/_genesis/configs/moe_tuning/E=256,N=512,device_name=NVIDIA_RTX_A5000,dtype=fp8_w8a8,block_shape=[128,128].json`](vllm/_genesis/configs/moe_tuning/) — bind-mount it into the vLLM install path.

### Bug reports

Open an issue at [github.com/Sandermage/genesis-vllm-patches/issues](https://github.com/Sandermage/genesis-vllm-patches/issues) with:

- Hardware (GPU + driver + CUDA + RAM)
- Software (vLLM commit, PyTorch + Triton versions, Genesis version from `vllm/_genesis/__init__.py`)
- Container logs from server start (full apply_all matrix output)
- Reproducer (request body, expected vs actual)

---

## Credits

Genesis is built on top of work by many people. **Every backport names its upstream author + PR; every Genesis-original patch is explicitly labelled.**

Highlight contributors (see [`CREDITS.md`](CREDITS.md) for the full list):

- **DeepSeek-V3 team** — fp32 router upcast pattern (basis for P31)
- **@JartX** — TurboQuant author, `JartX/vllm#11` FP16 rotation (P20 prerequisite)
- **@jhsmith409** — endorsed Genesis Ampere investigation, pre-approved P22
- **@ZJY0516** — hybrid prefix cache design clarifications
- **@vibhavagarwal5** — collaborative PR scope guidance
- **@youkaichao** — memory profiler invariants documentation
- **@tdoublep** — vllm#40738 GDN+ngram state recovery (P60/P60b basis)
- **@noonghunna** — vllm#40807, vllm#40831 spec-decode investigation (P56/P57/P78 basis); cross-rig validation on RTX 3090
- **@tonyliu312** — vllm#40925 fp8 block-scaled MM low-M tuning (P81 basis)
- **@ZenoAFfectionate** — vllm#39055 Qwen3 reasoning embedded tool_call recovery (P59 basis)
- **@ExtReMLapin** — vllm#40783 Qwen3 multi-tool first-occurrence + streaming overlap (P61, P61b basis)
- **@sfbemerk, @cicirori** — vllm#36138, vllm#34650 structured-output spec-decode timing (P62 basis)
- **@kotori-yan** — vllm#39598 qwen3coder MTP streaming (P64 basis)
- **@z1ying** — vllm#40768 async-scheduler placeholder fix (P58 basis)
- **@bhaktatejas922** — vllm#39273 GDN+ngram interaction (P60 prior art)
- **Z. Golpayegani** — vllm#40819 block-verify rejection sampler (P71 basis); Sun et al. 2024 ICLR (arXiv 2403.10444) algorithm
- **@gemini-code-assist (bot)** — review on vllm#40914 catching the buffer-reuse fix (v7.45 P67b)
- **Arctic Inference team** — vllm#25784 Suffix Decoding (P75 enabler), arXiv 2411.04975
- **SGLang team** ([sgl-project/sglang](https://github.com/sgl-project/sglang)) — `speculative_sampling.cuh` threshold_single OR-clause (P82 basis); `adaptive_spec_params.py` adaptive K controller (P77 basis)
- **Nightjar authors** — arXiv 2512.22420 MAB-style auto-disable (P77 extension)
- **vLLM core team** (@WoosukKwon, @zhuohan123, @robertgshaw2-redhat, @bnellnm) — responsive community, educational codebase

**Per-kernel attribution lives in each module's docstring. Per-patch attribution lives in [`PATCHES.md`](PATCHES.md) and [`CREDITS.md`](CREDITS.md).**

---

## Author

**Sandermage(Sander)-Barzov Aleksandr**
Ukraine, Odessa
GitHub: [@Sandermage](https://github.com/Sandermage)
Project: [genesis-vllm-patches](https://github.com/Sandermage/genesis-vllm-patches)

For sponsors / supporters see [`SPONSORS.md`](SPONSORS.md).

---

## License

Apache-2.0 — see [`LICENSE`](LICENSE).

---

*Genesis vLLM Master Plan v7.0 / current production v7.53.*
*Canonical reference: [`Genesis_Doc/common/GENESIS_VLLM_MASTER_PLAN_v7.0_20260424.md`](https://github.com/Sandermage/Genesis_Doc) (private).*
*Latest sprint reports: [`docs/sprint_reports/`](docs/sprint_reports/).*
