# Genesis vLLM Patches

**Runtime patches for [vLLM](https://github.com/vllm-project/vllm) — Qwen3.6-class inference on consumer NVIDIA Ampere with TurboQuant k8v4 KV cache, MTP K=3 spec-decode, tool-calling, and 256K-class context.**

> **Status:** v7.62.x (2026-04-29). Production stack runs 24/7 on 2× RTX A5000 with Qwen3.6-35B-A3B-FP8 and Qwen3.6-27B-int4-AutoRound. Cross-rig validated on community RTX 3090 / 4090 deployments via [@noonghunna](https://github.com/noonghunna), [@thc1006](https://github.com/thc1006), [@Quentin-M](https://github.com/Quentin-M) and others.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![35B-A3B-FP8](https://img.shields.io/badge/35B--A3B--FP8-183_TPS-brightgreen.svg)](#reference-configs)
[![27B-INT4](https://img.shields.io/badge/27B--INT4-89_TPS_(short)-green.svg)](#reference-configs)
[![Long context](https://img.shields.io/badge/27B-256K_verified-blue.svg)](#reference-configs)
[![Patches](https://img.shields.io/badge/runtime_patches-65+-orange.svg)](PATCHES.md)

---

## Headline numbers

Measured on 2× RTX A5000 24 GB (Ampere SM 8.6), driver 580.126.09, vLLM nightly pin `8cd174fa3`, with the new `tools/genesis_bench_suite.py` decode-only TPOT methodology (port of [thc1006](https://github.com/thc1006)'s `bench_v3_clean_ab.py`):

| Workload | Model | Wall TPS | Decode TPOT | CV | Notes |
|---|---|---:|---:|---:|---|
| Free-form, 1K ctx | Qwen3.6-35B-A3B-FP8 + MTP K=3 + TQ k8v4 | **183.05** | 5.33 ms | 8.92% | N=500 stress, 200 min continuous |
| Tool-call, short ctx | Qwen3.6-27B-int4-AutoRound + MTP K=3 + fp8_e5m2 | **89.23** | 11.18 ms | 9.97% | N=500, tool-call 4/4 |
| Long context, 256K | Qwen3.6-27B-int4-AutoRound (v791b config) | ~80 | 12 ms | n/a | 262 104 token prompt, 311 s prefill, 845 t/s prefill |
| TQ k8v4 on hybrid | Qwen3.6-27B-int4 + TQ k8v4 + P98 | **90.49** | 11.01 ms | 10.08% | First-time TQ on hybrid GDN; +1.9% NS vs fp8_e5m2 |

**PN8 VRAM savings**: 23.7 GB → 22.66 GB per GPU (~1066 MiB), zero throughput cost. Combined with the lighter long-ctx config we now run **256K context on 2× A5000** — previously thought to require Blackwell.

---

## Table of contents

1. [What's new in v7.62.x — 36-hour session timeline](#whats-new-in-v762x--36-hour-session-timeline)
2. [Quick start](#quick-start)
3. [Reference configs](#reference-configs)
4. [Genesis Benchmark Suite](#genesis-benchmark-suite)
5. [Patch catalog](#patch-catalog)
6. [Per-GPU recommendations](#per-gpu-recommendations)
7. [Empirical findings — what worked, what didn't](#empirical-findings--what-worked-what-didnt)
8. [Hardware and operating envelope](#hardware-and-operating-envelope)
9. [How patches work](#how-patches-work)
10. [Acknowledgments](#acknowledgments)
11. [License / disclaimer](#license--disclaimer)

---

## What's new in v7.62.x — 36-hour session timeline

This release captures a focused 36-hour optimization sprint (2026-04-28 to 2026-04-29) that turned into a much bigger story than expected. We measured, disproved, fixed, and shipped — all empirically. The TL;DR up front, then the chronological narrative below.

### TL;DR — what changed

**Configuration & validation:**
- Real PROD baseline recalibrated: **183.05 TPS** on 35B-A3B-FP8 (CV 8.92%, N=500). The previously-tracked 162 TPS was a *one-off measurement using older methodology* — the same stack measured properly is 13% higher.
- 27B Lorbus INT4 stable at **TRUE 256K context** (262 104 tokens) via `v791b` config (util 0.90, max-num-seqs 2, max-num-batched-tokens 2048). Previously OOMed at 16K with the aggressive `v771b` PROD config — was config-aggressiveness, not a model limit.
- TurboQuant k8v4 unlocked on hybrid GDN models (Qwen3.6-27B Lorbus) via the **P4 + P98** combo. First time TQ runs on hybrid in Genesis. Tool-call 4/4 clean.

**New patches and infrastructure:**
- `PN8` — MTP/draft online-quant propagation (vllm#40849 backport). Frees ~1 GiB VRAM per GPU on FP8 + MTP.
- `PN9` — independent drafter attention backend (vllm#39930 self-retired upstream merged in our pin).
- `PN11` — GDN a/b contiguity defensive fix (vllm#41142, **by [Quentin Machu](https://github.com/Quentin-M)** via fork PR — first community-contributed Genesis patch).
- `P40` signature drift fix — backport of vllm#40792 brought into the current pin (was crashing with `_fwd_kernel_stage2 missing 3 args`).
- `P4` drift markers added for vllm#41123 self-retirement.

**Tooling:**
- `tools/genesis_bench_suite.py` — flagship community-grade benchmark suite. 7 phases, configurable contexts (1K → 512K), Welch's t-test compare. Outputs JSON + Markdown.
- `vllm/_genesis/gpu_profile.py` — per-GPU patch recommendation engine. 18-card datasheet (Ampere consumer + workstation + Ada + Hopper + **all 5 Blackwell PRO workstation cards**). Auto-prints `[REC]/[OFF]` per patch at boot.
- `docs/BENCHMARK_GUIDE.md` — 5-environment run guide (bare metal / Docker / Proxmox / WSL2 / RunPod).

**Disproven (do NOT enable on Ampere consumer):**
- P83 + P84 + P85 (prefix-cache "cake-and-eat" stack) → -29% TPS regression. Even with the supposedly-root-cause P84.
- P40 default-on (only ~+1% on Ampere consumer; needs L2 ≥ 24 MB to deliver — recommended for 4090, 5090, H100, Blackwell).

### Day 1 — 2026-04-28: 5-agent parallel research sweep

The session opened with a multi-source intelligence gathering pass:

- **Agent A (noonghunna fork audit)** — Reviewed both `noonghunna/qwen36-dual-3090` and `noonghunna/qwen36-27b-single-3090` repos for cross-rig configs, reported numbers, and divergences from our PROD.
- **Agent B (other downstream forks)** — Audited `thc1006/qwen3.6-vllm-2x3090`, `danbedford/qwen36-dual-3090-nvlink`, `AlexsJones/llmfit` plus three lighter community repos.
- **Agent C (vLLM main vs nightly diff, 6 mo)** — Found 47 commits ahead of our pin worth tracking, 13 high-relevance PRs.
- **Agent D (vLLM PRs deep-dive, 6 mo)** — Read all open and recently-merged PRs touching Qwen3.x, MTP, GDN, TurboQuant, Marlin/AutoRound.
- **Agent E (vLLM Issues deep-dive, 6 mo)** — Catalogued the open-bug landscape relevant to our exact stack intersection (Qwen3.6 × MTP × hybrid × TQ × Ampere consumer).

Key day-1 findings written to memory:
- `feedback_v791_27b_long_ctx_breakthrough.md`
- `feedback_pn8_verified_vram_savings.md`
- `feedback_v792_lorbus_TQ_k8v4_works.md`
- `feedback_int8_gs128_torch_compile_crash.md`

The day-1 sweep produced 5 full reports (saved to gitignored `docs/_internal/`). Two findings turned into shipping patches (PN8, PN11) and one into a root-cause memory note (INT8-gs128 boot crash).

### Day 2 morning — 4-arm A/B on prefix caching

Going into Day 2 we'd queued several "let's enable this and bench" actions. Top of the list: enable `--enable-prefix-caching` on 35B PROD with the P83 + P84 + P85 stack (the "cake-and-eat" hypothesis from prior internal investigation).

We ran a controlled 4-arm A/B:

| Arm | Config | wall_TPS | decode TPOT |
|---|---|---:|---:|
| **A** | v775 baseline (no prefix-cache) | **183.27** ± 15.91 | 5.33 ms |
| **B** | + `--enable-prefix-caching`, no P83/P84/P85 | 128.07 ± 11.70 | 7.69 ms |
| **C** | + P83 + P85 (without root-cause P84) | 130.87 ± 12.55 | 7.53 ms |
| **D** | + P83 + P84 + P85 + `HASH=16` (full stack) | 129.07 ± 12.39 | 7.64 ms |

Verdict: **all three "fix" arms still regressed by ~29%.** The supposedly-root-cause P84 didn't help. Welch's t-test on Arm A vs Arm D gave p < 0.0001 — significant *regression*.

Conclusion (saved to `feedback_p83_p84_p85_cache_no_cake.md`): **do NOT enable prefix-cache on this stack.** The cache machinery overhead is real and the patches we have don't recover the missed throughput. Whether this is a current-pin regression or a fundamental vllm cache-design issue is for a future investigation.

### Day 2 noon — P40 signature drift, fixed but no measurable win

Next on the queue: enable P40 (TurboQuant k8v4 GQA grouping kernel — backport of vllm#40792). PR author claimed +27% on Hopper.

First attempt: instant crash:
```
TypeError: _fwd_kernel_stage2() missing 3 required positional arguments:
  'stride_lse_bs', 'BLOCK_DV', 'Lv'
```

Cause: signature drift. The upstream `_fwd_kernel_stage2` had grown 3 new args between when P40 was originally written and the current pin. Our patch was passing the old signature.

Fixed in [`vllm/_genesis/wiring/patch_40_tq_grouped_decode.py:320`](vllm/_genesis/wiring/patch_40_tq_grouped_decode.py#L320) — added `seq_lens`, `BLOCK_DV`, `Lv`, `OUTPUT_FP16` matching the upstream callsite.

Re-tested: no crash, but **+1.14% — Welch p=0.284 NOT SIGNIFICANT**. The +27% from upstream is real, but only on Hopper-class hardware where L2 (50+ MB) is large enough that the grouping benefit can land. On Ampere consumer with 4 MB L2 (A5000) we're memory-bandwidth-bound — the patch executes correctly but has nothing to optimize.

Lesson recorded to `feedback_p40_broken_on_current_pin.md`: **P40 default OFF on Ampere consumer; recommended on RTX 4090, 5090, all Blackwell, H100, H200.** This finding triggered the GPU profile recommendation system later in the session.

### Day 2 afternoon — PN8 PROMOTED (~1 GiB saved per GPU)

PN8 = MTP/draft online-quant propagation, backport of [vllm#40849](https://github.com/vllm-project/vllm/pull/40849) by [@bhoomit](https://github.com/bhoomit). Idea: when the target model uses online-quant (e.g., compressed-tensors FP8), the MTP draft head currently loads in BF16 even though it could inherit the same online-quant config — wasting ~600 MiB.

Bench result on 35B-A3B-FP8:

| Metric | Baseline (v775) | + PN8 (v780) | Delta |
|---|---:|---:|---:|
| wall_TPS | 183.27 | 184.47 | +0.65% (NS, p=0.561) |
| decode TPOT (ms) | 5.33 | 5.30 | -0.6% (NS) |
| Tool-call | 3/4 | 3/4 | unchanged |
| **GPU 0 memory** | **23.7 GB** | **22.66 GB** | **-1066 MiB (-4.5%)** |

Throughput stays in the noise. **VRAM savings are real** and predictable. With 2× TP we recovered ~2.1 GB total — enough to bump `gpu-memory-utilization` from 0.90 → 0.93 if you want bigger KV pool, OR bump max-num-seqs, OR push max-model-len.

Saved to `feedback_pn8_verified_vram_savings.md`. PROMOTED → v780. **Default ON for any FP8 + MTP target.**

### Day 2 afternoon — INT8 sprint aborted (vllm bug, not us)

Plan was to tackle Minachist Qwen3.6-27B-INT8-gs128. Boot crashed with:
```
torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped
File "qwen3_next.py", line 408, in forward
    self.linear_attn(...)
File "torch/_library/custom_ops.py", line 152
    schema_str = torch.library.infer_schema(...)
```

Re-tested with v764d (Sander's untouched original config, no PN8/PN9, no Genesis modifications). Same crash. Confirmed **the issue is a vllm + torch.compile + qwen3_next linear_attn `@custom_op` interaction in the current pin** — not Genesis patches. Saved to `feedback_int8_gs128_torch_compile_crash.md`.

Workaround: use Lorbus INT4 (different code path, no crash). Will revisit when pin bumps past v0.20.2.

### Day 2 evening — 256K context on 27B Lorbus

The day's biggest empirical surprise. Original v771b PROD config:
```
--gpu-memory-utilization 0.95
--max-model-len 131072
--max-num-seqs 4
--max-num-batched-tokens 8192
```

OOMed at 16K context. We assumed this was a fundamental memory ceiling. It wasn't.

New config v791b:
```
--gpu-memory-utilization 0.90    # 0.95 → 0.90 (frees ~2.4 GB headroom)
--max-model-len 280000           # raised from 131072
--max-num-seqs 2                 # halved KV pool footprint
--max-num-batched-tokens 2048    # smaller chunked-prefill chunks
```

GPU memory drop: 22.69 GB (v771b) → 19.60 GB (v791b) = **3 GB freed for compile-time intermediate tensors.**

Progressive context probe (full results in `feedback_v791_27b_long_ctx_breakthrough.md`):

| Context | prompt_tokens | elapsed | prefill rate | Verdict |
|---|---:|---:|---:|---|
| 16K | 8 152 | 8.3 s | 1118 t/s | ✅ |
| 32K | 16 344 | 13.5 s | 1312 t/s | ✅ |
| 64K | 32 728 | 25.2 s | 1350 t/s | ✅ |
| 96K | 49 112 | 38.5 s | 1309 t/s | ✅ |
| 128K | 65 496 | 53.1 s | 1258 t/s | ✅ |
| 160K | 81 880 | 68.5 s | 1214 t/s | ✅ |
| 192K | 98 264 | 84.7 s | 1174 t/s | ✅ |
| **256K** | **131 032** | **120.7 s** | **1095 t/s** | **✅** |

Then: at util=0.90 the **TRUE 256K probe (262 104 actual prompt tokens)** also passed in 311 s (845 t/s prefill).

**The model wasn't the bottleneck. The config was.** Trade-off: ~5-10% lower peak TPS at small ctx in exchange for the full 256K window. Now shipped as the dedicated long-ctx variant — community users can pick `start_27b_int4_no_TQ_short.sh` for high-throughput chat or `start_27b_int4_no_TQ_long_256K.sh` for RAG / long-doc / agentic workloads.

### Day 2 night — TurboQuant k8v4 on hybrid 27B

vllm normally rejects `--kv-cache-dtype turboquant_k8v4` on hybrid GDN models with:
```
NotImplementedError: TurboQuant KV cache is not supported for hybrid
(attention + Mamba) models.
```

Genesis `P4` already removes this rejection (it's been in the tree since v7.0). What had been missing: when you actually exercise the TQ path on hybrid, the workspace manager (changed by the merged vllm#40941) hits an assertion:
```
AssertionError: Workspace is locked but allocation from
'turboquant_attn.py:1199:_decode_attention' requires 0.38 MB,
current size is 0.00 MB. Workspace growth is not allowed after locking.
```

Solution: enable `P98` (TQ WorkspaceManager revert, also opt-in). Combined `P4 + P98 + GENESIS_ENABLE_PN11_GDN_AB_CONTIGUOUS=1` makes TQ k8v4 work on Lorbus 27B-INT4.

Tool-call probe with `--max-tokens 1500` (the harness fix mentioned earlier):

| Case | Result |
|---|---|
| `think=false`, hermes parser, "weather in Paris" | `get_weather({"city":"Paris"})` ✅ |
| `think=true`, hermes parser, "Tokyo weather" | `get_weather({"city":"Tokyo"})` ✅ (with 85 reasoning tokens) |
| `think=false`, oai parser, "New York" | `get_weather({"city":"New York"})` ✅ |
| `think=true`, oai parser, "London with reasoning" | `get_weather({"city":"London"})` ✅ (265 reasoning tokens) |

**Tool-call 4/4 clean.** No `finish_reason="length"`, no garbage strings, no parser drift.

Throughput vs the no-TQ variant (Welch's t-test):

| | v791b (fp8_e5m2 KV) | v792 (TQ k8v4) | Welch p |
|---|---:|---:|---:|
| wall_TPS | 89.23 ± 8.87 | 90.49 ± 9.12 | 0.067 (NS) |
| decode TPOT | 11.18 ms | 11.01 ms | -1.9% (NS) |

Verdict: **+1.9% within statistical noise.** The dramatic +11.3% TQ bonus we measured on 35B-A3B-FP8 does NOT reproduce on Lorbus INT4 because Lorbus routes through `AllSparkLinearKernel` (not Marlin / compressed-tensors) — different memory traffic profile.

So for 27B Lorbus we ship **both** TQ and non-TQ variants, neither is strictly better. Saved to `feedback_v792_lorbus_TQ_k8v4_works.md`.

### Day 2 night — Quentin Machu's P64 fix

[Quentin Machu (@Quentin-M)](https://github.com/Quentin-M) opened a fork branch `fix/p64_indexerror` with a clean diagnosis and 40-line fix for an `IndexError` in `chat_completion_stream_generator`. Root cause:

> P64's widened `_should_check_for_unstreamed_tool_arg_tokens` returns True on `finish_reason` alone — to handle MTP/spec-decode finals where tool calls are in progress but the last delta carries no tool_calls chunk. However, the call site in `chat_completion_stream_generator` accesses `delta_message.tool_calls[0]` without checking that the list is non-empty. When the final delta has `tool_calls = []`, this crashes.

Fix: add `delta_message.tool_calls and` guard before `[0]` access. Cherry-picked into `vllm/_genesis/wiring/patch_64_qwen3coder_mtp_streaming.py` as sub-patch E. Excellent root-cause writeup + minimal correct fix — first community-contributed Genesis patch. Credit added to [CREDITS.md](CREDITS.md).

### Day 2 night — GPU profile recommendation system

A repeated theme during the day: a patch that's a clear win on one card class is irrelevant on another (P40 needs L2 ≥ 24 MB, etc). We codified this as a **per-GPU recommendation engine**:

`vllm/_genesis/gpu_profile.py` — static datasheet for 18 GPU classes (Ampere consumer + workstation, Ada Lovelace, Hopper, **all 5 Blackwell PRO workstation cards**), plus a per-patch predicate engine. At boot, Genesis prints `[REC]` / `[ON]` / `[OFF]` per patch based on the detected card:

```
[Genesis GPU profile] detected: NVIDIA RTX A5000
  canonical: RTX A5000  cc: (8, 6)  SM: 64  L2: 4 MB  BW: 768 GB/s  regime: bandwidth

  [OFF] P40    TQ k8v4 GQA grouping kernel  (correctly skipped — needs L2 ≥ 24 MB)
  [REC] P67    Multi-query verify kernel for spec-decode K+1
  [REC] P82    SGLang-style acceptance threshold OR-clause
  [REC] PN8    MTP/draft online-quant propagation
  [OFF] P83+P84+P85   (currently regressing on this stack)
```

Suggest-only — operator still must `export GENESIS_ENABLE_*=1`. See [Per-GPU recommendations](#per-gpu-recommendations) for the full table.

### Day 2 night — Genesis Benchmark Suite shipped

`tools/genesis_bench_suite.py` — single self-contained Python script (stdlib + requests, no scipy). 7-phase battery, configurable scope (`--quick` | `--mode standard` | `--mode full`), context window selectable (`--ctx 1K` through `--ctx 256K` or `all`). Output JSON + Markdown. Welch's t-test compare via `--compare A.json B.json`.

See [Genesis Benchmark Suite](#genesis-benchmark-suite) section below for full description. Also a dedicated 5-environment run guide at [`docs/BENCHMARK_GUIDE.md`](docs/BENCHMARK_GUIDE.md).

---

## Quick start

### Docker (recommended)

```bash
# 1. Set env paths
export MODELS_DIR=/path/to/your/models
export GENESIS_REPO=$HOME/genesis-vllm-patches
export HF_CACHE=$HOME/.cache/huggingface

# 2. Pick your config and launch
git clone https://github.com/Sandermage/genesis-vllm-patches "$GENESIS_REPO"
cd "$GENESIS_REPO"
./scripts/launch/start_35b_fp8_PROD.sh

# 3. Wait ~3-5 min for cold compile cache (subsequent boots ~1-2 min)
docker logs -f vllm-server-mtp-test

# 4. Health check
curl http://localhost:8000/v1/models -H "Authorization: Bearer genesis-local"

# 5. Run a quick benchmark
python3 tools/genesis_bench_suite.py --quick --host 127.0.0.1
```

### Bare metal (no Docker)

```bash
# Prereq
pip install vllm flashinfer-python

# Clone + launch
git clone https://github.com/Sandermage/genesis-vllm-patches ~/genesis-vllm-patches
export GENESIS_REPO=$HOME/genesis-vllm-patches
export MODEL_PATH=/path/to/Qwen3.6-35B-A3B-FP8

./scripts/launch/bare_metal_35b_fp8_PROD.sh
# (the script symlinks Genesis _genesis into the installed vllm package
#  on first run, then exec vllm serve ...)
```

For VM (Proxmox), WSL2, or RunPod see [`docs/BENCHMARK_GUIDE.md`](docs/BENCHMARK_GUIDE.md).

---

## Reference configs

Four PROD-ready configs ship in [`scripts/launch/`](scripts/launch/) — each in two flavors (Docker `start_*.sh` and bare-metal `bare_metal_*.sh`).

### 1. 35B-A3B-FP8 PROD (the daily driver)

```
Model:               Qwen3.6-35B-A3B-FP8
KV cache:            turboquant_k8v4
Spec-decode:         MTP K=3
Max context:         320 000
gpu-mem-util:        0.90
max-num-seqs:        2
max-num-batched-tk:  4 096
Tensor parallel:     2

Genesis env (PN8 + standard stack):
  GENESIS_ENABLE_P58/60/60b/61/61b/62/64/66/67/68/69/70/72/74=1
  GENESIS_ENABLE_P81=1, P82=1, P82_THRESHOLD_SINGLE=0.3
  GENESIS_ENABLE_P99=1, P101=1, P37=1
  GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT=1   ⭐ NEW
  GENESIS_ENABLE_PN11_GDN_AB_CONTIGUOUS=1       ⭐ defensive

Empirical:           wall_TPS 183.05, CV 8.92%, N=500 stress (200 min)
                     tool-call 3/4 (case 4 is max_tokens artifact, not regression)
                     GPU memory 22.66 GB after PN8 (-1 GiB vs v775)
                     128K context stable; 256K+ needs longer probe timeout
```

Files: [`scripts/launch/start_35b_fp8_PROD.sh`](scripts/launch/start_35b_fp8_PROD.sh) · [`bare_metal_35b_fp8_PROD.sh`](scripts/launch/bare_metal_35b_fp8_PROD.sh)

### 2. 27B-INT4-Lorbus short-ctx (high TPS for chat ≤8K)

```
Model:               Qwen3.6-27B-int4-AutoRound (community standard)
KV cache:            fp8_e5m2 (NOT TQ — Lorbus routes via AllSpark, no benefit)
Spec-decode:         MTP K=3
Max context:         131 072
gpu-mem-util:        0.95          ← aggressive — OK for ≤8K prompts
max-num-seqs:        4
max-num-batched-tk:  8 192
Prefix caching:      OFF           ← REQUIRED (DS conv state crash if ON)

Genesis env:
  GENESIS_ENABLE_P58/60/60b/61/61b/62/64/66/74=1
  GENESIS_ENABLE_P67=1, P83=1, P85=1, P87=1, P91=1
  GENESIS_ENABLE_P99=1, P100=1, P101=1
  GENESIS_ENABLE_PN11=1
  P82=0  (not yet swept on INT4)

Empirical:           wall_TPS 89.23, CV 9.97%, N=500 stress
                     tool-call 4/4 with max_tokens=1500
                     OOM at ≥16K prompts (config-aggressiveness, see #3)
```

Files: [`scripts/launch/start_27b_int4_no_TQ_short.sh`](scripts/launch/start_27b_int4_no_TQ_short.sh) · [`bare_metal_27b_int4_no_TQ_short.sh`](scripts/launch/bare_metal_27b_int4_no_TQ_short.sh)

### 3. 27B-INT4-Lorbus long-ctx 256K (RAG / agentic / long-doc)

```
Model:               Qwen3.6-27B-int4-AutoRound
KV cache:            fp8_e5m2
Spec-decode:         MTP K=3
Max context:         280 000       ← 192K and 256K both verified
gpu-mem-util:        0.90          ← reduced from 0.95 to free compile-time headroom
max-num-seqs:        2             ← halved (KV pool footprint)
max-num-batched-tk:  2 048         ← smaller chunked-prefill chunks

Same Genesis env as variant 2 plus PN8.

Empirical:           ~80 TPS at 1K (slightly lower than short-ctx variant)
                     128K stable: 53 s prefill, 1258 t/s
                     256K stable: 311 s prefill, 845 t/s
                     262 104 actual prompt tokens at TRUE 256K
                     GPU memory 19.60 GB after load (vs 22.69 GB short-ctx)
```

Files: [`scripts/launch/start_27b_int4_no_TQ_long_256K.sh`](scripts/launch/start_27b_int4_no_TQ_long_256K.sh) · [`bare_metal_27b_int4_no_TQ_long_256K.sh`](scripts/launch/bare_metal_27b_int4_no_TQ_long_256K.sh)

### 4. 27B-INT4-Lorbus + TurboQuant k8v4 (hybrid TQ)

```
Model:               Qwen3.6-27B-int4-AutoRound
KV cache:            turboquant_k8v4   ← unlocked via P4 + P98
Spec-decode:         MTP K=3
Max context:         280 000
gpu-mem-util:        0.90
max-num-seqs:        2
max-num-batched-tk:  2 048

Same Genesis env as variant 3 plus:
  GENESIS_ENABLE_P98=1   ⭐ REQUIRED — WorkspaceManager revert (vllm#40941 lock fix)

Empirical:           wall_TPS 90.49 (+1.9% NS vs no-TQ, Welch p=0.067)
                     tool-call 4/4 clean
                     256K capable (probe pending)
```

Files: [`scripts/launch/start_27b_int4_TQ_k8v4.sh`](scripts/launch/start_27b_int4_TQ_k8v4.sh) · [`bare_metal_27b_int4_TQ_k8v4.sh`](scripts/launch/bare_metal_27b_int4_TQ_k8v4.sh)

### Deferred — 27B-INT8-Minachist

Both Minachist `Qwen3.6-27B-INT8-AutoRound` (group_size=-1, AllSpark path) and `Qwen3.6-27B-INT8-gs128` (group_size=128, Marlin path) currently boot-crash on `torch._dynamo.exc.Unsupported: infer_schema` in `qwen3_next.py:408 self.linear_attn(...)`. **Crash is independent of Genesis patches** — reproduces with `v764d` minimal config. Wait for pin bump past v0.20.2 or upstream fix.

---

## Genesis Benchmark Suite

The flagship community-grade benchmark is **[`tools/genesis_bench_suite.py`](tools/genesis_bench_suite.py)** — a single self-contained Python script (816 LOC, stdlib + `requests` only, no scipy, no extra deps) that runs the full battery in one command.

### What it measures

Seven phases, configurable scope, JSON + Markdown output:

1. **Server discovery + GPU profile** — auto-detects card class via nvidia-smi, looks up bandwidth from the static datasheet
2. **Tool-call quality** — 8 cases (4 prompts × thinking on/off + 1 negative case to check the model can refuse)
3. **Decode-only TPOT bench** — N runs × M prompts × K decode tokens. Methodology adopted from [thc1006](https://github.com/thc1006)'s `bench_v3_clean_ab.py`. Strips TTFT + queue + scheduler from raw decode rate.
4. **Multi-turn TTFT** — 5 sequential same-prefix requests, smell test for cache benefit
5. **Stability stress** — configurable N iterations × M prompts. Tracks CV and any failures over a long run.
6. **Context window probe** — selectable max via `--ctx`: 1K / 4K / 8K / 16K / 32K / 64K / 128K / 256K / 512K / `all`. Stops at first OOM and reports the max stable.
7. **Output** — `<name>.json` (machine-readable, full per-trial data) + `<name>.md` (human summary)

### Run it

```bash
# Quick smoke test (~5 min)
python3 tools/genesis_bench_suite.py --quick

# Standard run (25 min) — bench + tool-call + multi-turn TTFT + ctx probe
python3 tools/genesis_bench_suite.py --mode standard --ctx 8K

# Full battery — pick the largest ctx your card can hold
python3 tools/genesis_bench_suite.py --mode full --ctx 256K

# Compare two arms (Welch's t-test on per-prompt decode TPOT)
python3 tools/genesis_bench_suite.py --compare run_A.json run_B.json
```

The default targets `localhost:8000` with API key `genesis-local`. Override with `--host`, `--port`, `--api-key`, `--model`. See `--help` for the full flag list.

### Sample output

```
========================================================================
Genesis Benchmark Suite — standard_2026-04-29T13-17-22Z
Mode: standard  ctx max: 8K  runs: 25  stress: 30
========================================================================

[0/7] Server discovery...
      reachable: True (HTTP 200)
      model: qwen3.6-35b-a3b
      local GPUs:
        GPU 0: NVIDIA RTX A5000  VRAM 22663/24564 MiB
        GPU 1: NVIDIA RTX A5000  VRAM 22663/24564 MiB

[1/7] Tool-call quality (8 cases)...
      4/7 positive cases passed (negative case scored separately)

[2/7] Decode bench (25 runs × 5 prompts × 1024)...
      wall_TPS = 184.471  CV 0.0873
      decode_TPOT_ms = 5.2953  CV 0.0824
      TTFT_ms = 159.50  CV 0.1382

[3/7] Multi-turn TTFT (5 turns)...
      turn 1: 171.3ms
      turn 2: 161.1ms
      turn 3: 155.5ms
      turn 4: 206.5ms
      turn 5: 164.5ms

[4/7] Stability stress (30 iters × 5 prompts)...
      duration 1217.4s  failures 0
      wall_TPS 184.81  CV 0.0899

[5/7] Context probe (max 8K)...
      max stable: 8K
========================================================================
```

### Run guide

For detailed step-by-step instructions covering bare metal, Docker, Proxmox VM, WSL2, and RunPod see **[`docs/BENCHMARK_GUIDE.md`](docs/BENCHMARK_GUIDE.md)**.

The guide also documents:

- How to interpret each metric (wall_TPS vs decode_TPOT_ms; CV; what "tool-call 3/4" actually means)
- Common failure modes (`ConnectionRefused` = vllm not yet booted; `FAIL_HTTP_500` at long ctx = OOM)
- Sharing protocol via [GitHub Discussions](https://github.com/Sandermage/genesis-vllm-patches/discussions) — the project is interested in community baselines on diverse hardware
- Privacy: bench does NOT phone home, NOT upload, NOT collect telemetry. All data stays local.

### Validation scripts (different purpose)

The bench suite measures **performance**. For **correctness validation** (apply matrix, smoke tests, pytest) use:

- [`validate_unit.sh`](validate_unit.sh) — CPU-only Python pytest in transient Docker container (~30 sec)
- [`validate_integration.sh`](validate_integration.sh) — GPU pytest + container health + chat-completion smoke + diagnostic probes
- [`scripts/run_validation_suite.sh <model_tag>`](scripts/run_validation_suite.sh) — universal per-model validation runner with new model tags (`qwen3_6_35b_fp8`, `qwen3_6_27b_int4_short`, `qwen3_6_27b_int4_long`, `qwen3_6_27b_int4_TQ`)

---

## Patch catalog

Genesis ships **65+ runtime patches** across categories. Each patch is opt-in via env var. See [`PATCHES.md`](PATCHES.md) for the canonical full reference; the table below highlights the most operator-facing ones.

### TurboQuant + KV cache (foundational)

| ID | Title | Default | Notes |
|---|---|:---:|---|
| P3 | TQ BF16→FP8 cast (Ampere fix) | ON | Required for FP8 on SM 8.6 |
| P4 | TQ hybrid model support | ON | Removes the hybrid rejection (Qwen3.6-27B etc) |
| P5 / P5b | KV cache page size unification | ON | LCM-pad for hybrid; v1 baseline |
| P6 | TQ-aware attention page size | ON | Aligns block size to TQ packed slot |
| P22 | TQ shared dequant prealloc | (auto-skipped) | PR #40655 merged upstream — auto-no-op |
| P36 | TQ shared decode buffers | (auto-skipped) | PR #40798 merged — auto-no-op |
| P38 | TQ continuation-prefill workspace | ON | Persistent K_full/V_full buffers |
| **P40** | **TQ k8v4 GQA grouping kernel** | **OFF** | **Recommend ON for 4090/5090/H100/Blackwell only** |
| P81 | FP8 block_scaled M ≤ 8 decode tune | ON | Targets MTP K=3 verify M=4 |
| **P98** | **TQ WorkspaceManager revert** | **conditional** | **REQUIRED for TQ on hybrid models** |
| P99 | WorkspaceManager.get_simultaneous memoize | ON | ~5× speedup per call, perf hotfix |
| P101 | TQ continuation 64-token slicing | ON | Selective backport of vllm#41123 |

### Spec-decode (MTP K=3)

| ID | Title | Default |
|---|---|:---:|
| P56/P57 | Spec-decode capture-safe buffers | OFF |
| P58 | Async-scheduler placeholder fix (vllm#40768) | ON |
| P63 | MTP GDN state recovery | OFF |
| P65 | TQ spec-decode CG downgrade | OFF |
| P66 | CUDA graph size divisibility filter | ON |
| P67 / P67b | Multi-query verify kernel for K+1 | ON |
| P70 | Auto-strict ngram (clean_rate ceiling fix) | ON |
| P77 | Adaptive ngram K controller (SGLang port) | OFF |
| P79b/c/d | Async × spec-decode race fixes | OFF |
| P82 | SGLang threshold_single OR-clause | ON (35B) |
| **PN8** | **MTP draft online-quant propagation** | **ON for FP8 + MTP** |
| **PN9** | **Independent drafter attention backend** | **(self-retired in pin)** |

### Hybrid GDN / Mamba

| ID | Title | Default |
|---|---|:---:|
| P7 / P7b | GDN dual-stream custom_op | ON |
| P28 | GDN core_attn_out prealloc | ON |
| P34 | Mamba zero-collapse deadlock guard | ON |
| P39 / P39a | FLA chunk_scaled_dot_kkt buffer pool | ON |
| P46 | GDN gating buffers | ON |
| P60 / P60b | GDN+ngram state recovery (vllm#40738) | ON |
| **PN11** | **GDN a/b contiguity (vllm#41142, by Quentin Machu)** | **ON (defensive)** |

### Tool-call + reasoning parsers

| ID | Title | Default |
|---|---|:---:|
| P12 | Tool-call reasoning extraction | ON |
| P15 | Qwen3 None-null normalization | ON |
| P27 | Reasoning-before-think parser fallback | ON |
| P59 | Qwen3 reasoning tool-call recovery (vllm#39055) | ON |
| P61 | Multi-tool first-occurrence (vllm#40783) | ON |
| P61b | Streaming overlap guard slice | ON |
| P62 | Reasoning-aware grammar (vllm#36138, sfbemerk) | ON |
| **P64** | **Streaming tool-call early-return + Quentin's IndexError fix** | **ON** |
| P68 / P69 | Auto force tool / long-ctx tool reminder | ON |
| P74 | Chunk clamp | ON |

### Marlin / quantization

| ID | Title | Default |
|---|---|:---:|
| P17/P18 | Marlin tuning | ON |
| P23 | Marlin FP32_REDUCE auto-disable on SM 8.6 | ON |
| P24 | MoE tuning | ON |
| P31 | Router softmax | ON |
| P37 | MoE intermediate cache pool | conditional |
| P87 | Marlin sub-tile pad-on-load (vllm#40361) | ON |
| P91 | AutoRound row-parallel cdiv (vllm#39460) | ON |
| P93 | AllSpark bypass for INT8 W8A16 | OFF |

### Disproven / not recommended

| ID | Why |
|---|---|
| P83 + P84 + P85 | Empirically -29% TPS regression with prefix-cache on current pin (this session) |
| P104 | -16.2% via L2 cache thrashing on 32+ layer transformer (prior session) |
| P105 | Variance noise on dequant kernel (prior session) |

For full per-patch documentation, opt-in env names, and credit lines see [PATCHES.md](PATCHES.md) and [CREDITS.md](CREDITS.md).

---

## Per-GPU recommendations

Genesis auto-detects your GPU at boot via [`vllm/_genesis/gpu_profile.py`](vllm/_genesis/gpu_profile.py) and prints `[REC]` (recommended) / `[OFF]` (not recommended) per patch in the apply log. The static datasheet covers 18 GPU classes:

| GPU class | Bandwidth | L2 | Regime | P40 | P67 | P82 | PN8 | P83/84/85 |
|---|---:|---:|---|:---:|:---:|:---:|:---:|:---:|
| RTX 3060 / 3070 / 3080 | 360-760 GB/s | 3-5 MB | bandwidth | OFF | REC | REC | REC | OFF |
| RTX 3090 | 936 GB/s | 6 MB | bandwidth | OFF | REC | REC | REC | OFF |
| RTX A4000 | 448 GB/s | 4 MB | bandwidth | OFF | REC | REC | REC | OFF |
| **RTX A5000 (Sander's PROD)** | **768 GB/s** | **4 MB** | **bandwidth** | **OFF** | **REC** | **REC** | **REC** | **OFF** |
| RTX A6000 | 768 GB/s | 6 MB | bandwidth | OFF | REC | REC | REC | OFF |
| RTX 4070 | 504 GB/s | 36 MB | mixed | REC | REC | REC | REC | OFF |
| RTX 4080 | 716 GB/s | 64 MB | mixed | REC | REC | REC | REC | OFF |
| RTX 4090 | 1008 GB/s | 72 MB | mixed | **REC** | REC | REC | REC | OFF |
| L40 / L40S | 864 GB/s | 96 MB | mixed | REC | REC | REC | REC | OFF |
| RTX 6000 Ada | 960 GB/s | 96 MB | mixed | REC | REC | REC | REC | OFF |
| RTX 5080 | 960 GB/s | 64 MB | mixed | REC | REC | REC | REC | OFF |
| **RTX 5090** | **1792 GB/s** | **88 MB** | **compute** | **REC** | REC | REC | REC | OFF |
| **RTX PRO 4000 Blackwell** (24 GB) | 672 GB/s | 24 MB | mixed | REC | REC | REC | REC | OFF |
| RTX PRO 4500 Blackwell (32 GB) | 896 GB/s | 32 MB | mixed | REC | REC | REC | REC | OFF |
| RTX PRO 5000 Blackwell (48 GB) | 1344 GB/s | 64 MB | compute | REC | REC | REC | REC | OFF |
| **RTX PRO 6000 Blackwell** (96 GB) | 1792 GB/s | 88 MB | compute | REC | REC | REC | REC | OFF |
| RTX PRO 6000 Blackwell Max-Q | 1792 GB/s | 88 MB | compute | REC | REC | REC | REC | OFF |
| A100 80GB | 2039 GB/s | 40 MB | compute | REC | REC | REC | REC | OFF |
| H100 / H200 | 3350-4800 GB/s | 50 MB | compute | REC | REC | REC | REC | OFF |
| B200 | 8000 GB/s | 80 MB+ | compute | REC | REC | REC | REC | OFF |

**Reading the table:** `REC` = empirically expected to deliver gain on this card class; `OFF` = either neutral or known-regressive. Auto-detection runs at every boot — operator pastes the `export GENESIS_ENABLE_*=1` lines from the printed recommendations into the launch script.

If your card isn't listed, Genesis falls back to "unknown regime — no recommendations" and you get the dispatcher's default behavior. Open a [GitHub Discussion](https://github.com/Sandermage/genesis-vllm-patches/discussions) with `nvidia-smi --query-gpu=name,memory.total --format=csv` output and we'll add it to the next release.

---

## Empirical findings — what worked, what didn't

Operators value honesty about both. From this session and prior:

### What worked (empirically validated, default ON or REC)

- **TurboQuant k8v4 KV cache** — measured **+11.3% TPS** on 35B-A3B-FP8 (Sander 2026-04-29 A/B vs `--kv-cache-dtype auto`). NOT just memory savings — the packed-slot layout helps L2 locality even on Ampere consumer.
- **MTP K=3 spec-decode** — accepts ~50-70% of speculative tokens depending on workload; net positive on every model class tested.
- **PN8 (online-quant draft propagation)** — saves ~1 GiB VRAM per GPU on FP8 + MTP. Throughput-neutral. **PROMOTE in any FP8 + MTP target.**
- **P82 (SGLang threshold_single OR-clause)** — cross-rig +12% on Sander's A5000 + FP8, +10.5% on noonghunna's 3090 + INT4. Two SM 8.6 datapoints, two quant paths.
- **P67 / P67b (multi-query verify kernel)** — +25-35% on spec-decode K+1 verify across all GPU classes tested.
- **256K context on 27B Lorbus via lighter config** — util 0.90 / max-num-seqs 2 / max-num-batched-tokens 2048. Was config-aggressiveness, not a model limit.
- **TurboQuant on hybrid GDN (P4 + P98)** — first time we got TQ k8v4 working on Lorbus 27B. Tool-call 4/4 clean.

### What didn't work (do NOT enable on Ampere consumer)

- **Prefix caching on 35B + MTP** — `--enable-prefix-caching` triggers a -29% TPS regression that **none of P83 / P84 / P85 mitigate**. Tested 4-arm A/B (this session). The supposed root-cause patch P84 didn't help. Conclusion: cache machinery overhead is real and the patches we have don't recover it.
- **P40 default-on for Ampere consumer (4 MB L2)** — fix landed (signature drift), kernel runs correctly, but +1.14% NS (Welch p=0.284). The +27% from upstream is real on Hopper-class L2 (50+ MB) but doesn't materialize when you're already memory-bandwidth-bound.
- **P104 (L2 persistence cache thrashing)** — -16.2% on 32+ layer transformer with KV >> L2. Each layer's pin evicts the previous. Architecture mismatch. Removed.
- **P105 (lmdeploy num_stages=3 hint on dequant kernel)** — variance noise (-1.4% retest after proper registration). Cross-engine kernel hints from prefill contexts don't apply to dequant utility kernels.
- **Triton 3.6 `disallow_acc_multi_buffer + loop_unroll_factor=2`** — -6.5% on small-BLOCK_M split-M kernel. Generic compiler hints don't transfer; always per-kernel A/B.
- **Aggressive `gpu-memory-utilization=0.95` + 4 streams + 8K batch on 27B** — OOMs at ≥16K context. Use 0.85-0.90 + 2 streams + 2K batch for long-ctx; the v791b config validates 256K stable.

### Special cases

- **Tool-call 3/4 on 35B** — case 4 (`think=true` + complex prompt + small `max_tokens`) is a **harness budget artifact, not a quality regression.** The model writes a long reasoning chain inside `<think>` and runs out of tokens before emitting the tool_call. Raise `max_tokens` from 300 to 1500 and you get 4/4. The bench suite uses 1024 by default; the test harness was upgraded to 1500.
- **TQ k8v4 on Lorbus 27B-INT4** — works (P4 + P98), tool-call 4/4 clean, but only +1.9% NS over fp8_e5m2. The +11.3% TQ bonus seen on 35B-A3B-FP8 doesn't reproduce because Lorbus routes through `AllSparkLinearKernel`, not Marlin / compressed-tensors. Different memory traffic profile. Both variants ship — no strict winner.

For the full chronological derivations see the memory notes referenced in each section above.

---

## Hardware and operating envelope

### Tested PROD environment

- 2× RTX A5000 24 GB (Ampere SM 8.6)
- Ubuntu 24.04, kernel 6.8
- NVIDIA driver `≥580.126.09` — **REQUIRED**, driver 570 puts PyTorch into a compatibility fallback ≈ 3× slower decode
- vLLM nightly pin `8cd174fa3` (image `vllm/vllm-openai:nightly`)
- PyTorch 2.11.0 + CUDA 13.0
- Triton 3.6.0, FlashInfer (MoE FP8 disabled per `VLLM_USE_FLASHINFER_MOE_FP8=0`)
- Genesis patch tree at the commit you cloned

### Cross-rig validated

- noonghunna's dual-3090 cluster (cross-rig +10.5% on P82, INT4)
- thc1006's 2×3090 cluster (provided the bench methodology + decode-only TPOT design)
- Quentin Machu's setup (P64 IndexError reproduced + fixed)

### Pin management

We hold pin at `8cd174fa3` deliberately. **Do NOT bump unless you've read [`docs/_internal/research_vllm_main_nightly_audit_20260429.md`](docs/_internal/) first** (gitignored, ask Sander) — there are several merged PRs in main that would break our patches simultaneously:

- vllm#40941 (TurboQuant share-dequant) — caused our prior 200→167 PROD regression, mitigated via P98/P99
- vllm#40860 (DSV4 mega-merge, +16k LOC) — breaks anchors for ~13 patches; ~2.5h of careful re-anchoring needed
- vllm#41184 (FusedMoE inversion, 108 files) — mass `intermediate_size_per_partition` → `intermediate_size` rename

The plan is to wait for v0.20.2 stable and do a single full TDD-gated re-anchor pass.

### Software you also need

- Docker 24+ with NVIDIA Container Toolkit (for the `start_*.sh` flow)
- Python 3.10+ + `requests` (for the bench suite)
- `gh` CLI (optional; for community sharing)
- `nsys` 2025.6+ (optional; for performance profiling — install via NVIDIA CUDA repo: `sudo apt install nsight-systems-2025.6.3` after adding `cuda-keyring`)

---

## How patches work

Genesis patches are **runtime modifications** — the engine loads stock vLLM, then `python3 -m vllm._genesis.patches.apply_all` modifies in-memory bytecode + monkey-patches Python objects to install the fixes. Three layers:

### 1. The dispatcher ([`vllm/_genesis/dispatcher.py`](vllm/_genesis/dispatcher.py))

Single source of truth — `PATCH_REGISTRY` dict with metadata for each patch (env flag, default state, applies-to predicate, credit line, upstream PR ref). Auto-evaluates per-patch predicates against detected model + GPU + spec-decode method, prints the apply matrix at boot.

### 2. The wiring layer ([`vllm/_genesis/wiring/`](vllm/_genesis/wiring/))

One file per patch (e.g., `patch_82_sglang_acceptance_threshold.py`). Each defines:
- `apply()` — the function `apply_all` calls
- `_make_patcher()` — for text-patches, defines anchor strings + replacement strings
- `UPSTREAM_DRIFT_MARKERS` — strings whose presence in upstream source triggers automatic self-retirement (when the upstream PR finally merges, our patch goes idempotent without operator intervention)

### 3. The GPU profile ([`vllm/_genesis/gpu_profile.py`](vllm/_genesis/gpu_profile.py))

18-card datasheet with bandwidth + L2 + SM count + compute capability + regime classification (bandwidth-bound / mixed / compute-bound). Per-patch predicates evaluate against the detected card. Prints `[REC]/[OFF]` recommendations at boot.

### Container R/W layer caveat

If you ran `docker compose stop` then `docker compose start`, the container's R/W layer **preserves the patched files** from the previous boot. Monolithic re-application can fail on anchor drift because the file is already mutated. Always use `docker compose down && docker compose up -d` (full container recreate) when re-applying patches. The text-patcher has idempotency markers, but the cleanest invariant is "fresh container per Genesis version".

---

## Acknowledgments

Genesis stands on the shoulders of the upstream vLLM project + the open-source community. This list captures the authors whose work directly powers Genesis. For the canonical full credit reference see [CREDITS.md](CREDITS.md).

### Community contributors (this release)

- **[Quentin Machu (@Quentin-M)](https://github.com/Quentin-M)** — first community-contributed Genesis patch. Diagnosed the IndexError in `chat_completion_stream_generator` final delta + landed the fix as P64 sub-patch E in [his fork](https://github.com/Quentin-M/genesis-vllm-patches/commit/09688b1d). Excellent root-cause writeup + minimal correct fix. Thank you Quentin.
- **[thc1006](https://github.com/thc1006)** — `bench_v3_clean_ab.py` decode-only TPOT methodology (the foundation of our bench suite). Also originated the prefix-cache OFF discovery (vllm#38182 adverse interaction).
- **[@noonghunna](https://github.com/noonghunna)** — community lead, two downstream forks (`qwen36-dual-3090`, `qwen36-27b-single-3090`), cross-rig validation on multiple rigs, contributed apples-to-apples baseline data for Sander's PRs.

### Upstream PR authors we backport

| Genesis patch | vllm PR | Author |
|---|---|---|
| P58 | [#40768](https://github.com/vllm-project/vllm/pull/40768) | z1ying |
| P59 | [#39055](https://github.com/vllm-project/vllm/pull/39055) | ZenoAFfectionate |
| P60, P60b | [#40738](https://github.com/vllm-project/vllm/pull/40738) | Thomas Parnell ([@tdoublep](https://github.com/tdoublep)) |
| P61, P61b | [#40783](https://github.com/vllm-project/vllm/pull/40783) | ExtReMLapin |
| P62 | [#36138](https://github.com/vllm-project/vllm/pull/36138) | sfbemerk |
| P64 | [#39598](https://github.com/vllm-project/vllm/pull/39598) | kotori-yan |
| P71 | [#40819](https://github.com/vllm-project/vllm/pull/40819) | Z. Golpayegani + gemini-code-assist (review) |
| P77 | SGLang `adaptive_spec_params.py` (Apache-2.0 port) | SGLang team + Nightjar paper authors |
| P81 | [#40925](https://github.com/vllm-project/vllm/pull/40925) | tonyliu312 |
| P82 | SGLang `speculative_sampling.cuh` ([SGLang](https://github.com/sgl-project/sglang)) | SGLang team |
| P86 | [#40876](https://github.com/vllm-project/vllm/pull/40876) | aaronagent |
| P87 | [#40361](https://github.com/vllm-project/vllm/pull/40361) | vLLM contributor |
| P91 | [#39460](https://github.com/vllm-project/vllm/pull/39460) | vLLM contributor |
| P94 | [#41043](https://github.com/vllm-project/vllm/pull/41043) | wangluochao902 |
| P100 | [#41127](https://github.com/vllm-project/vllm/pull/41127) | vLLM contributor |
| P101 | [#41123](https://github.com/vllm-project/vllm/pull/41123) (selective) | cderinbogaz |
| **PN8** | **[#40849](https://github.com/vllm-project/vllm/pull/40849)** | **[@bhoomit](https://github.com/bhoomit)** |
| **PN9** | **[#39930](https://github.com/vllm-project/vllm/pull/39930)** | **MatthewBonanni (merged upstream — auto-self-retires)** |
| **PN11** | **[#41142](https://github.com/vllm-project/vllm/pull/41142)** | **Yeuvoir (upstream PR) + [Quentin Machu](https://github.com/Quentin-M) (Genesis backport)** |

### Issue reporters whose investigation informed our patches

- **noonghunna** — vllm#40807 / #40831 (TQ + spec-decode + chunked-prefill `tolist()` crash; degenerate token loop)
- **SongXiaoMao** — vllm#40756 (MTP IMA on long sequences, sibling-bug)
- **bhaktatejas922** — vllm#39273 (GDN + ngram corruption original report)
- **cicirori (yinghui)** — vllm#34650 (MTP + reasoning + structured output `</think>` detection)
- **uOnePiece** + **@Angazenn** — vllm#38182 root-cause analysis (informed P83)
- **JartX** — vllm#39931 (broader hybrid TurboQuant work)

### vLLM core maintainers we depend on

The entire upstream vLLM team. Our patches are runtime modifications of their code; without their engine there is no Genesis. See [vLLM's CONTRIBUTORS](https://github.com/vllm-project/vllm/graphs/contributors).

### Special thanks

- Sander's homelab (2× RTX A5000 + Proxmox infra) for taking the pounding of 36+ hours of A/B testing without a single hardware crash
- The community that's reproduced findings cross-rig and reported back

---

## License / disclaimer

Apache-2.0 (matches vLLM upstream — see [LICENSE](LICENSE)).

Genesis is **NOT affiliated with or endorsed by** vLLM, NVIDIA, Alibaba, or any model author. It's a community downstream project. Patches are AS-IS — they work on the tested stack and are documented; no warranty implied.

Don't blindly enable everything. **Read the patch description, check the recommendation for your card, and bench in your environment** before promoting to production. The bench suite makes this fast — 25 minutes for a full standard run.

For commercial support / consulting / collaboration:

- **Author:** Sandermage (Sander) Barzov Aleksandr — Ukraine, Odessa
- **Repo:** https://github.com/Sandermage/genesis-vllm-patches
- **Discussions:** https://github.com/Sandermage/genesis-vllm-patches/discussions
- **License:** [Apache-2.0](LICENSE)

If you find Genesis useful and want to support continued development, see [SPONSORS.md](SPONSORS.md).

---

*Genesis vLLM Patches — empirical, attribution-rich, AS-IS. Built nights and weekends with the community.*
