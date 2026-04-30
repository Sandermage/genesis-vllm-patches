# Genesis Compatibility Matrix

What each opt-in patch does, which models it applies to, what's known to
conflict, and what to expect when you flip the env flag. Use this to
decide which patches to enable for YOUR workload.

> **TL;DR for operators:** every patch listed here is `default_on=False`
> (opt-in via env flag). The PROD launch scripts (`scripts/launch/`)
> already set the right combination for each model — you only need this
> table when customizing.

Conventions:

- 🟢 **SAFE** — empirically verified neutral or positive on the listed
  model. No tool-call regression, no stability issue.
- 🟡 **OPT-IN-ONLY** — patch is applied but its effect depends on
  workload (e.g. only fires on prefix-caching, only fires on a specific
  GQA shape, only fires on online quant).
- 🔴 **DO-NOT-ENABLE** — empirically broken on the listed model
  (regression confirmed in this repo's commit history).
- ⚠ **DEPRECATED** — superseded by another patch; kept for git history
  but should NOT be enabled.

---

## How patches interact

The dispatcher's `validate_registry()` call (runs at every boot) checks
for `requires_patches` and `conflicts_with` declarations and refuses to
boot if a violation exists. As of v7.63.x there are **0 declared
conflicts** between patches — they all compose additively. The
"conflicts" you'll see below are EMPIRICAL (observed at runtime), not
schema-declared.

The 50-entry registry splits into 15 categories. Within a category,
patches are typically **alternatives** (enable one, not all). Across
categories, patches are typically **additive**.

---

## spec_decode (22 patches)

The biggest category — anything that touches speculative-decoding
(MTP, ngram, suffix, eagle).

### Currently in PROD on 35B-A3B-FP8

| Patch | Status | Effect | Notes |
|---|---|---|---|
| **P58** Async-scheduler -1 placeholder fix | 🟢 SAFE | spec-decode + cudagraph workloads no longer loop or IMA | Backport vllm#40768 (z1ying), opt-in |
| **P60** GDN+ngram state recovery (Phase 1: SSM pre-copy) | 🟢 SAFE | tool-call recovery on hybrid models | Backport vllm#40738 |
| **P60b** GDN+ngram Triton kernel offset (Phase 2) | 🟢 SAFE | Composes with P60 | Backport vllm#40738 |
| **P66** cudagraph_capture_sizes divisibility filter | 🟢 SAFE | Boot 2-4× faster, less peak GPU memory | Genesis-original |
| **P67** TurboQuant multi-query kernel for spec-decode K+1 | 🟢 SAFE on 35B (GQA=8 = pow-2). 🔴 27B (GQA=6 non-pow-2) — guard auto-skips | Self-protected by Issue #7 power-of-2 guard | Genesis-original |
| **P67b** TurboQuant spec-verify forward() routing | 🟢 SAFE on 35B. Auto-skips on 27B same guard | Routes K+1 batches through P67 kernel before `_prefill_attention` | Genesis-original |
| **P82** SGLang threshold_single OR-clause acceptance | 🟢 SAFE on 35B. Biased — `GENESIS_P82_THRESHOLD_SINGLE=0.3` recommended | +5-10% acceptance rate on conservative draft scenarios | Backport SGLang |

### Currently in PROD on 27B Lorbus + fp8_e5m2 (v771b)

| Patch | Status | Effect | Notes |
|---|---|---|---|
| **P58** Async-scheduler -1 placeholder fix | 🟢 SAFE | same as 35B | Backport |
| **P60** + **P60b** GDN+ngram state recovery | 🟢 SAFE | same as 35B | Backport |
| **P66** cudagraph_capture_sizes filter | 🟢 SAFE | same | Genesis-original |
| **P67** + **P67b** TQ multi-query kernel | 🟡 OPT-IN-ONLY | Auto-skips 27B at runtime due to GQA=24/4=6 non-pow-2 (Issue #7 guard) — env flag is set but kernel doesn't fire | Genesis-original |

### Opt-in only on certain workloads

| Patch | When to enable | Why off by default |
|---|---|---|
| **P59** Qwen3 reasoning embedded tool_call recovery | Streaming clients only (LibreChat / OpenWebUI) | Overlaps with P62; dual-apply on same code is redundant |
| **P63** MTP/Eagle drafter GDN state recovery | Never (deprecated) | Hypothesis disproven 2026-04-26 |
| **P65** TurboQuant spec-decode cudagraph downgrade | Never on Ampere (superseded by P67b) | Forces PIECEWISE which costs ~5% TPS |
| **P70** Auto-strict-ngram (force `prompt_lookup_min>=8`) | ngram spec-decode workloads only | We use MTP, not ngram → no-op |
| **P71** Block-verify rejection sampler | Hopper / Blackwell experiments | Sampling change; needs retraining |
| **P75** Auto-enable Suffix Decoding | Arctic Inference workloads | Different acceptance heuristic |
| **P77** Adaptive ngram K controller | ngram only | We're on MTP |
| **P79b/P79c** Async × spec-decode proposer-sync | Async-scheduler workloads | OPEN PRs upstream — risk of upstream drift |
| **P83** + **P85** MTP keep-last-cached-block + hybrid fine-shadow prefix cache | If `--enable-prefix-caching` ON | PROD doesn't use prefix-caching (P83+P84+P85 stack regressed -29% in our 4-arm A/B) |
| **P86** ngram batch_propose O(N+K) | ngram only | Out of scope for MTP |
| **P94** Spec-decode prepare_next_token_ids_padded zero-alloc | Never (superseded by upstream merged #41043) | Auto-skips when upstream marker detected |
| **PN8** MTP/draft online-quant propagation | FP8 + MTP only | No-op on offline-quant INT4 (Lorbus) per `feedback_pn8_verified_vram_savings.md`. -1066 MiB VRAM on 35B FP8 |
| **PN9** Independent drafter attention backend | Never — already in upstream pin | Auto-skip (drift marker) |

### ⚠ Deprecated (do NOT enable)

| Patch | Why deprecated |
|---|---|
| **P56** TQ spec-decode safe-path guard | Superseded by P65 (which itself is now superseded by P67b) |
| **P57** TQ spec-decode capture-safe buffers | Research artifact; never reproduced positive effect |
| **P61** Qwen3 multi-tool first-occurrence | Superseded by P12 v2 (in `vllm/_genesis/wiring/legacy/`) |

---

## structured_output (7 patches)

Tool-call quality and reasoning parsing.

| Patch | Status | When | Effect |
|---|---|---|---|
| **P61b** Streaming partial-tag overlap guard | 🟢 SAFE on PROD | streaming tool-call | Stops `<tool_call` partial fragment from being closed prematurely |
| **P62** Structured-output spec-decode reasoning-end timing fix | 🟢 SAFE on PROD | spec-decode + grammar | Reasoning-aware grammar acceptance + spec-token validation |
| **P64** Qwen3coder MTP streaming early-return fix | 🟢 SAFE on PROD | streaming + MTP | Removes early `return` that drops parameters when MTP bundles last param + `</function>` |
| **P68** Auto force `tool_choice=required` for long-context tool calls | 🟢 SAFE on PROD | long-context (≥50K chars) | Threshold via `GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS` |
| **P69** Long-context tool-format reminder injection | 🟢 SAFE on PROD | long-context | Companion to P68 |

---

## perf_hotfix (4 patches)

Workspace / continuation perf tweaks.

| Patch | Status | Effect | Empirical (this repo) |
|---|---|---|---|
| **P98** TQ WorkspaceManager revert (vllm#40941 perf hotfix) | 🟡 OPT-IN-ONLY | "Expected +15-25% TPS recovery on Ampere small-batch single-stream" | **Tested 35B PROD 2026-04-30: +1.0% (within ±1% noise) — neutral on this workload** |
| **P99** WorkspaceManager.get_simultaneous memoization | 🟢 SAFE on 35B PROD (env=1) | "Per Sander 2026-04-28: 'if revert helps in test, this should help too'" | Already in PROD; effect baked into baseline |
| **P100** FlashInfer FULL CUDA graph for spec-decode (vllm#41127) | 🟢 SAFE on 35B PROD (env=1) | UNIFORM_BATCH cudagraph for K+1 spec-verify | NO-OP on TQ backend; only fires on FlashInfer backend |
| **P101** TQ continuation 64-token slicing (vllm#41123 SELECTIVE) | 🟢 SAFE on 35B PROD (env=1) | "+3-12% TPS on PROD long-context" | Already in PROD baseline |

---

## kv_cache (2 patches)

Prefix-cache / hybrid KV.

| Patch | Status | When | Notes |
|---|---|---|---|
| **P84** hash_block_size override (vllm#38182 actual root cause) | 🔴 DO NOT ENABLE on 27B PROD | Only with prefix-caching ON + GENESIS_P84_HASH_BLOCK_SIZE set | -29% TPS regression in 4-arm A/B (memory: `feedback_p83_p84_p85_cache_no_cake.md`) |
| **P85** Hybrid fine-shadow prefix cache (MambaManager fix) | 🔴 DO NOT ENABLE on 27B PROD | Companion to P84 | Same regression as above |

---

## kernel / kernel_perf / kernel_safety / quantization (4 patches)

| Patch | Status | When | Notes |
|---|---|---|---|
| **P81** fp8 block-scaled MM low-M decode tuning | 🟢 SAFE on 35B PROD (env=1) | FP8 block-scaled checkpoints (Minachist 35B) | Backport vllm#40925 |
| **P87** Marlin W4A16/W8A16 sub-tile output dim pad-on-load | 🟢 SAFE on 27B PROD (env=1) | INT4 AutoRound checkpoints | Backport vllm#40361 |
| **P91** AutoRound row-parallel group cdiv + start-idx fix | 🟢 SAFE on 27B PROD (env=1) | Row-parallel AutoRound INT4 | Backport vllm#39460 (CLOSED but valid) |
| **PN14** TQ decode IOOB safe_page_idx clamp | 🟡 OPT-IN-ONLY | Defensive on TQ k8v4 path | **Tested 35B PROD 2026-04-30: +0.1% (noise) — neutral** |

---

## compile_safety (3 patches)

Cudagraph capture / dynamo safety.

| Patch | Status | Effect | Empirical |
|---|---|---|---|
| **P72** profile_run M cap | 🟢 SAFE on 35B + 27B PROD (env=1) | Unblocks `--max-num-batched-tokens > 4096` on MoE by avoiding Dynamo fake-tensor shape mismatch | Already PROD |
| **P74** Auto chunk-clamp via long_prefill_token_threshold (P72 companion) | 🟢 SAFE on PROD (env=1) | Prevents prealloc buffer overflow when batched=8192 | Already PROD |
| **P78** TurboQuant `.tolist()` capture-guard | 🟡 OPT-IN-ONLY (Site B only) | Falls back to `flash_attn_varlen_func` during cudagraph capture | **Site A NOT applied** — substituting captured-time constant produces garbage output. See `feedback_27b_lorbus_compile_cache_regression_20260430.md` and v7.63.x P78 v3 commit |

---

## cudagraph_safety / model_correctness / memory_savings (5 patches)

| Patch | Status | When | Empirical |
|---|---|---|---|
| **PN11** GDN a/b contiguity (vllm#41142) | 🟢 SAFE on 35B PROD (env=1) | Hybrid GDN models — Quentin Machu fix | Defensive; community-validated |
| **PN12** FFN intermediate scratch pool (Cliff 1 fix) | 🟡 OPT-IN-ONLY | INT4 AutoRound 27B at long-ctx + tool-call (OOM mitigation) | **Tested 35B PROD 2026-04-30: +0.16% (noise) — neutral on this workload, but fires correctly** |
| **PN13** CUDAGraphWrapper `gc.collect`/`empty_cache` lambda arity (vllm#41235) | 🟡 OPT-IN-ONLY | Blackwell GB200 nightly | **Tested 35B PROD 2026-04-30: +0.7% (noise) — neutral on Ampere** |
| **PN17** FA2 softmax_lse runtime clamp (Issue #11 mech A) | 🟡 OPT-IN-ONLY | FA2 + spec-decode | Recent (2026-04-30); not yet PROD-tested |
| **PN19** Scoped `max_split_size_mb` during model load (vllm#41268) | 🟡 OPT-IN-ONLY | Cold-boot allocator hygiene | Recent; not yet PROD-tested |

---

## request_middleware / memory_hotfix / stability (3 patches)

| Patch | Status | When | Notes |
|---|---|---|---|
| **PN16** Lazy-reasoner request hook | 🟡 RESEARCH | Per-request `enable_thinking` decision (variant 5 prompt-engineering soft cap) | Phase 2 stub; Genesis-original |
| **P103** FLA Cliff 2 chunked fwd_h+fwd_o orchestrator | 🟡 OPT-IN-ONLY | Long-context > 50K on Lorbus 27B + DeltaNet GLA | Genesis-original (qwen36-27b-single-3090#1) |
| **P95** Marlin TP cudagraph cap on Ampere | 🟡 OPT-IN-ONLY | TP > 2 with Marlin kernels | Backport vllm#40385 (OPEN) |

---

## Empirical compatibility matrix (this repo, 2026-04-30)

| Patch | 35B-A3B-FP8 | 27B-Lorbus + fp8_e5m2 | 27B-Lorbus + TQ k8v4 | Notes |
|---|---|---|---|---|
| P58 | 🟢 PROD | 🟢 PROD | 🟢 PROD | Always safe |
| P60 + P60b | 🟢 PROD | 🟢 PROD | 🟢 PROD | Hybrid GDN — always |
| P66 | 🟢 PROD | 🟢 PROD | 🟢 PROD | Always safe |
| P67 + P67b | 🟢 fires (GQA=8) | 🟡 auto-skip (GQA=6) | 🟡 auto-skip (GQA=6) | Pow-2 guard |
| P67/P67b under FULL cudagraph | 🟢 PROD | n/a | 🔴 _prefill_attention .tolist() crash + repetition spam | Use PIECEWISE on 27B+TQ |
| P78 Site B | n/a (no .tolist() hit) | n/a | 🟢 boots, but doesn't fix the Site A crash | See P78 v3 commit |
| P82 (threshold=0.3) | 🟢 PROD | 🟡 OFF in v771b | 🟡 OFF | Workload-dependent acceptance |
| P98 | 🟡 +1% noise (35B) | not yet tested | required for boot on hybrid+TQ per memory | |
| P99 + P101 | 🟢 PROD | 🟡 OFF | n/a | Already optimized in PROD baseline |
| P83 + P84 + P85 | n/a (no prefix-cache) | n/a | n/a | -29% regression with prefix-cache ON |
| PN8 | 🟢 PROD (-1066 MiB) | 🟡 no-op (offline INT4) | 🟡 no-op (offline INT4) | Online-quant only |
| PN11 | 🟢 PROD | 🟡 OFF | 🟡 OFF | Hybrid defensive |
| PN12 | 🟡 +0.16% noise (35B) | not yet tested | not yet tested | Cliff 1 fix |
| PN13 | 🟡 +0.7% noise (35B) | not yet tested | not yet tested | Blackwell-targeted |
| PN14 | 🟡 +0.1% noise (35B) | not yet tested | not yet tested | TQ defensive |

Numbers in "noise" parentheses are from `tools/genesis_bench_suite.py
--mode standard --runs 25 --max-tokens 1024` runs on 2× RTX A5000
2026-04-30. CV is 5-6%, so anything under ±2% is sampling noise.

## How to add patches to your launch script

```bash
# v771b 27B PROD baseline already includes these — add MORE only if your
# workload needs them (and you want to opt into the empirical caveat
# above):

-e GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL=1 \
-e GENESIS_ENABLE_PN13_CUDA_GRAPH_LAMBDA_ARITY=1 \
-e GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP=1 \

# DO NOT enable these without reading the regression note:
# -e GENESIS_ENABLE_P83=1
# -e GENESIS_ENABLE_P84=1
# -e GENESIS_ENABLE_P85=1
# -e GENESIS_P84_HASH_BLOCK_SIZE=16
```

Inspect any patch via the unified CLI:

```bash
python3 -m vllm._genesis.compat.cli explain PN12
```

That prints the full record: applies_to predicate, lifecycle, upstream
PR, env flag, decision-today, credit text. Use it before flipping a
flag in production.
