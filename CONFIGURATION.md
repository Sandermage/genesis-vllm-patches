# Genesis vLLM Patches — Configuration Reference

Central reference for every environment variable that Genesis patches read.
Default behaviour is "off" / "safe" for opt-in patches; on-by-default
patches that are platform-gated (e.g. Ampere SM 8.0+) are noted.

> **Tested baseline (v7.59, 2026-04-28 — current PROD):**
>
> - vLLM `0.19.2rc1.dev212+g8cd174fa3` (image `vllm/vllm-openai:nightly`)
> - PyTorch 2.11.0+cu130, Triton 3.6.0, CUDA 13.0
> - **NVIDIA driver ≥ 580.126.09 REQUIRED** (570 → 3× slowdown)
> - 2× RTX A5000 (Ampere SM 8.6), TP=2
> - Qwen3.6-35B-A3B-FP8 + TurboQuant k8v4 + MTP K=3 + P67 multi-query kernel
> - **`--max-model-len 320000` (320K) + `--max-num-batched-tokens 4096`**
> - **220-317K context validated** (both think-ON + think-OFF modes)
> - **Stability + stress 30/30 + 30/30** (CV 6.7-6.8%)
> - Speed bench: 244 → 200 t/s (max_tokens 64 → 2048), GMU 0.90
> - **P67 safety gate** (v7.56): auto-disabled when no spec-decode in config
>
> Previous baseline (v7.52, 2026-04-27): max-model-len 262144 (256K),
> max-num-batched-tokens 8192. Same TPS class (CV practically identical).
> See `docs/reference/V759_320K_CONTEXT_EXPANSION_20260427.md` for full
> v759 vs v748 comparison + CV analysis.

---

## Table of contents

- [Production launch defaults (`scripts/launch/start_mtp.sh`)](#production-launch-defaults)
- [Patch enable / disable flags](#patch-enable--disable-flags)
- [Buffer-mode toggles (memory pool architecture)](#buffer-mode-toggles)
- [P67 multi-query kernel tuning](#p67-multi-query-kernel-tuning)
- [Diagnostic / observability](#diagnostic--observability)
- [PyTorch / CUDA / Triton standard env (recommended values)](#pytorch--cuda--triton-standard-env-recommended-values)
- [Rollback / debug overrides](#rollback--debug-overrides)

---

## Production launch defaults

The `scripts/launch/start_mtp.sh` script ships with a tested-on-prod set of
env vars. Each is described below. Override by exporting before invoking
the script, or edit the script directly for permanent changes.

| Concern | Default | Override env |
|---|---|---|
| GPU memory utilization | `0.90` | edit script `--gpu-memory-utilization` |
| Max context length | `262144` (256K) | edit script `--max-model-len` |
| Spec-decode method | `mtp` (K=3) | edit script `--speculative-config` |
| KV-cache dtype | `turboquant_k8v4` | edit script `--kv-cache-dtype` |
| TP size | `2` | edit script `--tensor-parallel-size` |
| Max num seqs | `2` | edit script `--max-num-seqs` |
| Max batched tokens | `8192` | edit script `--max-num-batched-tokens` |

---

## Patch enable / disable flags

All Genesis patches are opt-in via `GENESIS_ENABLE_<patch_id>=1`.
Production `start_mtp.sh` enables the validated set; opt-in patches stay off
unless explicitly engaged.

### On in production `start_mtp.sh`

| Env var | Patch | What it does |
|---|---|---|
| `GENESIS_ENABLE_P37=1` | P37 | MoE intermediate cache prealloc |
| `GENESIS_ENABLE_P58_ASYNC_PLACEHOLDER_FIX=1` | P58 | Async-scheduler `[-1]` placeholder fix (root cause for vllm#40831) |
| `GENESIS_ENABLE_P60_GDN_NGRAM_FIX=1` | P60 | GDN+ngram SSM state recovery (Phase 1, vllm#40738 backport) |
| `GENESIS_ENABLE_P60B_TRITON_KERNEL=1` | P60b | GDN+ngram conv state Triton kernel offset (Phase 2) |
| `GENESIS_ENABLE_P61_QWEN3_MULTI_TOOL=1` | P61 | Qwen3 multi-tool first-occurrence (vs LAST in upstream) |
| `GENESIS_ENABLE_P61B_STREAMING_OVERLAP=1` | P61b | Streaming partial-tag overlap guard (ExtReMLapin vllm#40783) |
| `GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING=1` | P62 | Reasoning-aware grammar acceptance + spec-token validation |
| `GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING=1` | P64 | qwen3coder streaming early-return fix (kotori-yan vllm#39598 backport) |
| `GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER=1` | P66 | cudagraph_capture_sizes spec-decode divisibility filter |
| `GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1` | P67 | TurboQuant multi-query Triton kernel for K+1 spec-verify (Genesis-original) |
| `GENESIS_P67_USE_UPSTREAM=1` | P67 | route to upstream `triton_turboquant_decode` instead of our v7.22 (drift-free) |
| `GENESIS_ENABLE_P68_AUTO_FORCE_TOOL=1` | P68 | Auto force tool_choice=required for long-ctx + tool calls |
| `GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER=1` | P69 | Long-context tool-format reminder injection |
| `GENESIS_ENABLE_P70_AUTO_STRICT_NGRAM=1` | P70 | Auto-strict-ngram (force prompt_lookup_min ≥ 8) |
| `GENESIS_ENABLE_P72_PROFILE_RUN_CAP=1` | P72 | profile_run M cap (unblocks `--max-num-batched-tokens > 4096`) |
| `GENESIS_ENABLE_P74_CHUNK_CLAMP=1` | P74 | Auto chunk-clamp via `long_prefill_token_threshold` (P72 companion) |
| `GENESIS_ENABLE_P81_FP8_BLOCK_SCALED_M_LE_8=1` | P81 | fp8 block-scaled MM low-M decode tuning (vllm#40925 backport, +23% per upstream) |

### Off by default (opt-in / experimental / deprecated)

| Env var | Patch | Note |
|---|---|---|
| `GENESIS_ENABLE_P56_SPEC_DECODE_GUARD` | P56 | Spec-decode safe-path guard. **Empirically deprecated**, kept for diagnostics only |
| `GENESIS_ENABLE_P57_SPEC_DECODE_CAPTURE_SAFE` | P57 | Capture-safe buffer expansion (experimental, fixes vllm#40831 root) |
| `GENESIS_ENABLE_P59_QWEN3_TOOL_RECOVERY` | P59 | Backport of vllm#39055. **Currently superseded by upstream PR #35687 in our pin — keep disabled** |
| `GENESIS_ENABLE_P63_MTP_GDN_STATE_RECOVERY` | P63 | **DEPRECATED** — wrong layer, hypothesis disproven. Kept for archival diagnostics only |
| `GENESIS_ENABLE_P65_TURBOQUANT_SPEC_CG_DOWNGRADE` | P65 | Cudagraph downgrade for spec-decode (workaround; replaced by P67/P67b) |
| `GENESIS_ENABLE_P71_BLOCK_VERIFY` | P71 | Block-verify rejection sampler (Sun 2024 ICLR + 2 critical bug-fixes from gemini bot review of vllm#40819). MTP-only |
| `GENESIS_ENABLE_P75_SUFFIX_DECODING` | P75 | Auto-enable Suffix Decoding (vllm#25784 Arctic Inference) |
| `GENESIS_ENABLE_P77_ADAPTIVE_NGRAM_K` | P77 | Adaptive ngram K controller (port of SGLang adaptive_spec_params.py + Nightjar arXiv 2512.22420 auto-disable) |
| `GENESIS_ENABLE_P78_TOLIST_CAPTURE_GUARD` | P78 | TurboQuant `.tolist()` capture-guard (adapted from @noonghunna's `patch_tolist_cudagraph.py`, Apache-2.0 attribution) |
| `GENESIS_ENABLE_P79B_ASYNC_PROPOSER_SYNC` | P79b | Async × spec-decode proposer-sync backport (vllm#40610, OPEN draft) |
| `GENESIS_ENABLE_P79C_STALE_SPEC_TOKEN_CLEANUP` | P79c | Stale spec_token_ids cleanup for unscheduled requests (vllm#37629, OPEN). v7.49 improvement: only clears `-1` placeholders, preserves real draft tokens |
| `GENESIS_ENABLE_P40` | P40 | TurboQuant grouped-decode Stage1 Triton kernel (vllm#40792 backport, +10-27% on Qwen3-32B GQA) |
| `GENESIS_ENABLE_P5B` | P5B | Page-size padded prealloc kernel (P5 follow-up — see `kernels/page_size_padded.py` history block) |
| `GENESIS_ENABLE_P7B` | P7B | GDN dual-stream `custom_op` variant (P7 follow-up — fuses two `in_proj_*` GEMMs) |
| `GENESIS_ENABLE_P41_RESPONSE_CACHE` | P41 | Response-level cache (above prefix-cache; full prompt → response). Memory or Redis backend. See P41 section below |
| `GENESIS_ENABLE_P78_TOLIST_CAPTURE_GUARD` | P78 | TurboQuant `.tolist()` cudagraph-capture guard (adapted from @noonghunna, Apache-2.0) |
| `GENESIS_ENABLE_P82` | P82 | SGLang per-token acceptance OR-clause (`speculative_sampling.cuh`). Opt-in; threshold via `GENESIS_P82_THRESHOLD_SINGLE` |
| `GENESIS_ENABLE_P83` | P83 | MTP keep-last-cached-block fix (force-pop disabled for hybrid models). Opt-in for hybrid (Qwen3-Next etc.) |
| `GENESIS_ENABLE_P84` | P84 | Override `hash_block_size` for hybrid prefix-cache (env `GENESIS_P84_HASH_BLOCK_SIZE`, defaults to layer block_size) |
| `GENESIS_ENABLE_P85` | P85 | Hybrid fine-shadow prefix cache (companion to P83/P84; opt-in for hybrid) |
| `GENESIS_ENABLE_P86` | P86 | Ngram batch propose linear scan (faster batch ngram proposer) |
| `GENESIS_ENABLE_P87` | P87 | Marlin sub-tile output-dim pad-on-load (vllm#40361 backport). v7.62.10 text-patch implementation |
| `GENESIS_ENABLE_P91` | P91 | AutoRound row-group cdiv quant dispatcher fix |
| `GENESIS_ENABLE_P94` | P94 | Spec-decode `prepare_next_token_ids_padded` zero-alloc (vllm#41043 backport, P99 TPOT -9.3% per author) |

> **P79d retired in v7.49** (vllm#38624 confirmed non-bug by njhill).
> **P22, P26, P28, P36, P38, P44, P46** are dispatcher-driven (always-on if platform supports).

---

## Buffer-mode toggles

Memory pool architecture — added v7.48 to control whether prealloc patches use shared singleton pool or legacy per-layer attached attributes.

| Env var | Default | Values | What it does |
|---|---|---|---|
| `GENESIS_BUFFER_MODE` | `shared` | `shared` / `per_layer` | Global mode for all prealloc patches |
| `GENESIS_BUFFER_MODE_<PID>` | (inherits global) | `shared` / `per_layer` | Per-patch override (e.g. `GENESIS_BUFFER_MODE_P38=per_layer`) |

`shared` = singleton pool via `GenesisPreallocBuffer` (memory-efficient, all 36 attention layers share one buffer).
`per_layer` = legacy attached-attribute path (rollback safety; recommended only if shared regresses on a specific model).

---

## P67 multi-query kernel tuning

| Env var | Default | What it does |
|---|---|---|
| `GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL` | `0` (off) | Enable P67 hook |
| `GENESIS_P67_USE_UPSTREAM` | `0` | Route through upstream `triton_turboquant_decode` instead of v7.22 (drift-free) |
| `GENESIS_P67_NUM_KV_SPLITS` | `32` | Number of KV-split partitions |
| `GENESIS_P67_BLOCK_KV` | `32` | KV tile width. Tested values 16/32/64 — 32 is optimum on A5000 (Step D sweep) |
| `GENESIS_P67_NUM_WARPS` | `8` (SM≥8.0) / `4` | Warps per CTA. 8 is optimum (Step D sweep — 4 regresses 4-5%) |
| `GENESIS_P67_NUM_STAGES` | `3` (SM≥8.0) / `2` | Pipeline depth. 3 is optimum on A5000 dequant-heavy kernel; 2 was -2 to -9% (Step E) |
| `GENESIS_P67_USE_FUSED` | `0` (off, opt-in) | **Experimental v7.52** — use fused-M kernel (BLOCK_M=K_PLUS_1*HEADS_PER_KV=32). REJECTED for prod default (-7% due to register spill on 64KB SM register file). Useful on A100/H100 or HEAD_DIM=64 models. |
| `GENESIS_P67_MAX_PRIOR_LEN` | `4096` | Max prior context len for P67 fast-path. **Baked at module load (v7.62.6 H2 fix).** Container-launch-time tunable only. |
| `GENESIS_P67_DEBUG_COMPARE` | `0` | Run reference CPU and assert match. **Baked at module load.** ~50× slower; use only for kernel debugging. |

---

## P75 Suffix Decoding tunables (opt-in via `GENESIS_ENABLE_P75_SUFFIX_DECODING=1`)

Activates upstream PR #25784 (Arctic Inference). All values pass through to vLLM's
`speculative_config`; defaults from PR's recommended profile.

| Env var | Default | What it does |
|---|---|---|
| `GENESIS_P75_TREE_DEPTH` | `24` | Suffix tree max depth |
| `GENESIS_P75_SPEC_FACTOR` | `2.0` | Max draft length factor (×K) |
| `GENESIS_P75_MIN_PROB` | `0.10` | Branch probability threshold (drop branches below this) |
| `GENESIS_P75_CACHE_REQS` | `10000` | Cross-request cache cap |

---

## P77 Adaptive Ngram-K Controller (opt-in via `GENESIS_ENABLE_P77_ADAPTIVE_NGRAM_K=1`)

Port of SGLang's `adaptive_spec_params.py` EMA + hysteresis logic + Nightjar
arXiv 2512.22420 auto-disable extension.

| Env var | Default | What it does |
|---|---|---|
| `GENESIS_P77_STEPS` | `0,1,3,5` | K-ladder steps the controller can pick from |
| `GENESIS_P77_EMA_ALPHA` | `0.2` | EMA smoothing factor for accept-rate |
| `GENESIS_P77_WARMUP_BATCHES` | `10` | Batches to observe before first decision |
| `GENESIS_P77_UPDATE_INTERVAL` | `5` | Batches between K decisions |
| `GENESIS_P77_HYSTERESIS_DOWN` | `0.25` | Drop K when accept-rate falls this much below threshold |
| `GENESIS_P77_HYSTERESIS_UP` | `0.0` | Raise K when accept-rate rises this much above threshold |
| `GENESIS_P77_DISABLE_THRESHOLD` | `0.30` | Auto-disable spec-decode entirely below this accept-rate (Nightjar) |
| `GENESIS_P77_PROBE_INTERVAL` | `100` | Batches between auto-disabled probes (re-test workload) |
| `GENESIS_P77_LOG_EVERY` | `20` | Log K decision every N batches |

---

## P82 SGLang Acceptance Threshold (opt-in via `GENESIS_ENABLE_P82=1`)

| Env var | Default | What it does |
|---|---|---|
| `GENESIS_P82_THRESHOLD_SINGLE` | (empty = disabled) | OR-clause threshold (`target_prob_single >= threshold_single`). Empirically tuned via prod sweep; biased rule, see `project_genesis_v7_53_p82_sglang_acceptance.md` |

---

## P41 Response Cache (opt-in via `GENESIS_ENABLE_P41_RESPONSE_CACHE=1`)

Response-level cache layered above vLLM's prefix-cache: full prompt → full
response, with TTL and weighted hit-rate metrics.

| Env var | Default | What it does |
|---|---|---|
| `GENESIS_P41_BACKEND` | `memory` | `memory` (in-process LRU) or `redis` |
| `GENESIS_P41_REDIS_URL` | (none) | e.g. `redis://192.168.1.10:6379/1` (required when backend=redis) |
| `GENESIS_P41_MAX_ENTRIES` | (impl default) | LRU cap for memory backend |
| `GENESIS_P41_TTL_SECONDS` | (impl default) | Expiry per cached entry |
| `GENESIS_P41_HIT_WEIGHTED` | `0` | Weight hit-rate metric by response length |
| `GENESIS_P41_HIT_ALPHA` | (impl default) | EMA alpha for weighted hit-rate |

---

## P83 / P85 debug knobs (opt-in via `GENESIS_ENABLE_P83=1` / `_P85=1`)

| Env var | Default | What it does |
|---|---|---|
| `GENESIS_P83_DEBUG` | `0` | Enable P83 debug log lines (per-decision MTP cached-block trace) |
| `GENESIS_P83_DEBUG_GCB` | `0` | Trace `get_computed_blocks` calls |
| `GENESIS_P83_DEBUG_HITS` | `0` | Trace cache hit decisions |
| `GENESIS_P83_DEBUG_STORE` | `0` | Trace block-store ops |
| `GENESIS_P85_DEBUG` | `0` | Enable P85 hybrid fine-shadow prefix-cache trace lines |
| `GENESIS_P84_HASH_BLOCK_SIZE` | (= layer `block_size`) | Override hash block size for hybrid prefix-cache |

---

## Memory / batched-token caps (kernel-side)

These cap kernel-side scratch buffers; useful when vLLM's
`--max-num-batched-tokens` differs from the kernel-baked default.

| Env var | Default | What it does |
|---|---|---|
| `GENESIS_TQ_MAX_BATCHED_TOKENS` | (= `--max-num-batched-tokens`) | Override TurboQuant dequant scratch sizing (kernels/dequant_buffer.py) |
| `GENESIS_GDN_MAX_BATCHED_TOKENS` | (= scheduler default) | Override GDN core-attn scratch sizing (kernels/gdn_core_attn_manager.py) |
| `GENESIS_MOE_MAX_BATCHED_TOKENS` | (= scheduler default) | Override MoE intermediate-cache sizing (kernels/moe_intermediate_cache.py) |
| `GENESIS_FLA_KKT_MAX_T` | (autodetect) | FLA KKT buffer max T-dim (patch_39_fla_kkt_buffer.py) |
| `GENESIS_FLA_KKT_MAX_B` | (autodetect) | FLA KKT buffer max B-dim (patch_39_fla_kkt_buffer.py) |

---

## Force / override / test infra

| Env var | Default | When to use |
|---|---|---|
| `GENESIS_DISABLE_P5` | `0` | Disable P5 page-size patch entirely (rollback) |
| `GENESIS_FORCE_APPLY_P36` | `0` | Force P36 to apply even if config_detect would skip (test only) |
| `GENESIS_FORCE_SPEC_DECODE` | (empty) | Force config_detect to report spec-decode active (test / pre-flight) |
| `GENESIS_FORCE_MARLIN_W8A16` | `0` | Force Marlin kernel for W8A16 (bypasses AllSpark dispatch). Set together with `VLLM_DISABLED_KERNELS=AllSparkLinearKernel`. P93 companion |
| `GENESIS_P71_USE_PYTORCH` | `0` | P71 block-verify: use PyTorch reference path instead of Triton kernel |
| `GENESIS_PROFILE_RUN_CAP_LOG` | `1` | P72: log when profile_run M is capped |
| `GENESIS_ENABLE_PERF_TESTS` | `0` | Run perf-benchmark tests (gated to keep CI fast) |
| `GENESIS_SKIP_PERF_TESTS` | `0` | Force-skip perf tests even when implicitly enabled |
| `GENESIS_VLLM_PIN_PATH` | (default file) | CI override for vLLM pin file location (test_v7_14_15_audit.py) |

---

## Diagnostic / observability

| Env var | Default | What it does |
|---|---|---|
| `GENESIS_DEBUG_INVARIANTS` | `0` | Enable assertions in patch hot paths (perf cost) |
| `VLLM_LOGGING_LEVEL` | `WARNING` (prod) | Set `INFO` to see Genesis dispatcher matrix per boot |
| `GENESIS_TQ_MAX_MODEL_LEN` | `262144` | Max model length for TQ prealloc sizing |
| `GENESIS_PREALLOC_TOKEN_BUDGET` | `4096` | Token budget for prefill output prealloc (P26) |
| `GENESIS_PROFILE_RUN_CAP_M` | `4096` | M cap for profile_run (P72) — unblocks `--max-num-batched-tokens > 4096` |
| `GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS` | `8000` | Char threshold for long-context tool-call hooks (P68/P69) |

---

## PyTorch / CUDA / Triton standard env (recommended values)

These are not Genesis env vars — they're vLLM / PyTorch / NCCL / Triton settings that interact with our patches in known ways. Production `start_mtp.sh` ships with the tested values.

| Env var | Recommended | Why |
|---|---|---|
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.6` | gc_threshold helps reclaim reserved-but-unallocated under high GMU |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | `1` | Profiler accounts for cudagraph buffers — important for P22/P26/P38 prealloc visibility |
| `VLLM_NO_USAGE_STATS` | `1` | Telemetry off |
| `VLLM_FLOAT32_MATMUL_PRECISION` | `high` | TF32 enabled for matmuls |
| `VLLM_USE_FLASHINFER_SAMPLER` | `1` | Faster sampling on Ampere |
| `VLLM_USE_FUSED_MOE_GROUPED_TOPK` | `1` | Fused MoE topk path |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | `1` | Marlin FP8 weight-only path Ampere optimization |
| `VLLM_MOE_USE_DEEP_GEMM` | `0` | DeepGEMM disabled (Hopper-only path; can break on Ampere) |
| `VLLM_USE_DEEP_GEMM` | `0` | Same as above |
| `VLLM_USE_FLASHINFER_MOE_FP8` | `0` | FlashInfer MoE FP8 disabled (Blackwell-only path) |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `1` | Allow `--max-model-len > model_max_position_embeddings` |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | Required for plugin-driven Genesis registration in TP workers |
| `NCCL_P2P_DISABLE` | `1` | A5000 has no NVLink; disable P2P to avoid NCCL probing overhead |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `8` | Prevents NCCL connection starvation on TP=2 |
| `OMP_NUM_THREADS` | `1` | Avoids OpenMP-vs-CUDA thread oversubscription |
| `TRITON_CACHE_DIR` | `/root/.triton/cache` (in container) | Persistent Triton compile cache for warm boots |

---

## Rollback / debug overrides

| Env var | When to use |
|---|---|
| `GENESIS_BUFFER_MODE=per_layer` | If shared buffer pool causes regression on a non-default model |
| `GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=0` | Disable P67 entirely (route through pure upstream attention) |
| `GENESIS_P67_USE_UPSTREAM=0` | Use Genesis v7.22 kernel instead of upstream |
| `GENESIS_P67_USE_FUSED=1` | A/B test fused-M kernel (default off, expected slower on A5000) |
| `VLLM_LOGGING_LEVEL=INFO` | Boot diagnostics: dispatcher apply matrix per patch |

For full revert paths, use git tags:

```bash
git checkout v7.52-stable-2026-04-27   # current production
git checkout v7.51-stable-2026-04-27   # pre-fused-experiment
git checkout v7.50-stable-2026-04-27   # pre-Step-D
```

Or use the server-side backup:

```bash
ls /home/sander/genesis-backups/
# v7.50-stable-20260427_0202/ contains RESTORE.md with step-by-step
```
