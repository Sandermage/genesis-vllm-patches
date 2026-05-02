# Genesis vLLM Patches — Complete Reference

This file is the **single source of truth** for every Genesis runtime patch.
For each patch you get: ID, title, what it does, status (ON / opt-in / deprecated),
env flag to toggle, upstream PR (if backported), and credit.

**Total PATCH_REGISTRY entries:** 99 (range P1–P103 + PN8–PN33 + sub-patches P5b/P7b/P15B/P18b/P38B/P39a/P67b/P67c/PN26b). The dispatcher's `PATCH_REGISTRY` is the schema-validated, lifecycle-tracked, opt-in surface — `genesis self-test` and the schema validator gate this set on every commit.

**Total apply_all `@register_patch`:** 98 entries. P68/P69 share one `apply_patch_68_long_ctx_tool_adherence` function but are registered as two `PATCH_REGISTRY` entries; that's the reason for the 1-entry delta. As of v7.65 (2026-05-02) all legacy P1–P46 patches are first-class registry entries with `lifecycle: legacy` — historical pre-dispatcher patches with minimal metadata, kept default-on for compatibility.

- **Source of truth:** `vllm/_genesis/dispatcher.py` `PATCH_REGISTRY` (range: P56-P103 + PN8-PN31, rich metadata) + `vllm/_genesis/patches/apply_all.py` `@register_patch` decorators (legacy P1-P55, dry-run diagnostic only).
- **All patches default OFF unless explicitly noted.** Production launch script enables a curated set via env flags.
- **Credits:** every backport names its upstream author + PR. Genesis-original patches are explicitly labelled. See [`CREDITS.md`](CREDITS.md) for the comprehensive attribution log.
- **Status legend:**
  - `default ON` — patch self-activates when its config gate passes
  - `opt-in` — requires `GENESIS_ENABLE_<patch>=1` env var
  - `deprecated` — superseded by another patch; kept for archeology, do not enable in new deployments
  - `library` — utility module loaded by other patches, no direct env flag

---

## How to enable / disable a patch

```bash
# Enable an opt-in patch:
docker run -e GENESIS_ENABLE_P82=1 -e GENESIS_P82_THRESHOLD_SINGLE=0.3 ... vllm/vllm-openai:nightly

# Disable a default-ON patch (rare — usually for A/B testing):
docker run -e GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=0 ... vllm/vllm-openai:nightly

# See full list of env vars + defaults:
# → CONFIGURATION.md
```

## Where the code lives

- **Wiring** (text-patcher hooks): `vllm/_genesis/wiring/<category>/patch_<id>_*.py` — organized by category (Phase 2.1, 9 subdirs: spec_decode, structured_output, perf_hotfix, compile_safety, kv_cache, kernels, hybrid, middleware, legacy). Resolution is layout-agnostic via `compat/categories.module_for(patch_id)`.
- **Kernels** (Triton / CUDA): `vllm/_genesis/kernels/`
- **Dispatcher metadata** (P56+): `vllm/_genesis/dispatcher.py:PATCH_REGISTRY`
- **Registration**: `vllm/_genesis/patches/apply_all.py:@register_patch`
- **Per-patch CHANGELOG entries**: `vllm/_genesis/CHANGELOG.md` (search by patch ID)

---

## Patches by category

### TurboQuant integration

| ID | Title | Status | Env Flag | Upstream | Credit |
|---|---|---|---|---|---|
| **P4** | TurboQuant hybrid model support | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P98** | TQ WorkspaceManager revert (vllm#40941 perf hotfix) | opt-in | `GENESIS_ENABLE_P98` | [#40941](https://github.com/vllm-project/vllm/pull/40941) | Reverts upstream PR #40941 WorkspaceManager indirection in turboquant_attn — required for TQ k8v4 on hybrid GDN models (else AssertionError on workspace lock) |
| **P101** | TQ continuation 64-token slicing (vllm#41123 SELECTIVE) | opt-in | `GENESIS_ENABLE_P101` | [#41123](https://github.com/vllm-project/vllm/pull/41123) | Selective backport of vllm#41123 TQ on hybrid models — takes the `_CONTINUATION_DECODE_SLICE` improvement only |
| **PN14** | TQ decode IOOB safe_page_idx clamp | opt-in | `GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP` | [#40074](https://github.com/vllm-project/vllm/pull/40074) | devarakondasrikanth @adobe (vllm#40074, OPEN) — fixes upstream issue: TQ decode page-index out-of-bounds clamp |

### Memory / buffers

| ID | Title | Status | Env Flag | Upstream | Credit |
|---|---|---|---|---|---|
| **P6** | TurboQuant-aware attention page size | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P5** | KV cache page size unification | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P22** | TurboQuant shared dequant prealloc | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P26** | TurboQuant prefill output prealloc | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P44** | TQ mixed-batch attn_out pool | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P46** | GDN gating buffer pool | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P39a** | FLA chunk_scaled_dot_kkt persistent A pool | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P38** | TQ _continuation_prefill persistent workspace | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P36** | TurboQuant shared decode buffers | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P32/P33** | TurboQuant cu_2 + synth_seq_lens preallocs | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P28** | GDN core_attn_out prealloc | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P14** | block_table tail zero-fill | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P20** | TurboQuant continuation-prefill FP16 rotate | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **PN12** | FFN intermediate scratch pool — Cliff 1 fix on TQ3 path | opt-in | `GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL` | [#34207](https://github.com/vllm-project/vllm/pull/34207) | Genesis-original 2026-04-29 — Cliff 1 fix on TQ3 path. Closes 138 MiB OOM at 192K context |
| **PN17** | FA2 softmax_lse runtime clamp (Cliff 1 mechanism A, Issue #11) | opt-in | `GENESIS_ENABLE_PN17_FA2_LSE_CLAMP` | — | Genesis-original 2026-04-30 in response to noonghunna's [Genesis Issue #11](https://github.com/Sandermage/genesis-vllm-patches/issues/11) cross-rig diagnosis (RTX 3090). Clamps `max_seqlen_k` from `attn_metadata.max_seq_len` (= max_model_len during cudagraph capture) to actual chunk max from `seqused_k.max()` at runtime. Closes Cliff 1 FA2 softmax_lse mechanism; widens long-text-no-vision safe envelope from ~150K to ~205K. Cudagraph-safe (capture-path falls back to original). Reference: Dao-AILab/flash-attention#1011. |
| **PN19** | Scoped max_split_size_mb during model load (vllm#41268) | opt-in | `GENESIS_ENABLE_PN19_SCOPED_MAX_SPLIT` | [#41268](https://github.com/vllm-project/vllm/pull/41268) | Backport of vllm#41268 (MatthewBonanni, OPEN). PyTorch 2.10+ introduced load-time allocator fragmentation; mitigates by temporarily setting `max_split_size_mb=20` (PyTorch minimum) during `Worker.load_model`, restores prior on exit. Cudagraph-safe (load-time only). Self-detects torch < 2.11 lacking `_accelerator_setAllocatorSettings` and falls through unchanged. Estimated 200-500 MiB headroom on H100; unverified on Ampere — operator should measure via nvidia-smi peak before relying on it. |
| **PN25** | SiluAndMul forward_native opaque-op pool — Cliff 1 mech B fix | opt-in | `GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE` | — | Genesis-original 2026-05-01 — closes Cliff 1 mech B (PN12 inductor inline bypass) on TQ3 + spec-decode + TP=1 configs. Registers `genesis::silu_and_mul_pooled` torch op as Inductor-opaque, ensuring FFN intermediate buffer pool fires even when Inductor inlines `forward_native`. Worker-spawn safe (issue #16 fix 2026-05-02). |
| **PN31** | FA varlen persistent out buffer (issue #15, sister to P38) | opt-in | `GENESIS_ENABLE_PN31_FA_VARLEN_PERSISTENT_OUT` | — | Genesis-original 2026-05-02 — closes #15 OOM at flash_attn_varlen_func on budget-constrained single-GPU. Per-shape persistent `out` buffer eliminates per-call malloc pressure inside FA C extension. Memory cost: ~16-64 MiB per shape × layer. NULL impact on 2×A5000 PROD (we have headroom); designed for 1×3090/1×4090 community users. |

### Kernel performance

| ID | Title | Status | Env Flag | Upstream | Credit |
|---|---|---|---|---|---|
| **P3** | TurboQuant BF16->FP8 cast (Ampere fix) | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P23** | Marlin FP32_REDUCE env override | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P31** | MoE router fp32 softmax | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P81** | fp8 block-scaled MM low-M decode tuning (vllm#40925) | opt-in | `GENESIS_ENABLE_P81_FP8_BLOCK_SCALED_M_LE_8` | [#40925](https://github.com/vllm-project/vllm/pull/40925) | Backport of vllm#40925 (tonyliu312, OPEN). Specializes w8a8_triton_block_scaled_mm default config for M<=8 (single-reque |
| **P87** | Marlin W4A16/W8A16 sub-tile output dim pad-on-load (vllm#40361) | opt-in | `GENESIS_ENABLE_P87` | [#40361](https://github.com/vllm-project/vllm/pull/40361) | Backport of vllm#40361 (OPEN). MarlinLinearKernel requires per-rank out_features divisible by GPTQ_MARLIN_MIN_THREAD_N=64. Sub-tile shards (Qwen3.5 GatedDeltaNet at TP>=2, Intel/Qwen3.6-35B-A3B-int4-AutoRound n=32 shard) fail can_implement and force a slow non-Marlin fallback. P87 zero-pads weights/scales/qzeros at load. Class-rebind on MarlinLinearKernel.{can_implement, process_weights_after_loading, apply_weights}. **NOTE 2026-04-28:** v764 boots with P87+P93 enabled trigger torch.dynamo "marked as skipped" error during cudagraph capture; deep dive deferred. Use at your own risk. |
| **P91** | AutoRound row-parallel group cdiv + start-idx fix (vllm#39460) | opt-in | `GENESIS_ENABLE_P91` | [#39460](https://github.com/vllm-project/vllm/pull/39460) | Backport of vllm#39460 non-MoE portion (CLOSED but valid). Fixes silent dequant corruption when AutoRound INT4/INT8 checkpoints have row-parallel layers whose input_size_per_partition is not divisible by group_size at TP>=2. Text-patch on gptq_marlin.py + parameter.py. Hypothesized cause of Lorbus INT4 < INT8 perf gap. |
| **P93** | AllSpark bypass for INT8 W8A16 group_size=-1 (force Marlin path) | opt-in | `GENESIS_FORCE_MARLIN_W8A16` | — | Genesis-original 2026-04-28. vLLM kernel selector picks AllSparkLinearKernel for AutoRound INT8 W8A16 group_size=-1 checkpoints (Minachist style); Marlin never fires, so P87/P91 are no-op. P93 prepends "AllSparkLinearKernel" to VLLM_DISABLED_KERNELS at plugin register() time, forcing fallback to MarlinLinearKernel. AllSpark on consumer Ampere is hardcoded for A100 (108 SMs, 40MB L2) — under-tuned on A5000 (64 SMs, 6MB L2). Functionally activates Marlin path; v764c boot crash blocks measurement until P87 wrapper is rewritten as text-patch. |
| **P72** | profile_run M cap (unblocks --max-num-batched-tokens>40 | opt-in | `GENESIS_ENABLE_P72_PROFILE_RUN_CAP` | — | Genesis-original (Dynamo fake-tensor mismatch workaround for moe_align_block_size symbolic shape) |
| **P37** | MoE intermediate cache pool (opt-in) | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P17/P18** | Marlin MoE per-SM tuning | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P1/P2** | FP8 kernel dispatcher | opt-in | — | — | Genesis (see source / CHANGELOG) |

### Spec-decode

| ID | Title | Status | Env Flag | Upstream | Credit |
|---|---|---|---|---|---|
| **P62** | structured-output spec-decode timing fix | opt-in | `GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING` | [#36138](https://github.com/vllm-project/vllm/pull/36138) | sfbemerk (vllm#36138), cicirori (vllm#34650) |
| **P60b** | GDN+ngram Triton kernel offset | opt-in | `GENESIS_ENABLE_P60B_TRITON_KERNEL` | [#40738](https://github.com/vllm-project/vllm/pull/40738) | tdoublep (vllm#40738) |
| **P60** | GDN+ngram state recovery | opt-in | `GENESIS_ENABLE_P60_GDN_NGRAM_FIX` | [#40738](https://github.com/vllm-project/vllm/pull/40738) | tdoublep (vllm#40738), bhaktatejas922 (#39273) |
| **P63** | MTP/Eagle drafter GDN state recovery | deprecated | `GENESIS_ENABLE_P63_MTP_GDN_STATE_RECOVERY` | — | Genesis-original (hypothesis disproven 2026-04-25) |
| **P64** | qwen3coder MTP streaming early-return fix | opt-in | `GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING` | [#39598](https://github.com/vllm-project/vllm/pull/39598) | kotori-yan (vllm#39598) |
| **P65** | TurboQuant spec-decode cudagraph downgrade | opt-in | `GENESIS_ENABLE_P65_TURBOQUANT_SPEC_CG_DOWNGRADE` | — | Genesis-original (root cause for noonghunna #40880) |
| **P66** | cudagraph_capture_sizes spec-decode divisibility filter | opt-in | `GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER` | [#23679](https://github.com/vllm-project/vllm/pull/23679) | Genesis-original (mirrors fhl2000 vllm#23679 closed) |
| **P70** | Auto-strict-ngram (force prompt_lookup_min>=8) | opt-in | `GENESIS_ENABLE_P70_AUTO_STRICT_NGRAM` | — | Genesis-original (vllm#40875 enforcement) |
| **P67** | TurboQuant multi-query kernel for spec-decode K+1 | opt-in | `GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL` | — | Genesis-original (proper fix for noonghunna #40880; replaces P65 workaround) |
| **P71** | Block-verify rejection sampler (vllm#40819 + gemini bug | opt-in | `GENESIS_ENABLE_P71_BLOCK_VERIFY` | [#40819](https://github.com/vllm-project/vllm/pull/40819) | Backport of vllm#40819 (Z. Golpayegani draft) + Sun et al. arXiv 2403.10444 + 2 critical fixes from gemini-code-assist r |
| **P77** | Adaptive ngram K controller (EMA + hysteresis + auto-di | opt-in | `GENESIS_ENABLE_P77_ADAPTIVE_NGRAM_K` | — | Genesis-original (port of SGLang adaptive_spec_params.py EMA+hysteresis Apache-2.0 + Nightjar arXiv 2512.22420 auto-disa |
| **P79b** | Async × spec-decode proposer-sync backport (vllm#40610) | opt-in | `GENESIS_ENABLE_P79B_ASYNC_PROPOSER_SYNC` | [#40610](https://github.com/vllm-project/vllm/pull/40610) | Backport of vllm#40610 (OPEN draft, tracked from #40608). Re-records prepare_inputs_event AFTER spec-decode proposer GPU |
| **P82** | SGLang threshold_single OR-clause acceptance (BIASED —  | opt-in | `GENESIS_ENABLE_P82` | — | SGLang team (sgl-project/sglang) speculative_sampling.cuh — port of the threshold_single OR-clause that breaks the struc |
| **P83** | MTP keep-last-cached-block (vllm#38182 downstream symptom) | opt-in (research) | `GENESIS_ENABLE_P83` | — | Genesis (root-cause analysis on vllm#38182 by uOnePiece + @Angazenn). Empirically DISPROVEN as actual cause for our workload; kept as research artifact. |
| **P84** | hash_block_size override (vllm#38182 actual root cause for hybrid) | opt-in (research) | `GENESIS_ENABLE_P84` + `GENESIS_P84_HASH_BLOCK_SIZE=16` | — | Genesis-original 2026-04-27. scheduler.py:234 + engine/core.py:209 dual-site override; lets prefix-cache hash at finer granularity than the LCM-padded hybrid block_size. |
| **P85** | Hybrid fine-shadow prefix cache (MambaManager fix) | opt-in (research) | `GENESIS_ENABLE_P85` | — | Genesis-original 2026-04-27. MambaManager scale-factor shadow-hash entries + eviction-safety verify. Requires P84 to give fine hashes. NOTE: bisect 2026-04-27 confirmed v756 sustained-load crash NOT caused by P85 (reproduced without it). |
| **P86** | ngram batch_propose O(N\*K) → O(N+K) direct-fill (vllm#40876) | opt-in | `GENESIS_ENABLE_P86` | [#40876](https://github.com/vllm-project/vllm/pull/40876) | Backport of vllm#40876 (aaronagent). Replaces O(N\*K) `i in valid_ngram_requests` membership scan with O(N+K) direct-fill loop. Algorithmic improvement, no behavioral change. |
| **P94** | Spec-decode prepare_next_token_ids_padded zero-alloc (vllm#41043) | opt-in | `GENESIS_ENABLE_P94` | [#41043](https://github.com/vllm-project/vllm/pull/41043) | Backport of vllm#41043 (wangluochao902, OPEN). Removes GPU→CPU `.tolist()` sync + list-comprehension + `np.array(...)` allocation in spec-decode hot path. PR author measured P99 TPOT -9.3% on Llama-3.1-8B + Eagle3 TP=4. Applies to all spec methods (Eagle, MTP, ngram, draft model). For our MTP K=3 single-stream: expected +2-4% wall TPS + tighter CV. |
| **P57** | TQ spec-decode capture-safe buffers | deprecated | `GENESIS_ENABLE_P57_SPEC_DECODE_CAPTURE_SAFE` | — | noonghunna (#40831), gdn_attn.py reference |
| **P56** | TQ spec-decode safe-path guard | deprecated | `GENESIS_ENABLE_P56_SPEC_DECODE_GUARD` | — | noonghunna (#40807, #40831) |
| **PN9** | Independent drafter attention backend (vllm#39930) | self-retired | — | [#39930](https://github.com/vllm-project/vllm/pull/39930) | MatthewBonanni (vllm#39930, MERGED). Self-retires when upstream merged — use `--speculative-config.attention_backend` instead. Removed from start scripts 2026-05-02. |
| **PN30** | DS conv state layout + spec-decode AL>1 fix (issue #17) | opt-in | `GENESIS_ENABLE_PN30_DS_LAYOUT_SPEC_DECODE` | — | Genesis-original 2026-05-02 (closes #17 noonghunna LCB v6 50/50 crash). Two-file text-patch: replaces `NotImplementedError` raise in `mamba_utils.py:get_conv_copy_spec` with `.contiguous()` copy + module-level temp-tensor list; adds stream sync + cleanup in `do_mamba_copy_block`. Cost ~10-50us per batch when path active. |

### Structured-output / Qwen3 parser

| ID | Title | Status | Env Flag | Upstream | Credit |
|---|---|---|---|---|---|
| **P15** | Qwen3 None/null tool arg parser | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P12** | Qwen3 <tool_call> implicit reasoning end | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P29** | tool parser IndexError guard | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P61b** | Qwen3 streaming partial-tag overlap guard | opt-in | `GENESIS_ENABLE_P61B_STREAMING_OVERLAP` | [#40783](https://github.com/vllm-project/vllm/pull/40783) | ExtReMLapin (vllm#40783) |
| **P61** | Qwen3 multi-tool first-occurrence | opt-in | `GENESIS_ENABLE_P61_QWEN3_MULTI_TOOL` | [#40783](https://github.com/vllm-project/vllm/pull/40783) | ExtReMLapin (vllm#40783) |
| **P68/P69** | long-context tool-call adherence | opt-in | `GENESIS_ENABLE_P68_AUTO_FORCE_TOOL` | — | Genesis-original (long-ctx tool adherence mitigation) |
| **P59** | Qwen3 reasoning embedded tool_call recovery | opt-in | `GENESIS_ENABLE_P59_QWEN3_TOOL_RECOVERY` | [#39055](https://github.com/vllm-project/vllm/pull/39055) | ZenoAFfectionate (vllm#39055) |
| **P40** | TurboQuant GQA-grouped decode stage1 (opt-in) | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P24** | fused_moe num_warps/num_stages overlay | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P18b** | TurboQuant decode stage1 tune | opt-in | — | — | Genesis (see source / CHANGELOG) |

### Hybrid / GDN / Mamba

| ID | Title | Status | Env Flag | Upstream | Credit |
|---|---|---|---|---|---|
| **P34** | Mamba zero-collapse deadlock guard | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P7b** | GDN dual-stream via torch.library.custom_op (opt-in) | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P7** | GDN dual-stream in_proj parallelism | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **PN11** | GDN a/b contiguity in fix_query_key_value_ordering (vllm#41142) | opt-in | `GENESIS_ENABLE_PN11_GDN_AB_CONTIGUOUS` | [#41142](https://github.com/vllm-project/vllm/pull/41142) | Yeuvoir (vllm#41142, OPEN). Fixes upstream issue #41112: in `GatedDeltaNet.fix_query_key_value_ordering` the `a` and `b` slices need explicit `.contiguous()` calls |

### Cudagraph

| ID | Title | Status | Env Flag | Upstream | Credit |
|---|---|---|---|---|---|
| **P78** | TurboQuant .tolist() capture-guard (adapted from noongh | opt-in | `GENESIS_ENABLE_P78_TOLIST_CAPTURE_GUARD` | — | Adapted from noonghunna's patch_tolist_cudagraph.py (Apache-2.0, github.com/noonghunna/qwen36-27b-single-3090). Surgical |
| **P67b** | TurboQuant spec-verify forward() routing (FULL CG enabl | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P95** | Marlin TP cudagraph cap on Ampere (vllm#40385) | opt-in | `GENESIS_ENABLE_P95` | [#40385](https://github.com/vllm-project/vllm/pull/40385) | Backport of vllm#40385 (OPEN as of 2026-04-28). Defensive cap of `max_cudagraph_capture_sizes` to avoid OOM on TP>=2 with Marlin kernels |
| **P100** | FlashInfer FULL CUDA graph for spec-decode (vllm#41127) | opt-in | `GENESIS_ENABLE_P100` | [#41127](https://github.com/vllm-project/vllm/pull/41127) | Backport of vllm#41127 (open 2026-04-28) — enables FlashInfer FULL cudagraph mode for spec-decode |
| **PN13** | CUDAGraphWrapper gc.collect/empty_cache lambda arity (vllm#41235) | opt-in | `GENESIS_ENABLE_PN13_CUDA_GRAPH_LAMBDA_ARITY` | [#41235](https://github.com/vllm-project/vllm/pull/41235) | roikoren755 (vllm#41235, OPEN). Fixes worker-death TypeError in CUDAGraphWrapper |

### Scheduler / chunked-prefill

| ID | Title | Status | Env Flag | Upstream | Credit |
|---|---|---|---|---|---|
| **P74** | Auto chunk-clamp via long_prefill_token_threshold (P72  | opt-in | `GENESIS_ENABLE_P74_CHUNK_CLAMP` | — | Genesis-original (zero-VRAM-cost prealloc-overflow safety net for P72-unblocked batched_tokens>4096) |
| **P58** | async-scheduler -1 placeholder fix | opt-in | `GENESIS_ENABLE_P58_ASYNC_PLACEHOLDER_FIX` | [#40768](https://github.com/vllm-project/vllm/pull/40768) | z1ying (vllm#40768) |

### Other

| ID | Title | Status | Env Flag | Upstream | Credit |
|---|---|---|---|---|---|
| **P8** | KV hybrid reporting (per-token capacity) | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P27** | Qwen3 BEFORE-THINK fallback | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P5b** | KV page-size pad-smaller-to-max (env-opt-in) | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P79c** | Stale spec_token_ids cleanup for unscheduled requests ( | opt-in | `GENESIS_ENABLE_P79C_STALE_SPEC_TOKEN_CLEANUP` | [#37629](https://github.com/vllm-project/vllm/pull/37629) | Backport of vllm#37629 (OPEN, fixes #36906). Cleanup pass after main scheduling loop clears spec_token_ids for unschedul |
| **P75** | Auto-enable Suffix Decoding (vllm#25784 Arctic Inferenc | opt-in | `GENESIS_ENABLE_P75_SUFFIX_DECODING` | [#25784](https://github.com/vllm-project/vllm/pull/25784) | Backport-enabler of vllm#25784 (Arctic Inference Suffix Decoding) — operator convenience: auto-swap method=ngram→suffix  |

---

## Category descriptions

### TurboQuant integration

Genesis uses Beidi Chen's [TurboQuant](https://github.com/Infini-AI-Lab/turboquant) k8v4 KV cache (FP8 K + 4-bit V) on Ampere. These patches add Ampere-specific casts, page-size unification, hybrid model support, and decode/prefill workspace management for the TurboQuant attention path.

### Memory / buffers

Pre-allocated singleton buffers shared across all 36 attention layers via `GenesisPreallocBuffer.get_or_create()`. Eliminates per-call `torch.empty` allocations on the hot path. Toggleable via `GENESIS_BUFFER_MODE=shared|per_layer`.

### Kernel performance

Triton / CUDA kernel optimizations: Marlin FP8 reduce, MoE warps/stages tuning, fp8 block-scaled MM low-M decode (P81), and the flagship **P67 multi-query kernel** for spec-decode K+1 verify (Genesis-original, +32% TPS).

### Spec-decode

Speculative decoding fixes and acceptance heuristics: GDN+ngram state recovery (P60/P60b backport), block-verify rejection sampler (P71, Sun 2024), MTP cudagraph fixes, async-scheduler placeholder, and **P82 SGLang threshold_single OR-clause** (production: +12% TPS at threshold=0.3).

### Structured-output / Qwen3 parser

Qwen3 thinking + tool-call output handling: reasoning-end timing (P62), multi-tool first-occurrence (P61), streaming overlap (P61b), embedded tool_call recovery (P59), MTP streaming early-return (P64), long-context tool-format reminder (P68/P69).

### Hybrid / GDN / Mamba

Patches for hybrid attention models (GDN, Mamba): dual-stream parallelism (P7/P7b/P28), gating buffers (P46), zero-collapse deadlock guard (P34), FLA chunk_scaled_dot_kkt persistent A pool (P39a).

### Cudagraph

Cudagraph capture safety: spec-decode divisibility filter (P66), TurboQuant CG downgrade (P65), .tolist() capture-guard (P78).

### Scheduler / chunked-prefill

Scheduler-level fixes for long-context decode + MoE: `profile_run` M cap (P72, unblocks `--max-num-batched-tokens > 4096`), auto chunk-clamp via `long_prefill_token_threshold` (P74).

### Other

Misc fixes: KV cache page size unification (P5/P5b), block_table tail zero-fill (P14), Marlin FP32_REDUCE env override (P23).

### Community-issue fixes (v7.65, 2026-05-02)

Patches addressing reported community bugs. Mix of default-OFF (cross-rig
validation pending) and default-ON (root-cause correctness fixes).

| ID | Title | Issue | Default | Env flag |
|---|---|---|---|---|
| P15B | FA varlen `max_seqlen_k` clamp on TQ path | #15 | OFF | `GENESIS_ENABLE_P15B_FA_VARLEN_CLAMP` |
| P38B | P38 compile-safe in-source hook (aot_compile-safe) | #14 | OFF | `GENESIS_ENABLE_P38B_COMPILE_SAFE_HOOK` |
| PN30 | DS conv state layout + spec-decode AL>1 | #17 | OFF | `GENESIS_ENABLE_PN30_DS_LAYOUT` |
| PN31 | FA varlen persistent `out` buffer (sister to P38) | #15 | OFF | `GENESIS_ENABLE_PN31_FA_VARLEN_OUT` |
| PN32 | GDN chunked-prefill (Cliff 2 single-24GB-GPU OOM) | noonghunna | OFF | `GENESIS_ENABLE_PN32_GDN_CHUNKED_PREFILL` |
| PN33 | Spec-decode warmup K-aware (vllm#37521 extended to MTP/ngram) | ampersandru/noonghunna | **ON** | `GENESIS_ENABLE_PN33_SPEC_DECODE_WARMUP_K` (disable: `GENESIS_DISABLE_PN33_SPEC_DECODE_WARMUP_K=1`) |

**PN33 is default-ON** because it's a root-cause correctness fix, not
experimental. The vanilla `_dummy_sampler_run` warms the rejection
sampler with K=1 dummy draft tokens regardless of real
`num_speculative_tokens`. This causes (a) under-counted KV-cache
profile → mid-stream OOM via `propose_draft_token_ids` (ampersandru on
club-3090#16), and (b) TurboQuant `WorkspaceManager` lock fails when
real K-token spec-decode tries to grow workspace beyond warmup-reserved
size (noonghunna on club-3090 disc #19). Both share one root cause;
PN33 fixes the root, both symptoms disappear. Genesis extends the
upstream PR (vllm-project/vllm#37521 by `itailang`) beyond its
`use_eagle()` gate to cover all spec-decode methods (EAGLE + MTP +
ngram + draft-model) since the rejection sampler memory footprint
scales with K regardless of method. Disable only if K-sized warmup
itself OOMs on a tight rig.

### Reasoning + middleware

- **PN16** — Lazy-reasoner request hook (per-request `enable_thinking`). Middleware variant; configured via `GENESIS_ENABLE_PN16_LAZY_REASONER`.

### DFlash spec-decode (PN21–PN24)

DFlash combine_hidden_states + SWA + aux-layer indexing fixes for spec-decode + DFlash hybrid:

- **PN21** — DFlash SWA support partial backport (vllm#40898). Env: `GENESIS_ENABLE_PN21_DFLASH_SWA`.
- **PN22** — Local argmax for TP draft (vllm#39419). Env: `GENESIS_ENABLE_PN22_TP_LOCAL_ARGMAX`.
- **PN23** — DFlash combine_hidden_states dtype cast (vllm#40334). Env: `GENESIS_ENABLE_PN23_DFLASH_DTYPE`.
- **PN24** — DFlash aux layer +1 indexing fix (vllm#40727). Env: `GENESIS_ENABLE_PN24_DFLASH_AUX_INDEX`.

### TurboQuant unified pack (PN26 / PN26b)

- **PN26** — TQ unified perf pack (centroids prebake + sparse-V scaffold). Genesis-original.
- **PN26b** — Sparse-V tile-skip kernel (BLASST λ=a/L for SM86; first NVIDIA Ampere implementation). Genesis-original Triton kernel.

### MoE / merge_attn_states / chunk_o numerics

- **PN27** — Revert MoERunnerInterface PluggableLayer (vllm#41440 backport).
- **PN28** — `merge_attn_states` NaN guard (vllm#39148 backport).
- **PN29** — GDN `chunk_o` scale-fold (vllm#41446 pattern (c) backport). Default OFF; empirical −0.4% on 27B (Welch p=0.008) — kept opt-in for future Blackwell.

---

## Adding a new patch

1. **Pick a free ID.** Run `grep -E '^@register_patch' vllm/_genesis/patches/apply_all.py | head` and `grep -E '"P[0-9]+' vllm/_genesis/dispatcher.py` to confirm the next available number. Don't reuse retired IDs (P56/P57/P63 are deprecated but kept).
2. **Write wiring**: `vllm/_genesis/wiring/<category>/patch_<id>_<name>.py`. Use [`patch_71_block_verify.py`](vllm/_genesis/wiring/spec_decode/patch_71_block_verify.py) or [`patch_82_sglang_acceptance_threshold.py`](vllm/_genesis/wiring/spec_decode/patch_82_sglang_acceptance_threshold.py) as templates. The category should match a `compat/categories.py` bucket — `_build_module_index` will rglob the new file in automatically.
3. **Register in dispatcher** (P56+): add an entry to `PATCH_REGISTRY` in [`vllm/_genesis/dispatcher.py`](vllm/_genesis/dispatcher.py).
4. **Hook in apply_all**: add `@register_patch(...)` + `apply_patch_<id>_*` function in [`vllm/_genesis/patches/apply_all.py`](vllm/_genesis/patches/apply_all.py).
5. **Document in CHANGELOG**: add a `vX.YZ` entry to [`vllm/_genesis/CHANGELOG.md`](vllm/_genesis/CHANGELOG.md) explaining the WHY, empirical data, and ship/reject decision.
6. **Validate**:
   - Static: `python3 -c 'import ast; ast.parse(open("vllm/_genesis/wiring/<category>/patch_<id>_*.py").read())'`
   - Container: `docker compose down && docker compose up -d` (NOT `stop/start` — see [`CONFIGURATION.md`](CONFIGURATION.md) Container R/W layer note)
   - Empirical: blue/green sweep with `genesis_quality_harness.py` + `genesis_bench_v3.py`. SHIP gate: ≥30/31 quality + ≥+5% TPS (or whatever the patch targets).
7. **Credit upstream** in the patch docstring + `CREDITS.md` if backporting from someone else's PR / project.

