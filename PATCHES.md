# Genesis vLLM Patches — Complete Reference

This file is the **single source of truth** for every Genesis runtime patch.
For each patch you get: ID, title, what it does, status (ON / opt-in / deprecated),
env flag to toggle, upstream PR (if backported), and credit.

**Total registered patches:** 59 (across 52 wiring files; some hooks share a wiring module).

- **Source of truth:** `vllm/_genesis/dispatcher.py` `PATCH_REGISTRY` (P56-P82, rich metadata) + `vllm/_genesis/patches/apply_all.py` `@register_patch` decorators (legacy P1-P55).
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

- **Wiring** (text-patcher hooks): `vllm/_genesis/wiring/patch_<id>_*.py`
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

### Kernel performance

| ID | Title | Status | Env Flag | Upstream | Credit |
|---|---|---|---|---|---|
| **P3** | TurboQuant BF16->FP8 cast (Ampere fix) | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P23** | Marlin FP32_REDUCE env override | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P31** | MoE router fp32 softmax | opt-in | — | — | Genesis (see source / CHANGELOG) |
| **P81** | fp8 block-scaled MM low-M decode tuning (vllm#40925) | opt-in | `GENESIS_ENABLE_P81_FP8_BLOCK_SCALED_M_LE_8` | [#40925](https://github.com/vllm-project/vllm/pull/40925) | Backport of vllm#40925 (tonyliu312, OPEN). Specializes w8a8_triton_block_scaled_mm default config for M<=8 (single-reque |
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
| **P57** | TQ spec-decode capture-safe buffers | deprecated | `GENESIS_ENABLE_P57_SPEC_DECODE_CAPTURE_SAFE` | — | noonghunna (#40831), gdn_attn.py reference |
| **P56** | TQ spec-decode safe-path guard | deprecated | `GENESIS_ENABLE_P56_SPEC_DECODE_GUARD` | — | noonghunna (#40807, #40831) |

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

### Cudagraph

| ID | Title | Status | Env Flag | Upstream | Credit |
|---|---|---|---|---|---|
| **P78** | TurboQuant .tolist() capture-guard (adapted from noongh | opt-in | `GENESIS_ENABLE_P78_TOLIST_CAPTURE_GUARD` | — | Adapted from noonghunna's patch_tolist_cudagraph.py (Apache-2.0, github.com/noonghunna/qwen36-27b-single-3090). Surgical |
| **P67b** | TurboQuant spec-verify forward() routing (FULL CG enabl | opt-in | — | — | Genesis (see source / CHANGELOG) |

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

---

## Adding a new patch

1. **Pick a free ID.** Run `grep -E '^@register_patch' vllm/_genesis/patches/apply_all.py | head` and `grep -E '"P[0-9]+' vllm/_genesis/dispatcher.py` to confirm the next available number. Don't reuse retired IDs (P56/P57/P63 are deprecated but kept).
2. **Write wiring**: `vllm/_genesis/wiring/patch_<id>_<name>.py`. Use [`patch_71_block_verify.py`](vllm/_genesis/wiring/patch_71_block_verify.py) or [`patch_82_sglang_acceptance_threshold.py`](vllm/_genesis/wiring/patch_82_sglang_acceptance_threshold.py) as templates.
3. **Register in dispatcher** (P56+): add an entry to `PATCH_REGISTRY` in [`vllm/_genesis/dispatcher.py`](vllm/_genesis/dispatcher.py).
4. **Hook in apply_all**: add `@register_patch(...)` + `apply_patch_<id>_*` function in [`vllm/_genesis/patches/apply_all.py`](vllm/_genesis/patches/apply_all.py).
5. **Document in CHANGELOG**: add a `vX.YZ` entry to [`vllm/_genesis/CHANGELOG.md`](vllm/_genesis/CHANGELOG.md) explaining the WHY, empirical data, and ship/reject decision.
6. **Validate**:
   - Static: `python3 -c 'import ast; ast.parse(open("vllm/_genesis/wiring/patch_<id>_*.py").read())'`
   - Container: `docker compose down && docker compose up -d` (NOT `stop/start` — see [`CONFIGURATION.md`](CONFIGURATION.md) Container R/W layer note)
   - Empirical: blue/green sweep with `genesis_quality_harness.py` + `genesis_bench_v3.py`. SHIP gate: ≥30/31 quality + ≥+5% TPS (or whatever the patch targets).
7. **Credit upstream** in the patch docstring + `CREDITS.md` if backporting from someone else's PR / project.

