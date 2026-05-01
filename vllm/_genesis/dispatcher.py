# SPDX-License-Identifier: Apache-2.0
"""Genesis Dispatcher v2 — unified patch decision matrix + diagnostics.

Builds on top of `model_detect.py` and `config_detect.py` to provide:

  1. **Per-patch should_apply()** — single-line gate decision for each patch.
     Wraps `config_detect.recommend()` + `model_detect` checks + env-flag
     overrides into one consistent API.

  2. **Apply matrix dump** — diagnostic command-line entry-point that prints
     the full per-patch decision table for the current vllm config. Useful
     for operators to see WHY a patch was applied or skipped without grep-ing
     boot logs.

  3. **Startup logging** — single condensed line at boot summarizing which
     patches got applied, skipped (with reason), or failed. Replaces the
     scattered per-patch INFO lines that flood the boot log.

Usage at runtime
----------------
From a patch wiring (`patch_NN_*.py::apply()`):

    from vllm._genesis.dispatcher import should_apply, log_decision

    decision, reason = should_apply("P60")
    if not decision:
        log_decision("P60", decision, reason)
        return "skipped", reason
    # ... do the patching ...

From CLI / diagnostic:

    python3 -m vllm._genesis.dispatcher

Output: full apply matrix as ASCII table.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("genesis.dispatcher")


# ─── Validation issue dataclass ──────────────────────────────────────────
# Reported by validate_registry() and validate_apply_plan(). Severity
# levels: "ERROR" (operator must fix), "WARNING" (likely-wrong, allow boot
# to proceed), "INFO" (informational only).

@dataclass(frozen=True)
class ValidationIssue:
    severity: str  # "ERROR" | "WARNING" | "INFO"
    patch_id: str
    message: str


# ─── Patch metadata registry ───────────────────────────────────────────────
# Each patch declares what it touches + which env flag enables/disables it.
# This is the SINGLE source of truth for patch-to-feature mapping.

PATCH_REGISTRY: dict[str, dict[str, Any]] = {
    "P56": {
        "title": "TQ spec-decode safe-path guard (deprecated — superseded by P65)",
        "env_flag": "GENESIS_ENABLE_P56_SPEC_DECODE_GUARD",
        "default_on": False,
        "deprecated": True,
        "category": "spec_decode",
        "credit": "noonghunna (#40807, #40831)",
        "upstream_pr": None,
        "conflicts_with": ["P65"],
        "deprecation_note": (
            "P56 was a routing-layer workaround forcing spec-decode through a "
            "'safe' path when CG-aware buffers misaligned. Real fix is P65 "
            "(TurboQuant CG downgrade) which addresses the root cause in the "
            "full-attention path under FULL cudagraph capture. Kept opt-in for "
            "configurations where P65 is intentionally disabled and a routing "
            "guard is still desired (no such config verified in production)."
        ),
    },
    "P57": {
        "title": "TQ spec-decode capture-safe buffers (deprecated — research artifact)",
        "env_flag": "GENESIS_ENABLE_P57_SPEC_DECODE_CAPTURE_SAFE",
        "default_on": False,
        "deprecated": True,
        "category": "spec_decode",
        "credit": "noonghunna (#40831), gdn_attn.py reference",
        "upstream_pr": None,
        "conflicts_with": ["P65"],
        "deprecation_note": (
            "P57 v2 enlarges per-layer capture buffers from ~530 KiB to ~2.1 MiB "
            "(see wiring/patch_57 docstring for derivation), which is the "
            "MINIMAL sufficient fix for the original symptom but pushes total "
            "spec-decode buffer memory from ~270 MiB to ~1080 MiB across 32 "
            "layers — unacceptable on consumer Ampere with 24 GB VRAM. P65 "
            "(CG downgrade) achieves the same correctness without the memory "
            "blow-up. Kept opt-in as a research artifact / reference for "
            "future hardware with larger VRAM budgets."
        ),
    },
    "P58": {
        "title": "Async-scheduler -1 placeholder fix",
        "env_flag": "GENESIS_ENABLE_P58_ASYNC_PLACEHOLDER_FIX",
        "default_on": False,
        "category": "spec_decode",
        "credit": "z1ying (vllm#40768)",
        "upstream_pr": 40768,
    },
    "P59": {
        "title": "Qwen3 reasoning embedded tool_call recovery",
        "env_flag": "GENESIS_ENABLE_P59_QWEN3_TOOL_RECOVERY",
        "default_on": False,
        "category": "structured_output",
        "credit": "ZenoAFfectionate (vllm#39055)",
        "upstream_pr": 39055,
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_moe", "qwen3_next"]},
    },
    "P60": {
        "title": "GDN+ngram state recovery (Phase 1: SSM pre-copy)",
        "env_flag": "GENESIS_ENABLE_P60_GDN_NGRAM_FIX",
        "default_on": False,
        "category": "spec_decode",
        "credit": "tdoublep (vllm#40738), bhaktatejas922 (#39273)",
        "upstream_pr": 40738,
        "applies_to": {"is_hybrid": [True]},
    },
    "P60b": {
        "title": "GDN+ngram Triton kernel offset (Phase 2)",
        "env_flag": "GENESIS_ENABLE_P60B_TRITON_KERNEL",
        "default_on": False,
        "category": "spec_decode",
        "credit": "tdoublep (vllm#40738)",
        "upstream_pr": 40738,
        "applies_to": {"is_hybrid": [True]},
        "requires_patches": ["P60"],
    },
    "P61": {
        "title": "Qwen3 multi-tool first-occurrence (DEPRECATED — superseded by P12 v2)",
        "env_flag": "GENESIS_ENABLE_P61_QWEN3_MULTI_TOOL",
        "default_on": False,
        "deprecated": True,
        "category": "structured_output",
        "credit": "ExtReMLapin (vllm#40783) — P61 was supposed to flip P12's LAST-occurrence to FIRST via post-anchor replacement, but its anchor 'tool_call_index = ...' never matched P12-emitted 'idx = ...' form, so it silent-skipped when P12 was active. v7.62.5 (2026-04-28): P12 emit updated to FIRST directly; P61 retired. Setting GENESIS_ENABLE_P61=1 is now a harmless no-op (anchor not found vs already-FIRST P12 output).",
        "upstream_pr": 40783,
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_moe", "qwen3_next"]},
    },
    "P62": {
        "title": "Structured-output spec-decode reasoning-end timing fix",
        "env_flag": "GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING",
        "default_on": False,
        "category": "structured_output",
        "credit": "sfbemerk (vllm#36138), cicirori (vllm#34650)",
        "upstream_pr": 36138,
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_moe", "qwen3_next"]},
    },
    "P61b": {
        "title": "Qwen3 streaming partial-tag overlap guard",
        "env_flag": "GENESIS_ENABLE_P61B_STREAMING_OVERLAP",
        "default_on": False,
        "category": "structured_output",
        "credit": "ExtReMLapin (vllm#40783)",
        "upstream_pr": 40783,
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_moe", "qwen3_next"]},
    },
    "P63": {
        "title": "MTP/Eagle drafter GDN state recovery (deprecated — wrong layer)",
        "env_flag": "GENESIS_ENABLE_P63_MTP_GDN_STATE_RECOVERY",
        "default_on": False,
        "deprecated": True,
        "category": "spec_decode",
        "credit": "Genesis-original (hypothesis disproven 2026-04-25)",
        "upstream_pr": None,
        "deprecation_note": (
            "P63 hypothesis was wrong: MTP module uses layer_type='full_attention' "
            "(Qwen3NextAttention), NOT GDN. GDNAttentionMetadataBuilder.build_for_drafting "
            "is never called for MTP drafter. Real fix is P65 (TurboQuant CG downgrade) — "
            "the bug is in the full_attention path under FULL cudagraph capture, not GDN. "
            "P63 may still be relevant for eagle/draft_model methods that use a separate "
            "drafter model with hybrid layers, but no such configuration is verified yet."
        ),
    },
    "P64": {
        "title": "qwen3coder MTP streaming early-return fix",
        "env_flag": "GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING",
        "default_on": False,
        "category": "structured_output",
        "credit": "kotori-yan (vllm#39598)",
        "upstream_pr": 39598,
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_moe", "qwen3_next"]},
    },
    "P65": {
        "title": "TurboQuant spec-decode cudagraph downgrade",
        "env_flag": "GENESIS_ENABLE_P65_TURBOQUANT_SPEC_CG_DOWNGRADE",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Genesis-original (root cause for noonghunna #40880)",
        "upstream_pr": None,
        "applies_to": {"is_turboquant": [True]},
    },
    "P66": {
        "title": "cudagraph_capture_sizes spec-decode divisibility filter",
        "env_flag": "GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Genesis-original (mirrors fhl2000 vllm#23679 closed)",
        "upstream_pr": 23679,
    },
    "P68": {
        "title": "Auto force tool_choice=required for long-context tool calls",
        "env_flag": "GENESIS_ENABLE_P68_AUTO_FORCE_TOOL",
        "default_on": False,
        "category": "structured_output",
        "credit": "Genesis-original (long-ctx tool adherence mitigation)",
        "upstream_pr": None,
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_moe", "qwen3_next"]},
    },
    "P69": {
        "title": "Long-context tool-format reminder injection",
        "env_flag": "GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER",
        "default_on": False,
        "category": "structured_output",
        "credit": "Genesis-original (long-ctx tool adherence mitigation)",
        "upstream_pr": None,
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_moe", "qwen3_next"]},
    },
    "P70": {
        "title": "Auto-strict-ngram (force prompt_lookup_min>=8)",
        "env_flag": "GENESIS_ENABLE_P70_AUTO_STRICT_NGRAM",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Genesis-original (vllm#40875 enforcement)",
        "upstream_pr": None,
    },
    "P67": {
        "title": "TurboQuant multi-query kernel for spec-decode K+1",
        "env_flag": "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Genesis-original (proper fix for noonghunna #40880; replaces P65 workaround)",
        "upstream_pr": None,
        "applies_to": {
            "is_turboquant": [True],
        },
        "conflicts_with": ["P65"],
    },
    "P67b": {
        "title": "TurboQuant spec-verify forward() routing (FULL CG enable)",
        # P67b reuses P67's env flag intentionally — they're a coupled pair,
        # P67b is the forward() routing companion that bypasses
        # _prefill_attention for K+1 verify batches (cudagraph-safe).
        "env_flag": "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Genesis-original (FULL CG enable for P67 multi-query kernel)",
        "upstream_pr": None,
        "applies_to": {
            "is_turboquant": [True],
        },
        "requires_patches": ["P67"],
        "conflicts_with": ["P65"],
    },
    "P72": {
        "title": "profile_run M cap (unblocks --max-num-batched-tokens>4096 on MoE)",
        "env_flag": "GENESIS_ENABLE_P72_PROFILE_RUN_CAP",
        "default_on": False,
        "category": "compile_safety",
        "credit": "Genesis-original (Dynamo fake-tensor mismatch workaround for moe_align_block_size symbolic shape)",
        "upstream_pr": None,
    },
    "P71": {
        "title": "Block-verify rejection sampler (Sun 2024 ICLR)",
        "env_flag": "GENESIS_ENABLE_P71_BLOCK_VERIFY",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport of vllm#40819 (Z. Golpayegani draft) + Sun et al. arXiv 2403.10444 + 2 critical fixes from gemini-code-assist review (shared u per request, denom==0 → 1.0)",
        "upstream_pr": 40819,
    },
    "P74": {
        "title": "Auto chunk-clamp via long_prefill_token_threshold (P72 companion)",
        "env_flag": "GENESIS_ENABLE_P74_CHUNK_CLAMP",
        "default_on": False,
        "category": "compile_safety",
        "credit": "Genesis-original (zero-VRAM-cost prealloc-overflow safety net for P72-unblocked batched_tokens>4096)",
        "upstream_pr": None,
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_moe", "qwen3_next"]},
        "requires_patches": ["P72"],
    },
    "P75": {
        "title": "Auto-enable Suffix Decoding (Arctic Inference, vllm#25784)",
        "env_flag": "GENESIS_ENABLE_P75_SUFFIX_DECODING",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport-enabler of vllm#25784 (Arctic Inference Suffix Decoding) — operator convenience: auto-swap method=ngram→suffix when env enabled. Algorithm: arxiv 2411.04975.",
        "upstream_pr": 25784,
    },
    "P77": {
        "title": "Adaptive ngram K controller (EMA + hysteresis + auto-disable)",
        "env_flag": "GENESIS_ENABLE_P77_ADAPTIVE_NGRAM_K",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Genesis-original (port of SGLang adaptive_spec_params.py EMA+hysteresis Apache-2.0 + Nightjar arXiv 2512.22420 auto-disable extension). Targets free-form ngram pathology (46 tok/s).",
        "upstream_pr": None,
    },
    "P78": {
        "title": "TurboQuant .tolist() capture-guard (adapted from noonghunna)",
        "env_flag": "GENESIS_ENABLE_P78_TOLIST_CAPTURE_GUARD",
        "default_on": False,
        "category": "compile_safety",
        "credit": "Adapted from noonghunna's patch_tolist_cudagraph.py (Apache-2.0, github.com/noonghunna/qwen36-27b-single-3090). Surgical safety-net for cudagraph capture; complements our P22/P26/P44 prealloc.",
        "upstream_pr": None,
        "applies_to": {
            "is_turboquant": [True],
            "quant_format": ["fp8", "compressed_tensors"],
        },
    },
    "P79b": {
        "title": "Async × spec-decode proposer-sync backport (vllm#40610)",
        "env_flag": "GENESIS_ENABLE_P79B_ASYNC_PROPOSER_SYNC",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport of vllm#40610 (OPEN draft, tracked from #40608). Re-records prepare_inputs_event AFTER spec-decode proposer GPU work in sample_tokens(). Fixes async × spec-decode race where next batch _update_states could mutate block_table while previous batch's proposer was still reading on GPU. Genesis prod uses sync ngram so direct value is minimal; protects users on async+EAGLE/MTP/ngram_gpu.",
        "upstream_pr": 40610,
    },
    "P79c": {
        "title": "Stale spec_token_ids cleanup for unscheduled requests (vllm#37629)",
        "env_flag": "GENESIS_ENABLE_P79C_STALE_SPEC_TOKEN_CLEANUP",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport of vllm#37629 (OPEN, fixes #36906). Cleanup pass after main scheduling loop clears spec_token_ids for unscheduled running requests. Prevents -1 placeholder leak into F.embedding() under budget-exhausted high-concurrency on async + EAGLE/MTP. Genesis prod (max_num_seqs=2, sync ngram) gains nothing direct; protects high-concurrency multimodal users.",
        "upstream_pr": 37629,
    },
    "P81": {
        "title": "fp8 block-scaled MM low-M decode tuning (vllm#40925)",
        "env_flag": "GENESIS_ENABLE_P81_FP8_BLOCK_SCALED_M_LE_8",
        "default_on": False,
        "category": "kernel_perf",
        "credit": "Backport of vllm#40925 (tonyliu312, OPEN). Specializes w8a8_triton_block_scaled_mm default config for M<=8 (single-request decode + MTP K=3 verify): BLOCK_SIZE_M 64->16, num_stages 2->3 (non-ROCm). Empirical +23% median decode on GB10. Direct hit for Genesis prod (Qwen3.6-A3B FP8 + max_num_seqs=2 + no pre-tuned JSON for A5000).",
        "upstream_pr": 40925,
        "applies_to": {
            "quant_format": ["fp8"],
        },
    },
    "P82": {
        "title": "SGLang threshold_single OR-clause acceptance (BIASED — opt-in research)",
        "env_flag": "GENESIS_ENABLE_P82",
        "default_on": False,
        "category": "spec_decode",
        "credit": "SGLang team (sgl-project/sglang) speculative_sampling.cuh — port of the threshold_single OR-clause that breaks the structural ceiling clean_rate ≈ accept_rate^num_spec. Targets v7.13 strict-ngram acceptance gap. BIASED rule (loses unbiased-sampling guarantee); requires empirical quality validation before prod. Threshold baked from env GENESIS_P82_THRESHOLD_SINGLE (default 0.3) at server start.",
        "upstream_pr": None,
    },
    "P83": {
        "title": "MTP keep-last-cached-block (vllm#38182 downstream symptom — P84 is real fix)",
        "env_flag": "GENESIS_ENABLE_P83",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Root-cause analysis: vllm#38182 by uOnePiece + @Angazenn comment identifying single_type_kv_cache_manager.py:457 force-pop last cached block when use_eagle=True. MTP gets caught up via config/speculative.py:890-891 (use_eagle returns True for 'mtp'). EMPIRICALLY DISPROVEN as the actual cause: Genesis debug instrumentation showed find_longest_cache_hit was NEVER called for our workload because num_hashes=0 (block_size > prompt_len after P5 LCM-pad). The L457 pop is a downstream symptom, not the upstream cause. P84 (hash_block_size override) is the real fix. P83 kept as opt-in research artifact for future workloads where the pop site IS reached.",
        "upstream_pr": None,
        "applies_to": {"is_hybrid": [True]},
    },
    "P84": {
        "title": "hash_block_size override (vllm#38182 actual root cause)",
        "env_flag": "GENESIS_ENABLE_P84",
        "default_on": False,
        "category": "kv_cache",
        "credit": "Genesis-original discovery 2026-04-27 via P83 DEBUG instrumentation. scheduler.py:234 hard-codes hash_block_size=self.block_size; on hybrid Qwen3.6-MoE with P5 LCM-pad this becomes 2048+, so request_block_hasher computes 0 hashes for prompts < 2048 tokens. Cache machinery runs with overhead but never produces hits. P84 text-patches scheduler.py to read hash_block_size from env GENESIS_P84_HASH_BLOCK_SIZE (recommended value: 16 = full-attention default). Engage via GENESIS_ENABLE_P84=1 + GENESIS_P84_HASH_BLOCK_SIZE=16. Constraint: must divide every group's block_size, else vLLM's own assertion fires at startup. Related: vllm#38182 identified WRONG root cause (the L457 pop); P84 attacks the upstream cause.",
        "upstream_pr": None,
        "applies_to": {"is_hybrid": [True]},
    },
    "P85": {
        "title": "Hybrid fine-shadow prefix cache (vllm#38182 followup, MambaManager fix)",
        "env_flag": "GENESIS_ENABLE_P85",
        "default_on": False,
        "category": "kv_cache",
        "credit": "Genesis-original 2026-04-27 — synthesis of 6-round empirical investigation + deep code analysis. Identified TWO mismatches in hybrid prefix cache: (A) MambaManager.cache_blocks early-returns for prompts < self.block_size (e.g., 1424 < 2048); (B) Mamba align-mode pads with null_blocks so num_full_blocks > 0 still inserts 0 entries. P85 patches MambaManager to: (1) register shadow fine-grained hash entries (scale_factor=block_size/hash_block_size duplicates) when caching, (2) walk fine hashes on lookup with eviction-safety re-derive verify. Memory layout / ref-count untouched. Requires P84 (fine hashes computed). Architectural limit: cannot help prompts < block_size (Mamba state genuinely uncached at sub-block boundaries).",
        "upstream_pr": None,
        "applies_to": {"is_hybrid": [True]},
        "requires_patches": ["P84"],
    },
    "P86": {
        "title": "ngram batch_propose O(N*K) → O(N+K) direct-fill (vllm#40876)",
        "env_flag": "GENESIS_ENABLE_P86",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport of vllm#40876 (aaronagent, OPEN). Replaces O(N*K) `i in valid_ngram_requests` membership scan in NgramProposer.batch_propose with O(N+K) direct-fill loop iterating only the valid ngram requests. Algorithmic improvement, no behavioral change. Negligible at Genesis prod max_num_seqs=2 (~ns); meaningful at high-concurrency multi-user serving (e.g. N=64, K=32 saves ~1952 list-membership ops per batch step).",
        "upstream_pr": 40876,
    },
    "P87": {
        "title": "Marlin W4A16/W8A16 sub-tile output dim pad-on-load (vllm#40361)",
        "env_flag": "GENESIS_ENABLE_P87",
        "default_on": False,
        "category": "kernel",
        "credit": "Backport of vllm#40361 (OPEN). MarlinLinearKernel requires per-rank out_features divisible by GPTQ_MARLIN_MIN_THREAD_N=64. Sub-tile shards (e.g. Qwen3.5 GatedDeltaNet.in_proj_ba at TP>=2 with num_v_heads=64, or Intel/Qwen3.6-35B-A3B-int4-AutoRound n=32 shard at TP=2) fail can_implement and force a slow non-Marlin fallback (or refuse to load entirely on Ampere where Machete/CutlassW4A8/AllSpark are unavailable or restricted). P87 wraps three MarlinLinearKernel methods to zero-pad qweight/scales/qzeros/bias along the output dim at load, swap config.partition_weight_shape to padded value so downstream transforms see consistent layout, and slice the extra columns off the output in apply_weights. Runtime cost is zero — padding is one-time at load. PR bench: +24% on 2x RTX 3090 SM 8.6 with Intel/Qwen3.6-35B-A3B-int4-AutoRound TP=2 (137 -> 170 t/s). Closes vllm#35924 generically.",
        "upstream_pr": 40361,
        "applies_to": {
            "quant_format": [
                "int8_w8a16", "int4_w4a16",
                "autoround_int8", "autoround_int4",
                "gptq_int4", "awq_int4", "compressed_tensors",
            ],
        },
    },
    "PN8": {
        "title": "MTP/draft online-quant propagation (vllm#40849)",
        "env_flag": "GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT",
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Backport of vllm#40849 (bhoomit, OPEN). Modifies "
            "`get_draft_quant_config()` so that, when the spec-decode draft "
            "model has no explicit quantization config, it inherits the "
            "target's `OnlineQuantizationConfig` (e.g. fp8_per_tensor). "
            "Frees ~600 MiB on FP8-target + Eagle3 / DFlash / MTP-as-external-"
            "draft worker (1.45 GiB BF16 → 0.88 GiB FP8 on Qwen3-32B + Eagle3 "
            "per PR author bench). Also catches ValueError/FileNotFoundError "
            "in the existing draft lookup path (online-quant methods crash "
            "through checkpoint-config because hf_overrides is callable). "
            "NO-OP for current Genesis prod (Lorbus/Minachist 27B do not run "
            "online-quant + external draft). Becomes valuable when DFlash / "
            "Eagle3 / FP8 stacks roll out."
        ),
        "upstream_pr": 40849,
        "applies_to": {
            # Predicate enforced naturally by the patched function — when
            # spec-decode is off OR target is not online-quantized, the new
            # branch falls through identical to vanilla. No model gating.
        },
    },
    "PN9": {
        "title": "Independent drafter attention backend (vllm#39930)",
        "env_flag": "GENESIS_ENABLE_PN9_INDEPENDENT_DRAFTER_ATTN",
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Backport of vllm#39930 (MatthewBonanni, MERGED). Allows the "
            "spec-decode drafter to use a different attention backend than "
            "the target model. Unblocks drafters with incompatible "
            "requirements (e.g. DFlash needs non-causal attention support, "
            "which TRITON_ATTN does not provide → ValueError on boot). "
            "Modifies `LLMBaseProposer._create_draft_vllm_config()` to "
            "always reset the drafter's attention backend (None = "
            "auto-select). Genesis minimal port: env "
            "GENESIS_PN9_DRAFTER_BACKEND chooses backend (e.g. FLASH_ATTN); "
            "unset/auto → drafter auto-selects. Does NOT add the new "
            "SpeculativeConfig.attention_backend pydantic field (too "
            "invasive at runtime for a frozen dataclass + field_validator). "
            "Unblocks DFlash spike sprint task without full pin bump risk "
            "from #40860 mega-merge. NO-OP for current Genesis prod (PROD "
            "uses ngram drafter, no attention backend conflict)."
        ),
        "upstream_pr": 39930,
        "applies_to": {
            # Patch only takes effect inside _create_draft_vllm_config which
            # is only called when spec-decode is active. No additional gate.
        },
    },
    "PN11": {
        "title": "GDN a/b contiguity in fix_query_key_value_ordering (vllm#41142)",
        "env_flag": "GENESIS_ENABLE_PN11_GDN_AB_CONTIGUOUS",
        "default_on": False,
        "category": "model_correctness",
        "credit": (
            "Backport of vllm#41142 (Yeuvoir, OPEN). Fixes upstream issue "
            "#41112: in `GatedDeltaNetAttention.fix_query_key_value_ordering` "
            "the reshape of `b` and `a` returns a non-contiguous view when "
            "num_v_heads == num_k_heads (np/ng == 1), breaking "
            "`fused_post_conv_prep` Triton kernel which assumes head-dim "
            "stride 1. Adds `.contiguous()` to both lines (zero cost when "
            "already contiguous; copy only on the buggy path). Symptom on "
            "affected configs: silent quality drift, no crash. For Genesis "
            "prod (Qwen3.6 27B has np/ng=8, 35B has no GDN) this is "
            "DEFENSIVE — installs guard against future model swaps."
        ),
        "upstream_pr": 41142,
        "applies_to": {
            # Patch only matters when GDN layer's fix_query_key_value_ordering
            # runs with np/ng==1. Genesis prod doesn't trigger it but the
            # patch is harmless (no-op .contiguous() call).
        },
    },
    "PN12": {
        "title": "FFN intermediate scratch pool — Cliff 1 fix on TQ3 path",
        "env_flag": "GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL",
        "default_on": False,
        "category": "memory_savings",
        "credit": (
            "Genesis-original 2026-04-29 — Cliff 1 fix on TQ3 path. "
            "Closes 138 MiB OOM at 192K + tool-call on RTX 3090 (noonghunna "
            "report). PN8 closes Cliff 1 on FP8 by freeing ~600 MiB persistent "
            "draft VRAM, but on TQ3 frees only ~230 MiB — not enough slack "
            "for the 138 MiB transient. Different memory class. PN12 pools "
            "the SiluAndMul output across layers (single buffer per "
            "(intermediate_size, dtype, device)) — reduces per-step allocator "
            "churn from ~4.7-18 GiB to ~73-285 MiB on Lorbus 27B-int4. "
            "Pointer-stable (cudagraph-safe). Cross-engine reference: "
            "TensorRT-LLM live-range activation reuse (gold standard); "
            "alternative paths: vLLM PR #34207 (silu_and_mul.out variant), "
            "SGLang PR #15927 (piecewise CUDA graph private pool). Tested "
            "via 17 unit tests in tests/test_ffn_intermediate_cache.py."
        ),
        "upstream_pr": 34207,  # would obsolete this patch if merged
        "applies_to": {
            # Patch matters when SiluAndMul / MulAndSilu is on the hot path
            # (any model with FFN gate-up + silu activation — qwen3, llama,
            # mistral, deepseek, etc.). For MoE models impact is per-expert.
        },
    },
    "PN19": {
        "title": "Scoped max_split_size_mb during model load (vllm#41268)",
        "env_flag": "GENESIS_ENABLE_PN19_SCOPED_MAX_SPLIT",
        "default_on": False,
        "category": "memory_savings",
        "credit": (
            "Backport of vllm#41268 (MatthewBonanni, OPEN 2026-04-30). "
            "PyTorch 2.10+ introduced load-time allocator fragmentation: "
            "weight segments split inside other segments, leaving "
            "200-500 MiB unusable. Mitigation: temporarily set "
            "max_split_size_mb=20 (PyTorch minimum) for the duration of "
            "model load, restore prior on exit. Cudagraph-safe (load-"
            "time only; capture phase uses restored allocator). "
            "Default OFF — operator should measure fragmentation gap "
            "via nvidia-smi peak during load before vs after to confirm "
            "win on Ampere SM 8.6 (PR #41268 measured on H100; A5000 "
            "behavior unverified). Cross-reference Genesis memory "
            "feedback_p104_l2_persistence_thrashing — hardware-mismatch "
            "patches are anti-pattern; measure first."
        ),
        "upstream_pr": 41268,
        "applies_to": {
            # Always applicable on CUDA. Self-detects torch < 2.11 lack
            # of _accelerator_setAllocatorSettings and falls through
            # unchanged.
        },
    },
    "PN23": {
        "title": "DFlash combine_hidden_states dtype cast (vllm#40334)",
        "env_flag": "GENESIS_ENABLE_PN23_DFLASH_DTYPE_FIX",
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Backport of vllm#40334 (ciphernaut, OPEN 2026-05-01). Six-line "
            "defensive cast in Qwen3DFlashModel.combine_hidden_states to handle "
            "mixed-precision targets (AWQ + non-quantized layers, FP8 + BF16 mix). "
            "Casts hidden_states to fc.params_dtype before FC layer call. Fixes "
            "RuntimeError on mixed-precision DFlash configs."
        ),
        "upstream_pr": 40334,
        "applies_to": {
            # DFlash-specific; auto-no-op when qwen3_dflash.py absent or anchor
            # already has params_dtype cast (upstream merge).
        },
        "conflicts_with": [],
        "requires_patches": [],
    },
    "PN21": {
        "title": "DFlash SWA support partial backport (vllm#40898)",
        "env_flag": "GENESIS_ENABLE_PN21_DFLASH_SWA",
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Partial backport of vllm#40898 (jianc99, OPEN 2026-05-01). "
            "Adds SWA config preservation in speculators/algos.py and forces "
            "causal=True on sliding-window layer attention metadata in "
            "v1/spec_decode/dflash.py. The qwen3_dflash.py model class "
            "changes (7+ sub-patches) are NOT backported. EMPIRICAL on 35B-A3B "
            "DFlash 160K: tool-call regresses 5-6/7 vs 7/7 baseline (without PN21) — "
            "metadata/compute mismatch (config says SWA, model computes full attn). "
            "DEFAULT OFF, NOT enabled in any launch script. Wait for upstream merge "
            "or full manual model class backport before enabling. Composes (no conflict) "
            "with PN24 if/when full enabler lands."
        ),
        "upstream_pr": 40898,
        "applies_to": {},
        "conflicts_with": [],
        "requires_patches": [],  # Pairs with PN24 but does not strictly require it
    },
    "PN22": {
        "title": "Local argmax for TP draft (vllm#39419 backport)",
        "env_flag": "GENESIS_ENABLE_PN22_LOCAL_ARGMAX_TP",
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Backport of vllm#39419 (EanWang, OPEN 2026-05-01). Adds "
            "get_top_tokens() plumbing to Qwen3 and Qwen3-DFlash model "
            "classes, enabling vocab-parallel argmax on each TP rank "
            "instead of all-gathering full logits. Wins +9.4-30.6% TPS "
            "on TP>=2 + draft model per PR author. LogitsProcessor."
            "get_top_tokens() callsite is already in our pin (PR #34049 "
            "merged). Llama and Eagle3 parts of the upstream PR are not "
            "backported — Genesis does not run those models in production."
        ),
        "upstream_pr": 39419,
        "applies_to": {},
        "conflicts_with": [],
        "requires_patches": [],
    },
    "PN24": {
        "title": "DFlash aux layer +1 indexing fix (vllm#40727)",
        "env_flag": "GENESIS_ENABLE_PN24_DFLASH_AUX_LAYER_FIX",
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Backport of vllm#40727 (benchislett, OPEN 2026-05-01). One-line "
            "semantic fix in _get_eagle3_aux_layers_from_config. DFlash stores "
            "target_layer_ids as 0-indexed; downstream Eagle3 aux machinery "
            "expects 1-indexed (layer 0 = embedding). +1 shift converts. "
            "Empirical: AL gsm8k 6.18→6.42 per PR author."
        ),
        "upstream_pr": 40727,
        "applies_to": {},
        "conflicts_with": [],
        "requires_patches": [],
    },
    "PN26": {
        "title": "TQ unified perf pack (centroids prebake + sparse V scaffold)",
        "env_flag": "GENESIS_ENABLE_PN26_TQ_UNIFIED",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Genesis-original 2026-05-01 unification of three OPEN upstream "
            "PRs (jasonkim8652): #41418 pre-baked Lloyd-Max centroids (drop-in "
            "safe, eliminates 50ms-2.5s JIT solver per shape on cold boot); "
            "#41422 sparse V tile-skip in decode kernel (scaffolded, OFF by "
            "default until NVIDIA Ampere correctness validation — author "
            "validated AMD MI300X only); #41414 head_dim pow-2 padding "
            "DROPPED — Qwen3.6 head_dim=128 already pow-2, would add dead "
            "code overhead. Genesis defensive addition: self-check at "
            "module-init asserts prebaked centroids equal solver output; on "
            "drift (e.g. upstream changes Lloyd-Max algo) auto-disables "
            "prebake and falls through to runtime solver with WARNING. No "
            "silent staleness. Composes with P67/P98/PN8 — orthogonal code "
            "paths."
        ),
        "upstream_pr": 41418,
        "applies_to": {},
        "conflicts_with": [],
        "requires_patches": [],
    },
    "PN25": {
        "title": "SiluAndMul.forward_native opaque-op pool (Cliff 1 mech B compile path)",
        "env_flag": "GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE",
        "default_on": False,
        "category": "memory_savings",
        "credit": (
            "Genesis-original 2026-05-01 in response to noonghunna's "
            "club-3090#16 (VolandBerlioz/ampersandru cross-rig OOM trace, "
            "RTX 3090 24 GB + Lorbus 27B + OpenCode 29K prefill). PN12 "
            "patches eager `forward_cuda` but `custom_ops=['none']` (default "
            "under V1 aot_compile_fullgraph) routes dispatch through "
            "`forward_native` which Inductor inlines and lowers to "
            "`empty_strided_cuda(...)`, bypassing PN12's pool. "
            "Sister-patch PN25 patches `forward_native` to dispatch through "
            "an opaque `genesis::silu_and_mul_pooled` torch.library.custom_op "
            "(Inductor cannot inline opaque ops). Both patches share the "
            "same FFNIntermediateCache pool. Recommended pairing for any "
            "inductor-heavy config."
        ),
        "upstream_pr": None,
        "applies_to": {},
        "conflicts_with": [],
        "requires_patches": [],  # complements PN12 but does not require it
    },
    "PN17": {
        "title": "FA2 softmax_lse runtime clamp (Cliff 1 mechanism A, Issue #11)",
        "env_flag": "GENESIS_ENABLE_PN17_FA2_LSE_CLAMP",
        "default_on": False,
        "category": "memory_savings",
        "credit": (
            "Genesis-original 2026-04-30 in response to noonghunna's "
            "Genesis Issue #11 cross-rig diagnosis (RTX 3090, 2026-04-29). "
            "FA2 `flash_attn_varlen_func` allocates softmax_lse buffer "
            "of shape [num_seqs, num_heads, max_seqlen_k] sized by "
            "the max_seqlen_k argument — NOT actual seqused_k. vLLM's "
            "gpu_model_runner sets attn_metadata.max_seq_len = "
            "max_model_len during cudagraph capture for shape stability "
            "(see vllm#40961 SWA case); this leaks into runtime "
            "decode/prefill, causing 50-100 MiB over-allocation at "
            "long context. Closes Cliff 1 mechanism A (FA2 path); "
            "widens long-text-no-vision safe envelope from ~150K to "
            "~205K. Mechanism B (FFN intermediate buffer 138 MiB on "
            "long-vision) is OUT OF SCOPE — requires upstream-FFN "
            "chunked forward, not addressable from Genesis text-patch "
            "layer. Cudagraph-safe: clamp only fires when "
            "is_current_stream_capturing() returns False; capture-time "
            "preserves max_model_len padding. Reference: "
            "Dao-AILab/flash-attention#1011 (open since 2024)."
        ),
        "upstream_pr": None,
        "applies_to": {
            # Applies whenever FA2 varlen path is active. Most relevant
            # at long context (>100K) where the cap-leak dominates.
        },
    },
    "PN16": {
        "title": "Lazy-reasoner request hook (per-request enable_thinking)",
        "env_flag": "GENESIS_ENABLE_PN16_LAZY_REASONER",
        "default_on": False,
        "category": "request_middleware",
        "credit": (
            "Genesis-original 2026-04-29. Hybrid policy on whether the "
            "model's `<think>...</think>` reasoning block adds value for "
            "this specific request. Variant 1 (pre-decision): force "
            "enable_thinking=False on short prompts (< "
            "GENESIS_PN16_THRESHOLD_CHARS, default 300) without tools, "
            "json_schema, or reasoning-signal patterns (math/code/CoT "
            "keywords). Variant 3 (client override): respect explicit "
            "chat_template_kwargs.enable_thinking from the client. Variant "
            "4 (LogitsProcessor cap): UPSTREAM-BLOCKED — vllm v1 rejects "
            "custom logits processors when speculative_config is set "
            "(Genesis PROD = MTP K=3). Variant 5 (prompt-engineering soft "
            "cap): fallback when max_thinking_tokens > 0; appends a "
            "concise-reasoning hint to the last user message. Soft cap, "
            "depends on model compliance, but works with spec-decode. "
            "Goal: reduce wasted reasoning tokens + TTFT on simple prompts "
            "without doubling latency or load. Stats counters exposed via "
            "`vllm._genesis.middleware.lazy_reasoner.get_stats()`."
        ),
        "upstream_pr": None,
    },
    "PN14": {
        "title": "TQ decode IOOB safe_page_idx clamp (vllm#40074)",
        "env_flag": "GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP",
        "default_on": False,
        "category": "kernel_safety",
        "credit": (
            "Backport of vllm#40074 (devarakondasrikanth @adobe, OPEN). "
            "Fixes upstream issue #39998 — Triton bounds-checker assertion "
            "in `_tq_decode_stage1` on long (>32k) sequences. The mask= "
            "argument guards the LOADED VALUE on masked-out lanes but not "
            "the address arithmetic; clamping page_idx to 0 via "
            "`tl.where(kv_mask, page_idx, 0)` keeps the pointer in-bounds "
            "even on lanes whose result is discarded. Originally reported "
            "on 4090 (sm_89); jhsmith409 confirmed clean apply on 5090 "
            "(sm_120) while stacking on top of #39931. Defensive on Genesis "
            "Ampere prod (sm_86 — assertion not seen). Becomes load-bearing "
            "on Sander's planned RTX PRO 6000 Blackwell upgrade. Self-"
            "retires via marker `safe_page_idx` when #40074 merges. "
            "Codepath fires when spec-decode OFF/K=1 OR P67 dispatch returns "
            "False (shape outside envelope) — runs in Genesis prod despite "
            "MTP K=3 being active."
        ),
        "upstream_pr": 40074,
        "applies_to": {
            "is_turboquant": [True],
        },
    },
    "PN13": {
        "title": "CUDAGraphWrapper gc.collect/empty_cache lambda arity (vllm#41235)",
        "env_flag": "GENESIS_ENABLE_PN13_CUDA_GRAPH_LAMBDA_ARITY",
        "default_on": False,
        "category": "cudagraph_safety",
        "credit": (
            "Backport of vllm#41235 (roikoren755, OPEN). Fixes worker-death "
            "TypeError in CUDAGraphWrapper.__call__: 0-arg lambdas patched "
            "over gc.collect / torch.accelerator.empty_cache fail when dynamo "
            "calls gc.collect(generation) with positional arg during nested "
            "@torch.compile recompile inside cudagraph capture region. "
            "Genesis-relevant because P67/P67b/P78/P85 family uses nested "
            "compiled kernels — dynamo can recompile mid-capture. Author "
            "reports 'consistent on GB200 nightly' → directly relevant for "
            "Sander's planned R6000 Pro Blackwell upgrade. Cost: 2-line text "
            "patch, zero runtime overhead, defensive only. NO-OP on workloads "
            "that don't trigger nested recompile."
        ),
        "upstream_pr": 41235,
        "applies_to": {
            # Patch only matters during cudagraph capture with nested compile
            # recompiles. Harmless no-op otherwise.
        },
    },
    "P94": {
        "title": "Spec-decode prepare_next_token_ids_padded zero-alloc (vllm#41043)",
        "env_flag": "GENESIS_ENABLE_P94",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport of vllm#41043 (wangluochao902, MERGED 2026-04-29). Removes GPU->CPU .tolist() sync + list-comp Python objects + np.array allocation in LLMBaseProposer.prepare_next_token_ids_padded hot path. PR author measured P99 TPOT -9.3% on Llama-3.1-8B + Eagle3 TP=4. For our MTP K=3 single-stream: expected +2-4% wall TPS + tighter CV. SUPERSEDED-ON-MERGE: when our pin advances past the merge SHA the patch will SKIP cleanly via drift detection on the original .tolist() anchor — at that point delete the wiring file + this entry.",
        "upstream_pr": 41043,
        "superseded_by": "vllm#41043 (merged 2026-04-29)",
        "applies_to": {
            # Applies whenever spec-decode is active. All spec methods.
        },
    },
    "P100": {
        "title": "FlashInfer FULL CUDA graph for spec-decode (vllm#41127)",
        "env_flag": "GENESIS_ENABLE_P100",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": "Backport of vllm#41127 (open 2026-04-28). Per Sander 'не ждём, изучаем, импортируем'. Native FlashInfer can route uniform query_len>1 (1+num_spec_tokens) batches through prefill wrapper in cudagraph mode (zero_rows padding bit-identical). Adds FISpecDecode dataclass + _get_spec_decode_prefill_wrapper method + per-row qo_indptr delta scan in build() + FISpecDecode case in forward(). 11 sub-patches on flashinfer.py. NO-OP for PROD (turboquant_attn). Active for 27B variants (FlashInfer + spec-decode + non-DCP). Expected: +5-10% TPS on Ampere SM 8.6.",
        "upstream_pr": 41127,
        "applies_to": {},  # FlashInfer auto-selected; gating via env_flag only
    },
    "P103": {
        "title": "FLA Cliff 2 chunked fwd_h+fwd_o orchestrator (qwen36-27b-single-3090#1)",
        "env_flag": "GENESIS_ENABLE_P103",
        "default_on": False,
        "category": "memory_hotfix",
        "credit": "Genesis-original 2026-04-28 in response to noonghunna Cliff 2 OOM report (qwen36-27b-single-3090#1). Wraps chunk.py::chunk_gated_delta_rule_fwd to split T-dim into MAX_T sub-prompts; runs fwd_h + fwd_o per sub-call, chains final_state, never materializes full (B, NT, H, V, K) h tensor. For Qwen3.6-27B at T=64K: peak h drops 4x (805 → 200 MiB per rank). Saves ~600 MiB headroom for long-context single-GPU users. Falls back to original for cu_seqlens != None or T <= MAX_T. Default OFF; opt-in via GENESIS_ENABLE_P103=1. Threshold: GENESIS_FLA_FWD_H_MAX_T (default 16384, rounded down to FLA_CHUNK_SIZE multiple). KDA path uncovered (separate model class).",
        "upstream_pr": None,
        "applies_to": {
            "model_arch": [
                "Qwen3MoeForCausalLM",
                "Qwen3_5ForConditionalGeneration",
                "Qwen3NextForCausalLM",
            ],
        },
    },
    "P101": {
        "title": "TQ continuation 64-token slicing (vllm#41123 SELECTIVE)",
        "env_flag": "GENESIS_ENABLE_P101",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": "Selective backport of vllm#41123 TQ on hybrid models. TAKE: _CONTINUATION_DECODE_THRESHOLD 128→64 + _CONTINUATION_DECODE_MAX_CACHED_LEN=32K + 64-token slicing loop in _prefill_attention. SKIP: cudagraph_support downgrade (would hurt PROD), hybrid boundary-skip (would break our explicit skip-layers). Expected: +3-12% TPS on PROD long-context. Composes with P98/P99.",
        "upstream_pr": 41123,
        "applies_to": {
            "kv_cache_dtype": [
                "turboquant_k8v4", "turboquant_4bit_nc",
                "turboquant_k3v4_nc", "turboquant_3bit_nc",
            ],
        },
    },
    "P99": {
        "title": "WorkspaceManager.get_simultaneous memoization (perf hotfix)",
        "env_flag": "GENESIS_ENABLE_P99",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": "Per Sander 2026-04-28: 'if revert gives speedup, look at kernel — maybe rewrite'. P99 keeps upstream WorkspaceManager design (shared memory, 60x savings) but adds memoization to bypass per-call list-comp + accumulate + _ensure_workspace_size. Cache hit ~5x faster than full computation. Composes with P98 (P98 reverts turboquant_attn to per-layer; P99 helps any other backend using WorkspaceManager).",
        "upstream_pr": 40941,
        "applies_to": {},  # applies whenever WorkspaceManager is used
    },
    "P98": {
        "title": "TQ WorkspaceManager revert (vllm#40941 perf hotfix)",
        "env_flag": "GENESIS_ENABLE_P98",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": "Reverts upstream PR #40941 (MERGED 2026-04-27). PR introduced WorkspaceManager indirection in turboquant_attn._decode_attention hot path. Diagnosis 2026-04-28: caused 17% TPS regression on PROD (200 → 167 TPS) due to current_workspace_manager().get_simultaneous() Python lookup × N layers × per-step. Restores OLD per-layer cached buffer pattern. Memory cost: O(num_layers) extra dequant buffers (~1GB for 64-layer model). DO NOT enable on H100/H200 high-concurrency where WorkspaceManager amortizes better. NOTE: this patch is a DELIBERATE INVERSE of merged upstream behavior (NOT a backport) — it remains a perf hotfix specifically for Ampere small-batch single-stream workloads even though the upstream PR is merged. Author: Sandermage.",
        "upstream_pr": 40941,
        "applies_to": {
            "kv_cache_dtype": [
                "turboquant_k8v4", "turboquant_4bit_nc",
                "turboquant_k3v4_nc", "turboquant_3bit_nc",
            ],
        },
    },
    "P95": {
        "title": "Marlin TP cudagraph cap on Ampere (vllm#40385)",
        "env_flag": "GENESIS_ENABLE_P95",
        "default_on": False,
        "category": "stability",
        "credit": "Backport of vllm#40385 (OPEN as of 2026-04-28). Defensive cap of max_cudagraph_capture_size to 8 when ALL of: TP>1, Ampere SM 8.0 family (covers SM 8.6 A5000), quantization endswith '_marlin', AND user did NOT set explicit cudagraph sizing. Mitigates vllm#40121 (illegal memory access during CG replay on TP>1 + Marlin + Ampere). NO-OP for our PROD (FP8, not Marlin); ACTIVE for Lorbus INT4 + Minachist gs128 (Marlin path). Operator override via --compilation-config bypasses entirely.",
        "upstream_pr": 40385,
        "applies_to": {
            "quant_format": [
                "gptq_int4", "gptq_int8", "awq_int4", "awq_int8",
                "compressed_tensors", "int4_w4a16", "int8_w8a16",
                "autoround_int4", "autoround_int8",
            ],
        },
    },
    "P91": {
        "title": "AutoRound row-parallel group cdiv + start-idx fix (vllm#39460)",
        "env_flag": "GENESIS_ENABLE_P91",
        "default_on": False,
        "category": "quantization",
        "credit": "Backport of non-MoE-specific portion of vllm#39460 (CLOSED). gptq_marlin.py:402-407 computes scales_and_zp_size = input_size_per_partition // group_size — when input_size_per_partition % group_size != 0 (AutoRound INT4/INT8 checkpoints with awkward shard sizes), this floor-div drops the trailing partial group of scales. Combined with parameter.py:222-225 load_row_parallel_weight using `tp_rank * shard_size` as start_idx (in scale-rows units, but the source tensor is indexed in scales-rows that map to input-element groups), rank-1 scales load from the wrong offset for partial-group shards → silent dequant corruption or fallback to slow non-Marlin path. P91 (a) replaces both floor-divs with cdiv(), (b) tags scales/qzeros with row_group_size + row_input_size_per_partition, (c) makes load_row_parallel_weight compute start_idx as (tp_rank * input_partition_size) // group_size when those tags present. Hypothesized as dominant cause of Lorbus INT4 < INT8 perf gap on our 2x A5000 (87/61/67 vs 93/77/86 t/s) — sister bug #38064 had 2.72x latency improvement when fixed. We do NOT port the MoE/gate_linear/gemma4 changes (those are Gemma4-specific).",
        "upstream_pr": 39460,
        "applies_to": {
            "quant_format": [
                "autoround_int8", "autoround_int4",
                "gptq_int4", "int8_w8a16", "int4_w4a16",
                "compressed_tensors",
            ],
        },
    },
}


# ─── Layer 2: model-aware applies_to gate ─────────────────────────────────

def _check_applies_to(
    patch_id: str, meta: dict[str, Any]
) -> tuple[bool, str]:
    """Layer 2 model-compatibility gate.

    Reads `meta["applies_to"]` (a dict mapping profile-key → list of allowed
    values), looks up the live model profile via `model_detect.get_model_profile()`,
    and returns (compatible, reason).

    Profile keys recognized: 'model_class', 'quant_format', 'kv_cache_dtype',
    'is_moe', 'is_hybrid', 'is_turboquant'. Any key whose actual value is None
    (detector couldn't resolve) is treated as compatible (conservative — let
    the patch apply and have its own call-site guards decide).

    Returns:
        (True, reason)  — model matches all applies_to constraints (or none
                          declared, or model couldn't be resolved)
        (False, reason) — explicit incompatibility: actual_value not in
                          allowed set for at least one key
    """
    applies_to = meta.get("applies_to")
    if not applies_to:
        return True, "no applies_to declared (model-class agnostic)"

    try:
        from vllm._genesis.model_detect import get_model_profile
        profile = get_model_profile()
    except Exception as e:
        return True, f"model_detect probe failed ({e}) — conservative apply"

    if not profile.get("resolved", False):
        return True, "model profile unresolved — conservative apply"

    # Map profile keys onto applies_to keys (some applies_to use boolean
    # aliases like is_moe / is_hybrid / is_turboquant for readability).
    key_aliases = {
        "is_moe": "moe",
        "is_hybrid": "hybrid",
        "is_turboquant": "turboquant",
    }

    # Build a flat profile dict that combines model_detect output with
    # boolean-alias mapping so the new predicates evaluator can read both
    # "is_turboquant" (used in applies_to) and "turboquant" (model_detect
    # native key) interchangeably.
    flat_profile: dict[str, Any] = dict(profile)
    for applies_key, profile_key in key_aliases.items():
        if profile_key in profile and applies_key not in flat_profile:
            flat_profile[applies_key] = profile[profile_key]

    # ─── Path A: richer predicate DSL (compat/predicates) ────────────────
    # Detect compound keys: all_of / any_of / not / none_of. If present,
    # delegate to the new evaluator. This lets new patches use the richer
    # syntax while old ones keep working unchanged.
    compound_keys = ("all_of", "any_of", "not", "none_of")
    if any(k in applies_to for k in compound_keys):
        try:
            from vllm._genesis.compat.predicates import evaluate
            ok, reason = evaluate(applies_to, flat_profile)
            return ok, ("applies_to satisfied" if ok
                        else f"MODEL-COMPAT: {reason}")
        except Exception as e:
            log.warning(
                "[Genesis dispatcher] %s: predicate evaluator raised (%s) — "
                "conservative apply. Check applies_to syntax.",
                patch_id, e,
            )
            return True, f"predicate evaluator error ({e}) — conservative apply"

    # ─── Path B: legacy flat-dict applies_to (backward compatible) ───────
    # Also pull version-related keys out and check via version_check.
    version_keys = (
        "vllm_version_range", "torch_version_min", "triton_version_min",
        "cuda_runtime_min", "nvidia_driver_min", "python_version_min",
        "compute_capability_min", "compute_capability_max",
    )
    version_constraints = {k: v for k, v in applies_to.items() if k in version_keys}
    profile_constraints = {k: v for k, v in applies_to.items() if k not in version_keys}

    # Version range checks
    if version_constraints:
        try:
            from vllm._genesis.compat.version_check import (
                check_version_constraints,
            )
            v_ok, v_results = check_version_constraints(version_constraints)
            if not v_ok:
                failed = [r for r in v_results if r.matched is False]
                if failed:
                    return False, f"VERSION: {failed[0].reason}"
                return False, "VERSION: constraint violation"
        except Exception as e:
            log.debug("[Genesis dispatcher] %s: version_check failed (%s) — "
                      "conservative apply", patch_id, e)

    # Legacy profile gates
    for key, allowed in profile_constraints.items():
        profile_key = key_aliases.get(key, key)
        actual = profile.get(profile_key)
        if actual is None:
            continue  # detector couldn't resolve → conservative
        if not isinstance(allowed, (list, tuple, set)):
            allowed = [allowed]
        if actual not in allowed:
            return False, (
                f"MODEL-COMPAT: {key}={actual!r} not in {list(allowed)!r}"
            )
    return True, "applies_to satisfied"


# ─── Single-call gate ─────────────────────────────────────────────────────

def should_apply(patch_id: str) -> tuple[bool, str]:
    """Unified gate: returns (apply_decision, reason).

    Combines:
      - env-flag check (`GENESIS_ENABLE_P<patch>=1` opt-in)
      - `applies_to` model-compatibility hard-skip (Layer 2, opt-in patches
        respect env override; default_on patches honor it strictly)
      - `config_detect.recommend(patch_id)` (model+config-aware decision)

    The decision rule:

      1. If env flag is truthy AND patch is opt-in (default_on=False) → apply,
         operator override wins over applies_to (logged as override).
      2. If env flag is unset/falsy AND patch is `default_on=False` → skip (opt-in)
      3. If applies_to declared and actual model profile mismatches → hard-skip
         with WARNING-class reason ("MODEL-COMPAT: ..."). For default_on=True
         patches this kicks in unconditionally; for env-truthy opt-ins it's
         logged but apply proceeds (override).
      4. Otherwise consult `config_detect.recommend()`:
         - "skip:..." → don't apply
         - "redundant:..." → don't apply
         - "deprecated:..." → don't apply
         - "neutral" / "apply" → apply

    Returns:
        (True, reason) — patch should apply
        (False, reason) — patch should skip, with human-readable reason
    """
    meta = PATCH_REGISTRY.get(patch_id)
    if meta is None:
        return False, f"unknown patch_id {patch_id!r}"

    env_flag = meta.get("env_flag")
    env_value = os.environ.get(env_flag, "") if env_flag else ""
    env_truthy = env_value.strip().lower() in ("1", "true", "yes", "on")

    # Operator override: env truthy = always apply (subject to anchor presence)
    if env_truthy:
        # Layer 2 applies_to is informational under env-override
        compat, compat_reason = _check_applies_to(patch_id, meta)
        if not compat:
            log.warning(
                "[Genesis dispatcher] %s: env OVERRIDE applies_to mismatch — "
                "%s. Proceeding because operator set %s=1.",
                patch_id, compat_reason, env_flag,
            )
        # Still consult config_detect to PRINT the recommendation as info
        try:
            from vllm._genesis.config_detect import recommend
            verdict, reason = recommend(patch_id)
            if verdict == "apply":
                return True, f"opt-in env + config recommends apply: {reason}"
            elif verdict == "neutral":
                return True, f"opt-in env (config: neutral)"
            else:
                return True, (
                    f"opt-in env OVERRIDE (config recommends {verdict}: "
                    f"{reason}) — proceeding because operator forced it"
                )
        except Exception as e:
            return True, f"opt-in env (config_detect probe failed: {e})"

    # Env flag unset/falsy
    if not meta.get("default_on", False):
        if meta.get("deprecated", False):
            return False, (
                f"opt-in only AND empirically deprecated — "
                f"keeping skip; set {env_flag}=1 only for diagnostics"
            )
        return False, f"opt-in only — set {env_flag}=1 to engage"

    # default_on=True: enforce applies_to as Layer 2 HARD skip.
    compat, compat_reason = _check_applies_to(patch_id, meta)
    if not compat:
        log.warning(
            "[Genesis dispatcher] %s HARD-SKIP — %s. Patch designed for a "
            "different model class; skipping to avoid overhead. Set %s=1 to "
            "force-apply if you know what you are doing.",
            patch_id, compat_reason, env_flag,
        )
        return False, compat_reason

    # default_on=True patches still consult config_detect
    try:
        from vllm._genesis.config_detect import recommend
        verdict, reason = recommend(patch_id)
        return (verdict in ("apply", "neutral")), f"config_detect: {verdict}:{reason}"
    except Exception as e:
        return False, f"config_detect failed: {e}"


# ─── Decision logging ─────────────────────────────────────────────────────

# Module-level cache of decisions made this boot, for matrix dump.
_DECISIONS: list[dict[str, Any]] = []


def log_decision(patch_id: str, applied: bool, reason: str) -> None:
    """Log + record a patch decision for the boot-time matrix dump.

    Single condensed line per patch. Operator can see all decisions at boot
    via `Genesis Dispatcher v2 decisions:` log block (called from apply_all).
    """
    meta = PATCH_REGISTRY.get(patch_id, {})
    title = meta.get("title", patch_id)
    status = "APPLY" if applied else "SKIP "
    log.info(
        "[Genesis Dispatcher] %s %s — %s | %s",
        status, patch_id, title, reason[:120],
    )
    _DECISIONS.append({
        "patch_id": patch_id,
        "title": title,
        "applied": applied,
        "reason": reason,
        "env_flag": meta.get("env_flag", ""),
        "credit": meta.get("credit", ""),
        "upstream_pr": meta.get("upstream_pr"),
    })


def get_apply_matrix() -> list[dict[str, Any]]:
    """Return the recorded apply matrix for this boot.

    Useful for tests + diagnostic dump.
    """
    return list(_DECISIONS)


def dump_apply_matrix() -> str:
    """Format the apply matrix as ASCII table (string for printing).

    Columns: patch_id, status, title, reason (truncated), credit.
    """
    if not _DECISIONS:
        return "(no decisions recorded — Genesis Dispatcher hasn't been used yet)"

    # Compute column widths
    rows = [
        (
            d["patch_id"],
            "APPLY" if d["applied"] else "SKIP",
            d["title"][:45],
            d["reason"][:60],
            d.get("credit", "")[:30],
        )
        for d in _DECISIONS
    ]
    widths = [max(len(r[i]) for r in rows) for i in range(5)]
    widths = [max(w, len(h)) for w, h in zip(widths,
              ["Patch", "Status", "Title", "Reason", "Credit"])]

    def _fmt_row(r):
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(r))

    lines = []
    lines.append(_fmt_row(["Patch", "Status", "Title", "Reason", "Credit"]))
    lines.append("-+-".join("-" * w for w in widths))
    for r in rows:
        lines.append(_fmt_row(r))
    return "\n".join(lines)


def log_apply_matrix() -> None:
    """Emit the apply matrix as a multi-line INFO block.

    Called by apply_all at end of boot to give operator a single readable
    summary instead of grep-ing through scattered INFO lines.
    """
    matrix = dump_apply_matrix()
    log.info(
        "[Genesis Dispatcher v2] apply matrix:\n%s",
        matrix,
    )


# ─── A3/D2 — PATCH_REGISTRY dependency / conflict validator ───────────────
# Two layers:
#   1. validate_registry()      — static structural check (boot-time)
#   2. validate_apply_plan(set) — runtime check on actual decisions
#
# Patch metadata may declare:
#   "requires_patches": ["P60"]      — list of patch_ids that must also apply
#   "conflicts_with":   ["P65"]      — list of patch_ids that MUST NOT apply
# Both fields default to [] when absent (no relationship declared).


def _coerce_list(value: Any) -> list[str]:
    """Normalize a metadata field into list[str]. Tolerates None / scalar."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return []


def validate_registry(
    registry: dict[str, dict[str, Any]] | None = None,
) -> list[ValidationIssue]:
    """Static validation of PATCH_REGISTRY shape.

    Checks (per declared `requires_patches` / `conflicts_with` field):
      - Every referenced id exists in the registry (else ERROR — typo class)
      - No patch references itself (else ERROR — clear bug)
      - No requires-cycle A→…→A (else ERROR — boot would loop or wedge)

    Returns a list of `ValidationIssue` (empty list = clean).
    """
    if registry is None:
        registry = PATCH_REGISTRY

    issues: list[ValidationIssue] = []
    keys = set(registry.keys())

    # 1. Reference existence + self-reference
    for pid, meta in registry.items():
        for ref in _coerce_list(meta.get("requires_patches")):
            if ref == pid:
                issues.append(ValidationIssue(
                    "ERROR", pid,
                    f"requires_patches contains self-reference {ref!r}",
                ))
            elif ref not in keys:
                issues.append(ValidationIssue(
                    "ERROR", pid,
                    f"requires_patches references unknown patch_id {ref!r}",
                ))
        for ref in _coerce_list(meta.get("conflicts_with")):
            if ref == pid:
                issues.append(ValidationIssue(
                    "ERROR", pid,
                    f"conflicts_with contains self-reference {ref!r}",
                ))
            elif ref not in keys:
                issues.append(ValidationIssue(
                    "ERROR", pid,
                    f"conflicts_with references unknown patch_id {ref!r}",
                ))

    # 2. Cycle detection on requires_patches graph (DFS three-color).
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {pid: WHITE for pid in registry}

    def _walk(pid: str, path: list[str]) -> None:
        if color[pid] == GRAY:
            cycle = path[path.index(pid):] + [pid]
            issues.append(ValidationIssue(
                "ERROR", pid,
                f"requires_patches cycle detected: {' → '.join(cycle)}",
            ))
            return
        if color[pid] == BLACK:
            return
        color[pid] = GRAY
        for ref in _coerce_list(registry.get(pid, {}).get("requires_patches")):
            if ref in color:
                _walk(ref, path + [pid])
        color[pid] = BLACK

    for pid in list(registry):
        if color[pid] == WHITE:
            _walk(pid, [])

    return issues


def validate_apply_plan(
    applied: set[str],
    registry: dict[str, dict[str, Any]] | None = None,
) -> list[ValidationIssue]:
    """Runtime validation: given the live APPLY set, surface dependency /
    conflict violations.

    Args:
        applied: set of patch_ids that the dispatcher actually decided to
            APPLY this boot (from `get_apply_matrix()` filtered by
            applied=True, or computed externally).
        registry: optional override for testing; defaults to PATCH_REGISTRY.

    Returns:
        list of ValidationIssue. Severities:
          - ERROR  : missing required, conflict-pair both applied
          - WARNING: applied set contains a patch_id not in registry

    Conflict pairs are reported once (canonicalized — sorted ids) even when
    the conflict is declared symmetrically on both sides.
    """
    if registry is None:
        registry = PATCH_REGISTRY

    issues: list[ValidationIssue] = []

    # Unknown ids in applied set
    for pid in applied:
        if pid not in registry:
            issues.append(ValidationIssue(
                "WARNING", pid,
                f"applied set contains unknown patch_id {pid!r}",
            ))

    # Required dependencies — only check patches that ARE applied
    for pid in applied:
        meta = registry.get(pid)
        if meta is None:
            continue  # already reported as unknown above
        for ref in _coerce_list(meta.get("requires_patches")):
            if ref not in applied:
                issues.append(ValidationIssue(
                    "ERROR", pid,
                    f"missing required dependency: {pid} requires {ref!r} "
                    f"to also be APPLY (currently SKIP)",
                ))

    # Conflicts — canonicalize pairs to avoid double-reporting
    seen_pairs: set[tuple[str, str]] = set()
    for pid in applied:
        meta = registry.get(pid)
        if meta is None:
            continue
        for ref in _coerce_list(meta.get("conflicts_with")):
            if ref in applied:
                pair = tuple(sorted([pid, ref]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                issues.append(ValidationIssue(
                    "ERROR", pid,
                    f"conflict: {pair[0]} and {pair[1]} are both APPLY but "
                    f"declared mutually exclusive — pick one",
                ))

    return issues


def log_validation_issues(issues: list[ValidationIssue]) -> None:
    """Emit issues at appropriate log severity. Operator-readable summary."""
    if not issues:
        log.info("[Genesis Dispatcher v2] validator: clean (no issues)")
        return
    for i in issues:
        msg = f"[Genesis Dispatcher v2] validator {i.severity}: {i.patch_id} — {i.message}"
        if i.severity == "ERROR":
            log.error(msg)
        elif i.severity == "WARNING":
            log.warning(msg)
        else:
            log.info(msg)


# ─── CLI entry-point ──────────────────────────────────────────────────────

def main() -> int:
    """`python3 -m vllm._genesis.dispatcher` — print apply matrix as table.

    Useful for diagnostics OUTSIDE a vllm boot (e.g. dry-run profiling).
    Walks the patch registry and dry-evaluates should_apply() for each.
    """
    print("Genesis Dispatcher v2 — patch decision matrix")
    print("=" * 80)
    print()

    # Run should_apply against every registered patch
    for patch_id in PATCH_REGISTRY:
        decision, reason = should_apply(patch_id)
        log_decision(patch_id, decision, reason)

    print(dump_apply_matrix())
    print()
    print("Note: this is a STATIC dispatch view. Some recommendations are")
    print("'skip' because vllm config isn't set in this dry-run context.")
    print("In real boot, get_runtime_profile() returns the actual config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
