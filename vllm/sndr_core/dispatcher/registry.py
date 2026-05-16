# SPDX-License-Identifier: Apache-2.0
"""SNDR Core dispatcher — PATCH_REGISTRY data (single source of truth).

This file is DATA-ONLY. It exceeds the 300-LOC cap on logic files
because it's a registry of all known patches with their metadata
(2000+ lines of dict entries). The cap exemption was approved by
Sander on 2026-05-07 (Idea 2 of Stage 2 acceptance) — data files
that grow with the patch count are not covered by the structural
LOC limit. Logic files MUST stay under 300 LOC.

Each entry declares:
  - title: short human-readable description
  - env_flag: env var that toggles this patch (without prefix; the
    SNDR_ENABLE_/GENESIS_ENABLE_ prefix is added at lookup time)
  - default_on: bool — whether the patch applies without explicit env
  - category: family bucket (deprecated by Stage 6 family taxonomy)
  - tier: "community" | "engine"  (added at Stage 5)
  - lifecycle: "active" | "deprecated" | "retired" | "legacy"
  - credit: provenance + design notes
  - upstream_pr: int | None (vllm-project/vllm PR number, if applicable)
  - applies_to: dict of profile-key → allowed values (model_class etc.)
  - requires_patches / conflicts_with: dependency declarations
  - composes_with: informational soft-link to related patches
  - apply_module: dotted path to the module providing apply()
    (added at Stage 6 reorg)

Migration history:
  - Original location: vllm/_genesis/dispatcher.py (2828-LOC monolith).
  - Stage 3 (CURRENT): split out as data-only module here.
"""
from __future__ import annotations

from typing import Any


# ─── Patch metadata registry ───────────────────────────────────────────────
# Each patch declares what it touches + which env flag enables/disables it.
# This is the SINGLE source of truth for patch-to-feature mapping.

PATCH_REGISTRY: dict[str, dict[str, Any]] = {
    # P56 + P57: archived 2026-05-05 to
    # ../Genesis_internal_docs/_archive/dead_patches/p56_p57_tq_specdec_deadends/
    # P56 = TQ spec-decode safe-path guard (superseded by P65)
    # P57 = TQ spec-decode capture-safe buffers (~1080 MiB regression, dead-end)
    # See archive README for the full investigation thread.
    "P58": {
        "title": "Async-scheduler -1 placeholder fix",
        "tier": "community",
        "family": "scheduler",
        "env_flag": "GENESIS_ENABLE_P58_ASYNC_PLACEHOLDER_FIX",
        "default_on": False,
        "category": "spec_decode",
        "credit": "z1ying (vllm#40768)",
        "upstream_pr": 40768,
        "apply_module": "vllm.sndr_core.integrations.scheduler.p58_async_scheduler_placeholder_fix",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P59": {
        "title": "Qwen3 reasoning embedded tool_call recovery",
        "tier": "community",
        "family": "reasoning",
        "env_flag": "GENESIS_ENABLE_P59_QWEN3_TOOL_RECOVERY",
        "default_on": False,
        "category": "structured_output",
        "credit": "ZenoAFfectionate (vllm#39055)",
        "upstream_pr": 39055,
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_6", "qwen3_moe", "qwen3_next"]},
        "apply_module": "vllm.sndr_core.integrations.reasoning.p59_qwen3_reasoning_tool_call_recovery",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P60": {
        "title": "GDN+ngram state recovery (Phase 1: SSM pre-copy)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_P60_GDN_NGRAM_FIX",
        "default_on": False,
        "category": "spec_decode",
        "credit": "tdoublep (vllm#40738), bhaktatejas922 (#39273)",
        "upstream_pr": 40738,
        "applies_to": {"is_hybrid": [True]},
        "apply_module": "vllm.sndr_core.integrations.attention.gdn.p60_gdn_ngram_state_recovery",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P60b": {
        "title": "GDN+ngram Triton kernel offset (Phase 2)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_P60B_TRITON_KERNEL",
        "default_on": False,
        "category": "spec_decode",
        "credit": "tdoublep (vllm#40738)",
        "upstream_pr": 40738,
        "applies_to": {"is_hybrid": [True]},
        "requires_patches": ["P60"],
        "apply_module": "vllm.sndr_core.integrations.attention.gdn.p60b_gdn_ngram_triton_kernel",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P61": {
        "title": "Qwen3 multi-tool first-occurrence (RETIRED — fully superseded by P12 v2)",
        "tier": "community",
        "family": "reasoning",
        "env_flag": "GENESIS_ENABLE_P61_QWEN3_MULTI_TOOL",
        "default_on": False,
        "lifecycle": "retired",  # migrated from `deprecated: True` to explicit retired lifecycle. P12 v2 directly emits FIRST-occurrence; P61 anchor never matched in active form. Safety-gated.
        "category": "structured_output",
        "credit": "ExtReMLapin (vllm#40783) — P61 was supposed to flip P12's LAST-occurrence to FIRST via post-anchor replacement, but its anchor 'tool_call_index = ...' never matched P12-emitted 'idx = ...' form, so it silent-skipped when P12 was active. v7.62.5 (2026-04-28): P12 emit updated to FIRST directly; P61 retired. Setting GENESIS_ENABLE_P61=1 is now a harmless no-op (anchor not found vs already-FIRST P12 output).",
        "upstream_pr": 40783,
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_6", "qwen3_moe", "qwen3_next"]},
        "apply_module": "vllm.sndr_core.integrations._retired.p61_qwen3_multi_tool_first_occurrence",
        "retired_waiver": True,  # P12 v2 directly emits FIRST-occurrence; P61 anchor never matched in active form. Harmless no-op.
        "implementation_status": "full",
    },
    "P62": {
        "title": "Structured-output spec-decode reasoning-end timing fix",
        "tier": "community",
        "family": "serving",
        "env_flag": "GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING",
        "default_on": False,
        "category": "structured_output",
        "credit": "sfbemerk (vllm#36138), cicirori (vllm#34650)",
        "upstream_pr": 36138,
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_6", "qwen3_moe", "qwen3_next"]},
        "conflicts_with": ["PN58"],
        "apply_module": "vllm.sndr_core.integrations.serving.p62_structured_output_spec_decode_timing",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P61c": {
        "title": "Qwen3Coder deferred-commit until <function= header (club-3090#72)",
        "tier": "community",
        "family": "tool_parsing",
        "env_flag": "GENESIS_ENABLE_P61C_QWEN3CODER_DEFERRED_COMMIT",
        "default_on": False,
        "category": "structured_output",
        "apply_module": "vllm.sndr_core.integrations.tool_parsing.p61c_qwen3coder_deferred_commit",
        "credit": (
            "Local mitigation for club-3090 issue #72 (troymroberts 2026-05-06). "
            "qwen3coder parser flips is_tool_call_started=True permanently on "
            "EITHER tool_call_start_token_id in delta_token_ids OR string match "
            "against `<tool_call>` in delta_text. Both trigger paths mis-fire "
            "when narrative output contains `<tool_call>` (e.g. agent reasoning "
            "describing the protocol). The flip is sticky: subsequent deltas "
            "all return None and vLLM serving layer drops them via "
            "`if delta_message is None: continue` — SSE wire goes silent for "
            "30-120+ s until max_tokens. Fix V2 (deferred): commit only after "
            "`<function=` header arrives in 64-char slack window; otherwise "
            "emit delta as content. Three paths (tokenizer-edge / confirmed / "
            "uncertain) all handled non-silently. Composes with P64 "
            "(vllm#39598) and PN56 (vllm#41466) — all touch same file but "
            "different sub-blocks. Default OFF until live verify on 27B PROD."
        ),
        "upstream_pr": None,  # club-3090 issue, not yet upstream PR
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_6", "qwen3_moe", "qwen3_next"]},
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P61b": {
        "title": "Qwen3 streaming partial-tag overlap guard",
        "tier": "community",
        "family": "reasoning",
        "env_flag": "GENESIS_ENABLE_P61B_STREAMING_OVERLAP",
        "default_on": False,
        "category": "structured_output",
        "credit": "ExtReMLapin (vllm#40783)",
        "upstream_pr": 40783,
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_6", "qwen3_moe", "qwen3_next"]},
        "apply_module": "vllm.sndr_core.integrations.reasoning.p61b_qwen3_streaming_overlap_guard",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P63": {
        "title": "MTP/Eagle drafter GDN state recovery (RETIRED — hypothesis disproven 2026-04-25)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_P63_MTP_GDN_STATE_RECOVERY",
        "default_on": False,
        "lifecycle": "retired",  # migrated from `deprecated: True`. Hypothesis disproven empirically; safety-gated.
        "retired_waiver": True,  # Genesis-original hypothesis disproven 2026-04-25; no upstream supersession (not a backport)
        "apply_module": "vllm.sndr_core.integrations._retired.p63_mtp_gdn_state_recovery",
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
        "implementation_status": "full",
    },
    "P64": {
        "title": "qwen3coder MTP streaming early-return fix",
        "tier": "community",
        "family": "tool_parsing",
        "env_flag": "GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING",
        "default_on": False,
        "category": "structured_output",
        "credit": "kotori-yan (vllm#39598)",
        "upstream_pr": 39598,
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_6", "qwen3_moe", "qwen3_next"]},
        "apply_module": "vllm.sndr_core.integrations.tool_parsing.p64_qwen3coder_mtp_streaming",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P65": {
        "title": "TurboQuant spec-decode cudagraph downgrade (FALLBACK — see P67 root-cause fix)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_P65_TURBOQUANT_SPEC_CG_DOWNGRADE",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Genesis-original (root cause for noonghunna #40880). Fallback safety-net workaround that downgrades cudagraph PIECEWISE → ~30% TPS hit. SUPERSEDED by P67/P67b root-cause fix (TurboQuant multi-query kernel for spec-decode K+1 verify). Default OFF; opt-in only when P67 unavailable or unstable.",
        "upstream_pr": None,
        "applies_to": {"is_turboquant": [True]},
        "conflicts_with": ["P67", "P67b"],
        # NOTE: P67/P67b is the root-cause fix (multi-query kernel) and
        # P65 is the safety-net fallback. Relationship explained in
        # `credit`. Not using `superseded_by` because P65 has no pin-gate
        # boundary (it's a runtime fallback choice, not version retire).
        "apply_module": "vllm.sndr_core.integrations.attention.turboquant.p65_turboquant_spec_cg_downgrade",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P66": {
        "title": "cudagraph_capture_sizes spec-decode divisibility filter",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Genesis-original (mirrors fhl2000 vllm#23679 closed)",
        "upstream_pr": 23679,
        "apply_module": "vllm.sndr_core.integrations.compile_safety.p66_cudagraph_size_divisibility_filter",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P68": {
        "title": "Auto force tool_choice=required for long-context tool calls",
        "tier": "community",
        "family": "serving",
        "env_flag": "GENESIS_ENABLE_P68_AUTO_FORCE_TOOL",
        "default_on": False,
        "category": "structured_output",
        "credit": "Genesis-original (long-ctx tool adherence mitigation)",
        "upstream_pr": None,
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_6", "qwen3_moe", "qwen3_next"]},
        "apply_module": "vllm.sndr_core.integrations.serving.p68_69_long_ctx_tool_adherence",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P69": {
        "title": "Long-context tool-format reminder injection",
        "tier": "community",
        "family": "serving",
        "env_flag": "GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER",
        "default_on": False,
        "category": "structured_output",
        "credit": "Genesis-original (long-ctx tool adherence mitigation)",
        "upstream_pr": None,
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_6", "qwen3_moe", "qwen3_next"]},
        "apply_module": "vllm.sndr_core.integrations.serving.p68_69_long_ctx_tool_adherence",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P70": {
        "title": "Auto-strict-ngram (force prompt_lookup_min>=8)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_P70_AUTO_STRICT_NGRAM",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Genesis-original (vllm#40875 enforcement)",
        "upstream_pr": None,
        "applies_to": {
            # [Genesis pin-gate 2026-05-11] PROD-active (GroupAB component
            # +9.2% on 27B). Validated dev9 → dev93.
            "vllm_version_range": (">=0.20.0", "<0.21.0"),
        },
        "apply_module": "vllm.sndr_core.integrations.spec_decode.p70_auto_strict_ngram",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN72": {
        "title": "Frequency-based ngram draft post-filter (llama.cpp-style)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN72_FREQUENCY_NGRAM_DRAFTER",
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Genesis-original 2026-05-06. Mirrors llama.cpp's "
            "draft_min_sample_size + draft_min_percent heuristic from "
            "common/ngram-cache.cpp — but as POST-filter on vllm's "
            "longest-match ngram drafter (because vllm uses numba JIT, "
            "text-patching it is risky). Wraps NgramProposer.propose, "
            "rejects drafts whose first token has < MIN_OBS occurrences "
            "in the recent window. Tunables: "
            "GENESIS_PN72_MIN_OBSERVATIONS (default 4), "
            "GENESIS_PN72_FREQUENCY_WINDOW (default 1024). Composable with "
            "P70 (orthogonal — both can be on for stricter drafting). "
            "Goal: reject spurious draft-acceptances from chat-template "
            "ambiguity (Qwen3-coder `<<` 2-token-suffix bug class — see "
            "Genesis v7.13 BREAKTHROUGH). Safety: graceful fallback to "
            "unfiltered drafts on any internal error."
        ),
        "upstream_pr": None,
        "apply_module": "vllm.sndr_core.integrations.spec_decode.pn72_frequency_ngram_drafter",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN77": {
        "title": "FP8 lm_head compression (BF16→FP8 e4m3 + per-channel scale)",
        "tier": "community",
        "family": "quantization",
        "env_flag": "GENESIS_ENABLE_PN77_FP8_LM_HEAD",
        "default_on": False,
        "category": "memory_savings",
        "credit": (
            "Genesis-original 2026-05-07 (Phase E MVP). Backport pattern of "
            "vllm PR #35696 (lucaspirola, OPEN) + PR #35694 (FP8 weight "
            "storage in unquantized linear/embedding apply). Compresses "
            "lm_head BF16 weight to float8_e4m3fn with per-channel scale. "
            "Saves ~606 MiB/rank on 27B Qwen3.6 (vocab=248320, hidden=5120), "
            "~243 MiB/rank on 35B (hidden=2048). MVP forward path: cast-back "
            "to BF16 on each call (~3 ms/token, absorbed by spec-decode "
            "latency). Phase E.3+ will replace with apply_fp8_marlin_linear "
            "for weight-only FP8 GEMM (no per-call cast). Quality gate: "
            "cosine_sim ≥ 0.999 verified in unit tests; integration A/B "
            "(tool-call 10/10 + reasoning probe) required before promoting. "
            "Tied-embedding detection prevents poisoning embed_tokens."
        ),
        "upstream_pr": None,  # PR #35696 OPEN; Genesis is preemptive backport
        "apply_module": "vllm.sndr_core.integrations.quantization.pn77_fp8_lm_head",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN80": {
        "title": "LoRA tensorizer device kwarg (vllm#41845 backport)",
        "tier": "community",
        "family": "lora",
        "env_flag": "GENESIS_ENABLE_PN80_LORA_TENSORIZER_DEVICE",
        "default_on": False,
        "lifecycle": "retired",  # 2026-05-11 v2 audit: byte-equivalent in dev209
        "vllm_version_range": ">=0.20.1rc1.dev16+g7a1eb8ac2,<0.20.2rc1.dev209+g5536fc0c0",  # active before upstream merge in dev209
        "apply_module": "vllm.sndr_core.integrations._retired.pn80_lora_tensorizer_device",
        "category": "memory_savings",
        "credit": (
            "Backport of vllm#41845 (Or Ozeri @ IBM, MERGED 2026-05-07 main "
            "HEAD — not in nightly image as of dev93+g51f22dcfd). Single-line "
            "fix: pass `device=device` to TensorDeserializer in "
            "lora/lora_model.py LoRA loading path. Without device kwarg, "
            "tensorizer first deserializes to host RAM (full tensor size, "
            "potentially 2-50 GB depending on LoRA rank), then transfers to "
            "GPU — peak host RAM blows up causing OOM on memory-constrained "
            "rigs. With kwarg, tensorizer streams directly to GPU. Genesis "
            "35B/27B PROD currently не использует LoRA — patch ready на "
            "случай community deployments или Sander запустит LoRA workload."
        ),
        "upstream_pr": 41845,
        "superseded_by": "vllm#41845 (merged 2026-05-07, byte-identical in dev209 lora_model.py:206 has `device=device` arg in TensorDeserializer call)",
        "applies_to": {
            # [Iron rule #11 retire 2026-05-11 v2 audit] PN80 was
            # never deployed on Genesis PROD (no LoRA workload). On
            # dev209 deep-diff verified: lora_model.py:203-206 has
            # the exact `device=device` arg PN80 was backporting.
            "vllm_version_range": "<0.20.2rc1.dev93",
        },
        "requires_patches": [],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN79": {
        "title": "In-place SSM state for GDN chunk prefill (vllm#41824 backport)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN79_INPLACE_SSM_STATE",
        "default_on": False,
        "apply_module": "vllm.sndr_core.integrations.attention.gdn.pn79_inplace_ssm_state",
        "lifecycle": "experimental",
        "category": "memory_savings",
        "credit": (
            "Backport of vllm#41824 (Kermit-C, OPEN as of 2026-05-06). "
            "Eliminates per-decode-step gather/scatter copies of "
            "initial_state and final_state in chunk_gated_delta_rule_fwd_h "
            "by passing ssm_state_indices directly to the Triton kernel. "
            "Author claims 4.5-36 GiB cumulative fp32 traffic eliminated "
            "per multi-turn session (Qwen3.5-0.8B → Qwen3.6-27B scale). "
            "ORTHOGONAL TO PN59: PN59 fixes prefill peak (h-allocation), "
            "PN79 fixes decode steady-state allocator pressure. Empirical "
            "Genesis 2026-05-06: PN59 _streaming_path almost never fires "
            "in real workloads (chunked-prefill T=64 chunks always "
            "bypass T<1024 gate; multi-seq batches always bypass single-seq "
            "guard) — therefore PN79 has higher actual ROI on our stack. "
            "Status 2026-05-07: FULL IMPLEMENTATION LANDED + LIVE A/B "
            "VALIDATED — 17 anchors across 3 files (Sub-1 chunk.py: 7 "
            "[1A/1B/1C/1D + 1E_SIG/1E_VAL/1E_APPLY_CALL high-level API], "
            "Sub-2 chunk_delta_h.py: 7 [2A heuristics, 2B kernel sig, 2C "
            "kernel main, 2D kernel epilogue, 2E wrapper sig, 2F wrapper "
            "body strides, 2G wrapper kernel call], Sub-3 gdn_linear_attn.py: "
            "3 [3A forward_cuda fallback, 3B forward_native passthrough, "
            "3C _forward_core gather/scatter elim]). All 17 OLD anchors "
            "match server live source (vllm pin dev60+ge47c98ef7) uniquely; "
            "pristine dev9 chunk_delta_h.py is bit-identical to live dev60 "
            "(kernel unchanged across 51 dev versions). Live boot 27B "
            "Lorbus INT4+TQ k8v4 successful (135-155s); GDN warmup all 32 "
            "layers clean. A/B bench 27B 2026-05-07: TPS +1.1% within "
            "noise (105.3 vs 104.2), VRAM identical (45485 MiB), tool 10/10 "
            "match — single-shot win unproven, multi-turn evidence pending. "
            "58 PN79 tests + 2260 full Genesis suite pass on Mac. Atomic "
            "3-file MultiFilePatchTransaction + conflicts_with [PN59, "
            "PN54] guard. Default OFF, lifecycle experimental. PN59/PN54 "
            "lifecycle migration pending Stage 4 multi-turn evidence "
            "(currently both stable in registry; PN79 docstring describes "
            "intended final state, not current registry state)."
        ),
        "upstream_pr": 41824,
        "applies_to": {"is_hybrid": [True]},
        "requires_patches": [],
        "conflicts_with": ["PN59", "PN54"],
        "implementation_status": "full",
    },
    "PN78": {
        "title": "[RETIRED] One-shot empty_cache() after CG warmup",
        "tier": "community",
        "family": "memory",
        "env_flag": "GENESIS_ENABLE_PN78_POST_WARMUP_CACHE_RELEASE",
        "default_on": False,
        "lifecycle": "retired",  # migrated from "deprecated" — upstream pin handles cache release internally; this wrap is permanent no-op.
        "vllm_version_range": ">=0.20.1rc1.dev16+g7a1eb8ac2,<0.20.2rc1.dev9+g01d4d1ad3",  # active before upstream pin handles cache release
        "apply_module": "vllm.sndr_core.integrations._retired.pn78_post_warmup_cache_release",
        "category": "memory_savings",
        "credit": (
            "DEPRECATED 2026-05-07: see deprecation_note. Patch retained "
            "for documentation; env flag honored but executes no-op."
        ),
        "deprecation_note": (
            "Investigation 2026-05-07 (MEMORY_DEEP_PLAN Этап 2.1): vllm "
            "pin already calls torch.accelerator.empty_cache() inside "
            "GPUModelRunner.capture_model (gpu_model_runner.py:6213 "
            "BEFORE capture, :6244 AFTER capture, before lock_workspace). "
            "PN78 wrap would be redundant 3rd call. Additionally, "
            "apply_all()-time monkey patches do NOT reach worker "
            "processes (VLLM_WORKER_MULTIPROC_METHOD=spawn → fresh "
            "interp). Genesis pattern for worker-visible patches is "
            "source-level edits to vllm core (e.g. PN59 in "
            "fla/ops/chunk.py). No upstream PR needed — upstream is "
            "already correct."
        ),
        "upstream_pr": None,
        "implementation_status": "retired",
        "superseded_by": "upstream torch.accelerator.empty_cache() calls in GPUModelRunner.capture_model (gpu_model_runner.py:6213/6244 in dev9+) — PN78 would be redundant 3rd call",
        "applies_to": {
            # [Iron rule #11 formal retire 2026-05-11] Promoted from
            # waiver — deprecation_note identified specific supersession
            # points in upstream code. PN78 wrap permanent no-op since
            # dev9+; pin-gate documents the boundary.
            "vllm_version_range": "<0.20.2rc1.dev9",
        },
    },
    "P67": {
        "title": "TurboQuant multi-query kernel for spec-decode K+1",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Genesis-original (proper fix for noonghunna #40880; replaces P65 workaround). 2026-05-05 NOTE: alternative upstream fix is OPEN as vllm#40914 (Sandermage) — uses synth_seq_lens routing through existing decode kernel instead of new Genesis-original kernel. If #40914 merges, P67 becomes one of two equivalent paths; defer retirement decision until empirical TPS A/B (P67 currently delivers +32% on 35B-A3B-FP8 PROD vs upstream baseline). Watch_for_drift_via vllm#40914.",
        "upstream_pr": None,
        "applies_to": {
            "is_turboquant": [True],
            # [Genesis pin-gate 2026-05-11] PROD-active patch (35B +32% TPS).
            # Validated dev16 → dev93. Broad range; drift detector handles
            # anchor-line breakage on bumps.
            "vllm_version_range": (">=0.20.0", "<0.21.0"),
        },
        "conflicts_with": ["P65"],
        "apply_module": "vllm.sndr_core.integrations.attention.turboquant.p67_tq_multi_query_kernel",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P67b": {
        "title": "TurboQuant spec-verify forward() routing (FULL CG enable)",
        "tier": "community",
        "family": "attention.turboquant",
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
        "apply_module": "vllm.sndr_core.integrations.attention.turboquant.p67b_spec_verify_routing",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P72": {
        "title": "profile_run M cap (unblocks --max-num-batched-tokens>4096 on MoE)",
        "tier": "community",
        "family": "worker",
        "env_flag": "GENESIS_ENABLE_P72_PROFILE_RUN_CAP",
        "default_on": False,
        "category": "compile_safety",
        "credit": "Genesis-original (Dynamo fake-tensor mismatch workaround for moe_align_block_size symbolic shape)",
        "upstream_pr": None,
        "applies_to": {
            # [Genesis pin-gate 2026-05-11] PROD-active (caps profile_run
            # M to GENESIS_PROFILE_RUN_CAP_M, unblocks batched_tokens>4096
            # on MoE). Validated dev9 → dev93.
            "vllm_version_range": (">=0.20.0", "<0.21.0"),
        },
        "apply_module": "vllm.sndr_core.integrations.worker.p72_profile_run_cap",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P71": {
        "title": "Block-verify rejection sampler (Sun 2024 ICLR)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_P71_BLOCK_VERIFY",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport of vllm#40819 (Z. Golpayegani draft) + Sun et al. arXiv 2403.10444 + 2 critical fixes from gemini-code-assist review (shared u per request, denom==0 → 1.0)",
        "upstream_pr": 40819,
        "apply_module": "vllm.sndr_core.integrations.spec_decode.p71_block_verify",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P74": {
        "title": "Auto chunk-clamp via long_prefill_token_threshold (P72 companion)",
        "tier": "community",
        "family": "scheduler",
        "env_flag": "GENESIS_ENABLE_P74_CHUNK_CLAMP",
        "default_on": False,
        "category": "compile_safety",
        "credit": "Genesis-original (zero-VRAM-cost prealloc-overflow safety net for P72-unblocked batched_tokens>4096)",
        "upstream_pr": None,
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_6", "qwen3_moe", "qwen3_next"]},
        "requires_patches": ["P72"],
        "apply_module": "vllm.sndr_core.integrations.scheduler.p74_chunk_clamp",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P75": {
        "title": "Auto-enable Suffix Decoding (Arctic Inference, vllm#25784)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_P75_SUFFIX_DECODING",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport-enabler of vllm#25784 (Arctic Inference Suffix Decoding) — operator convenience: auto-swap method=ngram→suffix when env enabled. Algorithm: arxiv 2411.04975.",
        "upstream_pr": 25784,
        "enables_upstream_feature": True,
        # [Iron rule #11 audit 2026-05-11 v2] P75 is NOT a backport —
        # it's a convenience activator ON TOP of merged upstream feature
        # (#25784 in pin since 2025-11). Audit script honors
        # `enables_upstream_feature: True` to exclude from NEWLY-MERGED
        # categorization. KEEP active — convenience value preserved.
        "apply_module": "vllm.sndr_core.integrations.spec_decode.p75_suffix_decoding_enable",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P77": {
        "title": "Adaptive ngram K controller (EMA + hysteresis + auto-disable)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_P77_ADAPTIVE_NGRAM_K",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Genesis-original (port of SGLang adaptive_spec_params.py EMA+hysteresis Apache-2.0 + Nightjar arXiv 2512.22420 auto-disable extension). Targets free-form ngram pathology (46 tok/s).",
        "upstream_pr": None,
        "apply_module": "vllm.sndr_core.integrations.spec_decode.p77_adaptive_ngram_k",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P78": {
        "title": "TurboQuant .tolist() capture-guard (adapted from noonghunna)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_P78_TOLIST_CAPTURE_GUARD",
        "default_on": False,
        "category": "compile_safety",
        "credit": "Adapted from noonghunna's patch_tolist_cudagraph.py (Apache-2.0, github.com/noonghunna/qwen36-27b-single-3090). Surgical safety-net for cudagraph capture; complements our P22/P26/P44 prealloc.",
        "upstream_pr": None,
        "applies_to": {
            "is_turboquant": [True],
            "quant_format": ["fp8", "compressed_tensors"],
        },
        "apply_module": "vllm.sndr_core.integrations.attention.turboquant.p78_tolist_capture_guard",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P79b": {
        "title": "Async × spec-decode proposer-sync backport (vllm#40610)",
        "tier": "community",
        "family": "worker",
        "env_flag": "GENESIS_ENABLE_P79B_ASYNC_PROPOSER_SYNC",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport of vllm#40610 (OPEN draft, tracked from #40608). Re-records prepare_inputs_event AFTER spec-decode proposer GPU work in sample_tokens(). Fixes async × spec-decode race where next batch _update_states could mutate block_table while previous batch's proposer was still reading on GPU. Genesis prod uses sync ngram so direct value is minimal; protects users on async+EAGLE/MTP/ngram_gpu.",
        "upstream_pr": 40610,
        "apply_module": "vllm.sndr_core.integrations.worker.p79b_async_proposer_sync",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P79c": {
        "title": "Stale spec_token_ids cleanup for unscheduled requests (vllm#37629)",
        "tier": "community",
        "family": "scheduler",
        "env_flag": "GENESIS_ENABLE_P79C_STALE_SPEC_TOKEN_CLEANUP",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport of vllm#37629 (OPEN, fixes #36906). Cleanup pass after main scheduling loop clears spec_token_ids for unscheduled running requests. Prevents -1 placeholder leak into F.embedding() under budget-exhausted high-concurrency on async + EAGLE/MTP. Genesis prod (max_num_seqs=2, sync ngram) gains nothing direct; protects high-concurrency multimodal users.",
        "upstream_pr": 37629,
        "apply_module": "vllm.sndr_core.integrations.scheduler.p79c_stale_spec_token_cleanup",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P79d": {
        "title": "Preempt async-discard backport (vllm#38624)",
        "tier": "community",
        "family": "scheduler",
        "env_flag": "GENESIS_ENABLE_P79D_PREEMPT_ASYNC_DISCARD",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport of vllm#38624 (CodersAcademy006, OPEN). Adds discard_latest_async_tokens=True + num_output_placeholders=0 to _preempt_request() — fixes silent token duplication ('the the', 'of of') after preemption-resume on async + EAGLE/MTP/ngram_gpu paths. Additive (does NOT remove from reset_prefix_cache like upstream does — defensive). Idempotent. Genesis prod (sync ngram) gains nothing direct; protects async users.",
        "upstream_pr": 38624,
        "apply_module": "vllm.sndr_core.integrations.scheduler.p79d_preempt_async_discard",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P81": {
        "title": "fp8 block-scaled MM low-M decode tuning (vllm#40925)",
        "tier": "community",
        "family": "quantization",
        "env_flag": "GENESIS_ENABLE_P81_FP8_BLOCK_SCALED_M_LE_8",
        "default_on": False,
        "category": "kernel_perf",
        "credit": "Backport of vllm#40925 (tonyliu312, OPEN). Specializes w8a8_triton_block_scaled_mm default config for M<=8 (single-request decode + MTP K=3 verify): BLOCK_SIZE_M 64->16, num_stages 2->3 (non-ROCm). Empirical +23% median decode on GB10. Direct hit for Genesis prod (Qwen3.6-A3B FP8 + max_num_seqs=2 + no pre-tuned JSON for A5000).",
        "upstream_pr": 40925,
        "applies_to": {
            "quant_format": ["fp8"],
        },
        "apply_module": "vllm.sndr_core.integrations.quantization.p81_fp8_block_scaled_m_le_8",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P82": {
        "title": "SGLang threshold_single OR-clause acceptance (BIASED — opt-in research)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_P82",
        "default_on": False,
        "category": "spec_decode",
        "credit": "SGLang team (sgl-project/sglang) speculative_sampling.cuh — port of the threshold_single OR-clause that breaks the structural ceiling clean_rate ≈ accept_rate^num_spec. Targets v7.13 strict-ngram acceptance gap. BIASED rule (loses unbiased-sampling guarantee); requires empirical quality validation before prod. Threshold baked from env GENESIS_P82_THRESHOLD_SINGLE (default 0.3) at server start.",
        "upstream_pr": None,
        "applies_to": {
            # [Genesis pin-gate 2026-05-11] PROD-active (Sprint 1 winner
            # +5.23% TPS at thr=0.1). Validated dev9 → dev93.
            "vllm_version_range": (">=0.20.0", "<0.21.0"),
        },
        "apply_module": "vllm.sndr_core.integrations.spec_decode.p82_sglang_acceptance_threshold",
        "lifecycle": "research",
        "research_note": (
            "BIASED rule — gives up the unbiased-sampling guarantee in "
            "exchange for breaking the clean_rate ≈ accept_rate^num_spec "
            "structural ceiling. Sprint 1 +5.23% TPS at threshold=0.1, "
            "but mathematical bias means output distribution drifts vs "
            "vanilla speculative decoding. Kept as opt-in research; "
            "promotion requires per-workload quality validation (tool-call "
            "score, JSON correctness, reasoning quality) — not just TPS."
        ),
        "implementation_status": "full",
    },
    "P83": {
        "title": "MTP keep-last-cached-block (vllm#38182 downstream symptom — P84 is real fix)",
        "tier": "community",
        "family": "kv_cache",
        "env_flag": "GENESIS_ENABLE_P83",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Root-cause analysis: vllm#38182 by uOnePiece + @Angazenn comment identifying single_type_kv_cache_manager.py:457 force-pop last cached block when use_eagle=True. MTP gets caught up via config/speculative.py:890-891 (use_eagle returns True for 'mtp'). EMPIRICALLY DISPROVEN as the actual cause: Genesis debug instrumentation showed find_longest_cache_hit was NEVER called for our workload because num_hashes=0 (block_size > prompt_len after P5 LCM-pad). The L457 pop is a downstream symptom, not the upstream cause. P84 (hash_block_size override) is the real fix. P83 kept as opt-in research artifact for future workloads where the pop site IS reached.",
        "upstream_pr": None,
        "applies_to": {"is_hybrid": [True]},
        "apply_module": "vllm.sndr_core.integrations.kv_cache.p83_mtp_keep_last_cached_block",
        "lifecycle": "research",
        "research_note": (
            "Empirically disproven as the root cause of vllm#38182: "
            "Genesis instrumentation found `find_longest_cache_hit` was "
            "never called for our workload (num_hashes=0 after P5 "
            "LCM-pad makes block_size > prompt_len). The L457 force-pop "
            "is a downstream symptom; P84 (hash_block_size override) is "
            "the actual upstream fix. P83 kept as opt-in research for "
            "future workloads where the pop site IS reachable (e.g. "
            "longer prompts that produce num_hashes>0 with smaller blocks)."
        ),
        "implementation_status": "full",
    },
    "P84": {
        "title": "hash_block_size override (vllm#38182 actual root cause)",
        "tier": "community",
        "family": "scheduler",
        "env_flag": "GENESIS_ENABLE_P84",
        "default_on": False,
        "category": "kv_cache",
        "credit": "Genesis-original discovery 2026-04-27 via P83 DEBUG instrumentation. scheduler.py:234 hard-codes hash_block_size=self.block_size; on hybrid Qwen3.6-MoE with P5 LCM-pad this becomes 2048+, so request_block_hasher computes 0 hashes for prompts < 2048 tokens. Cache machinery runs with overhead but never produces hits. P84 text-patches scheduler.py to read hash_block_size from env GENESIS_P84_HASH_BLOCK_SIZE (recommended value: 16 = full-attention default). Engage via GENESIS_ENABLE_P84=1 + GENESIS_P84_HASH_BLOCK_SIZE=16. Constraint: must divide every group's block_size, else vLLM's own assertion fires at startup. Related: vllm#38182 identified WRONG root cause (the L457 pop); P84 attacks the upstream cause.",
        "upstream_pr": None,
        "applies_to": {"is_hybrid": [True]},
        "apply_module": "vllm.sndr_core.integrations.scheduler.p84_hash_block_size_override",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P85": {
        "title": "Hybrid fine-shadow prefix cache (vllm#38182 followup, MambaManager fix)",
        "tier": "community",
        "family": "kv_cache",
        "env_flag": "GENESIS_ENABLE_P85",
        "default_on": False,
        "category": "kv_cache",
        "credit": "Genesis-original 2026-04-27 — synthesis of 6-round empirical investigation + deep code analysis. Identified TWO mismatches in hybrid prefix cache: (A) MambaManager.cache_blocks early-returns for prompts < self.block_size (e.g., 1424 < 2048); (B) Mamba align-mode pads with null_blocks so num_full_blocks > 0 still inserts 0 entries. P85 patches MambaManager to: (1) register shadow fine-grained hash entries (scale_factor=block_size/hash_block_size duplicates) when caching, (2) walk fine hashes on lookup with eviction-safety re-derive verify. Memory layout / ref-count untouched. Requires P84 (fine hashes computed). Architectural limit: cannot help prompts < block_size (Mamba state genuinely uncached at sub-block boundaries).",
        "upstream_pr": None,
        "applies_to": {"is_hybrid": [True]},
        "requires_patches": ["P84"],
        "apply_module": "vllm.sndr_core.integrations.kv_cache.p85_hybrid_fine_shadow_prefix_cache",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P86": {
        "title": "ngram batch_propose O(N*K) → O(N+K) direct-fill (vllm#40876)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_P86",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport of vllm#40876 (aaronagent, OPEN). Replaces O(N*K) `i in valid_ngram_requests` membership scan in NgramProposer.batch_propose with O(N+K) direct-fill loop iterating only the valid ngram requests. Algorithmic improvement, no behavioral change. Negligible at Genesis prod max_num_seqs=2 (~ns); meaningful at high-concurrency multi-user serving (e.g. N=64, K=32 saves ~1952 list-membership ops per batch step).",
        "upstream_pr": 40876,
        "apply_module": "vllm.sndr_core.integrations.spec_decode.p86_ngram_batch_propose_linear",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P87": {
        "title": "Marlin W4A16/W8A16 sub-tile output dim pad-on-load (vllm#40361)",
        "tier": "community",
        "family": "kernels",
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
        "apply_module": "vllm.sndr_core.integrations.kernels.p87_marlin_pad_sub_tile",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN8": {
        "title": "MTP/draft online-quant propagation (vllm#40849)",
        "tier": "community",
        "family": "loader",
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
        "apply_module": "vllm.sndr_core.integrations.loader.pn8_mtp_draft_online_quant_propagation",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN9": {
        "title": "Independent drafter attention backend (vllm#39930)",
        "tier": "community",
        "family": "spec_decode",
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
        "superseded_by": "vllm#39930 (merged 2026-04-28, in dev9+) — upstream provides full feature including SpeculativeConfig.attention_backend pydantic field; our PN9 backported only the env-driven subset (less invasive at runtime). Upstream is strictly more capable on dev9+.",
        "vllm_version_range": "<0.20.2rc1.dev9+g01d4d1ad3",  # active before upstream merge in dev9
        "apply_module": "vllm.sndr_core.integrations._retired.pn9_independent_drafter_attn_backend",
        "applies_to": {
            # Patch only takes effect inside _create_draft_vllm_config which
            # is only called when spec-decode is active. No additional gate.

            # [Genesis iron-rule-#11 retire 2026-05-11 v2 audit] #39930
            # MERGED 2026-04-28 → in dev9+. Upstream provides FULL feature
            # (including new SpeculativeConfig.attention_backend pydantic
            # field) which our PN9 deliberately did NOT backport (too
            # invasive at runtime for a frozen dataclass + field_validator).
            # Upstream is strictly more capable on dev9+; PN9 was a stop-gap.
            # Pin-gate upper bound documents the boundary. Wiring auto-skips
            # on dev9+. Cleanup: delete patch file in next refactor pass.
            "vllm_version_range": "<0.20.2rc1.dev9",
        },
        "lifecycle": "retired",  # 2026-05-11 v2 audit: upstream more capable, soft-retire formalized
        "implementation_status": "full",
    },
    "PN38": {
        "title": "DFlash drafter quantization support (PR #40425 backport)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN38_DFLASH_QUANT_DRAFTER",
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Backport of vllm-project/vllm#40425 (infatoshi, OPEN). "
            "Enables quantized DFlash drafter checkpoints (FP8 W8A8, "
            "NVFP4, AWQ, etc.) — correctness/compat fix per PR title, "
            "NOT throughput improvement claim. Today NO-OP for configured BF16 "
            "Qwen3.6-{27B,35B-A3B}-DFlash drafters "
            "(quant_config is None → original dense fast-path runs). "
            "Tomorrow: drop-in support for FP8/NVFP4 drafter checkpoints "
            "(e.g. AEON-7/Qwen3.6-NVFP4-DFlash, llm-compressor self-quant). "
            "Memory savings on adoption: BF16 drafter ~2.4 GB → FP8 ~1.2 GB "
            "per worker, ~2.4 GB total at TP=2 — frees KV-cache headroom. "
            "4 sub-patches into qwen3_dflash.py (Site A: F.linear→qkv_proj; "
            "B: pass quant_config to layer; C: conditional fused-KV; D: "
            "quantized fallback in precompute). Composable с PN40-A "
            "(different anchor surfaces in same file)."
        ),
        "upstream_pr": 40425,
        "applies_to": {
            "spec_method": ["dflash"],
        },
        "apply_module": "vllm.sndr_core.integrations.spec_decode.pn38_dflash_quant_drafter",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN40-classifier": {
        "title": "PN40 sub-D workload classifier (chat_completion middleware)",
        "tier": "community",
        "family": "middleware",
        "env_flag": "GENESIS_ENABLE_PN40_DFLASH_OMNIBUS",
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Genesis-original 2026-05-04 — companion to PN40 sub-D. "
            "Text-patches vllm/entrypoints/openai/chat_completion/serving.py "
            "(audit A-13 fix 2026-05-05: was incorrectly listed as "
            "serving_chat.py — actual target is chat_completion/serving.py) "
            "to classify each request as code/short_ctx/long_ctx/free_form "
            "and stash on `request._genesis_pn40_workload_class`. "
            "Consumer is the runtime K-trim hook in PN40 sub-C "
            "(scheduler.update_draft_token_ids). Toggled jointly with "
            "PN40 master via GENESIS_ENABLE_PN40_DFLASH_OMNIBUS — no "
            "separate enable flag (sub-D is universal companion to sub-C). "
            "Tier bias: code +1, long_ctx -1, others 0. Defensive on "
            "unknown class names (falls through to neutral bias)."
        ),
        "upstream_pr": None,
        "applies_to": {
            "spec_method": ["dflash", "mtp"],
        },
        "apply_module": "vllm.sndr_core.integrations.spec_decode.pn40_workload_classifier_hook",  # explicit ref — file exists but auto-derivation can't infer from PN40-classifier ID
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN40": {
        "title": "Spec-decode omnibus (A DFlash K-norm + B pool + C adaptive K + D sentinel)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN40_DFLASH_OMNIBUS",
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Genesis-original 2026-05-04 — 4-component omnibus spec-decode "
            "optimization with strict no-regression contract. "
            "Sub-A (DFlash-only): fused per-layer K-norm Triton kernel "
            "replaces L-iteration loop in qwen3_dflash.py:397-404. "
            "Numerical TDD 12/12 PASS rel_avg=0.0000. Microbench vs "
            "_custom_ops.rms_norm: 3.22x (27B L=5) / 5.32x (35B L=8). "
            "Per-draft-step saving +37us (27B) / +70us (35B). "
            "Sub-B (DFlash-only MVP): persistent K/V buffer pool, "
            "LRU-bounded, hit-rate tracked. Saves cudaMalloc churn. "
            "Sub-C (UNIVERSAL): adaptive K/N controller, mirrors SGLang "
            "tier policy + EMA hysteresis. Default tiers MTP K=3 [0,1,3], "
            "DFlash N=5 [0,1,3,5], DFlash N=3 [0,1,3]. NaN-trip safety. "
            "Applies to ALL 4 configs (27B/35B x MTP/DFlash). "
            "Sub-D (UNIVERSAL): workload classifier (code/short/long/"
            "free-form) + stability sentinel (sliding-window AL drop "
            "detector + NaN trip). Applies to ALL 4 configs. "
            "TDD: 12/12 (sub-A numerical) + 35/35 (sub-B/C/D logic). "
            "Per-sub env toggles GENESIS_PN40_ENABLE_SUB_{A,B,C,D}=0 "
            "to disable individually. Strict-superset throughout: "
            "any eligibility failure falls through to baseline. Default "
            "OFF master-gated until A/B prod-validates. Composes "
            "additively with PN21/PN23/PN24/P77."
        ),
        "upstream_pr": None,
        "applies_to": {
            "spec_method": ["dflash", "mtp"],  # C+D universal across both
        },
        "apply_module": "vllm.sndr_core.integrations.spec_decode.pn40_workload_classifier_hook",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    # PN37 archived 2026-05-04 to vllm/_genesis/_not_used_artifact/.
    # Premise (FA2 dead-zone for tiny-Q non-causal) was disproved by
    # microbench: torch SDPA already routes to FA2 packed-GQA path well.
    # Kernel + TDD (rel_avg < 0.01) preserved as research artifact;
    # entry intentionally NOT in PATCH_REGISTRY (no dispatcher matrix
    # row, no apply_all skip-noise on every boot).
    # PN36 was removed 2026-05-04 — was a misdiagnosis. The 5 `self.reasoner`
    # call-sites I found were inserted by OUR P62 backport (vllm#36138, still
    # OPEN), NOT by upstream. Upstream PR #41199 (MERGED 2026-05-01, included
    # in our pin) intentionally moved reasoner to per-request lazy build via
    # `self._get_reasoner(request)`. Pristine upstream code does NOT reference
    # `self.reasoner`. Fix path: disable P62 on this pin (it collides with
    # the rename refactor) and backport PR #40962 separately for the
    # post-reasoning-boundary spec-decode case.
    "PN50": {
        "title": "GDN proj fusion (SGLang#21019 backport — Qwen3.5/3.6 contiguous-projection Triton kernel)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN50_GDN_FUSED_PROJ",
        "default_on": False,
        "category": "perf_kernel",
        "credit": (
            "Backport of SGLang PR #21019 (MERGED 2026-03-23, commit "
            "5bdc07d). Original Triton kernel by Yuan Luo (@yuan-luo), "
            "Apache-2.0. Replaces the unfused split/reshape/cat/.contiguous() "
            "chain (5-6 launches + 2 explicit copies) in `gdn_linear_attn.py:562-566` "
            "Qwen3.5/3.6 contiguous projection branch with single fused Triton "
            "kernel `fused_qkvzba_split_reshape_cat_contiguous` (310 LOC). "
            "Pure data-copy kernel — no math, no reductions, no numerical drift. "
            "Output layout bit-identical to unfused PyTorch. Wrapper falls "
            "through to PyTorch reference on: non-contiguous input, non-pow2 "
            "head_dim, V_PER_GROUP non-integer, kernel launch failure. "
            "Affects: 27B Lorbus only (35B is Qwen3MoE — no GDN layers, "
            "patch never fires). Claimed gain on H200/Qwen3.5-35B-A3B (SGLang "
            "naming): +7.4% TPS, -10.8% TTFT, -31.2% ITL P95. On A5000 + "
            "27B Lorbus expect modest gain (memory-bound layer, A5000 PCIe "
            "slower than H200). Composable with PN11/PN29/PN32/P103 — verified "
            "no overlap (PN11 acts in interleaved branch, others in different "
            "files). Default OFF until live A/B prod-validates."
        ),
        "upstream_pr": None,
        "applies_to": {
            "model_class": ["qwen3_5", "qwen3_6"],
        },
        "apply_module": "vllm.sndr_core.integrations.attention.gdn.pn50_gdn_fused_proj",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN59": {
        "title": "Streaming-GDN orchestrator (Variant D Phase 2) — true Cliff 2b OOM fix",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN59_STREAMING_GDN",
        "default_on": False,
        "category": "hybrid",
        "credit": (
            "Genesis-original 2026-05-05, Variant D Phase 2. Replaces the "
            "(B, NT, H, V, K) full materialization in `chunk_gated_delta_rule_"
            "fwd_h → chunk_fwd_o` consumer pair with window-iterative driver. "
            "Eliminates Cliff 2b multi-turn OOM (Issue #19) — root cause: 805 "
            "MiB single allocation per layer per forward at T=64K Genesis 27B "
            "Lorbus shapes. Cross-engine validation: llama.cpp + MLX-LM use "
            "pure-streaming register-resident state, survive multi-turn; "
            "vLLM/SGLang/FLA materialize-full, hit Cliff 2b. **Independent "
            "confirmation** by noonghunna (issue #20, 2026-05-05): 'the "
            "limitation is the triton kernel for cliff 2; doesn't appear with "
            "llama.cpp'. Phase 1 numerical TDD proves bit-equivalence "
            "(rtol<1e-5) on 10 Genesis 27B shape cases. Composes with "
            "PN50/PN54/PN26b/P67 (orthogonal). Supersedes P103 outer chunked "
            "wrapper when both ON. Default OFF until live A/B prod-validates."
        ),
        "upstream_pr": None,  # FLA RFC #485, #190 pending — first-mover position
        "applies_to": {
            "model_class": ["qwen3_5", "qwen3_6"],  # 27B Lorbus hybrid only
        },
        # Symmetric with PN79's declaration (audit 2026-05-12): PN79 wraps
        # the GDN chunk-prefill path with in-place SSM state; this
        # streaming-GDN orchestrator targets the same call site differently.
        "conflicts_with": ["PN79"],
        "apply_module": "vllm.sndr_core.integrations.attention.gdn.pn59_streaming_gdn",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN58": {
        "title": "Spec-decode reasoning boundary validation — narrower alt to P62 (vllm#40962)",
        "tier": "community",
        "family": "reasoning",
        "env_flag": "GENESIS_ENABLE_PN58_SPEC_REASONING_BOUNDARY",
        "default_on": False,
        "category": "structured_output",
        "credit": (
            "Backport of vllm#40962 (OPEN). NARROWER "
            "alternative to our existing P62 (vllm#36138 broader pipeline-"
            "level fix). MUTUALLY EXCLUSIVE with P62 — both patch the same "
            "`should_advance` block in scheduler.update_from_output(). "
            "Apply check enforces P62 OFF requirement; SKIPS otherwise. "
            "PN58 modifies ONLY commit-time validation, doesn't touch "
            "bitmask/draft validation. Author warns: significant perf drop "
            "with multi-token reasoning markers (per-token boundary scan "
            "expensive). Engineering tradeoff: P62 = more correct (per-pos "
            "grammar masks), more invasive; PN58 = less correct in edge "
            "cases (commit-time only), cheaper hot-path. Multi-file "
            "(envs.py + abs_reasoning_parsers.py + basic_parsers.py + "
            "v1/structured_output/__init__.py + v1/core/sched/scheduler.py "
            "= 5 files, 6 sub-patches). Default OFF; current Genesis PROD "
            "uses P62 (broader). Enable PN58 only after measuring P62 perf "
            "hit on YOUR specific reasoning parser."
        ),
        "upstream_pr": 40962,
        "applies_to": {},
        "conflicts_with": ["P62"],
        "apply_module": "vllm.sndr_core.integrations.reasoning.pn58_spec_reasoning_boundary",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P107": {
        "title": "MTP truncation detector at reasoning→tool_call boundary (vllm#41467)",
        "tier": "community",
        "family": "serving",
        "env_flag": "GENESIS_ENABLE_P107_MTP_TRUNCATION_DETECTOR",
        "default_on": False,
        "category": "structured_output",
        "credit": (
            "Backport of vllm#41467 (ToastyTheBot, OPEN). При MTP K≥1 + "
            "tools + reasoning_parser возможна редкая (~0.25% per author "
            "on Qwen3.6 27B-FP8) ситуация: модель производит EOS на "
            "boundary reasoning→tool_call. finish_reason=stop, ни "
            "tool_calls, ни content. Defensive guard в "
            "chat_completion_stream_generator detect'ит combo и raise "
            "GenerationError (retryable) → клиент retries вместо silent "
            "stop. Author явно ссылается на наш P58/P59/P60/P61/P64 path. "
            "EXACT наш PROD config (27B Lorbus + MTP K=3 + tools). Defensive "
            "safety-net, не root-cause fix. Default OFF до live verify "
            "tool-call sweep на 27B PROD."
        ),
        "upstream_pr": 41467,
        "applies_to": {
            # [Genesis pin-gate 2026-05-11] Defensive safety-net (default
            # OFF). Validated dev9 → dev93. Self-retires when #41467 merges.
            "vllm_version_range": (">=0.20.0", "<0.21.0"),
        },
        "apply_module": "vllm.sndr_core.integrations.serving.p107_mtp_truncation_detector",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN56": {
        "title": "Qwen3Coder XML parse fallback (vllm#41466 backport)",
        "tier": "community",
        "family": "tool_parsing",
        "env_flag": "GENESIS_ENABLE_PN56_QWEN3CODER_XML_FALLBACK",
        "default_on": False,
        "category": "structured_output",
        "credit": (
            "Backport of vllm#41466 (ToastyTheBot, OPEN). When "
            "_parse_xml_function_call returns None or throws inside "
            "extract_tool_calls_streaming, prev_tool_call_arr keeps the "
            "header-step \"{}\" placeholder. Serving layer remainder check "
            "later double-emits {arguments:\"{}\"}. Strict OpenAI clients "
            "(Vercel AI SDK, OpenAI Node SDK) reject. Fix: track parse "
            "success, on failure restore prev_tool_call_arr from streamed "
            "args + closing brace. Composes with our P64 (vllm#39598) — P64 "
            "modified post-except flow but didn't touch try block. Affects "
            "ALL Genesis configs with qwen3_coder tool parser. Default OFF "
            "until live verify against tool-call sweep on 27B PROD."
        ),
        "upstream_pr": 41466,
        "applies_to": {},
        "apply_module": "vllm.sndr_core.integrations.tool_parsing.pn56_qwen3coder_xml_fallback",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN57": {
        "title": "TurboQuant centroids disk-persistent cache (vllm#41418-inspired)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_PN57_TQ_CENTROIDS_DISK_CACHE",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Inspired by vllm#41418 (TheTom, OPEN). Upstream PR pre-bakes 9 "
            "(d,bits) centroid tables inline (~1500 LOC of constants). "
            "Genesis approach: disk-persistent cache `~/.cache/genesis/"
            "turboquant_centroids.pkl` instead of inline constants. "
            "Lloyd-Max solver fully deterministic given (d,bits) → bit-"
            "identical to upstream pre-baked tables. Cold start: 200ms × N "
            "first-time shapes per fresh container. Subsequent boots / "
            "worker restarts: instant lookup. Saves ~205ms per worker on "
            "k8v4 path. Atomic write (tempfile+rename), defensive fall-"
            "through to solver on any cache failure. Default OFF until "
            "live-verified cold-start savings."
        ),
        "upstream_pr": 41418,
        "applies_to": {},
        "apply_module": "vllm.sndr_core.integrations.attention.turboquant.pn57_tq_centroids_disk_cache",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN122": {  # renamed 2026-05-14 from SPRINT26_CG_DISPATCH_TRACE — long ID violated P[N]?\d+ convention + auto-derivation
        "title": "Sprint 2.6 v2 — CUDA graph dispatch trace wire-in (formerly SPRINT26_CG_DISPATCH_TRACE)",
        "tier": "community",
        "family": "observability",  # 2026-05-11 audit fix: was "worker" but file lives under integrations/observability/ + category is observability
        "env_flag": "GENESIS_ENABLE_PN122_CG_DISPATCH_TRACE",  # legacy GENESIS_ENABLE_SPRINT26_CG_DISPATCH_TRACE accepted for 1 release
        "default_on": False,
        "category": "observability",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": (
            "vllm.sndr_core.integrations.observability."
            "pn122_sprint26_cudagraph_dispatch_trace"
        ),
        "lifecycle": "experimental",
        "experimental_note": (
            "Text-patch into gpu_model_runner.py decoder dispatch site "
            "to call record_dispatch(matched). Default OFF — opt in via "
            "this flag PLUS the runtime GENESIS_CUDAGRAPH_DISPATCH_TRACE=1 "
            "to actually emit per-N-requests summary lines. Wave 6 PN16 "
            "V1 regression root-cause: dispatch mismatch invisible to "
            "wall_TPS averaging."
        ),
        "credit": (
            "Genesis-original — Sandermage. Sprint 2.6 v2 closes the "
            "Sprint 2.6 v1 'instrumentation only, no wire-in' gap."
        ),
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {},
    },
    "PN132": {
        "title": "Triton top-k/top-p contiguous logits fix (backport vllm#42739)",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_ENABLE_PN132_TOPK_TOPP_CONTIGUOUS",
        "default_on": False,
        "category": "stability",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": (
            "vllm.sndr_core.integrations.compile_safety."
            "pn132_triton_topk_topp_contiguous"
        ),
        "lifecycle": "experimental",
        "experimental_note": (
            "Backport vllm#42739 (OPEN). Correctness fix: Triton "
            "top-k/top-p kernel computes row_ptr = base + row * VOCAB "
            "assuming contiguous row-major. Non-contiguous views (from "
            "index_select/slicing) → kernel reads garbage. PN132 wraps "
            "apply_top_k_top_p_triton с contiguous() guarantee. "
            "У нас VLLM_USE_FLASHINFER_SAMPLER=1 → Triton path обычно "
            "не используется (FlashInfer первый), но fallback возможен "
            "для unsupported combos. Defense-in-depth."
        ),
        "credit": "Backport vllm#42739 by Sandermage 2026-05-15.",
        "upstream_pr": 42739,
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {
            "vllm_version_range": (">=0.20.0", "<0.22.0"),
        },
    },
    "PN133": {
        "title": "MTP scheduler empty-output accounting fix (backport vllm#42722)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN133_MTP_EMPTY_OUTPUT_FIX",
        "default_on": False,
        "category": "stability",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": (
            "vllm.sndr_core.integrations.spec_decode."
            "pn133_mtp_scheduler_empty_output"
        ),
        "lifecycle": "experimental",
        "experimental_note": (
            "Backport vllm#42722 (OPEN). Fixes permanently-stuck request "
            "in MTP/spec-decode when model_runner returns empty "
            "generated_token_ids (request abortion, async race, OOM "
            "partial output). Pre-fix: scheduler doesn't account "
            "scheduled draft tokens as rejected → num_computed_tokens "
            "stays caught up → scheduler stops issuing work для "
            "unfinished request. Также fixes pre-existing crash через "
            "len([])-1 = -1 → Prometheus counter ValueError."
        ),
        "credit": "Backport vllm#42722 by Sandermage 2026-05-15.",
        "upstream_pr": 42722,
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {
            "vllm_version_range": (">=0.20.0", "<0.22.0"),
        },
    },
    "PN134": {
        "title": "torch.compile fullgraph patch для PyTorch 2.11 (backport vllm#42686) — BENCH-VALIDATED REGRESSOR, DO NOT ENABLE",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_ENABLE_PN134_TORCH_COMPILE_FULLGRAPH_211",
        "default_on": False,
        "category": "perf_hotfix",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": (
            "vllm.sndr_core.integrations.compile_safety."
            "pn134_torch_compile_fullgraph_211"
        ),
        "lifecycle": "retired",
        "retired_waiver": True,
        "retired_reason": (
            "Bench 2026-05-15 on Qwen3.6-35B-A3B-FP8 (dev371 nightly-bf610c2f, "
            "2× A5000 TP=2): enabling PN134 caused -25% TPS regression "
            "(211.38 → 158.0), TPOT +37%, TTFT 85→196 ms. Monkey-patching "
            "torch._inductor.ir.StorageBox.should_realize_on_reuse affects "
            "the ENTIRE model compilation graph, not just attention path — "
            "the size-aware cost model materializes too many intermediates "
            "for hybrid_gdn_moe layout, blowing the Inductor cache and "
            "forcing recompilation on every batch shape variant. "
            "DO NOT ENABLE on this model class. Module kept on disk for "
            "future investigation on dense-attention models where the "
            "cost model may behave correctly. See plan section 12.x for "
            "regression report."
        ),
        "experimental_note": (
            "RETIRED 2026-05-15 — bench-validated regressor. Original "
            "design backport vllm#42686 (OPEN), closes vLLM issue #27828. "
            "Patches torch._inductor.ir.StorageBox.should_realize_on_reuse "
            "с size-aware cost model для PyTorch 2.11 (fix landed в "
            "torch 2.12). Theory: без fix Inductor inline'ит residual в "
            "fused_add_rms_norm каждый раз → cascade re-computation. "
            "Reality on hybrid_gdn_moe: -25% TPS regression."
        ),
        "credit": "Backport vllm#42686 (pytorch#176994 simplified) by Sandermage 2026-05-15. Retired same day after bench regression.",
        "upstream_pr": 42686,
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {
            "vllm_version_range": (">=0.20.0", "<0.22.0"),
        },
    },
    "PN128": {
        "title": "Spec-decode helper kernel warmup (backport vllm#41481, 4 eagle kernels)",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_ENABLE_PN128_SPEC_DECODE_WARMUP",
        "default_on": False,
        "category": "perf_hotfix",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": (
            "vllm.sndr_core.integrations.compile_safety."
            "pn128_spec_decode_helper_warmup"
        ),
        "lifecycle": "experimental",
        "experimental_note": (
            "Genesis backport of vllm-project/vllm#41481 (OPEN). Закрывает "
            "4 из 8 JIT spikes на первом user request: "
            "eagle_prepare_next_token_padded_kernel, "
            "eagle_prepare_inputs_padded_kernel, "
            "copy_and_expand_eagle_inputs_kernel, "
            "eagle_step_slot_mapping_metadata_kernel. Wraps "
            "Worker.compile_or_warm_up_model + после оригинального "
            "warmup вызывает 4 dummy Triton kernel invokes с synthetic "
            "shapes (next_power_of_2(num_spec_tokens + 1)). Auto-skip "
            "V2_MODEL_RUNNER=1, enforce_eager=True. Issue #39790 H100 "
            "repro показал 25× первая-request регрессию pre-fix."
        ),
        "credit": "Backport of vllm-project/vllm#41481 by Sandermage 2026-05-15.",
        "upstream_pr": 41481,
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {
            "model_arch": [
                "Qwen3_5ForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
                "Qwen3NextForCausalLM",
                "Qwen3MoeForCausalLM",
            ],
            "vllm_version_range": (">=0.20.0", "<0.22.0"),
        },
    },
    "PN129": {
        "title": "V1 slot mapping kernel warmup (backport vllm#42165, 1 kernel + do_not_specialize)",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_ENABLE_PN129_SLOT_MAPPING_WARMUP",
        "default_on": False,
        "category": "perf_hotfix",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": (
            "vllm.sndr_core.integrations.compile_safety."
            "pn129_slot_mapping_warmup"
        ),
        "lifecycle": "experimental",
        "experimental_note": (
            "Genesis backport of vllm-project/vllm#42165 (OPEN). Закрывает "
            "_compute_slot_mapping_kernel JIT spike + (попытка) "
            "do_not_specialize='num_tokens' через private Triton API. "
            "Если do_not_specialize injection не работает на нашей "
            "версии Triton, остаётся warmup hook — kernel JIT'ится на "
            "boot вместо первого user request. Best-effort fix."
        ),
        "credit": "Backport of vllm-project/vllm#42165 by Sandermage 2026-05-15.",
        "upstream_pr": 42165,
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {
            "vllm_version_range": (">=0.20.0", "<0.22.0"),
        },
    },
    "PN130": {
        "title": "TurboQuant decode kernel warmup (backport vllm#42215, 1 kernel + workspace prealloc)",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_ENABLE_PN130_TQ_DECODE_WARMUP",
        "default_on": False,
        "category": "perf_hotfix",
        "implementation_status": "full",
        "source": "vllm_pr_backport",
        "apply_module": (
            "vllm.sndr_core.integrations.compile_safety."
            "pn130_turboquant_decode_warmup"
        ),
        "lifecycle": "experimental",
        "experimental_note": (
            "Genesis backport of vllm-project/vllm#42215 (OPEN). Закрывает "
            "_tq_grouped_decode_stage1 JIT spike + предотвращает workspace "
            "re-allocation после lock_workspace(). Итерирует Attention "
            "слои модели, dedupes по config-tuple, вызывает "
            "impl._decode_attention() с synthetic tensors. Auto-skip "
            "когда kv_cache_dtype != turboquant_*."
        ),
        "credit": "Backport of vllm-project/vllm#42215 by Sandermage 2026-05-15.",
        "upstream_pr": 42215,
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {
            "vllm_version_range": (">=0.20.0", "<0.22.0"),
        },
    },
    "PN127": {
        "title": "Qwen 3.5/3.6 enhanced chat-template auto-install (closes club-3090#53/#72)",
        "tier": "community",
        "family": "serving",
        "env_flag": "GENESIS_ENABLE_PN127_AUTO_CHAT_TEMPLATE",
        "default_on": False,
        "category": "stability",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": (
            "vllm.sndr_core.integrations.serving."
            "pn127_chat_template_qwen36"
        ),
        "lifecycle": "experimental",
        "experimental_note": (
            "Genesis-original 2026-05-15. Закрывает operator pain: enhanced "
            "chat-template для Qwen 3.5/3.6 hybrid_gdn_moe (interleaved-"
            "thinking + XML tool_call) ранее жил в HF репозиториях froggeric/"
            "Sandermage/club-3090 и operator должен был knew где искать и "
            "копировать .jinja вручную. PN127 запекает enhanced template как "
            "Genesis asset (vllm/sndr_core/assets/chat_templates/qwen3.6_"
            "enhanced.jinja) и на apply() копирует в writable location "
            "(/tmp/genesis/chat_templates/ или GENESIS_CHAT_TEMPLATE_DIR). "
            "Operator получает каноничный путь через log line; запускает "
            "vllm с --chat-template <path>. Закрывает 7 bugs в дефолтном "
            "template: empty <think></think>, </thinking> hallucination, "
            "unclosed think pre tool_call, no-user-query crash, developer "
            "role, multi-turn tool-call SSE deadlock (club-3090#72), think→"
            "tool_call boundary truncation."
        ),
        "credit": (
            "Genesis-original — Sandermage. Combines Sandermage v7.62 "
            "interleaved-thinking template + 7 froggeric fixes + "
            "club-3090 live-verify (30/30 tool regression PASS)."
        ),
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {
            "model_arch": [
                "Qwen3_5ForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
                "Qwen3NextForCausalLM",
                "Qwen3MoeForCausalLM",
            ],
            "vllm_version_range": (">=0.20.0", "<0.22.0"),
        },
    },
    "PN126": {
        "title": "V1 decode + spec-decode kernel warmup orchestrator (fixes JIT spikes on first request)",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_ENABLE_PN126_V1_DECODE_WARMUP",
        "default_on": False,  # bench-gate required
        "category": "perf_hotfix",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": (
            "vllm.sndr_core.integrations.compile_safety."
            "pn126_v1_decode_kernel_warmup"
        ),
        "lifecycle": "experimental",
        "experimental_note": (
            "Genesis-original 2026-05-15. Closes V1 vs V2 model runner gap: "
            "V2 calls warmup_kernels() at end of compile_or_warm_up_model "
            "which exercises decode + spec-decode + TQ kernels at boot; "
            "V1 only runs a sampler-only dummy_run with cudagraph_mode=NONE. "
            "Result on V1: vLLM jit_monitor warns 8-10 kernel JIT events "
            "on FIRST user request, causing TTFT spike of 5-25 s. "
            "PN126 wraps Worker.compile_or_warm_up_model to add 2 extra "
            "_dummy_run() passes after the original sampler warmup: "
            "(Pass 1) prefill at max_num_batched_tokens with PIECEWISE "
            "cudagraph dispatch — covers Mamba causal_conv1d at large T + "
            "prefill attention; (Pass 2) uniform decode at "
            "max_num_seqs × (1 + num_speculative_tokens) with FULL "
            "cudagraph dispatch — covers decode attention + TQ kernels + "
            "spec-decode draft prep kernels. Expected TTFT CV drop "
            "30%→~15% post-warmup. Auto-skips when VLLM_USE_V2_MODEL_RUNNER=1 "
            "(V2 native) or enforce_eager=True (no cudagraphs). "
            "Default OFF until bench: 35B 8K TPS unchanged, TTFT CV < 20%, "
            "boot time +3-10s acceptable."
        ),
        "credit": (
            "Genesis-original — Sandermage 2026-05-15. Pattern from V2 "
            "warmup_kernels (vllm/v1/worker/gpu/warmup.py in dev338+) "
            "adapted to V1 model runner via wrapped "
            "compile_or_warm_up_model. Related: PR #39822 (SSD kernel "
            "warmup at profile_run — landed upstream but only covers "
            "MambaMixer2 path, not GDN+spec_decode+TQ kernels that fire "
            "on first user request)."
        ),
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {
            # Hybrid models with spec_decode benefit most. Dense models
            # (Llama, Mistral) still benefit from prefill+decode warmup
            # but the gap there is smaller (no Mamba JIT, no TQ).
            "model_arch": [
                "Qwen3_5ForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
                "Qwen3NextForCausalLM",
                "Qwen3MoeForCausalLM",
            ],
            "vllm_version_range": (">=0.20.0", "<0.22.0"),
        },
    },
    "PN125": {
        "title": "Hybrid Qwen3.5/3.6 FULL_AND_PIECEWISE cudagraph_mode (redundant on dev338+)",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_ENABLE_PN125_HYBRID_FULL_AND_PIECEWISE",
        "default_on": False,  # measured no-op on dev338
        "category": "perf_hotfix",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": (
            "vllm.sndr_core.integrations.compile_safety."
            "pn125_hybrid_full_and_piecewise"
        ),
        "lifecycle": "experimental",
        "experimental_note": (
            "POST-BENCH NOTE (2026-05-15): on dev338 PN125 is empirically "
            "REDUNDANT. The vllm v1 default cudagraph resolver in "
            "config/compilation.py already sets FULL_AND_PIECEWISE when "
            "splitting_ops_contain_attention(); for hybrid_gdn_moe this "
            "branch fires unconditionally. Bench on 2× A5000 + 35B-A3B-FP8 "
            "+ TQ k8v4 + MTP K=3 (n=25, 5 runs × 5 prompts × 384 tok):\n"
            "  - PN125 OFF: 206.26 TPS / CV 5.5%\n"
            "  - PN125 ON:  206.23 TPS / CV 5.5%\n"
            "Delta within noise band (CV exceeds the 0.01% TPS gap by 500×).\n"
            "ORIGINAL motivation (likely incorrect for dev338): "
            "Qwen3_5ForConditionalGenerationConfig only updates "
            "mamba_ssm_cache_dtype and never calls "
            "MambaModelConfig.verify_and_update_config — assumed this "
            "leaked the FULL_AND_PIECEWISE setup. In practice the v1 "
            "resolver runs LATER in init and re-applies FULL_AND_PIECEWISE "
            "via the splitting-ops path regardless. PN125 monkey-patches "
            "verify_and_update_config to also invoke MambaModelConfig — "
            "essentially a NO-OP given the v1 resolver. Retained as "
            "audit-trail registry entry + safety net in case future vllm "
            "pins change the resolver default (auto-retire when "
            "splitting_ops_contain_attention drops the FULL branch). "
            "Default OFF until a bench shows measurable benefit on some "
            "pin/workload combination."
        ),
        "credit": (
            "Genesis-original 2026-05-15 — Sandermage. Source: "
            "https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/ "
            "(PyTorch blog claim of up to 91% throughput gain on hybrid "
            "Mamba models — measured 0% on our 35B / dev338 stack since "
            "vllm v1 default already engages FULL_AND_PIECEWISE)."
        ),
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {
            "model_arch": [
                "Qwen3_5ForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
                "Qwen3NextForCausalLM",
                "Qwen3MoeForCausalLM",
            ],
            # Valid for pins where MambaModelConfig.verify_and_update_config exists.
            "vllm_version_range": (">=0.20.0", "<0.21.0"),
        },
    },
    "PN96b": {
        "title": "Persistent Marlin MoE workspace (Wave 9 dev209 perf-restore)",
        "tier": "community",
        "family": "moe",
        "env_flag": "GENESIS_ENABLE_PN96B",  # renamed from PN96 2026-05-14 to avoid collision with kv_cache/PN96 emergency demote
        "default_on": True,
        "category": "perf_hotfix",
        "credit": (
            "Genesis-original 2026-05-12 (Wave 9 dev209 35B regression "
            "RCA). Upstream `experts/marlin_moe.py::MarlinExperts.apply` "
            "calls `fused_marlin_moe(...)` without passing the `workspace` "
            "kwarg, so `_fused_marlin_moe` allocates fresh via "
            "`marlin_make_workspace_new(device, 4)` on every MoE call. "
            "PN96 wraps MarlinExperts.apply to cache a persistent "
            "workspace per-instance + monkey-patches fused_marlin_moe to "
            "honor a thread-local default when workspace=None. Target: "
            "recover part of the -2.82% TPS / +2.86% TPOT 35B A3B-FP8 "
            "regression seen between dev93 and dev209. NO-OP on "
            "non-Marlin paths (27B hybrid GDN+Mamba INT4 unaffected). "
            "Auto-skips on dev93-era layout where experts/marlin_moe.py "
            "doesn't exist. Self-retires when upstream adds workspace= "
            "to the apply→fused_marlin_moe call site."
        ),
        "upstream_pr": None,  # not yet proposed upstream
        "applies_to": {},
        "apply_module": "vllm.sndr_core.integrations.moe.pn96b_marlin_persistent_workspace",
        "lifecycle": "experimental",
        "experimental_note": (
            "Runtime hook (no _make_patcher). default_on=True for 35B "
            "PROD. Recovers a portion of the dev209 MoE regression "
            "(exact gain TBD by post-deploy A/B bench). Composes with "
            "P17/P18 (per-SM tuning) + P22/P38 (TQ workspace) — "
            "different optimization vectors, no aliasing."
        ),
        "conflicts_with": [],
        # P18 is not a separate registry entry — it's the companion "B"
        # variant of P17 (Marlin MoE per-SM tuning) handled inline.
        "composes_with": ["P17", "P22", "P38"],
        "implementation_status": "full",
    },
    "PN95": {
        "title": "PN95 — Tier-aware KV cache + CPU offload + boot-time expansion (Path C, club-3090 #58)",
        "tier": "community",
        "family": "kv_cache",
        "env_flag": "GENESIS_ENABLE_PN95_TIER_AWARE_CACHE",
        "default_on": False,
        "category": "kv_cache",
        "implementation_status": "partial",
        "source": "genesis_original",
        "apply_module": "vllm.sndr_core.integrations.kv_cache.pn95_tier_aware_cache",
        "lifecycle": "experimental",
        "experimental_note": (
            "Eleven text-patch anchors wired into vLLM v1 KV cache hot path. "
            "Phase 1 (anchors 1-3): cache_blocks admit, block_pool touch, "
            "KVCacheManager mamba classifier + lazy TierManager init. "
            "Phase 2 (anchors 4-5): worker register_kv_caches + Scheduler tick. "
            "Phase 4 (anchors 6-8): prefix cache extension — block_pool register, "
            "demote-on-evict, promote-on-miss. Phase 5 partial (anchors 9, 11, 12): "
            "boot-time available_memory inflation, BlockPool metadata side-table, "
            "get_new_blocks virtual materialization. "
            "Phase 5 anchor #10 (KVCacheTensor physical-allocation cap via "
            "pn95_physical_num_blocks_cap()) helper exists in _pn95_runtime.py "
            "but text-patch wire-in is pending — boot-time virtualization needs "
            "live GPU validation before flipping that switch. CPU slab is "
            "torch.empty(pin_memory=True) when torch+CUDA available; "
            "MambaSpec groups are filtered from demote candidates."
        ),
        "credit": (
            "Genesis-original implementation. Addresses club-3090 #58 — Mamba "
            "SSM state lives outside the KV pool, so all upstream CPU-offload "
            "paths (vLLM cpu-offload-gb, SimpleCPUOffloadConnector, LMCache, "
            "SGLang HiCache) crash on hybrid-GDN models. PN95 filters MambaSpec "
            "groups out of demote candidates and drains MM/vision pages first "
            "(image tokens carry lower attention re-use than text-prefix pages). "
            "Reuses PN91 eviction policy ABC (LRU/2Q/ARC) per tier. Coordinates "
            "with P83 and P85 in the same vLLM source files."
        ),
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {},
        "related_upstream_prs": [],
    },
    "PN90": {
        "title": "Probabilistic draft rejection (vllm#40269 backport) — propagate draft_probs to verifier",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_PN90_PROBABILISTIC_DRAFT",
        "default_on": False,
        "lifecycle": "experimental",
        "implementation_status": "full",
        "category": "spec_decode",
        "apply_module": "vllm.sndr_core.integrations.spec_decode.pn90_probabilistic_draft_rejection",
        "credit": (
            "Backport of vllm-project/vllm#40269 (MERGED upstream 2026-05-14). "
            "Stock vLLM dev93 passed literal `None` for draft_probs in "
            "gpu_model_runner.py:3416 → rejection_sampler fell back to "
            "argmax-or-bonus rule. PN90 captures softmax probs in "
            "_greedy_sample (proposer side), accumulates per K-step, "
            "stacks into [total_drafts, vocab] 2D layout, and feeds to "
            "rejection_sampler. Probabilistic acceptance rule "
            "min(1, target_prob/draft_prob) is mathematically tighter "
            "and accepts more borderline tokens. Expected: +0.5-2% "
            "acceptance rate on MTP K=3 workloads. "
            "Genesis contribution: full text-patch implementation across "
            "llm_base_proposer.py (3 anchors) + gpu_model_runner.py "
            "(1 anchor), MultiFilePatchTransaction atomicity, drift "
            "markers, env-gated for safe back-compat. "
            "POST-MERGE NOTE (2026-05-15): upstream landed the same "
            "feature via `speculative_config.draft_sample_method = "
            "\"probabilistic\"` + `take_last_draft_probs()`. PN90 drift "
            "markers detect upstream-native symbols and self-skip on "
            "pins ≥dev338. Prefer the native path: set draft_sample_method "
            "(see commit 0e877eaf exposing the knob on SpecDecodeConfig) "
            "instead of GENESIS_ENABLE_PN90_*."
        ),
        "upstream_pr": 40269,
        "applies_to": {
            # Only fires for MTP/Eagle/DFlash drafters that go through
            # llm_base_proposer._greedy_sample. ngram drafter has its
            # own propose path that doesn't touch this code.

            # [Genesis pin-gate] PN90 anchors live in llm_base_proposer.py
            # (3 sites) + gpu_model_runner.py (1 site). Validated against:
            #   - dev93+g51f22dcfd (2026-05-07, PROD baseline, +7.4% TPS bench)
            # On <0.20.2 the anchor target lines do not yet exist (different
            # rejection_sampler path). On future pins, drift detector
            # auto-skips if anchors moved — pin-gate prevents the attempt
            # when version is known-incompatible.
            "vllm_version_range": (">=0.20.2rc1.dev9", "<0.21.0"),
        },
        "requires_patches": [],
        "conflicts_with": [],
    },
    "SNDR_WORKSPACE_001": {
        "title": "SNDR-WORKSPACE-001 — workspace grow-after-lock graceful fix",
        "tier": "community",
        "family": "worker",
        "env_flag": "GENESIS_ENABLE_SNDR_WORKSPACE_001",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "perf_hotfix",
        "apply_module": "vllm.sndr_core.integrations.worker.sndr_workspace_001_grow_after_lock",
        "source": "genesis_original",
        "credit": (
            "Genesis-original fix for the upstream vllm v1 workspace.py guard "
            "that raises AssertionError when any post-warmup path "
            "(decode_attention, continuation_prefill, Marlin GEMM scratch) "
            "needs to grow the GPU workspace. Without this patch the engine "
            "crashes on every request that touches a code path warmup did not "
            "pre-size. The patch replaces the raise with a warn + grow: the "
            "torch CUDA allocator handles the resize, the first call takes a "
            "non-graph slow path, subsequent calls hit the graph normally. "
            "Net effect: engine keeps serving instead of crashing. "
            "Update 2026-05-14 PR sweep: upstream vllm#42551 (jasonboukheir, "
            "DRAFT) proposes a more invasive fix to the same fault class — "
            "pre-reserve decode workspace in TurboQuantAttentionImpl.__init__ "
            "plus non-raising try_get_simultaneous() + torch.empty fallback "
            "in _decode_attention. When #42551 merges and a pin bump absorbs "
            "it, this patch self-retires via drift-marker. Until then we "
            "stay on the lighter warn+grow shape which is already running "
            "in PROD without issue (27B INT4 + TQ k8v4, 256K hardware-"
            "verified Wave 9). PN34 is being retired in this same wave "
            "(duplicate of this patch with the same fault site)."
        ),
        "applies_to": {
            "vllm_version_range": (">=0.20.2rc1.dev9", "<0.21.0"),
        },
        "requires_patches": [],
        "conflicts_with": [],
        "upstream_pr": 42551,  # 2026-05-14 PR sweep — pin-bump retire trigger
        "implementation_status": "full",
    },
    "PN202": {
        "title": "PN202 — per-layer KV tensor split (Tier 2.A enabler)",
        "tier": "community",
        "family": "streaming",  # was "kv_cache"; integration lives at integrations/streaming/
        "env_flag": "GENESIS_ENABLE_PN202_PER_LAYER_KV_SPLIT",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "memory",
        "apply_module": "vllm.sndr_core.integrations.streaming.pn202_per_layer_kv_split",
        "source": "genesis_original",
        "credit": (
            "Genesis-original Tier 2.A enabler. Replaces Branch-C of "
            "kv_cache_utils.py::get_kv_cache_config_from_groups (which "
            "emits group_size slabs each shared by representative layers) "
            "with one KVCacheTensor per layer (Branch-A semantics). Net "
            "bytes identical; enables per-layer memory policies "
            "(offload/evict/quantize) required by PN203. Zero speed "
            "and quality impact."
        ),
        "applies_to": {"vllm_version_range": (">=0.20.2rc1.dev9", "<0.21.0")},
        "requires_patches": [],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN203": {
        "title": "PN203 — cold-prefix CPU offload manager (Tier 3.A)",
        "tier": "community",
        "family": "streaming",  # was "kv_cache"; integration lives at integrations/streaming/
        "env_flag": "GENESIS_ENABLE_PN203_COLD_PREFIX_OFFLOAD",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "memory",
        "apply_module": "vllm.sndr_core.integrations.streaming.pn203_cold_prefix_offload",
        "source": "genesis_original",
        "credit": (
            "Genesis-original Tier 3.A — the architectural breakthrough "
            "piece. Demotes full-attention layer blocks older than "
            "active_window_tokens (default 32768) to PN95 pinned host RAM "
            "L2. Reuses existing PN95 infrastructure: pinned pool, stream "
            "pool, prefetch API, demote_on_evict. Mamba/GDN state kept "
            "GPU-resident (small fixed cost). Requires PN202 per-layer "
            "split. Math: 256K context fp8 KV = 8 GiB attention KV; "
            "active window 32K = 1 GiB → 7 GiB offloaded. Quality "
            "preserved (math identical). Decode adds 5-15ms per token "
            "when paging hits."
        ),
        "applies_to": {"vllm_version_range": (">=0.20.2rc1.dev9", "<0.21.0")},
        "requires_patches": ["PN202"],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN200": {
        "title": "PN200 — GDN outer-forward scratch pool (Tier 1.B)",
        "tier": "community",
        "family": "streaming",  # was "kv_cache"; integration lives at integrations/streaming/
        "env_flag": "GENESIS_ENABLE_PN200_GDN_SCRATCH_REUSE",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "memory",
        "apply_module": "vllm.sndr_core.integrations.streaming.pn200_gdn_scratch_reuse",
        "source": "genesis_original",
        "credit": (
            "Genesis-original Tier 1.B of three-tier memory plan. Routes "
            "gdn_linear_attn.py:765 core_attn_out (32 MiB × 48 layers per "
            "step) through the PN106 named-pool API with zero=True. "
            "Honors the vllm PR #28182 'must be zeroed' contract via "
            "explicit .zero_() on pool slice. Eliminates ~1.5 GiB alloc "
            "traffic per chunked-prefill step → ~500 MiB - 1 GiB GPU "
            "reclaim through reduced caching-allocator fragmentation. "
            "Speed impact 1-3% (memset offset by saved alloc latency). "
            "Composes with PN106 (inner FLA chunk scratch) and PN201 "
            "(scheduler empty_cache) for Tier 1 complete coverage."
        ),
        "applies_to": {"vllm_version_range": (">=0.20.2rc1.dev9", "<0.21.0")},
        "requires_patches": [],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN201": {
        "title": "PN201 — scheduler empty_cache hook (Tier 1.C)",
        "tier": "community",
        "family": "streaming",  # was "kv_cache"; integration lives at integrations/streaming/
        "env_flag": "GENESIS_ENABLE_PN201_SCHEDULER_EMPTY_CACHE",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "memory",
        "apply_module": "vllm.sndr_core.integrations.streaming.pn201_scheduler_empty_cache",
        "source": "genesis_original",
        "credit": (
            "Genesis-original Tier 1.C. Threshold-gated torch.cuda."
            "empty_cache() called from PN95 scheduler_tick when free_mib "
            "< 256 or free_blocks < threshold AND cooldown elapsed. "
            "Reclaims the 'reserved but unallocated' fragmentation that "
            "OOM logs show as 319 MiB stuck after long prefill runs. "
            "Cooldown (default 50 ticks ~= 5s) prevents hot-path stall. "
            "No text-patch — runtime hook in existing scheduler_tick path."
        ),
        "applies_to": {"vllm_version_range": (">=0.20.2rc1.dev9", "<0.21.0")},
        "requires_patches": [],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN106": {
        "title": "PN106 — GDN scratch tensor pool (architectural memory mgr)",
        "tier": "community",
        "family": "kv_cache",
        "env_flag": "GENESIS_ENABLE_PN106_GDN_H_POOL",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "memory",
        "apply_module": "vllm.sndr_core.integrations.kv_cache.pn106_gdn_h_pool",
        "source": "genesis_original",
        "credit": (
            "Genesis-original architectural memory manager. Replaces the "
            "per-call torch.empty / torch.empty_like patterns inside the "
            "48-GDN-layer hot path (chunk_delta_h.py, chunk_o.py) with "
            "slice views into named persistent pools. Eliminates 2.4-5.7 GiB "
            "of alloc/free traffic per chunked-prefill step and 200-400 MiB "
            "of steady-state fragmentation. Includes generic "
            "pn106_get_pooled_buf(name, shape, dtype, device) API for "
            "extending to other hot-path allocations (Marlin scratch, "
            "FlashAttention k_full/v_full, etc). Targets the exact crash "
            "site observed at chunk_o.py:168 on Qwen3.6-27B + 156K context."
        ),
        "applies_to": {
            "vllm_version_range": (">=0.20.2rc1.dev9", "<0.21.0"),
        },
        "requires_patches": [],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN105": {
        "title": "PN105 — PrefetchOffloader AutoRound INT4 compat (pin-assert relax)",
        "tier": "community",
        "family": "offload",
        "env_flag": "GENESIS_ENABLE_PN105_AUTOROUND_OFFLOAD_COMPAT",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "stability",  # was "compat"; normalize to existing VALID_CATEGORIES (AutoRound INT4 compat / pin-assert relax)
        "apply_module": "vllm.sndr_core.integrations.offload.pn105_prefetch_autoround_compat",
        "source": "genesis_original",
        "credit": (
            "Genesis-original fix unblocking vllm's PrefetchOffloader on "
            "AutoRound INT4 models. AutoRound's process_weights_after_"
            "loading replaces param.data with non-pinned INT tensors "
            "(g_idx, qzeros, scales). PrefetchOffloader asserts ALL "
            "cpu_storage pinned and crashes at engine startup. PN105 "
            "replaces the assertion with conditional blocking copy: "
            "non-blocking when pinned (fast path), blocking when not "
            "(slow fallback). Blocking copy from pageable memory IS "
            "correct, just slower. Combined with PN104 (UVA→Prefetch "
            "redirect) and Tier 1 GDN scratch pool, this unblocks "
            "cpu_offload_gb=8 on AutoRound → KV pool grows 4 GB → "
            "9-10 GB → 156K-176K context on single A5000."
        ),
        "applies_to": {"vllm_version_range": (">=0.20.2rc1.dev9", "<0.21.0")},
        "requires_patches": ["PN104"],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN104": {
        "title": "PN104 — redirect --cpu-offload-gb from UVA to Prefetch backend",
        "tier": "community",
        "family": "offload",
        "env_flag": "GENESIS_ENABLE_PN104_OFFLOAD_PREFETCH_REDIRECT",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "kernel",  # was "perf_critical"; normalize (kernel-level redirect from UVA to Prefetch backend)
        "apply_module": "vllm.sndr_core.integrations.offload.pn104_offload_backend_redirect",
        "source": "genesis_original",
        "credit": (
            "Genesis-original critical perf patch. vllm's --cpu-offload-gb "
            "defaults to UVAOffloader which uses cudaHostGetDevicePointer to "
            "map pinned host RAM as GPU-visible pointer — every GEMM kernel "
            "issues PCIe reads on every load instruction, with zero GPU "
            "caching. Empirically observed 24x slowdown (50K prefill: 30s "
            "→ 720s) on Genesis 27B INT4 single-A5000 with cpu_offload_gb=8. "
            "PN104 monkey-patches create_offloader so cpu_offload_gb auto-"
            "translates to PrefetchOffloader params (offload_group_size + "
            "num_in_group + prefetch_step=2). PrefetchOffloader does explicit "
            "cudaMemcpyAsync into static GPU buffers on a side stream, "
            "compute reads from GPU buffer, copy hidden behind previous "
            "layer's compute. Expected +30-50% TPS recovery — the single "
            "biggest win for any cpu_offload deployment."
        ),
        "applies_to": {
            "vllm_version_range": (">=0.20.2rc1.dev9", "<0.21.0"),
        },
        "requires_patches": [],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN97": {
        "title": "PN97 — physical-cap on KV tensor allocation (Phase 7 PoC)",
        "tier": "community",
        "family": "kv_cache",
        "env_flag": "GENESIS_ENABLE_PN97_TENSOR_PHYSICAL_CAP",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "memory",
        "apply_module": "vllm.sndr_core.integrations.kv_cache.pn97_tensor_physical_cap",
        "source": "genesis_original",
        "credit": (
            "Genesis-original Phase 7 PoC — missing Anchor #10. Caps "
            "each KVCacheTensor.size to physical GPU budget so VIRT=1 "
            "inflation does NOT cause CUDA OOM. Per-tensor cap derived "
            "from torch.cuda.mem_get_info(0) × 0.80 / n_tensors, or "
            "operator override via GENESIS_PN97_PHYSICAL_CAP_GIB. "
            "Prerequisite for 156K+ single-card via virtual block "
            "addressing — full support also requires PN98 (attention "
            "block_id translation) which is future work."
        ),
        "applies_to": {
            "vllm_version_range": (">=0.20.2rc1.dev9", "<0.21.0"),
        },
        "requires_patches": [],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN96": {
        "title": "PN96 — emergency-demote hook (Phase 6 PoC; allocator intercept)",
        "tier": "community",
        "family": "kv_cache",
        "env_flag": "GENESIS_ENABLE_PN96_EMERGENCY_DEMOTE",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "memory",
        "apply_module": "vllm.sndr_core.integrations.kv_cache.pn96_emergency_demote",
        "source": "genesis_original",
        "credit": (
            "Genesis-original Phase 6 PoC. Inserts an emergency-rescue "
            "branch at the entry of BlockPool.get_new_blocks: when vllm "
            "would raise 'Cannot get N free blocks', PN96 walks the "
            "free queue for already-cached (ref_cnt=0, block_hash!=None) "
            "blocks, captures their bytes to PN95 L2 via demote_on_evict, "
            "and clears the hash so vllm sees clean free slots. Recovers "
            "the engine from the allocation cliff without preempting "
            "active sequences. Helps multi-prefix workloads; does NOT "
            "extend single-user max_model_len above the physical pool — "
            "that requires virtual block_table addressing inside attention "
            "(Phase 7 / scheduler refactor)."
        ),
        "applies_to": {
            "vllm_version_range": (">=0.20.2rc1.dev9", "<0.21.0"),
        },
        "requires_patches": [],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN92": {
        "title": "PN92 — nixl_ep/deep_ep/mori trial-import guard (vllm PR #40154)",
        "tier": "community",
        "family": "worker",
        "env_flag": "GENESIS_ENABLE_PN92_NIXL_EP_TRIAL_IMPORT",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "loader",  # was "compat"; normalize (trial-import guard for nixl_ep/deep_ep/mori loader-time)
        "apply_module": "vllm.sndr_core.integrations.worker.pn92_nixl_ep_trial_import",
        "source": "genesis_original",
        "credit": (
            "Genesis-original backport of upstream vllm PR #40154 / "
            "issue #42525. New nightlies (>= dcacdf9a 2026-05-13) ship "
            "nixl_ep C++ extension compiled against CUDA 12 inside a "
            "CUDA-13 image. find_spec-only check in has_nixl_ep() lets "
            "the broken import cascade through fused_moe → all2all_utils "
            "and break ALL hybrid-MoE models (Qwen3.5/3.6, DeepSeek, "
            "Mixtral) on inspect. PN92 replaces with try/except trial "
            "import. Until upstream PR lands."
        ),
        "applies_to": {
            "vllm_version_range": (">=0.20.2rc1.dev209", "<0.21.0"),
        },
        "requires_patches": [],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN71": {
        "title": "PN71 — `</thinking>` hallucination runtime normalizer",
        "tier": "community",
        "family": "reasoning",
        "env_flag": "GENESIS_ENABLE_PN71_THINKING_TAG_NORMALIZE",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "reasoning",  # was "compat"; normalize (reasoning-subsystem thinking-tag hallucination normalizer)
        "apply_module": "vllm.sndr_core.integrations.reasoning.pn71_thinking_token_hallucination",
        "source": "genesis_original",
        "credit": (
            "Genesis-original parser-side normalizer for Qwen 3.6 "
            "`</thinking>` hallucination. The model occasionally emits "
            "the full word instead of the canonical `</think>` token. "
            "Without this patch, Qwen3ReasoningParser.extract_reasoning "
            "routes ALL output to reasoning channel with content=None. "
            "PN71 normalizes the tag at parser entry so the partition "
            "logic stays correct regardless of which chat template is "
            "active. Complements froggeric template fix (which handles "
            "the prompt side); PN71 handles the live-generation side."
        ),
        "applies_to": {
            "vllm_version_range": (">=0.20.2rc1.dev9", "<0.21.0"),
        },
        "requires_patches": [],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN73": {
        "title": "PN73 — safe `tool_calls.arguments` string→dict normalization",
        "tier": "community",
        "family": "serving",
        "env_flag": "GENESIS_ENABLE_PN73_TOOL_ARGS_SAFE_NORMALIZE",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "tool_parsing",  # was "compat"; normalize (tool_calls.arguments string→dict normalization)
        "apply_module": "vllm.sndr_core.integrations.serving.pn73_tool_args_safe_normalize",
        "source": "genesis_original",
        "credit": (
            "Genesis-original defensive normalizer for malformed "
            "tool_calls.arguments. Upstream chat_utils.py runs unguarded "
            "json.loads which raises HTTP 500 on (a) non-strict JSON, "
            "(b) non-string scalars, (c) double-encoded payloads. PN73 "
            "wraps in try/except and keeps the original string on failure "
            "instead of 500'ing. Strictly defensive."
        ),
        "applies_to": {
            "vllm_version_range": (">=0.20.2rc1.dev9", "<0.21.0"),
        },
        "requires_patches": [],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN91": {
        "title": "PN91 — `developer` role pre-render normalizer (OpenAI Responses API compat)",
        "tier": "community",
        "family": "serving",
        "env_flag": "GENESIS_ENABLE_PN91_DEVELOPER_ROLE",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "request_middleware",  # was "compat"; normalize (OpenAI `developer` role pre-render normalizer in request path)
        "apply_module": "vllm.sndr_core.integrations.serving.pn91_developer_role_normalizer",
        "source": "genesis_original",
        "credit": (
            "Genesis-original fix for OpenAI Responses API `role=developer` "
            "support. Maps developer→system at parser layer "
            "(_parse_chat_message_content) BEFORE chat template renders, so "
            "the fix holds regardless of which chat template is active — "
            "complements froggeric/enhanced jinja but does not require them."
        ),
        "applies_to": {
            "vllm_version_range": (">=0.20.2rc1.dev9", "<0.21.0"),
        },
        "requires_patches": [],
        "conflicts_with": [],
        "implementation_status": "full",
    },
    "PN82": {
        "title": "Mamba CUDA-graph stale `is_prefilling` padded rows — vllm#41873 backport",
        "tier": "community",
        "family": "worker",
        "env_flag": "GENESIS_ENABLE_PN82_MAMBA_CUDAGRAPH_PREFILL_ZERO",
        "default_on": False,
        "lifecycle": "experimental",
        "category": "perf_hotfix",
        "apply_module": "vllm.sndr_core.integrations.worker.pn82_mamba_cudagraph_prefill_zero",
        "credit": (
            "Backport of vllm-project/vllm#41873 (OPEN as of 2026-05-07). "
            "After CUDA-graph batch padding via condense(), the boolean "
            "`is_prefilling` slice keeps stale True values for padded "
            "rows beyond num_reqs. Mamba/hybrid attention backends read "
            "this slice to decide between prefill vs decode kernel paths "
            "and would treat padding rows as prefill, occasionally "
            "corrupting Mamba state on hybrid models. Fix is a single "
            "extra line right after the existing assignment: "
            "`is_prefilling[num_reqs:] = False`. Tiny, safe, default OFF "
            "until live smoke on hybrid CUDA-graph configs (27B INT4 + "
            "DFlash; 35B-A3B-FP8 not affected — no Mamba). Genesis "
            "contribution: integration, gating, idempotent TextPatch, "
            "drift markers, tests."
        ),
        "upstream_pr": 41873,
        "applies_to": {"is_hybrid": [True]},
        "implementation_status": "full",
    },
    "PN55": {
        "title": "wake_up nested KV cache crash fix — vllm#41602+#41896 unified backport",
        "tier": "community",
        "family": "worker",
        "env_flag": "GENESIS_ENABLE_PN55_WAKE_UP_HYBRID_KV",
        "default_on": False,
        "category": "perf_hotfix",
        "apply_module": "vllm.sndr_core.integrations.worker.pn55_wake_up_hybrid_kv",
        "credit": (
            "PR38 Day 2 (2026-05-08) upgrade from PN55v1 → PN55v2. "
            "Unified backport of vllm-project/vllm#41602 (kevglynn / "
            "Mistral, OPEN as of 2026-05-04) AND vllm-project/vllm#41896 "
            "(nested KV cache class). v1 only handled `list[Tensor]` "
            "for Mamba/DeltaNet hybrid; #41896 surfaced the same bug "
            "class with `tuple` and `Mapping` (block-scaled FP8 KV). "
            "Both PRs target the SAME wake_up zeroing site, so a "
            "separate PN83 would conflict on the anchor. PN55v2 collapses "
            "them into one recursive iterator that zero-s only real "
            "tensors and silently skips None / non-tensor sentinels. "
            "Affects 27B Lorbus Qwen3.6 hybrid (GDN = MambaSpec layers) "
            "and any future FP8 nested-KV deployment. Crash trigger: "
            "/sleep + /wake_up via management API. Genesis active "
            "scripts don't use sleep, but defensive backport recommended "
            "for any external mgmt-API trigger. Default OFF; enable "
            "when sleep/wake actively used."
        ),
        "upstream_pr": 41602,
        "related_upstream_prs": [41896],
        "applies_to": {},
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN54": {
        "title": "GDN contiguous-call deduplication (P0.7 Cliff 2b OOM mitigation)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN54_GDN_CONTIGUOUS_DEDUP",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Genesis-original 2026-05-04, inspired by MLX-LM PR #1077 "
            "(adurham, MIT) root-cause analysis: shared-buffer/slice-keeps-"
            "parent-alive class of bug. Removes 2 redundant `.contiguous()` "
            "calls in `gdn_linear_attn.py` already guaranteed contiguous by "
            "operator semantics OR re-enforced by FLA `@input_guard`. "
            "Sub-A: `ssm_state[non_spec_state_indices_tensor].contiguous()` "
            "(line ~985) — advanced index already produces fresh allocation; "
            "saves one full ssm_state-shape copy per prefill batch. Sub-B: "
            "LoRA branch `b/a.contiguous()` after `chunk(2, -1)` (lines 551-"
            "552) — chunk on last dim returns contiguous halves; LoRA-only, "
            "no-op on Genesis non-LoRA PROD. Target: Cliff 2b multi-turn "
            "OOM (Genesis Issue #19) — observed +1400 MiB/turn allocator "
            "delta; estimated saving 300-600 MiB/turn. Models: 27B Lorbus "
            "INT4 (all configs with GDN) — sub-A fires; 35B Qwen3MoE no GDN "
            "— patch never fires. Default OFF until live A/B Cliff 2b "
            "reproducer shows per-turn delta drops below ~900 MiB."
        ),
        "upstream_pr": None,
        "applies_to": {
            "model_class": ["qwen3_5", "qwen3_6"],
        },
        # Symmetric with PN79's declaration (audit 2026-05-12): PN79 wraps
        # `chunk_gated_delta_rule` while PN54 dedups `.contiguous()` calls
        # in the same gdn_linear_attn.py codepath. Together they double-pad
        # the same allocation in some prefill regimes.
        "conflicts_with": ["PN79"],
        "apply_module": "vllm.sndr_core.integrations.attention.gdn.pn54_gdn_contiguous_dedup",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN52": {
        "title": "prompt_logprobs eviction fix during chunked prefill (vllm#41411 backport)",
        "tier": "community",
        "family": "worker",
        "env_flag": "GENESIS_ENABLE_PN52_PROMPT_LOGPROBS_EVICTION",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Backport of vllm-project/vllm#41411 (MERGED 2026-05-04 18:46 UTC "
            "by Joachim Studnia, Mistral). Fixes TWO bugs in v1 gpu_worker "
            "prompt_logprobs path: (1) overly aggressive `-1` in "
            "`includes_prompt = computed_prefill < prompt_lens - 1` skipped "
            "the last prompt token's logprob when chunked-prefill boundary "
            "fell on `prompt_lens - 1`; (2) `in_progress_prompt_logprobs_cpu` "
            "stored on `input_batch` per-batch dict was lost on request "
            "eviction → silent corruption / IndexError on re-schedule. "
            "Multi-file text-patch: prompt_logprob.py + gpu_input_batch.py "
            "(field move) + gpu_model_runner.py (read/write per-request). "
            "Affects Genesis configs that combine `--enable-chunked-prefill` "
            "(all of ours) + spec-decode (MTP K=3 on PROD) + clients passing "
            "`prompt_logprobs=N`. Default OFF until live verify with Open "
            "WebUI / LibreChat workload that exercises prompt_logprobs."
        ),
        "upstream_pr": 41411,
        "superseded_by": "vllm#41411 (merged 2026-05-04, byte-equivalent on dev209+g5536fc0c0)",
        "vllm_version_range": "<0.20.2rc1.dev209+g5536fc0c0",  # active before upstream merge
        "apply_module": "vllm.sndr_core.integrations._retired.pn52_prompt_logprobs_eviction",
        "applies_to": {
            # [Genesis pin-gate 2026-05-11 iron rule #11] Deep-diff'd
            # upstream code in dev209: both fixes from PN52 now in vllm
            # natively — `< prompt_lens` (no -1) in prompt_logprob.py,
            # `in_progress_prompt_logprobs_cpu` moved to CachedRequestState
            # (gpu_input_batch.py:54 + gpu_model_runner.py:5200/5207/5264).
            # Auto-skip working via wiring's drift detector. Pin-gate adds
            # explicit boundary for `genesis explain` / audit reports.
            # Cleanup: delete patch file in next refactor pass.
            "vllm_version_range": "<0.20.2rc1.dev209",
        },
        "lifecycle": "retired",  # 2026-05-11: byte-equivalent with #41411 in dev209
        "implementation_status": "full",
    },
    "PN51": {
        "title": "Qwen3 streaming `enable_thinking=false` content routing (vllm#40816 backport)",
        "tier": "community",
        "family": "reasoning",
        "env_flag": "GENESIS_ENABLE_PN51_QWEN3_STREAMING_THINKING_DISABLED",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Backport of upstream issue vllm-project/vllm#40816 (OPEN, filed "
            "2026-04-22 by 'keehawkes'). When server is launched with "
            "--default-chat-template-kwargs '{\"enable_thinking\": false}' or "
            "the request passes chat_template_kwargs.enable_thinking=false, "
            "streaming responses incorrectly route every model token via "
            "delta.reasoning instead of delta.content. Mirrors the existing "
            "non-streaming short-circuit at qwen3_reasoning_parser.py:146-148. "
            "Affects ALL OpenAI-compatible streaming clients that read "
            "delta.content (Open WebUI, LibreChat, LobeChat, Cline, OpenCode). "
            "Single-line guard at extract_reasoning_streaming entry; no risk "
            "for thinking-enabled requests (guard False). Default OFF until "
            "Open WebUI / LibreChat repro proves the fix end-to-end on "
            "Genesis 27B/35B + reasoning-parser qwen3."
        ),
        "upstream_pr": 40816,
        "experimental_note": (
            "REACTIVATED 2026-05-15 after retired-patch audit of pin "
            "bf610c2f5 (dev371). Upstream PR #40816 is STILL OPEN and the "
            "streaming entry-point `extract_reasoning_streaming` (line 226 "
            "of qwen3_reasoning_parser.py) has NO `not self.thinking_enabled` "
            "short-circuit. The serving-layer `prompt_is_reasoning_end` "
            "bypass works for the common case (prompt has the pre-baked "
            "empty <think>\\n\\n</think>\\n\\n block), but defensive parser-"
            "entry recovery is still valuable when the bypass misses for "
            "any reason. Risk: zero — guard False for thinking-enabled "
            "(dominant case) and for legacy templates with </think> token."
        ),
        "apply_module": "vllm.sndr_core.integrations.reasoning.pn51_qwen3_streaming_thinking_disabled",
        "applies_to": {
            "vllm_version_range": (">=0.20.0", "<0.22.0"),
        },
        "lifecycle": "experimental",  # 2026-05-15 reactivated after retired-audit (gap confirmed in upstream parser)
        "implementation_status": "full",
    },
    "PN35": {
        "title": "Skip inputs_embeds buffer for text-only models (vllm#35975 backport)",
        "tier": "community",
        "family": "worker",
        "env_flag": "GENESIS_ENABLE_PN35_INPUTS_EMBEDS_OPTIONAL",
        "default_on": True,
        "category": "perf_hotfix",
        "credit": (
            "Backport of vllm-project/vllm#35975 by AjAnubolu (OPEN since "
            "2026-03-04). Skips the (max_num_tokens, hidden_size) GPU "
            "buffer allocation for text-only models in BOTH "
            "gpu_model_runner (~64 MiB GPU + 64 MiB pinned CPU) AND "
            "llm_base_proposer spec-decode proposer (~64 MiB GPU). For "
            "Qwen3.6-27B at max_num_tokens=4096 and hidden_size=8192: "
            "freed ~128 MiB GPU + 64 MiB pinned CPU per worker. "
            "Particularly relevant on borderline-OOM single-24GB-GPU "
            "configs (Cliff 2 fires at 50 MiB-free thresholds) and "
            "WSL2 setups with extra display overhead. Pattern credit: "
            "noonghunna club-3090 setup-time sidecar "
            "patch_inputs_embeds_optional.py 2026-05-02. Originally "
            "raised by club-3090#32 (RossNE99, GuiPerPT WSL2 OOM "
            "reports). Default ON — strict memory savings, no "
            "regression possible (the `if` guard preserves original "
            "allocation for multimodal models). Auto-retires when "
            "vllm#35975 merges upstream."
        ),
        "upstream_pr": 35975,
        # Promoted 2026-05-12 (Wave 9 dev209 + STABLE-prep): full ratchet
        # satisfied — `register_for_manifest()` added in wiring;
        # anchor_manifest.json covers PN35.Sub-1 (gpu_model_runner.py) +
        # PN35.Sub-2 (llm_base_proposer.py) with pristine fixtures from
        # the dev209 image. Production-validated default_on across Wave
        # 6→9 + dev93/dev209 with zero regressions. Strict-superset
        # (text-only guard preserves multimodal path verbatim).
        # Upstream vllm#35975 still OPEN — Genesis durably ahead;
        # auto-retires when upstream merges (registry-driven gate).
        "apply_module": "vllm.sndr_core.integrations.worker.pn35_inputs_embeds_optional",
        "lifecycle": "stable",
        "stable_kind": "text-patch",
        "stable_since": "v11.0.0+wave9_dev209",
        "implementation_status": "full",
    },
    "PN34": {
        "title": "WorkspaceManager runtime lock relaxation (PN33 companion for runtime decode)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_PN34_WORKSPACE_LOCK_RELAX",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Companion to PN33 — same root cause class but on the runtime "
            "decode path. PN33 fixes BOOT-time _dummy_sampler_run "
            "under-counting; PN34 relaxes the strict "
            "WorkspaceManager._ensure_workspace_size AssertionError that "
            "still fires at runtime decode "
            "(turboquant_attn.py:1350:_decode_attention) on rare paths. "
            "Direct port of noonghunna's club-3090 setup-time sidecar "
            "patch_workspace_lock_disable.py. "
            "Retired 2026-05-14 PR sweep: this entry is a near-duplicate "
            "of SNDR_WORKSPACE_001 (same fault site, same warn+grow "
            "shape, same vllm.v1.worker.workspace anchor). Both were "
            "carried side-by-side during the v7 → v11 migration; the "
            "audit identified PN34 as the older variant. Consolidated "
            "into SNDR_WORKSPACE_001 going forward. Upstream replacement "
            "is vllm#42551 (jasonboukheir, DRAFT — pre-reserve + non-"
            "raising try API in _decode_attention) — see "
            "SNDR_WORKSPACE_001 credit for the merge-and-retire plan."
        ),
        "upstream_pr": 42551,  # 2026-05-14 PR sweep — same retire trigger as SNDR_WORKSPACE_001
        "requires_patches": ["PN33"],
        "superseded_by": "SNDR_WORKSPACE_001",
        "lifecycle": "retired",  # 2026-05-14 PR sweep audit — duplicate of SNDR_WORKSPACE_001
        "vllm_version_range": ">=0.20.1rc1.dev16+g7a1eb8ac2,<0.20.2rc1.dev338+gbf0d2dc6d",  # was active on dev16-dev209; retired on dev338 pin in favor of SNDR_WORKSPACE_001
        "implementation_status": "retired",
    },
    "PN33": {
        "title": "Spec-decode warmup K-aware sizing (vllm#37521 extended to MTP/ngram)",
        "tier": "community",
        "family": "worker",
        "env_flag": "GENESIS_ENABLE_PN33_SPEC_DECODE_WARMUP_K",
        "default_on": True,
        "category": "spec_decode",
        "credit": (
            "Backport of vllm-project/vllm#37521 (itailang, OPEN at "
            "backport time 2026-05-02) EXTENDED beyond use_eagle() "
            "gate to cover all spec-decode methods (EAGLE + MTP + "
            "ngram + draft-model). Root-cause fix: gpu_model_runner."
            "_dummy_sampler_run() warmed up with dummy K=1 instead of "
            "real num_speculative_tokens, causing (a) KV-cache profile "
            "to over-estimate available headroom → mid-stream OOM via "
            "propose_draft_token_ids (ampersandru, club-3090#16 "
            "2026-05-01) AND (b) TurboQuant WorkspaceManager lock fails "
            "when real spec-decode tries to grow workspace beyond "
            "warmup-reserved size (noonghunna, club-3090 disc #19 "
            "2026-05-01). Same root cause for both bugs; one fix "
            "closes both. Default ON when spec-decode active — real "
            "correctness fix, not experimental. Disable via "
            "GENESIS_DISABLE_PN33_SPEC_DECODE_WARMUP_K=1 if K-sized "
            "warmup itself OOMs (better-than-runtime-OOM diagnosis)."
        ),
        "upstream_pr": 37521,
        "applies_to": {
            # Only fires when speculative_config is present at runtime.
            # The text-patch site itself is gated `if self.speculative_config:`
            # so non-spec-decode boots are NULL on this path.
        },
        # Promoted 2026-05-12 (Wave 9 dev209 + STABLE-prep): full ratchet
        # satisfied — `register_for_manifest()` added in wiring;
        # anchor_manifest.json covers PN33.Sub-1 (gpu_model_runner.py).
        # Correctness fix (not perf-experimental). Production-validated
        # default_on across Wave 6→9 + dev93/dev209 + both 27B+35B PROD
        # with zero regressions. EXTENDED upstream vllm#37521 beyond
        # use_eagle() to cover all spec-decode methods (EAGLE + MTP +
        # ngram + draft-model) so Genesis stays ahead until upstream
        # broadens its merge — disabling carries real correctness risk
        # on MTP/ngram/draft paths.
        "apply_module": "vllm.sndr_core.integrations.worker.pn33_spec_decode_warmup_k",
        "lifecycle": "stable",
        "stable_kind": "text-patch",
        "stable_since": "v11.0.0+wave9_dev209",
        "implementation_status": "full",
    },
    "PN32": {
        "title": "GDN _forward_core chunked-prefill v2 (Cliff 2 fix for single-24GB-GPU OOM)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN32_GDN_CHUNKED_PREFILL",
        "default_on": False,
        "category": "hybrid",
        # NOTE: v2 (v7.69) supersedes v1 (v7.65). v1 chunked at the WRONG
        # level (forward_cuda outer, didn't propagate cu_seqlens to inner
        # FLA call → empirically OOM'd EARLIER on club-3090 cross-rig).
        # v2 chunks _forward_core directly with chunk-local cu_seqlens
        # and threaded initial_state. See club-3090#19 finding 3.
        #
        # COMPOSITION: v2 chunks the OUTER FLA call (chunk_gated_delta_rule).
        # P103 chunks INSIDE chunk_gated_delta_rule_fwd's h tensor. Both
        # default OFF, COMPLEMENTARY. Recommended together for single-24GB-
        # GPU users hitting Cliff 2 (>50K single-prompt prefill on 1×3090
        # /4090/5090).
        #
        # DEPENDENCY: P28 (legacy persistent buffer pool) conflicts with
        # PN32 v2 — both modify gdn_linear_attn.py overlapping paths.
        # Operator MUST disable P28 before enabling PN32. P28 IS in this
        # dispatcher (legacy lifecycle since v7.65) — symmetric back-link
        # declared via conflicts_with below.
        "credit": (
            "Genesis-original v7.69 v2 (2026-05-02) — Cliff 2 fix per "
            "noonghunna's CLIFF2_INVESTIGATION_20260430.md + cross-rig "
            "club-3090#19 finding 3. v2 supersedes v1 (v7.65) which "
            "chunked at wrong level. v2: when single-seq prefill T > "
            "16384 (env-tunable), splits chunk_gated_delta_rule call "
            "into chunks of 8192. Each chunk: slice query/key/value/g/"
            "beta along T, build chunk-local cu_seqlens=[0, chunk_len], "
            "thread initial_state via prior chunk's last_recurrent_state, "
            "concat outputs. Multi-seq prefill bypasses to original. "
            "Default OFF. Composes with P103 (P103 chunks INSIDE FLA "
            "kernel; PN32 chunks the FLA CALL). Recommended pairing for "
            "single-24GB-GPU Cliff 2: GENESIS_ENABLE_P103=1 + GENESIS_"
            "ENABLE_PN32_GDN_CHUNKED_PREFILL=1. Cross-rig validation "
            "required (our 2×A5000 PROD with TP=2 doesn't hit Cliff 2)."
        ),
        "upstream_pr": None,
        "applies_to": {
            # Triggers in any hybrid GDN model with long single-prompts.
            # NULL on non-GDN paths (no GDN layers in 35B Qwen3MoE).
        },
        "requires_patches": [],
        "conflicts_with": ["P28", "PN108"],  # PN32+PN108 both touch GDN chunked prefill orchestrator — incompatible
        "apply_module": "vllm.sndr_core.integrations.attention.gdn.pn32_gdn_chunked_prefill",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN102": {
        "title": "PrefetchOffloader pinned-allocator prewarm pool",
        "tier": "community",
        "family": "offload",
        "env_flag": "GENESIS_ENABLE_PN102_PARAM_POOL",
        "default_on": False,
        "category": "memory_pool",  # was "performance"; normalize (PrefetchOffloader pinned-allocator prewarm pool)
        # Relevant only when --cpu-offload-gb > 0. Without prewarm,
        # vllm's PrefetchOffloader calls torch.empty_strided(pin_memory=True)
        # once per offloaded param: 27B INT4 Qwen3.6 with cpu_offload_gb=8
        # has ~768 cudaHostAlloc calls × ~50 ms each ≈ 38 s of pure
        # pinning overhead before the model is ready. PN102 prewarms a
        # single contiguous slab (configurable via
        # GENESIS_PN102_PREWARM_MB, default 1024) so the subsequent
        # per-param pinnings are served from PyTorch's cached pinned
        # allocator — drops startup overhead to single-digit seconds.
        # Idempotent; harmless if cpu_offload is off (then the
        # PrefetchOffloader is never touched).
        "credit": (
            "Genesis-original 2026-05-14. Closes the 38 s startup-wall "
            "introduced by per-param `cudaHostAlloc` when "
            "`--cpu-offload-gb > 0` on AutoRound INT4 models. Companion "
            "to PN104 (cpu_offload → Prefetch redirect) and PN105 "
            "(AutoRound metadata exclusion)."
        ),
        "upstream_pr": None,
        "applies_to": {},
        "requires_patches": [],
        "conflicts_with": [],
        "apply_module": "vllm.sndr_core.integrations.offload.pn102_pinned_alloc_pool",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN204": {
        "title": "GDN dual-stream input projection (port of vllm#42301)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN204_DUAL_STREAM_INPROJ",
        "default_on": False,
        "category": "kernel_perf",  # was "performance"; normalize (GDN dual-stream input projection kernel-level perf win)
        # PN204 replaces retired P7 (status=skipped, deferred — raw
        # torch.cuda.Stream not SymPy-graphable inside torch.compile
        # fullgraph). PN204 uses upstream vllm.utils.multi_stream_utils
        # .maybe_execute_in_parallel which is torch.compile-safe and
        # available in the pinned nightly dcacdf9a.
        #
        # Direct port of vllm PR #42301. Single text-patch anchor on
        # gdn_linear_attn.py::forward_cuda Part 1: replaces the serial
        # in_proj_qkvz/in_proj_ba pair with the parallel helper. Stream
        # and events are created lazily on the first forward call (naive
        # __init__ allocation crashed worker with a torch.Event type
        # error in our pinned vLLM). Auto-SKIPs when upstream lands
        # #42301 (drift marker `_in_proj_aux_stream`).
        "credit": (
            "Port of vllm-project/vllm#42301 (open as of 2026-05-14). "
            "Upstream measures -2.9% TPOT at qps=0.5 on Qwen3.5-35B-A3B "
            "via overlapping in_proj_qkvz/in_proj_ba GEMMs on an aux "
            "CUDA stream. PN204 ports the same change as a Genesis "
            "text-patch to unblock the win without waiting for upstream "
            "merge."
        ),
        "upstream_pr": 42301,
        "applies_to": {
            # Hybrid GDN models (Qwen3.5/3.6) on CUDA-alike platforms.
        },
        "requires_patches": [],
        # Mutually exclusive with retired P7 (same forward_cuda Part 1
        # target). Operator must keep P7 disabled when enabling PN204.
        "conflicts_with": ["P7"],
        "apply_module": "vllm.sndr_core.integrations.attention.gdn.pn204_dual_stream_inproj",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN108": {
        "title": "GDN fused_recurrent prefill dispatch (Cliff 2 memory-bound fix)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN108_FUSED_RECURRENT_PREFILL",
        "default_on": False,
        "category": "hybrid",
        # PN108 takes a different approach from PN32 / PN59. Instead of
        # chunking the chunk_gated_delta_rule call (PN32) or the inner
        # kernel orchestrator (PN59), PN108 switches BACKENDS for long
        # single-sequence prefill: it dispatches to fla's
        # fused_recurrent_gated_delta_rule, which has NO chunk-state h
        # buffer at all (the Cliff 2 OOM trigger).
        #
        # Trade-off: fused_recurrent is purely token-recurrent compute,
        # so prefill is ~3-8× slower above the threshold. In exchange:
        # zero h-tensor allocation, predictable memory across any T,
        # no fragmentation from a chunked-dispatch loop. For long-context
        # workflows on memory-constrained single-GPU rigs (1×A5000 24GB,
        # 1×3090, 1×4090), this is the only path that achieves stable
        # 150-200K prefill without TP=2.
        #
        # Mutually exclusive with PN32 (both patch the same _forward_core
        # prefill branch in gdn_linear_attn.py). PN108 also conflicts
        # with P28 for the same reason as PN32 — operator must disable
        # P28 before enabling PN108.
        "credit": (
            "Genesis-original 2026-05-14 — Cliff 2 memory-bound fix on "
            "single 24GB GPU. PN32/PN59 attempted to keep the chunkwise-"
            "parallel kernel and chunk dispatch around it; both either "
            "hit anchor drift (PN59, upstream added `core_attn_out` "
            "param) or added a torch.cat memory peak that didn't fit on "
            "saturated single-card budget (PN32). PN108 sidesteps by "
            "switching to fla.fused_recurrent_gated_delta_rule for long "
            "single-seq prefill — same output contract (B,T,HV,V), no "
            "chunk-state h buffer. Threshold env-tunable via GENESIS_"
            "PN108_FUSED_RECURRENT_THRESHOLD (default 32768)."
        ),
        "upstream_pr": None,
        "applies_to": {
            # Hybrid GDN/DeltaNet models with single-sequence long prefill.
            # NULL on non-GDN paths or short prompts.
        },
        "requires_patches": [],
        "conflicts_with": ["P28", "PN32"],
        "lifecycle": "retired",  # docstring says TOMBSTONED — fla recurrent kernel design conflict; synced to registry 2026-05-14
        "apply_module": "vllm.sndr_core.integrations._retired.pn108_fused_recurrent_prefill",
        "retired_waiver": True,  # design conflict — see docstring tombstone
        "implementation_status": "full",
    },
    "PN31": {
        "title": "FA varlen persistent out buffer (issue #15, sister to P38)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_PN31_FA_VARLEN_PERSISTENT_OUT",
        "default_on": False,
        "category": "memory_pool",
        "credit": (
            "Genesis-original sister patch to P38 (K_full/V_full persistent "
            "buffers). Closes issue #15 — OOM at flash_attn_varlen_func on "
            "budget-constrained single-GPU configs. Per-shape persistent "
            "out buffer eliminates per-call malloc pressure inside FA C "
            "extension. Memory cost: ~16-64 MiB per shape × layer. NULL "
            "impact on 2×A5000 PROD (we have headroom); designed for "
            "1×3090 / 1×4090 single-GPU community users."
        ),
        "upstream_pr": None,
        "applies_to": {
            # Triggers when TurboQuant attention is active. NULL on
            # non-TQ paths (FP8 KV, BF16 KV).
        },
        "conflicts_with": [],
        "requires_patches": [],
        "apply_module": "vllm.sndr_core.integrations.attention.turboquant.pn31_fa_varlen_persistent_out",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN30": {
        "title": "DS conv state layout + spec-decode AL>1 fix (issue #17)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN30_DS_LAYOUT_SPEC_DECODE",
        "default_on": False,
        "category": "model_correctness",
        "credit": (
            "Genesis-original fix for issue #17 (noonghunna, 2026-05-01). "
            "Replaces upstream NotImplementedError raise in "
            "`get_conv_copy_spec` for DS conv state layout + "
            "num_accepted_tokens > 1 (every spec-decode AL>1 prefill on "
            "DS-enabled hybrid GDN configs). Two-file text-patch: "
            "(1) mamba_utils.py uses .contiguous() copy + module-level "
            "temp tensor list; (2) v1/worker/mamba_utils.py wraps "
            "do_mamba_copy_block with stream sync + list clear after "
            "batch_memcpy. Cost: ~10-50us per batch when path active. "
            "Closes 50/50 LCB v6 failure on structured-CoT workloads."
        ),
        "upstream_pr": None,
        "applies_to": {
            # Triggers in any hybrid GDN model with DS layout + spec-decode.
            # Genesis A5000 PROD doesn't have --structured-outputs-config so
            # may not exercise this path; community single-3090 + structured
            # CoT does.
        },
        "conflicts_with": [],
        "requires_patches": [],
        "apply_module": "vllm.sndr_core.integrations.attention.gdn.pn30_ds_layout_spec_decode_align",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P67c": {
        "title": "Per-row vote sparse-V integration into P67 split-M kernel",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_P67_SPARSE_V",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Genesis-original 2026-05-01 — synthesizes PN26b proven uniform-"
            "scalar `if` pattern (Triton 3.6 scf.if), TRT-LLM #9821 sink "
            "protection design, TheTom #41422 threshold=0 bit-exact contract. "
            "Per-q_t skip via `if SPARSE_V: ...` constexpr-DCE'd to nothing "
            "at SPARSE_V=0 → byte-equivalent to pre-sparse-V P67. "
            "When SPARSE_V=1 + threshold=0: bit-exact (P_t = exp2(...) >= 0, "
            "so `p_t_max < 0` always False). When threshold > 0: per-q_t "
            "max-prob check skips V@P tl.dot for cold tiles past sink window. "
            "Greenfield in spec-decode K+1 verify (no upstream impl exists). "
            "Expected gain: +5-22% on long-context (16K+); NULL on short ctx."
        ),
        "upstream_pr": None,
        "applies_to": {"is_turboquant": [True]},
        "requires_patches": ["P67"],
        "conflicts_with": [],
        "apply_module": "vllm.sndr_core.integrations.attention.turboquant.p67c_sparse_v",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN29": {
        "title": "GDN chunk_o scale-fold (vllm#41446 pattern (c))",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN29_GDN_SCALE_FOLD",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Backport of vllm#41446 (zobinHuang, OPEN) pattern (c) only. "
            "Folds scale multiply in `chunk_fwd_kernel_o`: "
            "`b_o = (b_o + tl.dot(b_A, b_v)) * scale` instead of "
            "`b_o = b_o * scale + tl.dot(b_A, b_v) * scale`. "
            "One fewer fp32 multiply per inner iter. Distributive on "
            "fp32 accumulators (drift bounded by 1-2 ULP per element). "
            "Triton compiler does NOT auto-fuse across the +/- boundary, "
            "so explicit fold = guaranteed save. Hardware-agnostic; "
            "PR is MI300X-targeted but pattern (c) is NVIDIA-Triton "
            "compatible. Genesis-applicable: hybrid GDN models "
            "(Qwen3.5/3.6 27B); no-op on Qwen3MoE 35B."
        ),
        "upstream_pr": 41446,
        "applies_to": {
            # Triggers in any model using FLA chunk_fwd_kernel_o (hybrid
            # GDN). On Qwen3MoE without GDN, the kernel never fires →
            # patch is silently no-op even if env enabled.
        },
        "apply_module": "vllm.sndr_core.integrations.attention.gdn.pn29_gdn_chunk_o_scale_fold",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN11": {
        "title": "GDN a/b contiguity in fix_query_key_value_ordering (vllm#41142)",
        "tier": "community",
        "family": "attention.gdn",
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
        "apply_module": "vllm.sndr_core.integrations.attention.gdn.pn11_gdn_a_b_contiguous",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN12": {
        "title": "FFN intermediate scratch pool — Cliff 1 fix on TQ3 path",
        "tier": "community",
        "family": "kernels",
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

            # [Genesis pin-gate 2026-05-11] PROD-active (GroupAB component).
            # Validated dev9 → dev93. Self-retires when #34207 merges.
            "vllm_version_range": (">=0.20.0", "<0.21.0"),
        },
        "apply_module": "vllm.sndr_core.integrations.kernels.pn12_ffn_intermediate_pool",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN19": {
        "title": "Scoped max_split_size_mb during model load (vllm#41268)",
        "tier": "community",
        "family": "memory",
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
        "superseded_by": "vllm#41268 (merged 2026-04-30, byte-equivalent on dev209+g5536fc0c0)",
        "vllm_version_range": "<0.20.2rc1.dev209+g5536fc0c0",  # active before upstream merge in dev209
        "apply_module": "vllm.sndr_core.integrations._retired.pn19_scoped_max_split",
        "applies_to": {
            # Always applicable on CUDA. Self-detects torch < 2.11 lack
            # of _accelerator_setAllocatorSettings and falls through
            # unchanged.

            # [Iron rule #11 retire 2026-05-11] Deep-diff confirmed
            # #41268 byte-equivalent in dev209: `_scoped_allocator_max_split`
            # symbol present in vllm/v1/worker/gpu_worker.py (upstream's
            # context-manager-based scoped allocator with same 20 MiB
            # minimum + prior-value restoration semantics that PN19
            # implemented). Wiring auto-skips correctly; pin-gate
            # formalizes the retire boundary. Was actually mergeable into
            # dev93+ already (merge date 2026-04-30 < dev93 SHA 2026-05-07)
            # but lifecycle wasn't updated until this audit.
            # Cleanup: delete patch file in next refactor pass.
            "vllm_version_range": "<0.20.2rc1.dev93",
        },
        "lifecycle": "retired",  # 2026-05-11: byte-equivalent with #41268 in dev209
        "implementation_status": "full",
    },
    "PN23": {
        "title": "DFlash combine_hidden_states dtype cast (vllm#40334)",
        "tier": "community",
        "family": "spec_decode",
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
        "apply_module": "vllm.sndr_core.integrations.spec_decode.pn23_dflash_combine_hidden_dtype",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN21": {
        "title": "DFlash SWA support partial backport (vllm#40898)",
        "tier": "community",
        "family": "spec_decode",
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
        "apply_module": "vllm.sndr_core.integrations.spec_decode.pn21_dflash_swa_support",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN22": {
        "title": "Local argmax for TP draft (vllm#39419 backport)",
        "tier": "community",
        "family": "spec_decode",
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
        "apply_module": "vllm.sndr_core.integrations.spec_decode.pn22_local_argmax_tp",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN24": {
        "title": "DFlash aux layer +1 indexing fix (vllm#40727)",
        "tier": "community",
        "family": "worker",
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
        "apply_module": "vllm.sndr_core.integrations.worker.pn24_dflash_aux_layer_indexing",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN28": {
        "title": "merge_attn_states NaN guard (vllm#39148 backport)",
        "tier": "community",
        "family": "kernels",
        "env_flag": "GENESIS_ENABLE_PN28_MERGE_ATTN_NAN_GUARD",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Backport of vllm#39148 (jasonkim8652, OPEN 2026-05-01). "
            "Branchless NaN guard in Triton merge_attn_states kernel for "
            "both-LSE-(-inf) edge case (zero-context-length chunked prefill). "
            "Without guard: NaN propagates through exp()/division and silently "
            "corrupts output — one bad token can break tool-call JSON parsing. "
            "Fix: clamp max_lse to -1e30 finite floor + add 1e-10 epsilon to "
            "denominator. Quality-only — no perf impact. CUDA merge_attn_states "
            "kernel already had this guard; PN28 brings Triton to parity."
        ),
        "upstream_pr": 39148,
        "applies_to": {},
        "conflicts_with": [],
        "requires_patches": [],
        "apply_module": "vllm.sndr_core.integrations.kernels.pn28_merge_attn_states_nan_guard",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P15B": {
        "title": "FA varlen max_seqlen_k clamp on TQ path (Issue #15 fix)",
        "tier": "community",
        "family": "memory",
        "env_flag": "GENESIS_ENABLE_P15B_FA_VARLEN_CLAMP",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Genesis-original 2026-05-01 fix for noonghunna's Issue #15. "
            "PN17 clamps max_seqlen_k on the FA2 backend path, but TurboQuant "
            "code path bypasses PN17's coverage by calling vllm_flash_attn's "
            "vendored wrapper via turboquant_attn.py:_flash_attn_varlen. P15B "
            "extends the same clamp logic to that callsite via text-patch — "
            "computes actual span from cu_seqlens_k and clamps max_seqlen_k "
            "before invocation. Prevents 50 MiB workspace OOM on long-context "
            "continuation-prefill on tight VRAM (24 GB consumer cards). "
            "Trade-off: adds one GPU->CPU sync per call on the infrequent "
            "continuation-prefill path."
        ),
        "upstream_pr": None,
        "applies_to": {},
        "conflicts_with": [],
        "requires_patches": [],
        "apply_module": "vllm.sndr_core.integrations.memory.p15b_fa_varlen_clamp",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P38B": {
        "title": "P38 compile-safe in-source hook (Issue #14 fix)",
        "tier": "community",
        "family": "memory",
        "env_flag": "GENESIS_ENABLE_P38B_COMPILE_SAFE",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Genesis-original 2026-05-01 fix for noonghunna's Issue #14. "
            "Root cause: aot_compile_fullgraph captures _continuation_prefill "
            "original body at engine init; Python class-attribute rebind "
            "(P38's mechanism) doesn't propagate to compiled artifact. "
            "P38B injects an in-source delegate hook at the start of "
            "_continuation_prefill body via text-patch. Hook calls a "
            "dispatcher that returns Genesis result OR None (fall-through). "
            "Source-level edit means aot_compile captures the hook itself. "
            "Affects ALL TQ KV users with V0/V1 compile pipeline; fp8 KV "
            "configs unaffected (different code path). Composes with P38 "
            "(both share _genesis_continuation_prefill impl)."
        ),
        "upstream_pr": None,
        "applies_to": {},
        "conflicts_with": [],
        "requires_patches": [],  # P38 install order: P38 first (provides impl), P38B second (installs hook)
        "apply_module": "vllm.sndr_core.integrations.memory.p38b_compile_safe_hook",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN26b": {
        "title": "Sparse-V tile-skip Genesis kernel (BLASST λ=a/L for SM86)",
        "tier": "community",
        # Family corrected 2026-05-12: kernel operates on attention V tensors
        # (sparse-V tile-skip in the TQ decode path), not on memory pools.
        # Wiring lives in integrations/attention/turboquant/pn26_sparse_v_kernel.py.
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_PN26_SPARSE_V",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Genesis-original Triton kernel fork — first sparse-V tile-skip "
            "deployed for SM86 (Ampere consumer). Synthesized from 4-agent "
            "research 2026-05-01: vllm#41422 (TheTom, AMD-only validated) "
            "design template + BLASST arXiv 2512.12087 (Yuan et al. Dec 2025) "
            "λ=a/L threshold formula + tq-kv reference (CUDA, SM86-compatible) "
            "acc*re_scale skip semantics + StreamingLLM (arXiv 2309.17453) "
            "sink token protection (first 4 KV positions never skipped). "
            "Mechanism: when tl.max(p) < threshold for a KV tile, skip V load + "
            "dequant + weighted sum, just decay accumulator. Online softmax "
            "denominator/max still update so totals stay numerically exact "
            "for non-skipped tiles. Composes with PN26 main (centroids "
            "prebake) + P98 (workspace revert) + P67 (multi-query — separate "
            "code path, not affected). Default OFF; opt-in via "
            "GENESIS_ENABLE_PN26_SPARSE_V=1 + GENESIS_PN26_SPARSE_V_THRESHOLD "
            "(fixed) OR GENESIS_PN26_SPARSE_V_SCALE_FACTOR (BLASST adaptive)."
        ),
        "upstream_pr": 41422,
        "applies_to": {},
        "conflicts_with": [],
        "requires_patches": [],
        "apply_module": "vllm.sndr_core.integrations.attention.turboquant.pn26_sparse_v_kernel",
        "lifecycle": "research",
        "research_note": (
            "First sparse-V tile-skip kernel deployed on SM86 (Ampere). "
            "Upstream vllm#41422 (TheTom) is AMD-only validated; Genesis "
            "port adds SM86 Triton path + BLASST adaptive λ=a/L threshold "
            "+ StreamingLLM sink-token protection. Default OFF until "
            "long-context (32k+) bench data lands — at typical 4k-8k "
            "decode the per-tile skip-probability check overhead exceeds "
            "the savings. Quality risk: sparse-V drops V loads for tiles "
            "below threshold; for biased workloads (concentrated attention) "
            "this is exact, but for diffuse-attention workloads the "
            "BLASST adaptive threshold can drop signal. Promotion requires "
            "(a) 32k+ ctx bench data, (b) tool-call score ≥ 8/10 across "
            "3+ models, (c) decode_tpot regression < 2% on short-ctx."
        ),
        "implementation_status": "full",
    },
    "PN27": {
        "title": "Revert MoERunnerInterface PluggableLayer (vllm#41440)",
        "tier": "community",
        "family": "moe",
        "env_flag": "GENESIS_ENABLE_PN27_REVERT_PLUGGABLE_MOE",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Backport of vllm#41440 (auto-generated CI failure analyzer, OPEN "
            "2026-05-01). Reverts vllm#35178 (b55b2652, merged 2026-04-30) "
            "which made MoERunnerInterface inherit from PluggableLayer + "
            "introduced DefaultMoERunner split/recombine. Issue #41306 "
            "reports +21% TPOT / +59% TTFT / -19% throughput on Mixtral-8x7B "
            "(8× H200), with bnellnm confirming `--moe-backend=triton` "
            "restores v0.19 perf. Our pin (0.20.1rc1.dev16+g7a1eb8ac2) "
            "predates the merge by 2 days — PN27 is a PROACTIVE SCAFFOLD "
            "for the case when we eventually pin-bump past b55b2652 BEFORE "
            "#41440 (or equivalent fix-forward) merges. On our current pin, "
            "all 3 sub-patches SKIP as intended (anchors are pre-#35178)."
        ),
        "upstream_pr": 41440,
        "applies_to": {},
        "conflicts_with": [],
        "requires_patches": [],
        "apply_module": "vllm.sndr_core.integrations.moe.pn27_revert_pluggable_moe",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN26": {
        "title": "TQ unified perf pack (centroids prebake + sparse V scaffold)",
        "tier": "community",
        "family": "attention.turboquant",
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
        "apply_module": "vllm.sndr_core.integrations.attention.turboquant.pn26_sparse_v_kernel",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN25": {
        "title": "SiluAndMul.forward_native opaque-op pool (Cliff 1 mech B compile path)",
        "tier": "community",
        "family": "kernels",
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
        "apply_module": "vllm.sndr_core.integrations.kernels.pn25_silu_inductor_safe_pool",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN17": {
        "title": "FA2 softmax_lse runtime clamp (Cliff 1 mechanism A, Issue #11)",
        "tier": "community",
        "family": "attention.flash",
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
        "apply_module": "vllm.sndr_core.integrations.attention.flash.pn17_fa2_softmax_lse_clamp",
        "lifecycle": "experimental",
        # [Performance verified 2026-05-11 — DO NOT DISABLE]
        # Differential bench on 35B/dev209 (canonical genesis_bench_suite
        # --quick --ctx 8k, 5×5×1024):
        #   PN17=1: wall_TPS 231.08, decode_TPOT 4.01 ms, CV ~4-9%
        #   PN17=0: wall_TPS 178.95, decode_TPOT 5.15 ms, CV ~8-11%
        #   Delta: -22.6% TPS, +28.5% TPOT when DISABLED
        # PN17 is a PERFORMANCE WIN, not a regression source. The naive
        # static-analysis worry about per-call .item() sync was wrong:
        # CUDA stream scheduling absorbs the sync, while the alternative
        # (FA2 allocating softmax_lse for max_seqlen_k=320K per call) is
        # the actual dominant cost on long-context configs. The clamp
        # to seqused_k.max() reduces the per-call buffer 10-1000x for
        # decode workloads where actual context << max_model_len.
        # Iron rule #10 ("study before disabling") was the right call —
        # the empirical bench reversed the static-analysis hypothesis.
        # KEEP PN17 enabled on PROD. No optimization needed at this site;
        # the per-call .item() sync IS the cheap path here.
        "implementation_status": "full",
    },
    "PN16": {
        "title": "Lazy-reasoner request hook v2 (cache-safe; V7 max_tokens cap)",
        "tier": "community",
        "family": "middleware",
        "env_flag": "GENESIS_ENABLE_PN16_LAZY_REASONER",
        "default_on": False,
        "category": "request_middleware",
        "credit": (
            "Genesis-original 2026-04-29; v2 rearchitecture 2026-05-09 "
            "(Wave 6 closure). Per-request hybrid policy deciding "
            "whether the model's `<think>...</think>` block adds value. "
            "v2 production paths (cache-safe): "
            "V3 (client override) respects explicit "
            "chat_template_kwargs.enable_thinking; "
            "V5 (soft cap) injects a concise-reasoning hint into the "
            "last user message when GENESIS_PN16_MAX_THINKING_TOKENS > 0; "
            "V7 (NEW — max_tokens hard cap) clamps request.max_tokens "
            "to GENESIS_PN16_CLASSIFIER_MAX_TOKENS (default 0=off) when "
            "the short-prompt classifier hits — provides a hard ceiling "
            "on response length without mutating the chat template, so "
            "CUDA graph dispatch and MTP draft compatibility are "
            "preserved. "
            "⚠ V1 (template-flag mutation) RETIRED from default on "
            "2026-05-09 — Wave 6 bench measured 28%% wall_TPS drop with "
            "6× CV amplification (236.24 → 166.25 TPS, 6.3% → 37.6% CV) "
            "on Sander's 35B PROD because forcing enable_thinking=False "
            "renders a different chat-template prefix → CUDA graph "
            "dispatch miss + MTP draft divergence. V1 still callable as "
            "legacy via GENESIS_PN16_V1_LEGACY=1 (emits one-shot WARN at "
            "first hit). "
            "V4 (LogitsProcessor strict cap) deferred — vllm v1 rejects "
            "custom logits processors with speculative_config set. "
            "V6 (streaming-side `<think>` truncator) — future work for "
            "deterministic TTFT bounding. "
            "Stats via `vllm.sndr_core.middleware.lazy_reasoner.get_stats()`."
        ),
        "upstream_pr": None,
        "applies_to": {
            # [Genesis pin-gate 2026-05-11] PROD-active (V2 rearchitecture,
            # Wave 6 closure). V1 retired, V5/V7 cache-safe paths active.
            "vllm_version_range": (">=0.20.0", "<0.21.0"),
        },
        "apply_module": "vllm.sndr_core.integrations.middleware.pn16_v6_streaming_truncator",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN16_V6": {
        "title": "PN16 V6 — streaming `<think>` token-budget enforcer (Sprint 4)",
        "tier": "community",
        "family": "middleware",
        "env_flag": "GENESIS_ENABLE_PN16_V6_STREAMING_TRUNCATOR",
        "default_on": False,
        "category": "request_middleware",
        "implementation_status": "full",
        "source": "genesis_original",
        # Explicit apply_module — registry sub-id "PN16_V6" doesn't auto-resolve
        # via the pn{digit}_ regex; spell it out so iter_patch_specs() finds it.
        "apply_module": (
            "vllm.sndr_core.integrations.middleware.pn16_v6_streaming_truncator"
        ),
        "credit": (
            "Genesis-original Sprint 4 closure 2026-05-09 — deep fix for "
            "london_think failure class. Per-request stateful truncator "
            "(vllm/sndr_core/middleware/think_streaming_truncator.py) wraps "
            "OpenAIServingChat.chat_completion_stream_generator via "
            "class-rebind. When a request has tools attached AND "
            "GENESIS_PN16_MAX_THINKING_STREAM_TOKENS > 0, counts "
            "delta.reasoning_content chunks; once budget exceeded, drops "
            "subsequent reasoning chunks and emits one-shot [Genesis] "
            "truncation note as delta.content. tool_calls + plain content "
            "chunks always pass. Bounds client-visible TTFT-to-tool-call "
            "deterministically when the model ignores V8's prompt-engineering "
            "budget hint. Stacks cleanly on top of V8 — V8 nudges, V6 "
            "enforces. Cache-safe (no chat_template mutation), "
            "compute-neutral (model still generates internally; only "
            "client-visible stream is truncated)."
        ),
        "upstream_pr": None,
        "lifecycle": "experimental",
        "requires_patches": [],
        "conflicts_with": [],
    },
    "PN14": {
        "title": "TQ decode IOOB safe_page_idx clamp (vllm#40074)",
        "tier": "community",
        "family": "attention.turboquant",
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
            # [Genesis pin-gate 2026-05-11] Defensive on Ampere; load-
            # bearing on planned Blackwell upgrade. Validated dev9 → dev93.
            # Self-retires via marker when #40074 merges.
            "vllm_version_range": (">=0.20.0", "<0.21.0"),
        },
        "apply_module": "vllm.sndr_core.integrations.attention.turboquant.pn14_tq_decode_oob_clamp",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    # PN13 entry moved to legacy/retired section below (lifecycle: retired_2026-05-04)
    # Reason: vllm 0.20.2 commit c2fb013 merged identical change (#41235).
    # See PN13 entry near line 1289 for retirement metadata.

    "P94": {
        "title": "Spec-decode prepare_next_token_ids_padded zero-alloc (vllm#41043)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_P94",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport of vllm#41043 (wangluochao902, MERGED 2026-04-29). Removes GPU->CPU .tolist() sync + list-comp Python objects + np.array allocation in LLMBaseProposer.prepare_next_token_ids_padded hot path. PR author measured P99 TPOT -9.3% on Llama-3.1-8B + Eagle3 TP=4. For our MTP K=3 single-stream: expected +2-4% wall TPS + tighter CV. SUPERSEDED-ON-MERGE: when our pin advances past the merge SHA the patch will SKIP cleanly via drift detection on the original .tolist() anchor — at that point delete the wiring file + this entry.",
        "upstream_pr": 41043,
        "superseded_by": "vllm#41043 (merged 2026-04-29, byte-identical with deep-diff confirmed Wave 8 audit) — patch retained as audit trail",
        "vllm_version_range": "<0.20.2rc1.dev93+g51f22dcfd",  # active before upstream merge in dev93
        "apply_module": "vllm.sndr_core.integrations._retired.p94_spec_decode_zero_alloc",
        "applies_to": {
            # Applies whenever spec-decode is active. All spec methods.

            # [Genesis iron-rule-#11 retire 2026-05-11 v2 audit] Wave 8
            # deep-diff confirmed BYTE-IDENTICAL with upstream #41043
            # (merged 2026-04-29, in dev9+ → dev93+ → dev209+). On
            # post-merge pins drift detector auto-skips (anchor gone).
            # Pin-gate tightened to anchor-correct upper bound.
            "vllm_version_range": "<0.20.2rc1.dev9",
        },
        "lifecycle": "retired",  # 2026-05-11 v2 audit: formalized byte-identical retire
        "implementation_status": "full",
    },
    # ════════════════════════════════════════════════════════════════════
    # 2026-05-14 — vLLM upstream PR sweep on dev338+gbf0d2dc6d nightly.
    # Four entries here (P108, P109, PN110, PN111). All backport open
    # upstream fixes that target either bugs in our hot path (P108, P109,
    # PN110) or a measurable perf win on a future operator profile
    # (PN111, align-mode only). Default-on choice is per-patch.
    # ════════════════════════════════════════════════════════════════════
    "P108": {
        "title": "MTP draft-loop stream synchronization (vllm#42603)",
        "tier": "community",
        "family": "spec_decode",
        "env_flag": "GENESIS_ENABLE_P108",
        "default_on": True,
        "category": "spec_decode",
        "credit": (
            "Backport of vllm#42603 (z1ying, OPEN 2026-05-14). Closes a "
            "cudaErrorIllegalAddress race in LLMBaseProposer.propose() "
            "where the input_ids / hidden_states buffer writes on the "
            "default stream were not synchronized before downstream "
            "attention kernels ran on a different stream (FlashInfer "
            "default). Reproduced upstream on Qwen3.6-27B-FP8 + RTX 5090. "
            "Genesis ships ON for any spec-decode method; cost is a "
            "single-stream synchronize() — ~0 in the common case where "
            "the producer kernel has already finished."
        ),
        "upstream_pr": 42603,
        "applies_to": {
            "spec_method_any": ["mtp", "eagle", "dflash"],
        },
        "implementation_status": "full",
        "apply_module": "vllm.sndr_core.integrations.spec_decode.p108_mtp_draft_stream_sync",
        "source": "vllm_pr_backport",
        "lifecycle": "experimental",
    },
    "P109": {
        "title": "sampling_params vocab-range validators (vllm#42614)",
        "tier": "community",
        "family": "serving",
        "env_flag": "GENESIS_ENABLE_P109",
        "default_on": True,
        "category": "stability",
        "credit": (
            "Backport of vllm#42614 (jperezdealgaba, OPEN 2026-05-14). "
            "Adds explicit validation of stop_token_ids and "
            "logprob_token_ids against model vocab size in "
            "SamplingParams.verify(). Out-of-vocab ids previously OOB'd "
            "the V2 Triton _bias_kernel and crashed the worker; with "
            "this patch the request bounces with a clear 400 instead. "
            "Defense-in-depth for the public Proxy-AI surface (Cline / "
            "OpenWebUI / LibreChat clients sometimes pass malformed "
            "stop_token_ids). Bit-identical for valid inputs."
        ),
        "upstream_pr": 42614,
        "applies_to": {},  # generic safety; always applicable
        "implementation_status": "full",
        "apply_module": "vllm.sndr_core.integrations.serving.p109_sampling_params_vocab_bounds",
        "source": "vllm_pr_backport",
        "lifecycle": "experimental",
    },
    "PN110": {
        "title": "BlockPool.free_blocks deduplication (vllm#42615)",
        "tier": "community",
        "family": "kv_cache",
        "env_flag": "GENESIS_ENABLE_PN110",
        "default_on": True,
        "category": "stability",
        "credit": (
            "Backport of vllm#42615 (AkCodes23, OPEN 2026-05-14). "
            "Deduplicates by id(block) in BlockPool.free_blocks() so a "
            "caller that passes the same KVCacheBlock twice (sliding-"
            "window + offload-connector race) does not double-decrement "
            "ref_cnt or double-append into the free queue. Composes "
            "cleanly with PN95/PN96/PN97 — same family, no anchor "
            "conflict (dedup happens before ref_cnt -= and before "
            "append_n). Warns when duplicates are observed."
        ),
        "upstream_pr": 42615,
        "applies_to": {},  # generic defensive guard; always applicable
        "implementation_status": "full",
        "apply_module": "vllm.sndr_core.integrations.kv_cache.pn110_block_pool_free_dedup",
        "source": "vllm_pr_backport",
        "lifecycle": "experimental",
    },
    "PN111": {
        "title": "Skip-mamba-postprocess GPU->CPU sync (align-mode; vllm#42574)",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_ENABLE_PN111",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Backport of vllm#42574 (mamingyuan-nv, OPEN 2026-05-14). "
            "Skips the per-decode-step blocking GPU->CPU sync of "
            "num_accepted_tokens in the mamba_cache_mode==align branch "
            "of GPUModelRunner._update_states_after_model_execute when "
            "the downstream postprocess_mamba is provably a no-op this "
            "step (upper-bound proof: num_accepted <= n_draft + 1, no "
            "Mamba block boundary crossed). Adds can_skip_mamba_"
            "postprocess() to mamba_utils.py. Reported +17.4% TPS / "
            "-13.7% ITL on Nemotron-Super-120B-A12B-NVFP4 MTP=3 on "
            "GB300. ⚠ Genesis PROD presets currently DO NOT set "
            "--mamba-cache-mode align — patch is a no-op there. Win "
            "materialises only after an operator opts into align mode."
        ),
        "upstream_pr": 42574,
        "applies_to": {
            "is_hybrid": True,
            "spec_method_any": ["mtp", "eagle"],
        },
        "implementation_status": "full",
        "apply_module": "vllm.sndr_core.integrations.attention.gdn.pn111_skip_mamba_postprocess_sync",
        "source": "vllm_pr_backport",
        "lifecycle": "experimental",
        "conflicts_with": [],
        "requires_patches": [],
    },
    "PN116": {
        "title": "TurboQuant prefill max_seq_len fallback fix (regressor: vllm#41434)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_PN116",
        "default_on": True,
        "category": "perf_hotfix",
        "credit": (
            "Genesis-original fix for an Ampere-specific regression "
            "introduced by vllm#41434 (Perf 3/n: eliminate GPU<->CPU "
            "syncs in attention impls, merged 2026-05-08). The PR's "
            "new TurboQuant prefill fast path uses a CPU-resident "
            "seq_lens upper bound when populated and falls back to "
            "`attn_metadata.max_seq_len` otherwise — but max_seq_len is "
            "the FULL-BATCH max (includes decodes), so the fallback "
            "feeds the attention kernel an inflated upper bound. On "
            "Hopper GB200 the PR measured +4.8% TPS (fast path "
            "dominant). On 2× A5000 Ampere + 35B-A3B-FP8 + TurboQuant "
            "k8v4 + MTP K=3 we measure −9.7% wall_TPS (241→218 between "
            "dev93 and dev209+). 27B INT4 + GDN is unaffected. "
            "HW-aware: applies on SM<9.0 only by default; on Hopper "
            "and newer the upstream behaviour wins and we self-skip "
            "(GENESIS_PN116_FORCE=1 to override)."
        ),
        "upstream_pr": 41434,  # this is the regressor — patch self-retires when upstream re-fixes fallback
        "applies_to": {
            "is_turboquant": True,  # patch site is turboquant_attn.py
        },
        "implementation_status": "full",
        "apply_module": "vllm.sndr_core.integrations.attention.turboquant.pn116_tq_prefill_maxseq_fallback",
        "source": "genesis_original",
        "lifecycle": "experimental",
        "conflicts_with": [],
        "requires_patches": [],
    },
    "PN118": {
        "title": "TurboQuant workspace graceful-fallback (vllm#42551, P99-compat)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_PN118",
        "default_on": True,
        "category": "stability",
        "credit": (
            "P99-compatible backport of vllm#42551 (jasonboukheir, OPEN "
            "2026-05-14). Closes an AssertionError on first decode request "
            "for partial-TQ models (16/64 TQ layers, e.g. Lorbus/Qwen3.6-"
            "27B-int4-AutoRound — named in the PR). Adds two new methods "
            "to WorkspaceManager: try_get_simultaneous (returns None on "
            "locked-undersized instead of raising) and reserve (pre-"
            "allocates every ubatch slot before lock_workspace snapshot). "
            "TurboQuant __init__ calls reserve, _decode_attention uses "
            "try_get_simultaneous with torch.empty fallback. Composes with "
            "our P99 memoization on get_simultaneous (P99 stays intact)."
        ),
        "upstream_pr": 42551,
        "applies_to": {
            "is_turboquant": True,  # patch sites are TQ-specific
        },
        "implementation_status": "full",
        "apply_module": "vllm.sndr_core.integrations.attention.turboquant.pn118_tq_workspace_fallback",
        "source": "vllm_pr_backport",
        "lifecycle": "experimental",
        "conflicts_with": [],
        "requires_patches": [],
    },
    "PN119": {
        "title": "TurboQuant k8v4 GQA head grouping kernel (vllm#40792)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_PN119",
        "default_on": True,
        "category": "kernel_perf",
        "credit": (
            "Backport of vllm#40792 (hoseung2, OPEN 2026-05-11). Adds "
            "_tq_grouped_decode_stage1 Triton kernel that loads K once "
            "per BLOCK_H q-head tile (sharing across the GQA group) and "
            "uses tl.dot (tensor cores) instead of element-wise scoring. "
            "Updates dispatch in triton_turboquant_decode_attention to "
            "route GQA-ratio>1 traffic to the grouped kernel. Upstream "
            "measured +16.5–27.2 % TPS on A100/H100 with GQA-ratio "
            "∈ {4, 8, 24}. Our 27B + 35B both run GQA-ratio 8 so the "
            "expected win is at the high end on Ampere SM 8.6. Applied "
            "via bundled diff + `patch` subprocess with md5 pre-patch "
            "guard; self-retires on drift."
        ),
        "upstream_pr": 40792,
        "applies_to": {
            "is_turboquant": True,  # k8v4 decode path
        },
        "implementation_status": "full",
        "apply_module": "vllm.sndr_core.integrations.attention.turboquant.pn119_tq_gqa_grouping",
        "source": "vllm_pr_backport",
        "lifecycle": "experimental",
        "conflicts_with": [],
        "requires_patches": [],
    },
    "P100": {
        "title": "FlashInfer FULL CUDA graph for spec-decode (vllm#41127)",
        "tier": "community",
        "family": "attention.flash",
        "env_flag": "GENESIS_ENABLE_P100",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": "Backport of vllm#41127 (open 2026-04-28). Per Sander 'не ждём, изучаем, импортируем'. Native FlashInfer can route uniform query_len>1 (1+num_spec_tokens) batches through prefill wrapper in cudagraph mode (zero_rows padding bit-identical). Adds FISpecDecode dataclass + _get_spec_decode_prefill_wrapper method + per-row qo_indptr delta scan in build() + FISpecDecode case in forward(). 11 sub-patches on flashinfer.py. NO-OP for PROD (turboquant_attn). Active for 27B variants (FlashInfer + spec-decode + non-DCP). Expected: +5-10% TPS on Ampere SM 8.6. RECOMMENDED on Blackwell consumer (sm_120) where FlashInfer is the default backend and PIECEWISE downgrade was observed (apnar club-3090#51). Recommendation surfaced via gpu_profile.PATCH_RECOMMENDATIONS rule.",
        "upstream_pr": 41127,
        "applies_to": {},  # FlashInfer auto-selected; gating via env_flag only
        "apply_module": "vllm.sndr_core.integrations.attention.flash.p100_flashinfer_full_cg_specdec",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P103": {
        "title": "FLA Cliff 2 chunked fwd_h+fwd_o orchestrator (qwen36-27b-single-3090#1)",
        "tier": "community",
        "family": "attention.gdn",
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
            # [Genesis pin-gate 2026-05-11] PROD-active (GroupAB component
            # + long-context single-GPU users). Validated dev9 → dev93.
            "vllm_version_range": (">=0.20.0", "<0.21.0"),
        },
        "apply_module": "vllm.sndr_core.integrations.attention.gdn.p103_fla_cliff2_chunked",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P101": {
        "title": "TQ continuation 64-token slicing (vllm#41123 SELECTIVE)",
        "tier": "community",
        "family": "attention.turboquant",
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
        "apply_module": "vllm.sndr_core.integrations.attention.turboquant.p101_tq_continuation_slicing",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P99": {
        "title": "WorkspaceManager.get_simultaneous memoization (perf hotfix)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_ENABLE_P99",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": "Per Sander 2026-04-28: 'if revert gives speedup, look at kernel — maybe rewrite'. P99 keeps upstream WorkspaceManager design (shared memory, 60x savings) but adds memoization to bypass per-call list-comp + accumulate + _ensure_workspace_size. Cache hit ~5x faster than full computation. Composes with P98 (P98 reverts turboquant_attn to per-layer; P99 helps any other backend using WorkspaceManager).",
        "upstream_pr": 40941,
        "enables_upstream_feature": True,
        # [Iron rule #11 audit 2026-05-11 v2] P99 AUGMENTS the merged
        # upstream feature (#40941 WorkspaceManager) by wrapping
        # `get_simultaneous()` with memoization — it does NOT backport
        # the PR (the PR is already in our pin since dev9+). Case (b)
        # of iron rule #11: we do MORE on top of upstream. Audit script
        # honors `enables_upstream_feature: True` to exclude from
        # NEWLY-MERGED categorization. KEEP active. Cleanup queue:
        # if upstream upstreams the memoization, retire then.
        "applies_to": {},  # applies whenever WorkspaceManager is used
        "apply_module": "vllm.sndr_core.integrations.attention.turboquant.p99_workspace_manager_memoize",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P98": {
        "title": "TQ WorkspaceManager revert (vllm#40941 perf hotfix)",
        "tier": "community",
        "family": "attention.turboquant",
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
        "apply_module": "vllm.sndr_core.integrations.attention.turboquant.p98_tq_workspace_revert",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P95": {
        "title": "Marlin TP cudagraph cap on Ampere (vllm#40385)",
        "tier": "community",
        "family": "compile_safety",
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
        "apply_module": "vllm.sndr_core.integrations.compile_safety.p95_marlin_tp_cudagraph_cap",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "P91": {
        "title": "AutoRound row-parallel group cdiv + start-idx fix (vllm#39460)",
        "tier": "community",
        "family": "quantization",
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
        "apply_module": "vllm.sndr_core.integrations.quantization.p91_autoround_row_group_cdiv",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },

    # ─── Legacy patches (P1–P46 series, pre-dispatcher era) ─────────────
    # These patches predate the PATCH_REGISTRY metadata system. They have
    # been live in PROD since pre-v7.0 and don't currently read an env
    # flag — they apply unconditionally as part of `apply_all`. The
    # synthetic `GENESIS_LEGACY_P*` env_flags below exist purely so the
    # dispatcher / validator / `genesis explain` see a coherent registry
    # entry; setting them has no runtime effect (yet). Future work: wire
    # actual opt-out gating where it makes sense.
    #
    # Why register them: lets `apply_all_dispatcher_sync` test pass,
    # surfaces these patches in `genesis list-patches` / `genesis explain`,
    # and provides a stable shape for documentation tooling.

    "P1": {
        "title": "FP8 kernel dispatcher (P1/P2 — Ampere FP8 viability)",
        "tier": "community",
        "family": "quantization",
        "env_flag": "GENESIS_LEGACY_P1",
        "default_on": True,
        "implementation_status": "marker_only",
        "lifecycle": "legacy",
        "category": "quantization",
        "credit": "Pre-dispatcher legacy patch. Wires Ampere SM86 to FP8 kernel paths so consumer 3090/A5000 can serve FP8-quantized models.",
    },
    "P3": {
        "title": "TurboQuant BF16→FP8 cast (Ampere fix)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_LEGACY_P3",
        "default_on": True,
        "apply_module": "vllm.sndr_core.integrations.attention.turboquant.p3_tq_bf16_cast",
        "lifecycle": "legacy",
        "category": "quantization",
        "credit": "Pre-dispatcher legacy patch. Inserts BF16→FP8 cast on TQ ingress for SM86 where FP8 is software-emulated.",
        "implementation_status": "full",
    },
    "P4": {
        "title": "TurboQuant hybrid model support",
        "tier": "community",
        "family": "scheduler",
        "env_flag": "GENESIS_LEGACY_P4",
        "default_on": True,
        "apply_module": "vllm.sndr_core.integrations.scheduler.p4_tq_hybrid",
        "lifecycle": "legacy",
        "category": "kv_cache",
        "credit": "Pre-dispatcher legacy patch. Removes hybrid (GDN + full attention) model rejection in TQ path, enabling Qwen3.5/3.6 hybrid serving with TQ k8v4.",
        # 2026-05-05: SUPERSEDED upstream by vllm#39931 (MERGED 2026-05-05 00:14
        # UTC, JartX + jhsmith409 + Sandermage co-authors). Upstream now
        # detects hybrid via layer_types/layers_block_type/attn_type_list and
        # computes TQ page-size via lcm in `_align_hybrid_block_size` —
        # cleaner than P4. Plan: retire P4 on next pin bump past commit
        # 4f2af1a7c03aae2b3227dd7e69d726104d44a711. Verify hybrid TQ smoke test
        # boots cleanly with P4 OFF before final retirement.
        "superseded_by": "vllm#39931 (merged 2026-05-05)",
        "retire_after_pin": "0.20.2rc1+",
        "implementation_status": "full",
    },
    "P5": {
        "title": "KV cache page size unification",
        "tier": "community",
        "family": "kv_cache",
        "env_flag": "GENESIS_LEGACY_P5",
        "default_on": True,
        "apply_module": "vllm.sndr_core.integrations.kv_cache.p5_page_size",
        "lifecycle": "legacy",
        "category": "kv_cache",
        "credit": "Pre-dispatcher legacy patch. Unifies per-layer page size across hybrid attention layers so block manager doesn't fragment.",
        "implementation_status": "full",
    },
    "P5b": {
        "title": "KV page-size pad-smaller-to-max (env-flag coordinator for P5)",
        "tier": "community",
        "family": "memory",
        # Audit P2 fix 2026-05-05: registry was `GENESIS_ENABLE_P5B_PAGE_PAD`
        # but wiring code + docstrings use `GENESIS_ENABLE_P5B`. Aligned.
        "env_flag": "GENESIS_ENABLE_P5B",
        "default_on": False,
        # `coordinator` lifecycle (introduced 2026-05-06): the wiring module
        # has no real binding — it only reads the env gate and reports
        # `applied`/`skipped`. The actual text-patch (selecting v2 pad-smaller-
        # to-max body in `_align_hybrid_block_size` + stamping
        # `real_page_size_bytes` on TQFullAttentionSpec) is performed by P5
        # when GENESIS_ENABLE_P5B=1 is set. Kept as a separate registry entry
        # so operators can grep for the feature flag and so audit/preflight
        # can flag P5+P5b composability concerns explicitly.
        "apply_module": "vllm.sndr_core.integrations.memory.p5b_page_size_pad_smaller",
        "lifecycle": "coordinator",
        "category": "kv_cache",
        "credit": "Pre-dispatcher legacy patch. Opt-in companion to P5 — pads smaller pages up to max so all layers share one block-pool stride. Guarded by env (was always opt-in). Coordinator pattern: real binding in P5; this entry is a documented feature-flag handle.",
        "implementation_status": "full",
    },
    "P6": {
        "title": "TurboQuant-aware attention page size",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_LEGACY_P6",
        "default_on": True,
        "apply_module": "vllm.sndr_core.integrations.compile_safety.p6_tq_block_size_align",
        "lifecycle": "legacy",
        "category": "kv_cache",
        "credit": "Pre-dispatcher legacy patch. Selects TQ-aware page size (matches TQ packed slot stride) when TQ KV is active.",
        "implementation_status": "full",
    },
    "P7": {
        "title": "GDN dual-stream in_proj parallelism",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_LEGACY_P7",
        "default_on": True,
        "apply_module": "vllm.sndr_core.integrations.attention.gdn.p7_gdn_dual_stream",
        "lifecycle": "legacy",
        "category": "kernel_perf",
        "credit": "Pre-dispatcher legacy patch. Splits GDN in_proj across two CUDA streams so q/k/v projections overlap. Validated +8% decode on 35B.",
        "conflicts_with": ["P7b", "PN204"],  # PN204 = port of vllm#42301, same site as P7
        "implementation_status": "full",
    },
    "P7b": {
        "title": "GDN dual-stream via torch.library.custom_op (opt-in)",
        "tier": "community",
        "family": "attention.gdn",
        # Audit P2 fix 2026-05-05: registry was `GENESIS_ENABLE_P7B_DUAL_STREAM_CUSTOM_OP`
        # but wiring code + docstrings use `GENESIS_ENABLE_P7B`. Aligned.
        "env_flag": "GENESIS_ENABLE_P7B",
        "default_on": False,
        "apply_module": "vllm.sndr_core.integrations.attention.gdn.p7b_gdn_dual_stream_customop",
        "lifecycle": "legacy",
        "category": "kernel_perf",
        "credit": "Pre-dispatcher legacy patch. Custom-op variant of P7 dual-stream — opt-in alternative for cudagraph capture compatibility experiments.",
        "conflicts_with": ["P7"],
        "implementation_status": "full",
    },
    "PN13": {
        "title": "CUDAGraphWrapper lambda arity (vllm#41235 backport) — RETIRED 2026-05-04",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_ENABLE_PN13_CUDA_GRAPH_LAMBDA_ARITY",
        "default_on": False,
        "lifecycle": "retired",
        "notes": (
            "upstream_native_via_pr41235 — vllm 0.20.2 (commit "
            "c2fb013) merged identical change in cuda_graph.py: "
            "patch(\"gc.collect\", lambda *args, **kwargs: None) + "
            "patch(\"torch.accelerator.empty_cache\", lambda *args, "
            "**kwargs: None). Upstream code now matches our PN13 "
            "replacement byte-for-byte. PN13 anchor (pre-fix lambda: "
            "None pattern) no longer matches → silent skip. Per Sander "
            "rule (2026-05-04): «если код соответствует патчу, патч "
            "отключаем». Retired."
        ),
        "category": "compile_safety",
        "credit": "Backport of vllm#41235 by Roi Koren (NVIDIA). RETIRED — upstream natively fixes after vllm v0.20.2.",
        "upstream_pr": 41235,
        "superseded_by": "vllm#41235 (merged 2026-04-29, in commit c2fb013 / v0.20.2 — byte-equivalent on dev93+dev209)",
        "vllm_version_range": "<0.20.2",  # active before upstream merge in v0.20.2
        "apply_module": "vllm.sndr_core.integrations._retired.pn13_cuda_graph_lambda_arity",
        "applies_to": {
            # [Genesis pin-gate 2026-05-11] #41235 MERGED 2026-04-29 (in
            # commit c2fb013, vllm 0.20.2). Upper bound formalizes retire
            # — pin-gate adds belt-to-the-suspenders (lifecycle="retired"
            # already de-facto skips, this makes the version boundary
            # explicit for `genesis explain` and audit reports).
            "vllm_version_range": "<0.20.2",
        },
        "implementation_status": "full",
    },
    "P8": {
        "title": "KV hybrid reporting (per-token capacity) — RETIRED 2026-05-04",
        "tier": "community",
        "family": "scheduler",
        "env_flag": "GENESIS_LEGACY_P8",
        "default_on": False,
        "lifecycle": "retired",
        "notes": (
            "upstream_native_via_get_max_concurrency_refactor — vllm "
            "v0.20.2rc1.dev9 (commit 01d4d1ad3) refactored "
            "_report_kv_cache_config to use "
            "get_max_concurrency_for_kv_cache_config(vllm_config, "
            "kv_cache_config) which natively handles hybrid layouts "
            "(SWA / chunked-local groups with per-request block count "
            "capped by window). The new formula `max_concurrency * "
            "max_model_len` supersedes our P8 approach (excluding O(1) "
            "Mamba groups from per-token divisor). Engine now reports "
            "correct capacity natively without our patch. P8 anchors "
            "no longer match (kv_cache_utils.py refactored)."
        ),
        "category": "kv_cache",
        "credit": "Pre-dispatcher legacy patch. Reports KV capacity per-token (not per-block) for hybrid models so scheduler doesn't over-admit. RETIRED upstream natively fixes after vllm v0.20.2.",
        "superseded_by": "vllm dev9 commit 01d4d1ad3 — get_max_concurrency_for_kv_cache_config refactor handles hybrid layouts natively (no specific PR captured; verified via anchor mismatch on dev9+)",
        "vllm_version_range": "<0.20.2rc1.dev9+g01d4d1ad3",  # active before upstream refactor in dev9
        "apply_module": "vllm.sndr_core.integrations._retired.p8_kv_hybrid_reporting",
        "applies_to": {
            # [Iron rule #11 formal retire 2026-05-11] Promoted from
            # waiver — notes already had identifiable supersession.
            # Anchors don't match dev9+, so pin-gate upper bound is
            # cosmetic on the legacy auto-apply path (synthetic env_flag),
            # but documents the supersession boundary for `genesis explain`.
            "vllm_version_range": "<0.20.2rc1.dev9",
        },
        "implementation_status": "full",
    },
    "P12": {
        "title": "Qwen3 <tool_call> implicit reasoning end",
        "tier": "community",
        "family": "reasoning",
        "env_flag": "GENESIS_LEGACY_P12",
        "default_on": True,
        "apply_module": "vllm.sndr_core.integrations.reasoning.p12_tool_call_reasoning",
        "lifecycle": "legacy",
        "category": "structured_output",
        "credit": "Pre-dispatcher legacy patch. Treats <tool_call> emission as implicit </think>, fixing Qwen3 reasoning models that omit explicit </think> before tool calls. Updated v7.62.5 to FIRST-occurrence (was LAST), retiring P61.",
        "superseded_by": "upstream qwen3_parser refactor (marker '_tool_call_token_id' present on dev93+; auto-skip via wiring drift detector)",
        # [Iron rule #11 audit 2026-05-11] Wire detector auto-skips on
        # dev93+dev209 — upstream qwen3 reasoning_parser refactor added
        # `_tool_call_token_id` attribute which handles the implicit
        # </think> case natively (different impl but same outcome).
        # Lifecycle stays "legacy" (architectural — pre-dispatcher
        # auto-apply pattern); skip is correct + safe.
        "implementation_status": "full",
    },
    "P14": {
        "title": "block_table tail zero-fill",
        "tier": "community",
        "family": "kv_cache",
        "env_flag": "GENESIS_LEGACY_P14",
        "default_on": True,
        "apply_module": "vllm.sndr_core.integrations.kv_cache.p14_block_table",
        "lifecycle": "legacy",
        "category": "kernel_safety",
        "credit": "Pre-dispatcher legacy patch. Zero-fills block_table tail past valid sequences so out-of-bounds prefetch doesn't read stale page indices.",
        "implementation_status": "full",
    },
    "P15": {
        "title": "Qwen3 None/null tool arg parser",
        "tier": "community",
        "family": "tool_parsing",
        "env_flag": "GENESIS_LEGACY_P15",
        "default_on": True,
        "apply_module": "vllm.sndr_core.integrations.tool_parsing.p15_qwen3_none_null",
        "lifecycle": "legacy",
        "category": "structured_output",
        "credit": "Pre-dispatcher legacy patch. Tolerates None / null tool arguments in Qwen3 parser instead of raising.",
        "implementation_status": "full",
    },
    "P17": {
        "title": "Marlin MoE per-SM tuning (P17/P18)",
        "tier": "community",
        "family": "moe",
        "env_flag": "GENESIS_LEGACY_P17",
        "default_on": True,
        "implementation_status": "marker_only",
        "lifecycle": "legacy",
        "category": "kernel_perf",
        "credit": "Pre-dispatcher legacy patch. Per-SM (SM86) tuned configs for Marlin MoE kernel — bsm=8 selected on Ampere consumer cards.",
    },
    "P18b": {
        "title": "TurboQuant decode stage1 tune",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_LEGACY_P18B",
        "default_on": True,
        "implementation_status": "marker_only",
        "lifecycle": "legacy",
        "category": "kernel_perf",
        "credit": "Pre-dispatcher legacy patch. Tuned launch config for TQ decode stage1 kernel on SM86.",
    },
    "P20": {
        "title": "TurboQuant continuation-prefill FP16 rotate",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_LEGACY_P20",
        "default_on": True,
        "implementation_status": "marker_only",
        "lifecycle": "legacy",
        "category": "kernel_perf",
        "credit": "Pre-dispatcher legacy patch. FP16 rotation for TQ continuation-prefill path (JartX/vllm#11 prerequisite for v7.0+).",
    },
    "P22": {
        "title": "TurboQuant shared dequant prealloc",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_LEGACY_P22",
        "default_on": True,
        "apply_module": "vllm.sndr_core.integrations.attention.turboquant.p22_tq_prealloc",
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Preallocates shared dequant scratch buffer so TQ doesn't allocate-per-step (Genesis-original).",
        # [Iron rule #11 investigation queued 2026-05-11]
        # On dev209+g5536fc0c0 the drift detector auto-skips P22 because
        # upstream removed `_init_turboquant_buffers` (PR #40655-style
        # restructure landed without a formal merge). However, per our
        # patch's own docstring (lines 43-55), the alternative upstream
        # approaches LACK profiler visibility — which is the specific
        # value-add P22 provides (visible to memory-profiler → correct
        # KV cache sizing → no #40420-class OOM at long context).
        # Per iron rule #11 case (b) "our patch does MORE": should be
        # UPDATED to re-hook profiler visibility on top of the new
        # upstream restructure, NOT silently retired. Currently auto-skip
        # is safe (no crash, just loses our improvement); investigation
        # queued — read dev209's gpu_model_runner.capture_model + the
        # current TurboQuantAttentionImpl class to design new hook site.
        "implementation_status": "full",
    },
    "P23": {
        "title": "Marlin FP32_REDUCE env override",
        "tier": "community",
        "family": "kernels",
        "env_flag": "GENESIS_LEGACY_P23",
        "default_on": True,
        "implementation_status": "marker_only",
        "lifecycle": "legacy",
        "category": "kernel_perf",
        "credit": "Pre-dispatcher legacy patch. Honors VLLM_MARLIN_FP32_REDUCE env to force FP32 reduction in Marlin matmul (numerical-stability hedge).",
    },
    "P24": {
        "title": "fused_moe num_warps/num_stages overlay",
        "tier": "community",
        "family": "moe",
        "env_flag": "GENESIS_LEGACY_P24",
        "default_on": True,
        "apply_module": "vllm.sndr_core.integrations.moe.p24_moe_tune",
        "lifecycle": "legacy",
        "category": "kernel_perf",
        "credit": "Pre-dispatcher legacy patch. Overlays SM86-tuned num_warps/num_stages on fused_moe kernel selection.",
        "implementation_status": "full",
    },
    "P26": {
        "title": "TurboQuant prefill output prealloc",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_LEGACY_P26",
        "default_on": True,
        "apply_module": "vllm.sndr_core.integrations.attention.turboquant.p26_prefill_output",
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Preallocates TQ prefill output buffer to avoid per-step allocation churn.",
        "superseded_by": "upstream TQ prefill prealloc refactor (marker 'if not hasattr(self, \"_cu_2\")' present on dev93+; auto-skip via wiring drift detector)",
        # [Iron rule #11 audit 2026-05-11] Wire detector auto-skips on
        # dev93+dev209 — upstream's `_cu_2` lazy-init guard covers the
        # same prefill output buffer prealloc concern P26 addressed.
        # Different impl path (lazy hasattr check vs Genesis explicit
        # prealloc) but functionally equivalent for the regression P26
        # prevents. Lifecycle stays "legacy" (architectural).
        "implementation_status": "full",
    },
    "P27": {
        "title": "Qwen3 BEFORE-THINK fallback",
        "tier": "community",
        "family": "reasoning",
        "env_flag": "GENESIS_LEGACY_P27",
        "default_on": True,
        "apply_module": "vllm.sndr_core.integrations.reasoning.p27_reasoning_before_think",
        "lifecycle": "legacy",
        "category": "structured_output",
        "credit": "Pre-dispatcher legacy patch. Falls back to BEFORE-THINK parsing path when Qwen3 model emits tool_call before <think>.",
        "implementation_status": "full",
    },
    "P28": {
        "title": "GDN core_attn_out prealloc",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_LEGACY_P28",
        "default_on": True,
        "apply_module": "vllm.sndr_core.integrations.attention.gdn.p28_gdn_core_attn",
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Preallocates GDN core_attn_out as a layer-persistent buffer + zero()-on-reuse instead of torch.zeros() per-step. Reduces allocator pressure on GDN forward.",
        "conflicts_with": ["PN32", "PN108"],  # PN108 = GDN fused_recurrent prefill switches backend; incompatible with P28 buffer reuse
        "implementation_status": "full",
    },
    "P29": {
        "title": "tool parser IndexError guard",
        "tier": "community",
        "family": "tool_parsing",
        "env_flag": "GENESIS_LEGACY_P29",
        "default_on": True,
        "implementation_status": "marker_only",
        "lifecycle": "legacy",
        "category": "structured_output",
        "credit": "Pre-dispatcher legacy patch. Wraps tool-arg index access so malformed parser state returns empty instead of raising IndexError.",
    },
    "P31": {
        "title": "MoE router fp32 softmax",
        "tier": "community",
        "family": "moe",
        "env_flag": "GENESIS_LEGACY_P31",
        "default_on": True,
        "apply_module": "vllm.sndr_core.integrations.moe.p31_router_softmax",
        "lifecycle": "legacy",
        "category": "model_correctness",
        "credit": "Pre-dispatcher legacy patch. Upcasts MoE router softmax to fp32 (DeepSeek-V3 pattern, deepseek_v2.py:345 reference). Improves expert routing stability on consumer Ampere.",
        "implementation_status": "full",
    },
    "P32": {
        "title": "TurboQuant cu_2 + synth_seq_lens preallocs (P32/P33)",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_LEGACY_P32",
        "default_on": True,
        "implementation_status": "marker_only",
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Preallocates cu_2 and synth_seq_lens TQ scratch tensors as persistent buffers.",
    },
    "P34": {
        "title": "Mamba zero-collapse deadlock guard",
        "tier": "community",
        "family": "scheduler",
        "env_flag": "GENESIS_LEGACY_P34",
        "default_on": True,
        "apply_module": "vllm.sndr_core.integrations.scheduler.p34_mamba_deadlock_guard",
        "lifecycle": "legacy",
        "category": "stability",
        "credit": "Pre-dispatcher legacy patch. Guards against Mamba state collapse-to-zero deadlock when delta is exactly zero on hybrid models.",
        "implementation_status": "full",
    },
    "P36": {
        "title": "TurboQuant shared decode buffers",
        "tier": "community",
        "family": "kernels",
        "env_flag": "GENESIS_LEGACY_P36",
        "default_on": True,
        "apply_module": "vllm.sndr_core.integrations.kernels.p36_tq_shared_decode_buffers",
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Shared decode-stage scratch buffers across TQ layers to amortize allocation.",
        "implementation_status": "full",
    },
    "P37": {
        "title": "MoE intermediate cache pool (opt-in)",
        "tier": "community",
        "family": "moe",
        # Audit P1 fix 2026-05-05 (genesis_local_consistency_audit + runtime audit):
        # registry was `GENESIS_ENABLE_P37_MOE_INTER_CACHE` but wiring code,
        # apply_all docstring, AND launch scripts all use `GENESIS_ENABLE_P37`.
        # env_flag_guard was reporting it as suspicious typo. Aligned to short form.
        "env_flag": "GENESIS_ENABLE_P37",
        "default_on": False,
        "apply_module": "vllm.sndr_core.integrations.moe.p37_moe_intermediate_cache",
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Opt-in pool for MoE intermediate activations. noonghunna's club-3090 long-text recipe ships with this enabled.",
        "implementation_status": "full",
    },
    "P38": {
        "title": "TQ _continuation_prefill persistent workspace",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_LEGACY_P38",
        "default_on": True,
        "apply_module": "vllm.sndr_core.integrations.attention.turboquant.p38_tq_continuation_memory",
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Persistent workspace tensor for TQ continuation-prefill, addresses VolandBerlioz's OOM site at turboquant_attn.py. Companion: P38B (compile-safe in-source hook, see PATCH_REGISTRY).",
        "implementation_status": "full",
    },
    "P39a": {
        "title": "FLA chunk_scaled_dot_kkt persistent A pool",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_LEGACY_P39A",
        "default_on": True,
        "apply_module": "vllm.sndr_core.integrations.attention.gdn.p39a_fla_kkt_buffer",
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Persistent pool for FLA chunk_scaled_dot_kkt's A matrix to avoid per-step allocation in GDN backward.",
        "implementation_status": "full",
    },
    "P40": {
        "title": "TurboQuant GQA-grouped decode stage1 (opt-in)",
        "tier": "community",
        "family": "attention.turboquant",
        # Audit P1 fix 2026-05-05: same class as P37 — registry was
        # `GENESIS_ENABLE_P40_GQA_GROUPED_DECODE` but wiring/kernel/scripts
        # use `GENESIS_ENABLE_P40`. compat/presets.py also used yet another
        # variant (`GENESIS_ENABLE_P40_TQ_GROUPED_DECODE`). Aligned to short form.
        "env_flag": "GENESIS_ENABLE_P40",
        "default_on": False,
        "apply_module": "vllm.sndr_core.integrations.attention.turboquant.p40_tq_grouped_decode",
        "lifecycle": "legacy",
        "category": "kernel_perf",
        "credit": "Pre-dispatcher legacy patch (vllm#40792 backport candidate). Opt-in GQA-grouped TQ decode stage1 kernel. Welch t-test on 2x A5000 single-stream: not significant (p=0.284 vs baseline 183 TPS) — kept opt-in pending Blackwell retest.",
        "implementation_status": "full",
    },
    "P44": {
        "title": "TQ mixed-batch attn_out pool",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_LEGACY_P44",
        "default_on": True,
        "apply_module": "vllm.sndr_core.integrations.attention.turboquant.p44_tq_mixed_attn_out",
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Pool for TQ attn_out tensor under mixed prefill+decode batches.",
        "implementation_status": "full",
    },
    "P46": {
        "title": "GDN gating buffer pool",
        "tier": "community",
        "family": "attention.gdn",
        "env_flag": "GENESIS_LEGACY_P46",
        "default_on": True,
        "apply_module": "vllm.sndr_core.integrations.attention.gdn.p46_gdn_gating_buffers",
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Pool for GDN gating tensor to avoid per-layer allocation.",
        "implementation_status": "full",
    },
    "P51": {
        "title": "TQ-active runtime layer-level guard",
        "tier": "community",
        "family": "attention.turboquant",
        "env_flag": "GENESIS_LEGACY_P51",
        "default_on": True,
        "implementation_status": "marker_only",
        "lifecycle": "legacy",
        "category": "quantization",
        "credit": "Pre-dispatcher library patch. Runtime layer-level TQ-active detection in kernels/dequant_buffer.py — skips TQ preallocs on layers where TQ is not active. No env toggle (defensive runtime check). Companion to model_detect's config-level TQ check.",
    },
    "P102": {
        "title": "Unified spec-decode metadata + disagreement tracker (TRT-LLM style)",
        "tier": "community",
        "family": "kv_cache",
        "env_flag": "GENESIS_ENABLE_P102",
        "default_on": False,
        "implementation_status": "marker_only",  # exploratory design entry, no on-disk wiring
        "category": "spec_decode",
        "credit": "Genesis-original (Sander 2026-04-29). First-class spec_meta module that wraps spec-decode metadata into a unified object + tracks predicate disagreement (e.g. should_dispatch_p67 disagreements between proposer and verify paths). Diagnostic-only opt-in observability layer; emits log lines when divergence detected. Future hook for unified spec-decode dispatcher refactor.",
        "upstream_pr": None,
        "lifecycle": "experimental",
    },
    "PN60": {
        "title": "Quant arg vs config.json validator (preflight DX)",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_ENABLE_PN60",
        "default_on": True,
        "implementation_status": "marker_only",  # advisory validation, registered as recommendation
        "category": "stability",
        "credit": "Genesis-original 2026-05-05 (apnar club-3090#51 finding). Cross-checks operator's --quantization CLI arg against the model's config.json:quantization_config.quant_method BEFORE vLLM loads. Emits one-line remediation hint instead of a 30-line pydantic ValidationError. Doctor extension; runs at preflight, no monkey-patch.",
        "upstream_pr": None,
        "applies_to": {},
        "lifecycle": "legacy",
    },
    "PN61": {
        "title": "qwen3_vl loader KeyError → text-only auto-fallback (vllm-loader guard)",
        "tier": "community",
        "family": "loader",
        "env_flag": "GENESIS_ENABLE_PN61",
        "default_on": False,
        "category": "stability",
        "credit": "Genesis-original 2026-05-05 (apnar club-3090#51 NVFP4 finding). Catches `KeyError: 'blocks.0.attn.proj.weight'` in qwen3_vl.load_weights when an NVFP4 quant strips the ViT tower; emits WARN + auto-sets language_model_only=True instead of crashing. Same defensive pattern as P29 IndexError guard.",
        "upstream_pr": None,
        "applies_to": {"model_class": ["qwen3_vl"]},
        "apply_module": "vllm.sndr_core.integrations.loader.pn61_qwen3_vl_keyerror_guard",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN62": {
        "title": "Text-only ViT scratch skip via skip_mm_profiling flip (3-5 GiB save)",
        "tier": "community",
        "family": "multimodal",
        "env_flag": "GENESIS_ENABLE_PN62",
        "default_on": False,
        "category": "memory_savings",
        "credit": "Genesis-original 2026-05-05 (apnar club-3090#51); Wave 6 real hook 2026-05-09. When mm_limits_all_zero AND --language-model-only, PN62 wraps GPUModelRunner.profile_run and flips MultiModalConfig.skip_mm_profiling=True before profile_run executes, so vllm dev93's native short-circuit at the encoder profiling branch fires. Saves ~3-5 GiB ViT scratch on qwen3_vl + NVFP4 single-card boot. Real hook landed (Wave 6) — replaces the prior marker-only scaffold. Sister to PN35 (text-only inputs_embeds skip, vllm#35975 merged).",
        "upstream_pr": None,
        "applies_to": {"model_class": ["qwen3_vl"]},
        "implementation_status": "full",
        "apply_module": "vllm.sndr_core.integrations.multimodal.pn62_text_only_vit_skip",
        "lifecycle": "experimental",  # awaiting cross-rig live validation
    },
    "PN63": {
        "title": "fp8_e5m2 advisory for consumer Blackwell (gpu_profile recommendation)",
        "tier": "community",
        "family": "compile_safety",
        "env_flag": "GENESIS_ENABLE_PN63",
        "default_on": True,
        "implementation_status": "marker_only",  # advisory only, no patch site
        "category": "stability",
        "credit": "Genesis-original 2026-05-05 (apnar club-3090#51 empirical). Adds a per-GPU advisory entry to gpu_profile.PATCH_RECOMMENDATIONS that recommends --kv-cache-dtype fp8_e5m2 over fp8_e4m3 on consumer Blackwell (sm 12.0) until vLLM e4m3 codepath matures. Suggest-only; operator passes via CLI.",
        "upstream_pr": None,
        "lifecycle": "legacy",
    },
    "PN64": {
        "title": "Marlin MoE per-SM tuning placeholder for SM 12.0 (consumer Blackwell)",
        "tier": "community",
        "family": "kernels",
        "env_flag": "GENESIS_ENABLE_PN64",
        "default_on": False,
        "implementation_status": "placeholder",  # SM 12.0 stub, awaiting empirical Blackwell measurement
        "category": "kernel_perf",
        "credit": "Genesis-original 2026-05-05 (apnar club-3090#51 — boot log shows `[Genesis] skipped: P17/P18 Marlin MoE per-SM tuning — no tuning entry for SM (12, 0)`). PN64 adds a placeholder entry copying SM (9, 0) Hopper config until empirical sweep data lands from sm_120. Author-blocked: needs real 5090 sweep — solicit from apnar/jhsmith409.",
        "upstream_pr": None,
        "applies_to": {},
        "lifecycle": "experimental",
    },
    "PN65": {
        "title": "Genesis structured API access log middleware (operator UX)",
        "tier": "community",
        "family": "middleware",
        "env_flag": "GENESIS_ENABLE_PN65",
        "default_on": False,
        "category": "request_middleware",
        "credit": "Genesis-original 2026-05-05 (Sander request 'по апи лог невзрачный надо тоже проработать'). Replaces uvicorn's bare `INFO: 127.0.0.1:45116 - GET /v1/models 401 Unauthorized` with `[Genesis-API] 200  POST /v1/chat/completions  34ms  prompt=46t  completion=400t  tools=1  client=127.0.0.1`. Suppresses /health polling by default (GENESIS_PN65_LOG_HEALTH=1 to include). Status-aware level (2xx INFO / 4xx WARN / 5xx ERROR + exception type).",
        "upstream_pr": None,
        "applies_to": {},
        "apply_module": "vllm.sndr_core.integrations.middleware.pn65_access_log",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN66": {
        "title": "Multiturn </think> leak fix in DelegatingParser (vllm#41696 backport)",
        "tier": "community",
        "family": "reasoning",
        "env_flag": "GENESIS_ENABLE_PN66",
        "default_on": False,
        "category": "structured_output",
        "credit": "Backport of vllm#41696 (panpan0000, OPEN as of 2026-05-05). Removes the buggy `prompt_reasoning_checked` short-circuit in `vllm.parser.abstract_parser.DelegatingParser.parse_delta` that walked the FULL prompt looking for `</think>` and prematurely set `reasoning_ended=True` from a previous turn's `</think>`. Defensive backport for multi-turn DSML/Hermes/Qwen3 chat clients sending full history. Original report: DeepSeek V3.2 reasoning users.",
        "upstream_pr": 41696,
        "applies_to": {},
        "apply_module": "vllm.sndr_core.integrations.reasoning.pn66_multiturn_think_leak",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN67": {
        "title": "thinking_token_budget inverted-bool fix (vllm#41674 backport, 1-line)",
        "tier": "community",
        "family": "worker",
        "env_flag": "GENESIS_ENABLE_PN67",
        "default_on": False,
        "category": "stability",
        "credit": "Backport of vllm#41674 (JasonKeyiL, OPEN as of 2026-05-04). Single-token fix in `vllm/v1/worker/gpu_input_batch.py:879` — removes `not` from `or not thinking_budget_tracks_reqs`. Bug: thinking_token_budget was silently ignored for any request without penalty parameters. NULL on Genesis PROD (we don't enable thinking_token_budget); defensive for users who experiment with it. Trivial backport, zero risk.",
        "upstream_pr": 41674,
        "applies_to": {},
        "apply_module": "vllm.sndr_core.integrations.worker.pn67_thinking_budget_inverted_bool",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },
    "PN70": {
        "title": "Tool schema subset filter (combined `anyOf` xgrammar-clean) — companion to P68 v7.72.1",
        "tier": "community",
        "family": "serving",
        "env_flag": "GENESIS_ENABLE_PN70_TOOL_SCHEMA_FILTER",
        "default_on": False,
        "category": "structured_output",
        "credit": "Genesis-original — implements lexhoefsloot's option-3 fix for noonghunna/club-3090#57. Wraps `vllm.tool_parsers.utils._get_json_schema_from_tools` and filters tools containing xgrammar-unsupported JSON Schema keys (patternProperties / propertyNames / $ref / oneOf / etc.) BEFORE the combined `anyOf` is built and handed to xgrammar. Companion to P68's option-1 skip: where P68 refuses to upgrade tool_choice on dirty catalogs, PN70 keeps the upgrade and filters dirty tools out of grammar enforcement (model can still SEE all tools in context but grammar restricts callable subset). Reuses P68's `_scan_schema_for_unsupported_key` so the unsupported-key set is single-sourced. Off by default; enable per workload.",
        "applies_to": {},
        "composes_with": ["P68"],
        "apply_module": "vllm.sndr_core.integrations.serving.pn70_tool_schema_subset_filter",
        "lifecycle": "experimental",
        "implementation_status": "full",
    },

    # ── Gemma 4 family (2026-05-17) ────────────────────────────────────
    # 21 patches addressing every known Gemma 4 issue on Ampere SM 8.6:
    # FP8 double-scale (#39407), Marlin K-divisibility (#40354),
    # non-causal attention wall (#40382), AWQ MoE keys (#40886),
    # DFlash backend autoselect (#42069), KV-projection optimization
    # (#41944), SWA/global prefill chunker (#39914), FP8 e4nv guard
    # (#41014), per-token-head KV asymmetry (#40388 + WIP #40391),
    # tool-call-parser pad-token (#39392), vision-tower text-only,
    # FP16 overflow (#40124), perf kernels (fused RMSNorm + softcap),
    # FULL_AND_PIECEWISE cudagraph mode parallel to PN125.
    # Family: gemma4, location: vllm/sndr_core/integrations/gemma4/
    "G4_01": {
        "title": "Refuse FP8_BLOCK Gemma 4 checkpoint on Ampere SM 8.6 (closes vllm#39407 user pain)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_01_GEMMA4_FP8_BLOCK_GUARD",
        "default_on": True,
        "category": "stability",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "vllm.sndr_core.integrations.gemma4.g4_01_gemma4_ampere_fp8_block_guard",
        "lifecycle": "stable",
        "credit": "Refuses the known-broken FP8_BLOCK + Ampere combo at process_weights_after_loading. Saves operators a 30-min cold-boot-to-garbage debug cycle.",
        "upstream_pr": "https://github.com/vllm-project/vllm/issues/39407",
        "requires_patches": [],
        "conflicts_with": [],
        "superseded_by": ["G4_07"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_02": {
        "title": "Refuse Marlin MoE with K%64≠0 on Gemma 4 26B-A4B (closes vllm#40354 user pain)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_02_GEMMA4_MARLIN_KDIM_GUARD",
        "default_on": True,
        "category": "stability",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "vllm.sndr_core.integrations.gemma4.g4_02_gemma4_ampere_marlin_kdim_guard",
        "lifecycle": "stable",
        "credit": "Refuses 26B-A4B + Ampere Marlin combo at apply_weights time. K=352 (704/2 at TP=2) fails Marlin's min_thread_k=64 divisibility.",
        "upstream_pr": "https://github.com/vllm-project/vllm/issues/40354",
        "requires_patches": [],
        "conflicts_with": [],
        "superseded_by": ["G4_08"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration"]},
    },
    "G4_03": {
        "title": "Refuse non-causal drafter (Eagle3/DFlash) on Ampere SM 8.6 (closes vllm#40382 user pain)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_03_GEMMA4_NON_CAUSAL_DRAFTER_GUARD",
        "default_on": True,
        "category": "stability",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "vllm.sndr_core.integrations.gemma4.g4_03_gemma4_ampere_non_causal_drafter_guard",
        "lifecycle": "stable",
        "credit": "Refuses Eagle3/DFlash drafter on Gemma 4 + Ampere — no Ampere backend supports head_dim=256 + non-causal.",
        "upstream_pr": "https://github.com/vllm-project/vllm/issues/40382",
        "requires_patches": [],
        "conflicts_with": [],
        "superseded_by": ["G4_10"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_04": {
        "title": "Gemma 4 AWQ MoE keys remap (vendors vllm#40886)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_04_GEMMA4_AWQ_MOE_KEYS_REMAP",
        "default_on": True,
        "category": "loader",
        "implementation_status": "full",
        "source": "vendor_backport",
        "apply_module": "vllm.sndr_core.integrations.gemma4.g4_04_gemma4_awq_moe_keys_remap",
        "lifecycle": "stable",
        "credit": "Vendors vllm#40886 — AWQ MoE keys remap for Gemma 4 26B-A4B checkpoint compatibility.",
        "upstream_pr": "https://github.com/vllm-project/vllm/pull/40886",
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration"]},
    },
    "G4_05": {
        "title": "DFlash drafter backend autoselect (vendors vllm#42069)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_05_GEMMA4_DFLASH_BACKEND_AUTOSELECT",
        "default_on": True,
        "category": "loader",
        "implementation_status": "full",
        "source": "vendor_backport",
        "apply_module": "vllm.sndr_core.integrations.gemma4.g4_05_gemma4_dflash_backend_autoselect",
        "lifecycle": "stable",
        "credit": "Vendors vllm#42069 — 1-line backend=None to let DFlash drafter autoselect a non-causal-capable backend.",
        "upstream_pr": "https://github.com/vllm-project/vllm/pull/42069",
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_06": {
        "title": "v_head_size=0 for k_eq_v attention layers (vendors vllm#41944)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_06_GEMMA4_KV_PROJ_V0",
        "default_on": False,
        "category": "perf_hotfix",
        "implementation_status": "partial",
        "source": "vendor_backport",
        "apply_module": "vllm.sndr_core.integrations.gemma4.g4_06_gemma4_kv_proj_v_head_size_zero",
        "lifecycle": "experimental",
        "credit": "Vendors __init__ portion of vllm#41944. ~3% memory savings on V-slot weights for global attention layers.",
        "upstream_pr": "https://github.com/vllm-project/vllm/pull/41944",
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_07": {
        "title": "FP8_BLOCK double-scale fix — custom quant config (closes vllm#39407)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_07_GEMMA4_FP8_BLOCK_FIX",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "vllm.sndr_core.integrations.gemma4.g4_07_gemma4_fp8_block_double_scale_fix",
        "lifecycle": "experimental",
        "credit": "Registers gemma4_fp8_block_fix quantization config — bypasses double-scale bug by skipping the second activation quantization.",
        "upstream_pr": "https://github.com/vllm-project/vllm/issues/39407",
        "requires_patches": [],
        "conflicts_with": ["G4_01"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_08": {
        "title": "Marlin K-pad Triton MoE fallback (closes vllm#40354)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_08_MARLIN_KDIM_PAD",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "vllm.sndr_core.integrations.gemma4.g4_08_gemma4_marlin_kdim_pad_fallback",
        "lifecycle": "experimental",
        "credit": "Genesis Triton zero-pad MoE GEMM kernel routes K%64≠0 cases. Unblocks Gemma 4 26B-A4B at TP=2 on Ampere SM 8.6.",
        "upstream_pr": "https://github.com/vllm-project/vllm/issues/40354",
        "requires_patches": [],
        "conflicts_with": ["G4_02"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration"]},
    },
    "G4_09": {
        "title": "SWA→global prefill chunker (workaround vllm#39914 engine hang)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_09_GEMMA4_SWA_PREFILL_CHUNKER",
        "default_on": True,
        "category": "stability",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "vllm.sndr_core.integrations.gemma4.g4_09_gemma4_swa_global_prefill_chunker",
        "lifecycle": "stable",
        "credit": "Clamps scheduler.max_num_batched_tokens to 2048 + forces enable_chunked_prefill=True on Gemma 4 — bypasses #39914 engine hang at prefill>4K.",
        "upstream_pr": "https://github.com/vllm-project/vllm/issues/39914",
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_10": {
        "title": "Ampere non-causal head_dim=256 Triton attention backend (deep fix for vllm#40382)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_10_GEMMA4_AMPERE_NON_CAUSAL_BACKEND",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "vllm.sndr_core.integrations.gemma4.g4_10_gemma4_ampere_non_causal_attn_backend",
        "lifecycle": "experimental",
        "credit": "Genesis Triton attention kernel — head_dim=256 + non-causal block-parallel + BLOCK_M=64 BLOCK_DMODEL=128 (looped twice). Unblocks Eagle3/DFlash on Ampere SM 8.6.",
        "upstream_pr": "https://github.com/vllm-project/vllm/issues/40382",
        "requires_patches": [],
        "conflicts_with": ["G4_03"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_11": {
        "title": "Gemma 4 enhanced chat template install",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_11_GEMMA4_CHAT_TEMPLATE_INSTALL",
        "default_on": True,
        "category": "loader",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "vllm.sndr_core.integrations.gemma4.g4_11_gemma4_chat_template_install",
        "lifecycle": "stable",
        "credit": "Writes enhanced gemma4.jinja chat template to /tmp/genesis/chat_templates/ — supports tool calls + system role + thinking blocks.",
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_12": {
        "title": "Refuse FP8 e4nv on Ampere SM 8.6 (closes vllm#41014)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_12_GEMMA4_FP8_E4NV_GUARD",
        "default_on": True,
        "category": "stability",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "vllm.sndr_core.integrations.gemma4.g4_12_gemma4_fp8_e4nv_ampere_guard",
        "lifecycle": "stable",
        "credit": "Refuses FP8 e4nv Gemma 4 checkpoint on Ampere SM 8.6 at config-verify time. Ampere tensor cores don't support e4nv natively.",
        "upstream_pr": "https://github.com/vllm-project/vllm/issues/41014",
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_13": {
        "title": "Refuse asymmetric per-layer KV-head config (closes vllm#40388 silent corruption)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_13_GEMMA4_PER_TOKEN_HEAD_KV_GUARD",
        "default_on": True,
        "category": "stability",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "vllm.sndr_core.integrations.gemma4.g4_13_gemma4_per_token_head_kv_guard",
        "lifecycle": "stable",
        "credit": "Refuses 26B-A4B (sliding=8 KV-heads, full=2 KV-heads) at config-verify. Prevents silent quality regression from KV page-size mismatch.",
        "upstream_pr": "https://github.com/vllm-project/vllm/issues/40388",
        "requires_patches": [],
        "conflicts_with": [],
        "superseded_by": ["G4_18"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration"]},
    },
    "G4_14": {
        "title": "Gemma 4 tool-call parser pad-token strip (closes vllm#39392)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_14_GEMMA4_TOOL_CALL_PARSER_PAD",
        "default_on": True,
        "category": "stability",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "vllm.sndr_core.integrations.gemma4.g4_14_gemma4_tool_call_parser_pad_token",
        "lifecycle": "stable",
        "credit": "Strips <pad>/<eos>/turn-boundary control tokens from streaming tool-call JSON deltas. Fixes malformed function.arguments JSON.",
        "upstream_pr": "https://github.com/vllm-project/vllm/issues/39392",
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_15": {
        "title": "Fused RMSNorm Triton kernels for Gemma 4 (ported from SGLang)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_15_GEMMA4_FUSED_RMSNORM",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "partial",
        "source": "genesis_original",
        "apply_module": "vllm.sndr_core.integrations.gemma4.g4_15_gemma4_fused_rmsnorm_route",
        "lifecycle": "experimental",
        "credit": "Triton kernels port + integration hooks for Gemma 4 RMSNorm fusion (Q/K/V per-head + residual+scalar + dual-norm MoE reduction). Expected +5-10% TPS on decode at low concurrency. SM 8.6 budget-tuned.",
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_16": {
        "title": "Gemma 4 FULL_AND_PIECEWISE cudagraph mode (parallel to PN125 for gemma4)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_16_GEMMA4_FULL_AND_PIECEWISE",
        "default_on": True,
        "category": "perf_hotfix",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "vllm.sndr_core.integrations.gemma4.g4_16_gemma4_full_piecewise_cudagraph",
        "lifecycle": "stable",
        "credit": "Forces FULL_AND_PIECEWISE cudagraph mode on Gemma 4 dense path. Upstream's splitting_ops heuristic doesn't catch gemma4 model_type. Expected +10-30% TPS on decode at low batch.",
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_17": {
        "title": "Gemma 4 vision-tower text-only skip (closes vllm#41565)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_17_GEMMA4_VISION_SKIP",
        "default_on": False,
        "category": "memory_savings",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "vllm.sndr_core.integrations.gemma4.g4_17_gemma4_vision_tower_text_only_skip",
        "lifecycle": "experimental",
        "credit": "Stubs vision tower + multi_modal_projector when GENESIS_GEMMA4_TEXT_ONLY=1. Saves ~2.3 GB VRAM + ~30 sec cold boot.",
        "upstream_pr": "https://github.com/vllm-project/vllm/issues/41565",
        "requires_patches": [],
        "conflicts_with": ["G4_23"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration"]},
    },
    "G4_18": {
        "title": "Per-layer KV cache page-size for 26B-A4B (vendors WIP vllm#40391)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_18_GEMMA4_PER_LAYER_KV_PAGE_SIZE",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "partial",
        "source": "vendor_backport",
        "apply_module": "vllm.sndr_core.integrations.gemma4.g4_18_gemma4_per_layer_kv_page_size",
        "lifecycle": "experimental",
        "credit": "Hooks ModelConfig.get_num_kv_heads to return per-layer-type KV-head counts for asymmetric 26B-A4B. Closes vllm#40388 root cause (not just guard).",
        "upstream_pr": "https://github.com/vllm-project/vllm/pull/40391",
        "requires_patches": [],
        "conflicts_with": ["G4_13"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration"]},
    },
    "G4_23": {
        "title": "Gemma 4 vision-tower FP16 overflow fix (closes vllm#40124)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_23_GEMMA4_VISION_FP16_OVERFLOW",
        "default_on": True,
        "category": "stability",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "vllm.sndr_core.integrations.gemma4.g4_23_gemma4_vision_fp16_overflow_fix",
        "lifecycle": "stable",
        "credit": "Forces vision tower to BF16 (or soft-clip fallback) when operator chose FP16. Prevents NaN propagation from patch-embed overflow.",
        "upstream_pr": "https://github.com/vllm-project/vllm/issues/40124",
        "requires_patches": [],
        "conflicts_with": ["G4_17"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration"]},
    },
    "G4_24": {
        "title": "Fused softcap Triton kernel route for Gemma 4 (attention + final logits)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_24_GEMMA4_FUSED_SOFTCAP",
        "default_on": False,
        "category": "kernel",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "vllm.sndr_core.integrations.gemma4.g4_24_gemma4_fused_softcap_route",
        "lifecycle": "experimental",
        "credit": "Triton kernel fuses div+tanh+mul for softcap calls. Routes final-logit softcap via wrapper. Expected +3-5% TPS on decode at low batch.",
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "composes_with": ["G4_15"],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
    "G4_25": {
        "title": "Gemma 4 dual-RoPE base-freq divergence guard (long-context quality)",
        "tier": "community",
        "family": "gemma4",
        "env_flag": "GENESIS_ENABLE_G4_25_GEMMA4_RoPE_DUAL_BASE_GUARD",
        "default_on": True,
        "category": "stability",
        "implementation_status": "full",
        "source": "genesis_original",
        "apply_module": "vllm.sndr_core.integrations.gemma4.g4_25_gemma4_rope_dual_base_freq_guard",
        "lifecycle": "stable",
        "credit": "Diagnoses single-table-collapse when rope_theta == global_rope_theta. Warns operator to fix config.json to distinct values.",
        "upstream_pr": None,
        "requires_patches": [],
        "conflicts_with": [],
        "applies_to": {"model_arch": ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]},
    },
}


# ─── Legacy-register patch index (Entry 17 §6.8 P-2 documentation) ────
# `_apply_module_overlay.APPLY_MODULE_OVERLAY` is the index of 127
# patches that today still live in the monolithic
# `vllm/sndr_core/apply/_per_patch_dispatch.py`. The patches-prove gate
# (§6.8 rule P-2) consults `apply._state.PATCH_REGISTRY` for legacy
# register membership — Phase 10 migration moves each patch into
# `integrations/<family>/<patch>.py`, and the integration-tree walk in
# `dispatcher.spec._build_apply_module_map` picks up the new home
# automatically.
#
# IMPORTANT: this index is NOT applied to PATCH_REGISTRY entries. The
# spec-loop dry-run requires a callable `apply()` in the resolved
# module, which `_per_patch_dispatch.py` doesn't expose (it has 95
# `apply_patch_X` functions, not a generic apply). Writing the overlay
# into PATCH_REGISTRY would mislead the spec-loop into trying to import
# a non-existent `apply()`. Keeping the index file as documentation
# avoids that pitfall while still letting `discover_apply_modules.py`
# track migration progress.
