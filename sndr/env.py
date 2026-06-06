# SPDX-License-Identifier: Apache-2.0
"""SNDR Core env-flag registry — single source of truth for ALL env vars.

Per Sander Q4 decision (2026-05-07): all flags enumerated as constants
on the `Flags` class. `is_enabled()` adds the right prefix at lookup
time and supports back-compat between SNDR_ENABLE_* and GENESIS_ENABLE_*.

Per Sander Q2 decision (2026-05-07, mixed branding):
  - Backport patches keep `GENESIS_ENABLE_*` env name.
  - Sander-original (tier=engine) patches use `SNDR_ENABLE_*` canonical.
  - Both prefixes are recognized by `is_enabled()`. SNDR_* wins if
    BOTH are set (giving Sander-IP override priority on community
    deployments).

Why this file exists:
  Before Stage 2, each of 102 patch wirings rolled its own
    `os.environ.get("GENESIS_ENABLE_<flag>", "").strip().lower() in ("1", ...)`
  - No registry of "what flags exist" → typos silent (typoed flag
    just defaults off).
  - No SNDR_/GENESIS_ aliasing → tier=engine patches stuck with
    GENESIS_ branding.
  - No type-checking of flag names → drift between docstrings and code.

Now:
  - `Flags` class — every known flag enumerated as a class constant.
  - `is_enabled(flag, default)` — checks SNDR_ENABLE_* THEN GENESIS_ENABLE_*.
  - `known_flags()` — list for `sndr list-flags` CLI.
  - `boot_audit()` — warns on env vars that look like flags but aren't
    in the registry (catches typos like SNDR_ENABLE_P61C_DEFERED_COMMIT).

Migration status:
  Stage 2 (CURRENT) — registry created; legacy `os.environ.get()` calls
                      in patches still work unchanged.
  Stage 6+        — patches migrate to `is_enabled(Flags.X)`.
  Stage 13        — boot_audit becomes mandatory in apply orchestrator.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


class Flags:
    """All SNDR Core / Genesis env flags as named constants.

    Each constant value is the bare flag name (without prefix).
    `is_enabled()` adds `SNDR_ENABLE_` or `GENESIS_ENABLE_` at lookup.

    Convention:
      - Group by patch family (alphabetical within group).
      - Constant name mirrors patch ID + descriptive suffix.
      - Constant value matches what's currently in dispatcher.PATCH_REGISTRY
        (1:1 mapping for back-compat — Stage 5+ may normalize names).

    Adding a new flag:
      1. Add constant here in the right family group.
      2. Reference via `is_enabled(Flags.X)` from the patch.
      3. Register in dispatcher entry with `env_flag = Flags.X`.
    """

    # ── Backport patches (community tier — keep GENESIS_ENABLE_ env name) ──
    # tool_parsing family
    P15 = "P15"  # Qwen3 None/null tool arg fix (vllm#38996)
    P61C_QWEN3CODER_DEFERRED_COMMIT = "P61C_QWEN3CODER_DEFERRED_COMMIT"
    P64_QWEN3CODER_MTP_STREAMING = "P64_QWEN3CODER_MTP_STREAMING"
    PN56_QWEN3CODER_XML_FALLBACK = "PN56_QWEN3CODER_XML_FALLBACK"

    # reasoning family
    P59_QWEN3_TOOL_RECOVERY = "P59_QWEN3_TOOL_RECOVERY"
    P61_QWEN3_MULTI_TOOL = "P61_QWEN3_MULTI_TOOL"
    P61B_STREAMING_OVERLAP = "P61B_STREAMING_OVERLAP"
    PN51_QWEN3_STREAMING_THINKING_DISABLED = "PN51_QWEN3_STREAMING_THINKING_DISABLED"

    # serving family
    P107_MTP_TRUNCATION_DETECTOR = "P107_MTP_TRUNCATION_DETECTOR"
    P68_AUTO_FORCE_TOOL = "P68_AUTO_FORCE_TOOL"
    P69_LONG_CTX_TOOL_REMINDER = "P69_LONG_CTX_TOOL_REMINDER"
    PN16_LAZY_REASONER = "PN16_LAZY_REASONER"
    PN16_V6_STREAMING_TRUNCATOR = "PN16_V6_STREAMING_TRUNCATOR"
    PN58_SPEC_REASONING_BOUNDARY = "PN58_SPEC_REASONING_BOUNDARY"
    PN66 = "PN66"  # multiturn think leak
    PN70_TOOL_SCHEMA_FILTER = "PN70_TOOL_SCHEMA_FILTER"
    P109 = "P109"  # sampling_params vocab-range validators (vllm#42614)

    # attention.gdn family (memory + chunked + streaming)
    P7B = "P7B"
    PN11_GDN_AB_CONTIGUOUS = "PN11_GDN_AB_CONTIGUOUS"
    PN12_FFN_INTERMEDIATE_POOL = "PN12_FFN_INTERMEDIATE_POOL"
    PN13_CUDA_GRAPH_LAMBDA_ARITY = "PN13_CUDA_GRAPH_LAMBDA_ARITY"
    PN25_SILU_INDUCTOR_SAFE = "PN25_SILU_INDUCTOR_SAFE"
    PN32_GDN_CHUNKED_PREFILL = "PN32_GDN_CHUNKED_PREFILL"
    PN50_GDN_FUSED_PROJ = "PN50_GDN_FUSED_PROJ"
    PN54_GDN_CONTIGUOUS_DEDUP = "PN54_GDN_CONTIGUOUS_DEDUP"
    PN59_STREAMING_GDN = "PN59_STREAMING_GDN"
    PN108_FUSED_RECURRENT_PREFILL = "PN108_FUSED_RECURRENT_PREFILL"
    PN102_PARAM_POOL = "PN102_PARAM_POOL"
    PN204_DUAL_STREAM_INPROJ = "PN204_DUAL_STREAM_INPROJ"
    P103 = "P103"  # FLA Cliff 2 chunked
    PN111 = "PN111"  # skip-mamba-postprocess sync (align-mode only; vllm#42574)
    PN116 = "PN116"  # TurboQuant prefill max_seq_len fallback fix (regressor vllm#41434)
    PN119 = "PN119"  # TurboQuant k8v4 GQA head grouping kernel (backport vllm#40792)
    PN118 = "PN118"  # TurboQuant workspace graceful-fallback (backport vllm#42551, P99-compat)
    # PN118 v2 — md5+full-file PoC of the PN119 reference pattern,
    # scoped to workspace.py (one of pn118's two target files).
    # Companion to PN118 (not a replacement; composes via Genesis
    # marker that prevents pn118 from re-anchoring workspace.py once
    # v2 ran). Default OFF — opt-in to A/B test the md5 pattern.
    # v11.1.0 Phase 6 P3.1 closeout PoC.
    PN118_V2_MD5_WORKSPACE = "PN118_V2_MD5_WORKSPACE"
    # PN118 v2 — md5+full-file PoC, turboquant_attn.py scope (sibling
    # to workspace.py v2 patch above). Together the two v2 patches
    # cover pn118's full 2-file scope via md5+full-file pattern,
    # closing the master plan's P3.1 single-file PoC validation.
    # Companion to PN118 (composes, does NOT conflict). Default OFF.
    # v11.2.0 Phase 6 P3.1 continuation.
    PN118_V2_MD5_TURBOQUANT_ATTN = "PN118_V2_MD5_TURBOQUANT_ATTN"

    # GDN spec-decode subfamily
    P60_GDN_NGRAM_FIX = "P60_GDN_NGRAM_FIX"
    P60B_TRITON_KERNEL = "P60B_TRITON_KERNEL"
    PN79_INPLACE_SSM_STATE = "PN79_INPLACE_SSM_STATE"
    # PN79 v2 — md5+full-file PoC, chunk.py scope (sibling 1 of pn79's
    # multi-file conversion). pn79 originally targets 4 files: chunk.py,
    # chunk_delta_h.py, gdn_linear_attn.py, olmo_hybrid.py. The latter 2
    # have drifted out of upstream entirely (gdn split into model-specific
    # files under gdn/, olmo_hybrid removed). This v2 sibling covers
    # chunk.py — 3/7 pn79 anchors apply cleanly on current pin, 4 drifted.
    # md5 pattern documents the drift transparently. Default OFF.
    # v11.2.0 Phase 6 P3.1 continuation.
    PN79_V2_MD5_CHUNK = "PN79_V2_MD5_CHUNK"
    # PN79 v2 — md5+full-file PoC, chunk_delta_h.py scope (sibling 2).
    # 3/4 pn79 anchors apply cleanly on current pin, 1 drifted.
    # Default OFF. v11.2.0 Phase 6 P3.1 continuation.
    PN79_V2_MD5_CHUNK_DELTA_H = "PN79_V2_MD5_CHUNK_DELTA_H"

    # attention.turboquant family (community subset; tier=engine elsewhere)
    P38B_COMPILE_SAFE = "P38B_COMPILE_SAFE"
    P65_TURBOQUANT_SPEC_CG_DOWNGRADE = "P65_TURBOQUANT_SPEC_CG_DOWNGRADE"
    P78_TOLIST_CAPTURE_GUARD = "P78_TOLIST_CAPTURE_GUARD"
    P98 = "P98"   # TQ workspace revert
    P99 = "P99"   # TQ workspace memoize
    P101 = "P101"  # TQ continuation slicing
    PN14_TQ_DECODE_OOB_CLAMP = "PN14_TQ_DECODE_OOB_CLAMP"
    PN30_DS_LAYOUT_SPEC_DECODE = "PN30_DS_LAYOUT_SPEC_DECODE"
    PN31_FA_VARLEN_PERSISTENT_OUT = "PN31_FA_VARLEN_PERSISTENT_OUT"
    PN34_WORKSPACE_LOCK_RELAX = "PN34_WORKSPACE_LOCK_RELAX"

    # attention.flash family
    P15B_FA_VARLEN_CLAMP = "P15B_FA_VARLEN_CLAMP"
    PN17_FA2_LSE_CLAMP = "PN17_FA2_LSE_CLAMP"
    PN28_MERGE_ATTN_NAN_GUARD = "PN28_MERGE_ATTN_NAN_GUARD"
    P100 = "P100"  # FlashInfer full CG specdec

    # spec_decode family (community subset)
    P58_ASYNC_PLACEHOLDER_FIX = "P58_ASYNC_PLACEHOLDER_FIX"
    P66_CUDAGRAPH_SIZE_FILTER = "P66_CUDAGRAPH_SIZE_FILTER"
    P70_AUTO_STRICT_NGRAM = "P70_AUTO_STRICT_NGRAM"
    P71_BLOCK_VERIFY = "P71_BLOCK_VERIFY"
    P75_SUFFIX_DECODING = "P75_SUFFIX_DECODING"
    P77_ADAPTIVE_NGRAM_K = "P77_ADAPTIVE_NGRAM_K"
    P79B_ASYNC_PROPOSER_SYNC = "P79B_ASYNC_PROPOSER_SYNC"
    P79C_STALE_SPEC_TOKEN_CLEANUP = "P79C_STALE_SPEC_TOKEN_CLEANUP"
    P79D_PREEMPT_ASYNC_DISCARD = "P79D_PREEMPT_ASYNC_DISCARD"
    P83 = "P83"  # MTP keep last cached block
    P86 = "P86"  # ngram batch propose linear
    P94 = "P94"  # spec-decode zero-alloc
    # 2026-05-14 PR sweep — backports landed in v11.0.0+wave9_dev338_pr_sweep:
    P108 = "P108"  # MTP draft-loop stream synchronization (vllm#42603)
    PN8_MTP_DRAFT_ONLINE_QUANT = "PN8_MTP_DRAFT_ONLINE_QUANT"
    PN9_INDEPENDENT_DRAFTER_ATTN = "PN9_INDEPENDENT_DRAFTER_ATTN"
    PN33_SPEC_DECODE_WARMUP_K = "PN33_SPEC_DECODE_WARMUP_K"

    # scheduler family
    P62_STRUCT_OUT_SPEC_TIMING = "P62_STRUCT_OUT_SPEC_TIMING"
    P63_MTP_GDN_STATE_RECOVERY = "P63_MTP_GDN_STATE_RECOVERY"
    P74_CHUNK_CLAMP = "P74_CHUNK_CLAMP"
    P84 = "P84"  # hash block size override

    # worker family
    P72_PROFILE_RUN_CAP = "P72_PROFILE_RUN_CAP"
    P95 = "P95"  # marlin TP cudagraph cap
    PN19_SCOPED_MAX_SPLIT = "PN19_SCOPED_MAX_SPLIT"
    PN35_INPUTS_EMBEDS_OPTIONAL = "PN35_INPUTS_EMBEDS_OPTIONAL"
    PN52_PROMPT_LOGPROBS_EVICTION = "PN52_PROMPT_LOGPROBS_EVICTION"
    PN55_WAKE_UP_HYBRID_KV = "PN55_WAKE_UP_HYBRID_KV"
    PN67 = "PN67"  # thinking budget inverted bool
    PN78_POST_WARMUP_CACHE_RELEASE = "PN78_POST_WARMUP_CACHE_RELEASE"
    # PR38 Day 1 (2026-05-07): backport of vllm#41873 — Mamba CUDA-graph
    # padded rows kept stale `is_prefilling=True` after condense(),
    # misleading Mamba/hybrid backends into prefill on padding.
    PN82_MAMBA_CUDAGRAPH_PREFILL_ZERO = "PN82_MAMBA_CUDAGRAPH_PREFILL_ZERO"
    # Wave 3.1 / vllm#40269 — Probabilistic draft rejection (PN90)
    PN90_PROBABILISTIC_DRAFT = "PN90_PROBABILISTIC_DRAFT"
    # Path C v7.73.x / club-3090 #58 — tier-aware KV cache (PN95)
    PN95_TIER_AWARE_CACHE = "PN95_TIER_AWARE_CACHE"
    # Sprint 2.6 v2 — CUDA graph dispatch trace wire-in (PN122).
    # Canonical constant is defined further below at the dispatcher-trace
    # block (PN122_CG_DISPATCH_TRACE = "PN122_CG_DISPATCH_TRACE"). The
    # earlier `= "PN122"` alias and the legacy SPRINT26_CG_DISPATCH_TRACE
    # alias were dead Python code (zero callsites, produced an orphan
    # "PN122" value in known_flags()); dropped 2026-05-28 STAGE-6-
    # HARDENING.2B. The runtime legacy env-var alias
    # GENESIS_ENABLE_SPRINT26_CG_DISPATCH_TRACE continues to be read by
    # string literal in apply/_per_patch_dispatch.py — unaffected.
    # PN282 — Spec-decode acceptance proxy metric (production sibling
    # of PN248's debug log trace). Wraps rejection_sample and emits
    # sndr_spec_decode_* Prometheus series on the worker's existing
    # /metrics endpoint. Canonical env: SNDR_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC;
    # legacy alias GENESIS_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC warns once.
    # Boot-applied from sndr_core/__init__.py, not via dispatcher (matches
    # PN248 sibling pattern).
    # 2026-05-28 STAGE-6-HARDENING.2C — Flag value tracks the canonical
    # env-var tail (SNDR_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC) so the
    # bidirectional Flags ↔ registry coverage check passes after PN282
    # was registered as a coordinator entry. The attribute name keeps
    # the PN282_ prefix for ID symmetry with the rest of the Flags class;
    # is_enabled() reads the value, so the env var name resolves to
    # SNDR_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC as before.
    PN282_SPEC_DECODE_ACCEPTANCE_METRIC = "SPEC_DECODE_ACCEPTANCE_METRIC"

    # PN283 — vLLM v1 multiprocess Prometheus directory bootstrap.
    # Sibling coordinator of PN282: boot-applied from sndr_core/__init__.py
    # (not via dispatcher pipeline) so that PROMETHEUS_MULTIPROC_DIR is
    # writable BEFORE any patch hook (and the first
    # prometheus_client value-file open) runs. Canonical env name
    # SNDR_ENABLE_PN283_PROC_BRIDGE — keeps PN282-established SNDR_*
    # naming for non-dispatcher coordinator boot patches. Phase 10.5
    # 2026-06-01 registration closes the orphan-flag gap surfaced
    # by audit_config_keys / audit_v2_env_keys after the chat-K3
    # profile promotion declared the env in
    # gemma4-31b-tq-mtp-{chat-k3,structured-k4} profiles.
    PN283_PROC_BRIDGE = "PN283_PROC_BRIDGE"

    # SNDR_MTP_DYNAMIC_K_001 — Genesis-original adaptive K MTP proposer
    # (Sandermage port of vllm#26504 DynamicProposer to DraftModelProposer).
    # Canonical env: GENESIS_ENABLE_SNDR_MTP_DYNAMIC_K_001 (the SNDR_*
    # prefix appears in the suffix because the patch ID itself uses the
    # SNDR_* namespace per Sander's tier=engine naming for Genesis-original
    # patches; the env strips ENABLE_ leaving SNDR_MTP_DYNAMIC_K_001).
    # Closes Phase 10.5 enterprise sweep 2026-06-01 — bidirectional Flags
    # ↔ registry coverage check now passes for this entry.
    SNDR_MTP_DYNAMIC_K_001 = "SNDR_MTP_DYNAMIC_K_001"

    # SNDR_EAGLE3_AUX_HIDDEN_001 — Genesis-original EAGLE-3 model-side prep
    # (Sandermage). Provides the aux_hidden_state hook API surface so when
    # a Qwen3.6 EAGLE-3 drafter checkpoint lands the wire-up is <1 day.
    # Default OFF; with no caller invoking the helpers, zero runtime cost
    # on the target model. Layer-id selection via env
    # GENESIS_SNDR_EAGLE3_AUX_LAYER_IDS (comma-separated).
    # Phase 7 readiness — vllm#35029 / #35040 EAGLE-3 V2 ModelRunner is
    # already in our pin; #43132 Qwen3 EAGLE-3 still open.
    SNDR_EAGLE3_AUX_HIDDEN_001 = "SNDR_EAGLE3_AUX_HIDDEN_001"

    # G4_T1 — Gemma4 tool-parser PR #42006 vendor marker. Operator-side
    # bind-mount overlay (the upstream gemma4 tool-call parser file) is
    # active if and only if the operator launcher binds the vendored file
    # into the container; this flag's env tail
    # GENESIS_INFO_G4_T1_PR42006_OVERLAY_MOUNTED documents that mount
    # state for audit/explain tooling. INFO-semantic prefix (no toggle —
    # the flag reports an external condition rather than gating apply
    # behavior). Closes Phase 10.5 enterprise sweep 2026-06-01.
    G4_T1_PR42006_OVERLAY_MOUNTED = "G4_T1_PR42006_OVERLAY_MOUNTED"

    # kv_cache family
    P5B = "P5B"  # page size pad smaller
    P85 = "P85"  # hybrid fine shadow prefix cache
    P102 = "P102"
    PN110 = "PN110"  # BlockPool.free_blocks deduplication (vllm#42615)

    # moe family
    P37 = "P37"  # MoE intermediate cache
    PN27_REVERT_PLUGGABLE_MOE = "PN27_REVERT_PLUGGABLE_MOE"
    # Wave 9 dev209 perf-restore — persistent Marlin MoE workspace.
    # Canonical Flag is PN96B (defined elsewhere in this class). The
    # bare `PN96 = "PN96"` alias was dead Python code (zero callsites,
    # produced an orphan "PN96" value in known_flags()); dropped
    # 2026-05-28 STAGE-6-HARDENING.2B. The legacy env-var compat shim
    # GENESIS_ENABLE_PN96 / GENESIS_DISABLE_PN96 continues to be read
    # by string literal in
    # integrations/moe/pn96b_marlin_persistent_workspace.py (lines
    # 90-91 + 115) — unaffected by this cleanup.

    # quantization family
    P81_FP8_BLOCK_SCALED_M_LE_8 = "P81_FP8_BLOCK_SCALED_M_LE_8"
    P87 = "P87"  # marlin pad sub-tile
    P91 = "P91"  # autoround row group cdiv (core: auto_gptq + parameter)
    P91B = "P91B"  # autoround row group cdiv multi-scheme (inc + ct_wNa16 + ct_w4a8_fp8)
    PN77_FP8_LM_HEAD = "PN77_FP8_LM_HEAD"

    # loader family
    PN61 = "PN61"  # qwen3 VL key error guard

    # middleware family
    PN40_DFLASH_OMNIBUS = "PN40_DFLASH_OMNIBUS"  # also PN40-classifier in dispatcher
    PN62 = "PN62"  # text-only VIT skip (multimodal subsystem)
    PN65 = "PN65"  # access log

    # lora family
    PN80_LORA_TENSORIZER_DEVICE = "PN80_LORA_TENSORIZER_DEVICE"

    # ── Sander-original (tier=engine — canonical SNDR_ENABLE_ env name) ──
    # These flag values still mirror existing GENESIS_ENABLE_* names for
    # back-compat at Stage 2. Stage 10 may rename to SNDR_ENABLE_<canonical>.
    P67_TQ_MULTI_QUERY_KERNEL = "P67_TQ_MULTI_QUERY_KERNEL"
    P67_SPARSE_V = "P67_SPARSE_V"
    # Flags.P67B_SPEC_VERIFY_ROUTING was a reserved-for-future Flag that
    # never received an env-var binding (comment said "P67 reuse") and
    # accumulated zero callsites in the entire codebase. Dropped
    # 2026-05-28 STAGE-6-HARDENING.2D. If a future P67B patch lands, add
    # the Flag at that time with a concrete env name.
    P82 = "P82"  # SGLang acceptance threshold
    PN21_DFLASH_SWA = "PN21_DFLASH_SWA"
    PN22_LOCAL_ARGMAX_TP = "PN22_LOCAL_ARGMAX_TP"
    PN23_DFLASH_DTYPE_FIX = "PN23_DFLASH_DTYPE_FIX"
    PN24_DFLASH_AUX_LAYER_FIX = "PN24_DFLASH_AUX_LAYER_FIX"
    PN26_SPARSE_V = "PN26_SPARSE_V"
    PN26_TQ_UNIFIED = "PN26_TQ_UNIFIED"
    PN29_GDN_SCALE_FOLD = "PN29_GDN_SCALE_FOLD"
    PN38_DFLASH_QUANT_DRAFTER = "PN38_DFLASH_QUANT_DRAFTER"
    PN57_TQ_CENTROIDS_DISK_CACHE = "PN57_TQ_CENTROIDS_DISK_CACHE"
    PN72_FREQUENCY_NGRAM_DRAFTER = "PN72_FREQUENCY_NGRAM_DRAFTER"

    # ── PN60-PN64 default-on with bare flag names (legacy from registry) ──
    PN60 = "PN60"
    PN63 = "PN63"
    PN64 = "PN64"

    # ── Other bare-name flags ──────────────────────────────────────────
    P40 = "P40"  # TQ grouped decode

    # ── Legacy default-on patches (use is_legacy_active() / GENESIS_LEGACY_* env) ──
    # Disabled by setting GENESIS_LEGACY_X=0. These are patches from the
    # pre-v6 era that have proven stable enough to be default-on, but kept
    # an env knob for emergency disable. Lifecycle = "legacy" in registry.
    LEGACY_P1 = "P1"
    LEGACY_P3 = "P3"
    LEGACY_P4 = "P4"
    LEGACY_P5 = "P5"
    LEGACY_P6 = "P6"
    LEGACY_P7 = "P7"
    LEGACY_P8 = "P8"
    LEGACY_P12 = "P12"
    LEGACY_P14 = "P14"
    LEGACY_P15 = "P15"
    LEGACY_P17 = "P17"
    LEGACY_P18B = "P18B"
    LEGACY_P20 = "P20"
    LEGACY_P22 = "P22"
    LEGACY_P23 = "P23"
    LEGACY_P24 = "P24"
    LEGACY_P26 = "P26"
    LEGACY_P27 = "P27"
    LEGACY_P28 = "P28"
    LEGACY_P29 = "P29"
    LEGACY_P31 = "P31"
    LEGACY_P32 = "P32"
    LEGACY_P34 = "P34"
    LEGACY_P36 = "P36"
    LEGACY_P38 = "P38"
    LEGACY_P39A = "P39A"
    LEGACY_P44 = "P44"
    LEGACY_P46 = "P46"
    LEGACY_P51 = "P51"

    # ── Stage 7 (2026-05-07): bundle umbrella flags ────────────────────
    # Each bundle composes 2+ semantically-related patches via
    # MultiFilePatchTransaction. Setting the umbrella flag triggers
    # atomic apply of ALL sub-patches regardless of their individual
    # env flags. Idempotent — each TextPatcher's marker check ensures
    # no double-apply. See vllm/sndr_core/bundles/ for orchestrators.
    BUNDLE_TOOL_PARSING_QWEN3CODER = "BUNDLE_TOOL_PARSING_QWEN3CODER"
    BUNDLE_REASONING_QWEN3 = "BUNDLE_REASONING_QWEN3"
    BUNDLE_ATTENTION_GDN_SPEC = "BUNDLE_ATTENTION_GDN_SPEC"
    BUNDLE_ATTENTION_TQ_MULTI_QUERY = "BUNDLE_ATTENTION_TQ_MULTI_QUERY"
    BUNDLE_SPEC_DECODE_ASYNC_CLEANUP = "BUNDLE_SPEC_DECODE_ASYNC_CLEANUP"

    # ── Wave 10 backports + experimental patches (2026-05-12+) ─────────
    # Audit 2026-05-16: registry env_flags backfilled here so the
    # Flags class stays the single source of truth for known env vars.
    # Each constant value matches the registry env_flag suffix
    # (after stripping the GENESIS_ENABLE_ / SNDR_ENABLE_ prefix).
    PN71_THINKING_TAG_NORMALIZE = "PN71_THINKING_TAG_NORMALIZE"
    PN73_TOOL_ARGS_SAFE_NORMALIZE = "PN73_TOOL_ARGS_SAFE_NORMALIZE"
    PN91_DEVELOPER_ROLE = "PN91_DEVELOPER_ROLE"
    PN92_NIXL_EP_TRIAL_IMPORT = "PN92_NIXL_EP_TRIAL_IMPORT"
    PN96_EMERGENCY_DEMOTE = "PN96_EMERGENCY_DEMOTE"
    PN96B = "PN96B"
    PN97_TENSOR_PHYSICAL_CAP = "PN97_TENSOR_PHYSICAL_CAP"
    PN104_OFFLOAD_PREFETCH_REDIRECT = "PN104_OFFLOAD_PREFETCH_REDIRECT"
    PN105_AUTOROUND_OFFLOAD_COMPAT = "PN105_AUTOROUND_OFFLOAD_COMPAT"
    PN106_GDN_H_POOL = "PN106_GDN_H_POOL"
    PN122_CG_DISPATCH_TRACE = "PN122_CG_DISPATCH_TRACE"
    PN125_HYBRID_FULL_AND_PIECEWISE = "PN125_HYBRID_FULL_AND_PIECEWISE"
    PN126_V1_DECODE_WARMUP = "PN126_V1_DECODE_WARMUP"
    PN127_AUTO_CHAT_TEMPLATE = "PN127_AUTO_CHAT_TEMPLATE"
    PN128_SPEC_DECODE_WARMUP = "PN128_SPEC_DECODE_WARMUP"
    PN129_SLOT_MAPPING_WARMUP = "PN129_SLOT_MAPPING_WARMUP"
    PN130_TQ_DECODE_WARMUP = "PN130_TQ_DECODE_WARMUP"
    PN132_TOPK_TOPP_CONTIGUOUS = "PN132_TOPK_TOPP_CONTIGUOUS"
    PN133_MTP_EMPTY_OUTPUT_FIX = "PN133_MTP_EMPTY_OUTPUT_FIX"
    PN134_TORCH_COMPILE_FULLGRAPH_211 = "PN134_TORCH_COMPILE_FULLGRAPH_211"
    PN200_GDN_SCRATCH_REUSE = "PN200_GDN_SCRATCH_REUSE"
    PN201_SCHEDULER_EMPTY_CACHE = "PN201_SCHEDULER_EMPTY_CACHE"
    PN202_PER_LAYER_KV_SPLIT = "PN202_PER_LAYER_KV_SPLIT"
    PN203_COLD_PREFIX_OFFLOAD = "PN203_COLD_PREFIX_OFFLOAD"
    SNDR_WORKSPACE_001 = "SNDR_WORKSPACE_001"

    # ── Gemma 4 family (G4_NN — 2026-05-17) ────────────────────────────
    # 21 patches covering: refusal guards (G4_01/02/03/12/13), vendor
    # backports (G4_04/05/06/18), deep fixes (G4_07/08/09/10), perf kernels
    # (G4_15/16/24), compatibility (G4_11/14), vision-tower management
    # (G4_17/23), and diagnostic (G4_25). Family lives at
    # vllm/sndr_core/integrations/gemma4/. See FAMILY_README for the
    # operator-facing rollout matrix.
    #
    # Implementation status snapshot per audit GEMMA4_PATCH_OPTIMIZATION_PLAN_2026-05-17_RU:
    #   stable / full         : G4_01..G4_05, G4_07, G4_09, G4_11..G4_14, G4_16, G4_17, G4_23, G4_25
    #   partial / experimental: G4_06 (k_eq_v half), G4_08 (AWQ MoE stub),
    #                           G4_10 (DFlash backend brittle, class typo),
    #                           G4_15 (no-op hot path, needs G4_15b deep anchor patch),
    #                           G4_18 (only get_num_kv_heads, not KV spec build),
    #                           G4_24 (only final logits softcap, not attention softcap)
    G4_01_GEMMA4_FP8_BLOCK_GUARD = "G4_01_GEMMA4_FP8_BLOCK_GUARD"
    G4_02_GEMMA4_MARLIN_KDIM_GUARD = "G4_02_GEMMA4_MARLIN_KDIM_GUARD"
    G4_03_GEMMA4_NON_CAUSAL_DRAFTER_GUARD = "G4_03_GEMMA4_NON_CAUSAL_DRAFTER_GUARD"
    G4_04_GEMMA4_AWQ_MOE_KEYS_REMAP = "G4_04_GEMMA4_AWQ_MOE_KEYS_REMAP"
    G4_05_GEMMA4_DFLASH_BACKEND_AUTOSELECT = "G4_05_GEMMA4_DFLASH_BACKEND_AUTOSELECT"
    G4_06_GEMMA4_KV_PROJ_V0 = "G4_06_GEMMA4_KV_PROJ_V0"
    G4_07_GEMMA4_FP8_BLOCK_FIX = "G4_07_GEMMA4_FP8_BLOCK_FIX"
    G4_08_GEMMA4_MARLIN_KDIM_PAD = "G4_08_MARLIN_KDIM_PAD"
    G4_09_GEMMA4_SWA_PREFILL_CHUNKER = "G4_09_GEMMA4_SWA_PREFILL_CHUNKER"
    G4_10_GEMMA4_AMPERE_NON_CAUSAL_BACKEND = "G4_10_GEMMA4_AMPERE_NON_CAUSAL_BACKEND"
    G4_11_GEMMA4_CHAT_TEMPLATE_INSTALL = "G4_11_GEMMA4_CHAT_TEMPLATE_INSTALL"
    G4_12_GEMMA4_FP8_E4NV_GUARD = "G4_12_GEMMA4_FP8_E4NV_GUARD"
    G4_13_GEMMA4_PER_TOKEN_HEAD_KV_GUARD = "G4_13_GEMMA4_PER_TOKEN_HEAD_KV_GUARD"
    G4_14_GEMMA4_TOOL_CALL_PARSER_PAD = "G4_14_GEMMA4_TOOL_CALL_PARSER_PAD"
    G4_15_GEMMA4_FUSED_RMSNORM = "G4_15_GEMMA4_FUSED_RMSNORM"
    G4_16_GEMMA4_FULL_AND_PIECEWISE = "G4_16_GEMMA4_FULL_AND_PIECEWISE"
    G4_17_GEMMA4_VISION_SKIP = "G4_17_GEMMA4_VISION_SKIP"
    G4_18_GEMMA4_PER_LAYER_KV_PAGE_SIZE = "G4_18_GEMMA4_PER_LAYER_KV_PAGE_SIZE"
    G4_23_GEMMA4_VISION_FP16_OVERFLOW = "G4_23_GEMMA4_VISION_FP16_OVERFLOW"
    G4_24_GEMMA4_FUSED_SOFTCAP = "G4_24_GEMMA4_FUSED_SOFTCAP"
    # Attribute name kept all-uppercase so `known_flags()` (which filters
    # by `name.isupper()`) picks it up; env_var value preserves the
    # historical mixed-case `RoPE` form to remain compatible with the
    # operator-facing env var emitted in YAML / docker / docs.
    G4_25_GEMMA4_ROPE_DUAL_BASE_GUARD = "G4_25_GEMMA4_RoPE_DUAL_BASE_GUARD"

    # G4_19 — Genesis-original TurboQuant KV cache for Gemma 4 (256K unlock)
    # Companion to our Qwen 3.5/3.6 P67/PN116/PN118/PN119 stack — parallel
    # architecture pattern for gemma4 attention path. Implementation:
    # vllm/sndr_core/integrations/gemma4/kernels/turboquant/.
    G4_19_GEMMA4_TURBOQUANT_KV = "G4_19_GEMMA4_TURBOQUANT_KV"
    # G4_19b — compression-aware KV cache memory check for vLLM v1
    G4_19B_GEMMA4_TQ_KV_SPEC = "G4_19B_GEMMA4_TQ_KV_SPEC"
    # G4_19c — TurboQuantAttention overlay attribute wrap for spec contract
    G4_19C_ATTN_WRAP = "G4_19C_ATTN_WRAP"

    # ── G4_31/G4_32 — TQ dtype + validation overlays ───────────────────
    G4_31_TQ_DTYPE_PRESERVE = "G4_31_TQ_DTYPE_PRESERVE"
    G4_32_TQ_VALIDATION_BYPASS = "G4_32_TQ_VALIDATION_BYPASS"

    # ── G4_60 series — PR42637 TurboQuant overlay (10 verifiers) ───────
    # Each entry is a marker verifier introspecting the bind-mounted
    # overlay surface (vllm/sndr_core/integrations/attention/turboquant/
    # overlays/pr42637/). See module docstrings for runtime semantics.
    G4_60A_TQ_SLIDING_SPEC = "G4_60A_TQ_SLIDING_SPEC"
    G4_60B_TQ_ATTN_OVERLAY = "G4_60B_TQ_ATTN_OVERLAY"
    G4_60C_TQ_DECODE_OVERLAY = "G4_60C_TQ_DECODE_OVERLAY"
    G4_60D_TQ_STORE_OVERLAY = "G4_60D_TQ_STORE_OVERLAY"
    G4_60E_KV_CACHE_UTILS = "G4_60E_KV_CACHE_UTILS"
    G4_60G_TQ_DISPATCH = "G4_60G_TQ_DISPATCH"
    G4_60H_TQ_CONFIG_AUGMENT = "G4_60H_TQ_CONFIG_AUGMENT"
    G4_60K_TQ_ENGINE_CONFIG = "G4_60K_TQ_ENGINE_CONFIG"
    G4_60L_TQ_BACKEND_MM_PREFIX = "G4_60L_TQ_BACKEND_MM_PREFIX"

    # ── G4_61..G4_69 — TQ workspace / warmup / spec route ──────────────
    G4_61_TQ_SHARED_WORKSPACE = "G4_61_TQ_SHARED_WORKSPACE"
    G4_62_TQ_KERNEL_WARMUP = "G4_62_TQ_KERNEL_WARMUP"
    G4_67_TQ_SPEC_VERIFY_ROUTE = "G4_67_TQ_SPEC_VERIFY_ROUTE"
    G4_68_TQ_SPEC_CG_DOWNGRADE_OVERLAY = "G4_68_TQ_SPEC_CG_DOWNGRADE_OVERLAY"
    G4_69_SKIP_LAYERS_NATIVE_BACKEND = "G4_69_SKIP_LAYERS_NATIVE_BACKEND"

    # ── G4_70 series — PN259B/C alloc + routing variants ───────────────
    G4_70_PN259B_FAIL_FAST = "G4_70_PN259B_FAIL_FAST"
    G4_70_PN259B_MIXED_ALLOC = "G4_70_PN259B_MIXED_ALLOC"
    G4_70_PN259C_ROUTE_B = "G4_70_PN259C_ROUTE_B"

    # ── G4_71..G4_78 — DFlash drafter rerouting + Triton kernels ───────
    G4_71_DRAFTER_NATIVE_BACKEND = "G4_71_DRAFTER_NATIVE_BACKEND"
    G4_71B_DRAFTER_SLIDING_TRITON = "G4_71B_DRAFTER_SLIDING_TRITON"
    G4_72_DRAFTER_NATIVE_SPEC = "G4_72_DRAFTER_NATIVE_SPEC"
    G4_73_DRAFTER_PROFILE_SKIP = "G4_73_DRAFTER_PROFILE_SKIP"
    G4_74_DRAFTER_HND_LAYOUT = "G4_74_DRAFTER_HND_LAYOUT"
    G4_75_DRAFTER_HEAD512_TRITON = "G4_75_DRAFTER_HEAD512_TRITON"
    G4_76_DISABLE_DRAFTER_KV_SHARING = "G4_76_DISABLE_DRAFTER_KV_SHARING"
    G4_78_DRAFTER_TARGET_KV_BRIDGE = "G4_78_DRAFTER_TARGET_KV_BRIDGE"

    # ── PN spec-decode / TQ telemetry + safety opt-ins (R3 audit) ──────
    PN256_KPLUS1_RAW_KV = "PN256_KPLUS1_RAW_KV"
    PN261_TQ_NATIVE_CACHE_ASSERT = "PN261_TQ_NATIVE_CACHE_ASSERT"
    PN262_FLASH_ATTN_DRAFTER_TRACE = "PN262_FLASH_ATTN_DRAFTER_TRACE"
    PN262B_KV_ALLOC_TRACE = "PN262B_KV_ALLOC_TRACE"
    PN271_KV_CONTRACT_AUDIT = "PN271_KV_CONTRACT_AUDIT"
    PN275_DFLASH_MAX_CGS_ALIGN = "PN275_DFLASH_MAX_CGS_ALIGN"
    # PN274 uses SNDR_ALLOW_* prefix family for operator consent semantic.
    # Bare-name lookup still works through Flags introspection.
    SPEC_DECODE_KV_ADAPTER = "SPEC_DECODE_KV_ADAPTER"
    PN286_FA_LAYOUT_REVERT_SM86 = "PN286_FA_LAYOUT_REVERT_SM86"
    PN287_QWEN3CODER_ARGS_OBSERVER = "PN287_QWEN3CODER_ARGS_OBSERVER"
    PN288_TOOL_FINISH_REASON_OVERRIDE = "PN288_TOOL_FINISH_REASON_OVERRIDE"
    PN289_PROCESS_INFO = "PN289_PROCESS_INFO"

    # ── Meta flags (apply behavior, not patch enable) ──────────────────
    NO_PATCH_CACHE = "NO_PATCH_CACHE"           # disable file_cache fast-path
    DISABLE_BOOT_PATCHES = "DISABLE_BOOT_PATCHES"  # skip apply_all at boot
    TIER_OVERRIDE = "TIER_OVERRIDE"             # force community-only mode
    FORCE_REAPPLY = "FORCE_REAPPLY"             # bypass marker idempotency
    NO_VERIFY = "NO_VERIFY"                     # skip post-apply verify
    TELEMETRY = "TELEMETRY"                     # opt-in telemetry


# ── Public API ──────────────────────────────────────────────────────────


def is_enabled(flag: str, default: bool = False) -> bool:
    """Check `SNDR_ENABLE_<flag>` first, then `GENESIS_ENABLE_<flag>`.

    Both prefixes work; SNDR_* takes precedence if both are set
    (per Q2 mixed: SNDR_ wins so Sander-IP override is consistent).

    Args:
      flag: bare flag name (no prefix). Typically `Flags.P61C_…` etc.
      default: returned when neither env var is present.

    Returns: bool — True if flag is enabled, False otherwise.

    Usage:
      from vllm.sndr_core.env import Flags, is_enabled
      if not is_enabled(Flags.P61C_QWEN3CODER_DEFERRED_COMMIT):
          return "skipped", "P61C disabled"
    """
    sndr_var = f"SNDR_ENABLE_{flag}"
    genesis_var = f"GENESIS_ENABLE_{flag}"
    val = os.environ.get(sndr_var)
    if val is None:
        val = os.environ.get(genesis_var)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def is_disabled(flag: str, default: bool = False) -> bool:
    """Check `SNDR_DISABLE_<flag>` / `GENESIS_DISABLE_<flag>` opt-out.

    For default-on patches that operators may want to TURN OFF.
    Returns True when DISABLE env is set.
    """
    sndr_var = f"SNDR_DISABLE_{flag}"
    genesis_var = f"GENESIS_DISABLE_{flag}"
    val = os.environ.get(sndr_var)
    if val is None:
        val = os.environ.get(genesis_var)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def is_legacy_active(flag: str, default: bool = True) -> bool:
    """Check `SNDR_LEGACY_<flag>` / `GENESIS_LEGACY_<flag>` for legacy patches.

    Legacy default-on patches (lifecycle="legacy" in registry) honor an
    opt-out knob via this prefix. Set `GENESIS_LEGACY_P5=0` to disable
    P5. Default = True (legacy patches stay on unless explicitly disabled).

    Returns True if the legacy patch should be applied at boot.
    """
    sndr_var = f"SNDR_LEGACY_{flag}"
    genesis_var = f"GENESIS_LEGACY_{flag}"
    val = os.environ.get(sndr_var)
    if val is None:
        val = os.environ.get(genesis_var)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# ──────────────────────────────────────────────────────────────────────
# Generic SNDR/GENESIS-aliased env reader  (P1 naming migration)
# ──────────────────────────────────────────────────────────────────────
#
# `is_enabled/disabled/legacy` cover the well-known ENABLE_/DISABLE_/
# LEGACY_ patterns for patch flags. The spec_decode + gateway layers
# introduced new env vars whose suffixes don't fit those patterns:
#
#   ALLOW_SPEC_DECODE_KV_ADAPTER
#   ALLOW_SPEC_DECODE_FUNCTIONAL_UNKNOWN
#   DISABLE_SPEC_DECODE_SAFETY_GUARD
#   SPEC_DECODE_ARTIFACTS_DIR
#   GATEWAY_DEFAULT_URL
#   GATEWAY_STRUCTURED_URL
#   GATEWAY_PROFILE
#   GATEWAY_BIND_HOST / _PORT / _HEALTH_INTERVAL / _TIMEOUT / _LOG_LEVEL
#   GATEWAY_ADMIN_ALLOW_REMOTE
#
# `get_sndr_env(name, default)` resolves these with the same
# SNDR_/GENESIS_ alias semantics: SNDR_<name> wins; falls back to
# GENESIS_<name> with a one-shot deprecation warning per name.

# Per-name dedup so we warn once per process per env name
_deprecation_warned: set[str] = set()


def get_sndr_env(name: str, default: str | None = None,
                 *, warn_deprecated: bool = True) -> str | None:
    """Read an env var with SNDR_/GENESIS_ alias semantics.

    `name` is the suffix without any prefix (e.g.
    ``ALLOW_SPEC_DECODE_KV_ADAPTER``). Both ``SNDR_<name>`` and
    ``GENESIS_<name>`` are checked. SNDR_ wins if both are set.

    If only GENESIS_<name> is set, returns its value AND emits a
    one-shot deprecation log warning naming the new SNDR_<name>
    canonical form. ``warn_deprecated=False`` suppresses the warning
    (use sparingly, e.g. inside docstring-default config templates).

    Returns ``default`` if neither env is present.
    """
    sndr_var = f"SNDR_{name}"
    genesis_var = f"GENESIS_{name}"
    val = os.environ.get(sndr_var)
    if val is not None:
        return val
    val = os.environ.get(genesis_var)
    if val is not None:
        if warn_deprecated and name not in _deprecation_warned:
            _deprecation_warned.add(name)
            try:
                import logging as _logging
                _logging.getLogger("vllm.sndr_core.env").warning(
                    "%s is deprecated; rename to %s. The alias is "
                    "supported now but will be removed in a future "
                    "release.",
                    genesis_var, sndr_var,
                )
            except Exception:
                pass
        return val
    return default


def get_sndr_env_bool(name: str, default: bool = False) -> bool:
    """Boolean form of get_sndr_env. Treats 1/true/yes/on as True."""
    v = get_sndr_env(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def get_sndr_env_int(name: str, default: int) -> int:
    v = get_sndr_env(name)
    if v is None:
        return default
    try:
        return int(v.strip())
    except (TypeError, ValueError):
        return default


def get_sndr_env_float(name: str, default: float) -> float:
    v = get_sndr_env(name)
    if v is None:
        return default
    try:
        return float(v.strip())
    except (TypeError, ValueError):
        return default


def known_flags() -> list[str]:
    """Return all flag names declared on `Flags` class (sorted)."""
    return sorted([
        getattr(Flags, name) for name in dir(Flags)
        if name.isupper() and isinstance(getattr(Flags, name), str)
    ])


def boot_audit() -> list[str]:
    """Warn if env vars look like SNDR/GENESIS flags but aren't in registry.

    Catches typos: e.g. `SNDR_ENABLE_P61C_DEFERED_COMMIT` (typo in
    'DEFERRED') will produce a warning that the var is unrecognized.
    Returns list of human-readable warning strings.

    Used by apply orchestrator at boot. Returns empty list if all
    flagged env vars match a known flag.
    """
    warnings = []
    known = set(known_flags())

    # Canonical flags + meta flags don't need boot warnings.
    # Test-fixture flags (GENESIS_ENABLE_GOOD, etc.) trigger warnings
    # only if they appear outside of pytest.
    for var in os.environ:
        for prefix in ("SNDR_ENABLE_", "GENESIS_ENABLE_",
                       "SNDR_DISABLE_", "GENESIS_DISABLE_",
                       "SNDR_LEGACY_", "GENESIS_LEGACY_"):
            if var.startswith(prefix):
                bare = var[len(prefix):]
                if bare not in known:
                    warnings.append(
                        f"Unknown env flag {var!r} — not in Flags registry. "
                        f"Possible typo (registry has {len(known)} known flags). "
                        f"Run `sndr list-flags` for valid names."
                    )
                break
    return warnings


def is_meta_flag(flag: str) -> bool:
    """Return True if `flag` is an apply-behavior or orchestration meta flag.

    Two recognized meta families:

      1. Apply-behavior meta (NO_PATCH_CACHE, DISABLE_BOOT_PATCHES,
         TIER_OVERRIDE, FORCE_REAPPLY, NO_VERIFY, TELEMETRY) — control
         the dispatcher / apply pipeline itself, not individual patches.

      2. Stage-7 bundle umbrella flags (BUNDLE_*) — orchestration flags
         that compose 2+ semantically-related patches via
         MultiFilePatchTransaction (see vllm/sndr_core/bundles/). Setting
         an umbrella flag triggers atomic apply of the bundle's
         sub-patches regardless of their individual env flags. They do
         NOT belong in PATCH_REGISTRY because they are orchestrators,
         not patches — and so they are correctly absent from the 1:1
         Flags ↔ registry coverage check.
    """
    if flag.startswith("BUNDLE_"):
        return True
    return flag in (
        Flags.NO_PATCH_CACHE,
        Flags.DISABLE_BOOT_PATCHES,
        Flags.TIER_OVERRIDE,
        Flags.FORCE_REAPPLY,
        Flags.NO_VERIFY,
        Flags.TELEMETRY,
    )


__all__ = [
    "Flags",
    "is_enabled",
    "is_disabled",
    "is_legacy_active",
    "is_meta_flag",
    "known_flags",
    "boot_audit",
]
