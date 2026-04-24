# SPDX-License-Identifier: Apache-2.0
"""Genesis upstream-compat markers — detect when upstream merges our fixes.

Each Genesis patch targets a specific upstream issue or PR. When that
upstream change lands, we want to auto-skip our patch (no need to
re-apply something the engine already has).

This module centralizes the markers used by each patch's Layer 3 check.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations


# Upstream PR → marker string mapping.
#
# Each value is a marker string that, if present in the target file's
# source, indicates the upstream fix has landed and our patch should skip.
#
# Sources:
#   - Upstream PR descriptions (identifying code added by the PR)
#   - Verified via reading merged commit diffs
#
# Audit against vllm-project/vllm main @ commit cde8d24 (2026-04-24)
# -------------------------------------------------------------------
#   MERGED (skip our patch when marker present):
#     - PR #39016: _prepare_expert_assignment present in fused_moe.py ✓
#     - PR #39391: isnan||isinf in csrc/moe/topk_softmax_kernels.cu    ✓
#     - PR #40172: postprocess_mamba() uses .get() in mamba_utils.py   ✓
#
#   STILL NEEDED (NOT in upstream main):
#     - Marlin bsm env override (P17/P18a)
#     - TritonFp8BlockScaledMM Ampere-guard (P1/P2) — upstream says
#       is_supported=True on Ampere but kernel produces wrong numerics
#     - GDN dual-stream aux_stream (P7) — no multi_stream in gdn_linear_attn
#     - block_table tail zero-fill (P14)
#     - TQ decode stage1 env tune (P18b) — BLOCK_KV=4 hardcoded
#     - TQ continuation_prefill FP16 rotation (P20) — no Pi_half
#     - MoE router fp32 upcast (P31) — universal improvement
#
#   PARTIAL (upstream has lazy-alloc, we add profiler visibility):
#     - P22 TQ dequant buffers: upstream allocates in forward path
#       (profiler-invisible → #40420-class OOM); our patch allocates
#       in _ensure_on_device so profiler counts it before KV sizing.
UPSTREAM_MARKERS: dict[str, dict[str, str]] = {
    "PR_39016_moe_naive_block_fast_path": {
        "file": "model_executor/layers/fused_moe/fused_moe.py",
        "marker": "_prepare_expert_assignment",
        "description": "MoE Triton perf regression restored; helper function added",
        "merged_date": "2026-04-21",
        "affects_patch": "P9 MoE naive_block_assignment",
        "verified_in_main_2026_04_24": True,
    },

    "PR_39391_moe_nan_clamp_cuda_kernel": {
        "file": "csrc/moe/topk_softmax_kernels.cu",
        "marker": "if (isnan",
        "description": "CUDA-level NaN clamp; Python nan_to_num becomes defense-in-depth",
        "merged_date": "2026-04-21",
        "affects_patch": "P11 MoE NaN guard",
        "verified_in_main_2026_04_24": True,
    },

    "PR_39953_tq_int64_cast_ops": {
        "files": [
            "v1/attention/ops/triton_turboquant_decode.py",
            "v1/attention/ops/triton_turboquant_store.py",
        ],
        "marker_decode": "tl.cast(kv_head, tl.int64)",
        "marker_store": "head_idx_i64 = tl.cast(head_idx, tl.int64)",
        "description": "TurboQuant int64 stride overflow fix (ROCm-tagged)",
        "merged_date": "2026-04-17",
        "affects_patch": "P16 TQ int64 + FA2 compat",
    },

    "PR_40060_tq_backend_selector_guard": {
        "file": "v1/attention/backends/turboquant_attn.py",
        "marker": "(earlier PR; patch 7 dropped in v5.6)",
        "description": "TurboQuant backend selector guard",
        "merged_date": "2026-04-17",
        "affects_patch": "(was) P7 — dropped",
    },

    "PR_40105_marlin_in_block_kernel_selection": {
        "file": "model_executor/kernels/linear/scaled_mm/__init__.py",
        "marker": "issubclass(kernel_type, FP8ScaledMMLinearKernel)",
        "description": "Marlin added to block kernel list",
        "merged_date": "2026-04",
        "affects_patch": "P2 Marlin FP8 fallback",
    },

    "PR_40159_mypy_model_executor_layers": {
        "file": "model_executor/layers/mamba/gdn_linear_attn.py",
        "marker": "(removed: from vllm.v1.attention.backend import AttentionMetadata)",
        "description": "MyPy cleanup; removed unused import that was our P7 anchor reference",
        "merged_date": "2026-04-22",
        "affects_patch": "P7 Dual-stream GDN (anchor re-trim required)",
    },

    "PR_40172_mamba_postprocess_fused": {
        "file": "v1/worker/mamba_utils.py",
        "marker": "postprocess_mamba",
        "description": "Mamba state postprocessing uses dict.get() (our P25 is redundant)",
        "merged_date": "2026-04-24 VERIFIED",
        "affects_patch": "P25 mamba_utils .get() guard — MERGED, our patch auto-skips",
        "notes": "Fused-kernel variant (+15-17% decode) still tracked separately",
        "verified_in_main_2026_04_24": True,
    },

    "PR_40194_tq_random_signs_removal": {
        "file": "v1/attention/backends/turboquant_attn.py",
        "marker": "(removed: layer._tq_signs buffer; docstring cites HIGGS prior art)",
        "description": "TurboQuant: remove redundant random signs, add prior art attribution",
        "merged_date": "2026-04-18",
        "affects_patch": "P22 TQ shared dequant (anchor already post-#40194)",
    },

    "PR_40384_hybrid_kv_cache_exclude_mamba_groups": {
        "files": [
            "v1/core/kv_cache_utils.py",
            "v1/core/sched/scheduler.py",
        ],
        "marker": "token_capacity_kv_cache_groups",
        "description": "Exclude O(1) Mamba groups from hybrid KV cache token capacity (Sander co-author, commit b5e1a26)",
        "merged_date": "OPEN as of 2026-04-24",
        "affects_patch": "P8/P9 KV cache reporting",
    },

    "PR_40572_marlin_moe_relocation": {
        "files_removed": ["model_executor/layers/fused_moe/fused_marlin_moe.py"],
        "files_added": ["model_executor/layers/fused_moe/experts/marlin_moe.py"],
        "description": "Move Marlin MoE implementation to experts/ subpackage",
        "merged_date": "OPEN as of 2026-04-24",
        "affects_patch": "P17/P18 Marlin bsm env override (filepath migration needed)",
    },

    "PR_40633_jartx_int4_int2_kv": {
        "files_added": [
            "v1/attention/ops/triton_quant_kv/",
        ],
        "marker": "INT4_PER_TOKEN_HEAD",
        "description": "JartX next-gen INT4/INT2 per-token-head KV cache quantization",
        "merged_date": "OPEN as of 2026-04-24",
        "affects_patch": "New option: may supersede our turboquant_k8v4 path",
    },

    "PR_38479_turboquant_upstream_k8v4": {
        "file": "v1/attention/backends/turboquant_attn.py",
        "marker": "turboquant_k8v4",
        "description": "Upstream merged TurboQuant 2-bit KV cache compression",
        "merged_date": "2026-04 (in v0.20.0)",
        "affects_patch": "P3-P6, P20, P22 — verify our TQ coexists with upstream",
    },

    "PR_39591_block_table_tail_zero": {
        "file": "v1/worker/block_table.py",
        "marker": "#39589",
        "description": "block_table tail zero-fill — prevents stale IDs leaking "
                       "past num_blocks_per_row when a shorter request reuses a "
                       "previously-longer row slot",
        "merged_date": "2026-04 (verify via main)",
        "affects_patch": "P14 block_table tail zero-fill",
    },

    "PR_JARTX_11_continuation_prefill_fp16": {
        "file": "v1/attention/backends/turboquant_attn.py",
        "marker": "Pi_half",
        "description": "JartX/vllm#11 — FP16 rotation in _continuation_prefill "
                       "(halves peak memory at long prefill, fixes #40420 cliff)",
        "merged_date": "OPEN as of 2026-04-24 (vendor fork only)",
        "affects_patch": "P20 TQ _continuation_prefill peak-mem",
    },

    "PR_39589_tq_decode_stage1_tunables": {
        "file": "v1/attention/ops/triton_turboquant_decode.py",
        "marker": "VLLM_TQ_DECODE_BLOCK_KV",
        "description": "Env-driven TQ decode stage1 tune — exposes BLOCK_KV / "
                       "num_warps / num_stages so non-H100 cards (e.g. A5000) "
                       "can tune away from H100-shaped defaults",
        "merged_date": "NOT MERGED (Genesis-only, candidate PR target)",
        "affects_patch": "P18b TQ decode stage1 tune",
    },

    # ── Additional upstream audits (Phase 3 step 4, 2026-04-24) ──

    "PR_40074_tq_decode_int64": {
        "file": "v1/attention/ops/triton_turboquant_decode.py",
        "marker": "tl.cast",  # 3 matches of tl.cast → int64 casts are in
        "description": "TurboQuant decode IOOB fix via int64 index casts",
        "merged_date": "2026-04-24 VERIFIED in main",
        "affects_patch": "P13 TQ decode IOOB — MERGED, our patch auto-skips",
        "verified_in_main_2026_04_24": True,
    },

    "PR_38996_qwen3_none_null": {
        "file": "tool_parsers/qwen3coder_tool_parser.py",
        "marker": '("null", "none")',
        "description": "Qwen3 chat-template None vs JSON null — parser accepts both",
        "merged_date": "NOT MERGED (verified 2026-04-24)",
        "affects_patch": "P15 — our patch still required",
        "verified_in_main_2026_04_24": False,
    },

    "PR_39908_bf16_fp8_ampere": {
        "file": "v1/attention/ops/triton_turboquant_store.py",
        "marker": "tl.float16).to(tl.float8e4b15)",
        "description": "BF16→FP16→FP8 cast chain for Ampere (convert_custom_float8_sm80)",
        "merged_date": "NOT MERGED (verified 2026-04-24)",
        "affects_patch": "P3 TQ BF16→FP8 — our patch still required on Ampere",
        "verified_in_main_2026_04_24": False,
    },

    "PR_40572_marlin_moe_relocation": {
        "file_moved_from": "model_executor/layers/fused_moe/fused_marlin_moe.py",
        "file_moved_to": "model_executor/layers/fused_moe/experts/marlin_moe.py",
        "description": "Marlin MoE module moved into experts/ subpackage",
        "merged_date": "NOT MERGED (verified 2026-04-24 — old path still exists)",
        "affects_patch": "P17/P18 Marlin bsm env override — watch for anchor break",
        "verified_in_main_2026_04_24": False,
        "action_when_merged": "update anchor paths in marlin_tuning wiring",
    },
}


def get_marker(pr_key: str) -> dict[str, str] | None:
    """Fetch marker info for a given upstream PR key."""
    return UPSTREAM_MARKERS.get(pr_key)


def all_markers() -> dict[str, dict[str, str]]:
    """Return all upstream markers for audit/reporting."""
    return dict(UPSTREAM_MARKERS)
