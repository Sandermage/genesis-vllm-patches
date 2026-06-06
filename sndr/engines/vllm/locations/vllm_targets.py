# SPDX-License-Identifier: Apache-2.0
"""SNDR Core engine target paths — single source of truth.

This file is THE ONLY place in SNDR Core / Genesis where vllm engine
file paths are hardcoded as strings. All 63 paths that any patch
modifies are defined here as named constants.

Why this exists (per Sander Q3 2026-05-07, mixed centralized + per-subsystem):

    Before Stage 2, each of 102 patch wirings contained its own
    `resolve_vllm_file("tool_parsers/qwen3coder_tool_parser.py")`
    string. When upstream renamed a file (e.g. `tool_parsers/` →
    `tool_parsing/` in some future vllm version), updating required
    grep+replace across 102 files. Drift detection ran 102 times
    instead of once.

    Now there's ONE constant per engine target. Rename in upstream
    → patch ONE constant here. Boot-time validation in
    `dispatcher/audit.py` verifies every constant points at an
    existing file in the installed vllm pin and flags orphans.

Conventions:
  - UPPER_SNAKE_CASE for each constant.
  - Group by vllm top-level subdirectory (preserves vllm taxonomy).
  - One blank line between groups.
  - Comment with intent if path target is non-obvious or rarely
    referenced.
  - Constants reference vllm-relative paths (joined with
    `vllm_install_root()` at call time).

Per-subsystem helpers (e.g. `attention/turboquant/_paths.py`) MAY
re-export specific constants for import-locality, but NEVER
duplicate the path string. The single source of truth is here.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────
# Top-level files
# ─────────────────────────────────────────────────────────────────────
ENVS = "envs.py"

# ─────────────────────────────────────────────────────────────────────
# compilation/ — torch.compile passes + CUDA graph capture
# ─────────────────────────────────────────────────────────────────────
CUDA_GRAPH = "compilation/cuda_graph.py"

# ─────────────────────────────────────────────────────────────────────
# config/ — top-level configuration objects
# ─────────────────────────────────────────────────────────────────────
CONFIG_SCHEDULER = "config/scheduler.py"
CONFIG_SPECULATIVE = "config/speculative.py"
CONFIG_VLLM = "config/vllm.py"

# ─────────────────────────────────────────────────────────────────────
# engine/ — high-level engine entrypoints
# ─────────────────────────────────────────────────────────────────────
ENGINE_ARG_UTILS = "engine/arg_utils.py"

# ─────────────────────────────────────────────────────────────────────
# entrypoints/ — HTTP serving layer (OpenAI-compatible API)
# ─────────────────────────────────────────────────────────────────────
SERVING_CHAT = "entrypoints/openai/chat_completion/serving.py"

# ─────────────────────────────────────────────────────────────────────
# lora/ — LoRA adapter loading + tensorizer
# ─────────────────────────────────────────────────────────────────────
LORA_MODEL = "lora/lora_model.py"

# ─────────────────────────────────────────────────────────────────────
# model_executor/kernels/ — kernel-level (Marlin, mixed precision)
# ─────────────────────────────────────────────────────────────────────
MARLIN_KERNEL = "model_executor/kernels/linear/mixed_precision/marlin.py"

# ─────────────────────────────────────────────────────────────────────
# model_executor/layers/ — generic layers (activation, attention dispatch, linear)
# ─────────────────────────────────────────────────────────────────────
ACTIVATION = "model_executor/layers/activation.py"
ATTENTION_DISPATCH = "model_executor/layers/attention/attention.py"

# ─────────────────────────────────────────────────────────────────────
# model_executor/layers/fla/ops/ — FLA chunked SSM kernel ops
# ─────────────────────────────────────────────────────────────────────
FLA_CHUNK = "model_executor/layers/fla/ops/chunk.py"
FLA_CHUNK_DELTA_H = "model_executor/layers/fla/ops/chunk_delta_h.py"
FLA_CHUNK_O = "model_executor/layers/fla/ops/chunk_o.py"

# ─────────────────────────────────────────────────────────────────────
# model_executor/layers/fused_moe/ — MoE expert routing + fused dispatch
# ─────────────────────────────────────────────────────────────────────
# Upstream PR #40572 moved the Marlin MoE implementation into the
# `experts/` subpackage between dev93 and dev209. Pre-dev209 callers
# resolve FUSED_MARLIN_MOE; dev209+ callers should prefer
# FUSED_MARLIN_MOE_NEW. Patches that text-patch this file try the new
# location first and fall back to the legacy one.
FUSED_MARLIN_MOE = "model_executor/layers/fused_moe/fused_marlin_moe.py"
FUSED_MARLIN_MOE_NEW = "model_executor/layers/fused_moe/experts/marlin_moe.py"
FUSED_MOE = "model_executor/layers/fused_moe/fused_moe.py"
FUSED_MOE_LAYER = "model_executor/layers/fused_moe/layer.py"
FUSED_MOE_RUNNER = "model_executor/layers/fused_moe/runner/moe_runner.py"
FUSED_MOE_RUNNER_INTERFACE = "model_executor/layers/fused_moe/runner/moe_runner_interface.py"

# ─────────────────────────────────────────────────────────────────────
# model_executor/layers/mamba/ — GDN + state-space attention + causal-conv1d
# ─────────────────────────────────────────────────────────────────────
GDN_LINEAR_ATTN = "model_executor/layers/mamba/gdn_linear_attn.py"
MAMBA_UTILS = "model_executor/layers/mamba/mamba_utils.py"
CAUSAL_CONV1D = "model_executor/layers/mamba/ops/causal_conv1d.py"

# ─────────────────────────────────────────────────────────────────────
# model_executor/layers/quantization/ — quantization kernels + utilities
# ─────────────────────────────────────────────────────────────────────
GPTQ_MARLIN = "model_executor/layers/quantization/gptq_marlin.py"
TURBOQUANT_CENTROIDS = "model_executor/layers/quantization/turboquant/centroids.py"
FP8_UTILS = "model_executor/layers/quantization/utils/fp8_utils.py"

# ─────────────────────────────────────────────────────────────────────
# model_executor/model_loader/ — model loading hooks
# ─────────────────────────────────────────────────────────────────────
MODEL_LOADER_UTILS = "model_executor/model_loader/utils.py"

# ─────────────────────────────────────────────────────────────────────
# model_executor/models/ — model class definitions
# ─────────────────────────────────────────────────────────────────────
OLMO_HYBRID = "model_executor/models/olmo_hybrid.py"
QWEN3 = "model_executor/models/qwen3.py"
QWEN3_DFLASH = "model_executor/models/qwen3_dflash.py"
MODELS_UTILS = "model_executor/models/utils.py"

# ─────────────────────────────────────────────────────────────────────
# model_executor/parameter.py — Parameter wrappers (weight_loader hooks)
# ─────────────────────────────────────────────────────────────────────
PARAMETER = "model_executor/parameter.py"

# ─────────────────────────────────────────────────────────────────────
# parser/ — abstract output parser base class
# ─────────────────────────────────────────────────────────────────────
ABSTRACT_PARSER = "parser/abstract_parser.py"

# ─────────────────────────────────────────────────────────────────────
# platforms/ — platform abstraction (CUDA, ROCm, etc.)
# ─────────────────────────────────────────────────────────────────────
PLATFORM_INTERFACE = "platforms/interface.py"

# ─────────────────────────────────────────────────────────────────────
# reasoning/ — reasoning model output parsing (qwen3, deepseek, etc.)
# ─────────────────────────────────────────────────────────────────────
ABS_REASONING_PARSERS = "reasoning/abs_reasoning_parsers.py"
BASIC_PARSERS = "reasoning/basic_parsers.py"
QWEN3_REASONING_PARSER = "reasoning/qwen3_reasoning_parser.py"

# ─────────────────────────────────────────────────────────────────────
# tool_parsers/ — custom tool-call format extraction
# ─────────────────────────────────────────────────────────────────────
QWEN3CODER_TOOL_PARSER = "tool_parsers/qwen3coder_tool_parser.py"

# ─────────────────────────────────────────────────────────────────────
# transformers_utils/ — speculator config helpers
# ─────────────────────────────────────────────────────────────────────
SPECULATORS_ALGOS = "transformers_utils/configs/speculators/algos.py"

# ─────────────────────────────────────────────────────────────────────
# v1/attention/backends/ — attention backend implementations
# ─────────────────────────────────────────────────────────────────────
FLASH_ATTN_BACKEND = "v1/attention/backends/flash_attn.py"
FLASHINFER_BACKEND = "v1/attention/backends/flashinfer.py"
GDN_ATTN_BACKEND = "v1/attention/backends/gdn_attn.py"
TURBOQUANT_ATTN = "v1/attention/backends/turboquant_attn.py"

# ─────────────────────────────────────────────────────────────────────
# v1/attention/ops/ — attention-related Triton ops
# ─────────────────────────────────────────────────────────────────────
TRITON_MERGE_ATTN_STATES = "v1/attention/ops/triton_merge_attn_states.py"
TRITON_TURBOQUANT_DECODE = "v1/attention/ops/triton_turboquant_decode.py"
TRITON_TURBOQUANT_STORE = "v1/attention/ops/triton_turboquant_store.py"

# ─────────────────────────────────────────────────────────────────────
# v1/core/ — core scheduling + KV cache management
# ─────────────────────────────────────────────────────────────────────
KV_CACHE_MANAGER = "v1/core/kv_cache_manager.py"
KV_CACHE_UTILS = "v1/core/kv_cache_utils.py"
ASYNC_SCHEDULER = "v1/core/sched/async_scheduler.py"
SCHEDULER = "v1/core/sched/scheduler.py"
SINGLE_TYPE_KV_CACHE_MANAGER = "v1/core/single_type_kv_cache_manager.py"

# ─────────────────────────────────────────────────────────────────────
# v1/engine/ — engine core
# ─────────────────────────────────────────────────────────────────────
ENGINE_CORE = "v1/engine/core.py"

# ─────────────────────────────────────────────────────────────────────
# v1/request.py — Request object definition
# ─────────────────────────────────────────────────────────────────────
REQUEST = "v1/request.py"

# ─────────────────────────────────────────────────────────────────────
# v1/sample/ — token sampling (rejection sampling for spec-decode)
# ─────────────────────────────────────────────────────────────────────
REJECTION_SAMPLER = "v1/sample/rejection_sampler.py"

# ─────────────────────────────────────────────────────────────────────
# v1/spec_decode/ — speculative decoding (ngram, MTP, EAGLE, DFlash)
# ─────────────────────────────────────────────────────────────────────
DFLASH = "v1/spec_decode/dflash.py"
LLM_BASE_PROPOSER = "v1/spec_decode/llm_base_proposer.py"
NGRAM_PROPOSER = "v1/spec_decode/ngram_proposer.py"

# ─────────────────────────────────────────────────────────────────────
# v1/structured_output/ — grammar-guided decoding
# ─────────────────────────────────────────────────────────────────────
STRUCTURED_OUTPUT_INIT = "v1/structured_output/__init__.py"

# ─────────────────────────────────────────────────────────────────────
# v1/worker/ — per-GPU worker, model runner, input batches
# ─────────────────────────────────────────────────────────────────────
PROMPT_LOGPROB = "v1/worker/gpu/sample/prompt_logprob.py"
GPU_INPUT_BATCH = "v1/worker/gpu_input_batch.py"
GPU_MODEL_RUNNER = "v1/worker/gpu_model_runner.py"
GPU_WORKER = "v1/worker/gpu_worker.py"
WORKER_MAMBA_UTILS = "v1/worker/mamba_utils.py"
WORKSPACE = "v1/worker/workspace.py"


# ─────────────────────────────────────────────────────────────────────
# Public API — programmatic enumeration
# ─────────────────────────────────────────────────────────────────────
def all_engine_targets() -> dict[str, str]:
    """Enumerate all engine target constants as `{name: relative_path}` dict.

    Used by:
      - `dispatcher/audit.py` boot-time validation (every target must
        exist in the installed vllm pin).
      - `cli/list_targets` for `sndr list-targets` output.
      - Stage 2+ tests verifying we have full coverage.

    Returns 63 entries (validated at Stage 2).
    """
    import sys
    mod = sys.modules[__name__]
    return {
        name: getattr(mod, name)
        for name in sorted(dir(mod))
        if name.isupper() and isinstance(getattr(mod, name), str)
    }


__all__ = sorted(name for name in dir() if name.isupper() and not name.startswith("_"))
__all__.append("all_engine_targets")
