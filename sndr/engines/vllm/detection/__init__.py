# SPDX-License-Identifier: Apache-2.0
"""vLLM-specific detection helpers (Layer 1).

These modules read vllm's runtime state (vllm.config, vllm.model_executor,
vllm.transformers_utils) and translate it into engine-agnostic shapes for
the dispatcher.

Modules:
  - **config_detect**: Read vllm's resolved configuration.
  - **model_detect**: Probe the loaded model's HF config + quant_method.
  - **model_profile**: Typed wrapper + 2D dispatch via should_apply_patch_for_model.
  - **guards**: Version checks, resolve_vllm_file, hardware predicates.
  - **driver_check**: NVIDIA driver capability probe.
  - **runtime_caveat**: Known-issue annotations for specific runtimes.
  - **gpu_detect**: Facade re-exporting GPU predicates from guards.

These modules are vllm-tied. For engine-agnostic detection (gpu_arch_profile,
perf_model), see ``sndr.detection``.

Migration: moved from vllm/sndr_core/detection/ in Phase 3 (2026-06-05).
"""
# Public symbols are imported lazily by callers; we do not eagerly
# re-export here to avoid forcing vllm imports at sndr.engines.vllm
# package load time.
