# SPDX-License-Identifier: Apache-2.0
"""sndr.detection — engine-agnostic hardware detection and modeling.

Layer 0 (no engine dependencies). Contains:

  - **gpu_arch_profile**: GenesisGPUArchProfile — typed snapshot of GPU
    architecture (SM, shared mem, L2, HBM bandwidth) plus predicates
    (is_ampere_consumer, is_hopper, has_fp8_native, treat_as_blackwell_for_ssm).
  - **perf_model**: roofline model and Triton config cost prediction.
  - **gpu_class_map**: lookup table of known GPU device names to
    architecture-specific metadata.

These modules MUST NOT import vLLM or any other engine. They consume only
torch and stdlib. Engine adapters consume these modules; they do not consume
engine adapters.

For engine-specific detection (config_detect, model_detect, model_profile,
guards), see ``sndr/engines/<engine>/detection/``.

Migration note: moved from vllm/sndr_core/detection/ in Phase 3 of the
sndr-platform refactor (2026-06-05).
"""
from sndr.detection.gpu_arch_profile import (  # noqa: F401
    GenesisGPUArchProfile,
    get_gpu_arch_profile,
    get_max_safe_num_warps,
    is_sm86,
    is_sm9_or_newer,
    prune_triton_autotune_configs,
)

# perf_model functions are imported lazily to avoid potential torch import
# overhead during simple detection lookups.

__all__ = [
    "GenesisGPUArchProfile",
    "get_gpu_arch_profile",
    "get_max_safe_num_warps",
    "is_sm86",
    "is_sm9_or_newer",
    "prune_triton_autotune_configs",
]
