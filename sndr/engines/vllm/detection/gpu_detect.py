# SPDX-License-Identifier: Apache-2.0
"""SNDR Core detection — GPU generation + compute capability detection.

Facade over the platform-detection helpers currently living in
`vllm/_genesis/guards.py`. This module groups them under a coherent
name + makes them discoverable from the canonical SNDR Core path:

    from sndr.engines.vllm.detection.gpu_detect import (
        is_ampere_any, is_ada_lovelace, is_blackwell,
        get_compute_capability, get_gpu_name,
    )

Used by:
  - cli/install.py (Stage 11): hardware match → preset selection.
  - dispatcher/decision.py: `applies_to.gpu_generation` gating.
  - bundle orchestrators (Stage 7): gate engine-tier kernels by GPU.

Migration history:
  - Original location: vllm/_genesis/guards.py (Stage 0).
  - Stage 4 (CURRENT): facade module created here. Impl stays in
    guards.py.
  - Stage 12 (PLANNED): impl may move once monkey-patch contracts
    are migrated.
"""
from sndr.engines.vllm.detection.guards import (  # noqa: F401
    get_compute_capability,
    is_ada_lovelace,
    is_amd_rocm,
    is_ampere_any,
    is_ampere_consumer,
    is_ampere_datacenter,
    is_blackwell,
    is_blackwell_consumer,
    is_blackwell_datacenter,
    is_cpu_only,
    is_cuda_alike,
    is_hopper,
    is_intel_xpu,
    is_nvidia_cuda,
    is_sm_at_least,
    is_sm_exactly,
)


def get_gpu_generation() -> str:
    """Return human-readable GPU generation tag.

    Returns one of: "ampere_consumer", "ampere_datacenter",
    "ada_lovelace", "hopper", "blackwell_consumer",
    "blackwell_datacenter", "amd_rocm", "intel_xpu", "cpu_only",
    "unknown".
    """
    if is_amd_rocm():
        return "amd_rocm"
    if is_intel_xpu():
        return "intel_xpu"
    if is_cpu_only():
        return "cpu_only"
    if is_blackwell_consumer():
        return "blackwell_consumer"
    if is_blackwell_datacenter():
        return "blackwell_datacenter"
    if is_hopper():
        return "hopper"
    if is_ada_lovelace():
        return "ada_lovelace"
    if is_ampere_datacenter():
        return "ampere_datacenter"
    if is_ampere_consumer():
        return "ampere_consumer"
    return "unknown"


def get_gpu_name() -> str:
    """Best-effort GPU model name (e.g. "RTX A5000", "RTX 4090")."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "unknown"


def get_gpu_count() -> int:
    """Number of CUDA-visible GPUs. 0 if no CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass
    return 0


__all__ = [
    "get_compute_capability",
    "get_gpu_count",
    "get_gpu_generation",
    "get_gpu_name",
    "is_ada_lovelace",
    "is_amd_rocm",
    "is_ampere_any",
    "is_ampere_consumer",
    "is_ampere_datacenter",
    "is_blackwell",
    "is_blackwell_consumer",
    "is_blackwell_datacenter",
    "is_cpu_only",
    "is_cuda_alike",
    "is_hopper",
    "is_intel_xpu",
    "is_nvidia_cuda",
    "is_sm_at_least",
    "is_sm_exactly",
]
