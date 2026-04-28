# SPDX-License-Identifier: Apache-2.0
"""Genesis defensive guards — canonical vendor/chip/model/dependency detection.

Philosophy: МЫ ЧИНИМ, НЕ ЛОМАЕМ.
Every helper is fail-safe: returns a safe default (False/None) on any exception.
If detection cannot complete, we SKIP the patch — never crash the engine.

All detection patterns mirror upstream vLLM canonical sources:
  - vllm/platforms/interface.py   (Platform predicates, DeviceCapability)
  - vllm/platforms/cuda.py        (NvmlCudaPlatform.get_device_capability)
  - vllm/platforms/rocm.py        (_GCN_ARCH parsing)

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Optional

log = logging.getLogger("genesis.guards")


# ─── torch._dynamo.disable shim ────────────────────────────────────────────
# Genesis guards are eager-only diagnostic helpers (vendor / SM / version
# detection). They are commonly called from kernel paths that vLLM compiles
# with torch.compile / torch.dynamo (Marlin apply_weights, FP8 scaled MM,
# CUDA-graph capture, etc.).
#
# torch.dynamo IGNORES `@functools.lru_cache` / `@functools.cache` wrappers
# and traces the underlying function instead. When that traced body calls
# `current_platform.get_device_capability()` (which itself touches torch
# / pynvml internals dynamo can't trace), the engine init crashes with
# RuntimeError. This was empirically observed 2026-04-28 with
# GENESIS_FORCE_MARLIN_W8A16=1 on Qwen3.6-27B-INT8-AutoRound (Marlin path).
#
# Fix: wrap each guard with `@torch._dynamo.disable` BEFORE the
# `@functools.cache`. Dynamo then treats the call as an opaque eager
# function — no tracing into platform internals, cache wrapper still works
# at runtime. No effect on patches that aren't dynamo-traced.
try:
    import torch._dynamo as _torch_dynamo
    _dynamo_disable = _torch_dynamo.disable
except Exception:  # noqa: BLE001 — torch may be unavailable in pure tests
    def _dynamo_disable(fn=None, **_kwargs):  # type: ignore[no-redef]
        """No-op fallback when torch._dynamo is not importable."""
        if fn is None:
            return lambda f: f
        return fn


# ═══════════════════════════════════════════════════════════════════════════
#                          VENDOR / PLATFORM IDENTITY
# ═══════════════════════════════════════════════════════════════════════════

@_dynamo_disable
@functools.cache
def _current_platform() -> Optional[Any]:
    """Lazy import of vllm.platforms.current_platform.

    Returns None if vllm is not importable — allows guards to work during
    early initialization or in environments without vllm installed.
    """
    try:
        from vllm.platforms import current_platform
        return current_platform
    except Exception as e:
        log.debug("current_platform unavailable: %s", e)
        return None


@_dynamo_disable
@functools.cache
def is_nvidia_cuda() -> bool:
    """True ONLY on NVIDIA CUDA (NOT ROCm).

    Canonical source: vllm/platforms/interface.py:157
        def is_cuda(self) -> bool: return self._enum == PlatformEnum.CUDA

    CRITICAL: Use this (not is_cuda_alike) for NVIDIA-specific patches such as
    Marlin, CUDA streams, CUDA graph capture. ROCm's PyTorch build exposes
    torch.cuda namespace but with different semantics that can crash our patches.
    """
    p = _current_platform()
    try:
        return bool(p is not None and p.is_cuda())
    except Exception:
        return False


@_dynamo_disable
@functools.cache
def is_amd_rocm() -> bool:
    """True on AMD ROCm.

    Canonical source: vllm/platforms/interface.py:160
    """
    p = _current_platform()
    try:
        return bool(p is not None and p.is_rocm())
    except Exception:
        return False


@_dynamo_disable
@functools.cache
def is_intel_xpu() -> bool:
    """True on Intel XPU (Arc, Max Series).

    Canonical source: vllm/platforms/interface.py:166
    """
    p = _current_platform()
    try:
        return bool(p is not None and p.is_xpu())
    except Exception:
        return False


@_dynamo_disable
@functools.cache
def is_cpu_only() -> bool:
    """True on CPU-only build (no GPU accelerator).

    Canonical source: vllm/platforms/interface.py:169
    """
    p = _current_platform()
    try:
        return bool(p is not None and p.is_cpu())
    except Exception:
        return False


@_dynamo_disable
@functools.cache
def is_cuda_alike() -> bool:
    """CUDA OR ROCm (shares torch.cuda namespace).

    Canonical source: vllm/platforms/interface.py:184

    ⚠️ TRAP: Do NOT use for NVIDIA-specific patches. ROCm exposes torch.cuda
    Stream/Event but with weaker ordering guarantees. Use is_nvidia_cuda()
    strict check for Marlin/CUDA-graph/aux-stream patches.

    Safe to use for operations that work identically on both: torch tensor ops,
    numerical operations, generic optimizations.
    """
    p = _current_platform()
    try:
        return bool(p is not None and p.is_cuda_alike())
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
#                      NVIDIA COMPUTE CAPABILITY
# ═══════════════════════════════════════════════════════════════════════════

@_dynamo_disable
@functools.cache
def get_compute_capability() -> Optional[tuple[int, int]]:
    """Return (major, minor) compute capability for NVIDIA CUDA; None otherwise.

    Canonical source: vllm/platforms/cuda.py:592-599 (NvmlCudaPlatform)
    """
    if not is_nvidia_cuda():
        return None
    p = _current_platform()
    try:
        cc = p.get_device_capability()
        if cc is None:
            return None
        return (cc.major, cc.minor)
    except Exception as e:
        log.debug("get_device_capability failed: %s", e)
        return None


@_dynamo_disable
def is_sm_at_least(major: int, minor: int = 0) -> bool:
    """True if SM >= (major, minor). Mirrors vLLM's has_device_capability."""
    cc = get_compute_capability()
    if cc is None:
        return False
    return cc >= (major, minor)


@_dynamo_disable
def is_sm_exactly(major: int, minor: int) -> bool:
    """True if SM is exactly (major, minor)."""
    return get_compute_capability() == (major, minor)


def is_ampere_datacenter() -> bool:
    """NVIDIA A100 — SM 8.0."""
    return is_sm_exactly(8, 0)


def is_ampere_consumer() -> bool:
    """NVIDIA A5000 / A6000 / RTX 3090 — SM 8.6.

    Genesis prod baseline.
    """
    return is_sm_exactly(8, 6)


def is_ampere_any() -> bool:
    """Any Ampere (SM 8.x except Ada 8.9)."""
    cc = get_compute_capability()
    return cc is not None and cc[0] == 8 and cc[1] < 9


def is_ada_lovelace() -> bool:
    """NVIDIA RTX 4090 / L40 / RTX 6000 Ada — SM 8.9."""
    return is_sm_exactly(8, 9)


def is_hopper() -> bool:
    """NVIDIA H100 / H200 — SM 9.0."""
    return is_sm_exactly(9, 0)


def is_blackwell() -> bool:
    """NVIDIA B100 / B200 / RTX 5090 / RTX PRO 6000 — SM 10.x.

    Our future target after R6000 purchase.
    """
    cc = get_compute_capability()
    return cc is not None and cc[0] == 10


def has_native_fp8() -> bool:
    """True if GPU has native FP8 tensor cores (SM >= 8.9).

    Includes Ada Lovelace (8.9), Hopper (9.0), Blackwell (10.0).
    Ampere (8.6) does NOT have native FP8 — uses emulation.
    """
    return is_sm_at_least(8, 9)


def pdl_support_expected() -> bool:
    """True if platform is expected to support Programmatic Dependent Launch.

    PDL (Programmatic Dependent Launch) is a Hopper+ / Blackwell feature. On
    older GPUs (Ampere / Ada), enabling PDL-related env vars has no effect
    at best; at worst it can trigger vLLM issue #40742 (Inductor autotune
    calls torch.cuda.synchronize() inside CUDA graph capture → illegal
    cuda operation → server crash at startup).

    Returns:
      True on SM >= 9.0 (Hopper, Blackwell, future).
      False on SM < 9.0 (Ampere consumer/datacenter, Ada Lovelace, pre-Ampere).
      False on non-NVIDIA.
    """
    return is_sm_at_least(9, 0)


def detect_pdl_env_misconfig() -> list[str]:
    """Detect user-set PDL env vars that aren't safe on this GPU.

    Reference: vLLM issue #40742 (2026-04-23) — CUDA graph capture crash when
    `TRTLLM_ENABLE_PDL=1` / `TORCHINDUCTOR_ENABLE_PDL=1` is set on GPUs where
    PDL is not fully supported, because Inductor autotune inserts a
    `torch.cuda.synchronize()` inside an active graph capture.

    Returns:
      List of env-var names that are set to truthy values but shouldn't be
      on this platform. Empty list means safe.
    """
    if pdl_support_expected():
        return []

    import os as _os
    misconfigured: list[str] = []
    for var in (
        "TRTLLM_ENABLE_PDL",
        "TORCHINDUCTOR_ENABLE_PDL",
    ):
        val = _os.environ.get(var, "").strip().lower()
        if val in ("1", "true", "yes", "on"):
            misconfigured.append(var)
    return misconfigured


# ═══════════════════════════════════════════════════════════════════════════
#                       AMD ROCm ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════

@_dynamo_disable
@functools.cache
def _gcn_arch() -> str:
    """GCN architecture string (e.g. 'gfx942'). Empty string if not ROCm.

    Canonical source: vllm/platforms/rocm.py:187
    """
    if not is_amd_rocm():
        return ""
    try:
        from vllm.platforms import rocm as _rocm
        return getattr(_rocm, "_GCN_ARCH", "") or ""
    except Exception:
        return ""


def is_rocm_cdna2() -> bool:
    """AMD MI210 / MI250 — gfx90a (CDNA2 datacenter)."""
    return "gfx90a" in _gcn_arch()


def is_rocm_cdna3() -> bool:
    """AMD MI300X / MI325X — gfx942 / gfx950 (CDNA3 datacenter)."""
    arch = _gcn_arch()
    return "gfx942" in arch or "gfx950" in arch


def is_rocm_rdna() -> bool:
    """AMD Radeon RDNA3/4 — gfx11xx / gfx12xx (consumer)."""
    arch = _gcn_arch()
    return "gfx11" in arch or "gfx12" in arch


# ═══════════════════════════════════════════════════════════════════════════
#                   EXTERNAL DEPENDENCY VERSIONS (NEW v7.0)
# ═══════════════════════════════════════════════════════════════════════════

@_dynamo_disable
@functools.cache
def get_torch_version() -> Optional[tuple[int, int]]:
    """Returns (major, minor) torch version, or None on failure."""
    try:
        import torch
        parts = torch.__version__.split(".")
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return None


def is_torch_211_plus() -> bool:
    """True if torch >= 2.11 (required for vLLM v0.20.0+)."""
    v = get_torch_version()
    return v is not None and v >= (2, 11)


def is_torch_212_plus() -> bool:
    """True if torch >= 2.12 (forthcoming late 2026)."""
    v = get_torch_version()
    return v is not None and v >= (2, 12)


@_dynamo_disable
@functools.cache
def get_transformers_version() -> Optional[tuple[int, int, int]]:
    """Returns (major, minor, patch) transformers version, or None on failure."""
    try:
        import transformers
        parts = transformers.__version__.split(".")[:3]
        # Handle versions like "5.5.0rc1" by stripping non-digit suffix
        return tuple(int(''.join(c for c in p if c.isdigit())) for p in parts)
    except Exception:
        return None


def is_transformers_v5_plus() -> bool:
    """True if transformers >= 5.0.0 (required for vLLM v0.19.1+)."""
    v = get_transformers_version()
    return v is not None and v[0] >= 5


def is_transformers_v55_plus() -> bool:
    """True if transformers >= 5.5.0 (required for Gemma 4 support)."""
    v = get_transformers_version()
    return v is not None and v >= (5, 5, 0)


@_dynamo_disable
@functools.cache
def get_vllm_version_tuple() -> Optional[tuple[int, ...]]:
    """Returns (major, minor, patch) vllm version tuple, or None on failure.

    Example: vllm 0.20.0 -> (0, 20, 0)
    """
    try:
        import vllm
        parts = vllm.__version__.split(".")[:3]
        # Handle versions like "0.19.2rc1.dev8" by taking only leading digits
        result = []
        for p in parts:
            digits = ''.join(c for c in p.split('rc')[0].split('+')[0] if c.isdigit())
            result.append(int(digits) if digits else 0)
        return tuple(result)
    except Exception:
        return None


def is_vllm_020_plus() -> bool:
    """True if vllm >= 0.20.0."""
    v = get_vllm_version_tuple()
    return v is not None and v >= (0, 20, 0)


@_dynamo_disable
@functools.cache
def get_flash_attn_major_version() -> Optional[int]:
    """Try to detect FlashAttention version (FA2 / FA3 / FA4).

    Returns major version int or None if not available or detection failed.

    Canonical source: vllm/v1/attention/backends/fa_utils.py:get_flash_attn_version
    """
    try:
        # Try vllm's own helper first
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
        v = get_flash_attn_version(head_size=128)
        return int(v) if v else None
    except Exception:
        pass
    try:
        # Fallback: check flash_attn module directly
        import flash_attn
        return int(flash_attn.__version__.split(".")[0])
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
#                      MODEL ARCHITECTURE DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def is_model_arch(model_config: Any, arch_name: str) -> bool:
    """Case-insensitive substring match against model_config.architectures.

    Canonical source: vllm/model_executor/models/registry.py resolution logic.

    Examples:
      is_model_arch(cfg, "Qwen3")       # True for Qwen3.5 / Qwen3.6 / Qwen3-Next
      is_model_arch(cfg, "DeepSeekV3")  # True for DeepSeek V3 family
      is_model_arch(cfg, "Llama")       # True for any Llama variant
    """
    if model_config is None:
        return False
    try:
        archs = getattr(model_config, "architectures", None) or []
        needle = arch_name.lower()
        return any(needle in (a or "").lower() for a in archs)
    except Exception:
        return False


def is_qwen3_family(model_config: Any) -> bool:
    """True for Qwen3.5 / Qwen3.6 / Qwen3-Next / Qwen3-Coder family."""
    return is_model_arch(model_config, "Qwen3")


def is_deepseek_v3(model_config: Any) -> bool:
    """True for DeepSeek V3 family (uses MLA attention, distinct from Qwen)."""
    return is_model_arch(model_config, "DeepseekV3") or is_model_arch(model_config, "DeepSeek-V3")


def is_llama_family(model_config: Any) -> bool:
    """True for Llama 3.x / 4.x family."""
    return is_model_arch(model_config, "Llama")


def is_gemma_family(model_config: Any) -> bool:
    """True for Gemma family (uses sliding-window hybrid)."""
    return is_model_arch(model_config, "Gemma")


def is_mixtral_family(model_config: Any) -> bool:
    """True for Mixtral MoE family (different router than Qwen3)."""
    return is_model_arch(model_config, "Mixtral")


# ═══════════════════════════════════════════════════════════════════════════
#                     BACKEND / KERNEL DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def has_turboquant_support(cache_dtype: Optional[str]) -> bool:
    """True if TurboQuant path is active (cache_dtype starts with 'turboquant_').

    Canonical source: vllm/platforms/cuda.py:134 — routing key in CacheConfig.
    """
    return bool(cache_dtype and cache_dtype.startswith("turboquant_"))


def is_marlin_selected(fused_moe_layer: Any) -> bool:
    """Best-effort introspection: is Marlin kernel selected for this MoE layer?

    Returns False (safe default) if detection fails — we prefer to skip
    a Marlin-specific patch than apply it wrong.
    """
    try:
        kernel = getattr(fused_moe_layer, "kernel", None)
        if kernel is None:
            # Try alternate attribute paths
            kernel = getattr(fused_moe_layer, "quant_method", None)
        name = type(kernel).__name__ if kernel else ""
        return "marlin" in name.lower()
    except Exception:
        return False


def is_flash_attn_backend(attn_backend: Any) -> bool:
    """True if FlashAttention backend selected."""
    try:
        name = getattr(attn_backend, "name", "") or type(attn_backend).__name__
        return "flash" in name.lower() and "attn" in name.lower()
    except Exception:
        return False


def is_turboquant_backend(attn_backend: Any) -> bool:
    """True if TurboQuant backend selected."""
    try:
        name = getattr(attn_backend, "name", "") or type(attn_backend).__name__
        return "turboquant" in name.lower()
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
#                      FILE PATH RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════

@_dynamo_disable
@functools.cache
def vllm_install_root() -> Optional[str]:
    """Returns absolute path to installed vllm package.

    CRITICAL: Replaces hardcoded /usr/local/lib/python3.12/dist-packages/vllm/
    that was in earlier patch versions — those break on:
      - macOS dev environments
      - venv installations
      - Python 3.13+ coming 2027
      - Docker slim/distroless images

    vllm.__file__ is the canonical universal way to locate the package.
    """
    try:
        import vllm
        import os
        return os.path.dirname(vllm.__file__)
    except Exception:
        return None


def resolve_vllm_file(relative_path: str) -> Optional[str]:
    """Returns absolute path to file within installed vllm, or None if missing.

    Example:
        resolve_vllm_file("v1/attention/backends/turboquant_attn.py")
        -> "/path/to/vllm/v1/attention/backends/turboquant_attn.py" or None
    """
    import os
    root = vllm_install_root()
    if root is None:
        return None
    full = os.path.join(root, relative_path)
    return full if os.path.exists(full) else None


# ═══════════════════════════════════════════════════════════════════════════
#                         SUMMARY / DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════

def platform_summary() -> dict[str, Any]:
    """Return full platform diagnostic dict for logging/debugging.

    Useful during patch application to log context:
        log.info("[Genesis] Platform: %s", json.dumps(platform_summary()))
    """
    return {
        "vendor": {
            "is_nvidia_cuda": is_nvidia_cuda(),
            "is_amd_rocm": is_amd_rocm(),
            "is_intel_xpu": is_intel_xpu(),
            "is_cpu_only": is_cpu_only(),
        },
        "nvidia": {
            "compute_capability": get_compute_capability(),
            "is_ampere_datacenter": is_ampere_datacenter(),
            "is_ampere_consumer": is_ampere_consumer(),
            "is_ada_lovelace": is_ada_lovelace(),
            "is_hopper": is_hopper(),
            "is_blackwell": is_blackwell(),
            "has_native_fp8": has_native_fp8(),
        },
        "amd": {
            "gcn_arch": _gcn_arch(),
            "is_cdna2": is_rocm_cdna2(),
            "is_cdna3": is_rocm_cdna3(),
            "is_rdna": is_rocm_rdna(),
        } if is_amd_rocm() else {},
        "versions": {
            "torch": get_torch_version(),
            "transformers": get_transformers_version(),
            "vllm": get_vllm_version_tuple(),
            "flash_attn_major": get_flash_attn_major_version(),
        },
        "paths": {
            "vllm_install_root": vllm_install_root(),
        },
    }
