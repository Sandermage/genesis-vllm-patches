# SPDX-License-Identifier: Apache-2.0
"""Genesis kernel-level drop-in replacements for vLLM weak spots.

Each kernel module provides a professional, platform-aware implementation
of a vLLM code path that was identified as broken, suboptimal, or crude.

Design goals per kernel:
  - Works on NVIDIA CUDA / AMD ROCm / Intel XPU / CPU (graceful skip).
  - Matches or exceeds upstream behavior in all metrics.
  - TDD-first: test suite before implementation.
  - Upstream-ready: code structure suitable for submission as vLLM PR.

Modules:
  router_softmax   — fp32-upcast MoE router softmax (Patch 31, universal)
  dequant_buffer   — TurboQuant shared pre-allocation manager (Patch 22)
  gdn_dual_stream  — Platform-aware dual-stream dispatcher (Patch 7)
  marlin_tuning    — Per-SM Marlin kernel auto-tuner (Patch 17/18)
  fp8_dispatcher   — Ampere/Ada/Hopper FP8 path selector (Patch 1/2)

Lazy import contract (audit A-02 fix 2026-05-06):
  This package previously eagerly imported `router_softmax` (which pulls
  in torch) at package-load time. That broke the offline/torch-optional
  contract for sibling helpers like `ngram_frequency_filter` (numpy-only)
  and the wiring `legacy/patch_5b_*` modules that import non-torch
  kernel sub-modules — Python evaluates `__init__.py` first regardless
  of which sub-module is being imported. The fix below uses PEP 562
  module-level `__getattr__` to resolve attributes lazily, so
  `from sndr.engines.vllm.kernels_legacy.ngram_frequency_filter import ...`
  no longer transitively imports torch.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

__all__ = [
    "router_softmax",
]


def __getattr__(name: str):
    """PEP 562 lazy attribute resolution.

    Avoids eager-importing torch-heavy modules at package load. Sub-modules
    that don't need torch can be imported by their full dotted path without
    triggering this __getattr__, since Python's import system bypasses
    package-level __getattr__ when resolving fully-qualified sub-module
    imports.
    """
    if name == "router_softmax":
        from sndr.engines.vllm.kernels_legacy.router_softmax import router_softmax as _fn
        return _fn
    raise AttributeError(f"module 'sndr.engines.vllm.kernels_legacy' has no attribute {name!r}")
