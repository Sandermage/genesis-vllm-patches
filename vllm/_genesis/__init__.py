# SPDX-License-Identifier: Apache-2.0
"""Genesis vLLM patches — modular package.

This is the v7.0 architecture replacing monolithic patch_genesis_unified.py.
All patches migrate to modular kernel-level professional replacements под vendor-safe
defensive guards.

Mission: МЫ ЧИНИМ, НЕ ЛОМАЕМ (we fix, we don't break).
Each kernel works on NVIDIA CUDA / AMD ROCm / Intel XPU / CPU with graceful skip.

Sub-packages:
  guards    — canonical vendor/chip/model detection helpers
  prealloc  — safe pre-allocation framework (graph-safe, profiler-visible)
  kernels   — professional drop-in replacements for upstream-weak code paths
  patches   — thin monkey-patch bridges to upstream (legacy overlay)
  tests     — TDD-first pytest suite

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
Project: github.com/Sandermage/genesis-vllm-patches
Version: 7.0.0-dev
"""

__version__ = "7.0.0-dev"
__author__ = "Sandermage(Sander)-Barzov Aleksandr"
__location__ = "Ukraine, Odessa"
__project__ = "https://github.com/Sandermage/genesis-vllm-patches"

# Public API — what gets imported from vllm._genesis
from vllm._genesis import guards
from vllm._genesis import prealloc

__all__ = [
    "guards",
    "prealloc",
    "__version__",
    "__author__",
]
