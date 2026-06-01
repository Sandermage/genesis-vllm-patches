# SPDX-License-Identifier: Apache-2.0
"""SNDR Core patches — engine-subsystem-first taxonomy.

Each subsystem corresponds to a vllm-engine subdirectory or coherent
concern. Patches grouped by PRIMARY engine target file (Sander Q1/Q6
decision 2026-05-07).

Family taxonomy
---------------

The PATCH_REGISTRY ``family`` field exposes 23 distinct values as of
2026-06-01:

  attention.flash, attention.gdn, attention.turboquant,
  compile_safety, gemma4, kernels, kv_cache, loader, lora, memory,
  middleware, moe, multimodal, observability, offload, quantization,
  reasoning, scheduler, serving, spec_decode, streaming, tool_parsing,
  worker

This module's ``__all__`` exposes the 20 importable subpackages under
``integrations/`` — ``attention`` is exposed as a single submodule
(its three sub-families ``flash`` / ``gdn`` / ``turboquant`` are
nested under ``integrations.attention.*``), ``gemma4`` is exposed at
``integrations.model_compat.gemma4`` (the dir under
``integrations/gemma4/`` holds only README + upstream-overlay
artefacts), and ``kernels`` lives at ``vllm.sndr_core.kernels``
(sibling to ``integrations/``). The remaining 20 entries below map
1:1 to family-name → import-name.
"""

from __future__ import annotations

__all__ = [
    "attention",
    "compile_safety",
    "kernels",
    "kv_cache",
    "loader",
    "lora",
    "memory",
    "middleware",
    "moe",
    "multimodal",
    "observability",
    "offload",
    "quantization",
    "reasoning",
    "scheduler",
    "serving",
    "spec_decode",
    "streaming",
    "tool_parsing",
    "worker",
]

def __getattr__(name: str):
    """Lazy submodule loader (P0-1 fix, audit 2026-05-08).

    Eager `from . import <patch>` cascaded torch imports → torch-less
    hosts (CI / Mac dev / preflight) couldn't import the patches
    package at all. Now patches load only on attribute access.
    """
    import importlib
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
