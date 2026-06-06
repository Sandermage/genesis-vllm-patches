# SPDX-License-Identifier: Apache-2.0
"""SNDR Core patches — `lora` family (LoRA adapter loading + tensorizer).

Engine subsystem: `vllm/lora/` and surrounding adapter-loading code.

Patches in this family:
  - pn80_lora_tensorizer_device — vllm#41845 backport (Sandermage). Fixes
    LoRA tensorizer crash when adapter is loaded onto a non-default GPU
    by passing `device` kwarg through the call stack. Ampere/Ada/Blackwell
    relevant — affects any multi-GPU deployment with LoRA adapters.

Future patches expected here: any LoRA / PEFT / adapter loading fixes.
"""

from __future__ import annotations

__all__ = [
    # "pn80_lora_tensorizer_device",  # moved to _retired/ 2026-05-14
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
