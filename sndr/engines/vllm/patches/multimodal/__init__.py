# SPDX-License-Identifier: Apache-2.0
"""SNDR Core patches — `multimodal` family.

Engine subsystem: multimodal-aware paths (VL/vision encoder skipping for
text-only inputs, multimodal token routing, etc.).

Patches in this family:
  - pn62_text_only_vit_skip — Genesis-original. When the running model
    is multimodal (qwen3-VL, etc.) but the request contains zero image
    tokens, skip the ViT forward pass + vision-projection materialize.
    Saves ~150-300ms / request when 90%+ of traffic is text-only.

Future patches expected here: vision token batching, mm cache eviction,
audio-modal skipping (when whisper-style models added).
"""

from __future__ import annotations

__all__ = [
    "pn62_text_only_vit_skip",
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
