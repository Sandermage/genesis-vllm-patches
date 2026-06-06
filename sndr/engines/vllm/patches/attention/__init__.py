# SPDX-License-Identifier: Apache-2.0
"""SNDR Core patches — `attention` family group.

Sub-families: gdn (Mamba/GDN), turboquant (TQ backend),
flash (FlashAttention varlen).
"""

from __future__ import annotations

__all__ = [
    "gdn",
    "turboquant",
    "flash",
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
