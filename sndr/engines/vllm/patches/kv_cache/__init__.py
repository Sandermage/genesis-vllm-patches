# SPDX-License-Identifier: Apache-2.0
"""SNDR Core patches — `kv_cache` family.

KV cache manager + block pool

Stage 6 (2026-05-07): 4 patches reorganized here from
the legacy `vllm/_genesis/wiring/<old_cat>/` layout. Old paths remain
as back-compat shims forwarding to this canonical home.
"""

from __future__ import annotations

__all__ = [
    "p14_block_table",
    "p5_page_size",
    "p83_mtp_keep_last_cached_block",
    "p85_hybrid_fine_shadow_prefix_cache",
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
