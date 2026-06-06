# SPDX-License-Identifier: Apache-2.0
"""SNDR Core patches — `memory` family.

memory pool / buffer / VRAM management patches

Stage 6 (2026-05-07): 5 patches reorganized here from
the legacy `vllm/_genesis/wiring/<old_cat>/` layout. Old paths remain
as back-compat shims forwarding to this canonical home.
"""

from __future__ import annotations

__all__ = [
    "p15b_fa_varlen_clamp",
    "p38b_compile_safe_hook",
    "p5b_page_size_pad_smaller",
    # "pn19_scoped_max_split",  # moved to _retired/ 2026-05-14
    # "pn78_post_warmup_cache_release",  # moved to _retired/ 2026-05-14
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
