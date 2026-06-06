# SPDX-License-Identifier: Apache-2.0
"""SNDR Core patches — `serving` family.

OpenAI-API serving layer (vllm/entrypoints/openai/)

Stage 6 (2026-05-07): 4 patches reorganized here from
the legacy `vllm/_genesis/wiring/<old_cat>/` layout. Old paths remain
as back-compat shims forwarding to this canonical home.
"""

from __future__ import annotations

__all__ = [
    "p107_mtp_truncation_detector",
    "p62_structured_output_spec_decode_timing",
    "p68_69_long_ctx_tool_adherence",
    "pn70_tool_schema_subset_filter",
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
