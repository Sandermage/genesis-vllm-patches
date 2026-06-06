# SPDX-License-Identifier: Apache-2.0
"""SNDR Core patches — `reasoning` family.

reasoning model output parsers (vllm/reasoning/)

Stage 6 (2026-05-07): 8 patches reorganized here from
the legacy `vllm/_genesis/wiring/<old_cat>/` layout. Old paths remain
as back-compat shims forwarding to this canonical home.
"""

from __future__ import annotations

__all__ = [
    "p12_tool_call_reasoning",
    "p27_reasoning_before_think",
    "p59_qwen3_reasoning_tool_call_recovery",
    # "p61_qwen3_multi_tool_first_occurrence",  # moved to _retired/ 2026-05-14
    "p61b_qwen3_streaming_overlap_guard",
    "pn51_qwen3_streaming_thinking_disabled",  # reactivated 2026-05-15 after retired-audit gap confirm
    "pn58_spec_reasoning_boundary",
    "pn66_multiturn_think_leak",
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
