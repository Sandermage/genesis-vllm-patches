# SPDX-License-Identifier: Apache-2.0
"""SNDR Core patches — `kernels` family.

Triton/CUDA kernel-level patches (Marlin, FA, fp8)

Stage 6 (2026-05-07): 5 patches reorganized here from
the legacy `vllm/_genesis/wiring/<old_cat>/` layout. Old paths remain
as back-compat shims forwarding to this canonical home.
"""

from __future__ import annotations

__all__ = [
    "p36_tq_shared_decode_buffers",
    "p87_marlin_pad_sub_tile",
    "pn12_ffn_intermediate_pool",
    "pn25_silu_inductor_safe_pool",
    "pn28_merge_attn_states_nan_guard",
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
