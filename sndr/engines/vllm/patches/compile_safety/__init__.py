# SPDX-License-Identifier: Apache-2.0
"""SNDR Core patches — `compile_safety` family.

torch.compile / cudagraph capture-time guards

Stage 6 (2026-05-07): 4 patches reorganized here from
the legacy `vllm/_genesis/wiring/<old_cat>/` layout. Old paths remain
as back-compat shims forwarding to this canonical home.
"""

from __future__ import annotations

__all__ = [
    "p66_cudagraph_size_divisibility_filter",
    # "p6_tq_block_size_align",  # moved to _archive/ 2026-06-11 (retired, preflight triage par.3)
    "p95_marlin_tp_cudagraph_cap",
    # "pn13_cuda_graph_lambda_arity",  # moved to _retired/ 2026-05-14
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
