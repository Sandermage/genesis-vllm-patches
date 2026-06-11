# SPDX-License-Identifier: Apache-2.0
"""SNDR Core patches — `scheduler` family.

request scheduler (v1/core/sched/)

Stage 6 (2026-05-07): 8 patches reorganized here from
the legacy `vllm/_genesis/wiring/<old_cat>/` layout. Old paths remain
as back-compat shims forwarding to this canonical home.
"""

from __future__ import annotations

__all__ = [
    "p34_mamba_deadlock_guard",
    # "p4_tq_hybrid",  # moved to _archive/ 2026-06-11 (retired, preflight triage par.3)
    "p58_async_scheduler_placeholder_fix",
    "p74_chunk_clamp",
    "p79c_stale_spec_token_cleanup",
    "p79d_preempt_async_discard",
    # "p84_hash_block_size_override",  # moved to _archive/ 2026-06-11 (retired, preflight triage par.3)
    # "p8_kv_hybrid_reporting",  # moved to _retired/ 2026-05-14
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
