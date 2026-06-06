# SPDX-License-Identifier: Apache-2.0
"""SNDR Core patches — `attention.turboquant` family.

TurboQuant attention backend + KV centroids

Stage 6 (2026-05-07): 20 patches reorganized here from
the legacy `vllm/_genesis/wiring/<old_cat>/` layout. Old paths remain
as back-compat shims forwarding to this canonical home.
"""

from __future__ import annotations

__all__ = [
    "p101_tq_continuation_slicing",
    "p22_tq_prealloc",
    "p26_prefill_output",
    "p38_tq_continuation_memory",
    "p3_tq_bf16_cast",
    "p40_tq_grouped_decode",
    "p44_tq_mixed_attn_out",
    "p65_turboquant_spec_cg_downgrade",
    "p67_tq_multi_query_kernel",
    "p67b_spec_verify_routing",
    "p67c_sparse_v",
    "p78_tolist_capture_guard",
    "p98_tq_workspace_revert",
    "p99_workspace_manager_memoize",
    "pn14_tq_decode_oob_clamp",
    "pn26_sparse_v_kernel",
    "pn26_tq_unified_perf",
    "pn31_fa_varlen_persistent_out",
    "pn34_workspace_lock_runtime_relax",
    "pn57_tq_centroids_disk_cache",
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
