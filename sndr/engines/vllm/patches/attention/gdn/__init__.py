# SPDX-License-Identifier: Apache-2.0
"""SNDR Core patches — `attention.gdn` family.

GDN/Mamba attention (mamba_linear, FLA chunked ops)

Stage 6 (2026-05-07): 17 patches reorganized here from
the legacy `vllm/_genesis/wiring/<old_cat>/` layout. Old paths remain
as back-compat shims forwarding to this canonical home.
"""

from __future__ import annotations

__all__ = [
    "p103_fla_cliff2_chunked",
    "p28_gdn_core_attn",
    "p39a_fla_kkt_buffer",
    "p46_gdn_gating_buffers",
    "p60_gdn_ngram_state_recovery",
    "p60b_gdn_ngram_triton_kernel",
    # "p63_mtp_gdn_state_recovery",  # moved to _retired/ 2026-05-14
    "p7_gdn_dual_stream",
    # "p7b_gdn_dual_stream_customop",  # moved to _archive/ 2026-06-11 (retired, preflight triage par.3)
    "pn11_gdn_a_b_contiguous",
    # PN29 + PN298 consolidated into one module 2026-06-19 (both patch the
    # same engine file model_executor/layers/fla/ops/chunk_o.py at disjoint
    # regions). Replaces pn29_gdn_chunk_o_scale_fold + pn298_fla_chunk_o_arch_warps.
    "pn29_pn298_chunk_o_consolidated",
    "pn30_ds_layout_spec_decode_align",
    "pn32_gdn_chunked_prefill",
    "pn50_gdn_fused_proj",
    # "pn54_gdn_contiguous_dedup",  # moved to _archive/ 2026-06-11 (retired, preflight triage par.3)
    "pn59_streaming_gdn",
    "pn79_inplace_ssm_state",
    # "pn108_fused_recurrent_prefill",  # moved to _retired/ 2026-05-14
    "pn204_dual_stream_inproj",
    "pn365_gdn_qkvz_ba_fuse_gemm",
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
