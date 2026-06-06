# SPDX-License-Identifier: Apache-2.0
"""SNDR Core patches — `spec_decode` family.

speculative decoding (ngram, MTP, EAGLE, DFlash, async cleanup)

Stage 6 (2026-05-07): 15 patches reorganized here from
the legacy `vllm/_genesis/wiring/<old_cat>/` layout. Old paths remain
as back-compat shims forwarding to this canonical home.
"""

from __future__ import annotations

__all__ = [
    "p70_auto_strict_ngram",
    "p71_block_verify",
    "p75_suffix_decoding_enable",
    "p77_adaptive_ngram_k",
    "p82_sglang_acceptance_threshold",
    "p86_ngram_batch_propose_linear",
    # "p94_spec_decode_zero_alloc",  # moved to _retired/ 2026-05-14
    "pn21_dflash_swa_support",
    "pn22_local_argmax_tp",
    "pn23_dflash_combine_hidden_dtype",
    "pn38_dflash_quant_drafter",
    "pn40_dflash_omnibus",
    "pn40_workload_classifier_hook",
    "pn72_frequency_ngram_drafter",
    # "pn9_independent_drafter_attn_backend",  # moved to _retired/ 2026-05-14
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
