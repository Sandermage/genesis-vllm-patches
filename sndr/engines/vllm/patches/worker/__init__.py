# SPDX-License-Identifier: Apache-2.0
"""SNDR Core patches — `worker` family.

per-GPU worker (gpu_model_runner, input_batch, workspace)

Stage 6 (2026-05-07): 8 patches reorganized here from
the legacy `vllm/_genesis/wiring/<old_cat>/` layout. Old paths remain
as back-compat shims forwarding to this canonical home.
"""

from __future__ import annotations

__all__ = [
    "p72_profile_run_cap",
    "p79b_async_proposer_sync",
    "pn24_dflash_aux_layer_indexing",
    "pn33_spec_decode_warmup_k",
    "pn35_inputs_embeds_optional",
    # "pn52_prompt_logprobs_eviction",  # moved to _retired/ 2026-05-14
    "pn55_wake_up_hybrid_kv",
    "pn67_thinking_budget_inverted_bool",
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
