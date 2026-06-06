# SPDX-License-Identifier: Apache-2.0
"""Retired Genesis patches — archived wirings, kept for audit trail.

A patch lands here when one of these holds:
 1. Upstream merged the same fix (within a known vllm_version_range).
 2. Hypothesis disproven empirically (retired_waiver=True with explanation).
 3. Duplicate of another active patch (superseded_by set).
 4. Deprecated mechanism (workaround replaced by root-cause fix).

Registry entries stay (with `lifecycle: "retired"`) so dispatcher logs and
audit gates can report drift. Modules here are imported via legacy hooks
(`_per_patch_dispatch.py`) and the `bundles/reasoning_qwen3.py` bundle to
preserve boot order; their `apply()` returns "skipped: retired" or is a
harmless no-op (anchor never matches in retire-eligible state).

Policy doc: ./README.md
"""
from __future__ import annotations

__all__ = [
    "p8_kv_hybrid_reporting",
    "p61_qwen3_multi_tool_first_occurrence",
    "p63_mtp_gdn_state_recovery",
    "p94_spec_decode_zero_alloc",
    "pn9_independent_drafter_attn_backend",
    "pn13_cuda_graph_lambda_arity",
    "pn19_scoped_max_split",
    "pn51_qwen3_streaming_thinking_disabled",
    "pn52_prompt_logprobs_eviction",
    "pn78_post_warmup_cache_release",
    "pn80_lora_tensorizer_device",
    "pn108_fused_recurrent_prefill",
]


def __getattr__(name: str):
    """Lazy submodule loader (audit P0-1 pattern, see other family __init__.py)."""
    if name in __all__:
        import importlib
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
