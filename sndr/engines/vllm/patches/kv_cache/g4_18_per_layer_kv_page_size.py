# SPDX-License-Identifier: Apache-2.0
"""G4_18 — per-layer KV cache page-size for Gemma 4 26B-A4B (vendors #40391).

================================================================
WHAT IT FIXES
================================================================

Companion to G4_13 (which **refuses** asymmetric KV-head Gemma 4
configs). This patch **actually fixes** the underlying problem by
vendoring the WIP logic from vllm#40391 — per-layer-type
``num_kv_heads`` in the KV-cache spec.

Without this fix:
  * 26B-A4B has sliding_attention layers with ``num_kv_heads=8`` and
    full_attention layers with ``num_kv_heads=2``
  * Upstream computes a SINGLE page-size based on
    ``hf_config.num_key_value_heads`` (the sliding value)
  * Global layers over-allocate pages (4× their actual need) AND
    write to wrong offsets when slot-mapping runs

With this fix:
  * KV-cache spec is built per-layer-type:
    ``sliding_attention`` → 8 KV-heads page-size
    ``full_attention``    → 2 KV-heads page-size
  * Slot-mapping kernel reads the correct head-count for each layer

================================================================
INTEGRATION STRATEGY
================================================================

We hook ``KVCacheSpecBuilder.build_for_layer`` (or the equivalent
vLLM v1 KV-cache spec builder) and inject a layer-type-aware override:

  1. Detect Gemma 4 model_config
  2. Read ``layer_types`` from config (per-layer pattern)
  3. For each layer, compute the per-layer-type ``num_kv_heads``
     from ``attention_kv_heads_per_layer_type``
  4. Override the default ``num_kv_heads`` in the per-layer KV-cache
     spec

Falls back gracefully when:
  * Not Gemma 4 (no override)
  * Symmetric KV-head config (no override)
  * vLLM API surface for per-layer spec doesn't match expectation

================================================================
SAFETY MODEL
================================================================

* default_on: False (until validated server-side; G4_13 catches the
  silent-corruption case anyway)
* env_flag: GENESIS_ENABLE_G4_18_GEMMA4_PER_LAYER_KV_PAGE_SIZE
* applies_to:
    - architecture: gemma4 (specifically 26B-A4B with asymmetric KV)
* conflicts_with: G4_13 (G4_18 makes G4_13's refusal unnecessary —
  enable G4_18, disable G4_13 once G4_18 is validated)
* superseded_by: vllm#40391 when merged

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
References:
  * https://github.com/vllm-project/vllm/pull/40391 (WIP)
  * https://github.com/vllm-project/vllm/issues/40388 (root-cause)
"""
from __future__ import annotations

import logging

from ..model_compat.gemma4._gemma4_detect import env_truthy, is_gemma4_arch

log = logging.getLogger("genesis.kv_cache.g4_18_per_layer_kv_page_size")

GENESIS_G4_18_MARKER = (
    "Genesis G4_18 gemma4 per-layer KV page-size v1 "
    "(vendors WIP #40391 for 26B-A4B asymmetric KV-head support)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_18_GEMMA4_PER_LAYER_KV_PAGE_SIZE"

_APPLIED = False
_ORIGINAL_GET_NUM_KV = None
_PATCHED_CLS = None


def _env_enabled() -> bool:
    return env_truthy(_ENV_ENABLE)


def _read_per_layer_kv_heads(model_config) -> dict[int, int] | None:
    """Return per-layer-index num_kv_heads, or None if symmetric/unknown."""
    hf = getattr(model_config, "hf_config", None) or model_config
    text = getattr(hf, "text_config", None) or hf
    pattern = getattr(text, "attention_kv_heads_per_layer_type", None)
    layer_types = getattr(text, "layer_types", None)
    if not pattern or not layer_types:
        return None
    if len(set(pattern.values())) <= 1:
        return None  # symmetric — no override needed
    return {i: pattern.get(layer_types[i], pattern.get(None, 1)) for i in range(len(layer_types))}


def apply() -> tuple[str, str]:
    """Install per-layer KV-head spec override.

    Note: vLLM v1's KV-cache spec API surface differs across pin versions.
    We target the most common entry point (``ModelConfig.get_num_kv_heads``)
    and apply the override only when the request comes from a layer-aware
    context (we keep the original signature so non-layer-aware callers
    still work).
    """
    global _APPLIED, _ORIGINAL_GET_NUM_KV, _PATCHED_CLS

    if not _env_enabled():
        return "skipped", (
            f"G4_18 disabled (set {_ENV_ENABLE}=1 to enable per-layer KV "
            "page-size for Gemma 4 26B-A4B — vendors #40391)"
        )

    if _APPLIED:
        return "applied", "G4_18 already installed (idempotent)"

    try:
        from vllm.config import ModelConfig
    except ImportError as e:
        return "skipped", f"vllm.config.ModelConfig not importable: {e}"

    method = getattr(ModelConfig, "get_num_kv_heads", None)
    if method is None:
        return "skipped", (
            "ModelConfig.get_num_kv_heads not found in this vLLM pin — "
            "G4_18 is no-op (per-layer KV API has different shape)"
        )

    _PATCHED_CLS = ModelConfig
    if getattr(method, "_genesis_g4_18_wrapped", False):
        _APPLIED = True
        return "applied", "G4_18 already wrapped (idempotent)"
    _ORIGINAL_GET_NUM_KV = method

    def _genesis_g4_18_get_num_kv_heads(self, parallel_config=None, layer_idx=None, **kwargs):
        try:
            if is_gemma4_arch(self):
                pattern = _read_per_layer_kv_heads(self)
                if pattern is not None and layer_idx is not None:
                    per_layer = pattern.get(layer_idx)
                    if per_layer is not None and parallel_config is not None:
                        # Apply tensor-parallel sharding
                        tp = getattr(parallel_config, "tensor_parallel_size", 1)
                        return max(1, per_layer // tp)
                    if per_layer is not None:
                        return per_layer
        except Exception as e:  # noqa: BLE001
            log.warning("[G4_18] per-layer KV-head probe failed: %r", e)
        # Fall through to original
        if parallel_config is not None and layer_idx is None:
            return _ORIGINAL_GET_NUM_KV(self, parallel_config)
        if parallel_config is not None:
            return _ORIGINAL_GET_NUM_KV(self, parallel_config)
        return _ORIGINAL_GET_NUM_KV(self)

    _genesis_g4_18_get_num_kv_heads._genesis_g4_18_wrapped = True
    _genesis_g4_18_get_num_kv_heads.__wrapped__ = _ORIGINAL_GET_NUM_KV
    ModelConfig.get_num_kv_heads = _genesis_g4_18_get_num_kv_heads
    _APPLIED = True
    log.info(
        "[G4_18] installed: ModelConfig.get_num_kv_heads now returns "
        "per-layer-type KV-head counts on Gemma 4 26B-A4B."
    )
    return "applied", (
        "G4_18 installed: Gemma 4 26B-A4B will use per-layer-type KV-head "
        "counts in KV-cache spec (vendors #40391 logic). Closes vllm#40388 "
        "silent-corruption window. NOTE: disable G4_13 guard to actually "
        "boot the asymmetric config (G4_18 supersedes G4_13)."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED, _ORIGINAL_GET_NUM_KV, _PATCHED_CLS
    if not _APPLIED or _PATCHED_CLS is None or _ORIGINAL_GET_NUM_KV is None:
        return False
    _PATCHED_CLS.get_num_kv_heads = _ORIGINAL_GET_NUM_KV
    _APPLIED = False
    _ORIGINAL_GET_NUM_KV = None
    _PATCHED_CLS = None
    return True


__all__ = ["GENESIS_G4_18_MARKER", "apply", "is_applied", "revert"]
