# SPDX-License-Identifier: Apache-2.0
"""Shared detection utilities for Gemma 4 family patches.

These helpers answer the three questions every G4 patch asks at apply
time:

  * ``is_gemma4_arch(...)``     — is the live model a Gemma 4 variant?
  * ``is_ampere_sm86(...)``     — are we on Ampere consumer / RTX A5000?
  * ``infer_moe_kdim(...)``     — what shape does the next Marlin GEMM see?

Importing torch / vllm is lazy so the module survives test collection
on hosts without CUDA.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger("genesis.gemma4.detect")


# ─── Architecture detection ──────────────────────────────────────────


GEMMA4_ARCHITECTURES: frozenset[str] = frozenset({
    "Gemma4ForConditionalGeneration",       # multimodal wrapper (31B / 26B / E4B / E2B)
    "Gemma4ForCausalLM",                    # text-only path
    "Gemma4TextModel",                      # rare; embedding extractor
    "Gemma4_AssistantForCausalLM",          # MTP assistant drafter
    "Gemma4AssistantForCausalLM",           # spelling drift
})


def is_gemma4_arch(value: object) -> bool:
    """Return True when ``value`` looks like a Gemma 4 architecture marker.

    Accepts:
      * a hf_config object (reads ``architectures`` list)
      * a string (single arch name)
      * a list/tuple/set of strings
      * a model_type string ("gemma4", "gemma4_assistant")
      * a vllm_config whose ``model_config.hf_config.architectures``
        contains a Gemma 4 entry
    """
    if value is None:
        return False
    # Probe vllm_config
    mc = getattr(value, "model_config", None)
    if mc is not None:
        hf = getattr(mc, "hf_config", None)
        if hf is not None:
            return is_gemma4_arch(hf)
    # Probe hf_config-like
    archs = getattr(value, "architectures", None)
    if archs is not None:
        for arch in archs:
            if str(arch) in GEMMA4_ARCHITECTURES:
                return True
    model_type = getattr(value, "model_type", None)
    if isinstance(model_type, str) and model_type.startswith("gemma4"):
        return True
    # Direct string forms
    if isinstance(value, str):
        if value in GEMMA4_ARCHITECTURES:
            return True
        if value.startswith("gemma4") or value.startswith("Gemma4"):
            return True
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(is_gemma4_arch(item) for item in value)
    return False


# ─── Hardware detection ──────────────────────────────────────────────


def get_compute_capability() -> Optional[tuple[int, int]]:
    """Return ``(major, minor)`` for device 0 or None if torch/cuda absent.

    Cached after first call to avoid repeated cudaGetDeviceProperties.
    """
    global _cached_cap
    cap = globals().get("_cached_cap", _SENTINEL)
    if cap is not _SENTINEL:
        return cap
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            cap = None
        else:
            cap = torch.cuda.get_device_capability(0)
    except Exception as e:  # noqa: BLE001
        log.debug("cuda probe failed: %s", e)
        cap = None
    globals()["_cached_cap"] = cap
    return cap


_SENTINEL = object()


def is_ampere_sm86() -> bool:
    """True on RTX 3090 / 4090 / A5000 / A6000 (SM 8.6) consumer Ampere.

    Note: RTX 3060/3070 are also SM 8.6, RTX 4090/4080 are SM 8.9 (Ada).
    SM 8.0 is A100/A800 (datacenter Ampere) — we accept 8.x ≤ 8.6 for
    this guard since the same Marlin / non-causal-attention constraints
    apply to all Ampere consumer chips, but not to Hopper (9.0+) or
    Blackwell (10.x / 12.x).
    """
    cap = get_compute_capability()
    if cap is None:
        return False
    major, minor = cap
    if major != 8:
        return False
    return minor <= 6  # 8.0 / 8.6 = Ampere; 8.9 = Ada (Lovelace) — different FP8 support


def is_hopper_or_later() -> bool:
    """True on Hopper (9.x), Blackwell (10.x / 12.x), or newer."""
    cap = get_compute_capability()
    if cap is None:
        return False
    return cap[0] >= 9


# ─── Quant + shape probes ────────────────────────────────────────────


def detect_fp8_block_format(quant_config: object) -> bool:
    """Return True for the llm-compressor FP8_BLOCK checkpoint shape that
    bug vllm#39407 affects.

    Detection signature (from #39407 root-cause analysis):
      * quant_method in {"compressed-tensors", "fp8"}
      * format == "float-quantized"
      * weight_quant.strategy == "block"
      * weight_quant.block_structure is not None (e.g. [128, 128])
      * activation scales are absorbed into weights (no separate input_scale)
    """
    if quant_config is None:
        return False
    qm = getattr(quant_config, "quant_method", None) or getattr(quant_config, "method", None)
    if qm is not None and str(qm) not in ("compressed-tensors", "fp8", "compressed_tensors"):
        return False
    fmt = getattr(quant_config, "format", None) or getattr(quant_config, "quant_format", None)
    if fmt is not None and str(fmt) != "float-quantized":
        return False
    wq = getattr(quant_config, "weight_quant", None) or getattr(quant_config, "config_groups", None)
    if wq is None:
        return False
    # Walk into the strategy field — schema differs per quant_method
    strat = _probe_strategy(wq)
    if strat is None:
        return False
    return str(strat).lower() == "block"


def _probe_strategy(obj: object) -> Optional[str]:
    """Best-effort: find the ``strategy`` field anywhere in a quant config."""
    if obj is None:
        return None
    strat = getattr(obj, "strategy", None)
    if strat is not None:
        return str(strat).lower()
    # Dict-style (some quant configs are plain dicts)
    if isinstance(obj, dict):
        if "strategy" in obj:
            return str(obj["strategy"]).lower()
        for v in obj.values():
            r = _probe_strategy(v)
            if r is not None:
                return r
    # config_groups list of group objects
    if isinstance(obj, (list, tuple)):
        for item in obj:
            r = _probe_strategy(item)
            if r is not None:
                return r
    return None


def infer_marlin_kdim_for_moe(layer: object) -> Optional[int]:
    """Best-effort: return the prob_k value the next Marlin MoE GEMM will see.

    Used by G4_02 to fail fast before the Marlin tile-finder raises an
    obscure ``Invalid thread config`` error. Returns None when probe
    fails so the guard is fail-open (doesn't refuse boot if we can't
    tell).
    """
    # For MoE, the failing GEMM is the down_proj path: K=intermediate_size_per_partition,
    # N=hidden_size. The layer should expose the per-partition intermediate via
    # one of these attribute names.
    for attr in (
        "intermediate_size_per_partition",
        "_g4_intermediate_real",
        "moe_intermediate_size_per_partition",
    ):
        val = getattr(layer, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    # Last resort: read from w2_weight tensor's K dim
    w2 = getattr(layer, "w2_weight", None)
    if w2 is not None and hasattr(w2, "shape") and len(w2.shape) >= 2:
        # w2 typically [num_experts, hidden_size, intermediate_size_per_partition]
        return int(w2.shape[-1])
    return None


def marlin_kdim_supported(prob_k: int, strict_python_check: bool = True) -> bool:
    """Return True when Marlin tile-finder will accept ``prob_k``.

    C++ side requires ``prob_k % min_thread_k(=64) == 0``; the Python-side
    GPTQ Marlin verifier in ``marlin_utils.py`` is stricter and requires
    ``prob_k % 128 == 0``. The Python path is what runs first for our
    compressed-tensors W8A8 FP8 MoE call site.
    """
    if prob_k <= 0:
        return False
    if strict_python_check:
        return prob_k % 128 == 0
    return prob_k % 64 == 0


# ─── Spec-decode drafter detection ───────────────────────────────────


def detect_non_causal_drafter(speculative_config: object) -> Optional[str]:
    """Return the drafter method name when it requires non-causal attention.

    Affects:
      * ``method == "eagle3"``  — EAGLE-3 block drafter (non-causal)
      * ``method == "dflash"`` — DFlash block-parallel drafter (non-causal)

    Returns the method name on detection, None otherwise.
    """
    if speculative_config is None:
        return None
    method = getattr(speculative_config, "method", None)
    if method is None and isinstance(speculative_config, dict):
        method = speculative_config.get("method")
    if isinstance(method, str) and method.lower() in ("eagle3", "dflash"):
        return method.lower()
    return None


# ─── Common env-flag helpers ─────────────────────────────────────────


def env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def env_disable(name: str) -> bool:
    """Mirror env-flag for explicit operator override (DISABLE wins over ENABLE)."""
    return env_truthy(name)


__all__ = [
    "GEMMA4_ARCHITECTURES",
    "is_gemma4_arch",
    "get_compute_capability",
    "is_ampere_sm86",
    "is_hopper_or_later",
    "detect_fp8_block_format",
    "infer_marlin_kdim_for_moe",
    "marlin_kdim_supported",
    "detect_non_causal_drafter",
    "env_truthy",
    "env_disable",
]
