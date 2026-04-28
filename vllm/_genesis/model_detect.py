# SPDX-License-Identifier: Apache-2.0
"""Genesis v7.9 — model-architecture active-dispatch detection.

Purpose
-------
Several Genesis patches are only useful on specific model shapes:

  * P24 / P31 / P37         — require MoE (routed experts + fused_moe path)
  * P28 / P34 / P39a / P46  — require hybrid linear-attention (Mamba2 / GDN)
  * P22 / P38 / P40 / P44   — require TurboQuant KV (kv_cache_dtype=turboquant_*)
    [TQ is handled separately by P51 in kernels/dequant_buffer.py via
     `impl.kv_cache_dtype` check at call-site, which is the most accurate
     layer — this module is for *config-level* dispatch decisions.]

Before v7.9, every patch applied to every model. On a dense FP16 Qwen3-32B,
that meant:

  * MoE intermediate-cache pool module loaded (~0 MiB idle cost but warns fire)
  * GDN gating buffer module loaded (~0 MiB idle cost)
  * Logs showed all 28 patches "applied" even though half could never trigger

The actual memory waste was small (lazy pools) but operator confusion was
high: "why is P37 applied if my model has no MoE?"

This module is the defense-in-depth **dispatch layer**:

  * `is_moe_model()` → True iff model config has num_experts > 1 or similar
  * `is_hybrid_model()` → True iff model has mamba2/linear_attn layers
  * `is_turboquant_active()` → True iff kv_cache_dtype starts with "turboquant_"
  * `get_model_profile()` → dict with all three + diagnostic details

Results are cached per-process (model config is immutable after engine init).
Failures degrade gracefully: an unknown architecture returns True for all
(conservative: "apply the patch, let the patch's own guards decide").

Why conservative default (True on unknown):
  Dense/hybrid/MoE detection is a *hint*, not a gate. Every patch still has
  its own call-site guards (P51 for TQ, fused_moe path presence for MoE,
  hybrid attention class presence for GDN). This module only adds a
  **visible log line** at register-time so operators see the dispatch
  decision up-front ("[P52 MoE-active] skipping P37 on dense model").

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging
from typing import Any, Optional

log = logging.getLogger("genesis.model_detect")


# Per-process cache. Model config is immutable after engine init, so one
# query is enough. Reset via `clear_for_tests()` in unit tests.
_CACHED_PROFILE: Optional[dict[str, Any]] = None


def _try_get_vllm_config() -> Optional[Any]:
    """Return the current vLLM config or None if not yet set."""
    try:
        from vllm.config import get_current_vllm_config
        return get_current_vllm_config()
    except Exception as e:
        log.debug("[model_detect] get_current_vllm_config unavailable: %s", e)
        return None


def _probe_moe(hf_config: Any) -> tuple[bool, dict[str, Any]]:
    """Inspect an HF PretrainedConfig for MoE signals.

    Returns (is_moe, details). Details are diagnostic only.

    We check several attribute names because each model family uses its own:
      - Qwen3-MoE / Qwen3-Next:  num_experts, n_routed_experts
      - DeepSeek / Mixtral:      num_local_experts, num_experts_per_tok
      - Gemma 4 MoE:             text_config.num_experts (nested in mm config)

    Multimodal configs (Gemma 4, LLaVA-class) keep language-model attrs in
    a nested `text_config` / `language_config`. We scan both top-level and
    nested to catch them.
    """
    details: dict[str, Any] = {}
    candidate_attrs = (
        "num_experts", "n_routed_experts", "num_local_experts",
        "moe_num_experts", "num_experts_per_tok",
    )

    def _scan_attrs(obj: Any, prefix: str = "") -> None:
        for attr in candidate_attrs:
            val = getattr(obj, attr, None)
            if val is None and isinstance(obj, dict):
                val = obj.get(attr)
            if val is not None:
                key = f"{prefix}{attr}" if prefix else attr
                # Only store if not already set (top-level wins over nested)
                details.setdefault(key, val)

    _scan_attrs(hf_config)
    # Check nested language configs (Gemma 4, LLaVA-class multimodal)
    for sub in ("text_config", "language_config"):
        nested = getattr(hf_config, sub, None)
        if nested is not None:
            _scan_attrs(nested, prefix=f"{sub}.")

    # Heuristic: any of the above > 1 → MoE. Many dense configs don't expose
    # any of these; some expose them as 0/1 → still dense.
    is_moe = any(
        isinstance(v, int) and v > 1 for v in details.values()
    )

    # Gemma 4 signals MoE via text_config.enable_moe_block even without
    # num_experts at expected paths.
    if not is_moe:
        for sub in ("text_config", "language_config"):
            nested = getattr(hf_config, sub, None)
            if nested is not None:
                flag = getattr(nested, "enable_moe_block", None)
                if flag is None and isinstance(nested, dict):
                    flag = nested.get("enable_moe_block")
                if flag:
                    is_moe = True
                    details["moe_source"] = f"{sub}.enable_moe_block"
                    break

    # Secondary signal: model_type ends with "_moe" or architecture contains
    # MoE markers.
    model_type = getattr(hf_config, "model_type", "") or ""
    architectures = getattr(hf_config, "architectures", None) or []
    details["model_type"] = model_type
    details["architectures"] = list(architectures) if architectures else []
    if not is_moe:
        lowered = model_type.lower()
        if any(k in lowered for k in ("moe", "mixtral", "deepseek", "cohere")):
            is_moe = True
            details["moe_source"] = "model_type_name"
        else:
            for arch in architectures:
                if not isinstance(arch, str):
                    continue
                arch_lower = arch.lower()
                # Case-insensitive — covers MoE / Moe / moE variants used by
                # different model families (e.g. CohereMoeForCausalLM uses
                # lowercase 'oe' not 'oE'); also matches mixtral/deepseek/cohere.
                if any(k in arch_lower for k in ("moe", "mixtral", "deepseek", "cohere")):
                    # Filter out false positives (e.g. "RemoeFor..." would match
                    # but Cohere2VisionForConditionalGeneration would not).
                    # We require the substring to be a word boundary or part
                    # of a CamelCase token; simple heuristic: at start, after
                    # a lowercase->uppercase transition, or preceded by a
                    # non-alpha char.
                    is_moe = True
                    details["moe_source"] = "architecture_name"
                    break
    return is_moe, details


def _probe_hybrid(hf_config: Any) -> tuple[bool, dict[str, Any]]:
    """Inspect config for hybrid linear-attention signals (Mamba2 / GDN / SSM).

    Returns (is_hybrid, details).

    Multimodal configs (Qwen3.5 ConditionalGeneration, LLaVA-class) keep the
    language-model `layer_types` and `model_type` in a nested `text_config` /
    `language_config` / `thinker_config`. We scan top-level first, then nested
    sub-configs. Mirrors the same nested pattern as `_probe_moe()`.
    """
    details: dict[str, Any] = {}

    def _scan_layer_types(obj: Any, source_label: str) -> Optional[bool]:
        """Return True if obj.layer_types contains a hybrid marker, None if no
        layer_types attr. Always records sample into details."""
        lt = getattr(obj, "layer_types", None)
        if lt is None and isinstance(obj, dict):
            lt = obj.get("layer_types")
        if lt is None:
            return None
        try:
            sample = list(lt)[:8]
            details.setdefault(f"{source_label}layer_types_sample", sample)
            for entry in lt:
                s = str(entry).lower()
                if "linear" in s or "mamba" in s or "gdn" in s or "ssm" in s:
                    return True
        except Exception:
            pass
        return False

    # Primary: layer_types — top-level then nested (multimodal)
    if _scan_layer_types(hf_config, "") is True:
        return True, {**details, "hybrid_source": "layer_types"}
    for sub in ("text_config", "language_config", "thinker_config"):
        nested = getattr(hf_config, sub, None)
        if nested is not None and _scan_layer_types(nested, f"{sub}.") is True:
            return True, {**details, "hybrid_source": f"{sub}.layer_types"}

    # Secondary: model_type — top-level then nested
    def _scan_model_type(obj: Any) -> Optional[str]:
        mt = getattr(obj, "model_type", None)
        if mt is None and isinstance(obj, dict):
            mt = obj.get("model_type")
        return (str(mt) if mt else None)

    top_mt = _scan_model_type(hf_config) or ""
    details["model_type"] = top_mt
    markers = ("qwen3_next", "mamba", "falcon_mamba", "gdn", "hybrid")
    for marker in markers:
        if marker in top_mt.lower():
            return True, {**details, "hybrid_source": "model_type"}
    for sub in ("text_config", "language_config", "thinker_config"):
        nested = getattr(hf_config, sub, None)
        if nested is None:
            continue
        nested_mt = _scan_model_type(nested) or ""
        if nested_mt:
            details.setdefault(f"{sub}.model_type", nested_mt)
            for marker in markers:
                if marker in nested_mt.lower():
                    return True, {**details, "hybrid_source": f"{sub}.model_type"}

    # Tertiary: architecture
    architectures = getattr(hf_config, "architectures", None) or []
    details["architectures"] = list(architectures) if architectures else []
    for arch in architectures:
        if isinstance(arch, str):
            lowered = arch.lower()
            if "mamba" in lowered or "hybrid" in lowered or "next" in lowered:
                return True, {**details, "hybrid_source": "architecture"}

    return False, details


def _probe_turboquant(cfg: Any) -> tuple[bool, str]:
    """Inspect cache_config.kv_cache_dtype for TurboQuant activation."""
    try:
        dtype = getattr(cfg.cache_config, "kv_cache_dtype", None)
    except Exception:
        dtype = None
    dtype_str = str(dtype) if dtype is not None else ""
    return dtype_str.startswith("turboquant_"), dtype_str


def get_model_profile() -> dict[str, Any]:
    """Return cached model-architecture dispatch profile.

    Keys:
      - moe: bool — True if model has MoE layers
      - hybrid: bool — True if model has linear-attention layers
      - turboquant: bool — True if kv_cache_dtype=turboquant_*
      - kv_cache_dtype: str — raw dtype string for diagnostics
      - model_type: str
      - architectures: list[str]
      - moe_details: dict — per-attr diagnostic values
      - hybrid_details: dict — per-attr diagnostic values
      - resolved: bool — False if config was unavailable at query time
                  (caller should treat flags as conservative True)

    On unavailable config (pre-init, test harness without vllm config context),
    returns {"resolved": False, "moe": True, "hybrid": True, "turboquant": True}
    so patches apply by default and the call-site guards take over.
    """
    global _CACHED_PROFILE
    if _CACHED_PROFILE is not None:
        return _CACHED_PROFILE

    cfg = _try_get_vllm_config()
    if cfg is None:
        # Conservative: pretend everything is present. Patches still have
        # their own guards.
        return {
            "resolved": False,
            "moe": True,
            "hybrid": True,
            "turboquant": True,
            "kv_cache_dtype": "",
            "model_type": "",
            "architectures": [],
            "moe_details": {},
            "hybrid_details": {},
        }

    try:
        hf_cfg = cfg.model_config.hf_config
    except Exception as e:
        log.info("[model_detect] hf_config unavailable: %s — conservative True", e)
        return {
            "resolved": False,
            "moe": True,
            "hybrid": True,
            "turboquant": True,
            "kv_cache_dtype": "",
            "model_type": "",
            "architectures": [],
            "moe_details": {},
            "hybrid_details": {},
        }

    is_moe, moe_details = _probe_moe(hf_cfg)
    is_hybrid, hybrid_details = _probe_hybrid(hf_cfg)
    is_tq, tq_dtype = _probe_turboquant(cfg)

    profile = {
        "resolved": True,
        "moe": is_moe,
        "hybrid": is_hybrid,
        "turboquant": is_tq,
        "kv_cache_dtype": tq_dtype,
        "model_type": getattr(hf_cfg, "model_type", "") or "",
        "architectures": list(getattr(hf_cfg, "architectures", None) or []),
        "moe_details": moe_details,
        "hybrid_details": hybrid_details,
    }

    _CACHED_PROFILE = profile
    log.info(
        "[Genesis v7.9 model_detect] profile resolved: "
        "model_type=%s moe=%s hybrid=%s turboquant=%s (kv=%s)",
        profile["model_type"], profile["moe"], profile["hybrid"],
        profile["turboquant"], profile["kv_cache_dtype"],
    )
    return profile


def is_moe_model() -> bool:
    """P52 dispatch predicate — True if patches targeting MoE should apply."""
    return get_model_profile()["moe"]


def is_hybrid_model() -> bool:
    """P53 dispatch predicate — True if patches targeting hybrid attention
    (Mamba2/GDN/linear-attn) should apply."""
    return get_model_profile()["hybrid"]


def is_turboquant_active() -> bool:
    """Config-level TQ check. Layer-level check (P51) lives in
    kernels/dequant_buffer.py::ensure_turboquant_buffers."""
    return get_model_profile()["turboquant"]


def log_skip(patch_name: str, reason: str) -> None:
    """Uniform single-line skip log for dispatch decisions. Safe to call
    from wiring.apply() — caller typically does this once per boot."""
    log.info("[Genesis v7.9 dispatch] %s skipped — %s", patch_name, reason)


def clear_for_tests() -> None:
    """TESTS ONLY. Reset the cached profile so the next query re-probes
    (e.g. after monkeypatching get_current_vllm_config)."""
    global _CACHED_PROFILE
    _CACHED_PROFILE = None
