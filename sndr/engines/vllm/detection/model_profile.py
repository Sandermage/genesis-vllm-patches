# SPDX-License-Identifier: Apache-2.0
"""Genesis Model Profile — typed wrapper + dispatch API.

Thin bridge over `model_detect.py` (the canonical detection layer).
Adds:
  - `GenesisModelProfile` typed dataclass with predicates
  - `_derive_hot_kernels()` — kernel family classifier
  - `should_apply_patch_for_model()` — explicit 2D dispatch helper
  - `patches_relevant_for_model()` — auto-build env from model+arch

Detection itself (hybrid/moe/turboquant/quant_format/model_class) is
DELEGATED to `model_detect.get_model_profile()` which is the
single source of truth and handles AutoRound INT4/INT8 bit refinement,
compressed_tensors format normalization, nested layer_types fallback,
and proper TurboQuant cache_config probe.

This module deliberately does NOT re-implement any detection probes.
Earlier drafts contained `_detect_family`, `_detect_topology`,
`_detect_quant`, and arch sets (_QWEN3_HYBRID_ARCHS, etc.) duplicating
canonical helpers — all removed 2026-06-05 after duplication audit.

Author: Sandermage (Sander) Barzov Aleksandr, 2026-06-05.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger("genesis.detection.model_profile")


# ─── Typed dataclass ────────────────────────────────────────────────


@dataclass(frozen=True)
class GenesisModelProfile:
    """Frozen snapshot of model architecture + quantization + spec-decode
    + topology, with typed predicates.

    Detection sourced from `model_detect.get_model_profile()` canonical.
    """

    # Raw identification
    architecture: str           # e.g. "Qwen3_5MoeForConditionalGeneration"
    model_name: str             # short, best-effort from served_model_name
    family: str                 # canonical model_class: "qwen3" / "gemma" / "llama" / ...

    # Topology
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    is_hybrid: bool             # has linear_attn / GDN / Mamba layers
    is_moe: bool                # has expert layers
    is_dense: bool              # neither hybrid nor MoE

    # Quantization
    weight_dtype: str           # "fp8_e4m3" / "int4" / "int8" / "fp16" / "bf16"
    activation_dtype: str       # "fp16" / "bf16" / "fp8"
    quant_method: str           # "auto_round" / "gptq" / "awq" / "fp8" / "compressed_tensors" / "none"
    kv_cache_dtype: str         # "turboquant_k8v4" / "fp8" / "auto"

    # Spec decode
    spec_method: Optional[str]
    spec_K: int

    # Parallelism
    tensor_parallel_size: int
    pipeline_parallel_size: int

    # Hot kernel families — drives 2D dispatch decisions
    hot_kernels: tuple[str, ...] = field(default_factory=tuple)

    # Convenience predicates
    @property
    def is_qwen3_hybrid(self) -> bool:
        return self.family == "qwen3" and self.is_hybrid

    @property
    def uses_gdn(self) -> bool:
        return "gdn" in self.hot_kernels

    @property
    def uses_marlin(self) -> bool:
        return any(k.startswith("marlin") for k in self.hot_kernels)

    @property
    def uses_tq(self) -> bool:
        return self.kv_cache_dtype.startswith("turboquant")

    @property
    def has_mtp(self) -> bool:
        return self.spec_method == "mtp" and self.spec_K > 0

    @property
    def is_fp8_quant(self) -> bool:
        return self.weight_dtype == "fp8_e4m3" or self.quant_method == "fp8"


# ─── Quant translation (canonical format → triple) ──────────────────


def _quant_from_canonical(
    canonical_format: str,
    model_config: Any,
) -> tuple[str, str, str]:
    """Translate canonical `model_detect.quant_format` into our triple
    (quant_method, weight_dtype, activation_dtype).

    Canonical values from `_probe_quant_format` (model_detect.py):
      'fp8', 'autoround_int4', 'autoround_int8', 'awq_int4', 'gptq_int4',
      'int4_w4a16', 'int8_w8a16', 'int8_w8a8', 'compressed_tensors',
      'fp16', 'bf16', 'unknown'
    """
    fmt = (canonical_format or "unknown").lower()

    # NOTE: previously this section contained a workaround for the
    # `"autoround" not in "auto_round"` substring bug in canonical
    # `_probe_quant_format`. That bug is FIXED upstream (2026-06-05
    # — added "auto_round" and "auto-round" markers in model_detect.py).
    # When canonical now returns "unknown" it means truly unknown
    # quantization, not a missed substring. Workaround removed.

    # quant_method
    if "autoround" in fmt:
        quant_method = "auto_round"
    elif "awq" in fmt:
        quant_method = "awq"
    elif "gptq" in fmt:
        quant_method = "gptq"
    elif "compressed" in fmt:
        quant_method = "compressed_tensors"
    elif fmt == "fp8":
        quant_method = "fp8"
    else:
        quant_method = "none"

    # weight_dtype
    if "fp8" in fmt:
        weight_dtype = "fp8_e4m3"
    elif "int4" in fmt:
        weight_dtype = "int4"
    elif "int8" in fmt:
        weight_dtype = "int8"
    elif fmt == "bf16":
        weight_dtype = "bf16"
    elif fmt == "fp16":
        weight_dtype = "fp16"
    elif fmt == "compressed_tensors":
        dtype = str(getattr(model_config, "dtype", "")).lower()
        if "float8" in dtype:
            weight_dtype = "fp8_e4m3"
        elif "bfloat16" in dtype:
            weight_dtype = "bf16"
        else:
            weight_dtype = "fp16"
    else:
        weight_dtype = "fp16"

    # activation_dtype (dequant target for quantized weights)
    if weight_dtype in ("int4", "int8", "fp8_e4m3"):
        activation_dtype = "fp16"
    elif weight_dtype == "bf16":
        activation_dtype = "bf16"
    else:
        activation_dtype = "fp16"

    return quant_method, weight_dtype, activation_dtype


# ─── Hot kernel classifier ──────────────────────────────────────────


def _derive_hot_kernels(profile_fields: dict) -> tuple[str, ...]:
    """Determine which kernel families are exercised on hot path.

    Derived from topology + quant_method + kv_cache_dtype + spec_method.
    The kernel-family vocabulary is the input to `should_apply_patch_for_model`.
    """
    hot = []

    # GDN hot for hybrid Qwen3+ family
    if profile_fields["family"] == "qwen3" and profile_fields["is_hybrid"]:
        hot.append("gdn")

    # Classic Mamba SSM hot for pure Mamba/Jamba families
    if profile_fields["family"] == "mamba":
        hot.append("mamba_ssm")

    # TurboQuant decode hot when KV uses turboquant
    if profile_fields["kv_cache_dtype"].startswith("turboquant"):
        hot.append("tq_decode")

    # Marlin: INT4 / INT8 / FP8 quantization paths
    if profile_fields["quant_method"] in ("gptq", "awq", "auto_round"):
        if profile_fields["weight_dtype"] == "int8":
            hot.append("marlin_int8")
        else:
            hot.append("marlin_int4")
    elif profile_fields["weight_dtype"] == "fp8_e4m3":
        hot.append("marlin_fp8")
    elif profile_fields["quant_method"] == "fp8":
        hot.append("marlin_fp8")
    elif profile_fields["quant_method"] == "compressed_tensors":
        # Compressed tensors covers FP8 + INT8 + INT4 mixed; default Marlin path
        hot.append("marlin_fp8" if profile_fields["weight_dtype"] == "fp8_e4m3"
                  else "marlin_int4")

    # FlashAttention hot for any model with attention (most)
    hot.append("flash_attn")

    # MoE routing hot for MoE models
    if profile_fields["is_moe"]:
        hot.append("moe_routing")

    # Spec-decode draft model paths
    if profile_fields["spec_method"] == "mtp":
        hot.append("mtp_draft")
    elif profile_fields["spec_method"] == "eagle":
        hot.append("eagle_draft")

    return tuple(hot)


# ─── Detection bridge ───────────────────────────────────────────────


_PROFILE: Optional[GenesisModelProfile] = None
_DETECTION_ATTEMPTED = False


def get_model_profile(vllm_config: Any = None) -> Optional[GenesisModelProfile]:
    """Return cached model profile. Detection on first call.

    Args:
        vllm_config: Optional VllmConfig instance. If not provided,
                     `model_detect` will look it up via current engine state.

    Note: if first call had no vllm_config (None result) and a subsequent
    call provides one, we RETRY detection instead of returning the cached
    None. Prevents a stuck "no profile" state across boot phases.
    """
    global _PROFILE, _DETECTION_ATTEMPTED
    if _PROFILE is not None:
        return _PROFILE
    if _DETECTION_ATTEMPTED and vllm_config is None:
        return _PROFILE
    _DETECTION_ATTEMPTED = True
    _PROFILE = _detect(vllm_config)
    return _PROFILE


def reset_model_profile() -> None:
    """Test helper: reset cached profile."""
    global _PROFILE, _DETECTION_ATTEMPTED
    _PROFILE = None
    _DETECTION_ATTEMPTED = False


def _detect(vllm_config: Any) -> Optional[GenesisModelProfile]:
    """Build the profile from vllm_config via canonical `model_detect`."""
    if vllm_config is None:
        log.info("[model_profile] no vllm_config provided — detection skipped")
        return None

    try:
        model_cfg = vllm_config.model_config
        hf_cfg = getattr(model_cfg, "hf_config", None)
        parallel_cfg = vllm_config.parallel_config
        spec_cfg = getattr(vllm_config, "speculative_config", None)
    except Exception as e:
        log.warning("[model_profile] config introspection failed: %s", e)
        return None

    # Canonical detection — single source of truth for hybrid/moe/TQ/quant.
    # We call the probes DIRECTLY with our known vllm_config (not via
    # model_detect.get_model_profile() which reads a global config that
    # may be None in lazy-retry boot context). This is the canonical fix
    # for "family=unknown" regression observed 2026-06-05 in PN302 retry.
    #
    # v12.0 (2026-06-06): import the probes from the CANONICAL path. The
    # legacy ``vllm.sndr_core.detection.model_detect`` shim uses
    # ``from X import *`` which does NOT re-export private ``_probe_*``
    # functions; the AttributeError on the first ``_md._probe_moe(...)``
    # access would be caught by the ``except`` below and silently fall
    # back to all-False, which caused the live Qwen3_5MoeForConditionalGeneration
    # to be classified as dense=True moe=False — silently disabling every
    # patch that gates on ``is_moe`` / ``is_hybrid``.
    try:
        from sndr.engines.vllm.detection import model_detect as _md
        is_moe, _moe_details = _md._probe_moe(hf_cfg)
        is_hybrid_raw, _hyb_details = _md._probe_hybrid(hf_cfg)
        is_tq, tq_dtype = _md._probe_turboquant(vllm_config)
        quant_format = _md._probe_quant_format(vllm_config, hf_cfg)
        model_class = _md._probe_model_class(hf_cfg)
        canonical = {
            "resolved": True,
            "moe": is_moe,
            "hybrid": is_hybrid_raw,
            "turboquant": is_tq,
            "kv_cache_dtype": tq_dtype,
            "quant_format": quant_format,
            "model_class": model_class,
            "architectures": list(getattr(hf_cfg, "architectures", []) or []),
            "model_type": getattr(hf_cfg, "model_type", "") or "",
        }
    except Exception as e:
        log.warning("[model_profile] model_detect bridge failed: %s — "
                    "falling back to empty canonical", e)
        canonical = {
            "resolved": False, "moe": False, "hybrid": False,
            "turboquant": False, "kv_cache_dtype": "auto",
            "quant_format": "unknown", "model_class": "unknown",
            "architectures": [], "model_type": "",
        }

    # Architecture string (raw, for diagnostic logging)
    architecture = "unknown"
    if hf_cfg is not None:
        archs = getattr(hf_cfg, "architectures", None) or []
        if archs:
            architecture = archs[0]

    # Family — canonical model_class (single source). model_detect returns
    # versioned identifiers like "qwen3_5", "qwen3_6", "qwen3_next" — we
    # normalize to the base family name for downstream `family == "qwen3"`
    # predicates (used by `_derive_hot_kernels` GDN classifier and by
    # `should_apply_patch_for_model`).
    _raw_class = canonical.get("model_class", "unknown") or "unknown"
    if _raw_class.startswith("qwen3"):
        family = "qwen3"
    elif _raw_class.startswith("gemma"):
        family = "gemma"
    elif _raw_class.startswith("llama"):
        family = "llama"
    elif _raw_class.startswith("mixtral"):
        family = "mixtral"
    elif _raw_class.startswith("mamba") or _raw_class.startswith("jamba"):
        family = "mamba"
    else:
        family = _raw_class

    # Topology — canonical only
    is_hybrid = bool(canonical.get("hybrid", False))
    is_moe = bool(canonical.get("moe", False))
    is_dense = not is_hybrid and not is_moe

    # Quantization — canonical format → typed triple
    quant_format = canonical.get("quant_format", "unknown")
    quant_method, weight_dtype, activation_dtype = _quant_from_canonical(
        quant_format, model_cfg
    )

    # KV cache dtype: vllm normalizes "turboquant_k8v4" → "auto" in
    # cache_config at runtime, so canonical probe sees "auto". Three
    # fallback signals to recover the TQ truth (any one wins):
    #   1) canonical.turboquant boolean (works at config-build time)
    #   2) GENESIS_ENABLE_*TQ* env vars set by launcher (operator signal)
    #   3) attn_backend.name contains "turboquant" (runtime introspection)
    kv_cache_dtype = canonical.get("kv_cache_dtype", "auto") or "auto"
    tq_active = canonical.get("turboquant", False)
    if not tq_active:
        import os as _os
        # Signal 2: any GENESIS_ENABLE_*TQ* env var set → operator declared TQ
        for _k, _v in _os.environ.items():
            if "TQ" in _k and "GENESIS_ENABLE" in _k and _v.lower() in (
                "1", "true", "yes", "on"
            ):
                tq_active = True
                break
        # Signal 3: GENESIS_TQ_MAX_MODEL_LEN set (launcher TQ marker)
        if not tq_active and _os.environ.get("GENESIS_TQ_MAX_MODEL_LEN"):
            tq_active = True
    if tq_active and not kv_cache_dtype.startswith("turboquant"):
        kv_cache_dtype = "turboquant_k8v4"

    # Spec decode
    spec_method = None
    spec_K = 0
    if spec_cfg is not None:
        spec_method = getattr(spec_cfg, "method", None)
        spec_K = getattr(spec_cfg, "num_speculative_tokens", 0) or 0

    # Hidden config — Qwen3.5/3.6/Gemma 4 multimodal configs keep the
    # language-model fields nested under ``text_config`` (or
    # ``language_config`` / ``thinker_config``). Walk the same nesting
    # pattern as the MoE/hybrid probes so we don't silently report
    # layers=0 / hidden=0 / GQA=0/0 for every multimodal model.
    def _resolve_attr(name: str, default: int = 0) -> int:
        if hf_cfg is None:
            return default
        val = getattr(hf_cfg, name, None)
        if val is not None:
            return val
        for sub in ("text_config", "language_config", "thinker_config"):
            nested = getattr(hf_cfg, sub, None)
            if nested is None:
                continue
            nested_val = getattr(nested, name, None)
            if nested_val is None and isinstance(nested, dict):
                nested_val = nested.get(name)
            if nested_val is not None:
                return nested_val
        return default

    num_layers = _resolve_attr("num_hidden_layers", 0)
    hidden_size = _resolve_attr("hidden_size", 0)
    num_heads = _resolve_attr("num_attention_heads", 0)
    num_kv_heads = _resolve_attr("num_key_value_heads", num_heads)

    # Short name
    model_name = getattr(model_cfg, "served_model_name", None) or family

    fields = {
        "family": family,
        "is_hybrid": is_hybrid,
        "is_moe": is_moe,
        "weight_dtype": weight_dtype,
        "quant_method": quant_method,
        "kv_cache_dtype": kv_cache_dtype,
        "spec_method": spec_method,
    }
    hot_kernels = _derive_hot_kernels(fields)

    profile = GenesisModelProfile(
        architecture=architecture,
        model_name=str(model_name),
        family=family,
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_kv_heads=num_kv_heads,
        is_hybrid=is_hybrid,
        is_moe=is_moe,
        is_dense=is_dense,
        weight_dtype=weight_dtype,
        activation_dtype=activation_dtype,
        quant_method=quant_method,
        kv_cache_dtype=kv_cache_dtype,
        spec_method=spec_method,
        spec_K=spec_K,
        tensor_parallel_size=getattr(parallel_cfg, "tensor_parallel_size", 1),
        pipeline_parallel_size=getattr(parallel_cfg, "pipeline_parallel_size", 1),
        hot_kernels=hot_kernels,
    )
    log.warning(
        "[Genesis Model Profile] %s (family=%s, layers=%d, hidden=%d, GQA=%d/%d) | "
        "quant=%s weight=%s kv=%s | "
        "topology: hybrid=%s moe=%s dense=%s | "
        "spec=%s K=%d | TP=%d | hot_kernels=%s",
        profile.architecture, profile.family,
        profile.num_hidden_layers, profile.hidden_size,
        profile.num_attention_heads, profile.num_kv_heads,
        profile.quant_method, profile.weight_dtype, profile.kv_cache_dtype,
        profile.is_hybrid, profile.is_moe, profile.is_dense,
        profile.spec_method, profile.spec_K, profile.tensor_parallel_size,
        list(profile.hot_kernels),
    )
    return profile


# ─── Unified dispatch — combines arch + model profiles ──────────────


def should_apply_patch_for_model(
    patch_name: str,
    model_profile: Optional[GenesisModelProfile],
    arch_profile: Optional[Any] = None,
) -> tuple[bool, str]:
    """Decide if a Genesis patch is RELEVANT for the current (model, arch).

    Args:
        patch_name: Genesis patch ID e.g. "PN286" / "PN298" / "PN300"
        model_profile: GenesisModelProfile
        arch_profile: GenesisGPUArchProfile (optional)

    Returns:
        (should_apply, reason)

    Note: this is an EXPLICIT dispatch API for new model-aware patches.
    The legacy mechanism `dispatcher/decision.py:_check_applies_to()` reads
    declarative `applies_to` from registry; this function complements it
    for patches that want imperative 2D decisions (arch × model × workload).
    """
    if model_profile is None:
        return True, "no model profile — defer to existing patch logic"

    # PN286: FA layout revert for SM 8.6 — only matters if model uses FA
    if patch_name == "PN286":
        if "flash_attn" not in model_profile.hot_kernels:
            return False, "model doesn't use FlashAttention"
        if arch_profile is not None and not arch_profile.is_ampere_consumer:
            return False, f"PN286 SM 8.6 only — current arch SM {arch_profile.sm_string}"
        return True, "Qwen3.6 hybrid + FA + SM 8.6 = PN286 fires"

    # PN298/PN299/PN300: arch-aware Triton autotune for GDN/Mamba kernels
    if patch_name in ("PN298", "PN299", "PN300"):
        if not (model_profile.uses_gdn or "mamba_ssm" in model_profile.hot_kernels):
            return False, "model doesn't use GDN/Mamba kernels"
        return True, "GDN/Mamba kernels hot for this model"

    # PN302: the detector itself — unconditional
    if patch_name == "PN302":
        return True, "PN302 is the model profile detector — unconditional"

    # PN303 (planned): Marlin FP8 arch-aware — only for FP8 models on
    # archs WITHOUT native FP8 tensor cores (A5000/A6000 SM 8.6 etc).
    if patch_name == "PN303":
        if not model_profile.is_fp8_quant:
            return False, "model not FP8 quantized"
        if arch_profile is not None and arch_profile.has_fp8_native:
            return False, "arch has native FP8 TCs — no patch needed"
        return True, "FP8 model on arch without native FP8 — patch helps"

    # P67/P67b: TurboQuant multi-query kernel
    if patch_name in ("P67", "P67b"):
        if not model_profile.uses_tq:
            return False, "model doesn't use TurboQuant KV"
        if not model_profile.has_mtp:
            return False, "model doesn't use MTP spec-decode"
        return True, "TQ k8v4 + MTP K>0 = P67 fires"

    # Default: defer to patch's own gating
    return True, "patch has own gating; defer"


def patches_relevant_for_model(
    model_profile: Optional[GenesisModelProfile],
    arch_profile: Optional[Any] = None,
) -> tuple[str, ...]:
    """Return tuple of Genesis patch IDs that should be active for this model.

    Useful for `genesis patches diff` or operator scripts that auto-build
    launcher env from model+arch detection.
    """
    if model_profile is None:
        return ()
    relevant = []
    for patch_id in (
        "PN286", "PN293", "PN294", "PN296", "PN298", "PN299", "PN300",
        "PN302", "PN303", "P67", "P67b",
    ):
        ok, _ = should_apply_patch_for_model(patch_id, model_profile, arch_profile)
        if ok:
            relevant.append(patch_id)
    return tuple(relevant)
