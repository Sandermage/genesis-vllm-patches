# SPDX-License-Identifier: Apache-2.0
"""PN271 KV contract audit.

Runtime guard auditing spec-decode KV contract invariants. Lives at
spec_decode root (not probes/) per Task F §11.1 — operator decision
keeps runtime guards separate from diagnostic probes. Stays dormant
until the operator enables it via its env-flag. Resolves the Phase 3
relocation stash-pop conflict (old `integrations/gemma4/` path was
removed during the move).
"""
from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from typing import Any

log = logging.getLogger("genesis.spec_decode.pn271_spec_decode_kv_contract_audit")

GENESIS_PN271_MARKER = "Genesis PN271 SpecDecode KV contract audit"

_ENV_ENABLE = "GENESIS_ENABLE_PN271_KV_CONTRACT_AUDIT"
_APPLIED = False
_ORIGINAL_INIT_TENSORS = None
_DUMPED = False

# ----------------------- Compatibility verdicts -----------------------

VERDICT_EXACT = "EXACT_COPY"
VERDICT_GQA = "GQA_REPEAT"
VERDICT_LAYOUT = "LAYOUT_ADAPTER"
VERDICT_DEQUANT = "DEQUANT"
VERDICT_UNSUPPORTED = "UNSUPPORTED"
# PN271b — kernel-vs-storage / kernel-vs-layout contract mismatches.
VERDICT_KERNEL_STORAGE = "KERNEL_STORAGE_DTYPE_MISMATCH"
VERDICT_KERNEL_LAYOUT = "KERNEL_LAYOUT_CONTRACT_MISMATCH"


def _kernel_expects_quantized(impl_class: Any) -> bool | None:
    if not impl_class or impl_class == "<absent>":
        return None
    s = str(impl_class).lower()
    if "turboquant" in s:
        return True
    if "fp8" in s:
        return True
    if "flashattn" in s or "flashattention" in s:
        return False
    if "tritonattn" in s or "tritonattention" in s:
        return False
    return None


def _kernel_expected_layout(impl_class: Any) -> str | None:
    if not impl_class or impl_class == "<absent>":
        return None
    s = str(impl_class).lower()
    if "flashattn" in s or "flashattention" in s:
        return "HND"
    if "tritonattn" in s or "tritonattention" in s:
        return "NHD"
    if "turboquant" in s:
        return "NHD"
    return None


def _storage_is_native(kv_dtype_decl: Any, kv_dtype_real: Any) -> bool | None:
    """True if storage label is plain native, False if quantized."""
    s_decl = str(kv_dtype_decl or "").lower()
    if "turboquant" in s_decl or "fp8" in s_decl or (
            "quant" in s_decl and s_decl not in ("auto", "default", "none")):
        return False
    if s_decl in ("auto", "default", "none", "", "<absent>"):
        return True
    return None


def _safe_attr(obj: Any, name: str, default: Any = "<absent>") -> Any:
    try:
        return getattr(obj, name, default)
    except Exception as _e:
        return f"<err: {_e!r}>"


def _t_norm(t: Any) -> Any:
    try:
        if t is None:
            return None
        return float(t.float().norm().item())
    except Exception as _e:
        return f"<err: {_e!r}>"


def _t_head(t: Any, n: int = 3) -> Any:
    try:
        if t is None:
            return None
        flat = t.flatten()
        return flat[:n].tolist()
    except Exception as _e:
        return f"<err: {_e!r}>"


# ----------------------- Module discovery helpers -----------------------

def _unwrap(m: Any) -> Any:
    seen = set()
    for _ in range(12):
        if m is None or id(m) in seen:
            return m
        seen.add(id(m))
        for attr in ("runnable_model", "module", "model", "orig_module",
                     "_orig_mod", "wrapped", "inner"):
            inner = getattr(m, attr, None)
            if (inner is not None and inner is not m
                    and hasattr(inner, "named_modules")):
                m = inner
                break
        else:
            return m
    return m


def _find_module_by_prefix(root: Any, prefix_suffix: str) -> Any:
    """Find module whose registered name ENDS WITH prefix_suffix.

    e.g., prefix_suffix='language_model.model.layers.58.self_attn.attn'.
    """
    try:
        for name, mod in root.named_modules():
            if name.endswith(prefix_suffix):
                return mod
    except Exception as _e:
        log.warning("[PN271] _find_module_by_prefix(%s) failed: %s",
                    prefix_suffix, _e)
    return None


def _find_drafter_layers(runner: Any) -> list[tuple[int, Any]]:
    """Return [(layer_idx, drafter_self_attn_module), ...]."""
    out: list[tuple[int, Any]] = []
    drafter = getattr(runner, "drafter", None)
    if drafter is None:
        return out
    dmodel = getattr(drafter, "model", None)
    if dmodel is None:
        return out
    dmodel = _unwrap(dmodel)
    # Walk: ".layers.N.self_attn"
    try:
        for name, mod in dmodel.named_modules():
            if not name.endswith(".self_attn"):
                continue
            parts = name.split(".")
            if "layers" not in parts:
                continue
            try:
                idx = int(parts[parts.index("layers") + 1])
            except (ValueError, IndexError):
                continue
            out.append((idx, mod))
    except Exception as _e:
        log.warning("[PN271] drafter layer scan failed: %s", _e)
    return sorted(out, key=lambda x: x[0])


# ----------------------- Mapping providers -----------------------

class _MappingProvider:
    """Abstract: provides {drafter_layer_idx: target_full_attn_prefix}."""

    name = "abstract"

    def get_mapping(self, runner: Any, drafter_layers: list[tuple[int, Any]],
                    target_root: Any) -> dict[int, str]:
        raise NotImplementedError


class _Gemma4MappingProvider(_MappingProvider):
    """Re-runs vLLM's _setup_gemma4_kv_sharing logic in read-only mode."""

    name = "Gemma4"

    @classmethod
    def detect(cls, runner: Any) -> bool:
        try:
            mc = getattr(runner, "model_config", None)
            if mc is None:
                return False
            hf = getattr(mc, "hf_config", None)
            if hf is None:
                return False
            return "gemma" in type(hf).__name__.lower() or "gemma" in str(
                getattr(hf, "model_type", "")).lower()
        except Exception:
            return False

    def get_mapping(self, runner, drafter_layers, target_root):
        try:
            vllm_cfg = getattr(runner, "vllm_config", None)
            if vllm_cfg is None:
                # try to reach through drafter
                drafter = getattr(runner, "drafter", None)
                vllm_cfg = getattr(drafter, "vllm_config", None)
            if vllm_cfg is None:
                return {}

            target_hf = vllm_cfg.model_config.hf_config
            target_text = target_hf.get_text_config() if hasattr(
                target_hf, "get_text_config") else target_hf
            target_layer_types = getattr(target_text, "layer_types", []) or []
            target_num_kv_shared = getattr(
                target_text, "num_kv_shared_layers", 0) or 0
            num_non_shared = len(target_layer_types) - target_num_kv_shared

            type_to_target_indices: dict[str, list[int]] = defaultdict(list)
            for idx, lt in enumerate(target_layer_types[:num_non_shared]):
                type_to_target_indices[lt].append(idx)

            # Find target prefix from any drafter attn that already had
            # kv_sharing_target_layer_name. Fallback to 'language_model.model.layers'.
            target_prefix = None
            for _, draft_self_attn in drafter_layers:
                inner = getattr(draft_self_attn, "attn", None)
                if inner is None:
                    continue
                stored = getattr(inner, "kv_sharing_target_layer_name", None)
                if isinstance(stored, str) and ".layers." in stored:
                    target_prefix = stored.split(".layers.")[0] + ".layers"
                    break
            if target_prefix is None:
                # Search via root: find any attn with prefix "*.layers.*.self_attn.attn"
                try:
                    for n, _ in target_root.named_modules():
                        if (n.endswith(".self_attn.attn")
                                and ".layers." in n
                                and "draft" not in n):
                            target_prefix = n.rsplit(".layers.", 1)[0] + ".layers"
                            break
                except Exception:
                    pass
            if target_prefix is None:
                target_prefix = "language_model.model.layers"

            drafter_hf = vllm_cfg.speculative_config.draft_model_config.hf_config
            drafter_text = drafter_hf.get_text_config() if hasattr(
                drafter_hf, "get_text_config") else drafter_hf
            drafter_layer_types = getattr(
                drafter_text, "layer_types", []) or []

            mapping: dict[int, str] = {}
            for draft_idx, _ in drafter_layers:
                draft_lt = (
                    drafter_layer_types[draft_idx]
                    if draft_idx < len(drafter_layer_types)
                    else "full_attention"
                )
                candidates = type_to_target_indices.get(draft_lt, [])
                if not candidates:
                    log.warning(
                        "[PN271] no target candidate of type '%s' for "
                        "drafter[%d]", draft_lt, draft_idx,
                    )
                    continue
                target_idx = candidates[-1]
                mapping[draft_idx] = (
                    f"{target_prefix}.{target_idx}.self_attn.attn"
                )
            return mapping
        except Exception as _e:
            log.warning("[PN271] Gemma4 mapping derivation failed: %s", _e)
            return {}


# ----------------------- Per-pair audit -----------------------

def _collect_attn_facts(attn: Any) -> dict[str, Any]:
    """Field collection for one Attention (inner .attn) module."""
    facts: dict[str, Any] = {}
    facts["class"] = type(attn).__qualname__ if attn is not None else None
    if attn is None:
        return facts
    for k in ("num_kv_heads", "num_heads", "head_size", "head_dim",
             "scale", "logits_soft_cap", "kv_cache_dtype",
             "kv_sharing_target_layer_name", "sliding_window"):
        facts[k] = _safe_attr(attn, k)
    # impl class (post-init bound)
    impl = _safe_attr(attn, "impl")
    facts["impl_class"] = (type(impl).__qualname__
                           if impl is not None and impl != "<absent>" else None)
    # backend marker
    facts["attn_backend"] = _safe_attr(attn, "attn_backend")
    # kv_cache (set after bind)
    kv = _safe_attr(attn, "kv_cache")
    if kv is None or kv == "<absent>":
        facts["kv_cache_shape"] = None
        facts["kv_cache_dtype_real"] = None
        facts["kv_cache_layout"] = "unknown"
    else:
        try:
            facts["kv_cache_shape"] = tuple(kv.shape)
            facts["kv_cache_dtype_real"] = str(kv.dtype)
            if len(kv.shape) >= 2 and int(kv.shape[0]) == 2:
                facts["kv_cache_layout"] = "HND"
            elif len(kv.shape) >= 2 and int(kv.shape[1]) == 2:
                facts["kv_cache_layout"] = "NHD"
            else:
                facts["kv_cache_layout"] = "unknown"
        except Exception:
            facts["kv_cache_shape"] = "<err>"
    return facts


def _collect_self_attn_facts(self_attn: Any) -> dict[str, Any]:
    """Field collection for the outer self_attn (Gemma4MTPAttention or
    target self_attn). Captures q_norm, k_norm, RoPE, projections."""
    facts: dict[str, Any] = {}
    if self_attn is None:
        return facts
    facts["self_attn_class"] = type(self_attn).__qualname__
    facts["q_norm_weight_norm"] = _t_norm(
        _safe_attr(getattr(self_attn, "q_norm", None), "weight", None)
    ) if hasattr(self_attn, "q_norm") else None
    facts["k_norm_weight_norm"] = _t_norm(
        _safe_attr(getattr(self_attn, "k_norm", None), "weight", None)
    ) if hasattr(self_attn, "k_norm") else None
    rope = getattr(self_attn, "rotary_emb", None)
    if rope is not None:
        facts["rope_class"] = type(rope).__qualname__
        facts["rope_base"] = _safe_attr(rope, "base")
        facts["rope_max_position_embeddings"] = _safe_attr(
            rope, "max_position_embeddings"
        )
        facts["rope_rotary_dim"] = _safe_attr(rope, "rotary_dim")
        facts["rope_inv_freq_head"] = _t_head(
            _safe_attr(rope, "inv_freq", None), 3
        )
    facts["q_proj_present"] = getattr(self_attn, "q_proj", None) is not None
    facts["k_proj_present"] = getattr(self_attn, "k_proj", None) is not None
    facts["v_proj_present"] = getattr(self_attn, "v_proj", None) is not None
    facts["qkv_proj_present"] = getattr(
        self_attn, "qkv_proj", None) is not None
    facts["o_proj_present"] = getattr(self_attn, "o_proj", None) is not None
    return facts


def _audit_pair(
    drafter_idx: int,
    drafter_self_attn: Any,
    target_prefix: str,
    target_self_attn: Any,
) -> dict[str, Any]:
    drafter_inner = getattr(drafter_self_attn, "attn", None)
    target_inner = getattr(target_self_attn, "attn", None) if target_self_attn else None

    da = _collect_attn_facts(drafter_inner)
    ta = _collect_attn_facts(target_inner)
    dsa = _collect_self_attn_facts(drafter_self_attn)
    tsa = _collect_self_attn_facts(target_self_attn)

    # --- comparisons + divergence reasons
    divergences: list[str] = []
    verdicts: list[str] = []

    def _num(x):
        try:
            return int(x)
        except (TypeError, ValueError):
            try:
                return float(x)
            except (TypeError, ValueError):
                return None

    d_kvh = _num(da.get("num_kv_heads"))
    t_kvh = _num(ta.get("num_kv_heads"))
    d_hs = _num(da.get("head_size") or da.get("head_dim"))
    t_hs = _num(ta.get("head_size") or ta.get("head_dim"))

    if d_hs is not None and t_hs is not None and d_hs != t_hs:
        divergences.append(
            f"head_size mismatch drafter={d_hs} target={t_hs}"
        )
        verdicts.append(VERDICT_UNSUPPORTED)

    if d_kvh is not None and t_kvh is not None:
        if d_kvh == t_kvh:
            pass  # ok
        elif d_kvh > t_kvh and d_kvh % t_kvh == 0:
            verdicts.append(VERDICT_GQA)
            divergences.append(
                f"num_kv_heads ratio drafter={d_kvh} target={t_kvh} "
                f"(GQA repeat={d_kvh // t_kvh})"
            )
        else:
            divergences.append(
                f"num_kv_heads not divisor: drafter={d_kvh} target={t_kvh}"
            )
            verdicts.append(VERDICT_UNSUPPORTED)

    # Scale comparison
    d_scale = _num(da.get("scale"))
    t_scale = _num(ta.get("scale"))
    if d_scale is not None and t_scale is not None:
        # Tolerance: ratio within 1.0% considered equal
        if d_scale != 0 and t_scale != 0:
            ratio = d_scale / t_scale
            if not (0.99 <= ratio <= 1.01):
                expected_scale = (
                    1.0 / math.sqrt(d_hs)
                    if d_hs is not None and d_hs > 0
                    else None
                )
                divergences.append(
                    f"scale mismatch drafter={d_scale} target={t_scale} "
                    f"(ratio={ratio:.4f}; expected 1/sqrt(head_size)={expected_scale})"
                )
                verdicts.append(VERDICT_UNSUPPORTED)

    # Soft-cap comparison
    d_sc = da.get("logits_soft_cap")
    t_sc = ta.get("logits_soft_cap")
    if d_sc != t_sc and d_sc not in ("<absent>", None) or t_sc not in ("<absent>", None):
        divergences.append(
            f"logits_soft_cap drafter={d_sc} target={t_sc}"
        )
        if d_sc != t_sc:
            verdicts.append(VERDICT_UNSUPPORTED)

    # RoPE
    d_rb = dsa.get("rope_base")
    t_rb = tsa.get("rope_base")
    if d_rb != t_rb:
        divergences.append(
            f"rope_base drafter={d_rb} target={t_rb}"
        )
        verdicts.append(VERDICT_UNSUPPORTED)
    d_inv = dsa.get("rope_inv_freq_head")
    t_inv = tsa.get("rope_inv_freq_head")
    if d_inv != t_inv:
        divergences.append(
            f"rope_inv_freq[:3] drafter={d_inv} target={t_inv}"
        )

    # Layout
    d_layout = da.get("kv_cache_layout")
    t_layout = ta.get("kv_cache_layout")
    if (d_layout and t_layout and d_layout != t_layout
            and "unknown" not in (d_layout, t_layout)):
        divergences.append(
            f"kv_cache layout drafter={d_layout} target={t_layout}"
        )
        verdicts.append(VERDICT_LAYOUT)

    # Quantization (kv_cache_dtype)
    d_q = str(da.get("kv_cache_dtype") or "")
    t_q = str(ta.get("kv_cache_dtype") or "")
    if "turboquant" in t_q.lower() or "quant" in t_q.lower():
        if "turboquant" not in d_q.lower() and "quant" not in d_q.lower():
            divergences.append(
                f"target is quantized ({t_q}); drafter expects native ({d_q})"
            )
            verdicts.append(VERDICT_DEQUANT)

    # PN271b — consumer-kernel-vs-source-storage contract.
    d_impl = da.get("impl_class")
    t_impl = ta.get("impl_class")
    d_kernel_q = _kernel_expects_quantized(d_impl)
    d_kernel_layout = _kernel_expected_layout(d_impl)
    t_storage_native = _storage_is_native(
        ta.get("kv_cache_dtype"), ta.get("kv_cache_dtype_real"))
    # Storage native + drafter kernel expects quantized -> MISREAD.
    if d_kernel_q is True and t_storage_native is True:
        divergences.append(
            f"KERNEL_STORAGE_DTYPE_MISMATCH: drafter.impl={d_impl!r} "
            f"expects quantized bytes; target storage declared "
            f"{ta.get('kv_cache_dtype')!r} (native).  Drafter would "
            f"misread target's bf16 cache as TQ-packed."
        )
        verdicts.append(VERDICT_KERNEL_STORAGE)
    elif d_kernel_q is False and t_storage_native is False:
        divergences.append(
            f"KERNEL_STORAGE_DTYPE_MISMATCH: drafter.impl={d_impl!r} "
            f"expects native bytes; target storage declared "
            f"{ta.get('kv_cache_dtype')!r} (quantized)."
        )
        verdicts.append(VERDICT_KERNEL_STORAGE)
    # Kernel layout contract.
    if d_kernel_layout and ta.get("kv_cache_layout") not in (
            "unknown", None, "<absent>"):
        if d_kernel_layout != ta.get("kv_cache_layout"):
            divergences.append(
                f"KERNEL_LAYOUT_CONTRACT_MISMATCH: drafter.impl={d_impl!r} "
                f"expects {d_kernel_layout}; target storage layout="
                f"{ta.get('kv_cache_layout')}"
            )
            verdicts.append(VERDICT_KERNEL_LAYOUT)

    # Determine final verdict (worst case wins).
    # Kernel/storage mismatches outrank adapter-required verdicts —
    # the consumer kernel would misread bytes.
    if VERDICT_UNSUPPORTED in verdicts:
        final = VERDICT_UNSUPPORTED
    elif VERDICT_KERNEL_STORAGE in verdicts:
        final = VERDICT_KERNEL_STORAGE
    elif VERDICT_KERNEL_LAYOUT in verdicts:
        final = VERDICT_KERNEL_LAYOUT
    elif VERDICT_DEQUANT in verdicts:
        final = VERDICT_DEQUANT
    elif VERDICT_LAYOUT in verdicts and VERDICT_GQA in verdicts:
        final = "LAYOUT_ADAPTER + GQA_REPEAT"
    elif VERDICT_LAYOUT in verdicts:
        final = VERDICT_LAYOUT
    elif VERDICT_GQA in verdicts:
        final = VERDICT_GQA
    else:
        final = VERDICT_EXACT

    return {
        "drafter_idx": drafter_idx,
        "target_prefix": target_prefix,
        "drafter_attn": da,
        "target_attn": ta,
        "drafter_self_attn": dsa,
        "target_self_attn": tsa,
        "divergences": divergences,
        "verdict": final,
    }


def _run_audit(runner: Any) -> None:
    log.warning("[PN271] === SpecDecode KV contract audit BEGIN ===")
    log.warning("[PN271] runner_class=%s", type(runner).__qualname__)

    drafter_layers = _find_drafter_layers(runner)
    log.warning("[PN271] drafter layers found: %s",
                [(i, type(m).__qualname__) for i, m in drafter_layers])
    if not drafter_layers:
        log.warning("[PN271] no drafter layers — abort")
        log.warning("[PN271] === audit END (no_drafter) ===")
        return

    target_root_raw = getattr(runner, "model", None)
    target_root = _unwrap(target_root_raw) if target_root_raw is not None else None

    # Pick mapping provider
    if _Gemma4MappingProvider.detect(runner):
        provider: _MappingProvider = _Gemma4MappingProvider()
    else:
        provider = _Gemma4MappingProvider()  # only one implemented for now
        log.warning("[PN271] no model-specific provider matched; falling "
                    "back to Gemma4-style mapping (may be wrong)")
    log.warning("[PN271] mapping provider=%s", provider.name)

    mapping = provider.get_mapping(runner, drafter_layers, target_root)
    log.warning("[PN271] mapping: %s", mapping)

    overall_verdicts: list[str] = []
    pair_reports: list[dict[str, Any]] = []
    for draft_idx, draft_self_attn in drafter_layers:
        target_prefix = mapping.get(draft_idx)
        target_self_attn = None
        if target_prefix and target_root is not None:
            # target_prefix is like ".../layers.N.self_attn.attn"
            # we want the .self_attn (parent of .attn)
            parent_prefix = target_prefix.rsplit(".attn", 1)[0]
            target_self_attn = _find_module_by_prefix(target_root, parent_prefix)
        report = _audit_pair(
            draft_idx, draft_self_attn, target_prefix or "<unmapped>",
            target_self_attn,
        )
        pair_reports.append(report)
        log.warning("[PN271] --- drafter[%d] -> %s VERDICT=%s ---",
                    draft_idx, target_prefix, report["verdict"])
        log.warning("[PN271]   drafter.attn facts=%s", report["drafter_attn"])
        log.warning("[PN271]   target.attn facts=%s", report["target_attn"])
        log.warning("[PN271]   drafter.self_attn facts=%s",
                    report["drafter_self_attn"])
        log.warning("[PN271]   target.self_attn facts=%s",
                    report["target_self_attn"])
        if report["divergences"]:
            for d in report["divergences"]:
                log.warning("[PN271]   DIVERGE: %s", d)
        overall_verdicts.append(report["verdict"])

    # --- final summary ---
    log.warning("[PN271] ----- AUDIT SUMMARY -----")
    for r in pair_reports:
        log.warning("[PN271]  drafter[%d] -> %s : %s : %d divergences",
                    r["drafter_idx"], r["target_prefix"], r["verdict"],
                    len(r["divergences"]))
    if VERDICT_UNSUPPORTED in overall_verdicts:
        overall = VERDICT_UNSUPPORTED
    elif VERDICT_KERNEL_STORAGE in overall_verdicts:
        overall = VERDICT_KERNEL_STORAGE
    elif VERDICT_KERNEL_LAYOUT in overall_verdicts:
        overall = VERDICT_KERNEL_LAYOUT
    elif VERDICT_DEQUANT in overall_verdicts:
        overall = VERDICT_DEQUANT
    elif any("LAYOUT_ADAPTER" in v for v in overall_verdicts) and any(
            "GQA" in v for v in overall_verdicts):
        overall = "LAYOUT_ADAPTER + GQA_REPEAT"
    elif any("LAYOUT_ADAPTER" in v for v in overall_verdicts):
        overall = VERDICT_LAYOUT
    elif any("GQA" in v for v in overall_verdicts):
        overall = VERDICT_GQA
    else:
        overall = VERDICT_EXACT
    log.warning("[PN271] OVERALL VERDICT: %s", overall)
    if overall in (VERDICT_KERNEL_STORAGE, VERDICT_KERNEL_LAYOUT):
        recommendation = (
            "MTP UNSAFE — consumer kernel will misread bytes. NOT "
            "overridable; change backend route or storage layout."
        )
    elif overall == VERDICT_UNSUPPORTED:
        recommendation = "MTP UNSAFE — disable"
    elif "ADAPTER" in overall or "GQA" in overall or "DEQUANT" in overall:
        recommendation = "MTP may be enabled with adapter"
    else:
        recommendation = "MTP safe — exact bridge sufficient"
    log.warning("[PN271] PRODUCTION RECOMMENDATION: %s", recommendation)
    log.warning("[PN271] === audit END ===")


# ----------------------- Patch glue -----------------------

def apply() -> tuple[str, str]:
    global _APPLIED, _ORIGINAL_INIT_TENSORS

    if os.environ.get(_ENV_ENABLE, "").strip().lower() not in (
            "1", "true", "yes", "on"):
        return "skipped", f"PN271 disabled (set {_ENV_ENABLE}=1)"
    if _APPLIED:
        return "applied", "PN271 already installed"

    log.warning("[PN271] apply() entered")

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as e:  # noqa: BLE001
        log.warning("[PN271] SKIP: GPUModelRunner not importable: %s", e)
        return "skipped", f"GPUModelRunner not importable: {e!r}"

    if not hasattr(GPUModelRunner, "initialize_kv_cache_tensors"):
        return "skipped", "GPUModelRunner.initialize_kv_cache_tensors missing"

    original = GPUModelRunner.initialize_kv_cache_tensors
    if getattr(original, "_genesis_pn271_wrapped", False):
        _APPLIED = True
        return "applied", "initialize_kv_cache_tensors already wrapped"
    _ORIGINAL_INIT_TENSORS = original

    def _wrapped(self, kv_cache_config, kernel_block_sizes):
        result = original(self, kv_cache_config, kernel_block_sizes)
        global _DUMPED
        if not _DUMPED:
            try:
                _run_audit(self)
                _DUMPED = True
            except Exception as e:  # noqa: BLE001
                log.warning("[PN271] audit pass failed: %s", e)
        return result

    _wrapped._genesis_pn271_wrapped = True  # type: ignore[attr-defined]
    GPUModelRunner.initialize_kv_cache_tensors = _wrapped  # type: ignore[method-assign]
    _APPLIED = True
    log.warning(
        "[PN271] INSTALLED: SpecDecode KV contract audit will run "
        "once on first initialize_kv_cache_tensors call."
    )
    return "applied", "PN271 installed (audit-only)"


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED, _ORIGINAL_INIT_TENSORS, _DUMPED
    if not _APPLIED or _ORIGINAL_INIT_TENSORS is None:
        return False
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        GPUModelRunner.initialize_kv_cache_tensors = _ORIGINAL_INIT_TENSORS  # type: ignore[method-assign]
    except ImportError:
        return False
    _APPLIED = False
    _ORIGINAL_INIT_TENSORS = None
    _DUMPED = False
    return True


__all__ = ["GENESIS_PN271_MARKER", "apply", "is_applied", "revert"]
