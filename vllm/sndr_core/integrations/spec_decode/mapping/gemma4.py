# SPDX-License-Identifier: Apache-2.0
"""mapping/gemma4 — Gemma 4 MTP drafter→target mapping provider.

Re-runs the logic of vLLM's
``Gemma4Proposer._setup_gemma4_kv_sharing`` in read-only mode:

  for each drafter layer:
      candidates = [i for i,t in target_layer_types[:non_kv_shared_cutoff]
                    if t == drafter_layer_types[i]]
      target_idx = candidates[-1]    # last layer of same type

The result is a list of LayerMapping with the live nn.Module handles
for both sides. Errors are non-fatal: provider returns [] when it
can't reach into the model tree.

Provenance: extracted from
``integrations/gemma4/pn271_spec_decode_kv_contract_audit.py``
2026-05-20 per architectural directive (PN273).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from .base import LayerMapping, MappingProvider

log = logging.getLogger("genesis.spec_decode.mapping.gemma4")


def _walk_chain(start: Any) -> list[Any]:
    chain: list[Any] = []
    cur = start
    for _ in range(12):
        if cur is None:
            break
        chain.append(cur)
        nxt = None
        for attr in ("runnable_model", "module", "orig_module", "_orig_mod",
                     "wrapped", "inner", "model"):
            cand = getattr(cur, attr, None)
            if (cand is not None and cand is not cur
                    and hasattr(cand, "named_modules")):
                nxt = cand
                break
        if nxt is None:
            break
        cur = nxt
    return chain


def _find_drafter_predictor(runner: Any) -> Any:
    """Find Gemma4MultiTokenPredictor by signature."""
    drafter = getattr(runner, "drafter", None)
    dmodel = getattr(drafter, "model", None) if drafter is not None else None
    for cand in _walk_chain(dmodel):
        if (hasattr(cand, "layers")
                and hasattr(cand, "embed_tokens")
                and hasattr(cand, "pre_projection")):
            return cand
    return None


def _find_target_root(runner: Any) -> Any:
    rm = getattr(runner, "model", None)
    if rm is None:
        return None
    # Unwrap CUDAGraphWrapper / torch.compile / etc.
    for cand in _walk_chain(rm):
        # Pick the deepest module that has named_modules
        pass
    chain = _walk_chain(rm)
    return chain[-1] if chain else rm


def _find_target_attn_module(target_root: Any, target_prefix: str) -> Any:
    """target_prefix like 'language_model.model.layers.58.self_attn.attn'.
    We want the parent '.self_attn' module.
    """
    if target_root is None or not target_prefix:
        return None
    parent_prefix = target_prefix.rsplit(".attn", 1)[0]
    try:
        for name, mod in target_root.named_modules():
            if name.endswith(parent_prefix):
                return mod
    except Exception as _e:
        log.warning("[mapping.gemma4] named_modules walk failed: %s", _e)
    return None


def _is_gemma4_config(hf_config: Any) -> bool:
    if hf_config is None:
        return False
    cls_name = type(hf_config).__qualname__.lower()
    model_type = str(getattr(hf_config, "model_type", "")).lower()
    return ("gemma" in cls_name) or ("gemma" in model_type)


def _is_mtp_method(spec_cfg: Any) -> bool:
    if spec_cfg is None:
        return False
    method = getattr(spec_cfg, "method", None)
    if method is None:
        return False
    return str(method).strip().lower() == "mtp"


def _is_quantized_kv(model_cfg: Any) -> bool:
    if model_cfg is None:
        return False
    dt = getattr(model_cfg, "kv_cache_dtype", None) or getattr(
        model_cfg, "kv_cache_dtype_str", None)
    if dt is None:
        return False
    s = str(dt).lower()
    return ("turboquant" in s) or ("fp8" in s) or (
        "quant" in s and s not in ("auto", "default", "none")
    )


class Gemma4MappingProvider(MappingProvider):
    name = "Gemma4"

    def supports(self, runner: Any) -> bool:
        try:
            mc = getattr(runner, "model_config", None)
            hf = getattr(mc, "hf_config", None) if mc is not None else None
            return _is_gemma4_config(hf)
        except Exception:
            return False

    def supports_config(self, vllm_config: Any) -> bool:
        """Match Gemma 4 ANY-form MTP at config time.

        We engage the guard for Gemma 4 + MTP. The actual verdict in
        ``evaluate_from_config`` differentiates between native bf16
        (EXACT_COPY -> allow) and quantized target KV
        (FUNCTIONAL_UNVERIFIED -> deny by default).
        """
        try:
            mc = getattr(vllm_config, "model_config", None)
            spec_cfg = getattr(vllm_config, "speculative_config", None)
            if mc is None or spec_cfg is None:
                return False
            hf = getattr(mc, "hf_config", None)
            return _is_gemma4_config(hf) and _is_mtp_method(spec_cfg)
        except Exception:
            return False

    def evaluate_from_config(self, vllm_config: Any) -> tuple[Any, str]:
        """Return (Verdict, reason) for a matched Gemma 4 MTP config.

        Decision rule:
          - target KV is quantized (turboquant_* / fp8) ->
            ADAPTER_STRUCTURAL_OK_FUNCTIONAL_UNVERIFIED
            (backend mix breaks physical kv_sharing; bridge required;
            runtime acceptance not yet proven)
          - target KV is native bf16/fp16 ->
            EXACT_COPY (kv_sharing works as designed; no bridge)
        """
        from ..kv_contract import Verdict
        mc = getattr(vllm_config, "model_config", None)
        if _is_quantized_kv(mc):
            dt = getattr(mc, "kv_cache_dtype", None) or getattr(
                mc, "kv_cache_dtype_str", "?")
            return (
                Verdict.ADAPTER_STRUCTURAL_OK_FUNCTIONAL_UNVERIFIED,
                f"Gemma4 MTP with quantized target KV (kv_cache_dtype={dt!r}) "
                f"breaks physical kv_sharing — bridge required; runtime "
                f"acceptance not validated for this configuration.",
            )
        return (
            Verdict.EXACT_COPY,
            "Gemma4 MTP with native KV — physical kv_sharing works as "
            "designed.",
        )

    def get_mapping(self, runner: Any) -> list[LayerMapping]:
        predictor = _find_drafter_predictor(runner)
        if predictor is None:
            log.warning("[mapping.gemma4] no Gemma4MultiTokenPredictor — empty")
            return []

        # Build drafter layer index -> self_attn module list
        drafter_layers: list[tuple[int, Any]] = []
        try:
            layers = list(predictor.layers)
            for idx, layer in enumerate(layers):
                sa = getattr(layer, "self_attn", None)
                if sa is not None:
                    drafter_layers.append((idx, sa))
        except Exception as _e:
            log.warning("[mapping.gemma4] drafter layer iter failed: %s", _e)
            return []

        # Resolve target layer indices via vLLM's own rule
        try:
            vllm_cfg = (getattr(runner, "vllm_config", None)
                        or getattr(getattr(runner, "drafter", None),
                                   "vllm_config", None))
            if vllm_cfg is None:
                log.warning("[mapping.gemma4] vllm_config not found — empty")
                return []
            target_hf = vllm_cfg.model_config.hf_config
            target_text = (target_hf.get_text_config()
                           if hasattr(target_hf, "get_text_config")
                           else target_hf)
            target_layer_types = list(getattr(target_text, "layer_types", []))
            target_num_kv_shared = int(getattr(
                target_text, "num_kv_shared_layers", 0) or 0)
            num_non_shared = len(target_layer_types) - target_num_kv_shared

            type_to_target_indices: dict[str, list[int]] = defaultdict(list)
            for idx, lt in enumerate(target_layer_types[:num_non_shared]):
                type_to_target_indices[lt].append(idx)

            drafter_hf = vllm_cfg.speculative_config.draft_model_config.hf_config
            drafter_text = (drafter_hf.get_text_config()
                            if hasattr(drafter_hf, "get_text_config")
                            else drafter_hf)
            drafter_layer_types = list(getattr(drafter_text, "layer_types", []))
        except Exception as _e:
            log.warning("[mapping.gemma4] config read failed: %s", _e)
            return []

        target_root = _find_target_root(runner)

        # Discover target prefix by inspecting the live target module
        # tree for an existing self_attn.attn matching the layer regex.
        target_prefix_base = None
        try:
            if target_root is not None:
                for name, _mod in target_root.named_modules():
                    if (name.endswith(".self_attn.attn")
                            and ".layers." in name
                            and "draft" not in name):
                        target_prefix_base = name.rsplit(
                            ".layers.", 1)[0] + ".layers"
                        break
        except Exception:
            pass
        if target_prefix_base is None:
            target_prefix_base = "language_model.model.layers"

        results: list[LayerMapping] = []
        for draft_idx, draft_sa in drafter_layers:
            draft_lt = (drafter_layer_types[draft_idx]
                        if draft_idx < len(drafter_layer_types)
                        else "full_attention")
            candidates = type_to_target_indices.get(draft_lt, [])
            if not candidates:
                log.warning(
                    "[mapping.gemma4] no target candidate of type %r for "
                    "drafter[%d]", draft_lt, draft_idx,
                )
                continue
            target_idx = candidates[-1]
            target_prefix = (
                f"{target_prefix_base}.{target_idx}.self_attn.attn"
            )
            target_sa = _find_target_attn_module(target_root, target_prefix)
            results.append(LayerMapping(
                drafter_idx=draft_idx,
                target_full_prefix=target_prefix,
                drafter_self_attn=draft_sa,
                target_self_attn=target_sa,
            ))
        return results


__all__ = ["Gemma4MappingProvider"]
