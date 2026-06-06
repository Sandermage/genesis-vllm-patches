# SPDX-License-Identifier: Apache-2.0
"""PN270 drafter KV-proj audit.

Diagnostic probe for drafter KV projection alignment. Stays dormant until the operator
enables it via its env-flag; canonical location is this file itself.
Resolves the Phase 3 relocation stash-pop conflict (old
`integrations/gemma4/` path was removed during the move).
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("genesis.spec_decode.pn270_drafter_kv_proj_audit")

GENESIS_PN270_MARKER = "Genesis PN270 drafter K/V projection audit"

_ENV_ENABLE = "GENESIS_ENABLE_PN270_DRAFTER_KV_PROJ_AUDIT"
_APPLIED = False
_ORIGINAL_INIT_TENSORS = None
_DUMPED = False  # one-shot guard

DRAFTER_SELF_ATTN_PREFIX = "draft_model.layers."
DRAFTER_SELF_ATTN_SUFFIX = ".self_attn"

def _unwrap_model(m: Any) -> Any:
    """Strip common wrappers (CUDAGraphWrapper, torch.compile, etc.) to
    reach the underlying nn.Module that owns the actual layers."""
    seen = set()
    for _ in range(12):  # bounded
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

def _find_drafter_attached(runner: Any) -> list[tuple[str, Any]]:
    """Search GPUModelRunner for the drafter sub-model. Returns
    list of (root_attr_path, root_module) candidates to walk."""
    candidates: list[tuple[str, Any]] = []
    # Direct: runner.drafter.model
    drafter = getattr(runner, "drafter", None)
    if drafter is not None:
        for attr in ("model", "draft_model", "runnable_model"):
            cand = getattr(drafter, attr, None)
            if cand is not None and hasattr(cand, "named_modules"):
                candidates.append((f"drafter.{attr}", cand))
    # Other common proposer attribute names on the runner
    for attr in ("proposer", "speculative_decoder", "spec_decode_proposer"):
        prop = getattr(runner, attr, None)
        if prop is not None:
            for ma in ("model", "draft_model"):
                cand = getattr(prop, ma, None)
                if cand is not None and hasattr(cand, "named_modules"):
                    candidates.append((f"{attr}.{ma}", cand))
    # As fallback, look at runner.model AFTER unwrap (target model might
    # contain draft_model as a child for some MTP integrations).
    rm = getattr(runner, "model", None)
    if rm is not None:
        candidates.append(("model", rm))
    return candidates

def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )

def _safe(value: Any, default: str = "<?>") -> str:
    try:
        return repr(value)
    except Exception:
        return default

def _module_attrs(mod: Any, names: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for n in names:
        try:
            out[n] = getattr(mod, n, "<absent>")
        except Exception as _e:
            out[n] = f"<err: {_e!r}>"
    return out

def _describe_param(proj_name: str, full_name: str, proj: Any) -> dict[str, Any]:
    """Return a dict summarizing one projection module."""
    if proj is None:
        return {"name": proj_name, "status": "MISSING"}
    info: dict[str, Any] = {
        "name": proj_name,
        "status": "present",
        "class": type(proj).__qualname__,
    }
    weight = getattr(proj, "weight", None)
    if weight is None:
        info["weight"] = "<absent>"
        return info
    try:
        info["weight_shape"] = tuple(weight.shape)
        info["weight_dtype"] = str(weight.dtype)
        info["weight_data_ptr"] = int(weight.data_ptr())
        info["weight_numel"] = int(weight.numel())
        try:
            info["weight_norm"] = float(weight.float().norm().item())
        except Exception as _e:
            info["weight_norm"] = f"<err: {_e!r}>"
        # Try to detect a bias too
        bias = getattr(proj, "bias", None)
        if bias is not None:
            try:
                info["bias_shape"] = tuple(bias.shape)
                info["bias_norm"] = float(bias.float().norm().item())
            except Exception:
                info["bias"] = "<err>"
    except Exception as _e:
        info["err"] = f"{_e!r}"
    return info

def _dump_drafter_audit(runner: Any) -> None:
    """Find drafter sub-model on the runner and log full audit."""
    import torch

    log.warning("[PN270] === Drafter K/V projection audit BEGIN ===")
    log.warning("[PN270] runner_class=%s", type(runner).__qualname__)

    # List attributes on the runner that might hold the drafter model
    runner_attrs = sorted([
        a for a in dir(runner)
        if not a.startswith("_") and any(
            k in a.lower() for k in
            ("draft", "spec", "propos", "model")
        )
    ])
    log.warning("[PN270] runner candidate attrs: %s", runner_attrs)

    seed_candidates = _find_drafter_attached(runner)
    log.warning("[PN270] seed candidates: %s",
                [(p, type(m).__qualname__) for p, m in seed_candidates])

    candidates: list[tuple[str, Any]] = []
    for seed_path, seed_model in seed_candidates:
        unwrapped = _unwrap_model(seed_model)
        log.warning("[PN270] walking %s -> unwrapped=%s",
                    seed_path, type(unwrapped).__qualname__)
        try:
            for name, module in unwrapped.named_modules():
                # Accept paths whose tail looks like
                # "...draft_model.layers.N.self_attn" OR plain
                # "layers.N.self_attn" (when seed is the drafter root).
                if name.endswith(DRAFTER_SELF_ATTN_SUFFIX) and (
                    DRAFTER_SELF_ATTN_PREFIX in name
                    or name.startswith("layers.")
                    or ".layers." in name
                ):
                    candidates.append((f"{seed_path}.{name}", module))
        except Exception as _e:
            log.warning("[PN270] walk %s failed: %s", seed_path, _e)

    log.warning("[PN270] found %d candidate self_attn modules: %s",
                len(candidates), [n for n, _ in candidates])

    proj_attrs = ("q_proj", "k_proj", "v_proj", "qkv_proj", "kv_proj", "o_proj")
    attn_attrs = (
        "kv_sharing_target_layer_name",
        "attn_type",
        "num_kv_heads",
        "num_heads",
        "head_size",
        "head_dim",
        "kv_cache_dtype",
        "scale",
        "use_qk_norm",
        "is_sliding",
        "sliding_window",
    )

    for layer_name, self_attn in candidates:
        log.warning("[PN270] --- %s ---", layer_name)
        log.warning("[PN270] %s class=%s", layer_name,
                    type(self_attn).__qualname__)
        # self_attn attributes
        attrs = _module_attrs(self_attn, attn_attrs)
        log.warning("[PN270] %s self_attn attrs=%s", layer_name, attrs)
        # inner .attn (the Attention layer) attrs
        inner_attn = getattr(self_attn, "attn", None)
        if inner_attn is not None:
            inner_attrs = _module_attrs(inner_attn, attn_attrs)
            log.warning(
                "[PN270] %s.attn class=%s attrs=%s",
                layer_name, type(inner_attn).__qualname__, inner_attrs,
            )

        # Per-projection inspect
        proj_info: dict[str, dict[str, Any]] = {}
        for p in proj_attrs:
            proj = getattr(self_attn, p, None)
            proj_info[p] = _describe_param(p, layer_name, proj)
            log.warning("[PN270] %s.%s -> %s", layer_name, p, proj_info[p])

        # Tied-storage / allclose for k vs v
        kp = getattr(self_attn, "k_proj", None)
        vp = getattr(self_attn, "v_proj", None)
        if kp is not None and vp is not None:
            kp_w = getattr(kp, "weight", None)
            vp_w = getattr(vp, "weight", None)
            if kp_w is not None and vp_w is not None:
                tied = int(kp_w.data_ptr()) == int(vp_w.data_ptr())
                same_shape = kp_w.shape == vp_w.shape
                allclose = "<n/a>"
                if same_shape:
                    try:
                        allclose = bool(torch.allclose(
                            kp_w.float(), vp_w.float(), atol=1e-8
                        ))
                    except Exception as _e:
                        allclose = f"<err: {_e!r}>"
                log.warning(
                    "[PN270] %s k_proj vs v_proj: tied_storage=%s "
                    "same_shape=%s allclose=%s",
                    layer_name, tied, same_shape, allclose,
                )
        # Fused qkv: split-check
        qkv = getattr(self_attn, "qkv_proj", None)
        if qkv is not None:
            w = getattr(qkv, "weight", None)
            if w is not None:
                log.warning(
                    "[PN270] %s.qkv_proj fused weight shape=%s -- "
                    "split mismatch impossible (single weight)",
                    layer_name, tuple(w.shape),
                )

    # ---- state_dict key audit across all candidate models ----
    for seed_path, seed_model in seed_candidates:
        unwrapped = _unwrap_model(seed_model)
        try:
            sd_keys = list(unwrapped.state_dict().keys())
        except Exception as _e:
            log.warning("[PN270] state_dict(%s) failed: %s", seed_path, _e)
            continue
        drafter_keys = [
            k for k in sd_keys
            if "draft" in k.lower() or k.startswith("layers.")
        ]
        kv_keys = [
            k for k in drafter_keys
            if any(p in k for p in (".k_proj.", ".v_proj.", ".q_proj.",
                                    ".qkv_proj.", ".kv_proj.", ".o_proj."))
        ]
        log.warning(
            "[PN270] %s.state_dict total=%d draft-like=%d K/V/Q keys "
            "(count=%d): %s",
            seed_path, len(sd_keys), len(drafter_keys), len(kv_keys),
            kv_keys[:32] + (["..."] if len(kv_keys) > 32 else []),
        )

    log.warning("[PN270] === Drafter K/V projection audit END ===")

def apply() -> tuple[str, str]:
    global _APPLIED, _ORIGINAL_INIT_TENSORS

    if not _env_enabled():
        return "skipped", f"PN270 disabled (set {_ENV_ENABLE}=1)"
    if _APPLIED:
        return "applied", "PN270 already installed"

    log.warning("[PN270] apply() entered")

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as e:  # noqa: BLE001
        log.warning("[PN270] SKIP: GPUModelRunner not importable: %s", e)
        return "skipped", f"GPUModelRunner not importable: {e!r}"

    if not hasattr(GPUModelRunner, "initialize_kv_cache_tensors"):
        return "skipped", "GPUModelRunner.initialize_kv_cache_tensors missing"

    original = GPUModelRunner.initialize_kv_cache_tensors
    if getattr(original, "_genesis_pn270_wrapped", False):
        _APPLIED = True
        return "applied", "initialize_kv_cache_tensors already wrapped"
    _ORIGINAL_INIT_TENSORS = original

    def _wrapped(self, kv_cache_config, kernel_block_sizes):
        result = original(self, kv_cache_config, kernel_block_sizes)
        global _DUMPED
        if not _DUMPED:
            try:
                _dump_drafter_audit(self)
                _DUMPED = True
            except Exception as e:  # noqa: BLE001
                log.warning("[PN270] audit pass failed: %s", e)
        return result

    _wrapped._genesis_pn270_wrapped = True  # type: ignore[attr-defined]
    GPUModelRunner.initialize_kv_cache_tensors = _wrapped  # type: ignore[method-assign]
    _APPLIED = True
    log.warning(
        "[PN270] INSTALLED: drafter K/V projection audit will run "
        "once on first initialize_kv_cache_tensors call."
    )
    return "applied", "PN270 installed (audit-only)"

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

__all__ = ["GENESIS_PN270_MARKER", "apply", "is_applied", "revert"]

