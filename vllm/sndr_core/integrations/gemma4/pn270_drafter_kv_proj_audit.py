# SPDX-License-Identifier: Apache-2.0
"""PN270 — Drafter K/V projection audit (read-only).

================================================================
WHY
================================================================

G4_78-A v1 verdict (2026-05-19): bridge mechanically correct
(target_slot match byte-exact for prefill + decode), output coherent,
no crash — but acceptance stays 0/9 calls. PN269 + G4_78 traces
revealed a suspicious signal:

  drafter's key stats ≈ drafter's value stats
    e.g., prompt prefill: K mean=2.86e-3 std=0.122
                          V mean=2.86e-3 std=0.122  (IDENTICAL stats)

If drafter's k_proj and v_proj produce identical output, then EITHER:
  (a) they share weights (tied)
  (b) one is missing / stubbed / identity, and the other is computed
  (c) kv_sharing originally skipped the projections entirely;
      G4_76 disabled sharing but the projections never existed in
      the loaded checkpoint
  (d) coincidence (unlikely — exactly identical mean+std at multiple
      call sites is not a coincidence)

If (a/b/c) is true, then G4_78's FA-forward K/V substitution is
applied AFTER the broken projection — too late. The right hook
would be at the Attention.forward / kv_proj boundary, or by
redesigning kv_sharing as bridge rather than alias.

PN270 settles the question with weight-level inspection.

================================================================
WHAT IT CHECKS
================================================================

After model load (hooked at GPUModelRunner.initialize_kv_cache_tensors,
which runs strictly after model.load_model()), walks the loaded model
with `named_modules()`, filters to `draft_model.layers.0..3.self_attn`,
and for each layer dumps:

  PER PROJECTION (q_proj, k_proj, v_proj, qkv_proj, kv_proj):
    - exists / missing
    - module class name
    - weight.shape / dtype / data_ptr
    - weight.float().norm()
    - parameter name in state_dict (if attributable)

  PER LAYER:
    - k_proj.weight.data_ptr() == v_proj.weight.data_ptr() (tied storage)
    - torch.allclose(k_proj.weight, v_proj.weight) (shape permitting)
    - attn submodule attributes:
        kv_sharing_target_layer_name
        attn_type
        num_kv_heads, num_heads, head_size
        kv_cache_dtype

================================================================
ENV
================================================================

  GENESIS_ENABLE_PN270_DRAFTER_KV_PROJ_AUDIT=1

================================================================
NO BEHAVIOR CHANGE — DIAGNOSTIC ONLY
================================================================

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("genesis.gemma4.pn270_drafter_kv_proj_audit")

GENESIS_PN270_MARKER = "Genesis PN270 drafter K/V projection audit"

_ENV_ENABLE = "GENESIS_ENABLE_PN270_DRAFTER_KV_PROJ_AUDIT"
_APPLIED = False
_ORIGINAL_INIT_TENSORS = None
_DUMPED = False  # one-shot guard

DRAFTER_SELF_ATTN_PREFIX = "draft_model.layers."
DRAFTER_SELF_ATTN_SUFFIX = ".self_attn"


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


def _dump_drafter_audit(model: Any) -> None:
    """Walk model, find drafter self_attn modules, log full audit."""
    import torch

    log.warning("[PN270] === Drafter K/V projection audit BEGIN ===")
    log.warning("[PN270] model_class=%s", type(model).__qualname__)

    candidates: list[tuple[str, Any]] = []
    try:
        for name, module in model.named_modules():
            if (
                name.startswith(DRAFTER_SELF_ATTN_PREFIX)
                and name.endswith(DRAFTER_SELF_ATTN_SUFFIX)
                # exclude .self_attn.attn or other inner pieces
                and name.count(".") == 3  # draft_model . layers . N . self_attn
            ):
                candidates.append((name, module))
    except Exception as _e:
        log.warning("[PN270] named_modules() failed: %s", _e)
        return

    log.warning("[PN270] found %d drafter self_attn modules: %s",
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

    # ---- state_dict key audit ----
    try:
        sd_keys = list(model.state_dict().keys())
        drafter_keys = [k for k in sd_keys if k.startswith("draft_model.")]
        kv_keys = [
            k for k in drafter_keys
            if any(p in k for p in (".k_proj.", ".v_proj.", ".q_proj.",
                                    ".qkv_proj.", ".kv_proj."))
        ]
        log.warning(
            "[PN270] state_dict drafter K/V/Q keys (count=%d): %s",
            len(kv_keys),
            kv_keys[:24] + (["..."] if len(kv_keys) > 24 else []),
        )
    except Exception as _e:
        log.warning("[PN270] state_dict audit failed: %s", _e)

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
                model = getattr(self, "model", None)
                if model is None:
                    log.warning("[PN270] no self.model on GPUModelRunner; "
                                "deferring audit to next call")
                else:
                    _dump_drafter_audit(model)
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
