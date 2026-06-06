# SPDX-License-Identifier: Apache-2.0
"""PN267 KV bridge trace.

Diagnostic probe for spec-decode KV bridge events. Stays dormant until the operator
enables it via its env-flag; canonical location is this file itself.
Resolves the Phase 3 relocation stash-pop conflict (old
`integrations/gemma4/` path was removed during the move).
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("genesis.spec_decode.pn267_kv_bridge_trace")

GENESIS_PN267_MARKER = (
    "Genesis PN267 K/V bridge feasibility trace (G4_78-0 probe — "
    "target[58]/[59] + drafter[0] forward shapes)"
)

_ENV_ENABLE = "GENESIS_ENABLE_PN267_KV_BRIDGE_TRACE"
_APPLIED = False
_ORIGINAL_INIT_TENSORS = None
_ORIGINAL_FA_FORWARD = None
_ORIGINAL_TRITON_FORWARD = None

# Captured at post-bind time:
_TARGET_58_ATTN: Any = None
_TARGET_59_ATTN: Any = None
_DRAFTER_ATTNS: dict[str, Any] = {}

_FA_CALL_COUNT = [0]
_TRITON_CALL_COUNT = [0]

DRAFTER_PREFIX = "draft_model."
TARGET_58_NAME = "language_model.model.layers.58.self_attn.attn"
TARGET_59_NAME = "language_model.model.layers.59.self_attn.attn"

def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )

def _safe_tensor_info(t: Any, name: str = "<?>") -> str:
    if t is None:
        return f"{name}=<None>"
    try:
        return (
            f"{name}: shape={tuple(t.shape)} "
            f"stride={tuple(t.stride())} "
            f"dtype={t.dtype} "
            f"contig={bool(t.is_contiguous())} "
            f"data_ptr=0x{int(t.data_ptr()):x} "
            f"ndim={int(t.dim())} "
            f"numel={int(t.numel())}"
        )
    except Exception as _e:  # noqa: BLE001
        return f"{name}: introspection-failed: {_e!r}"

def _resolve_target_refs(fwd_ctx: dict) -> None:
    """Walk static_forward_context once to find target[58]/[59]."""
    global _TARGET_58_ATTN, _TARGET_59_ATTN
    if _TARGET_58_ATTN is not None and _TARGET_59_ATTN is not None:
        return
    for name, attn in fwd_ctx.items():
        if not isinstance(name, str):
            continue
        if name == TARGET_58_NAME or name.endswith(".layers.58.self_attn.attn"):
            _TARGET_58_ATTN = attn
        if name == TARGET_59_NAME or name.endswith(".layers.59.self_attn.attn"):
            _TARGET_59_ATTN = attn
        if isinstance(name, str) and name.startswith(DRAFTER_PREFIX):
            _DRAFTER_ATTNS[name] = attn

def apply() -> tuple[str, str]:
    global _APPLIED, _ORIGINAL_INIT_TENSORS, _ORIGINAL_FA_FORWARD, _ORIGINAL_TRITON_FORWARD

    if not _env_enabled():
        return "skipped", f"PN267 disabled (set {_ENV_ENABLE}=1)"

    if _APPLIED:
        return "applied", "PN267 already installed"

    log.warning("[PN267] apply() entered — beginning import phase")

    # --- Wrap GPUModelRunner.initialize_kv_cache_tensors (post-bind capture)
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as e:  # noqa: BLE001
        log.warning("[PN267] SKIP: GPUModelRunner not importable: %s", e)
        return "skipped", f"GPUModelRunner not importable: {e!r}"

    if hasattr(GPUModelRunner, "initialize_kv_cache_tensors"):
        original_init_tensors = GPUModelRunner.initialize_kv_cache_tensors
        if not getattr(original_init_tensors, "_genesis_pn267_wrapped", False):
            _ORIGINAL_INIT_TENSORS = original_init_tensors

            def _wrapped_init_tensors(self, kv_cache_config, kernel_block_sizes):
                result = original_init_tensors(self, kv_cache_config, kernel_block_sizes)
                try:
                    fwd_ctx = self.compilation_config.static_forward_context
                    _resolve_target_refs(fwd_ctx)
                    log.warning(
                        "[PN267/post-bind] resolved target_58=%s target_59=%s "
                        "n_drafter_attns=%d",
                        "OK" if _TARGET_58_ATTN is not None else "MISSING",
                        "OK" if _TARGET_59_ATTN is not None else "MISSING",
                        len(_DRAFTER_ATTNS),
                    )
                    if _TARGET_58_ATTN is not None:
                        log.warning(
                            "[PN267/post-bind] target[58].kv_cache: %s",
                            _safe_tensor_info(
                                getattr(_TARGET_58_ATTN, "kv_cache", None),
                                "kv_cache",
                            ),
                        )
                    if _TARGET_59_ATTN is not None:
                        log.warning(
                            "[PN267/post-bind] target[59].kv_cache: %s",
                            _safe_tensor_info(
                                getattr(_TARGET_59_ATTN, "kv_cache", None),
                                "kv_cache",
                            ),
                        )
                    for dn, da in sorted(_DRAFTER_ATTNS.items()):
                        log.warning(
                            "[PN267/post-bind] %s.kv_cache: %s",
                            dn,
                            _safe_tensor_info(
                                getattr(da, "kv_cache", None),
                                "kv_cache",
                            ),
                        )
                except Exception as _e:  # noqa: BLE001
                    log.warning("[PN267/post-bind] introspection failed: %s", _e)
                return result

            _wrapped_init_tensors._genesis_pn267_wrapped = True  # type: ignore[attr-defined]
            GPUModelRunner.initialize_kv_cache_tensors = _wrapped_init_tensors  # type: ignore[method-assign]

    # --- Wrap FlashAttentionImpl.forward + TritonAttentionImpl.forward
    def _make_forward_wrapper(original_forward, label: str, counter: list):
        def _wrapped(self, *args, **kwargs):
            # args[0] = Attention wrapper, args[1] = query, args[2] = key,
            # args[3] = value, args[4] = kv_cache, args[5] = attn_metadata
            layer = args[0] if len(args) >= 1 else None
            prefix = (
                getattr(layer, "prefix", None)
                or getattr(layer, "layer_name", None)
                or ""
            )
            is_drafter = (
                isinstance(prefix, str) and prefix.startswith(DRAFTER_PREFIX)
            )
            if is_drafter and counter[0] < 6:
                counter[0] += 1
                try:
                    query = args[1] if len(args) >= 2 else kwargs.get("query")
                    key = args[2] if len(args) >= 3 else kwargs.get("key")
                    value = args[3] if len(args) >= 4 else kwargs.get("value")
                    kv_cache = args[4] if len(args) >= 5 else kwargs.get("kv_cache")
                    attn_metadata = (
                        args[5] if len(args) >= 6 else kwargs.get("attn_metadata")
                    )
                    sm = getattr(attn_metadata, "slot_mapping", None)
                    sl = getattr(attn_metadata, "seq_lens", None)
                    nat = getattr(attn_metadata, "num_actual_tokens", "?")
                    log.warning(
                        "[PN267/%s] drafter forward call #%d: layer=%r\n"
                        "  query: %s\n"
                        "  key:   %s\n"
                        "  value: %s\n"
                        "  kv_cache: %s\n"
                        "  attn_metadata: type=%s num_actual_tokens=%s\n"
                        "  slot_mapping[:8]=%s seq_lens[:4]=%s\n"
                        "  target[58].kv_cache: %s\n"
                        "  target[59].kv_cache: %s",
                        label, counter[0], prefix,
                        _safe_tensor_info(query, "query"),
                        _safe_tensor_info(key, "key"),
                        _safe_tensor_info(value, "value"),
                        _safe_tensor_info(kv_cache, "kv_cache"),
                        type(attn_metadata).__qualname__
                        if attn_metadata is not None else "<None>",
                        nat,
                        sm[:8].tolist() if (sm is not None and hasattr(sm, "tolist")) else sm,
                        sl[:4].tolist() if (sl is not None and hasattr(sl, "tolist")) else sl,
                        _safe_tensor_info(
                            getattr(_TARGET_58_ATTN, "kv_cache", None)
                            if _TARGET_58_ATTN is not None else None,
                            "kv_cache",
                        ),
                        _safe_tensor_info(
                            getattr(_TARGET_59_ATTN, "kv_cache", None)
                            if _TARGET_59_ATTN is not None else None,
                            "kv_cache",
                        ),
                    )
                except Exception as _e:  # noqa: BLE001
                    log.warning("[PN267/%s] introspection failed: %s", label, _e)
            elif is_drafter and counter[0] == 6:
                counter[0] += 1
                log.warning("[PN267/%s] further drafter trace logs suppressed (>6)", label)
            return original_forward(self, *args, **kwargs)
        return _wrapped

    try:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
        if not getattr(FlashAttentionImpl.forward, "_genesis_pn267_wrapped", False):
            _ORIGINAL_FA_FORWARD = FlashAttentionImpl.forward
            wrapped_fa = _make_forward_wrapper(
                _ORIGINAL_FA_FORWARD, "FA", _FA_CALL_COUNT,
            )
            wrapped_fa._genesis_pn267_wrapped = True  # type: ignore[attr-defined]
            FlashAttentionImpl.forward = wrapped_fa  # type: ignore[method-assign]
    except Exception as e:  # noqa: BLE001
        log.warning("[PN267] FlashAttentionImpl wrap failed: %s", e)

    try:
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
        if not getattr(TritonAttentionImpl.forward, "_genesis_pn267_wrapped", False):
            _ORIGINAL_TRITON_FORWARD = TritonAttentionImpl.forward
            wrapped_tr = _make_forward_wrapper(
                _ORIGINAL_TRITON_FORWARD, "Triton", _TRITON_CALL_COUNT,
            )
            wrapped_tr._genesis_pn267_wrapped = True  # type: ignore[attr-defined]
            TritonAttentionImpl.forward = wrapped_tr  # type: ignore[method-assign]
    except Exception as e:  # noqa: BLE001
        log.warning("[PN267] TritonAttentionImpl wrap failed: %s", e)

    _APPLIED = True
    log.warning(
        "[PN267] INSTALLED: initialize_kv_cache_tensors post-bind hook + "
        "FlashAttentionImpl.forward + TritonAttentionImpl.forward wrapped "
        "(drafter prefix=%r). target[58]/[59] references captured at "
        "post-bind; drafter forward logged for first 6 calls per backend.",
        DRAFTER_PREFIX,
    )
    return "applied", "PN267 installed (trace-only, no behavior change)"

def is_applied() -> bool:
    return _APPLIED

def revert() -> bool:
    """Best-effort revert (test isolation only)."""
    global _APPLIED, _ORIGINAL_INIT_TENSORS, _ORIGINAL_FA_FORWARD, _ORIGINAL_TRITON_FORWARD
    if not _APPLIED:
        return False
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        if _ORIGINAL_INIT_TENSORS is not None:
            GPUModelRunner.initialize_kv_cache_tensors = _ORIGINAL_INIT_TENSORS  # type: ignore[method-assign]
    except ImportError:
        pass
    try:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
        if _ORIGINAL_FA_FORWARD is not None:
            FlashAttentionImpl.forward = _ORIGINAL_FA_FORWARD  # type: ignore[method-assign]
    except ImportError:
        pass
    try:
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
        if _ORIGINAL_TRITON_FORWARD is not None:
            TritonAttentionImpl.forward = _ORIGINAL_TRITON_FORWARD  # type: ignore[method-assign]
    except ImportError:
        pass
    _APPLIED = False
    _ORIGINAL_INIT_TENSORS = None
    _ORIGINAL_FA_FORWARD = None
    _ORIGINAL_TRITON_FORWARD = None
    return True

__all__ = ["GENESIS_PN267_MARKER", "apply", "is_applied", "revert"]

