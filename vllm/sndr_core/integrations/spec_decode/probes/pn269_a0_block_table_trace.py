# SPDX-License-Identifier: Apache-2.0
"""PN269 A0 block table trace.

Diagnostic probe for A0 block-table mutations. Stays dormant until the operator
enables it via its env-flag; canonical location is this file itself.
Resolves the Phase 3 relocation stash-pop conflict (old
`integrations/gemma4/` path was removed during the move).
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("genesis.spec_decode.pn269_a0_block_table_trace")

GENESIS_PN269_MARKER = (
    "Genesis PN269 G4_78-A0 trace: target/drafter block_table accessibility"
)

_ENV_ENABLE = "GENESIS_ENABLE_PN269_A0_BLOCK_TABLE_TRACE"
_ENV_MAX_CALLS = "GENESIS_PN269_MAX_LOG_CALLS"
_APPLIED = False
_ORIGINAL_FA_FORWARD = None
_ORIGINAL_TRITON_FORWARD = None

DRAFTER_PREFIX = "draft_model."
TARGET_58_SUFFIX = ".layers.58.self_attn.attn"
TARGET_59_SUFFIX = ".layers.59.self_attn.attn"

# Captured per-call state for target[58] and target[59]
_LAST_TARGET_STATE: dict[str, dict[str, Any]] = {}
_CALL_COUNT = [0]

def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )

def _max_calls() -> int:
    try:
        return int(os.environ.get(_ENV_MAX_CALLS, "16"))
    except ValueError:
        return 16

def _safe_attr(obj: Any, name: str, default: Any = "<absent>") -> Any:
    return getattr(obj, name, default)

def _tensor_head(t: Any, n: int = 8) -> Any:
    if t is None:
        return "<None>"
    try:
        if t.dim() == 0:
            return t.item()
        flat = t.flatten()
        return flat[:n].tolist()
    except Exception as _e:
        return f"<tensor_head_err: {_e!r}>"

def _block_table_head(bt: Any) -> Any:
    if bt is None:
        return "<None>"
    try:
        if bt.dim() < 2:
            return _tensor_head(bt, 8)
        return bt[0, :8].tolist()
    except Exception as _e:
        return f"<block_table_head_err: {_e!r}>"

def _reconstruct_positions(query_start_loc: Any, seq_lens: Any,
                           num_actual_tokens: int) -> list[int]:
    """Reconstruct logical positions per token from attn_metadata fields.

    For seq i, positions are seq_lens[i] - query_len + range(query_len).
    """
    try:
        qsl = query_start_loc.tolist() if hasattr(query_start_loc, "tolist") else list(query_start_loc)
        sl = seq_lens.tolist() if hasattr(seq_lens, "tolist") else list(seq_lens)
        positions = []
        num_seqs = len(sl)
        for i in range(num_seqs):
            qs = qsl[i]
            qe = qsl[i + 1] if i + 1 < len(qsl) else num_actual_tokens
            query_len = qe - qs
            seq_total = sl[i]
            for j in range(query_len):
                positions.append(seq_total - query_len + j)
        return positions
    except Exception as _e:
        log.warning("[PN269] position reconstruction failed: %s", _e)
        return []

def _identify_layer(args: tuple) -> tuple[str, str]:
    """Return (full_layer_name, kind) where kind in {'target_58',
    'target_59', 'drafter_X', 'other'}."""
    if len(args) < 1:
        return "<unknown>", "other"
    layer = args[0]
    prefix = (
        getattr(layer, "prefix", None)
        or getattr(layer, "layer_name", None)
        or ""
    )
    if not isinstance(prefix, str):
        return "<unknown>", "other"
    if prefix.endswith(TARGET_58_SUFFIX):
        return prefix, "target_58"
    if prefix.endswith(TARGET_59_SUFFIX):
        return prefix, "target_59"
    if prefix.startswith(DRAFTER_PREFIX):
        return prefix, f"drafter_{prefix.split('.')[2]}"  # layers.{N}
    return prefix, "other"

def _capture_metadata(args: tuple, kwargs: dict) -> dict[str, Any]:
    """Pull out the fields we want to log + persist."""
    key = args[2] if len(args) >= 3 else kwargs.get("key")
    value = args[3] if len(args) >= 4 else kwargs.get("value")
    kv_cache = args[4] if len(args) >= 5 else kwargs.get("kv_cache")
    attn_metadata = args[5] if len(args) >= 6 else kwargs.get("attn_metadata")

    info = {
        "num_actual_tokens": _safe_attr(attn_metadata, "num_actual_tokens"),
        "max_query_len": _safe_attr(attn_metadata, "max_query_len"),
        "query_start_loc": _safe_attr(attn_metadata, "query_start_loc"),
        "seq_lens": _safe_attr(attn_metadata, "seq_lens"),
        "block_table": _safe_attr(attn_metadata, "block_table"),
        "slot_mapping": _safe_attr(attn_metadata, "slot_mapping"),
        "key": key,
        "value": value,
        "kv_cache": kv_cache,
        "attn_metadata_class": type(attn_metadata).__qualname__
            if attn_metadata is not None else "<None>",
    }
    return info

def apply() -> tuple[str, str]:
    global _APPLIED, _ORIGINAL_FA_FORWARD, _ORIGINAL_TRITON_FORWARD

    if not _env_enabled():
        return "skipped", f"PN269 disabled (set {_ENV_ENABLE}=1)"

    if _APPLIED:
        return "applied", "PN269 already installed"

    log.warning("[PN269] apply() entered")
    max_calls = _max_calls()

    def _make_wrap(original, backend_label: str):
        def _wrapped(self, *args, **kwargs):
            if _CALL_COUNT[0] >= max_calls:
                return original(self, *args, **kwargs)

            layer_name, kind = _identify_layer(args)
            if kind == "other":
                return original(self, *args, **kwargs)

            info = _capture_metadata(args, kwargs)
            # Skip warmup / profile / cudagraph-capture calls where
            # attn_metadata is None — they don't represent real KV
            # geometry. Don't consume the call budget on them.
            attn_md = (
                args[5] if len(args) >= 6 else kwargs.get("attn_metadata")
            )
            if attn_md is None:
                return original(self, *args, **kwargs)
            num_actual = info["num_actual_tokens"]
            if not isinstance(num_actual, int) and hasattr(num_actual, "item"):
                try:
                    num_actual = int(num_actual)
                except Exception:
                    pass

            qsl = info["query_start_loc"]
            sl = info["seq_lens"]
            bt = info["block_table"]
            sm = info["slot_mapping"]
            key = info["key"]
            value = info["value"]
            kv_cache = info["kv_cache"]

            # Reconstruct positions
            positions = []
            try:
                if qsl is not None and sl is not None:
                    positions = _reconstruct_positions(
                        qsl, sl,
                        num_actual if isinstance(num_actual, int) else 0,
                    )
            except Exception as _e:
                log.warning("[PN269] position reconstruction error: %s", _e)

            # Is this a prefill-like call?
            is_prefill_like = False
            try:
                qsl_list = qsl.tolist() if hasattr(qsl, "tolist") else qsl
                sl_list = sl.tolist() if hasattr(sl, "tolist") else sl
                if qsl_list and sl_list:
                    # prefill: query_len > 1 (more than just one new token)
                    is_prefill_like = (qsl_list[1] - qsl_list[0]) > 1
            except Exception:
                pass

            _CALL_COUNT[0] += 1
            call_idx = _CALL_COUNT[0]

            # Capture key/value/kv_cache shapes + checksums
            def _ck(t):
                if t is None:
                    return "<None>"
                try:
                    f = t.float() if t.dtype != "torch.float32" else t
                    return (f"shape={tuple(t.shape)} dtype={t.dtype} "
                            f"sum={f.sum().item():.4e} "
                            f"mean={f.mean().item():.4e} "
                            f"std={f.std().item():.4e}")
                except Exception as _e:
                    return f"<ck_err: {_e!r}>"

            log.warning(
                "[PN269 #%d] %s (%s) impl=%s\n"
                "  attn_metadata=%s num_actual_tokens=%s max_query_len=%s "
                "is_prefill_like=%s\n"
                "  query_start_loc=%s seq_lens=%s\n"
                "  block_table.shape=%s [0,:8]=%s\n"
                "  slot_mapping[:8]=%s\n"
                "  positions[:8]=%s\n"
                "  key: %s\n"
                "  value: %s\n"
                "  kv_cache: shape=%s data_ptr=0x%x",
                call_idx, layer_name, kind,
                _safe_attr(getattr(self, "__class__", None), "__name__", "?"),
                info["attn_metadata_class"], num_actual,
                info["max_query_len"],
                is_prefill_like,
                _tensor_head(qsl, 4), _tensor_head(sl, 4),
                _safe_attr(bt, "shape", "<absent>"), _block_table_head(bt),
                _tensor_head(sm, 8),
                positions[:8],
                _ck(key),
                _ck(value),
                _safe_attr(kv_cache, "shape", "<absent>"),
                int(kv_cache.data_ptr()) if kv_cache is not None and hasattr(kv_cache, "data_ptr") else 0,
            )

            # Persist target state for cross-layer lookup
            if kind in ("target_58", "target_59"):
                try:
                    state = {
                        "block_table": bt.clone() if hasattr(bt, "clone") else None,
                        "slot_mapping": sm.clone() if hasattr(sm, "clone") else None,
                        "positions": positions[:],
                        "query_start_loc": qsl.clone() if hasattr(qsl, "clone") else None,
                        "seq_lens": sl.clone() if hasattr(sl, "clone") else None,
                        "call_idx": call_idx,
                    }
                    _LAST_TARGET_STATE[kind] = state
                except Exception as _e:
                    log.warning(
                        "[PN269 #%d] %s state persistence failed: %s",
                        call_idx, kind, _e,
                    )

            # For drafter: cross-reference with saved target state
            if kind.startswith("drafter_"):
                target_kind = "target_58" if kind in ("drafter_0", "drafter_1", "drafter_2") else "target_59"
                ts = _LAST_TARGET_STATE.get(target_kind)
                if ts is None:
                    log.warning(
                        "[PN269 #%d] %s: no stored %s state yet (drafter "
                        "called before target?)",
                        call_idx, kind, target_kind,
                    )
                else:
                    try:
                        t_positions = ts["positions"]
                        t_sm = ts["slot_mapping"]
                        t_bt = ts["block_table"]
                        target_block_size = 32 if target_kind == "target_58" else 64
                        validation_rows = []
                        for i in range(min(8, len(positions))):
                            P = positions[i]
                            drafter_slot = (
                                sm[i].item() if (sm is not None and i < len(sm))
                                else "?"
                            )
                            # Method 1: direct from target's current slot_mapping
                            t_slot_direct = None
                            if P in t_positions:
                                idx = t_positions.index(P)
                                if t_sm is not None and idx < len(t_sm):
                                    t_slot_direct = int(t_sm[idx].item())
                            # Method 2: via target block_table
                            t_slot_via_bt = None
                            try:
                                if t_bt is not None:
                                    b_idx = P // target_block_size
                                    b_off = P % target_block_size
                                    target_block_id = int(t_bt[0, b_idx].item())
                                    t_slot_via_bt = target_block_id * target_block_size + b_off
                            except Exception as _e:
                                t_slot_via_bt = f"<err: {_e!r}>"
                            validation_rows.append(
                                f"P={P} drafter_slot={drafter_slot} "
                                f"target_slot_direct={t_slot_direct} "
                                f"target_slot_via_bt={t_slot_via_bt}"
                            )
                        log.warning(
                            "[PN269 #%d] %s -> %s cross-ref (last_target_call=%d):\n  %s",
                            call_idx, kind, target_kind,
                            ts.get("call_idx", -1),
                            "\n  ".join(validation_rows),
                        )
                    except Exception as _e:
                        log.warning(
                            "[PN269 #%d] %s cross-ref failed: %s",
                            call_idx, kind, _e,
                        )

            return original(self, *args, **kwargs)

        return _wrapped

    try:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
        if not getattr(FlashAttentionImpl.forward, "_genesis_pn269_wrapped", False):
            _ORIGINAL_FA_FORWARD = FlashAttentionImpl.forward
            wrapped = _make_wrap(_ORIGINAL_FA_FORWARD, "FA")
            wrapped._genesis_pn269_wrapped = True
            FlashAttentionImpl.forward = wrapped
    except Exception as e:
        log.warning("[PN269] FA wrap failed: %s", e)

    try:
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
        if not getattr(TritonAttentionImpl.forward, "_genesis_pn269_wrapped", False):
            _ORIGINAL_TRITON_FORWARD = TritonAttentionImpl.forward
            wrapped = _make_wrap(_ORIGINAL_TRITON_FORWARD, "Triton")
            wrapped._genesis_pn269_wrapped = True
            TritonAttentionImpl.forward = wrapped
    except Exception as e:
        log.warning("[PN269] Triton wrap failed: %s", e)

    _APPLIED = True
    log.warning(
        "[PN269] INSTALLED: FA + Triton forward wrapped for target[58/59] "
        "and drafter[0..3]; max_log_calls=%d.",
        max_calls,
    )
    return "applied", "PN269 installed"

def is_applied() -> bool:
    return _APPLIED

def revert() -> bool:
    global _APPLIED, _ORIGINAL_FA_FORWARD, _ORIGINAL_TRITON_FORWARD
    if not _APPLIED:
        return False
    try:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
        if _ORIGINAL_FA_FORWARD is not None:
            FlashAttentionImpl.forward = _ORIGINAL_FA_FORWARD
    except ImportError:
        pass
    try:
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
        if _ORIGINAL_TRITON_FORWARD is not None:
            TritonAttentionImpl.forward = _ORIGINAL_TRITON_FORWARD
    except ImportError:
        pass
    _APPLIED = False
    _ORIGINAL_FA_FORWARD = None
    _ORIGINAL_TRITON_FORWARD = None
    return True

__all__ = ["GENESIS_PN269_MARKER", "apply", "is_applied", "revert"]

