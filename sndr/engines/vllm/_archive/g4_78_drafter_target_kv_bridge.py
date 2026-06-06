# SPDX-License-Identifier: Apache-2.0
"""G4_78-A v2 — drafter[0..3] <- target[58]/[59] K/V bridge (prefill + decode).

================================================================
WHY v2 vs v1
================================================================

PN270 audit (2026-05-19) proved drafter has NO k_proj / v_proj /
qkv_proj at all — Gemma4MTPAttention is Q-only, K/V come from the
target via kv_sharing. After G4_76 disabled kv_sharing (necessary
because G4_74 broke the physical alias), drafter's forward passes
`torch.empty(...)` as both K and V to its inner Attention layer.
That's uninitialized memory.

The Gemma4MTPAttention source (gemma4_mtp.py:148):

    # Attention reads K/V from the target's cache via KV sharing;
    # these dummy tensors are never consumed but required by the API.
    num_tokens = q.shape[0]
    kv_dummy = torch.empty(num_tokens, self.num_kv_heads * self.head_dim, ...)
    attn_output = self.attn(q, kv_dummy, kv_dummy)

So in the current state (G4_76 ON, no bridge):
  - drafter cache fills with torch.empty garbage
  - subsequent reads return garbage
  - drafter draft tokens are essentially random
  - accept_rate = 0%

The G4_78 bridge IS the replacement for the disabled physical
kv_sharing — it substitutes meaningful K/V at the FA/Triton forward
boundary BEFORE the kernel call, so:
  - cache gets sane data
  - attention is computed against target's actual K/V
  - drafter operates as designed

================================================================
SCOPE — v2
================================================================

- All 4 drafter layers (0, 1, 2, 3)
- Prompt prefill AND decode steps (PN269 proved exactness for both)
- Mapping:
    drafter[0..2] (FA backend, head=8x256)  <- target[58] (Triton, sliding, block=32)
    drafter[3]    (Triton, head=8x512)      <- target[59] (Triton, full,    block=64)
- Layer 3 GQA replication: target[59] has num_kv_heads=2,
  drafter[3] expects num_kv_heads=8. Replicate each KV head 4x
  via target_kv.repeat_interleave(4, dim=heads_axis).

================================================================
HOOKS
================================================================

(1) TritonAttentionImpl.forward — both capture AND bridge:
    a. If layer is target[58] or target[59]: capture state (kv_cache
       reference, block_table clone, query_start_loc, seq_lens).
    b. If layer is drafter[3]: bridge K/V from target[59] state
       with GQA replication factor 4.
    c. Else: passthrough.

(2) FlashAttentionImpl.forward — bridge only (no capture):
    For drafter[0..2]: bridge K/V from target[58] state.

================================================================
SAFETY
================================================================

Every guarded path. Any check failure -> log no-op reason + passthrough.
NEVER crash. Counters track apply/skip per layer.

================================================================
ENV
================================================================

  GENESIS_ENABLE_G4_78_DRAFTER_TARGET_KV_BRIDGE=1

================================================================

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("genesis.gemma4.g4_78_drafter_target_kv_bridge")

GENESIS_G4_78_MARKER = (
    "Genesis G4_78-A v2 drafter[0..3] <- target[58]/[59] K/V bridge "
    "(prefill + decode; GQA repeat=4 for layer 3)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_78_DRAFTER_TARGET_KV_BRIDGE"

DRAFTER_PREFIX = "draft_model."
TARGET_58_SUFFIX = ".layers.58.self_attn.attn"
TARGET_59_SUFFIX = ".layers.59.self_attn.attn"

# Block sizes per target layer (PN269 evidence: target[58] sliding=32,
# target[59] full=64). KV cache shape: (num_blocks, 2, block_size, num_kv_heads, head_size)
TARGET_58_BLOCK_SIZE = 32
TARGET_59_BLOCK_SIZE = 64

# GQA replication for drafter[3]: target[59] has num_kv_heads=2,
# drafter[3] expects num_kv_heads=8. Each target head consumed by
# 4 drafter heads.
DRAFTER_3_GQA_REPEAT = 4

_APPLIED = False
_ORIGINAL_FA_FORWARD = None
_ORIGINAL_TRITON_FORWARD = None


class _State:
    target_58_kv_cache: Any = None
    target_58_block_table: Any = None  # cloned per-call
    target_58_seq_lens: Any = None
    target_58_query_start_loc: Any = None
    target_58_capture_count: int = 0

    target_59_kv_cache: Any = None
    target_59_block_table: Any = None
    target_59_seq_lens: Any = None
    target_59_query_start_loc: Any = None
    target_59_capture_count: int = 0

    bridge_apply_count: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    bridge_skip_count: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}


_STATE = _State()


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _identify_drafter_layer_idx(layer: Any) -> int | None:
    """Return layer index N if prefix is 'draft_model.layers.N...', else None."""
    prefix = getattr(layer, "prefix", None) or getattr(layer, "layer_name", None)
    if not isinstance(prefix, str) or not prefix.startswith(DRAFTER_PREFIX):
        return None
    parts = prefix.split(".")
    if len(parts) < 3 or parts[1] != "layers":
        return None
    try:
        return int(parts[2])
    except ValueError:
        return None


def _is_target_58(layer: Any) -> bool:
    prefix = getattr(layer, "prefix", None) or getattr(layer, "layer_name", None)
    return isinstance(prefix, str) and prefix.endswith(TARGET_58_SUFFIX)


def _is_target_59(layer: Any) -> bool:
    prefix = getattr(layer, "prefix", None) or getattr(layer, "layer_name", None)
    return isinstance(prefix, str) and prefix.endswith(TARGET_59_SUFFIX)


def _reconstruct_positions(query_start_loc: Any, seq_lens: Any) -> list[int]:
    try:
        qsl = query_start_loc.tolist() if hasattr(query_start_loc, "tolist") else list(query_start_loc)
        sl = seq_lens.tolist() if hasattr(seq_lens, "tolist") else list(seq_lens)
        positions: list[int] = []
        num_seqs = len(sl)
        for i in range(num_seqs):
            qs = qsl[i]
            qe = qsl[i + 1]
            seq_total = sl[i]
            query_len = qe - qs
            for j in range(query_len):
                positions.append(seq_total - query_len + j)
        return positions
    except Exception as _e:  # noqa: BLE001
        log.warning("[G4_78] position reconstruction failed: %s", _e)
        return []


def _capture_target_state(label: str, args: tuple, kwargs: dict) -> None:
    """Capture target_58 or target_59 state from forward args."""
    attn_md = args[5] if len(args) >= 6 else kwargs.get("attn_metadata")
    kv_cache = args[4] if len(args) >= 5 else kwargs.get("kv_cache")
    if attn_md is None or kv_cache is None:
        return
    try:
        bt = getattr(attn_md, "block_table", None)
        qsl = getattr(attn_md, "query_start_loc", None)
        sl = getattr(attn_md, "seq_lens", None)
        if label == "target_58":
            _STATE.target_58_kv_cache = kv_cache  # reference
            _STATE.target_58_block_table = (
                bt.clone() if (bt is not None and hasattr(bt, "clone")) else None
            )
            _STATE.target_58_query_start_loc = (
                qsl.clone() if (qsl is not None and hasattr(qsl, "clone")) else None
            )
            _STATE.target_58_seq_lens = (
                sl.clone() if (sl is not None and hasattr(sl, "clone")) else None
            )
            _STATE.target_58_capture_count += 1
        elif label == "target_59":
            _STATE.target_59_kv_cache = kv_cache
            _STATE.target_59_block_table = (
                bt.clone() if (bt is not None and hasattr(bt, "clone")) else None
            )
            _STATE.target_59_query_start_loc = (
                qsl.clone() if (qsl is not None and hasattr(qsl, "clone")) else None
            )
            _STATE.target_59_seq_lens = (
                sl.clone() if (sl is not None and hasattr(sl, "clone")) else None
            )
            _STATE.target_59_capture_count += 1
    except Exception as _e:  # noqa: BLE001
        log.warning("[G4_78] %s capture failed: %s", label, _e)


def _build_bridged_kv(
    layer_idx: int,
    target_kv: Any,
    target_bt: Any,
    block_size: int,
    positions: list[int],
    key_shape_template: Any,
    value_shape_template: Any,
    torch_module: Any,
) -> tuple[Any, Any, list[int], list[int], list[int]] | None:
    """Build new_key, new_value from target_kv at positions.

    target_kv layout: (num_blocks, 2, block_size, num_kv_heads, head_size).
    For drafter[3]: target has num_kv_heads=2, drafter expects 8 ->
    GQA replication factor 4 along the heads axis.

    Returns (new_key, new_value, block_ids, offsets, target_slots) or
    None on failure.
    """
    try:
        block_ids: list[int] = []
        offsets: list[int] = []
        for P in positions:
            b_idx = P // block_size
            b_off = P % block_size
            block_id = int(target_bt[0, b_idx].item())
            block_ids.append(block_id)
            offsets.append(b_off)

        bid_t = torch_module.tensor(block_ids, dtype=torch_module.long,
                                    device=target_kv.device)
        off_t = torch_module.tensor(offsets, dtype=torch_module.long,
                                    device=target_kv.device)

        # K = target_kv[bid_t, 0, off_t]  -> (N, num_kv_heads_target, head_size)
        new_key_raw = target_kv[bid_t, 0, off_t]
        new_value_raw = target_kv[bid_t, 1, off_t]

        # Layer 3: GQA replicate heads (2 -> 8) on axis 1
        if layer_idx == 3:
            new_key_raw = new_key_raw.repeat_interleave(
                DRAFTER_3_GQA_REPEAT, dim=1
            )
            new_value_raw = new_value_raw.repeat_interleave(
                DRAFTER_3_GQA_REPEAT, dim=1
            )

        new_key = new_key_raw.to(key_shape_template.dtype)
        new_value = new_value_raw.to(value_shape_template.dtype)

        target_slots = [
            block_ids[i] * block_size + offsets[i] for i in range(len(positions))
        ]
        return new_key, new_value, block_ids, offsets, target_slots
    except Exception as _e:  # noqa: BLE001
        log.warning("[G4_78] layer=%d gather/replicate failed: %s",
                    layer_idx, _e)
        return None


def _try_bridge(
    layer_idx: int,
    args: tuple,
    kwargs: dict,
    target_kind: str,
    block_size: int,
    torch_module: Any,
) -> tuple | None:
    """Attempt K/V substitution. Returns new args tuple on success
    OR None to passthrough."""
    attn_md = args[5] if len(args) >= 6 else kwargs.get("attn_metadata")
    if attn_md is None:
        _STATE.bridge_skip_count[layer_idx] += 1
        return None

    qsl = getattr(attn_md, "query_start_loc", None)
    sl = getattr(attn_md, "seq_lens", None)
    if qsl is None or sl is None:
        _STATE.bridge_skip_count[layer_idx] += 1
        return None

    if target_kind == "target_58":
        target_kv = _STATE.target_58_kv_cache
        target_bt = _STATE.target_58_block_table
        capture_count = _STATE.target_58_capture_count
    else:
        target_kv = _STATE.target_59_kv_cache
        target_bt = _STATE.target_59_block_table
        capture_count = _STATE.target_59_capture_count

    if target_kv is None or target_bt is None:
        _STATE.bridge_skip_count[layer_idx] += 1
        log.warning(
            "[G4_78] layer=%d no %s state yet — no-op (captures=%d)",
            layer_idx, target_kind, capture_count,
        )
        return None

    key = args[2]
    value = args[3]
    if key is None or value is None:
        _STATE.bridge_skip_count[layer_idx] += 1
        return None

    # Safety checks
    try:
        if target_kv.ndim != 5:
            raise ValueError(f"target_kv.ndim={target_kv.ndim} != 5")
        if int(target_kv.shape[1]) != 2:
            raise ValueError(f"target_kv.shape[1]={target_kv.shape[1]} != 2")
        # head_size must match (target axis 4 == drafter axis 2)
        if int(target_kv.shape[4]) != int(key.shape[-1]):
            raise ValueError(
                f"head_size mismatch target={target_kv.shape[4]} "
                f"drafter={key.shape[-1]}"
            )
        # num_kv_heads check (with GQA accommodation)
        target_kv_heads = int(target_kv.shape[3])
        drafter_kv_heads = int(key.shape[-2])
        if layer_idx == 3:
            if drafter_kv_heads != target_kv_heads * DRAFTER_3_GQA_REPEAT:
                raise ValueError(
                    f"GQA replication mismatch: drafter_kv_heads={drafter_kv_heads} "
                    f"!= target_kv_heads({target_kv_heads}) * "
                    f"{DRAFTER_3_GQA_REPEAT}"
                )
        else:
            if drafter_kv_heads != target_kv_heads:
                raise ValueError(
                    f"num_kv_heads mismatch: drafter={drafter_kv_heads} "
                    f"target={target_kv_heads} (no GQA for layer {layer_idx})"
                )
        if key.shape != value.shape:
            raise ValueError(
                f"key.shape={tuple(key.shape)} != value.shape={tuple(value.shape)}"
            )
    except Exception as _e:  # noqa: BLE001
        _STATE.bridge_skip_count[layer_idx] += 1
        log.warning("[G4_78] layer=%d safety check fail: %s — no-op",
                    layer_idx, _e)
        return None

    positions = _reconstruct_positions(qsl, sl)
    if len(positions) != int(key.shape[0]):
        _STATE.bridge_skip_count[layer_idx] += 1
        log.warning(
            "[G4_78] layer=%d positions(%d) != key.shape[0](%d) — no-op",
            layer_idx, len(positions), int(key.shape[0]),
        )
        return None

    built = _build_bridged_kv(
        layer_idx, target_kv, target_bt, block_size, positions,
        key, value, torch_module,
    )
    if built is None:
        _STATE.bridge_skip_count[layer_idx] += 1
        return None

    new_key, new_value, block_ids, offsets, target_slots = built
    _STATE.bridge_apply_count[layer_idx] += 1

    # Logging: one canonical log on layer 0 and layer 3 (boundaries)
    if layer_idx in (0, 3):
        try:
            k_b = key.float()
            k_a = new_key.float()
            v_b = value.float()
            v_a = new_value.float()
            log.warning(
                "[G4_78] layer=%d %s repeat=%dx tokens=%d "
                "key.shape %s -> %s value.shape %s -> %s "
                "replaced=True positions[:8]=%s target_slots[:8]=%s",
                layer_idx, target_kind,
                (DRAFTER_3_GQA_REPEAT if layer_idx == 3 else 1),
                int(key.shape[0]),
                tuple(key.shape), tuple(new_key.shape),
                tuple(value.shape), tuple(new_value.shape),
                positions[:8], target_slots[:8],
            )
            log.warning(
                "[G4_78] layer=%d key mean/std before=%.4e/%.4e -> "
                "after=%.4e/%.4e",
                layer_idx,
                k_b.mean().item(), k_b.std().item(),
                k_a.mean().item(), k_a.std().item(),
            )
            log.warning(
                "[G4_78] layer=%d value mean/std before=%.4e/%.4e -> "
                "after=%.4e/%.4e",
                layer_idx,
                v_b.mean().item(), v_b.std().item(),
                v_a.mean().item(), v_a.std().item(),
            )
        except Exception as _e:  # noqa: BLE001
            log.warning("[G4_78] layer=%d stats log failed: %s",
                        layer_idx, _e)

    new_args = list(args)
    new_args[2] = new_key
    new_args[3] = new_value
    return tuple(new_args)


def apply() -> tuple[str, str]:
    global _APPLIED, _ORIGINAL_FA_FORWARD, _ORIGINAL_TRITON_FORWARD

    if not _env_enabled():
        return "skipped", f"G4_78 disabled (set {_ENV_ENABLE}=1)"
    if _APPLIED:
        return "applied", "G4_78 already installed"

    log.warning("[G4_78] apply() entered (v2)")

    try:
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
    except Exception as e:  # noqa: BLE001
        log.warning("[G4_78] SKIP: TritonAttentionImpl import failed: %s", e)
        return "skipped", f"TritonAttentionImpl import failed: {e!r}"

    try:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    except Exception as e:  # noqa: BLE001
        log.warning("[G4_78] SKIP: FlashAttentionImpl import failed: %s", e)
        return "skipped", f"FlashAttentionImpl import failed: {e!r}"

    if getattr(TritonAttentionImpl.forward, "_genesis_g4_78_wrapped", False):
        _APPLIED = True
        return "applied", "Triton.forward already wrapped"

    _ORIGINAL_TRITON_FORWARD = TritonAttentionImpl.forward
    _ORIGINAL_FA_FORWARD = FlashAttentionImpl.forward

    import torch  # local import; torch guaranteed at this point

    def _triton_wrapped(self, *args, **kwargs):
        if len(args) < 1:
            return _ORIGINAL_TRITON_FORWARD(self, *args, **kwargs)
        layer = args[0]
        # Capture target state (passthrough)
        if _is_target_58(layer):
            _capture_target_state("target_58", args, kwargs)
            return _ORIGINAL_TRITON_FORWARD(self, *args, **kwargs)
        if _is_target_59(layer):
            _capture_target_state("target_59", args, kwargs)
            return _ORIGINAL_TRITON_FORWARD(self, *args, **kwargs)
        # Bridge for drafter[3] (Triton because head_size=512 via G4_75)
        layer_idx = _identify_drafter_layer_idx(layer)
        if layer_idx == 3:
            new_args = _try_bridge(
                layer_idx, args, kwargs,
                target_kind="target_59",
                block_size=TARGET_59_BLOCK_SIZE,
                torch_module=torch,
            )
            if new_args is not None:
                return _ORIGINAL_TRITON_FORWARD(self, *new_args, **kwargs)
        return _ORIGINAL_TRITON_FORWARD(self, *args, **kwargs)

    _triton_wrapped._genesis_g4_78_wrapped = True  # type: ignore[attr-defined]
    TritonAttentionImpl.forward = _triton_wrapped  # type: ignore[method-assign]

    def _fa_wrapped(self, *args, **kwargs):
        if len(args) < 6:
            return _ORIGINAL_FA_FORWARD(self, *args, **kwargs)
        layer = args[0]
        layer_idx = _identify_drafter_layer_idx(layer)
        if layer_idx not in (0, 1, 2):
            return _ORIGINAL_FA_FORWARD(self, *args, **kwargs)
        new_args = _try_bridge(
            layer_idx, args, kwargs,
            target_kind="target_58",
            block_size=TARGET_58_BLOCK_SIZE,
            torch_module=torch,
        )
        if new_args is not None:
            return _ORIGINAL_FA_FORWARD(self, *new_args, **kwargs)
        return _ORIGINAL_FA_FORWARD(self, *args, **kwargs)

    _fa_wrapped._genesis_g4_78_wrapped = True  # type: ignore[attr-defined]
    FlashAttentionImpl.forward = _fa_wrapped  # type: ignore[method-assign]

    _APPLIED = True
    log.warning(
        "[G4_78] v2 INSTALLED: target_58 + target_59 capture (Triton); "
        "drafter[0..2] bridge (FlashAttn target_58); drafter[3] bridge "
        "(Triton target_59 with GQA repeat=%d)",
        DRAFTER_3_GQA_REPEAT,
    )
    return "applied", "G4_78 v2 installed"


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED, _ORIGINAL_FA_FORWARD, _ORIGINAL_TRITON_FORWARD
    if not _APPLIED:
        return False
    try:
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
        if _ORIGINAL_TRITON_FORWARD is not None:
            TritonAttentionImpl.forward = _ORIGINAL_TRITON_FORWARD
    except ImportError:
        pass
    try:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
        if _ORIGINAL_FA_FORWARD is not None:
            FlashAttentionImpl.forward = _ORIGINAL_FA_FORWARD
    except ImportError:
        pass
    _APPLIED = False
    _ORIGINAL_FA_FORWARD = None
    _ORIGINAL_TRITON_FORWARD = None
    return True


__all__ = ["GENESIS_G4_78_MARKER", "apply", "is_applied", "revert"]
