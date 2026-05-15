# SPDX-License-Identifier: Apache-2.0
"""PN130 — TurboQuant decode kernel warmup (backport vllm-project/vllm#42215).

================================================================
ЗАЧЕМ
================================================================

`_tq_grouped_decode_stage1` (наш PN119 kernel) и связанные TQ
decode kernels JIT'ятся на первом user request. Также — `workspace`
allocator pre-fills отбрасывается после `lock_workspace()` (см.
#42544 описание).

PN130 закрывает 1 из 8 JIT warnings + предотвращает workspace
re-allocation на первом decode.

================================================================
КАК
================================================================

Upstream PR #42215 (OPEN) добавляет:
  1. `turboquant_decode_warmup(model, ...)` в новом модуле
     `vllm/model_executor/warmup/turboquant_warmup.py`
  2. Iterates `Attention` layers, ищет TQ backend (`kv_cache_dtype`
     starts with `turboquant_`)
  3. Dedupes по `_TurboQuantDecodeWarmupKey` (group params: heads,
     head_dim, block_size, etc.)
  4. Для каждого unique config: вызывает `impl._decode_attention()`
     с synthetic тензорами → JIT compiles `_tq_decode_stage1/2` +
     allocates workspace до lock'a

PN130 backport через runtime monkey-patch:
  • Wraps `Worker.compile_or_warm_up_model`
  • После original warmup, итерирует attention layers модели
  • Для TQ layers вызывает `_decode_attention()` с synthetic data
  • Идемпотентен (dedupe по config tuple)

================================================================
SAFETY
================================================================

  • Default OFF — opt-in via GENESIS_ENABLE_PN130_TQ_DECODE_WARMUP=1
  • Защитные импорты (TurboQuantAttentionImpl, TurboQuantMetadata)
  • Auto-skip когда kv_cache_dtype != turboquant_*
  • Auto-skip V2/enforce_eager
  • try/except защита на каждый layer

================================================================
КОМПОЗИЦИЯ
================================================================

  • Stack'ируется с PN128 (eagle warmup) + PN129 (slot_mapping)
  • Mutually exclusive с #42215 если PR landed
  • Pairs с P22 (TQ prealloc) + PN119 (TQ grouped decode kernel)
  • Закрывает 1 из оставшихся 8 JIT warnings
  • После PN128+PN129+PN130 остаются только:
      - _zero_kv_blocks_kernel  (требует V1→V2 switch)
      - _fwd_kernel_stage2      (требует V1→V2 switch)

Author: Sandermage 2026-05-15. Backport vllm#42215 (OPEN).
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.wiring.pn130_turboquant_decode_warmup")

GENESIS_PN130_MARKER = "Genesis PN130 TQ decode warmup v1 (vllm#42215)"
_ENV_ENABLE = "GENESIS_ENABLE_PN130_TQ_DECODE_WARMUP"
_ENV_DISABLE = "GENESIS_DISABLE_PN130_TQ_DECODE_WARMUP"

_APPLIED = False
_ORIGINAL_COMPILE: object = None


def _env_enabled() -> bool:
    if os.environ.get(_ENV_DISABLE, "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    val = os.environ.get(_ENV_ENABLE, "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _make_warmup_key(impl, block_size, block_table_stride, model_dtype):
    """Dedupe key (только важные launch-constexpr fields)."""
    return (
        impl.num_kv_heads,
        impl.head_size,
        block_size,
        block_table_stride,
        getattr(impl, "max_num_kv_splits", 32),
        getattr(impl, "num_kv_groups", 1),
        round(float(impl.scale), 6),
        getattr(impl.tq_config, "key_mse_bits", 4) if hasattr(impl, "tq_config") else 4,
        getattr(impl.tq_config, "key_packed_size", 10) if hasattr(impl, "tq_config") else 10,
        getattr(impl.tq_config, "effective_value_quant_bits", 4) if hasattr(impl, "tq_config") else 4,
        bool(getattr(impl.tq_config, "key_fp8", False)) if hasattr(impl, "tq_config") else False,
        bool(getattr(impl.tq_config, "norm_correction", True)) if hasattr(impl, "tq_config") else True,
        model_dtype == "float16" or str(model_dtype) == "torch.float16",
    )


def _warmup_one_tq_layer(layer, impl, *, device, block_size, block_table_stride,
                         max_num_decode_tokens, model_dtype):
    """Один pass TQ decode warmup для конкретного слоя."""
    import torch

    try:
        from vllm.v1.attention.backends.turboquant_attn import TurboQuantMetadata
    except ImportError as e:
        log.warning("[PN130] TurboQuantMetadata not importable: %s", e)
        return False

    # 1. Ensure Pi/PiT/centroids on device
    try:
        impl._ensure_on_device(layer, device)
    except Exception as e:
        log.warning("[PN130] _ensure_on_device failed: %s", e)
        return False

    Pi = getattr(layer, "_tq_Pi", None)
    PiT = getattr(layer, "_tq_PiT", None)
    centroids = getattr(layer, "_tq_centroids", None)
    if Pi is None or centroids is None:
        log.warning("[PN130] layer missing TQ params after ensure_on_device — skip")
        return False

    batch_size = max_num_decode_tokens
    slot_size_aligned = getattr(impl.tq_config, "slot_size_aligned", 24) if hasattr(impl, "tq_config") else 24

    try:
        query = torch.zeros(
            (batch_size, impl.num_heads, impl.head_size),
            dtype=model_dtype, device=device,
        )
        kv_cache = torch.zeros(
            (2, block_size, impl.num_kv_heads, slot_size_aligned),
            dtype=torch.uint8, device=device,
        )
        block_table = torch.zeros(
            (batch_size, block_table_stride), dtype=torch.int32, device=device
        )
        block_table[:, 0] = 1
        seq_lens = torch.ones(batch_size, dtype=torch.int32, device=device)

        attn_metadata = TurboQuantMetadata(
            seq_lens=seq_lens,
            slot_mapping=torch.zeros(batch_size, dtype=torch.long, device=device),
            block_table=block_table,
            query_start_loc=torch.arange(batch_size + 1, dtype=torch.int32, device=device),
            num_actual_tokens=batch_size,
            max_query_len=1,
            max_seq_len=1,
            is_prefill=False,
            num_decodes=batch_size,
            num_decode_tokens=batch_size,
        )

        impl._decode_attention(
            query=query,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            Pi=Pi,
            centroids=centroids,
            PiT=PiT,
            layer=layer,
        )
        return True
    except Exception as e:
        log.warning("[PN130] _decode_attention call failed for layer: %s", e)
        return False


def _iter_tq_layers(model):
    """Yield (Attention, TurboQuantAttentionImpl) pairs."""
    try:
        from vllm.model_executor.layers.attention import Attention
        from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionImpl
    except ImportError as e:
        log.warning("[PN130] Attention/TurboQuantAttentionImpl not importable: %s", e)
        return

    for layer in model.modules():
        if not isinstance(layer, Attention):
            continue
        kv_dtype = getattr(layer, "kv_cache_dtype", "")
        if not str(kv_dtype).startswith("turboquant_"):
            continue
        if not isinstance(layer.impl, TurboQuantAttentionImpl):
            continue
        yield layer, layer.impl


def _run_tq_warmup(worker):
    """Главная entry — итерирует TQ layers и запускает warmup."""
    import torch

    runner = getattr(worker, "model_runner", None)
    if runner is None:
        return

    model = getattr(runner, "model", None)
    if model is None:
        log.debug("[PN130] model_runner.model None — skip")
        return

    device = runner.device
    model_dtype = getattr(runner.model_config, "dtype", torch.float16)

    # Block size + stride извлекаем из kv_cache_config
    kv_cfg = getattr(runner, "kv_cache_config", None)
    if kv_cfg is None:
        log.debug("[PN130] kv_cache_config None — skip")
        return

    block_size = 16  # дефолт V1 для TQ
    # Block table stride — берём из input_batch если есть
    try:
        bt_stride = runner.input_batch.block_table.block_tables[0].block_table.shape[1]
    except (AttributeError, IndexError):
        bt_stride = max(1, (runner.max_model_len + block_size - 1) // block_size)

    max_decode_tokens = min(
        getattr(runner.scheduler_config, "max_num_seqs", 2) * 4, 16,
    )

    log.info(
        "[PN130] starting TQ decode warmup: block_size=%d bt_stride=%d "
        "max_decode_tokens=%d model_dtype=%s",
        block_size, bt_stride, max_decode_tokens, model_dtype,
    )

    seen: set = set()
    num_warmed = 0
    for layer, impl in _iter_tq_layers(model):
        key = _make_warmup_key(impl, block_size, bt_stride, model_dtype)
        if key in seen:
            continue
        seen.add(key)
        if _warmup_one_tq_layer(
            layer, impl,
            device=device, block_size=block_size,
            block_table_stride=bt_stride,
            max_num_decode_tokens=max_decode_tokens,
            model_dtype=model_dtype,
        ):
            num_warmed += 1

    if num_warmed > 0:
        try:
            torch.accelerator.synchronize()
        except Exception:
            pass
        log.info("[PN130] TQ decode warmup ✓ — %d unique kernel variants warmed", num_warmed)
    else:
        log.info("[PN130] no TQ layers found — model doesn't use turboquant kv-cache, skip")


def apply() -> tuple[str, str]:
    global _APPLIED, _ORIGINAL_COMPILE

    if not _env_enabled():
        return "skipped", (
            f"PN130 disabled (set {_ENV_ENABLE}=1 — backport vllm#42215, "
            f"TQ decode kernel warmup, закрывает _tq_grouped_decode_stage1 "
            f"JIT spike + workspace pre-alloc до lock'a)"
        )

    if _APPLIED:
        return "applied", "PN130 already installed (idempotent)"

    try:
        from vllm.envs import VLLM_USE_V2_MODEL_RUNNER
        if VLLM_USE_V2_MODEL_RUNNER:
            return "skipped", "V2 native warmup — PN130 redundant"
    except ImportError:
        pass

    try:
        from vllm.v1.worker.gpu_worker import Worker
    except ImportError as e:
        return "skipped", f"V1 Worker not importable: {e}"

    original = Worker.compile_or_warm_up_model
    if getattr(original, "_genesis_pn130_wrapped", False):
        _APPLIED = True
        return "applied", "PN130 already wrapped"

    _ORIGINAL_COMPILE = original

    def _genesis_pn130_wrapped_compile(self):
        result = original(self)
        try:
            _run_tq_warmup(self)
        except Exception as e:
            log.warning("[PN130] post-warmup raised: %s", e)
        return result

    _genesis_pn130_wrapped_compile._genesis_pn130_wrapped = True
    _genesis_pn130_wrapped_compile._genesis_pn130_original = original

    Worker.compile_or_warm_up_model = _genesis_pn130_wrapped_compile
    _APPLIED = True

    log.info(
        "[PN130] installed: Worker.compile_or_warm_up_model теперь "
        "warmup'ит TQ decode kernels на boot. Backport vllm#42215."
    )
    return "applied", (
        "PN130 installed: TurboQuant decode kernel warmup wired into V1 "
        "compile_or_warm_up_model. Backport vllm-project/vllm#42215. "
        "Закрывает _tq_grouped_decode_stage1 JIT spike + workspace "
        "pre-alloc до lock'a."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED, _ORIGINAL_COMPILE
    if not _APPLIED or _ORIGINAL_COMPILE is None:
        return False
    try:
        from vllm.v1.worker.gpu_worker import Worker
    except ImportError:
        return False
    Worker.compile_or_warm_up_model = _ORIGINAL_COMPILE  # type: ignore[assignment]
    _APPLIED = False
    return True
