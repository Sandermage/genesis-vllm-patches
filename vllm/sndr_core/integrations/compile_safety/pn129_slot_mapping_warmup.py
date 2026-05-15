# SPDX-License-Identifier: Apache-2.0
"""PN129 — V1 slot mapping kernel warmup (backport vllm-project/vllm#42165).

================================================================
ЗАЧЕМ
================================================================

`_compute_slot_mapping_kernel` JIT'ится во время первого user-request
(см. JIT monitor warnings). Это потому что:
  1. Kernel @triton.jit без `do_not_specialize` → специализирует по
     `num_tokens` параметру → перекомпилируется на каждом новом
     batch size
  2. V1 _dummy_run не вызывает `block_table.compute_slot_mapping()`
     с настоящими kv blocks → kernel не warmed at boot

================================================================
КАК
================================================================

Upstream PR #42165 (OPEN) делает 2 вещи:

  1. **Структурный fix**: добавляет `do_not_specialize=["num_tokens"]`
     к `@triton.jit` decorator на `_compute_slot_mapping_kernel`.
     Single compilation для всех batch sizes — никакого пересборки
     при изменении num_tokens.

  2. **Warmup hook**: `warmup_v1_slot_mapping_kernel(model_runner)` —
     вызывает compute_slot_mapping с synthetic block_id=1 на 1
     request × 1 token → JIT compiles kernel до активации
     jit_monitor.

PN129 backport через runtime monkey-patch:
  • Monkey-patches `BlockTable._compute_slot_mapping_kernel`'s
    underlying triton.jit'ed function — добавляет
    `do_not_specialize` через декоратор reconfig (если возможно
    через triton API).
  • Wraps `Worker.compile_or_warm_up_model` для вызова
    warmup logic ДО `jit_monitor.activate()`.

================================================================
NB про do_not_specialize
================================================================

Triton JIT decorator's `do_not_specialize` контролируется через
JITFunction.do_not_specialize атрибут. Monkey-patch меняет:

  from vllm.v1.worker.block_table import _compute_slot_mapping_kernel
  _compute_slot_mapping_kernel.do_not_specialize = ("num_tokens",)
  _compute_slot_mapping_kernel.cache.clear()  # invalidate stale entries

Это **возможный** механизм — но рискованный (private Triton API).
Если он не работает на нашей версии Triton, остаётся только
warmup hook (вторая часть PR). Тогда warmup hit будет одной
compilation, и при первом user-request с другим num_tokens
JIT перекомпилируется ещё раз. Не идеально, но даёт +1 фикс
по сравнению с pre-PN129.

================================================================
SAFETY
================================================================

  • Default OFF — opt-in via GENESIS_ENABLE_PN129_SLOT_MAPPING_WARMUP=1
  • Защитные импорты + try/except
  • Auto-skip V2_MODEL_RUNNER + enforce_eager
  • Идемпотентен

Author: Sandermage 2026-05-15. Backport vllm#42165 (OPEN).
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.wiring.pn129_slot_mapping_warmup")

GENESIS_PN129_MARKER = "Genesis PN129 V1 slot mapping warmup v1 (vllm#42165)"
_ENV_ENABLE = "GENESIS_ENABLE_PN129_SLOT_MAPPING_WARMUP"
_ENV_DISABLE = "GENESIS_DISABLE_PN129_SLOT_MAPPING_WARMUP"

_APPLIED = False
_ORIGINAL_COMPILE: object = None


def _env_enabled() -> bool:
    if os.environ.get(_ENV_DISABLE, "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    val = os.environ.get(_ENV_ENABLE, "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _try_apply_do_not_specialize() -> bool:
    """Попытка добавить do_not_specialize="num_tokens" на упомянутый kernel.

    Triton public API не имеет post-init способа изменить
    `do_not_specialize`. Best-effort через private attribute access.
    Если не работает, warmup hook всё равно даёт частичное закрытие.
    """
    try:
        from vllm.v1.worker.block_table import _compute_slot_mapping_kernel
    except ImportError:
        log.warning("[PN129] _compute_slot_mapping_kernel not importable")
        return False
    try:
        # JITFunction в Triton 3.x хранит specialization config
        existing = getattr(_compute_slot_mapping_kernel, "do_not_specialize", None)
        if existing and "num_tokens" in existing:
            log.info("[PN129] do_not_specialize='num_tokens' already set")
            return True
        # Pickle JIT internals — это hacky path. Если упадёт — то ok,
        # выпадем на warmup-only mode.
        if hasattr(_compute_slot_mapping_kernel, "do_not_specialize"):
            new_list = list(existing or []) + ["num_tokens"]
            _compute_slot_mapping_kernel.do_not_specialize = new_list
            # Invalidate stale compiled binaries
            if hasattr(_compute_slot_mapping_kernel, "cache"):
                _compute_slot_mapping_kernel.cache.clear()
            log.info("[PN129] do_not_specialize='num_tokens' добавлен — single compilation для всех batch sizes")
            return True
    except Exception as e:
        log.warning("[PN129] do_not_specialize injection failed: %s — fallback to warmup-only", e)
    return False


def _run_slot_mapping_warmup(worker) -> None:
    """Запускает warmup_v1_slot_mapping_kernel логику на model_runner."""
    import torch

    runner = getattr(worker, "model_runner", None)
    if runner is None:
        return

    input_batch = getattr(runner, "input_batch", None)
    if input_batch is None:
        log.debug("[PN129] input_batch не доступен — skip")
        return
    block_table = getattr(input_batch, "block_table", None)
    if block_table is None:
        log.debug("[PN129] block_table не доступен — skip")
        return
    if not getattr(block_table, "block_tables", None):
        log.debug("[PN129] block_tables empty — skip")
        return

    kv_cfg = getattr(runner, "kv_cache_config", None)
    if kv_cfg is None or kv_cfg.num_blocks <= 1:
        log.debug("[PN129] kv_cache_config.num_blocks <= 1 — skip")
        return

    device = runner.device
    log.info("[PN129] starting slot_mapping warmup (block_id=1, 1 req × 1 token)...")

    # Setup ровно как в PR
    try:
        # Block 0 — null block. Используем block 1 (safe).
        block_table.add_row(tuple([1] for _ in block_table.block_tables), 0)
        block_table.commit_block_table(1)
        query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
        positions = torch.zeros(1, dtype=torch.int64, device=device)

        try:
            block_table.compute_slot_mapping(1, query_start_loc, positions)
            torch.accelerator.synchronize()
            log.info("[PN129] slot_mapping warmup ✓ — _compute_slot_mapping_kernel JIT'нулось на boot")
        finally:
            block_table.clear_row(0)
            block_table.commit_block_table(1)
    except Exception as e:
        log.warning("[PN129] slot_mapping warmup failed (%s) — kernel will JIT on first user request", e)


def apply() -> tuple[str, str]:
    global _APPLIED, _ORIGINAL_COMPILE

    if not _env_enabled():
        return "skipped", (
            f"PN129 disabled (set {_ENV_ENABLE}=1 — backport vllm#42165, "
            f"slot_mapping warmup + do_not_specialize, закрывает 1 из 8 "
            f"JIT spikes + структурный fix против пересборки по batch size)"
        )

    if _APPLIED:
        return "applied", "PN129 already installed (idempotent)"

    try:
        from vllm.envs import VLLM_USE_V2_MODEL_RUNNER
        if VLLM_USE_V2_MODEL_RUNNER:
            return "skipped", "V2 native warmup — PN129 redundant"
    except ImportError:
        pass

    try:
        from vllm.v1.worker.gpu_worker import Worker
    except ImportError as e:
        return "skipped", f"V1 Worker not importable: {e}"

    original = Worker.compile_or_warm_up_model
    if getattr(original, "_genesis_pn129_wrapped", False):
        _APPLIED = True
        return "applied", "PN129 already wrapped"

    _ORIGINAL_COMPILE = original

    # Step 1: попытка добавить do_not_specialize (структурный fix)
    dns_ok = _try_apply_do_not_specialize()

    # Step 2: wrap compile_or_warm_up_model для warmup hook
    def _genesis_pn129_wrapped_compile(self):
        result = original(self)
        try:
            _run_slot_mapping_warmup(self)
        except Exception as e:
            log.warning("[PN129] post-warmup raised: %s", e)
        return result

    _genesis_pn129_wrapped_compile._genesis_pn129_wrapped = True
    _genesis_pn129_wrapped_compile._genesis_pn129_original = original
    _genesis_pn129_wrapped_compile._genesis_pn129_dns_applied = dns_ok

    Worker.compile_or_warm_up_model = _genesis_pn129_wrapped_compile
    _APPLIED = True

    msg = (
        f"PN129 installed: slot_mapping warmup wired (vllm#42165). "
        f"do_not_specialize='num_tokens' "
        f"{'applied' if dns_ok else 'NOT applied (fallback to warmup-only)'}"
    )
    log.info("[PN129] %s", msg)
    return "applied", msg


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
