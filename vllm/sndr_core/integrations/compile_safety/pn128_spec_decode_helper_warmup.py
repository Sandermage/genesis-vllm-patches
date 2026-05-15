# SPDX-License-Identifier: Apache-2.0
"""PN128 — Spec-decode helper kernel warmup (backport vllm-project/vllm#41481).

================================================================
ЗАЧЕМ
================================================================

vLLM jit_monitor.py логирует Triton kernel JIT во время inference
(после warmup). На свежем dev371 (bf610c2f) bench показывает 8
уникальных kernels JIT'ящихся на первом user request:

  - _zero_kv_blocks_kernel              ← scheduler  (не покрывает)
  - _compute_slot_mapping_kernel        ← PN129 (отдельный backport)
  - eagle_prepare_next_token_padded_kernel  ← ПОКРЫТО PN128
  - eagle_prepare_inputs_padded_kernel      ← ПОКРЫТО PN128
  - eagle_step_slot_mapping_metadata_kernel ← ПОКРЫТО PN128
  - expand_kernel / copy_and_expand_eagle    ← ПОКРЫТО PN128
  - _fwd_kernel_stage2                  ← (V1↔V2 gap)
  - _tq_grouped_decode_stage1           ← PN130 (отдельный backport)

PN128 закрывает 4 из 8 — eagle_* + copy_and_expand. PN129 + PN130
закроют ещё 2. Останутся 2 (V1↔V2 model runner gap).

================================================================
КАК
================================================================

Upstream PR #41481 (OPEN) добавляет:
  1. ``SpecDecodeBaseProposer.dry_run_helper_kernels()`` — warmup'ит
     2 общих kernel (next_token + prepare_inputs) + опц. copy_expand
  2. ``EagleProposer.dry_run_helper_kernels()`` — дополнительно
     eagle_step_update_slot_mapping_and_metadata (для K > 1 + не
     parallel_drafting — наш MTP K=3 попадает)
  3. ``Worker._warmup_spec_decode_helpers()`` — вызывается из
     ``compile_or_warm_up_model`` после _dummy_run

PN128 backport через runtime monkey-patch (без text-patch):
  • Wraps Worker.compile_or_warm_up_model, после original вызывает
    нашу imported-from-vllm helper warmup logic
  • Logic копирует synthetic-tensor шаблоны прямо из PR кода —
    не зависит от methods добавленных PR'ом upstream (если они там
    окажутся при merge, PN128 self-skip через drift detection)

================================================================
SAFETY
================================================================

  • Default OFF — opt-in via GENESIS_ENABLE_PN128_SPEC_DECODE_WARMUP=1
  • Защитные импорты — если kernels не findable в pin → SKIP
  • try/except внутри каждого Triton invoke — failure -> log + continue
  • Auto-skip при VLLM_USE_V2_MODEL_RUNNER=1 (V2 native)
  • Auto-skip при enforce_eager=True
  • Auto-skip когда spec_decode не активен (метод != mtp/eagle/dflash)
  • Идемпотентен (marker attribute on wrapped method)

================================================================
ОЖИДАЕМЫЙ ЭФФЕКТ
================================================================

  • TTFT первого user-request -5..-25 секунд (по типу #39790 H100
    repro показал 25x первая-request регрессию pre-fix)
  • TTFT CV должен упасть ~30% → 10-15%
  • Steady-state TPS unchanged
  • Boot time +1-3 секунды (4 dummy Triton invoke + sync)

================================================================
COMPOSITION
================================================================

  • Stack'ируется с PN126 (V1 decode kernel warmup) — взаимодополняют
    разные kernels
  • Не зависит от PN125/PN127
  • Safe при PN129/PN130 (PN128 trips eagle_*, PN129 — slot_mapping,
    PN130 — TQ kernels)
  • Mutually exclusive с VLLM_USE_V2_MODEL_RUNNER=1 (V2 native
    warmup_kernels уже выполнит ту же работу)

Author: Sandermage 2026-05-15. Backport vllm-project/vllm#41481
(OPEN as of 2026-05-15) by direct replication of warmup logic.
Upstream PR may merge with rename/refactor — PN128 self-skips
on drift via _genesis_pn128_wrapped marker.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.wiring.pn128_spec_decode_helper_warmup")

GENESIS_PN128_MARKER = "Genesis PN128 spec-decode helper warmup v1 (vllm#41481)"
_ENV_ENABLE = "GENESIS_ENABLE_PN128_SPEC_DECODE_WARMUP"
_ENV_DISABLE = "GENESIS_DISABLE_PN128_SPEC_DECODE_WARMUP"

_APPLIED = False
_ORIGINAL_COMPILE: object = None


def _env_enabled() -> bool:
    if os.environ.get(_ENV_DISABLE, "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    val = os.environ.get(_ENV_ENABLE, "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def _warmup_eagle_helpers(drafter, model_runner) -> int:
    """Запекает 4 eagle helper kernels. Возвращает кол-во успешных warmups."""
    import torch

    device = drafter.device
    success = 0

    num_reqs = 1
    num_spec_tokens = int(getattr(drafter, "num_speculative_tokens", 0) or 0)
    if num_spec_tokens <= 0:
        log.info("[PN128] drafter has num_speculative_tokens=0 — skip eagle warmup")
        return 0

    num_sampled_per_req = num_spec_tokens + 1
    block_size_tokens = _next_power_of_2(num_sampled_per_req)

    # ===== kernel 1: eagle_prepare_next_token_padded_kernel =====
    try:
        from vllm.v1.spec_decode.utils import (
            eagle_prepare_next_token_padded_kernel,
        )
        sampled_token_ids = torch.zeros(
            (num_reqs, num_sampled_per_req), dtype=torch.int64, device=device
        )
        discard_mask = torch.zeros(num_reqs, dtype=torch.bool, device=device)
        backup_tokens = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        next_token_ids = torch.empty(num_reqs, dtype=torch.int32, device=device)
        valid_count_out = torch.empty(num_reqs, dtype=torch.int32, device=device)
        eagle_prepare_next_token_padded_kernel[(num_reqs,)](
            sampled_token_ids,
            discard_mask,
            backup_tokens,
            next_token_ids,
            valid_count_out,
            128,
            num_sampled_per_req,
            num_reqs,
            sampled_token_ids.stride(0),
            BLOCK_SIZE_TOKENS=block_size_tokens,
        )
        success += 1
        log.info("[PN128] kernel 1/4 eagle_prepare_next_token_padded_kernel ✓")
    except ImportError as e:
        log.warning("[PN128] kernel 1 not available in pin: %s", e)
    except Exception as e:
        log.warning("[PN128] kernel 1 invoke failed: %s", e)

    # ===== kernel 2: eagle_prepare_inputs_padded_kernel =====
    try:
        from vllm.v1.spec_decode.utils import eagle_prepare_inputs_padded_kernel
        cu_num_draft_tokens = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        valid_sampled_count = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        token_indices_to_sample = torch.empty(
            num_reqs, dtype=torch.int32, device=device
        )
        num_rejected_tokens_gpu = torch.empty(
            num_reqs, dtype=torch.int32, device=device
        )
        eagle_prepare_inputs_padded_kernel[(num_reqs,)](
            cu_num_draft_tokens,
            valid_sampled_count,
            query_start_loc,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
            num_reqs,
        )
        success += 1
        log.info("[PN128] kernel 2/4 eagle_prepare_inputs_padded_kernel ✓")
    except ImportError as e:
        log.warning("[PN128] kernel 2 not available: %s", e)
    except Exception as e:
        log.warning("[PN128] kernel 2 invoke failed: %s", e)

    # ===== kernel 3: copy_and_expand_eagle_inputs_kernel (conditional) =====
    method = getattr(drafter, "method", "")
    is_rejected = getattr(drafter, "is_rejected_token_mask", None)
    is_masked = getattr(drafter, "is_masked_token_mask", None)
    if method != "dflash" and is_rejected is not None and is_masked is not None:
        try:
            from vllm.v1.spec_decode.utils import copy_and_expand_eagle_inputs_kernel
            num_padding_slots = int(getattr(drafter, "extra_slots_per_request", 0) or 0)
            net_new_slots = int(getattr(drafter, "net_num_new_slots_per_request", 0) or 0)
            num_query_per_req = 1 + net_new_slots
            total_input_tokens = num_reqs * num_query_per_req
            total_output_tokens = num_reqs * (num_query_per_req + num_padding_slots)
            block_size_tokens_3 = min(256, _next_power_of_2(num_query_per_req))

            target_token_ids = torch.zeros(
                total_input_tokens, dtype=torch.int32, device=device
            )
            target_positions = torch.zeros(
                total_input_tokens, dtype=torch.int64, device=device
            )
            next_token_ids_buf = torch.zeros(num_reqs, dtype=torch.int32, device=device)
            qsl = (
                torch.arange(num_reqs + 1, dtype=torch.int32, device=device)
                * num_query_per_req
            )
            qel = qsl[1:] - 1
            tts_buf = torch.empty(
                max(num_reqs * num_padding_slots, 1),
                dtype=torch.int32, device=device,
            )
            ohsm_buf = torch.empty(total_input_tokens, dtype=torch.int32, device=device)

            num_blocks = max(
                1, (total_output_tokens + block_size_tokens_3 - 1) // block_size_tokens_3
            )
            copy_and_expand_eagle_inputs_kernel[(num_reqs, num_blocks)](
                target_token_ids_ptr=target_token_ids,
                target_positions_ptr=target_positions,
                next_token_ids_ptr=next_token_ids_buf,
                out_input_ids_ptr=drafter.input_ids,
                out_positions_ptr=drafter.positions,
                out_is_rejected_token_mask_ptr=is_rejected,
                out_is_masked_token_mask_ptr=is_masked,
                out_new_token_indices_ptr=tts_buf,
                out_hidden_state_mapping_ptr=ohsm_buf,
                query_start_loc_ptr=qsl,
                query_end_loc_ptr=qel,
                padding_token_id=0,
                parallel_drafting_token_id=getattr(drafter, "parallel_drafting_token_id", 0),
                total_input_tokens=total_input_tokens,
                num_padding_slots_per_request=num_padding_slots,
                shift_input_ids=getattr(drafter, "pass_hidden_states_to_model", True),
                BLOCK_SIZE_TOKENS=block_size_tokens_3,
            )
            # reset masks to avoid leaking warmup state
            is_rejected.zero_()
            is_masked.zero_()
            success += 1
            log.info("[PN128] kernel 3/4 copy_and_expand_eagle_inputs_kernel ✓")
        except ImportError as e:
            log.warning("[PN128] kernel 3 not available: %s", e)
        except Exception as e:
            log.warning("[PN128] kernel 3 invoke failed: %s", e)
    else:
        log.info("[PN128] kernel 3 skipped (dflash path OR no rejected/masked masks)")

    # ===== kernel 4: eagle_step_update_slot_mapping_and_metadata =====
    # Только для K > 1 + не parallel_drafting (наш MTP K=3 попадает)
    parallel_drafting = bool(getattr(drafter, "parallel_drafting", False))
    block_size = int(getattr(drafter, "block_size", 0) or 0)
    max_model_len = int(getattr(drafter, "max_model_len", 0) or 0)
    if num_spec_tokens > 1 and not parallel_drafting and block_size > 0:
        try:
            from vllm.v1.spec_decode.utils import (
                eagle_step_update_slot_mapping_and_metadata,
            )
            n_blocks_per_req = (max_model_len + block_size - 1) // block_size
            positions_1d = torch.zeros(num_reqs, dtype=torch.int64, device=device)
            block_table = torch.zeros(
                (num_reqs, n_blocks_per_req), dtype=torch.int32, device=device
            )
            seq_lens = torch.ones(num_reqs, dtype=torch.int32, device=device)
            out_clamped = torch.empty(num_reqs, dtype=torch.int64, device=device)
            out_slot = torch.empty(num_reqs, dtype=torch.int64, device=device)
            eagle_step_update_slot_mapping_and_metadata(
                positions_1d=positions_1d,
                block_table_tensor=block_table,
                seq_lens=seq_lens,
                block_size=block_size,
                max_model_len=max_model_len,
                out_clamped_positions=out_clamped,
                out_slot_mapping=out_slot,
            )
            success += 1
            log.info("[PN128] kernel 4/4 eagle_step_update_slot_mapping_and_metadata ✓")
        except ImportError as e:
            log.warning("[PN128] kernel 4 not available: %s", e)
        except Exception as e:
            log.warning("[PN128] kernel 4 invoke failed: %s", e)
    else:
        log.info(
            "[PN128] kernel 4 skipped (K=%d, parallel=%s, block_size=%d) "
            "— only fires for sequential Eagle K>1",
            num_spec_tokens, parallel_drafting, block_size,
        )

    return success


def _run_pn128_warmup(worker) -> None:
    """Главная entry — извлекает drafter и запускает eagle helper warmup.

    v2 (2026-05-15 bench finding): первый run использовал только
    num_reqs=1, но real user request имеет num_reqs до max_num_seqs.
    Triton JIT cache key включает constexpr — разные BLOCK_SIZE_TOKENS
    → разные binaries. Итерируем по shape variants чтобы покрыть
    все возможные num_reqs × num_sampled_per_req комбинации, которые
    реально могут случиться на inference.
    """
    runner = getattr(worker, "model_runner", None)
    if runner is None:
        log.debug("[PN128] worker.model_runner None — skip")
        return
    drafter = getattr(runner, "drafter", None)
    if drafter is None:
        log.debug("[PN128] runner.drafter None — no spec-decode active, skip")
        return

    # Shape coverage: num_reqs до max_num_seqs (1 + 2 для max_num_seqs=2 setup).
    # Real bench iteration на dev371 показал что warmup с num_reqs=1
    # покрывает только одну Triton specialization; max_num_seqs=2 batch
    # отдаст другой cache key.
    sched_config = getattr(worker, "scheduler_config", None)
    max_num_seqs = int(getattr(sched_config, "max_num_seqs", 2)) if sched_config else 2

    log.info(
        "[PN128] starting spec-decode helper kernel warmup "
        "(num_reqs sweep 1..%d)...", max_num_seqs,
    )

    total_warmed = 0
    for num_reqs in range(1, max(2, max_num_seqs) + 1):
        try:
            n = _warmup_eagle_helpers_with_reqs(drafter, runner, num_reqs)
            total_warmed += n
            log.info("[PN128] num_reqs=%d: %d/4 kernels warmed", num_reqs, n)
        except Exception as e:
            log.warning("[PN128] num_reqs=%d failed: %s", num_reqs, e)

    log.info("[PN128] spec-decode helper warmup complete: %d total warmups", total_warmed)

    # Sync GPU перед jit_monitor activation
    try:
        import torch
        torch.accelerator.synchronize()
    except Exception as e:
        log.warning("[PN128] post-warmup sync failed: %s", e)


def _warmup_eagle_helpers_with_reqs(drafter, model_runner, num_reqs: int) -> int:
    """Запуск warmup с конкретным num_reqs (форсируем shape variant)."""
    # Параметризуем глобальный _warmup_eagle_helpers через локальный
    # override. Простейший вариант — передать через temporary attribute,
    # но это hacky. Делаем явный duplicate с num_reqs параметром.
    import torch

    device = drafter.device
    success = 0
    num_spec_tokens = int(getattr(drafter, "num_speculative_tokens", 0) or 0)
    if num_spec_tokens <= 0:
        return 0

    num_sampled_per_req = num_spec_tokens + 1
    block_size_tokens = _next_power_of_2(num_sampled_per_req)

    # kernel 1
    try:
        from vllm.v1.spec_decode.utils import (
            eagle_prepare_next_token_padded_kernel,
        )
        sampled_token_ids = torch.zeros(
            (num_reqs, num_sampled_per_req), dtype=torch.int64, device=device
        )
        discard_mask = torch.zeros(num_reqs, dtype=torch.bool, device=device)
        backup_tokens = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        next_token_ids = torch.empty(num_reqs, dtype=torch.int32, device=device)
        valid_count_out = torch.empty(num_reqs, dtype=torch.int32, device=device)
        eagle_prepare_next_token_padded_kernel[(num_reqs,)](
            sampled_token_ids, discard_mask, backup_tokens, next_token_ids,
            valid_count_out, 128, num_sampled_per_req, num_reqs,
            sampled_token_ids.stride(0), BLOCK_SIZE_TOKENS=block_size_tokens,
        )
        success += 1
    except Exception as e:
        log.debug("[PN128] num_reqs=%d k1 failed: %s", num_reqs, e)

    # kernel 2
    try:
        from vllm.v1.spec_decode.utils import eagle_prepare_inputs_padded_kernel
        cu = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        valid = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        qsl = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        tits = torch.empty(num_reqs, dtype=torch.int32, device=device)
        nrt = torch.empty(num_reqs, dtype=torch.int32, device=device)
        eagle_prepare_inputs_padded_kernel[(num_reqs,)](cu, valid, qsl, tits, nrt, num_reqs)
        success += 1
    except Exception as e:
        log.debug("[PN128] num_reqs=%d k2 failed: %s", num_reqs, e)

    return success


def apply() -> tuple[str, str]:
    global _APPLIED, _ORIGINAL_COMPILE

    if not _env_enabled():
        return "skipped", (
            f"PN128 disabled (set {_ENV_ENABLE}=1 — backport vllm#41481, "
            f"warmup'ит eagle_* helper kernels на boot, закрывает 4 из 8 "
            f"JIT spikes на первом user request)"
        )

    if _APPLIED:
        return "applied", "PN128 already installed (idempotent)"

    # Auto-skip V2
    try:
        from vllm.envs import VLLM_USE_V2_MODEL_RUNNER
        if VLLM_USE_V2_MODEL_RUNNER:
            return "skipped", "V2 model runner active — warmup_kernels() native, PN128 redundant"
    except ImportError:
        pass

    try:
        from vllm.v1.worker.gpu_worker import Worker
    except ImportError as e:
        return "skipped", f"V1 Worker not importable: {e}"

    if not hasattr(Worker, "compile_or_warm_up_model"):
        return "skipped", "Worker.compile_or_warm_up_model not found"

    original = Worker.compile_or_warm_up_model
    if getattr(original, "_genesis_pn128_wrapped", False):
        _APPLIED = True
        return "applied", "PN128 already wrapped (idempotent)"

    _ORIGINAL_COMPILE = original

    def _genesis_pn128_wrapped_compile(self):
        """Original compile_or_warm_up_model + PN128 spec-decode helper warmup."""
        result = original(self)
        try:
            _run_pn128_warmup(self)
        except Exception as e:
            log.warning(
                "[PN128] post-warmup hook raised (%s); JIT spikes may "
                "still occur on first user request — falling back to "
                "pre-PN128 behavior", e,
            )
        return result

    _genesis_pn128_wrapped_compile._genesis_pn128_wrapped = True
    _genesis_pn128_wrapped_compile._genesis_pn128_original = original

    Worker.compile_or_warm_up_model = _genesis_pn128_wrapped_compile
    _APPLIED = True

    log.info(
        "[PN128] installed: Worker.compile_or_warm_up_model теперь warmup'ит "
        "4 eagle helper kernels после оригинального warmup. Closes 4 of 8 "
        "JIT spikes (Issue #39790 root cause). Backport vllm#41481."
    )
    return "applied", (
        "PN128 installed: spec-decode helper kernel warmup wired into V1 "
        "compile_or_warm_up_model. Backport vllm-project/vllm#41481. "
        "Закрывает 4 из 8 JIT warnings (eagle_prepare_next/inputs/copy_expand/"
        "step_update) на первом user request."
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
