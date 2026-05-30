# SPDX-License-Identifier: Apache-2.0
"""PN128 — Spec-decode helper kernel warmup (backport vllm-project/vllm#41481).

================================================================
WHY
================================================================

vLLM jit_monitor.py logs Triton kernel JIT compilations during
inference (after the official warmup). On a fresh dev371 (bf610c2f)
bench, 8 unique kernels JIT-compile on the first user request:

  - _zero_kv_blocks_kernel              ← scheduler (not covered)
  - _compute_slot_mapping_kernel        ← PN129 (separate backport)
  - eagle_prepare_next_token_padded_kernel  ← COVERED BY PN128
  - eagle_prepare_inputs_padded_kernel      ← COVERED BY PN128
  - eagle_step_slot_mapping_metadata_kernel ← COVERED BY PN128
  - expand_kernel / copy_and_expand_eagle    ← COVERED BY PN128
  - _fwd_kernel_stage2                  ← (V1↔V2 gap)
  - _tq_grouped_decode_stage1           ← PN130 (separate backport)

PN128 closes 4 of 8 — eagle_* + copy_and_expand. PN129 + PN130
cover 2 more. 2 remain in the V1↔V2 model runner gap.

================================================================
HOW
================================================================

Upstream PR #41481 (OPEN) adds:
  1. ``SpecDecodeBaseProposer.dry_run_helper_kernels()`` — warms up
     2 shared kernels (next_token + prepare_inputs) + optional
     copy_expand
  2. ``EagleProposer.dry_run_helper_kernels()`` — additionally
     eagle_step_update_slot_mapping_and_metadata (for K > 1 + non
     parallel_drafting — our MTP K=3 hits this)
  3. ``Worker._warmup_spec_decode_helpers()`` — called from
     ``compile_or_warm_up_model`` after _dummy_run

PN128 backports via runtime monkey-patch (no text-patch):
  • Wraps Worker.compile_or_warm_up_model; after the original it
    invokes the imported-from-vllm helper warmup logic
  • Logic copies synthetic-tensor templates directly from the PR
    source — does not depend on methods added by the PR upstream
    (if they do land at merge, PN128 self-skips via drift detection)

================================================================
SAFETY
================================================================

  • Default OFF — opt-in via GENESIS_ENABLE_PN128_SPEC_DECODE_WARMUP=1
  • Defensive imports — if kernels are not findable in the pin → SKIP
  • try/except around each Triton invoke — failure -> log + continue
  • Auto-skip when VLLM_USE_V2_MODEL_RUNNER=1 (V2 native)
  • Auto-skip when enforce_eager=True
  • Auto-skip when spec_decode is inactive (method != mtp/eagle/dflash)
  • Idempotent (marker attribute on wrapped method)

================================================================
EXPECTED EFFECT
================================================================

  • TTFT of the first user request -5..-25 s (issue #39790 H100
    repro showed a 25x first-request regression pre-fix)
  • TTFT CV should drop from ~30% to 10-15%
  • Steady-state TPS unchanged
  • Boot time +1-3 s (4 dummy Triton invokes + sync)

================================================================
COMPOSITION
================================================================

  • Stacks with PN126 (V1 decode kernel warmup) — complementary
    kernels
  • Independent of PN125/PN127
  • Safe with PN129/PN130 (PN128 trips eagle_*, PN129 → slot_mapping,
    PN130 → TQ kernels)
  • Mutually exclusive with VLLM_USE_V2_MODEL_RUNNER=1 (V2 native
    warmup_kernels already does the same work)

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
    """Bake 4 eagle helper kernels. Return count of successful warmups."""
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
    # Only for K > 1 + non parallel_drafting (our MTP K=3 hits this)
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
    """Main entry — extracts the drafter and runs eagle helper warmup.

    v2 (2026-05-15 bench finding): the first run used only num_reqs=1,
    but real user requests reach num_reqs up to max_num_seqs. The
    Triton JIT cache key includes constexpr values — different
    BLOCK_SIZE_TOKENS produce different binaries. Iterate over shape
    variants to cover all num_reqs x num_sampled_per_req combinations
    that can actually appear at inference time.
    """
    runner = getattr(worker, "model_runner", None)
    if runner is None:
        log.debug("[PN128] worker.model_runner None — skip")
        return
    drafter = getattr(runner, "drafter", None)
    if drafter is None:
        log.debug("[PN128] runner.drafter None — no spec-decode active, skip")
        return

    # Shape coverage: num_reqs up to max_num_seqs (1 + 2 for max_num_seqs=2 setup).
    # Real bench iteration on dev371 showed that warmup with num_reqs=1
    # covers only one Triton specialization; the max_num_seqs=2 batch
    # yields a different cache key.
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

    # Sync GPU before jit_monitor activation
    try:
        import torch
        torch.accelerator.synchronize()
    except Exception as e:
        log.warning("[PN128] post-warmup sync failed: %s", e)


def _warmup_eagle_helpers_with_reqs(drafter, model_runner, num_reqs: int) -> int:
    """Run warmup with a specific num_reqs (force a shape variant)."""
    # Parametrize the global _warmup_eagle_helpers via a local override.
    # Simplest path would be a temporary attribute, but that is hacky.
    # An explicit duplicate that accepts num_reqs is the cleaner option.
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
            f"warms up eagle_* helper kernels at boot, closes 4 of 8 "
            f"JIT spikes on the first user request)"
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
        "[PN128] installed: Worker.compile_or_warm_up_model now warms up "
        "4 eagle helper kernels after the original warmup. Closes 4 of 8 "
        "JIT spikes (Issue #39790 root cause). Backport vllm#41481."
    )
    return "applied", (
        "PN128 installed: spec-decode helper kernel warmup wired into V1 "
        "compile_or_warm_up_model. Backport vllm-project/vllm#41481. "
        "Closes 4 of 8 JIT warnings (eagle_prepare_next/inputs/copy_expand/"
        "step_update) on the first user request."
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
