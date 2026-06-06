# SPDX-License-Identifier: Apache-2.0
"""PN129 — V1 slot mapping kernel warmup (backport vllm-project/vllm#42165).

================================================================
WHY
================================================================

`_compute_slot_mapping_kernel` JIT-compiles during the first user
request (see JIT monitor warnings). Two root causes:
  1. The kernel is @triton.jit without `do_not_specialize`, so it
     specialises on the `num_tokens` parameter and is recompiled on
     every new batch size.
  2. V1 _dummy_run does not invoke `block_table.compute_slot_mapping()`
     with real kv blocks, so the kernel is not warmed at boot.

================================================================
HOW
================================================================

Upstream PR #42165 (OPEN) does two things:

  1. **Structural fix**: adds `do_not_specialize=["num_tokens"]`
     to the `@triton.jit` decorator on `_compute_slot_mapping_kernel`.
     Single compilation for all batch sizes — no recompiles when
     num_tokens changes.

  2. **Warmup hook**: `warmup_v1_slot_mapping_kernel(model_runner)`
     invokes compute_slot_mapping with a synthetic block_id=1 over
     1 request x 1 token, JIT-compiling the kernel before
     jit_monitor activates.

PN129 backports via runtime monkey-patch:
  • Monkey-patches `BlockTable._compute_slot_mapping_kernel`'s
    underlying triton.jit'ed function — adds `do_not_specialize`
    via decorator reconfig (where the Triton API allows it).
  • Wraps `Worker.compile_or_warm_up_model` to invoke the warmup
    logic BEFORE `jit_monitor.activate()`.

================================================================
NOTE on do_not_specialize
================================================================

Triton's `do_not_specialize` is controlled through the private
JITFunction.do_not_specialize attribute. The monkey-patch sets:

  from vllm.v1.worker.block_table import _compute_slot_mapping_kernel
  _compute_slot_mapping_kernel.do_not_specialize = ("num_tokens",)
  _compute_slot_mapping_kernel.cache.clear()  # invalidate stale entries

This is a **possible** mechanism, but risky (private Triton API).
If it does not work on our Triton version, only the warmup hook
(part 2 of the PR) remains. The warmup hit then covers a single
compilation, and a first user request with a different num_tokens
will JIT-recompile once more. Not ideal, but still +1 fix versus
pre-PN129.

================================================================
SAFETY
================================================================

  • Default OFF — opt-in via GENESIS_ENABLE_PN129_SLOT_MAPPING_WARMUP=1
  • Defensive imports + try/except
  • Auto-skip V2_MODEL_RUNNER + enforce_eager
  • Idempotent

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
    """Attempt to add do_not_specialize="num_tokens" on the named kernel.

    Triton's public API has no post-init way to change
    `do_not_specialize`. Best-effort via private attribute access.
    If this fails the warmup hook still provides partial closure.
    """
    try:
        from vllm.v1.worker.block_table import _compute_slot_mapping_kernel
    except ImportError:
        log.warning("[PN129] _compute_slot_mapping_kernel not importable")
        return False
    try:
        # JITFunction in Triton 3.x carries the specialization config.
        existing = getattr(_compute_slot_mapping_kernel, "do_not_specialize", None)
        if existing and "num_tokens" in existing:
            log.info("[PN129] do_not_specialize='num_tokens' already set")
            return True
        # Poking JIT internals — a hacky path. If it raises, we
        # fall back to warmup-only mode.
        if hasattr(_compute_slot_mapping_kernel, "do_not_specialize"):
            new_list = list(existing or []) + ["num_tokens"]
            _compute_slot_mapping_kernel.do_not_specialize = new_list
            # Invalidate stale compiled binaries
            if hasattr(_compute_slot_mapping_kernel, "cache"):
                _compute_slot_mapping_kernel.cache.clear()
            log.info("[PN129] do_not_specialize='num_tokens' added — single compilation for all batch sizes")
            return True
    except Exception as e:
        log.warning("[PN129] do_not_specialize injection failed: %s — fallback to warmup-only", e)
    return False


def _run_slot_mapping_warmup(worker) -> None:
    """Run the warmup_v1_slot_mapping_kernel logic on model_runner."""
    import torch

    runner = getattr(worker, "model_runner", None)
    if runner is None:
        return

    input_batch = getattr(runner, "input_batch", None)
    if input_batch is None:
        log.debug("[PN129] input_batch unavailable — skip")
        return
    block_table = getattr(input_batch, "block_table", None)
    if block_table is None:
        log.debug("[PN129] block_table unavailable — skip")
        return
    if not getattr(block_table, "block_tables", None):
        log.debug("[PN129] block_tables empty — skip")
        return

    kv_cfg = getattr(runner, "kv_cache_config", None)
    if kv_cfg is None or kv_cfg.num_blocks <= 1:
        log.debug("[PN129] kv_cache_config.num_blocks <= 1 — skip")
        return

    device = runner.device
    log.info("[PN129] starting slot_mapping warmup (block_id=1, 1 req x 1 token)...")

    # Setup matches the PR exactly.
    try:
        # Block 0 is the null block. Use block 1 (safe).
        block_table.add_row(tuple([1] for _ in block_table.block_tables), 0)
        block_table.commit_block_table(1)
        query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
        positions = torch.zeros(1, dtype=torch.int64, device=device)

        try:
            block_table.compute_slot_mapping(1, query_start_loc, positions)
            torch.accelerator.synchronize()
            log.info("[PN129] slot_mapping warmup ✓ — _compute_slot_mapping_kernel JIT'd at boot")
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
            f"slot_mapping warmup + do_not_specialize, closes 1 of 8 "
            f"JIT spikes + structural fix preventing recompile-on-batch-size churn)"
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

    # Step 1: attempt do_not_specialize (structural fix)
    dns_ok = _try_apply_do_not_specialize()

    # Step 2: wrap compile_or_warm_up_model for the warmup hook
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
