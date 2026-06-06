# SPDX-License-Identifier: Apache-2.0
"""PN126 — V1 decode + spec-decode kernel warmup orchestrator.

================================================================
THE PROBLEM
================================================================

vLLM V1 model runner has incomplete JIT-kernel warmup at boot:

  ┌────────────────────────────────────────────────────────────┐
  │ V2 path  (use_v2_model_runner=True)                        │
  │   compile_or_warm_up_model →                               │
  │     warmup_kernels(model_runner, execute_model,            │
  │                    sample_tokens)                          │
  │   • Synthetic prefill batch through execute_model          │
  │   • Sample tokens                                          │
  │   • Synthetic decode batch with spec_decode_tokens         │
  │   • Cleanup                                                │
  │   → All hot-path Triton kernels JIT-compile at boot        │
  └────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────┐
  │ V1 path  (default — what we run)                           │
  │   compile_or_warm_up_model →                               │
  │     model_runner._dummy_run(                               │
  │       num_tokens=max_num_seqs,                             │
  │       cudagraph_runtime_mode=CUDAGraphMode.NONE)           │
  │     model_runner._dummy_sampler_run(...)                   │
  │   → ONLY sampler-related kernels warm up                   │
  │   → Decode-path kernels JIT on FIRST USER REQUEST          │
  │   → Spec-decode draft kernels JIT on FIRST USER REQUEST    │
  └────────────────────────────────────────────────────────────┘

This is observable via vLLM's own `jit_monitor.py` — after boot
completes (warmup activates JIT monitor), the FIRST user request
triggers 8-10 warnings of the form:

  WARNING jit_monitor.py:103 Triton kernel JIT compilation during
    inference: <name>. This causes a latency spike; consider
    extending warmup to cover this shape/config.

Observed kernels (35B-A3B-FP8 + TQ k8v4 + MTP K=3, our prod path):
  - _zero_kv_blocks_kernel             ─ KV scheduler
  - _compute_slot_mapping_kernel       ─ attention metadata
  - eagle_prepare_next_token_padded    ─ MTP draft
  - eagle_step_slot_mapping_metadata   ─ MTP draft
  - eagle_prepare_inputs_padded        ─ MTP draft
  - expand_kernel                      ─ spec-decode expansion
  - _tq_grouped_decode_stage1          ─ TQ decode (PN119)
  - _tq_full_dequant_kv                ─ TQ dequant
  - _fwd_kernel_stage2                 ─ Triton decode attention
  - _causal_conv1d_fwd_kernel          ─ Mamba prefill (decode shape)

Cost: 1-30 s per kernel JIT × 8-10 kernels on first request.
This matches our observed TTFT CV of 30%+ in benches.

================================================================
THE FIX
================================================================

Hook into Worker.compile_or_warm_up_model AFTER the existing V1
sampler-only warmup runs. Issue additional `_dummy_run()` calls
that exercise the decode + spec-decode code paths with proper
shapes:

  Pass 1: prefill warmup
    _dummy_run(num_tokens=max_num_batched_tokens,
               uniform_decode=False,
               cudagraph_runtime_mode=PIECEWISE)
    → triggers prefill attention + Mamba causal_conv1d at large T

  Pass 2: uniform decode warmup (1 + num_spec_tokens per req)
    decode_query_len = 1 + num_speculative_tokens
    _dummy_run(num_tokens=max_num_seqs * decode_query_len,
               uniform_decode=True,
               cudagraph_runtime_mode=FULL)
    → triggers decode attention + TQ kernels + spec-decode draft
      prep kernels with the exact query_len shape

Pass 1 covers _causal_conv1d_fwd_kernel + prefill attention.
Pass 2 covers all spec-decode + TQ + decode-attention kernels.

================================================================
SAFETY MODEL
================================================================

- Default OFF — opt-in via GENESIS_ENABLE_PN126_V1_DECODE_WARMUP=1
- Hooks AFTER existing warmup so we don't fight upstream's pass
- Wrapped in try/except — any failure logs WARNING and proceeds;
  worst case is the pre-PN126 behavior (JIT on first request)
- Idempotency via marker attribute on the wrapped method
- Auto-skip when use_v2_model_runner=True (V2 already does this)
- Auto-skip when enforce_eager=True (no cudagraph capture needed)

================================================================
EXPECTED IMPACT
================================================================

- Extra +3-10 s at boot (one prefill pass + one decode pass)
- TTFT on first request DROPS by ~5-25 s (no JIT mid-inference)
- Steady-state TPS: UNCHANGED (same kernels, just warmer)
- TTFT CV: should DROP from ~30% to ~10-15%
- jit_monitor warnings on first request: should drop from 8-10
  to 0-2 (the remaining ones are PIECEWISE-only kernels that
  upstream already warms, but for completeness Pass 1 covers them)

================================================================
COMPOSITION
================================================================

- Mutually exclusive with VLLM_USE_V2_MODEL_RUNNER=1 (V2 has
  this built-in; our extras would duplicate work). The patch
  detects V2 mode and self-skips.
- Safe with P66 (cudagraph_capture_sizes filter), P95 (Marlin TP
  cudagraph cap), P101 (FlashInfer FULL CG spec-decode) — all
  orthogonal layers.
- Safe with PN125 — both target cudagraph_mode setup, but at
  different layers (PN125 = config-time; PN126 = warmup-time).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Backports: V2 model runner's `warmup_kernels()` logic (location
           vllm/v1/worker/gpu/warmup.py in dev338+) into V1 path.
Source:    https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/
           (PR #39822 for SSD kernel warmup, related pattern)
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.wiring.pn126_v1_decode_kernel_warmup")

GENESIS_PN126_MARKER = (
    "Genesis PN126 V1 decode + spec-decode kernel warmup v1"
)

_ENV_ENABLE = "GENESIS_ENABLE_PN126_V1_DECODE_WARMUP"
_ENV_DISABLE = "GENESIS_DISABLE_PN126_V1_DECODE_WARMUP"

_APPLIED = False
_ORIGINAL_COMPILE: object = None


def _env_enabled() -> bool:
    """Default OFF — bench-gate required before flipping default."""
    if os.environ.get(_ENV_DISABLE, "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    val = os.environ.get(_ENV_ENABLE, "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _genesis_pn126_run_warmup_extras(worker) -> None:
    """Execute the extra warmup passes that V1 skips.

    Two passes:
      1. Prefill warmup (uniform_decode=False) — triggers prefill
         attention + Mamba causal_conv1d at decode-shape.
      2. Uniform-decode warmup — triggers decode attention + TQ
         decode + spec-decode draft prep kernels.

    All exceptions are swallowed and logged; the worst case is
    the pre-PN126 behavior (JIT on first request).
    """
    try:
        from vllm.config.compilation import CUDAGraphMode  # noqa: F401 — import-availability probe
    except ImportError:
        log.warning("[PN126] CUDAGraphMode not importable; skip extras")
        return

    if worker.model_config is not None and worker.model_config.enforce_eager:
        log.debug("[PN126] enforce_eager=True — no cudagraphs to warm; skip")
        return

    runner = worker.model_runner
    sched_config = worker.scheduler_config
    spec_config = worker.vllm_config.speculative_config

    max_num_seqs = sched_config.max_num_seqs
    max_num_batched = sched_config.max_num_batched_tokens

    # ────────────────────────────────────────────────────────
    # Pass 1: Prefill warmup — auto-dispatch (let runner decide)
    # ────────────────────────────────────────────────────────
    # NOTE 2026-05-15 fix: passing cudagraph_runtime_mode=PIECEWISE
    # explicitly triggered "Cudagraph runtime mode mismatch in
    # dummy_run. Expected NONE, but got PIECEWISE." The cudagraph
    # dispatcher selects NONE for large/non-uniform shapes regardless
    # of the global cudagraph_mode setting. Pass None so the runner's
    # built-in dispatcher resolves the right mode for this shape.
    prefill_tokens = min(max_num_batched, 4096)  # cap for safety
    log.info(
        "[PN126] Pass 1: prefill warmup num_tokens=%d cudagraph=auto",
        prefill_tokens,
    )
    try:
        runner._dummy_run(
            num_tokens=prefill_tokens,
            cudagraph_runtime_mode=None,  # auto-dispatch
            uniform_decode=False,
            skip_eplb=True,
            is_profile=False,
        )
        log.info("[PN126] Pass 1 done")
    except Exception as e:
        log.warning("[PN126] Pass 1 (prefill) failed: %s — continuing", e)

    # ────────────────────────────────────────────────────────
    # Pass 2: Uniform decode warmup — auto-dispatch (FULL for capture)
    # ────────────────────────────────────────────────────────
    # num_tokens = max_num_seqs × (1 + num_speculative_tokens). For our
    # MTP K=3 + max_num_seqs=2 this gives 8 tokens, which is in the
    # default cudagraph_capture_sizes list [4, 8, 16] on dev371. The
    # dispatcher picks FULL for that uniform-decode shape, so the
    # explicit FULL pass through worked. Switching to auto-dispatch
    # here keeps it robust if cudagraph_capture_sizes changes.
    num_spec_tokens = 0
    if spec_config is not None and spec_config.num_speculative_tokens is not None:
        num_spec_tokens = spec_config.num_speculative_tokens
    decode_query_len = 1 + num_spec_tokens
    decode_tokens = max_num_seqs * decode_query_len
    log.info(
        "[PN126] Pass 2: uniform decode num_tokens=%d (max_num_seqs=%d × "
        "1+spec_tokens=%d) cudagraph=auto",
        decode_tokens, max_num_seqs, decode_query_len,
    )
    try:
        runner._dummy_run(
            num_tokens=decode_tokens,
            cudagraph_runtime_mode=None,  # auto-dispatch
            uniform_decode=True,
            skip_eplb=True,
            is_profile=False,
        )
        log.info("[PN126] Pass 2 done")
    except Exception as e:
        log.warning(
            "[PN126] Pass 2 (uniform decode) failed: %s — continuing", e
        )

    # ────────────────────────────────────────────────────────
    # Pass 3: Additional cudagraph_capture_sizes coverage
    # ────────────────────────────────────────────────────────
    # The dispatcher's capture-size list (cudagraph_capture_sizes) on
    # dev371 defaults to [4, 8, 16] for max_num_seqs=2. Pass 2 covered
    # size=8 (max_num_seqs × decode_query_len). Iterate the remaining
    # captured sizes so user-request shapes that land on 4 or 16 don't
    # JIT mid-inference. Cheap: each call is a couple of ms once
    # kernels are cached.
    try:
        capture_sizes = list(
            worker.vllm_config.compilation_config.cudagraph_capture_sizes or []
        )
    except Exception:
        capture_sizes = []
    extra_sizes = [s for s in capture_sizes if s != decode_tokens]
    if extra_sizes:
        log.info(
            "[PN126] Pass 3: covering extra capture sizes %s (decode shapes)",
            extra_sizes,
        )
        for size in extra_sizes:
            try:
                runner._dummy_run(
                    num_tokens=size,
                    cudagraph_runtime_mode=None,  # auto-dispatch
                    uniform_decode=True,
                    skip_eplb=True,
                    is_profile=False,
                )
            except Exception as e:
                log.warning(
                    "[PN126] Pass 3 size=%d failed: %s — continuing", size, e,
                )

    log.info("[PN126] all extra warmup passes complete; first user request "
             "should hit cache for decode + spec-decode kernels")


def apply() -> tuple[str, str]:
    """Install the compile_or_warm_up_model wrapper. Never raises."""
    global _APPLIED, _ORIGINAL_COMPILE

    if not _env_enabled():
        return "skipped", (
            f"PN126 disabled (set {_ENV_ENABLE}=1 to enable extra "
            f"warmup pass on V1 model runner — fixes JIT spikes on first "
            f"request for hybrid_gdn_moe + spec-decode + TQ workloads)"
        )

    if _APPLIED:
        return "applied", "PN126 already installed (idempotent)"

    # Skip if V2 model runner is active — V2 has warmup_kernels built-in
    try:
        from vllm.envs import VLLM_USE_V2_MODEL_RUNNER
        if VLLM_USE_V2_MODEL_RUNNER:
            return "skipped", (
                "VLLM_USE_V2_MODEL_RUNNER=1 — V2 has warmup_kernels() "
                "built-in; PN126 is redundant and would duplicate work"
            )
    except ImportError:
        pass  # very old pin; assume V1

    try:
        from vllm.v1.worker.gpu_worker import Worker
    except ImportError as e:
        return "skipped", f"V1 Worker class not importable: {e}"

    if not hasattr(Worker, "compile_or_warm_up_model"):
        return "skipped", (
            "Worker.compile_or_warm_up_model not found — vllm pin layout "
            "drifted; PN126 needs anchor update"
        )

    original = Worker.compile_or_warm_up_model
    if getattr(original, "_genesis_pn126_wrapped", False):
        _APPLIED = True
        return "applied", "PN126 already wrapped (idempotent)"

    _ORIGINAL_COMPILE = original

    def _genesis_pn126_wrapped_compile(self):
        """Call original V1 warmup, then run PN126 extras."""
        result = original(self)
        try:
            _genesis_pn126_run_warmup_extras(self)
        except Exception as e:
            log.warning(
                "[PN126] extras warmup raised (%s); JIT spikes may still "
                "occur on first user request — falling back to pre-PN126 "
                "behavior", e,
            )
        return result

    _genesis_pn126_wrapped_compile._genesis_pn126_wrapped = True
    _genesis_pn126_wrapped_compile._genesis_pn126_original = original

    Worker.compile_or_warm_up_model = _genesis_pn126_wrapped_compile
    _APPLIED = True

    log.info(
        "[PN126] installed: V1 compile_or_warm_up_model now runs +2 "
        "extra dummy_run passes (prefill PIECEWISE + uniform decode FULL) "
        "to JIT-compile decode + spec-decode kernels at boot. Expected: "
        "lower TTFT CV (~30% → ~10-15%) on first user request."
    )
    return "applied", (
        "PN126 installed: extra prefill + uniform-decode warmup passes "
        "wired into V1 compile_or_warm_up_model. JIT-compiles decode + "
        "spec-decode kernels at boot instead of on first user request. "
        "Expected: lower TTFT spike on first request after container start."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    """Restore upstream Worker.compile_or_warm_up_model."""
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
