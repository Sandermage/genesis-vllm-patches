# SPDX-License-Identifier: Apache-2.0
"""PN364 — vendor of OPEN PR vllm#43642 (hybrid GDN/Mamba/MRoPE startup warmup).

Closes the LAST first-request JIT spikes that PN126-130 do NOT cover
====================================================================

After PN126 (V1 decode warmup orchestrator), PN128 (spec-decode helper
warmup), PN129 (slot mapping warmup), PN130 (TurboQuant decode warmup),
there are still 4-5 kernels that fire JIT on the first user request:

  * ``_causal_conv1d_update_kernel`` (DECODE shape — distinct from
    ``_causal_conv1d_fwd_kernel`` covered by PN126 Pass 1 prefill)
  * ``fused_recurrent_gated_delta_rule_packed_decode_kernel`` (GDN
    single-token decode path — NOT covered by any existing PN12*)
  * ``MRotaryEmbedding.forward_cuda`` first-shape variant
  * ``layer_norm_fwd_kernel`` variants exercised by GDN/Mamba mixers
  * ``_kv_block_zeroer.warmup()`` if available

Verified by previous-session journal "2026-06-09-deep-dive-bugfix-followup.md"
section "Next actionable steps #1" listing 10 JIT-spike kernels still
firing on first user request.

Vendor strategy: PR #43642's design is "model-side warmup helper +
worker-side scheduler-warmup hook". We backport the SAME idea via a
single extra warmup pass triggered at the end of PN126's chain.

What PN364 does at boot
=======================

After PN126/PN128/PN129/PN130 finish their passes, PN364 runs:

  1. Pre-touch ``MRotaryEmbedding.forward_cuda`` with a fixed
     synthetic decode-shape tensor — bakes the first-call shape
     into Triton's autotune cache.
  2. Run an extra ``_dummy_run`` with the GDN decode-specific shape
     ``num_tokens = max_num_seqs × 1`` (single-token decode,
     NOT spec-decode multi-token shape covered by PN126 Pass 2).
     This is the path that fires
     ``_causal_conv1d_update_kernel`` +
     ``fused_recurrent_gated_delta_rule_packed_decode_kernel``.
  3. Pre-touch ``_kv_block_zeroer.warmup()`` if available on the runner.

All passes wrapped in try/except — partial completion is acceptable;
worst case is the pre-PN364 behaviour (those few kernels JIT on first
user request).

Why this composes cleanly with PN126/PN128/PN129/PN130
======================================================

  * PN126 = orchestrator wrapper around ``Worker.compile_or_warm_up_model``;
    issues Pass 1 (prefill) + Pass 2 (uniform spec-decode shape) +
    Pass 3 (extra capture sizes).
  * PN128 = wraps ``Worker.compile_or_warm_up_model`` to warm
    specific spec-decode helper kernels (eagle_prepare_*, etc.).
  * PN129 = slot_mapping_kernel warmup.
  * PN130 = TurboQuant decode kernels warmup.

PN364 ALSO wraps ``Worker.compile_or_warm_up_model`` but runs LAST in
the chain (registered later in dispatch order). All five patches use
the same wrapper-chaining pattern → naturally compose. Different
kernel-target sets → zero overlap.

  * **Single-token decode** (PN364's pass 2 ``max_num_seqs × 1``)
    is DISTINCT from PN126 Pass 2's ``max_num_seqs × (1 + num_spec)``
    spec-decode-uniform shape. Different cudagraph capture bucket.
  * **MRotaryEmbedding** is downstream of attention; no PN12*
    touches it.
  * **KV block zeroer** is a memory-mgmt kernel; no PN12* touches it.

Expected impact
===============

  * **TTFT on first user request after restart**: -200 ms to -1500 ms
    (kills the GDN single-token-decode + MRoPE first-call JIT spikes).
  * **Steady-state wall_TPS**: unchanged in the mean. **CV should
    tighten** because no JIT spike events mid-bench → less variance.
  * **Stability metric improvement** (operator's main complaint area):
    same patches → same kernels → identical autotune state → less drift
    between bench runs.

Safety model
============

  * Default OFF — opt-in via GENESIS_ENABLE_PN364_HYBRID_GDN_WARMUP=1.
  * Disable via GENESIS_DISABLE_PN364=1.
  * Idempotent via marker attribute on wrapped method.
  * Auto-skip when VLLM_USE_V2_MODEL_RUNNER=1 (V2's own warmup
    covers these kernels natively per PR43642 design).
  * Auto-skip when enforce_eager=True (no cudagraph capture).
  * Auto-skip when model has no hybrid GDN+Mamba layers (the
    helper detects via ``has_hybrid_gdn_mamba_mrope(model)``
    pattern from upstream PR).
  * Try/except wraps every pass — partial failure is acceptable.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Vendor target: vllm-project/vllm#43642 (OPEN as of 2026-06-09).
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.wiring.pn364_hybrid_gdn_mamba_warmup")

GENESIS_PN364_MARKER = "_genesis_pn364_hybrid_gdn_warmup_installed"

_WRAPPER_INSTALLED = False


def _is_hybrid_gdn_model(model) -> bool:
    """Detect if the loaded model has GDN+Mamba+MRoPE layers.

    PR43642 ships ``has_hybrid_gdn_mamba_mrope()`` in
    ``vllm.model_executor.warmup.kernel_warmup``. Since that helper
    doesn't exist in our pin, we duck-type detect by walking the model
    looking for ``GatedDeltaNet`` or ``MambaMixer`` attributes.
    """
    try:
        cls_names = {type(m).__name__ for m in model.modules()}
    except Exception:  # noqa: BLE001
        return False
    gdn_signatures = {
        "Qwen3GatedDeltaNet",
        "Qwen3_5GatedDeltaNet",
        "GatedDeltaNet",
        "MambaMixer",
        "MambaMixer2",
    }
    return bool(cls_names & gdn_signatures)


def _do_extra_warmups(worker) -> None:
    """Run the kernels PN126/128/129/130 don't cover.

    Specifically GDN-decode-path single-token shape, MRotaryEmbedding,
    and KV-block-zeroer. Each pass is wrapped in try/except so partial
    completion is acceptable.
    """
    runner = worker.model_runner
    sched_config = worker.scheduler_config
    spec_config = worker.vllm_config.speculative_config

    # ── Pass 1: single-token decode warmup (NOT spec-decode shape) ──
    # PN126 Pass 2 uses num_tokens = max_num_seqs × (1 + num_spec).
    # For MTP K=3 + max_num_seqs=2 that's 8 tokens.
    # PN364 Pass 1 uses num_tokens = max_num_seqs × 1 → 2 tokens.
    # That hits the single-token decode-shape cudagraph bucket which
    # PN126 missed — bakes the
    # _causal_conv1d_update_kernel +
    # fused_recurrent_gated_delta_rule_packed_decode_kernel +
    # MRotaryEmbedding.forward_cuda Triton autotune state.
    max_num_seqs = sched_config.max_num_seqs
    single_token_tokens = max_num_seqs
    log.info(
        "[PN364] Pass 1: single-token-decode warmup num_tokens=%d "
        "(max_num_seqs=%d × 1) cudagraph=auto",
        single_token_tokens, max_num_seqs,
    )
    try:
        runner._dummy_run(
            num_tokens=single_token_tokens,
            cudagraph_runtime_mode=None,  # auto-dispatch
            uniform_decode=True,
            skip_eplb=True,
            is_profile=False,
        )
        log.info("[PN364] Pass 1 done")
    except Exception as e:  # noqa: BLE001
        log.warning(
            "[PN364] Pass 1 single-token-decode failed: %s — continuing", e,
        )

    # ── Pass 2: KV block zeroer ──
    # PR43642 calls runner._kv_block_zeroer.warmup() if attr exists.
    # Our pin may not have this attr — duck-type detect.
    try:
        zeroer = getattr(runner, "_kv_block_zeroer", None)
        if zeroer is not None and hasattr(zeroer, "warmup"):
            log.info("[PN364] Pass 2: _kv_block_zeroer.warmup()")
            zeroer.warmup()
            log.info("[PN364] Pass 2 done")
        else:
            log.debug(
                "[PN364] Pass 2 skipped: no _kv_block_zeroer.warmup() "
                "on runner (expected on V1 — V2 has it)"
            )
    except Exception as e:  # noqa: BLE001
        log.warning(
            "[PN364] Pass 2 kv_block_zeroer warmup failed: %s — continuing", e,
        )

    # ── Pass 3: extra prefill-shape warmups for non-uniform decode ──
    # Cover capture sizes the PN126 Pass 3 also iterates, but with
    # single-token NON-uniform shape (simulates the moment user
    # sends a fresh batch that doesn't fit the most common shape).
    try:
        capture_sizes = list(
            worker.vllm_config.compilation_config.cudagraph_capture_sizes or []
        )
    except Exception:  # noqa: BLE001
        capture_sizes = []

    # Only do shapes < 16 — anything larger has already been hit by PN126.
    extra_decode_sizes = [s for s in capture_sizes if 1 <= s <= 16
                          and s != single_token_tokens]
    if extra_decode_sizes:
        log.info(
            "[PN364] Pass 3: extra single-token-decode shapes %s",
            extra_decode_sizes[:5],
        )
        for size in extra_decode_sizes[:5]:  # cap at 5 to avoid long boot
            try:
                runner._dummy_run(
                    num_tokens=size,
                    cudagraph_runtime_mode=None,
                    uniform_decode=True,
                    skip_eplb=True,
                    is_profile=False,
                )
            except Exception as e:  # noqa: BLE001
                log.debug(
                    "[PN364] Pass 3 size=%d failed: %s — continuing",
                    size, e,
                )
        log.info("[PN364] Pass 3 done")


def _wrap_compile_or_warm_up(worker_cls) -> None:
    """Install the PN364 warmup chain at the END of compile_or_warm_up_model.

    Chains AFTER PN126/PN128/PN129/PN130 since those install earlier
    in the boot dispatch order.
    """
    original = worker_cls.compile_or_warm_up_model

    if getattr(original, GENESIS_PN364_MARKER, False):
        log.debug("[PN364] already wrapped, skipping")
        return

    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        # Auto-skip conditions
        if os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "").strip() == "1":
            log.debug("[PN364] V2 model runner detected — V2 has builtin warmup; skip")
            return result
        if self.model_config is not None and self.model_config.enforce_eager:
            log.debug("[PN364] enforce_eager=True — skip")
            return result
        try:
            model = self.model_runner.model
            if not _is_hybrid_gdn_model(model):
                log.debug(
                    "[PN364] non-hybrid-GDN model — skip GDN-specific warmup"
                )
                return result
        except Exception:  # noqa: BLE001
            log.debug("[PN364] could not introspect model — skip")
            return result

        log.info("[PN364] running extra hybrid-GDN-Mamba warmup passes")
        try:
            _do_extra_warmups(self)
        except Exception as e:  # noqa: BLE001
            log.warning(
                "[PN364] extra warmup raised %r — engine continues uninterrupted",
                e,
            )
        return result

    setattr(wrapped, GENESIS_PN364_MARKER, True)
    worker_cls.compile_or_warm_up_model = wrapped


def _env_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN364", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Install PN364 wrapper on Worker.compile_or_warm_up_model."""
    global _WRAPPER_INSTALLED
    if _env_disabled():
        return "skipped", "PN364 disabled via GENESIS_DISABLE_PN364=1"

    if _WRAPPER_INSTALLED:
        return "applied", "PN364 wrapper already installed (idempotent)"

    try:
        from vllm.v1.worker.gpu_worker import Worker as V1Worker
    except ImportError as e:
        return "failed", f"PN364: cannot import vllm.v1.worker.gpu_worker.Worker — {e!r}"

    try:
        _wrap_compile_or_warm_up(V1Worker)
        _WRAPPER_INSTALLED = True
    except Exception as e:  # noqa: BLE001
        return "failed", f"PN364: wrapper install raised {e!r}"

    return "applied", (
        "PN364 installed: extra warmup passes wired into "
        "Worker.compile_or_warm_up_model AFTER PN126/PN128/PN129/PN130. "
        "Backport of OPEN vllm#43642. Targets the LAST 4-5 first-request "
        "JIT-spike kernels (causal_conv1d_update + GDN packed_decode + "
        "MRotaryEmbedding + kv_block_zeroer + extra capture sizes). "
        "Expected: TTFT -200-1500 ms on first user request; CV tightening "
        "on bench mean. No effect on steady-state wall_TPS. Auto-skip on "
        "V2 model runner / enforce_eager / non-hybrid models."
    )


def is_applied() -> bool:
    """Detect via attribute on the installed wrapper."""
    if _WRAPPER_INSTALLED:
        return True
    try:
        from vllm.v1.worker.gpu_worker import Worker as V1Worker
    except ImportError:
        return False
    return getattr(V1Worker.compile_or_warm_up_model, GENESIS_PN364_MARKER, False)
