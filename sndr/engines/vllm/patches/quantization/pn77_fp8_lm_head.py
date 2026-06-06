# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN77 — FP8 lm_head compression (Phase E.5 redesign).

ARCHITECTURAL CHANGE FROM PHASE E.2-3
======================================

Phase E.2-3 (BROKEN): two text-patches —
  1. `qwen3_5.py:Qwen3_5ForCausalLMBase.load_weights` post-hook — replaced
     `lm_head.weight = nn.Parameter(weight_fp8)` directly. ORPHANED `weight_loader`
     callback → TP-shard fail on second-iteration weight load.
  2. `vocab_parallel_embedding.py:UnquantizedEmbeddingMethod.apply` — decompress
     hook for cast-back path.

Phase E.5 (NEW): single text-patch + subclass —
  1. Text-patch on `model_loader/utils.py:process_weights_after_loading` walker
     to inject `maybe_swap_pn77_quant_method(module, quant_method)` BEFORE
     `quant_method.process_weights_after_loading(module)` is called.
  2. Subclass `Genesis_FP8_LMHead_EmbeddingMethod` (in `kernels/lm_head_fp8_method.py`)
     overrides `process_weights_after_loading` (using `replace_parameter` —
     preserves `weight_loader`!) and `apply` (hardware-tier dispatch:
     Marlin on Ampere, scaled_mm on Ada+, cast-back fallback).

WHY THIS WORKS
===============

1. `process_weights_after_loading` is vllm's CANONICAL post-load hook. Used by
   all native quant methods (`Fp8LinearMethod`, AWQ Marlin, GPTQ, etc.). Fires
   AFTER `load_weights()` fully returns AND `tie_weights` resolved AND
   `device_loading_context` is set. No re-iteration of weight load happens
   after this point — Parameter swap is safe.

2. `replace_parameter()` (from `vllm.model_executor.utils`) preserves the
   `weight_loader` attribute on the new Parameter — proven pattern in
   `Fp8LinearMethod.process_weights_after_loading` (`fp8.py:530`).

3. Generic — works on ANY model class with a `ParallelLMHead` (Llama, Mistral,
   Qwen3, Qwen3.5, Qwen3.6 hybrid, future Qwen3.7+). No model-specific
   text-patch needed.

4. Hardware-aware — subclass dispatches to Marlin/scaled_mm/cast_back at
   `apply()` time based on detected GPU tier.

5. Drift-resistant — when upstream lands PR #41000 (config-driven Fp8Config
   ParallelLMHead support), `lm_head_quantized` marker in `fp8.py` triggers
   self-retire.

SAFETY MODEL
=============

- Default OFF (opt-in via `GENESIS_ENABLE_PN77_FP8_LM_HEAD=1`).
- Single text-patch with `required=True` anchor — if upstream walker shape
  changes → SKIPPED, source stays vanilla, zero regression.
- Helper `maybe_swap_pn77_quant_method` always returns (never raises) — env
  unset, non-lm_head, hardware unsupported → returns original method.
- Idempotent via `_already_called_process_weights_after_loading` marker.
- Tied embeddings: `replace_parameter` doc explicitly says "should not be
  used on tied/shared param" — handled by guard in helper. Tied lm_head
  has `weight.data_ptr() == embed_tokens.weight.data_ptr()` — detected and
  skipped.

EXPECTED IMPACT
===============

- 27B Qwen3.6 (vocab=248320, hidden=5120, BF16 lm_head):
  ~606 MiB/rank saved on 2× A5000 = ~1212 MiB total.
- 35B Qwen3.6-A3B (hidden=2048): ~243 MiB/rank.
- TPS impact:
  - Ampere (Marlin): ~0% — weight-only FP8, compute parity vs BF16
  - Ada/Hopper (scaled_mm): +1-3% — native FP8 GEMM faster than BF16
  - Cast-back fallback: ~3 ms/token cast cost, absorbed by spec-decode

Author: Sandermage(Sander) Barzov Aleksandr — Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

from vllm.sndr_core.detection.guards import resolve_vllm_file
from vllm.sndr_core.core import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.pn77_fp8_lm_head")

GENESIS_PN77_MARKER = (
    "Genesis PN77 FP8 lm_head — process_weights_after_loading swap (E.5) v7.71"
)

_ENV_FLAG = "GENESIS_ENABLE_PN77_FP8_LM_HEAD"


def _is_enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "").strip().lower() in (
        "1", "true", "yes", "y", "on",
    )


# ─── Single text-patch: inject swap hook into vllm's PWAL walker ──────


PN77_PWAL_OLD = (
    "    for _, module in model.named_modules():\n"
    "        quant_method = getattr(module, \"quant_method\", None)\n"
    "        if isinstance(quant_method, QuantizeMethodBase):\n"
    "            # When quant methods need to process weights after loading\n"
    "            # (for repacking, quantizing, etc), they expect parameters\n"
    "            # to be on the global target device. This scope is for the\n"
    "            # case where cpu offloading is used, where we will move the\n"
    "            # parameters onto device for processing and back off after.\n"
    "            with device_loading_context(module, target_device):\n"
    "                quant_method.process_weights_after_loading(module)\n"
)

PN77_PWAL_NEW = (
    "    for _, module in model.named_modules():\n"
    "        quant_method = getattr(module, \"quant_method\", None)\n"
    "        if isinstance(quant_method, QuantizeMethodBase):\n"
    "            # [Genesis PN77] swap UnquantizedEmbeddingMethod → FP8 method on lm_head\n"
    "            try:\n"
    "                from vllm.sndr_core.kernels.lm_head_fp8_method import (\n"
    "                    maybe_swap_pn77_quant_method as _pn77_swap,\n"
    "                )\n"
    "                quant_method = _pn77_swap(module, quant_method)\n"
    "            except Exception as _pn77_e:\n"
    "                import logging as _pn77_logging\n"
    "                _pn77_logging.getLogger(\"genesis.pn77\").warning(\n"
    "                    \"[PN77] swap helper crashed (%s) — keeping original method\",\n"
    "                    type(_pn77_e).__name__,\n"
    "                )\n"
    "            # When quant methods need to process weights after loading\n"
    "            # (for repacking, quantizing, etc), they expect parameters\n"
    "            # to be on the global target device. This scope is for the\n"
    "            # case where cpu offloading is used, where we will move the\n"
    "            # parameters onto device for processing and back off after.\n"
    "            with device_loading_context(module, target_device):\n"
    "                quant_method.process_weights_after_loading(module)\n"
)


# Drift markers — auto-retire when upstream lands FP8 lm_head support.
PN77_UPSTREAM_DRIFT_MARKERS = [
    # PR #41000 marker — config-driven Fp8Config dispatch on ParallelLMHead
    "lm_head_quantized",
    # PR #35696 marker — naive cast-only path
    "maybe_compress_lm_head_to_fp8",
]


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/model_loader/utils.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN77 model_loader/utils.py — PWAL hook for FP8 lm_head swap",
        target_file=str(target),
        marker=GENESIS_PN77_MARKER,
        sub_patches=[
            TextPatch(
                name="pn77_pwal_swap_hook",
                anchor=PN77_PWAL_OLD,
                replacement=PN77_PWAL_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=PN77_UPSTREAM_DRIFT_MARKERS,
    )


# ─── Genesis dispatcher entry points ──────────────────────────────────


def should_apply() -> bool:
    """Always install wiring (dormant unless env=1).

    The text-patch installs the swap hook unconditionally — runtime gate is
    the env check inside `maybe_swap_pn77_quant_method`. This means we can
    flip env=0/1 without re-deploying patches.
    """
    return True


def apply() -> tuple[str, str]:
    """Apply text-patch; never raises. Returns dispatcher (status, reason)."""
    p = _make_patcher()
    if p is None:
        return "skipped", "model_loader/utils.py not found in vllm install"
    result, failure = p.apply()
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown skip"
    if result == TextPatchResult.FAILED:
        return "failed", failure.reason if failure else "unknown failure"

    if not _is_enabled():
        return "applied", (
            "PN77 wiring installed (PWAL swap hook in process_weights_after_loading); "
            f"env {_ENV_FLAG}=0 → swap helper is no-op. "
            "Set env=1 to enable FP8 lm_head compression."
        )
    return "applied", (
        f"PN77 wiring installed AND {_ENV_FLAG}=1 → lm_head will be "
        f"compressed via Genesis_FP8_LMHead_EmbeddingMethod swap during "
        f"process_weights_after_loading. Hardware-tier dispatch active."
    )
