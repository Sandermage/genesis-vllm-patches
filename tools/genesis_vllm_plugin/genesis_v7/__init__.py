# SPDX-License-Identifier: Apache-2.0
"""Genesis v7.0 vLLM plugin entry point.

Registered via `pyproject.toml` under `vllm.general_plugins` so vLLM's
`load_general_plugins()` calls `register()` automatically at process
start in every rank / engine process.

DO NOT add vllm imports at module top-level here — this file must be
importable even in a vLLM-less environment (for static analysis, test
collection, etc.). All vllm-touching work happens inside `register()`.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.plugin")


def register() -> None:
    """vLLM plugin entry point.

    Called by `vllm.plugins.load_general_plugins()` once per process. Must be:
      - Idempotent (safe to call multiple times — vLLM may re-trigger)
      - Non-fatal on error (log, don't raise — do not break the engine)
      - Fast (< 1 sec on green path; text-patches can add a few ms)
    """
    # Allow opt-out via env for troubleshooting / rollback.
    if os.environ.get("GENESIS_DISABLE", "").strip().lower() in ("1", "true", "yes"):
        log.info("[Genesis plugin] GENESIS_DISABLE set — skipping registration")
        return

    # ────────────────────────────────────────────────────────────────────
    # [Genesis P93] AllSpark bypass for INT8 W8A16 + group_size=-1 paths
    # ────────────────────────────────────────────────────────────────────
    # When GENESIS_FORCE_MARLIN_W8A16=1, prepend "AllSparkLinearKernel" to
    # VLLM_DISABLED_KERNELS so vLLM's mixed-precision selector falls
    # through to MarlinLinearKernel for AutoRound INT8 W8A16 group_size=-1
    # checkpoints (e.g. Minachist/Qwen3.6-27B-INT8-AutoRound). This is
    # the only way to engage P87 (Marlin sub-tile pad-on-load) and P91
    # (gptq_marlin row-parallel group cdiv) on those checkpoints —
    # without this flag the selector picks AllSpark first and the Marlin
    # patches never fire.
    #
    # Marlin SUPPORTS group_size=-1 + uint8b128 (per
    # vllm/model_executor/layers/quantization/utils/marlin_utils.py:30,75)
    # — we are not relying on a workaround, just changing the
    # kernel-selection priority via the existing VLLM_DISABLED_KERNELS hook.
    #
    # AllSpark vs Marlin perf on consumer Ampere SM 8.6 (A5000) at
    # M=1..8 spec-decode is unbenched publicly; needs A/B before
    # promoting to default-on. See memory record
    # `project_genesis_allspark_research_20260428` for full analysis.
    #
    # Must run BEFORE vLLM engine init reads VLLM_DISABLED_KERNELS, so it
    # lives in plugin register() (called early by load_general_plugins)
    # rather than in apply_all (which runs later in init order).
    if os.environ.get(
        "GENESIS_FORCE_MARLIN_W8A16", ""
    ).strip().lower() in ("1", "true", "yes"):
        existing = os.environ.get("VLLM_DISABLED_KERNELS", "")
        if "AllSparkLinearKernel" not in existing:
            os.environ["VLLM_DISABLED_KERNELS"] = (
                existing + ("," if existing else "") + "AllSparkLinearKernel"
            )
            log.info(
                "[Genesis P93] GENESIS_FORCE_MARLIN_W8A16=1 — added "
                "AllSparkLinearKernel to VLLM_DISABLED_KERNELS=%r so vLLM "
                "selector falls through to Marlin (P87 + P91 will now fire "
                "on AutoRound INT8 group_size=-1 checkpoints).",
                os.environ["VLLM_DISABLED_KERNELS"],
            )

    try:
        from vllm._genesis.patches.apply_all import run
    except ImportError as e:
        log.warning(
            "[Genesis plugin] vllm._genesis package not importable — skipping. "
            "Cause: %s. Check mount of vllm/_genesis into vLLM site-packages.",
            e,
        )
        return

    try:
        # apply=True enables the actual wiring (text-patches + monkey-patches).
        # apply=False is diagnostic-only (orchestrator reports what WOULD happen).
        apply_mode = (
            os.environ.get("GENESIS_WIRING_APPLY", "1").strip().lower()
            not in ("0", "false", "no")
        )
        stats = run(verbose=True, apply=apply_mode)
        log.info(
            "[Genesis plugin] register() complete: %d applied / %d skipped / %d failed (apply=%s)",
            stats.applied_count, stats.skipped_count, stats.failed_count, apply_mode,
        )
    except Exception as e:
        # Never block vLLM startup on plugin error.
        log.exception("[Genesis plugin] register() failed: %s", e)


# Optional: explicit alias for easier manual invocation + debugging
apply_genesis = register
