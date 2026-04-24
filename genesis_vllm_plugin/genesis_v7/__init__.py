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
