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

log = logging.getLogger("genesis.plugin")


def register() -> None:
    """vLLM plugin entry point — back-compat thin shim.

    Called by `vllm.plugins.load_general_plugins()` once per process. Must be:
      - Idempotent (safe to call multiple times — vLLM may re-trigger)
      - Non-fatal on error (log, don't raise — do not break the engine)
      - Fast (< 1 sec on green path; text-patches can add a few ms)

    UNIFIED ROOT BUG fix (2026-06-22): this shim now DELEGATES to the
    canonical `sndr.plugin.register` instead of re-implementing the apply
    logic. There is exactly ONE in-process apply path
    (`sndr.plugin.register` → `sndr.apply.run`), so there is no risk of a
    second, divergent apply running. The legacy subdir's pyproject now
    also registers `sndr.plugin:register` directly; this module is kept
    only so `import genesis_v7` keeps working for v7.x operators.
    """
    try:
        from sndr.plugin import register as _canonical_register
    except ImportError as e:
        log.warning(
            "[Genesis plugin] sndr.plugin not importable — skipping. "
            "Cause: %s. Check that the sndr package is installed/importable "
            "in the same environment vllm uses for plugin discovery.",
            e,
        )
        return
    _canonical_register()


# Optional: explicit alias for easier manual invocation + debugging.
# Delegates to the canonical sndr.plugin surface via register() above.
apply_genesis = register
