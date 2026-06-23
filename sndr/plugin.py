# SPDX-License-Identifier: Apache-2.0
"""SNDR Core — `vllm.general_plugins` entry point.

P0-6 fix (audit 2026-05-08): the root `pyproject.toml` previously
declared `genesis_v7 = "genesis_v7:register"`, pointing at a
standalone `genesis_v7` package that was NOT included in the core
wheel. After `pip install vllm-sndr-core`, the entry point pointed
at a nonexistent module and vllm would log a "plugin failed to load"
warning.

This module is the canonical entry point. The root pyproject now
declares `genesis_v7 = "sndr.plugin:register"`.

A separate `genesis_v7` shim package may still exist on hosts where
operators installed the plugin via pip during the v7.x era; it
re-exports `apply_genesis` from this module for back-compat.

Behavioral guarantees (vllm plugin contract):
  - Importable in a vLLM-less environment (no top-level vllm imports).
  - Idempotent — vLLM may re-trigger registration.
  - Non-fatal on error — log + return rather than raise.
  - Fast (< 1 sec on green path; text-patches add a few ms).

Author: Sandermage(Sander)-Barzov Aleksandr.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.plugin")


def register() -> None:
    """vLLM general-plugin entry point.

    Called by `vllm.plugins.load_general_plugins()` once per process
    (engine + every worker rank).
    """
    # G4_19/G4_19b ALWAYS-APPLY override: when G4_19 is explicitly enabled
    # via env, apply it + G4_19b even if GENESIS_DISABLE is set. This lets
    # operators run ONLY the G4-TurboQuant KV cache path (256K context unlock)
    # without enabling the entire Genesis patch stack which can corrupt
    # Gemma 4 config max_model_len in current dev371 pin.
    #
    # Critical for EngineCore subprocess: vLLM v1 spawns EngineCore which
    # re-loads plugins. Our G4_19b monkey-patches _check_enough_kv_cache_memory
    # and must run BEFORE the check fires in EngineCore. Apply happens here
    # at plugin register time, before vLLM engine init.
    _G4_19_ENABLED = os.environ.get(
        "GENESIS_ENABLE_G4_19_GEMMA4_TURBOQUANT_KV", ""
    ).strip().lower() in ("1", "true", "yes")
    _G4_19B_ENABLED = os.environ.get(
        "GENESIS_ENABLE_G4_19B_GEMMA4_TQ_KV_SPEC", ""
    ).strip().lower() in ("1", "true", "yes")

    def _apply_g4_19_pair() -> None:
        """Apply G4_19 + G4_19b selectively (no full Genesis apply)."""
        try:
            # Pre-import Gemma 4 module so Gemma4Config exists for G4_19 wrapper
            import vllm.model_executor.models.gemma4  # noqa: F401
        except ImportError:
            log.warning(
                "[Genesis plugin] gemma4 module not importable; G4_19 will skip"
            )
        try:
            if _G4_19_ENABLED:
                from sndr.engines.vllm.patches.attention.turboquant import (
                    g4_19_turboquant_kv_cache as _g4_19,
                )
                s, m = _g4_19.apply()
                log.info("[Genesis plugin] G4_19: %s — %s", s, m[:200])
            if _G4_19B_ENABLED:
                from sndr.engines.vllm.patches.attention.turboquant import (
                    g4_19b_tq_kv_spec_integration as _g4_19b,
                )
                s, m = _g4_19b.apply()
                log.info("[Genesis plugin] G4_19b: %s — %s", s, m[:200])
        except Exception as e:  # noqa: BLE001
            log.warning("[Genesis plugin] selective G4 apply failed: %r", e)

    if os.environ.get("GENESIS_DISABLE", "").strip().lower() in ("1", "true", "yes"):
        if _G4_19_ENABLED or _G4_19B_ENABLED:
            log.info(
                "[Genesis plugin] GENESIS_DISABLE set BUT G4_19/19b explicitly "
                "enabled — running selective apply for those only"
            )
            _apply_g4_19_pair()
        else:
            log.info("[Genesis plugin] GENESIS_DISABLE set — skipping registration")
        return

    # When full Genesis stack is enabled, G4_19/19b are part of apply.run() too,
    # so no need to call _apply_g4_19_pair() — apply.run() handles them via
    # registry default_on / env-enable checks.

    # ── Operator-opt-in safety valve: route AutoRound INT8 W8A16 ─────────
    # group_size=-1 checkpoints through Marlin instead of AllSpark by
    # prepending AllSparkLinearKernel to VLLM_DISABLED_KERNELS.
    #
    # Background: vllm's mixed-precision selector picks AllSpark first
    # for those checkpoints, which means P87 (Marlin sub-tile pad-on-load)
    # and P91 (gptq_marlin row-parallel group cdiv) never fire. This
    # hook is the only way to engage them on those checkpoints.
    #
    # Must run BEFORE vllm engine init reads VLLM_DISABLED_KERNELS, so
    # it lives in plugin register() (called early by load_general_plugins)
    # rather than in apply_all (which runs later).
    #
    # See memory `project_genesis_allspark_research_20260428` for the
    # full A/B picture; default OFF until benched on consumer Ampere.
    if os.environ.get(
        "GENESIS_FORCE_MARLIN_W8A16", ""
    ).strip().lower() in ("1", "true", "yes"):
        existing = os.environ.get("VLLM_DISABLED_KERNELS", "")
        if "AllSparkLinearKernel" not in existing:
            os.environ["VLLM_DISABLED_KERNELS"] = (
                existing + ("," if existing else "") + "AllSparkLinearKernel"
            )
            log.info(
                "[Genesis] GENESIS_FORCE_MARLIN_W8A16=1 — added "
                "AllSparkLinearKernel to VLLM_DISABLED_KERNELS=%r so the "
                "vllm selector falls through to Marlin (P87 + P91 will now "
                "fire on AutoRound INT8 group_size=-1 checkpoints).",
                os.environ["VLLM_DISABLED_KERNELS"],
            )

    # ── apply_all = canonical auto-wire of all PATCH_REGISTRY entries ────
    try:
        from sndr.apply import run
    except ImportError as e:
        log.warning(
            "[Genesis plugin] sndr.apply not importable — "
            "skipping (cause: %s). Check that the sndr package is installed "
            "in the same environment vllm uses for plugin discovery.", e,
        )
        return

    try:
        apply_mode = (
            os.environ.get("GENESIS_WIRING_APPLY", "1").strip().lower()
            not in ("0", "false", "no")
        )
        stats = run(verbose=True, apply=apply_mode)
        log.info(
            "[Genesis plugin] register() complete: applied=%d skipped=%d "
            "failed=%d (apply=%s)",
            stats.applied_count, stats.skipped_count, stats.failed_count,
            apply_mode,
        )
    except Exception as e:
        # Never block vLLM startup on plugin error — log and return.
        log.exception("[Genesis plugin] register() failed: %s", e)


# Manual-invocation alias kept for parity with the legacy
# v7.x `genesis_v7` package surface (back-compat for operators who
# installed it as a separate pip package).
apply_genesis = register


__all__ = ["register", "apply_genesis"]
