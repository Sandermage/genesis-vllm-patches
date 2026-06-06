# SPDX-License-Identifier: Apache-2.0
"""SNDR Core bundles — shared `run_bundle()` orchestrator helper.

Every bundle's apply() does the same 5 steps:
  1. Umbrella env flag check (skip if not enabled).
  2. Tier-gate (skip if tier=engine AND sndr_engine not installed,
     OR SNDR_ENABLE_TIER_OVERRIDE forces community-only).
  3. Compose patchers by calling each `_make_*_patcher` lazily (at
     bundle-apply time, not module-import time).
  4. Drop None patchers (target file not found / vllm unresolved)
     while keeping the bundle viable for the rest.
  5. Atomic MultiFilePatchTransaction commit.

This module factors out steps 1+2+5 so each bundle is just a list of
patcher-callables + bundle metadata.

Migration: Stage 7 (2026-05-07).
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from vllm.sndr_core.core.multi_file import MultiFilePatchTransaction
from vllm.sndr_core.core.text_patch import TextPatcher
from vllm.sndr_core.env import Flags, is_enabled

log = logging.getLogger("genesis.bundles")

PatcherFactory = Callable[[], Optional[TextPatcher]]


def run_bundle(
    *,
    name: str,
    umbrella_flag: str,
    tier: str,
    patcher_factories: list[PatcherFactory],
) -> tuple[str, str]:
    """Apply a bundle atomically. Common orchestrator for all bundles.

    Args:
      name: human-readable bundle name (e.g. "tool_parsing_qwen3coder").
      umbrella_flag: `Flags.BUNDLE_*` value gating this bundle.
      tier: "community" | "engine" — engine tier requires sndr_engine
            package + honors SNDR_ENABLE_TIER_OVERRIDE escape hatch.
      patcher_factories: list of callables (typically `_make_patcher`
            functions from individual patch modules). Called at apply
            time. Returning None from a factory means the patch's
            target file is unresolvable on this pin — that factory is
            DROPPED from the transaction (sibling patches still apply).
            If ALL factories return None, the bundle skips with a
            clear reason.

    Returns:
      ("applied", "<n>/<m> sub-patchers committed atomically")
      ("skipped", reason) — bundle disabled / tier-gated / no targets
      ("failed",  reason) — commit-phase failure with rollback details
    """
    # 1. Umbrella enable check
    if not is_enabled(umbrella_flag):
        return "skipped", f"bundle {name} disabled (set SNDR_ENABLE_{umbrella_flag}=1)"

    # 2. Tier-gate
    if tier == "engine":
        if is_enabled(Flags.TIER_OVERRIDE):
            return "skipped", (
                f"bundle {name}: tier=engine but TIER_OVERRIDE forces "
                "community-only mode"
            )
        try:
            import vllm.sndr_engine  # noqa: F401
        except ImportError:
            return "skipped", (
                f"bundle {name}: tier=engine but vllm.sndr_engine not "
                "installed — requires commercial SNDR Engine license"
            )

    # 3+4. Compose patchers, drop None (target file unresolved on this pin)
    patchers: list[TextPatcher] = []
    skipped_factories: list[str] = []
    for factory in patcher_factories:
        try:
            p = factory()
        except Exception as e:
            return "failed", (
                f"bundle {name}: patcher factory "
                f"{factory.__qualname__!r} raised {type(e).__name__}: {e}"
            )
        if p is None:
            skipped_factories.append(factory.__qualname__)
            continue
        patchers.append(p)

    if not patchers:
        return "skipped", (
            f"bundle {name}: no resolvable target files — every "
            "patcher factory returned None (unsupported vllm install?)"
        )

    if skipped_factories:
        log.info(
            "[bundle %s] %d/%d factories skipped (target unresolved): %s",
            name, len(skipped_factories),
            len(patcher_factories), ", ".join(skipped_factories),
        )

    # 5. Atomic commit via MultiFilePatchTransaction (Stage 3 infrastructure)
    txn = MultiFilePatchTransaction(patchers, name=name)
    return txn.apply_or_skip()


__all__ = ["run_bundle", "PatcherFactory"]
