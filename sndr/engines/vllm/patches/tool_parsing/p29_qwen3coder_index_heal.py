# SPDX-License-Identifier: Apache-2.0
"""P29 IMPROVE — qwen3coder tool parser bounded-index guards (2 unguarded sites).

Upstream qwen3coder_tool_parser.py already added bounded-index guards at
lines 372 + 619 (visible in current container). Earlier Genesis dispatch
correctly detected those and printed "upstream already contains bounded-
index guards (no-op)". But two HOT sites remain raw:

(A) **Line 442** — json_started branch
    `self.streamed_args_for_tool[self.current_tool_index] += "{"`
    Raises IndexError if `current_tool_index` has advanced past list.

(B) **Line 287** — bare increment
    `self.current_tool_index += 1`
    No symmetric guarantee that `len(streamed_args_for_tool) > new_index`.
    This is the root cause; (A) is the symptom site.

Genesis P29 IMPROVE (2026-06-04) adds heal logic on BOTH sites so the
list expands lazily on advance, and the indexed write self-heals on
desync. Result: 500 errors on streaming tool-call deltas stop being
possible from this class of off-by-one.

Both guards are pure defensive — semantics preserved on the happy path
(list length already correct → guard is no-op).

Author: Sandermage (Sander) Barzov Aleksandr, 2026-06-04 P23/P29 fix-wire pass.
"""
from __future__ import annotations

import logging
import os

from sndr.kernel import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)
from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root

log = logging.getLogger("genesis.wiring.p29_qwen3coder_index_heal")

GENESIS_P29_HEAL_MARKER = "Genesis P29 qwen3coder index heal (2 raw sites) v1"


# ─────────────────── Site B (line 287): bare increment ───────────────────
_SITE287_OLD = (
    "            if tool_ends > self.current_tool_index:\n"
    "                # This tool has ended, advance to next\n"
    "                self.current_tool_index += 1\n"
    "                self.header_sent = False\n"
)

_SITE287_NEW = (
    "            if tool_ends > self.current_tool_index:\n"
    "                # This tool has ended, advance to next\n"
    "                self.current_tool_index += 1\n"
    "                # [Genesis P29] heal list length to match new index —\n"
    "                # prevents IndexError at downstream indexed writes.\n"
    "                while len(self.streamed_args_for_tool) <= self.current_tool_index:\n"
    "                    self.streamed_args_for_tool.append(\"\")\n"
    "                self.header_sent = False\n"
)

# ─────────────────── Site A (line 442): json_started write ───────────────────
_SITE442_OLD = (
    "            if not self.json_started:\n"
    "                self.json_started = True\n"
    "                self.streamed_args_for_tool[self.current_tool_index] += \"{\"\n"
)

_SITE442_NEW = (
    "            if not self.json_started:\n"
    "                self.json_started = True\n"
    "                # [Genesis P29] heal-on-write — if a prior advance\n"
    "                # path missed the append (race or out-of-order tool\n"
    "                # boundary detection), grow the list so the indexed\n"
    "                # write below doesn't raise IndexError.\n"
    "                while len(self.streamed_args_for_tool) <= self.current_tool_index:\n"
    "                    self.streamed_args_for_tool.append(\"\")\n"
    "                self.streamed_args_for_tool[self.current_tool_index] += \"{\"\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("tool_parsers/qwen3coder_tool_parser.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P29 qwen3coder index heal (sites 287 + 442)",
        target_file=str(target),
        marker=GENESIS_P29_HEAL_MARKER,
        sub_patches=[
            TextPatch(
                name="p29_heal_site287_advance",
                anchor=_SITE287_OLD,
                replacement=_SITE287_NEW,
                required=True,
            ),
            TextPatch(
                name="p29_heal_site442_json_started",
                anchor=_SITE442_OLD,
                replacement=_SITE442_NEW,
                required=True,
            ),
        ],
    )


_APPLIED = False


def apply() -> tuple[str, str]:
    """Apply P29 IMPROVE — qwen3coder index heal on 2 unguarded sites."""
    global _APPLIED

    if os.environ.get("GENESIS_ENABLE_P29_QWEN3CODER_INDEX_HEAL", "").lower() not in ("1", "true", "yes", "on"):
        return "skipped", (
            "P29 heal default OFF — set GENESIS_ENABLE_P29_QWEN3CODER_INDEX_HEAL=1 "
            "to engage. Closes 2 raw IndexError sites in qwen3coder tool parser "
            "(line 287 advance + line 442 json_started write) that upstream's "
            "bounded-index guard (lines 372 + 619) does not cover."
        )

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "tool_parsers/qwen3coder_tool_parser.py not found"

    result, failure = patcher.apply()
    if result == TextPatchResult.FAILED:
        return "failed", failure.reason if failure else "unknown TextPatch failure"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown TextPatch skip"
    applied = patcher.applied_sub_patches or [sp.name for sp in patcher.sub_patches]
    _APPLIED = True
    return "applied", (
        f"P29 heal installed: 2 raw IndexError sites in qwen3coder "
        f"tool parser now have heal-on-advance + heal-on-write guards. "
        f"Sub-patches: {', '.join(applied)}."
    )


def is_applied() -> bool:
    return _APPLIED
