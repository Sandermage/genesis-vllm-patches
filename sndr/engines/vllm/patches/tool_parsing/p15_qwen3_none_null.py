# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 15 — Qwen3 chat-template `None` vs `null` tool-call fix.

Problem
-------
Qwen3.5+ chat templates use Jinja's `| string` filter for scalar tool-call
arguments, which produces Python's `repr()` form `None` instead of the JSON
literal `null`. The qwen3coder tool parser only recognises lowercase `null`,
so a `None` slips through as the literal string `"None"` and breaks any
tool with a nullable parameter.

Reference: vLLM PR [#38996](https://github.com/vllm-project/vllm/pull/38996)
            issue [#38885](https://github.com/vllm-project/vllm/issues/38885).

Fix
---
Accept both `null` and `none` (case-insensitive) in `_convert_param_value`:

    # before:
    if param_value.lower() == "null":
        return None
    # after:
    if param_value.lower() in ("null", "none"):
        return None

Platform compatibility: vendor-agnostic — pure Python parser logic.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""

# Legacy auto-apply note (audit 2026-05-11): registry env_flag
# `GENESIS_LEGACY_P15` is synthetic — flag exists for registry/audit
# coherence but has no runtime effect. Patch applies unconditionally
# via dispatcher's legacy auto-apply path (`is_legacy_active` in
# vllm/sndr_core/dispatcher/decision.py). See registry.py "Legacy
# patches" section (~line 2083) for full context.

from __future__ import annotations

import logging

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import (
    TextPatch, TextPatcher, TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p15_qwen3_none_null")

GENESIS_P15_MARKER = "Genesis P15 Qwen3 None/null tool arg v7.0"

UPSTREAM_DRIFT_MARKERS = [
    # Self-collision lint (triage plan §6 2026-06-11): former entry
    # '("null", "none")' is baked verbatim by our own replacement — it
    # cannot distinguish a real PR #38996 merge from our residue (false
    # "upstream_merged" skip, PN369 class). The single-quoted variant
    # below is a strictly-upstream-only spelling, never emitted by us.
    "'null', 'none'",
]


# Pre-dev354 form (qwen3coder_tool_parser.py:_convert_param_value inline).
_OLD_LEGACY = (
    "        # Handle null value for any type\n"
    '        if param_value.lower() == "null":\n'
    "            return None"
)

_NEW_LEGACY = (
    "        # [Genesis P15] Handle null/none value for any type (PR #38996).\n"
    "        # Qwen3.5+ chat template emits Python repr 'None' (Jinja `| string`)\n"
    "        # instead of JSON 'null'. Accept both case-insensitively.\n"
    '        if param_value.lower() in ("null", "none"):\n'
    "            return None"
)

# dev354+ form: upstream moved null-check from qwen3coder_tool_parser.py
# into tool_parsers/utils.py:coerce_to_schema_type (4-space indent inside
# coerce_to_schema_type's `for candidate_type in type_priority:` loop).
_OLD_DEV354 = (
    '        if candidate_type == "null":\n'
    '            if value.lower() == "null":\n'
    "                return None\n"
    "            continue"
)

_NEW_DEV354 = (
    '        if candidate_type == "null":\n'
    "            # [Genesis P15] Accept Python repr 'None' alongside JSON 'null'.\n"
    '            if value.lower() in ("null", "none"):\n'
    "                return None\n"
    "            continue"
)


def _make_patcher() -> TextPatcher | None:
    # Prefer dev354+ location (tool_parsers/utils.py).
    target_new = resolve_vllm_file("tool_parsers/utils.py")
    target_legacy = resolve_vllm_file("tool_parsers/qwen3coder_tool_parser.py")
    if target_new is not None:
        return TextPatcher(
            patch_name="P15 Qwen3 None/null tool arg (dev354 utils.py form)",
            target_file=target_new,
            marker=GENESIS_P15_MARKER,
            sub_patches=[
                TextPatch(
                    name="p15_none_null_utils",
                    anchor=_OLD_DEV354,
                    replacement=_NEW_DEV354,
                    required=True,
                ),
            ],
            upstream_drift_markers=UPSTREAM_DRIFT_MARKERS,
        )
    if target_legacy is not None:
        return TextPatcher(
            patch_name="P15 Qwen3 None/null tool arg (legacy qwen3coder form)",
            target_file=target_legacy,
            marker=GENESIS_P15_MARKER,
            sub_patches=[
                TextPatch(
                    name="p15_none_null",
                    anchor=_OLD_LEGACY,
                    replacement=_NEW_LEGACY,
                    required=True,
                ),
            ],
            upstream_drift_markers=UPSTREAM_DRIFT_MARKERS,
        )
    return None


def apply() -> tuple[str, str]:
    """Apply P15 wiring. Never raises."""
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "tool_parsers/utils.py and qwen3coder_tool_parser.py both not found"

    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return "applied", "None/none mapping added to tool param parser"
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied this image layer (idempotent)"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown skip"
    return "failed", failure.reason if failure else "unknown failure"
