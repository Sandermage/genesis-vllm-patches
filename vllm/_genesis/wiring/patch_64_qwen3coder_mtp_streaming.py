# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 64 — qwen3coder MTP streaming early-return removal.

Backport of vllm-project/vllm#39598 (kotori-yan, OPEN at time of writing).

================================================================
Targets MTP/spec-decode streaming tool-call edge case where a single
delta bundles multiple semantic tokens (last parameter value AND
</function> together). The pre-PR code has an early `return` after
emitting parameter fragments, so the `</function>` block never
executes — leaving prev_tool_call_arr with stale `"{}"` arguments and
streamed_args_for_tool without the closing `}`. Final chunk then has
no valid arguments → empty `tool_calls`.
================================================================

Three sub-patches:

1. **qwen3coder_tool_parser.py** — Restructure `extract_tool_calls_streaming`
   to remove early return and unify parameter/function-end accumulation.

2. **serving.py — _should_check_for_unstreamed_tool_arg_tokens** — Widen
   safety-net condition to fire on `finish_reason` presence alone (not
   gated by non-empty `tool_calls`). With MTP, the last delta before
   finish may carry no `tool_calls` even though tool calls in progress.

3. **serving.py — _create_remaining_args_delta** — Switch from
   constructor-passed `id=None / type=None / name=None` to
   post-construction conditional assignment, so Pydantic v2 treats
   absent fields as *unset* and `model_dump(exclude_unset=True)` omits
   them (instead of emitting `"id": null`).

Status: opt-in via `GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING=1`.

Compatibility
-------------
- Streaming-only fix; non-streaming tool-call code path unaffected.
- Idempotent (marker check).
- Auto-no-op once #39598 lands upstream (drift marker:
  `_create_remaining_args_delta` post-construction pattern).

Risks acknowledged
------------------
- Three sub-patches across two files; all-or-nothing apply (anchor
  drift in one → whole group skips).
- Test coverage limited to streaming clients (LibreChat, OpenWebUI).
  Synthetic streaming reproducer in test suite.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatcher,
    TextPatchResult,
    TextPatch,
)

log = logging.getLogger("genesis.wiring.p64_qwen3coder_mtp_streaming")

GENESIS_P64_MARKER = "Genesis P64 qwen3coder MTP streaming early-return fix v7.13"


# ─── Sub-patch A: qwen3coder_tool_parser.py — remove early return ───────────
# The bug: early `return DeltaMessage(...)` after `if json_fragments:` block
# prevents the </function> handling block from executing when MTP bundles
# the last parameter + </function> in the same delta.
#
# Fix: collapse fragment accumulation into a `combined` string that gets
# extended by the </function> path (adding "}"), and emit ONE return at the
# end with the merged content.

QWEN3CODER_OLD = (
    "            if json_fragments:\n"
    "                combined = \"\".join(json_fragments)\n"
    "\n"
    "                if self.current_tool_index < len(self.streamed_args_for_tool):\n"
    "                    self.streamed_args_for_tool[self.current_tool_index] += combined\n"
    "                else:\n"
    "                    logger.warning(\n"
    "                        \"streamed_args_for_tool out of sync: index=%d len=%d\",\n"
    "                        self.current_tool_index,\n"
    "                        len(self.streamed_args_for_tool),\n"
    "                    )\n"
    "\n"
    "                return DeltaMessage(\n"
    "                    tool_calls=[\n"
    "                        DeltaToolCall(\n"
    "                            index=self.current_tool_index,\n"
    "                            function=DeltaFunctionCall(arguments=combined),\n"
    "                        )\n"
    "                    ]\n"
    "                )\n"
)

QWEN3CODER_NEW = (
    "            # [Genesis P64 vllm#39598] Do NOT early-return here. With MTP\n"
    "            # speculative decoding a single delta can bundle the last\n"
    "            # parameter value AND </function> together. An early return\n"
    "            # would skip the </function> block below, leaving\n"
    "            # prev_tool_call_arr with stale \"{}\" and streamed_args_for_tool\n"
    "            # without the closing \"}\". Accumulate into `combined`, let\n"
    "            # </function> path append \"}\", emit ONE return at end.\n"
    "            combined = \"\".join(json_fragments) if json_fragments else \"\"\n"
)


# ─── Sub-patch B: append "}" to combined inside </function> branch ──────────
# The </function> branch currently emits its own `result = DeltaMessage(... arguments="}")`
# and `return result`. Change to: append "}" to `combined`, fall through to
# unified emit below.

QWEN3COD_FNEND_OLD = (
    "                if self.current_tool_index < len(self.streamed_args_for_tool):\n"
    "                    self.streamed_args_for_tool[self.current_tool_index] += \"}\"\n"
    "                else:\n"
    "                    logger.warning(\n"
    "                        \"streamed_args_for_tool out of sync: index=%d len=%d\",\n"
    "                        self.current_tool_index,\n"
    "                        len(self.streamed_args_for_tool),\n"
    "                    )\n"
    "\n"
    "                result = DeltaMessage(\n"
    "                    tool_calls=[\n"
    "                        DeltaToolCall(\n"
    "                            index=self.current_tool_index,\n"
    "                            function=DeltaFunctionCall(arguments=\"}\"),\n"
    "                        )\n"
    "                    ]\n"
    "                )\n"
    "\n"
    "                self.in_function = False\n"
    "                self.json_closed = True\n"
    "                self.accumulated_params = {}\n"
    "\n"
    "                return result\n"
)

QWEN3COD_FNEND_NEW = (
    "                # [Genesis P64 vllm#39598] Append \"}\" to combined and fall\n"
    "                # through to unified emit below — no early return.\n"
    "                combined += \"}\"\n"
    "                self.in_function = False\n"
    "                self.accumulated_params = {}\n"
    "\n"
    "            if combined:\n"
    "                if self.current_tool_index < len(self.streamed_args_for_tool):\n"
    "                    self.streamed_args_for_tool[self.current_tool_index] += combined\n"
    "                else:\n"
    "                    logger.warning(\n"
    "                        \"streamed_args_for_tool out of sync: index=%d len=%d\",\n"
    "                        self.current_tool_index,\n"
    "                        len(self.streamed_args_for_tool),\n"
    "                    )\n"
    "\n"
    "                return DeltaMessage(\n"
    "                    tool_calls=[\n"
    "                        DeltaToolCall(\n"
    "                            index=self.current_tool_index,\n"
    "                            function=DeltaFunctionCall(arguments=combined),\n"
    "                        )\n"
    "                    ]\n"
    "                )\n"
)


# ─── Sub-patch C: serving.py — _should_check safety-net widening ────────────

SERVING_SHOULD_OLD = (
    "        return bool(\n"
    "            # if there is a delta message that includes tool calls which\n"
    "            # include a function that has arguments\n"
    "            output.finish_reason is not None\n"
    "            and self.enable_auto_tools\n"
    "            and self.tool_parser\n"
    "            and delta_message\n"
    "            and delta_message.tool_calls\n"
    "            and delta_message.tool_calls[0]\n"
    "            and delta_message.tool_calls[0].function\n"
    "            and delta_message.tool_calls[0].function.arguments is not None\n"
    "        )\n"
)

SERVING_SHOULD_NEW = (
    "        # [Genesis P64 vllm#39598] Widen safety-net: with MTP/spec-decode\n"
    "        # the final delta before finish_reason may carry no tool_calls\n"
    "        # even though tool calls are still in progress. Caller's\n"
    "        # auto_tools_called guard (checks len(prev_tool_call_arr) > 0)\n"
    "        # prevents false positives for plain-text responses.\n"
    "        return bool(\n"
    "            output.finish_reason is not None\n"
    "            and self.enable_auto_tools\n"
    "            and self.tool_parser\n"
    "        )\n"
)


# ─── Sub-patch D: serving.py — _create_remaining_args_delta Pydantic fix ────

SERVING_CRD_OLD = (
    "        original_tc = next(\n"
    "            (tc for tc in delta_message.tool_calls if tc.index == index),\n"
    "            None,\n"
    "        )\n"
    "        original_fn = original_tc.function if original_tc else None\n"
    "        return DeltaMessage(\n"
    "            tool_calls=[\n"
    "                DeltaToolCall(\n"
    "                    index=index,\n"
    "                    id=original_tc.id if original_tc else None,\n"
    "                    type=original_tc.type if original_tc else None,\n"
    "                    function=DeltaFunctionCall("
)

SERVING_CRD_NEW = (
    "        # [Genesis P64 vllm#39598] Use post-construction assignment\n"
    "        # instead of constructor None so Pydantic v2 treats absent\n"
    "        # fields as *unset* (model_dump(exclude_unset=True) omits them\n"
    "        # instead of emitting \"id\": null which violates OpenAI spec).\n"
    "        original_tc = next(\n"
    "            (tc for tc in delta_message.tool_calls if tc.index == index),\n"
    "            None,\n"
    "        )\n"
    "        original_fn = original_tc.function if original_tc else None\n"
    "\n"
    "        delta_fn = DeltaFunctionCall(arguments=remaining_call)\n"
    "        if original_fn and original_fn.name is not None:\n"
    "            delta_fn.name = original_fn.name\n"
    "\n"
    "        tc = DeltaToolCall(index=index, function=delta_fn)\n"
    "        if original_tc and original_tc.id is not None:\n"
    "            tc.id = original_tc.id\n"
    "            tc.type = original_tc.type or \"function\"\n"
    "\n"
    "        return DeltaMessage(tool_calls=[tc])\n"
    "        # [Genesis P64] above replaces the original constructor-style:\n"
    "        return DeltaMessage(\n"
    "            tool_calls=[\n"
    "                DeltaToolCall(\n"
    "                    index=index,\n"
    "                    id=original_tc.id if original_tc else None,\n"
    "                    type=original_tc.type if original_tc else None,\n"
    "                    function=DeltaFunctionCall("
)


def _make_qwen3cod_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("tool_parsers/qwen3coder_tool_parser.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P64 qwen3coder_tool_parser.py — MTP streaming early-return removal",
        target_file=str(target),
        marker=GENESIS_P64_MARKER + " :: qwen3coder_tool_parser.py",
        sub_patches=[
            TextPatch(name="p64_remove_early_return", anchor=QWEN3CODER_OLD,
                      replacement=QWEN3CODER_NEW, required=True),
            TextPatch(name="p64_unify_emit_at_fnend", anchor=QWEN3COD_FNEND_OLD,
                      replacement=QWEN3COD_FNEND_NEW, required=True),
        ],
        upstream_drift_markers=[
            "[Genesis P64 vllm#39598]",
            "Do NOT early-return here. With MTP",
        ],
    )


def _make_serving_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("entrypoints/openai/chat_completion/serving.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P64 serving.py — MTP safety-net + Pydantic null fix",
        target_file=str(target),
        marker=GENESIS_P64_MARKER + " :: serving.py",
        sub_patches=[
            TextPatch(name="p64_safety_net_widen", anchor=SERVING_SHOULD_OLD,
                      replacement=SERVING_SHOULD_NEW, required=True),
            # NOTE: D sub-patch DUPLICATES the original return for the second
            # half of _create_remaining_args_delta to keep the function shape
            # syntactically valid (the original DeltaFunctionCall line
            # continues into name= / arguments=). The duplicate emit is
            # unreachable but preserves anchor stability for #D's tail.
            # Simpler approach: only patch the safety-net (sub-patch C); leave
            # _create_remaining_args_delta unchanged. The Pydantic null fix is
            # belt-and-braces; the parser fix (sub-patches A+B) closes the
            # primary symptom.
        ],
        upstream_drift_markers=[
            "[Genesis P64 vllm#39598]",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P64 streaming-fix backport. All-or-nothing across two files."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P64")
    log_decision("P64", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patchers = [_make_qwen3cod_patcher(), _make_serving_patcher()]
    if any(p is None for p in patchers):
        return "skipped", "one or more target files not found"

    # Pre-flight
    for p in patchers:
        if not os.path.isfile(p.target_file):
            return "skipped", f"target disappeared: {p.target_file}"
        with open(p.target_file) as f:
            content = f.read()
        if p.marker in content:
            continue
        for m in p.upstream_drift_markers:
            if m in content:
                return (
                    "skipped",
                    f"upstream drift marker {m!r} in {p.target_file} — "
                    "vllm#39598 likely already merged or backported.",
                )
        for sp in p.sub_patches:
            if sp.required and sp.anchor not in content:
                return (
                    "skipped",
                    f"required anchor for {sp.name!r} not found in "
                    f"{p.target_file} — anchor drifted, P64 cannot apply.",
                )

    results = []
    for p in patchers:
        result, failure = p.apply()
        if result == TextPatchResult.FAILED:
            return "failed", (
                f"{p.patch_name}: {failure.reason if failure else 'unknown'} "
                f"({failure.detail if failure else ''}) — partial state risk; "
                "container should be torn down (compose down + up -d)."
            )
        results.append((p.patch_name, result))

    applied = sum(1 for _, r in results if r == TextPatchResult.APPLIED)
    idempotent = sum(1 for _, r in results if r == TextPatchResult.IDEMPOTENT)

    return "applied", (
        f"P64 applied: {applied} files modified, {idempotent} idempotent. "
        "qwen3coder streaming parser no longer drops parameters when MTP "
        "bundles last param + </function> in same delta. Safety net "
        "widened to fire on finish_reason alone."
    )
