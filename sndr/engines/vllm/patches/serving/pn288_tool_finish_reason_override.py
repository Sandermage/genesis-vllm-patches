# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN288 — qwen3_coder tool_call finish_reason override
(§1.3 of the unified plan, Phase B / dry-run scaffold).

The decision logic lives in the companion middleware module
``sndr.engines.vllm.middleware.pn288_finish_reason_override``. This file
is the text-patch overlay that wires that logic into
``OpenAIServingChat._create_chat_completion`` at two anchors:

  1. **Streaming** (serving.py:821-828 on pin 0.22.1rc1.dev259) — the
     if-block that assigns ``finish_reason_`` inside the choice-data loop.
  2. **Non-streaming** (serving.py:1246-1250) — the
     ``is_finish_reason_tool_calls`` bool assignment.

Anchor strategy
---------------
v1 verified live against the running
``vllm-gemma4-31b-tq-mtp-structured-k4-k4`` container 2026-05-30 (pin
626fa9bb). v2 re-anchored 2026-06-11 for pin
0.22.1rc1.dev259+g303916e93: upstream removed ``auto_tools_called``
from the streaming condition (the variable now exists only in the
non-streaming full generator), so the streaming anchor, the injected
call and its except-fallback dropped every reference to it. Sub-patch 2
is byte-identical to v1 (pristine serving.py:1246-1250 unchanged).
P107 targets the same streaming block — see the apply-order chain note
above ``PN288_STREAMING_OLD``.

Both replacements are wrapped in ``try / except Exception`` so any
import or runtime failure in the middleware logic falls back to the
upstream verdict — observability must not break the request.

Gates
-----
  * ``GENESIS_ENABLE_PN288_TOOL_FINISH_REASON_OVERRIDE`` — install
    gate. Default OFF; no text patch is written unless this is ``1``.
  * ``GENESIS_PN288_DRY_RUN`` — Phase B vs Phase C selector, read
    inside the middleware module at decision time (NOT at patch time)
    so an operator can flip it on a live container without re-patching.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import (
    resolve_vllm_file, vllm_install_root,
)
from sndr.kernel import (
    TextPatch, TextPatcher, TextPatchResult,
)


log = logging.getLogger("genesis.wiring.pn288_tool_finish_reason_override")

GENESIS_PN288_MARKER = (
    "Genesis PN288 tool-call finish_reason override v1 (Phase B dry-run)"
)


# ─── Sub-patch 1: streaming anchor ──────────────────────────────────────
#
# v2 (2026-06-11, pin 0.22.1rc1.dev259+g303916e93): upstream REMOVED the
# ``auto_tools_called`` OR-clause from the streaming finish_reason
# condition — on this pin the variable exists only in the non-streaming
# full generator (pristine serving.py:1069+). Anchor refreshed to the
# pristine block at chat_completion/serving.py:821-828 (count==1
# verified against /private/tmp/candidate_pin_current):
#
#     if (tools_streamed[i] and not tool_choice_function_name) or (
#         self.use_harmony and harmony_tools_streamed[i]
#     ):
#         finish_reason_ = "tool_calls"
#     else:
#         finish_reason_ = (
#             output.finish_reason if output.finish_reason else "stop"
#         )
#
# APPLY-ORDER CHAIN with P107: P107 v3's ANCHOR_OLD spans this same
# block PLUS the following ``choice_data =
# ChatCompletionResponseStreamChoice(`` line, and its ANCHOR_NEW keeps
# the block verbatim — so PN288 applies cleanly on BOTH pristine and
# post-P107 content. The reverse order does NOT compose: PN288's
# replacement re-indents the block (+4 spaces) inside the
# except-fallback, destroying P107's anchor. The registry therefore
# declares ``requires_patches: ["P107"]`` on PN288 (ordering-only —
# PN288 still applies standalone when P107 is disabled). Proven in
# tests/unit/integrations/serving/test_pn288_p107_anchor_coordination.py.


PN288_STREAMING_OLD = (
    "                        if (tools_streamed[i] and not tool_choice_function_name) or (\n"
    "                            self.use_harmony and harmony_tools_streamed[i]\n"
    "                        ):\n"
    "                            finish_reason_ = \"tool_calls\"\n"
    "                        else:\n"
    "                            finish_reason_ = (\n"
    "                                output.finish_reason if output.finish_reason else \"stop\"\n"
    "                            )\n"
)


PN288_STREAMING_NEW = (
    "                        # [Genesis PN288] args-validity-aware finish_reason\n"
    "                        # decision. Phase B (dry-run) by default — see\n"
    "                        # sndr.engines.vllm.middleware.pn288_finish_reason_override.\n"
    "                        try:\n"
    "                            from sndr.engines.vllm.middleware.pn288_finish_reason_override import (  # noqa: E501\n"
    "                                decide_streaming_finish_reason as _genesis_pn288_decide_streaming,\n"
    "                            )\n"
    "                            finish_reason_ = _genesis_pn288_decide_streaming(\n"
    "                                tools_streamed_i=tools_streamed[i],\n"
    "                                tool_choice_function_name=tool_choice_function_name,\n"
    "                                use_harmony=self.use_harmony,\n"
    "                                harmony_tools_streamed_i=harmony_tools_streamed[i],\n"
    "                                output=output,\n"
    "                                request=request,\n"
    "                                tool_parser=getattr(self, \"tool_parser\", None),\n"
    "                            )\n"
    "                        except Exception:\n"
    "                            # Defensive fallback — replicate upstream verbatim\n"
    "                            # (pristine serving.py:821-828). The pre-0.22 auto\n"
    "                            # tool-call flag no longer exists in the streaming\n"
    "                            # generator — referencing it here would raise a\n"
    "                            # NameError that ESCAPES this defensive wrapper.\n"
    "                            if (tools_streamed[i] and not tool_choice_function_name) or (\n"
    "                                self.use_harmony and harmony_tools_streamed[i]\n"
    "                            ):\n"
    "                                finish_reason_ = \"tool_calls\"\n"
    "                            else:\n"
    "                                finish_reason_ = (\n"
    "                                    output.finish_reason if output.finish_reason else \"stop\"\n"
    "                                )\n"
)


# ─── Sub-patch 2: non-streaming anchor ─────────────────────────────────


PN288_NONSTREAMING_OLD = (
    "            is_finish_reason_tool_calls = auto_tools_called or (\n"
    "                request.tool_choice\n"
    "                and request.tool_choice == \"required\"\n"
    "                and output.finish_reason == \"stop\"\n"
    "            )\n"
)


PN288_NONSTREAMING_NEW = (
    "            # [Genesis PN288] args-validity-aware bool. Phase B dry-run\n"
    "            # default — see sndr.engines.vllm.middleware.pn288_finish_reason_override.\n"
    "            try:\n"
    "                from sndr.engines.vllm.middleware.pn288_finish_reason_override import (  # noqa: E501\n"
    "                    decide_non_streaming_is_tool_calls as _genesis_pn288_decide_non_streaming,\n"
    "                )\n"
    "                is_finish_reason_tool_calls = _genesis_pn288_decide_non_streaming(\n"
    "                    auto_tools_called=auto_tools_called,\n"
    "                    request=request,\n"
    "                    output=output,\n"
    "                    tool_parser=getattr(self, \"tool_parser\", None),\n"
    "                )\n"
    "            except Exception:\n"
    "                # Defensive fallback — replicate upstream verbatim.\n"
    "                is_finish_reason_tool_calls = auto_tools_called or (\n"
    "                    request.tool_choice\n"
    "                    and request.tool_choice == \"required\"\n"
    "                    and output.finish_reason == \"stop\"\n"
    "                )\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(
        "entrypoints/openai/chat_completion/serving.py"
    )
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN288 serving.py — tool-call finish_reason override "
            "(Phase B dry-run)"
        ),
        target_file=str(target),
        marker=GENESIS_PN288_MARKER,
        sub_patches=[
            TextPatch(
                name="pn288_streaming_finish_reason",
                anchor=PN288_STREAMING_OLD,
                replacement=PN288_STREAMING_NEW,
                required=True,
            ),
            TextPatch(
                name="pn288_non_streaming_finish_reason",
                anchor=PN288_NONSTREAMING_OLD,
                replacement=PN288_NONSTREAMING_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            # Self-collision lint (triage plan §6 2026-06-11): former
            # entries "_genesis_pn288_decide_streaming" /
            # "_genesis_pn288_decide_non_streaming" were baked by our own
            # replacements — residue coverage stays with the
            # "[Genesis PN288" banner.
            "[Genesis PN288",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN288 text patch.

    Phase B semantics: when the gate
    ``GENESIS_ENABLE_PN288_TOOL_FINISH_REASON_OVERRIDE=1`` is set, both
    anchors are replaced with the middleware-delegating calls. The
    text-patched code itself dispatches dry-run vs actual override at
    request time via the middleware's ``is_dry_run()`` check, which
    reads ``GENESIS_PN288_DRY_RUN`` live on every call.
    """
    from sndr.dispatcher import should_apply, log_decision
    from sndr.engines.vllm.middleware.pn288_finish_reason_override import (
        setup_prometheus_counters,
    )

    decision, reason = should_apply("PN288")
    log_decision("PN288", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", (
            "vllm/entrypoints/openai/chat_completion/serving.py "
            "not found"
        )

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"

    with open(patcher.target_file) as f:
        content = f.read()

    # Idempotency + upstream-drift check.
    if patcher.marker in content:
        pass  # already applied
    else:
        for m in patcher.upstream_drift_markers:
            if m in content:
                return (
                    "skipped",
                    f"upstream drift marker {m!r} already in "
                    f"{patcher.target_file} — PN288 already injected "
                    "or upstream landed an equivalent fix.",
                )
        # Pre-flight: confirm both anchors are present before we start.
        for sub in patcher.sub_patches:
            if sub.anchor not in content:
                return (
                    "skipped",
                    f"required anchor {sub.name!r} not found in "
                    f"{patcher.target_file} — upstream drift; refresh "
                    "anchor against the current pin.",
                )

    result, failure = patcher.apply()
    if result == TextPatchResult.SKIPPED:
        _r = failure.reason if failure else "anchor drift / not eligible"
        _d = f" ({failure.detail})" if (failure and failure.detail) else ""
        return "skipped", f"{patcher.patch_name}: {_r}{_d}"
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: "
            f"{failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )

    # Register the Prometheus counter once the patch is in place.
    prom_ready = setup_prometheus_counters()
    prom_note = (
        " + Prometheus counter registered "
        "(vllm:pn288_finish_reason_override_total{model,channel,action})"
        if prom_ready else
        " (prometheus_client unavailable; module-global dict only)"
    )

    return "applied", (
        "PN288 finish_reason override installed at both serving.py "
        "anchors (streaming + non-streaming). Phase B dry-run is "
        "ACTIVE by default; set GENESIS_PN288_DRY_RUN=0 to enable "
        "Phase C behavior change after evidence review."
        + prom_note
    )


def is_applied() -> bool:
    """Filesystem-level marker check — True iff serving.py carries the
    PN288 patch marker. Cheap; used by audit / shadow CLI."""
    target = resolve_vllm_file(
        "entrypoints/openai/chat_completion/serving.py"
    )
    if target is None:
        return False
    try:
        with open(target) as f:
            return GENESIS_PN288_MARKER in f.read()
    except OSError:
        return False
