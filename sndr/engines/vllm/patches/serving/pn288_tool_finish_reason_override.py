# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN288 — qwen3_coder tool_call finish_reason override
(§1.3 of the unified plan, Phase B / dry-run scaffold).

The decision logic lives in the companion middleware module
``vllm.sndr_core.middleware.pn288_finish_reason_override``. This file
is the text-patch overlay that wires that logic into
``OpenAIServingChat._create_chat_completion`` at two anchors:

  1. **Streaming** (serving.py:884-893 on pin 626fa9bb) — the if-block
     that assigns ``finish_reason_`` inside the choice-data loop.
  2. **Non-streaming** (serving.py:1306-1310) — the
     ``is_finish_reason_tool_calls`` bool assignment.

Anchor strategy
---------------
Verified live against the running ``vllm-gemma4-31b-tq-mtp-structured-k4-k4``
container 2026-05-30 (pin 626fa9bb). Both anchors are stable across
the 0.21.1rc0 → 0.21.1rc1+g626fa9bba window — none of the merged PRs
between dev371 and 626fa9bb touched the finish_reason emission sites.

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

from vllm.sndr_core.detection.guards import (
    resolve_vllm_file, vllm_install_root,
)
from vllm.sndr_core.core import (
    TextPatch, TextPatcher, TextPatchResult,
)


log = logging.getLogger("genesis.wiring.pn288_tool_finish_reason_override")

GENESIS_PN288_MARKER = (
    "Genesis PN288 tool-call finish_reason override v1 (Phase B dry-run)"
)


# ─── Sub-patch 1: streaming anchor ──────────────────────────────────────


PN288_STREAMING_OLD = (
    "                        if (\n"
    "                            auto_tools_called\n"
    "                            or (tools_streamed[i] and not tool_choice_function_name)\n"
    "                            or (self.use_harmony and harmony_tools_streamed[i])\n"
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
    "                        # vllm.sndr_core.middleware.pn288_finish_reason_override.\n"
    "                        try:\n"
    "                            from vllm.sndr_core.middleware.pn288_finish_reason_override import (  # noqa: E501\n"
    "                                decide_streaming_finish_reason as _genesis_pn288_decide_streaming,\n"
    "                            )\n"
    "                            finish_reason_ = _genesis_pn288_decide_streaming(\n"
    "                                auto_tools_called=auto_tools_called,\n"
    "                                tools_streamed_i=tools_streamed[i],\n"
    "                                tool_choice_function_name=tool_choice_function_name,\n"
    "                                use_harmony=self.use_harmony,\n"
    "                                harmony_tools_streamed_i=harmony_tools_streamed[i],\n"
    "                                output=output,\n"
    "                                request=request,\n"
    "                                tool_parser=getattr(self, \"tool_parser\", None),\n"
    "                            )\n"
    "                        except Exception:\n"
    "                            # Defensive fallback — replicate upstream verbatim.\n"
    "                            if (\n"
    "                                auto_tools_called\n"
    "                                or (tools_streamed[i] and not tool_choice_function_name)\n"
    "                                or (self.use_harmony and harmony_tools_streamed[i])\n"
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
    "            # default — see vllm.sndr_core.middleware.pn288_finish_reason_override.\n"
    "            try:\n"
    "                from vllm.sndr_core.middleware.pn288_finish_reason_override import (  # noqa: E501\n"
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
            "[Genesis PN288",
            "_genesis_pn288_decide_streaming",
            "_genesis_pn288_decide_non_streaming",
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
    from vllm.sndr_core.dispatcher import should_apply, log_decision
    from vllm.sndr_core.middleware.pn288_finish_reason_override import (
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
