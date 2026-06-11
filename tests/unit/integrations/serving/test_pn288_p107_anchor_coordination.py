# SPDX-License-Identifier: Apache-2.0
"""PN288 ↔ P107 anchor coordination (2026-06-11 pin re-anchor batch).

Both patches target the SAME pristine streaming finish_reason block in
``vllm/entrypoints/openai/chat_completion/serving.py`` (lines 821-828 on
pin 0.22.1rc1.dev259+g303916e93):

    if (tools_streamed[i] and not tool_choice_function_name) or (
        self.use_harmony and harmony_tools_streamed[i]
    ):
        finish_reason_ = "tool_calls"
    else:
        finish_reason_ = (
            output.finish_reason if output.finish_reason else "stop"
        )

P107 v3's ANCHOR_OLD spans this block PLUS the following
``choice_data = ChatCompletionResponseStreamChoice(`` line; its
ANCHOR_NEW preserves the block verbatim and only inserts the detector
before the choice_data line. PN288's streaming anchor is the bare block.

Consequence (proved below): the pair composes ONLY in P107-then-PN288
order. PN288's replacement re-indents the block inside its
except-fallback (+4 spaces), destroying P107's anchor — so the registry
declares ``requires_patches: ["P107"]`` on PN288 (apply-order chain
convention; ordering-only, PN288 still applies standalone on pristine).
"""
from __future__ import annotations

import ast
import os

import pytest


PRISTINE_SERVING = (
    "/private/tmp/candidate_pin_current/vllm/entrypoints/openai/"
    "chat_completion/serving.py"
)

# Six nesting levels -> innermost body at 24-space indent, matching the
# real streaming generator's loop depth around serving.py:821.
_STREAM_HEADER = (
    "async def _synthetic_stream_generator():\n"
    "    if 1:\n"
    "        if 1:\n"
    "            if 1:\n"
    "                if 1:\n"
    "                    if 1:\n"
)

# Two nesting levels -> innermost body at 12-space indent, matching the
# full (non-streaming) generator depth around serving.py:1246.
_FULL_HEADER = (
    "async def _synthetic_full_generator():\n"
    "    if 1:\n"
    "        if 1:\n"
)


def _pn288():
    from sndr.engines.vllm.patches.serving import (
        pn288_tool_finish_reason_override as M,
    )
    return M


def _p107():
    from sndr.engines.vllm.patches.serving import (
        p107_mtp_truncation_detector as M,
    )
    return M


def _stream_fixture() -> str:
    """Synthetic pristine streaming region: P107.ANCHOR_OLD *is*
    byte-exact pristine content (block + choice_data open-paren line);
    close the call so ``ast.parse`` accepts the whole snippet."""
    return _STREAM_HEADER + _p107().ANCHOR_OLD + ")\n"


def _full_fixture() -> str:
    return _FULL_HEADER + _pn288().PN288_NONSTREAMING_OLD


# ─── Anchor shape: chain precondition ───────────────────────────────────


def test_pn288_streaming_anchor_is_prefix_of_p107_anchor():
    """P107's anchor = PN288's anchor + choice_data line; P107's
    replacement must keep PN288's anchor verbatim exactly once —
    the precondition for the P107-then-PN288 chain."""
    pn288, p107 = _pn288(), _p107()
    assert p107.ANCHOR_OLD.startswith(pn288.PN288_STREAMING_OLD)
    assert p107.ANCHOR_NEW.count(pn288.PN288_STREAMING_OLD) == 1


def test_streaming_new_must_not_reference_auto_tools_called():
    """Upstream removed ``auto_tools_called`` from the streaming
    generator on this pin — any reference in the injected text (kwarg
    OR except-fallback) is a latent NameError; the fallback variant
    would even ESCAPE the defensive try/except wrapper."""
    pn288 = _pn288()
    assert "auto_tools_called" not in pn288.PN288_STREAMING_NEW
    assert "auto_tools_called" not in pn288.PN288_STREAMING_OLD


def test_non_streaming_sub_patch_untouched():
    """Sub-patch 2 targets the full generator where the variable still
    exists (pristine serving.py:1246-1250) — it must keep passing
    ``auto_tools_called``."""
    pn288 = _pn288()
    assert "auto_tools_called" in pn288.PN288_NONSTREAMING_OLD
    assert "auto_tools_called" in pn288.PN288_NONSTREAMING_NEW


def test_no_self_collision_between_old_and_new():
    """Idempotency safety: the except-fallback replicates the anchor
    shifted +4 spaces — it must NOT contain the anchor as a substring,
    or a marker-less re-apply would double-patch."""
    pn288 = _pn288()
    assert pn288.PN288_STREAMING_OLD not in pn288.PN288_STREAMING_NEW
    assert pn288.PN288_NONSTREAMING_OLD not in pn288.PN288_NONSTREAMING_NEW


# ─── Order A: P107 then PN288 — must compose ────────────────────────────


def test_order_p107_then_pn288_composes_and_parses():
    pn288, p107 = _pn288(), _p107()
    body = _stream_fixture()
    assert body.count(p107.ANCHOR_OLD) == 1
    body = body.replace(p107.ANCHOR_OLD, p107.ANCHOR_NEW)
    # P107's output preserves PN288's anchor exactly once.
    assert body.count(pn288.PN288_STREAMING_OLD) == 1
    body = body.replace(
        pn288.PN288_STREAMING_OLD, pn288.PN288_STREAMING_NEW
    )
    assert "_genesis_pn288_decide_streaming" in body
    assert "[Genesis P107" in body
    ast.parse(body)  # both injections — still valid python

    full = _full_fixture()
    assert full.count(pn288.PN288_NONSTREAMING_OLD) == 1
    full = full.replace(
        pn288.PN288_NONSTREAMING_OLD, pn288.PN288_NONSTREAMING_NEW
    )
    assert "_genesis_pn288_decide_non_streaming" in full
    ast.parse(full)


# ─── Order B: PN288 then P107 — proven collision, hence the chain ───────


def test_order_pn288_then_p107_collides():
    """PN288 first kills P107's anchor (block re-indented +4 inside the
    except-fallback). This is WHY the registry chains PN288 after P107
    instead of claiming order-independence."""
    pn288, p107 = _pn288(), _p107()
    body = _stream_fixture()
    assert body.count(pn288.PN288_STREAMING_OLD) == 1
    body = body.replace(
        pn288.PN288_STREAMING_OLD, pn288.PN288_STREAMING_NEW
    )
    ast.parse(body)  # PN288 alone is valid python...
    assert body.count(p107.ANCHOR_OLD) == 0  # ...but P107 can no longer apply


def test_registry_declares_p107_chain():
    """Chain convention: requires_patches orders P107 before PN288 in
    the topological apply path (dependency-first per spec.py)."""
    from sndr.dispatcher import PATCH_REGISTRY
    from sndr.dispatcher.spec import _topological_order

    assert "P107" in (PATCH_REGISTRY["PN288"].get("requires_patches") or [])
    order = _topological_order(PATCH_REGISTRY)
    assert order.index("P107") < order.index("PN288")


# ─── Middleware: kwarg becomes optional ─────────────────────────────────


def test_middleware_streaming_accepts_call_without_auto_tools_called():
    """The new injected call site no longer passes ``auto_tools_called``
    — the middleware must default it to False and return upstream's
    verdict for the plain stop path."""
    import inspect

    from sndr.engines.vllm.middleware import (
        pn288_finish_reason_override as mw,
    )

    sig = inspect.signature(mw.decide_streaming_finish_reason)
    param = sig.parameters["auto_tools_called"]
    assert param.default is False

    class _Out:
        finish_reason = None

    verdict = mw.decide_streaming_finish_reason(
        tools_streamed_i=False,
        tool_choice_function_name=None,
        use_harmony=False,
        harmony_tools_streamed_i=False,
        output=_Out(),
        request=None,
        tool_parser=None,
    )
    assert verdict == "stop"

    verdict = mw.decide_streaming_finish_reason(
        tools_streamed_i=True,
        tool_choice_function_name=None,
        use_harmony=False,
        harmony_tools_streamed_i=False,
        output=_Out(),
        request=None,
        tool_parser=None,
    )
    assert verdict == "tool_calls"


# ─── Pristine-tree byte-exactness (skipped when tree absent) ────────────


@pytest.mark.skipif(
    not os.path.isfile(PRISTINE_SERVING),
    reason="pristine candidate pin tree not present on this machine",
)
def test_anchors_byte_exact_on_pristine_tree():
    pristine = open(PRISTINE_SERVING).read()
    pn288, p107 = _pn288(), _p107()
    assert pristine.count(pn288.PN288_STREAMING_OLD) == 1
    assert pristine.count(pn288.PN288_NONSTREAMING_OLD) == 1
    assert pristine.count(p107.ANCHOR_OLD) == 1
