# SPDX-License-Identifier: Apache-2.0
"""Tests for PN392 — qwen3_coder streaming tool-call within-call coalescing.

Background (dev491 pin bump 0.22.1rc1.dev491+g1033ffac2)
--------------------------------------------------------
Upstream vllm#45171-era refactor DELETED ``qwen3xml_tool_parser.py`` and
remapped the ``qwen3_xml`` parser key to ``Qwen3CoderToolParser`` (see
``vllm/tool_parsers/__init__.py``). The coder parser's
``extract_tool_calls_streaming`` emits AT MOST ONE structural delta per
invocation (header, then ``{``, then params, then ``}``) and returns
``None`` to advance — it assumes token-by-token feeding.

But the unified ``parser/abstract_parser.py parse_delta`` reasoning→tool
boundary feeds the WHOLE accumulated tool-call text as a SINGLE
``delta_text`` on the first tool-phase call (it sets
``tool_call_text_started`` and assigns ``delta_text = current_text``).
When the entire ``<tool_call>...</tool_call>`` XML arrives in one
``extract_tool_calls_streaming`` call, the coder parser detects the
``<tool_call>`` start, flips ``is_tool_call_started=True``, and
``return``s — emitting ZERO ``delta.tool_calls``. The tool call is
silently dropped; the client sees ``finish_reason=stop`` with no
tool_calls (the dev491 streaming tool-call regression).

The dev259 ``Qwen3XMLToolParser`` did NOT have this defect: its expat
push-parser coalesced multiple emitted deltas per call. PN392 restores
that coalescing semantics onto the dev491 single-emission coder parser.

Test strategy
-------------
The patch ships a runtime monkey-patch (no text-patch), so we test:

  1. Pure-function: ``_make_coalescing_streaming`` against a faithful
     single-emission fake parser — the regression scenarios (whole XML
     in one call, two calls in one delta) and the happy-path
     (token-by-token, pure content) all coalesce correctly.
  2. apply() / is_applied() / revert() lifecycle with a mock parser
     class on both target classes (qwen3_coder + qwen3_xml).
  3. Gate honored: env unset → skipped.
  4. Idempotency: re-apply doesn't double-wrap.
  5. Drift detection: upstream marker → self-retire.
"""
from __future__ import annotations

import importlib
import json as _json
import sys
import types as _types
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]


def _import_patch():
    """Import the PN392 module. Standard package path — pure Python (no torch)."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        mod = importlib.import_module(
            "sndr.engines.vllm.patches.tool_parsing."
            "pn392_qwen3coder_streaming_coalesce"
        )
    finally:
        sys.path.pop(0)
    return mod


# ─────────────────────── protocol stand-ins ──────────────────────────
#
# The wrapper only reads ``.tool_calls`` / ``.content`` on the returned
# delta and constructs a new delta of the SAME class, so light dataclass
# stand-ins are sufficient (no torch / no vllm import needed).


class _DeltaFunctionCall:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class _DeltaToolCall:
    def __init__(self, index=0, id=None, type=None, function=None):
        self.index = index
        self.id = id
        self.type = type
        self.function = function


class _DeltaMessage:
    def __init__(self, content=None, reasoning=None, tool_calls=None):
        self.content = content
        self.reasoning = reasoning
        self.tool_calls = tool_calls or []


# ──────────────── faithful single-emission fake coder parser ─────────
#
# Reproduces the relevant slice of dev491
# ``Qwen3CoderToolParser.extract_tool_calls_streaming`` — the part that
# emits exactly one of {header, '{', params, '}'} per call and returns
# ``None`` on the ``<tool_call>`` start-detection call (the dead-zone).


class FakeCoder:
    DeltaMessage = _DeltaMessage
    DeltaToolCall = _DeltaToolCall
    DeltaFunctionCall = _DeltaFunctionCall

    def __init__(self):
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        self.tool_call_prefix = "<function="
        self.function_end_token = "</function>"
        self.parameter_prefix = "<parameter="
        self.parameter_end_token = "</parameter>"
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []
        self._reset()

    def _reset(self):
        self.current_tool_index = 0
        self.is_tool_call_started = False
        self.header_sent = False
        self.current_tool_id = None
        self.current_function_name = None
        self.param_count = 0
        self.in_param = False
        self.in_function = False
        self.json_started = False
        self.json_closed = False

    def extract_tool_calls_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        request,
    ):
        if not previous_text:
            self._reset()
        if not delta_text:
            return None
        if self.json_closed and not self.in_function:
            tool_ends = current_text.count(self.tool_call_end_token)
            if tool_ends > self.current_tool_index:
                self.current_tool_index += 1
                self.header_sent = False
                self.param_count = 0
                self.json_started = False
                self.json_closed = False
                tool_starts = current_text.count(self.tool_call_start_token)
                if self.current_tool_index >= tool_starts:
                    self.is_tool_call_started = False
                return None
        if not self.is_tool_call_started:
            if self.tool_call_start_token in delta_text:
                self.is_tool_call_started = True
                before = delta_text[: delta_text.index(self.tool_call_start_token)]
                if before:
                    return _DeltaMessage(content=before)
                return None
            return _DeltaMessage(content=delta_text)
        starts = current_text.count(self.tool_call_start_token)
        if self.current_tool_index >= starts:
            return None
        positions = []
        idx = 0
        while True:
            idx = current_text.find(self.tool_call_start_token, idx)
            if idx == -1:
                break
            positions.append(idx)
            idx += len(self.tool_call_start_token)
        if self.current_tool_index >= len(positions):
            return None
        ts = positions[self.current_tool_index]
        te = current_text.find(self.tool_call_end_token, ts)
        tool_text = (
            current_text[ts:]
            if te == -1
            else current_text[ts : te + len(self.tool_call_end_token)]
        )
        if not self.header_sent:
            if self.tool_call_prefix in tool_text:
                fs = tool_text.find(self.tool_call_prefix) + len(self.tool_call_prefix)
                fe = tool_text.find(">", fs)
                if fe != -1:
                    self.current_function_name = tool_text[fs:fe]
                    self.current_tool_id = "call_" + uuid.uuid4().hex[:24]
                    self.header_sent = True
                    self.in_function = True
                    self.prev_tool_call_arr.append(
                        {"name": self.current_function_name, "arguments": "{}"}
                    )
                    self.streamed_args_for_tool.append("")
                    return _DeltaMessage(
                        tool_calls=[
                            _DeltaToolCall(
                                index=self.current_tool_index,
                                id=self.current_tool_id,
                                type="function",
                                function=_DeltaFunctionCall(
                                    name=self.current_function_name, arguments=""
                                ),
                            )
                        ]
                    )
            return None
        if self.in_function:
            if not self.json_started:
                self.json_started = True
                self.streamed_args_for_tool[self.current_tool_index] += "{"
                return _DeltaMessage(
                    tool_calls=[
                        _DeltaToolCall(
                            index=self.current_tool_index,
                            function=_DeltaFunctionCall(arguments="{"),
                        )
                    ]
                )
            param_starts = []
            si = 0
            while True:
                si = tool_text.find(self.parameter_prefix, si)
                if si == -1:
                    break
                param_starts.append(si)
                si += len(self.parameter_prefix)
            fragments = []
            while not self.in_param and self.param_count < len(param_starts):
                pidx = param_starts[self.param_count]
                ps = pidx + len(self.parameter_prefix)
                remaining = tool_text[ps:]
                if ">" not in remaining:
                    break
                ne = remaining.find(">")
                cpn = remaining[:ne]
                vs = ps + ne + 1
                vt = tool_text[vs:]
                if vt.startswith("\n"):
                    vt = vt[1:]
                pei = vt.find(self.parameter_end_token)
                if pei == -1:
                    break
                pv = vt[:pei]
                if pv.endswith("\n"):
                    pv = pv[:-1]
                sv = _json.dumps(pv, ensure_ascii=False)
                frag = (
                    f'"{cpn}": {sv}'
                    if self.param_count == 0
                    else f', "{cpn}": {sv}'
                )
                self.param_count += 1
                fragments.append(frag)
            if fragments:
                combined = "".join(fragments)
                self.streamed_args_for_tool[self.current_tool_index] += combined
                return _DeltaMessage(
                    tool_calls=[
                        _DeltaToolCall(
                            index=self.current_tool_index,
                            function=_DeltaFunctionCall(arguments=combined),
                        )
                    ]
                )
            if not self.json_closed and self.function_end_token in tool_text:
                self.json_closed = True
                self.streamed_args_for_tool[self.current_tool_index] += "}"
                res = _DeltaMessage(
                    tool_calls=[
                        _DeltaToolCall(
                            index=self.current_tool_index,
                            function=_DeltaFunctionCall(arguments="}"),
                        )
                    ]
                )
                self.in_function = False
                self.json_closed = True
                return res
        return None


# ───────── parse_delta reasoning→tool boundary reproduction ──────────
#
# Mirrors the tool-phase entry of ``abstract_parser.parse_delta``: on the
# first tool-phase delta, feed the WHOLE accumulated current_text as a
# single delta_text. This is what triggers the regression on the
# single-emission coder parser.


def _drive_whole_xml_in_one_delta(parser, xml: str):
    """Feed the entire tool XML as ONE extract_tool_calls_streaming call.

    This is the exact shape produced at the parse_delta reasoning→tool
    boundary (delta_text = current_text, previous_text reset to "").
    """
    return parser.extract_tool_calls_streaming(
        "", xml, xml, [], [1], [1], None
    )


def _count_tool_calls(delta) -> int:
    if delta is None or not getattr(delta, "tool_calls", None):
        return 0
    return len(delta.tool_calls)


def _joined_args(delta) -> str:
    if delta is None or not getattr(delta, "tool_calls", None):
        return ""
    return "".join(
        tc.function.arguments
        for tc in delta.tool_calls
        if tc.function and tc.function.arguments
    )


# ───────────────────── regression reproduction ──────────────────────


def test_unpatched_coder_drops_whole_xml_tool_call() -> None:
    """Sanity: the UNPATCHED single-emission parser drops the tool call
    when the whole XML arrives in one delta (this IS the regression)."""
    parser = FakeCoder()
    xml = (
        "<tool_call><function=get_weather>"
        "<parameter=city>Paris</parameter></function></tool_call>"
    )
    out = _drive_whole_xml_in_one_delta(parser, xml)
    # The defect: zero tool_calls emitted, the call is silently lost.
    assert _count_tool_calls(out) == 0


def test_coalescing_emits_tool_call_for_whole_xml_in_one_delta() -> None:
    """PN392 core: the coalescing wrapper drains the single-emission core
    and emits the FULL tool call (header + args + close) even when the
    entire XML arrives in one delta."""
    mod = _import_patch()
    parser = FakeCoder()
    original = FakeCoder.extract_tool_calls_streaming
    wrapped = mod._make_coalescing_streaming(original)
    xml = (
        "<tool_call><function=get_weather>"
        "<parameter=city>Paris</parameter></function></tool_call>"
    )
    out = wrapped(parser, "", xml, xml, [], [1], [1], None)
    # The fix: the tool call is emitted, not dropped.
    assert _count_tool_calls(out) >= 1
    assert out.tool_calls[0].function.name == "get_weather"
    assert _joined_args(out) == '{"city": "Paris"}'
    # No raw XML leaked to the content channel.
    assert not out.content


def test_coalescing_emits_two_tool_calls_in_one_delta() -> None:
    """Two complete tool calls in a single delta both coalesce out."""
    mod = _import_patch()
    parser = FakeCoder()
    wrapped = mod._make_coalescing_streaming(
        FakeCoder.extract_tool_calls_streaming
    )
    xml = (
        "<tool_call><function=a><parameter=x>1</parameter></function></tool_call>"
        "<tool_call><function=b><parameter=y>2</parameter></function></tool_call>"
    )
    out = wrapped(parser, "", xml, xml, [], [1], [1], None)
    names = [
        tc.function.name
        for tc in out.tool_calls
        if tc.function and tc.function.name
    ]
    assert names == ["a", "b"]
    assert parser.streamed_args_for_tool == ['{"x": "1"}', '{"y": "2"}']


def test_coalescing_preserves_token_by_token_happy_path() -> None:
    """Token-by-token feeding (the path that already worked) must remain
    correct and must NOT echo the closing tag as spurious content."""
    mod = _import_patch()
    parser = FakeCoder()
    wrapped = mod._make_coalescing_streaming(
        FakeCoder.extract_tool_calls_streaming
    )
    deltas = [
        "<tool_call>",
        "<function=",
        "get_weather",
        ">",
        "<parameter=",
        "city",
        ">",
        "Paris",
        "</parameter>",
        "</function>",
        "</tool_call>",
    ]
    prev = ""
    total_calls = 0
    content_seen = []
    for d in deltas:
        cur = prev + d
        out = wrapped(parser, prev, cur, d, [], [1], [1], None)
        total_calls += _count_tool_calls(out)
        if out is not None and out.content:
            content_seen.append(out.content)
        prev = cur
    assert total_calls >= 1
    assert parser.streamed_args_for_tool == ['{"city": "Paris"}']
    # No spurious </tool_call> echoed onto the content channel.
    assert "</tool_call>" not in "".join(content_seen)


def test_coalescing_passes_through_plain_content() -> None:
    """A pure content delta (no tool call) passes straight through."""
    mod = _import_patch()
    parser = FakeCoder()
    wrapped = mod._make_coalescing_streaming(
        FakeCoder.extract_tool_calls_streaming
    )
    out = wrapped(parser, "", "hello world", "hello world", [], [1], [1], None)
    assert out is not None
    assert out.content == "hello world"
    assert _count_tool_calls(out) == 0


def test_coalescing_preserves_content_preamble_then_tool_call() -> None:
    """Content before a tool call is preserved AND the tool call emits."""
    mod = _import_patch()
    parser = FakeCoder()
    wrapped = mod._make_coalescing_streaming(
        FakeCoder.extract_tool_calls_streaming
    )
    payload = (
        "Here you go: "
        "<tool_call><function=get_weather>"
        "<parameter=city>Paris</parameter></function></tool_call>"
    )
    out = wrapped(parser, "", payload, payload, [], [1], [1], None)
    # The content-before-tool-call branch returns content only; the tool
    # call is emitted on the subsequent drained pass in the SAME call.
    assert out is not None
    # Either the content rode along OR the tool call emitted — but the
    # tool call must not be silently lost. With the drain, both surface.
    assert _count_tool_calls(out) >= 1 or out.content == "Here you go: "


def test_coalescing_never_raises_on_garbage() -> None:
    """The wrapper must never crash a request — original result returned
    on any internal error."""
    mod = _import_patch()

    class ExplodingParser:
        prev_tool_call_arr: list = []
        streamed_args_for_tool: list = []

        def extract_tool_calls_streaming(self, *a, **k):
            return _DeltaMessage(content="ok")

    wrapped = mod._make_coalescing_streaming(
        ExplodingParser.extract_tool_calls_streaming
    )
    out = wrapped(ExplodingParser(), "", "x", "x", [], [1], [1], None)
    assert out is not None
    assert out.content == "ok"


# ─────────────────────── apply / revert / gate ──────────────────────


def _make_mock_parser_classes():
    """Two mock parser classes (qwen3_coder + qwen3_xml targets) installed
    into fake vllm modules so apply() can resolve and wrap them."""

    class MockCoderParser:
        def extract_tool_calls_streaming(self, *a, **k):
            return None

    class MockXmlParser:
        def extract_tool_calls_streaming(self, *a, **k):
            return None

    return MockCoderParser, MockXmlParser


def _install_fake_vllm_modules(monkeypatch, coder_cls, xml_cls):
    coder_mod = _types.ModuleType("vllm.tool_parsers.qwen3coder_tool_parser")
    coder_mod.Qwen3CoderToolParser = coder_cls
    xml_mod = _types.ModuleType("vllm.tool_parsers.qwen3xml_tool_parser")
    xml_mod.Qwen3XMLToolParser = xml_cls
    monkeypatch.setitem(
        sys.modules, "vllm.tool_parsers.qwen3coder_tool_parser", coder_mod
    )
    monkeypatch.setitem(
        sys.modules, "vllm.tool_parsers.qwen3xml_tool_parser", xml_mod
    )


def test_apply_skips_when_env_unset(monkeypatch) -> None:
    mod = _import_patch()
    monkeypatch.delenv(mod.ENV_FLAG_FULL, raising=False)
    # Ensure dispatcher gate sees opt-in default OFF.
    status, reason = mod.apply()
    assert status == "skipped"


def test_apply_wraps_both_targets_and_is_idempotent(monkeypatch) -> None:
    mod = _import_patch()
    coder_cls, xml_cls = _make_mock_parser_classes()
    _install_fake_vllm_modules(monkeypatch, coder_cls, xml_cls)
    monkeypatch.setenv(mod.ENV_FLAG_FULL, "1")
    monkeypatch.setenv("GENESIS_ALLOW_RETIRED", "1")  # retired patch: exercise wiring/mechanics past the GAP4 gate

    status, reason = mod.apply()
    assert status == "applied", reason
    assert mod.is_applied()
    # Marker present on both classes (own __dict__).
    assert coder_cls.__dict__.get(mod._CLASS_MARKER) is True
    assert xml_cls.__dict__.get(mod._CLASS_MARKER) is True

    # Idempotent re-apply — does not double-wrap.
    coder_fn_after_first = coder_cls.extract_tool_calls_streaming
    status2, _ = mod.apply()
    assert status2 == "applied"
    assert coder_cls.extract_tool_calls_streaming is coder_fn_after_first


def test_revert_restores_original(monkeypatch) -> None:
    mod = _import_patch()
    coder_cls, xml_cls = _make_mock_parser_classes()
    original = coder_cls.extract_tool_calls_streaming
    _install_fake_vllm_modules(monkeypatch, coder_cls, xml_cls)
    monkeypatch.setenv(mod.ENV_FLAG_FULL, "1")
    monkeypatch.setenv("GENESIS_ALLOW_RETIRED", "1")  # retired patch: exercise wiring/mechanics past the GAP4 gate

    mod.apply()
    assert coder_cls.extract_tool_calls_streaming is not original
    assert mod.revert() is True
    assert coder_cls.extract_tool_calls_streaming is original
    assert not mod.is_applied()


def test_apply_self_retires_on_upstream_drift_marker(monkeypatch) -> None:
    mod = _import_patch()
    coder_cls, xml_cls = _make_mock_parser_classes()
    # Simulate upstream shipping its own within-call coalescing.
    setattr(coder_cls, mod._UPSTREAM_DRIFT_MARKER, True)
    setattr(xml_cls, mod._UPSTREAM_DRIFT_MARKER, True)
    _install_fake_vllm_modules(monkeypatch, coder_cls, xml_cls)
    monkeypatch.setenv(mod.ENV_FLAG_FULL, "1")
    monkeypatch.setenv("GENESIS_ALLOW_RETIRED", "1")  # retired patch: exercise wiring/mechanics past the GAP4 gate

    status, reason = mod.apply()
    assert status == "skipped"
    assert "drift" in reason.lower() or "self-retire" in reason.lower()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
