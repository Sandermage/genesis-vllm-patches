# SPDX-License-Identifier: Apache-2.0
"""G4_T1 overlays — MTP-split streaming regression corpus (PR #45068/#44844).

Until this file landed (sweep 2026-06-11, chunk-2 Theme B item) the
Genesis G4_T1 overlay stack had ZERO streaming-parser tests: the only
coverage was the pure ``_parse_gemma4_args`` corpus in
``test_g4_t1_dict_key_sentinel_strip.py``. The streaming path is exactly
where the PROD failures live (Gemma-4 + MTP K=3/K=4 packs several
delimiter events into one delta), so this module ports the regression
corpus from two OPEN upstream PRs against the SAME root issue #41967:

* PR #45068 (Raf38) — the MTP-split parallel-tool corpus: a single
  streaming delta carrying ``}<tool_call|><|tool_call>call:next{`` (end
  token + start token + the next call header in one chunk), plus the
  token-id start-gate this suite locks in for the v2 overlay.
* PR #44844 (ishaan-smallest) — the span-based rewrite corpus:
  coalesced complete calls, content sharing a delta with a complete
  call, stripped ``call:name{...}`` bodies (markers eaten upstream),
  partial-args finalization.

Overlays under test (both shipped files, imported as real modules with
the ``vllm`` import surface stubbed — see ``_load_overlay``):

* ``g4_t1_v2_gemma4_tool_parser_pr42237_overlay.py`` — CURRENT live
  bind-mount (accumulated-text rescan).
* ``g4_t1_v3_gemma4_tool_parser_pr44844_overlay.py`` — v3 PREP vendor
  (verbatim PR #44844 head + Genesis reset-guard hardening + the
  PR #44877 quoted-key hunk). NOT bind-mounted anywhere yet — the
  A/B vs v2 happens at the server stage.

Keep-alive state leak (the 31/35): the v2 empirical bench (2026-05-31,
gemma4-31B AWQ + TQ4bit_nc + MTP K=4, 7 cases x 5 runs) measured 35/35
with ``Connection: close`` but 31/35 over HTTP keep-alive. Diagnosis
(v2 overlay header): vLLM re-uses the parser instance across requests
on the same keep-alive socket, and the ``if not previous_text:`` reset
guard does NOT fire when the follow-up request's first parser
invocation arrives with non-empty ``previous_text`` — stale
``streamed_args_for_tool``/``prev_tool_call_arr``/``_sent_content_idx``
then corrupt the new request's diffs (observed as the missing closing
``"}`` on nested-object case 5). ``test_keep_alive_instance_reuse``
reproduces that exact shape: xfail(strict) on v2 (documented, not
fixed — the live workaround stays ``Connection: close``), PASS on v3
whose Genesis reset-guard hardening detects the new request by
``current_text`` no longer extending the previously seen text.

The token-id start-gate (PR #45068) is v2-only by design: v3/#44844
deliberately treats stripped ``call:name{`` TEXT as a tool call even
when the special tokens never appear, which is the opposite trade-off.

Stub fidelity: ``find_common_prefix`` and ``partial_tag_overlap`` are
logic-equivalent ports of the pin's ``vllm/tool_parsers/utils.py``
(read from /private/tmp/candidate_pin_current/vllm 2026-06-11); the
``ToolParser`` stub mirrors the pin's ``abstract_tool_parser.ToolParser``
attribute contract. ``_simulate_streaming`` is a faithful port of the
upstream test helper from ``tests/tool_parsers/test_gemma4_tool_parser
.py`` at the PR #45068 head (token id 48 = start, 49 = end, 0 = other).
"""
from __future__ import annotations

import json
import logging
import re as _stdlib_re
import sys
import types
import uuid
from importlib import util as importlib_util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
PATCH_DIR = REPO_ROOT / "sndr" / "engines" / "vllm" / "patches" / "tool_parsing"

OVERLAYS = {
    "v2_pr42237_current": PATCH_DIR
    / "g4_t1_v2_gemma4_tool_parser_pr42237_overlay.py",
    "v3_pr44844_prep": PATCH_DIR
    / "g4_t1_v3_gemma4_tool_parser_pr44844_overlay.py",
}

TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"
STRING_DELIM = '<|"|>'


# ---------------------------------------------------------------------------
# vllm import-surface stubs (dev venv has no vllm; the overlays import it
# at module top). Protocol stand-ins follow the pin's attribute contract.
# ---------------------------------------------------------------------------


class _DeltaFunctionCall:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments

    def model_dump(self, exclude_none: bool = False) -> dict:
        data = {"name": self.name, "arguments": self.arguments}
        if exclude_none:
            data = {k: v for k, v in data.items() if v is not None}
        return data


class _DeltaToolCall:
    def __init__(self, index=None, type=None, id=None, function=None):  # noqa: A002
        self.index = index
        self.type = type
        self.id = id
        self.function = function


class _DeltaMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = list(tool_calls) if tool_calls is not None else []


class _FunctionCall:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, type=None, function=None, id=None):  # noqa: A002
        self.type = type
        self.function = function
        self.id = id


class _ExtractedToolCallInformation:
    def __init__(self, tools_called=False, tool_calls=None, content=None):
        self.tools_called = tools_called
        self.tool_calls = tool_calls if tool_calls is not None else []
        self.content = content


class _ChatCompletionRequest:
    pass


class _ResponsesRequest:
    pass


class _TokenizerLike:
    pass


class _Tool:
    pass


class _ToolParser:
    """Mirror of the pin's abstract ToolParser attribute contract."""

    def __init__(self, tokenizer, tools=None):
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []
        self.model_tokenizer = tokenizer
        self.tools = list(tools) if tools else []

    @property
    def vocab(self) -> dict:
        return self.model_tokenizer.get_vocab()

    def adjust_request(self, request):
        return request


def _find_common_prefix(s1: str, s2: str) -> str:
    """Logic-equivalent port of the pin's tool_parsers.utils helper."""
    prefix = ""
    for a, b in zip(s1, s2):
        if a != b:
            break
        prefix += a
    return prefix


def _partial_tag_overlap(text: str, tag: str) -> int:
    """Logic-equivalent port of the pin's tool_parsers.utils helper."""
    max_check = min(len(tag) - 1, len(text))
    for k in range(max_check, 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


def _make_tool_call_id() -> str:
    return f"chatcmpl-tool-{uuid.uuid4().hex}"


def _build_stub_modules() -> dict[str, types.ModuleType]:
    """Build the vllm module tree the overlay files import from."""

    def mod(name: str) -> types.ModuleType:
        return types.ModuleType(name)

    stubs: dict[str, types.ModuleType] = {}
    for name in (
        "vllm",
        "vllm.entrypoints",
        "vllm.entrypoints.chat_utils",
        "vllm.entrypoints.openai",
        "vllm.entrypoints.openai.chat_completion",
        "vllm.entrypoints.openai.chat_completion.protocol",
        "vllm.entrypoints.openai.engine",
        "vllm.entrypoints.openai.engine.protocol",
        "vllm.entrypoints.openai.responses",
        "vllm.entrypoints.openai.responses.protocol",
        "vllm.logger",
        "vllm.tokenizers",
        "vllm.tool_parsers",
        "vllm.tool_parsers.abstract_tool_parser",
        "vllm.tool_parsers.utils",
    ):
        stubs[name] = mod(name)

    stubs["vllm.entrypoints.chat_utils"].make_tool_call_id = _make_tool_call_id
    stubs[
        "vllm.entrypoints.openai.chat_completion.protocol"
    ].ChatCompletionRequest = _ChatCompletionRequest
    proto = stubs["vllm.entrypoints.openai.engine.protocol"]
    proto.DeltaFunctionCall = _DeltaFunctionCall
    proto.DeltaMessage = _DeltaMessage
    proto.DeltaToolCall = _DeltaToolCall
    proto.ExtractedToolCallInformation = _ExtractedToolCallInformation
    proto.FunctionCall = _FunctionCall
    proto.ToolCall = _ToolCall
    stubs[
        "vllm.entrypoints.openai.responses.protocol"
    ].ResponsesRequest = _ResponsesRequest
    stubs["vllm.logger"].init_logger = logging.getLogger
    stubs["vllm.tokenizers"].TokenizerLike = _TokenizerLike
    abstract = stubs["vllm.tool_parsers.abstract_tool_parser"]
    abstract.Tool = _Tool
    abstract.ToolParser = _ToolParser
    utils = stubs["vllm.tool_parsers.utils"]
    utils.find_common_prefix = _find_common_prefix
    utils.partial_tag_overlap = _partial_tag_overlap

    # The v1/v3 overlays import the third-party `regex` package
    # (engine-side dependency); the stdlib module is API-compatible
    # for the parser's patterns.
    stubs["regex"] = _stdlib_re
    return stubs


_LOADED: dict[str, types.ModuleType] = {}


def _load_overlay(key: str) -> types.ModuleType:
    """Import a shipped overlay file with the vllm surface stubbed."""
    if key in _LOADED:
        return _LOADED[key]

    path = OVERLAYS[key]
    assert path.is_file(), f"overlay file missing: {path}"

    stubs = _build_stub_modules()
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib_util.spec_from_file_location(
            f"g4_t1_overlay_under_test_{key}", path
        )
        module = importlib_util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        for name, prior in saved.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior

    _LOADED[key] = module
    return module


class _FakeTokenizer:
    def get_vocab(self) -> dict:
        return {TOOL_CALL_START: 48, TOOL_CALL_END: 49, STRING_DELIM: 52}


def _make_parser(key: str):
    module = _load_overlay(key)
    return module.Gemma4ToolParser(_FakeTokenizer())


# ---------------------------------------------------------------------------
# Streaming simulation — faithful port of the upstream test helper at the
# PR #45068 head (token id 48 = start, 49 = end, 0 = other; multi-token
# MTP deltas are modeled as multi-event chunks).
# ---------------------------------------------------------------------------


def _chunk_token_ids(chunk: str, *, suppress_special_ids: bool) -> list[int]:
    if suppress_special_ids:
        return [0]
    if TOOL_CALL_START in chunk:
        return [48]
    if TOOL_CALL_END in chunk:
        return [49]
    return [0]


def _simulate_streaming(
    parser,
    chunks: list[str],
    *,
    suppress_special_ids: bool = False,
    initial_previous_text: str = "",
):
    """Feed chunks through the streaming parser and collect results.

    ``initial_previous_text`` reproduces the keep-alive instance-reuse
    shape: the first parser invocation of a request arriving with
    ``previous_text`` already non-empty, so the upstream
    ``if not previous_text:`` reset guard never fires.
    """
    results = []
    previous_text = initial_previous_text
    previous_token_ids: list[int] = (
        _chunk_token_ids(
            initial_previous_text, suppress_special_ids=suppress_special_ids
        )
        if initial_previous_text
        else []
    )

    for chunk in chunks:
        current_text = previous_text + chunk
        delta_token_ids = _chunk_token_ids(
            chunk, suppress_special_ids=suppress_special_ids
        )
        current_token_ids = previous_token_ids + delta_token_ids

        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=tuple(previous_token_ids),
            current_token_ids=tuple(current_token_ids),
            delta_token_ids=tuple(delta_token_ids),
            request=object(),
        )
        results.append((delta, current_text))
        previous_text = current_text
        previous_token_ids = list(current_token_ids)

    return results


def _collect_content(results) -> str:
    return "".join(
        delta.content
        for delta, _ in results
        if delta is not None and delta.content
    )


def _function_field(function, key: str):
    if function is None:
        return None
    if isinstance(function, dict):
        return function.get(key)
    return getattr(function, key, None)


def _collect_tool_calls(results) -> dict:
    """Collect streamed tool calls by index (PR #44844 test helper)."""
    calls: dict[int, dict] = {}
    for delta, _ in results:
        if not delta or not delta.tool_calls:
            continue
        for tc in delta.tool_calls:
            entry = calls.setdefault(tc.index, {"name": None, "arguments": ""})
            name = _function_field(tc.function, "name")
            arguments = _function_field(tc.function, "arguments") or ""
            if name:
                entry["name"] = name
            if arguments:
                entry["arguments"] += arguments
    return calls


def _assert_tool_call(calls, index, name, arguments) -> None:
    assert index in calls, f"tool call index {index} never streamed: {calls}"
    assert calls[index]["name"] == name, calls
    assert calls[index]["arguments"], f"no arguments streamed: {calls}"
    assert json.loads(calls[index]["arguments"]) == arguments, calls


# ---------------------------------------------------------------------------
# Shared corpus — both overlay generations must pass every case
# ---------------------------------------------------------------------------


@pytest.fixture(params=sorted(OVERLAYS), ids=sorted(OVERLAYS))
def parser(request):
    return _make_parser(request.param)


class TestStreamingCorpus:
    def test_basic_streaming_single_tool(self, parser):
        """PR #45068 base case: token-by-token single tool call."""
        chunks = [
            "<|tool_call>",
            "call:get_weather{",
            'location:<|"|>Paris',
            ", France",
            '<|"|>',
            "}",
            "<tool_call|>",
        ]

        results = _simulate_streaming(parser, chunks)
        calls = _collect_tool_calls(results)

        _assert_tool_call(
            calls, 0, "get_weather", {"location": "Paris, France"}
        )

    def test_basic_streaming_parallel_tools_mtp_split(self, parser):
        """PR #45068 headline case: the MTP-split delta.

        MTP K=3 packs the end token of call 0, the start token of call 1,
        and the next call header into ONE streaming delta
        (``}<tool_call|><|tool_call>call:get_time{``). Chunk shapes are
        verbatim from the upstream regression test.
        """
        chunks = [
            "<|tool_call>",
            "call:get_weather{",
            'location:<|"|>Paris',
            '<|"|>',
            "}<tool_call|><|tool_call>call:get_time{",
            'location:<|"|>France',
            '<|"|>',
            "}<tool_call|>",
        ]

        results = _simulate_streaming(parser, chunks)
        calls = _collect_tool_calls(results)

        _assert_tool_call(calls, 0, "get_weather", {"location": "Paris"})
        _assert_tool_call(calls, 1, "get_time", {"location": "France"})

    def test_streaming_complete_tool_call_in_single_delta(self, parser):
        """PR #44844: a full tool call can arrive in one MTP step."""
        chunks = [
            '<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}'
            "<tool_call|>"
        ]

        results = _simulate_streaming(parser, chunks)
        calls = _collect_tool_calls(results)

        _assert_tool_call(calls, 0, "get_weather", {"location": "Paris"})

    def test_streaming_content_and_complete_call_in_single_delta(
        self, parser
    ):
        """PR #44844: content before a coalesced call is still emitted."""
        content = "I will record this.\n"
        chunks = [
            f'{content}<|tool_call>call:get_weather{{location:<|"|>Paris'
            '<|"|>}<tool_call|>'
        ]

        results = _simulate_streaming(parser, chunks)
        calls = _collect_tool_calls(results)

        assert _collect_content(results) == content
        _assert_tool_call(calls, 0, "get_weather", {"location": "Paris"})

    def test_streaming_coalesced_multiple_tool_calls(self, parser):
        """PR #44844: back-to-back complete calls in one delta."""
        chunks = [
            '<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}'
            "<tool_call|>"
            '<|tool_call>call:get_time{location:<|"|>Paris<|"|>}'
            "<tool_call|>"
        ]

        results = _simulate_streaming(parser, chunks)
        calls = _collect_tool_calls(results)

        _assert_tool_call(calls, 0, "get_weather", {"location": "Paris"})
        _assert_tool_call(calls, 1, "get_time", {"location": "Paris"})

    def test_streaming_coalesced_complete_and_open_call(self, parser):
        """PR #44844: one delta finishes a call and opens the next."""
        chunks = [
            '<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}'
            "<tool_call|>"
            "<|tool_call>call:get_time{",
            'location:<|"|>Paris<|"|>}',
            "<tool_call|>",
        ]

        results = _simulate_streaming(parser, chunks)
        calls = _collect_tool_calls(results)

        _assert_tool_call(calls, 0, "get_weather", {"location": "Paris"})
        _assert_tool_call(calls, 1, "get_time", {"location": "Paris"})

    def test_streaming_content_between_two_tool_calls(self, parser):
        """PR #44844: content between two complete calls is emitted."""
        middle = "Some text"
        chunks = [
            '<|tool_call>call:a{x:<|"|>1<|"|>}<tool_call|>'
            f"{middle}"
            '<|tool_call>call:b{y:<|"|>2<|"|>}<tool_call|>'
        ]

        results = _simulate_streaming(parser, chunks)
        calls = _collect_tool_calls(results)

        assert _collect_content(results) == middle
        _assert_tool_call(calls, 0, "a", {"x": "1"})
        _assert_tool_call(calls, 1, "b", {"y": "2"})

    def test_streaming_nested_object_args(self, parser):
        """PR #44844: nested-object args — the keep-alive case-5 shape."""
        chunks = [
            "<|tool_call>",
            "call:fn{",
            'nested:{inner:<|"|>val<|"|>}',
            "}",
            "<tool_call|>",
        ]

        results = _simulate_streaming(parser, chunks)
        calls = _collect_tool_calls(results)

        _assert_tool_call(calls, 0, "fn", {"nested": {"inner": "val"}})

    def test_streaming_partial_args_then_finalize(self, parser):
        """PR #44844: partial argument diffs flushed by finalization."""
        chunks = [
            "<|tool_call>",
            "call:fn{",
            'a:<|"|>hello',
            ' world<|"|>}',
            "<tool_call|>",
        ]

        results = _simulate_streaming(parser, chunks)
        calls = _collect_tool_calls(results)

        _assert_tool_call(calls, 0, "fn", {"a": "hello world"})


# ---------------------------------------------------------------------------
# v2-only — token-id start-gate (borrowed from PR #45068)
# ---------------------------------------------------------------------------


class TestV2TokenIdStartGate:
    """The v2 overlay gates the streaming fast path on the START token ID
    (``tool_call_start_token_id in current_token_ids``), not on the start
    token TEXT. Token ids are the detokenizer's ground truth: the special
    token always arrives atomically with its id, while lookalike TEXT can
    be produced by ordinary tokens and must stay content.

    v3 (#44844) deliberately makes the opposite trade-off (it recovers
    stripped ``call:name{`` bodies with no markers at all), so this gate
    is v2-only.
    """

    def test_lookalike_text_without_start_id_stays_content(self):
        """Literal marker text with no start token id is plain content."""
        parser = _make_parser("v2_pr42237_current")
        chunks = [
            "The literal marker <|tool_call> is plain text here",
            " and stays on the content channel.",
        ]

        results = _simulate_streaming(
            parser, chunks, suppress_special_ids=True
        )
        calls = _collect_tool_calls(results)

        assert calls == {}, f"phantom tool call from lookalike text: {calls}"
        assert _collect_content(results) == "".join(chunks)

    def test_gate_content_then_real_tool_call_no_reemission(self):
        """Content emitted through the gate is not re-emitted after the
        real start token arrives (the gate must advance the parser's
        sent-content index)."""
        parser = _make_parser("v2_pr42237_current")
        intro = "Intro text. "
        chunks = [
            intro,
            '<|tool_call>call:fn{x:<|"|>v<|"|>}<tool_call|>',
        ]

        results = _simulate_streaming(parser, chunks)
        calls = _collect_tool_calls(results)

        assert _collect_content(results) == intro
        _assert_tool_call(calls, 0, "fn", {"x": "v"})


# ---------------------------------------------------------------------------
# v3-only — stripped-marker recovery (PR #44844 design goal)
# ---------------------------------------------------------------------------


class TestV3StrippedMarkerRecovery:
    def test_streaming_stripped_tool_call_body(self):
        """Stripped Gemma4 grammar must not leak as content (v3)."""
        parser = _make_parser("v3_pr44844_prep")
        chunks = [
            'call:record_status{item:<|"|>sprocket<|"|>,'
            'status:<|"|>empty<|"|>}'
        ]

        results = _simulate_streaming(parser, chunks)
        calls = _collect_tool_calls(results)

        assert _collect_content(results) == ""
        _assert_tool_call(
            calls, 0, "record_status", {"item": "sprocket", "status": "empty"}
        )

    def test_streaming_stripped_tool_call_after_content(self):
        """A stripped call after content must not leak as content (v3)."""
        parser = _make_parser("v3_pr44844_prep")
        content = "I will record this.\n"
        chunks = [
            content,
            'call:record_status{item:<|"|>sprocket<|"|>,'
            'status:<|"|>empty<|"|>}',
        ]

        results = _simulate_streaming(parser, chunks)
        calls = _collect_tool_calls(results)

        assert _collect_content(results) == content
        _assert_tool_call(
            calls, 0, "record_status", {"item": "sprocket", "status": "empty"}
        )


# ---------------------------------------------------------------------------
# Keep-alive instance-reuse state leak (the v2 31/35) — xfail-then-document
# ---------------------------------------------------------------------------

_KEEP_ALIVE_PARAMS = [
    pytest.param(
        "v2_pr42237_current",
        marks=pytest.mark.xfail(
            strict=True,
            reason=(
                "DOCUMENTED v2 keep-alive state leak (31/35, bench "
                "2026-05-31): the `if not previous_text:` reset guard "
                "does not fire when the follow-up request's first parser "
                "invocation arrives with non-empty previous_text on a "
                "re-used instance; stale streamed_args_for_tool/"
                "prev_tool_call_arr/_sent_content_idx corrupt the new "
                "request (missing name delta + clipped argument JSON). "
                "NOT fixed in v2 by design — live workaround stays "
                "`Connection: close`; the fix ships as the v3 overlay's "
                "Genesis reset-guard hardening (A/B at the server stage). "
                "If this test ever XPASSes, v2 grew a reset guard — "
                "re-run the 7x5 keep-alive bench and update the overlay "
                "header."
            ),
        ),
    ),
    pytest.param("v3_pr44844_prep"),
]


@pytest.mark.parametrize("overlay_key", _KEEP_ALIVE_PARAMS)
def test_keep_alive_instance_reuse_second_request_clean(overlay_key):
    """Second request on a re-used parser instance must parse clean.

    Request A (nested-object args, the case-5 shape) completes normally.
    Request B then lands on the SAME parser instance with its first
    parser invocation already carrying non-empty ``previous_text`` —
    the diagnosed keep-alive shape under which the upstream
    ``if not previous_text:`` guard never fires.
    """
    parser = _make_parser(overlay_key)

    chunks_a = [
        "<|tool_call>",
        "call:store{",
        'nested:{inner:<|"|>val<|"|>}',
        "}",
        "<tool_call|>",
    ]
    results_a = _simulate_streaming(parser, chunks_a)
    calls_a = _collect_tool_calls(results_a)
    _assert_tool_call(calls_a, 0, "store", {"nested": {"inner": "val"}})

    chunks_b = [
        "call:get_weather{",
        'location:<|"|>Paris<|"|>',
        "}",
        "<tool_call|>",
    ]
    results_b = _simulate_streaming(
        parser, chunks_b, initial_previous_text="<|tool_call>"
    )
    calls_b = _collect_tool_calls(results_b)

    _assert_tool_call(calls_b, 0, "get_weather", {"location": "Paris"})
