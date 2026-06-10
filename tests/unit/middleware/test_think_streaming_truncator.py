# SPDX-License-Identifier: Apache-2.0
"""Sprint 4: tests for PN16 V6 streaming `<think>` truncator.

Covers:

  • Inactive (budget=0 OR no tools) → strict passthrough
  • Active (budget>0 + has_tools) → token counting + truncation
  • Reasoning chunks dropped after budget
  • One-shot truncation note in delta.content
  • Tool-call chunks always pass through (even after truncation)
  • Plain content chunks always pass
  • Defensive: dict shape AND object shape (pydantic models)
  • Empty / heartbeat / role-only chunks pass
  • Stats counters increment correctly
  • filter_stream() async helper drops + injects correctly
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from sndr.engines.vllm.middleware import think_streaming_truncator as tst
from sndr.engines.vllm.middleware.think_streaming_truncator import (
    ThinkStreamingTruncator,
    filter_stream,
    get_stats,
    max_thinking_stream_tokens,
    reset_stats,
)


@pytest.fixture(autouse=True)
def _reset_stats():
    reset_stats()
    yield
    reset_stats()


# ─── Chunk builders ────────────────────────────────────────────────────


def reasoning_chunk(text: str = "x") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {"reasoning_content": text},
            "finish_reason": None,
        }],
    }


def content_chunk(text: str = "answer") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {"content": text},
            "finish_reason": None,
        }],
    }


def tool_call_chunk() -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": ""},
                }],
            },
            "finish_reason": None,
        }],
    }


def role_chunk() -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None,
        }],
    }


def finish_chunk(reason: str = "tool_calls") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": reason,
        }],
    }


# ─── Inactive: budget=0 OR no tools ────────────────────────────────────


class TestInactivePassthrough:
    def test_zero_budget_passes_everything(self):
        t = ThinkStreamingTruncator(budget_tokens=0, has_tools=True)
        chunks = [reasoning_chunk("a"), reasoning_chunk("b"), tool_call_chunk()]
        out = [t.filter_chunk(c) for c in chunks]
        assert out == chunks
        assert t.truncated is False

    def test_no_tools_passes_everything(self):
        t = ThinkStreamingTruncator(budget_tokens=10, has_tools=False)
        chunks = [reasoning_chunk("a"), reasoning_chunk("b")] * 100
        for c in chunks:
            assert t.filter_chunk(c) is c  # identity, no copy
        assert t.truncated is False

    def test_inactive_increments_passthrough_stat(self):
        ThinkStreamingTruncator(budget_tokens=0, has_tools=True)
        ThinkStreamingTruncator(budget_tokens=10, has_tools=False)
        s = get_stats()
        assert s["passthrough_only"] == 2


# ─── Active: budget+tools ──────────────────────────────────────────────


class TestActiveTruncation:
    def test_under_budget_passes_through(self):
        t = ThinkStreamingTruncator(budget_tokens=3, has_tools=True)
        for _ in range(3):
            c = reasoning_chunk("tok")
            assert t.filter_chunk(c) is c
        assert t.truncated is False
        assert t.reasoning_token_count == 3

    def test_first_over_budget_emits_note_chunk(self):
        t = ThinkStreamingTruncator(budget_tokens=2, has_tools=True)
        # First two reasoning chunks pass through
        t.filter_chunk(reasoning_chunk("a"))
        t.filter_chunk(reasoning_chunk("b"))
        assert t.truncated is False

        # Third triggers truncation + note injection
        third = reasoning_chunk("c")
        out = t.filter_chunk(third)
        assert t.truncated is True
        assert t.note_injected is True
        # Output is a NEW chunk with content note (not original reasoning)
        assert out is not third
        delta = out["choices"][0]["delta"]
        assert "content" in delta
        assert "Genesis" in delta["content"]
        # reasoning_content NOT in note chunk
        assert "reasoning_content" not in delta

    def test_subsequent_reasoning_chunks_dropped_silently(self):
        t = ThinkStreamingTruncator(budget_tokens=1, has_tools=True)
        t.filter_chunk(reasoning_chunk("a"))  # under budget
        t.filter_chunk(reasoning_chunk("b"))  # triggers truncation + note
        # Subsequent reasoning chunks: dropped (return None)
        for _ in range(50):
            assert t.filter_chunk(reasoning_chunk("x")) is None

    def test_tool_call_chunks_pass_after_truncation(self):
        """Critical contract — once truncation fires, tool_call chunks
        STILL pass through. That's the whole point of V6."""
        t = ThinkStreamingTruncator(budget_tokens=1, has_tools=True)
        t.filter_chunk(reasoning_chunk("a"))
        t.filter_chunk(reasoning_chunk("b"))  # truncation
        # Tool call after truncation
        tc = tool_call_chunk()
        out = t.filter_chunk(tc)
        assert out is tc

    def test_content_chunks_pass_after_truncation(self):
        t = ThinkStreamingTruncator(budget_tokens=1, has_tools=True)
        t.filter_chunk(reasoning_chunk("a"))
        t.filter_chunk(reasoning_chunk("b"))  # truncation
        c = content_chunk("answer")
        assert t.filter_chunk(c) is c

    def test_finish_reason_chunk_passes(self):
        t = ThinkStreamingTruncator(budget_tokens=2, has_tools=True)
        for _ in range(5):
            t.filter_chunk(reasoning_chunk("x"))  # over budget
        finish = finish_chunk("tool_calls")
        assert t.filter_chunk(finish) is finish


# ─── Counters ──────────────────────────────────────────────────────────


class TestCounters:
    def test_reasoning_seen_increments(self):
        t = ThinkStreamingTruncator(budget_tokens=10, has_tools=True)
        for _ in range(3):
            t.filter_chunk(reasoning_chunk("x"))
        assert t.counters["reasoning_chunks_seen"] == 3
        assert t.counters["reasoning_chunks_dropped"] == 0

    def test_dropped_increments_only_after_truncation(self):
        t = ThinkStreamingTruncator(budget_tokens=2, has_tools=True)
        for _ in range(5):
            t.filter_chunk(reasoning_chunk("x"))
        # 2 passed, 1 became note, 2 dropped silently
        assert t.counters["reasoning_chunks_seen"] == 5
        assert t.counters["reasoning_chunks_dropped"] == 3  # 1 note + 2 silent

    def test_module_level_truncations_fired(self):
        t = ThinkStreamingTruncator(budget_tokens=1, has_tools=True)
        t.filter_chunk(reasoning_chunk())
        t.filter_chunk(reasoning_chunk())  # truncates
        # Truncate again with another instance — should also count
        t2 = ThinkStreamingTruncator(budget_tokens=1, has_tools=True)
        t2.filter_chunk(reasoning_chunk())
        t2.filter_chunk(reasoning_chunk())
        s = get_stats()
        assert s["truncations_fired"] == 2


# ─── Defensive: object-shaped chunks ───────────────────────────────────


class _FakeDelta:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _FakeChoice:
    def __init__(self, delta, index=0, finish_reason=None):
        self.delta = delta
        self.index = index
        self.finish_reason = finish_reason


class _FakeChunk:
    def __init__(self, choices, id="chatcmpl-fake"):
        self.id = id
        self.choices = choices


class TestObjectShape:
    def test_object_chunk_passthrough_inactive(self):
        t = ThinkStreamingTruncator(budget_tokens=0, has_tools=False)
        chunk = _FakeChunk(choices=[
            _FakeChoice(_FakeDelta(reasoning_content="x")),
        ])
        out = t.filter_chunk(chunk)
        assert out is chunk

    def test_object_chunk_truncated_returns_dict(self):
        """When truncating, we degrade to dict shape (don't pydantic-mutate)."""
        t = ThinkStreamingTruncator(budget_tokens=1, has_tools=True)
        # First reasoning passes through
        chunk1 = _FakeChunk(choices=[
            _FakeChoice(_FakeDelta(reasoning_content="a")),
        ])
        out1 = t.filter_chunk(chunk1)
        assert out1 is chunk1
        # Second triggers truncation → returns dict with note
        chunk2 = _FakeChunk(choices=[
            _FakeChoice(_FakeDelta(reasoning_content="b")),
        ])
        out2 = t.filter_chunk(chunk2)
        assert isinstance(out2, dict)
        assert "Genesis" in out2["choices"][0]["delta"]["content"]


# ─── Edge cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_chunk_passes(self):
        t = ThinkStreamingTruncator(budget_tokens=10, has_tools=True)
        empty = {"id": "x", "choices": []}
        assert t.filter_chunk(empty) is empty

    def test_no_choices_field(self):
        t = ThinkStreamingTruncator(budget_tokens=10, has_tools=True)
        assert t.filter_chunk({"id": "x"}) == {"id": "x"}

    def test_role_only_chunk_passes(self):
        t = ThinkStreamingTruncator(budget_tokens=10, has_tools=True)
        r = role_chunk()
        assert t.filter_chunk(r) is r

    def test_truncation_note_env_override(self, monkeypatch):
        monkeypatch.setenv(
            "GENESIS_PN16_TRUNCATION_NOTE", "[Custom] thinking cut off",
        )
        t = ThinkStreamingTruncator(budget_tokens=1, has_tools=True)
        t.filter_chunk(reasoning_chunk())
        out = t.filter_chunk(reasoning_chunk())  # triggers note
        assert "[Custom]" in out["choices"][0]["delta"]["content"]


# ─── max_thinking_stream_tokens helper ─────────────────────────────────


class TestEnvHelper:
    def test_default_zero(self, monkeypatch):
        monkeypatch.delenv("GENESIS_PN16_MAX_THINKING_STREAM_TOKENS", raising=False)
        assert max_thinking_stream_tokens() == 0

    def test_env_int(self, monkeypatch):
        monkeypatch.setenv("GENESIS_PN16_MAX_THINKING_STREAM_TOKENS", "150")
        assert max_thinking_stream_tokens() == 150

    def test_invalid_falls_back_to_zero(self, monkeypatch):
        monkeypatch.setenv("GENESIS_PN16_MAX_THINKING_STREAM_TOKENS", "garbage")
        assert max_thinking_stream_tokens() == 0

    def test_negative_clamped(self, monkeypatch):
        monkeypatch.setenv("GENESIS_PN16_MAX_THINKING_STREAM_TOKENS", "-5")
        assert max_thinking_stream_tokens() == 0


# ─── filter_stream async helper ────────────────────────────────────────


async def _async_iter(items):
    for it in items:
        yield it


async def _collect(source, **kw):
    out = []
    async for chunk in filter_stream(source, **kw):
        out.append(chunk)
    return out


class TestFilterStream:
    """Async filter_stream tests — driven via asyncio.run() to avoid
    pytest-asyncio dependency."""

    def test_passthrough_when_inactive(self):
        chunks = [reasoning_chunk("a"), tool_call_chunk(), finish_chunk()]
        result = asyncio.run(_collect(
            _async_iter(chunks), budget_tokens=0, has_tools=True,
        ))
        assert result == chunks

    def test_truncates_after_budget(self):
        chunks = [
            reasoning_chunk("a"), reasoning_chunk("b"), reasoning_chunk("c"),
            tool_call_chunk(), finish_chunk("tool_calls"),
        ]
        result = asyncio.run(_collect(
            _async_iter(chunks), budget_tokens=2, has_tools=True,
        ))
        assert len(result) == 5
        notes = [c for c in result
                 if "content" in c["choices"][0]["delta"]
                 and "Genesis" in c["choices"][0]["delta"].get("content", "")]
        assert len(notes) == 1
        tool = [c for c in result
                if c["choices"][0]["delta"].get("tool_calls")]
        assert len(tool) == 1


class TestLondonThinkScenario:
    def test_graphomania_then_tool_call(self):
        """Simulates the london_think failure: 200 reasoning tokens
        then tool_call. Budget=50 → truncator drops chunks 51-200,
        surfacing tool_call to client promptly."""
        chunks = (
            [reasoning_chunk(f"r{i}") for i in range(200)]
            + [tool_call_chunk(), finish_chunk("tool_calls")]
        )
        result = asyncio.run(_collect(
            _async_iter(chunks), budget_tokens=50, has_tools=True,
        ))
        # 50 reasoning + 1 note + 1 tool_call + 1 finish = 53
        assert len(result) == 53
        for i in range(50):
            assert result[i]["choices"][0]["delta"].get("reasoning_content")
        assert "Genesis" in result[50]["choices"][0]["delta"].get("content", "")
        assert result[51]["choices"][0]["delta"].get("tool_calls")
        assert result[52]["choices"][0]["finish_reason"] == "tool_calls"
