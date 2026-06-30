# SPDX-License-Identifier: Apache-2.0
"""Tests for the gateway OpenAI-response helpers.

The augment -> forward -> capture orchestration is verified end-to-end via the
real route in `test_gateway_route.py`; here we cover the parsing helpers it uses
to capture the assistant reply (non-stream message + streamed SSE deltas).
"""
from __future__ import annotations

from sndr.memory.gateway import assistant_text_from_sse, extract_assistant_text


class TestExtractAssistant:
    def test_openai_shape(self):
        resp = {"choices": [{"message": {"content": "hi"}}]}
        assert extract_assistant_text(resp) == "hi"

    def test_missing_is_empty(self):
        assert extract_assistant_text({}) == ""
        assert extract_assistant_text({"choices": []}) == ""


class TestSseAccumulator:
    def test_reassembles_streamed_assistant(self):
        sse = (
            'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        assert assistant_text_from_sse(sse) == "Hello world"

    def test_tolerates_noise_and_empty(self):
        sse = "data: \n\n: comment\n\ndata: not-json\n\ndata: [DONE]\n\n"
        assert assistant_text_from_sse(sse) == ""
