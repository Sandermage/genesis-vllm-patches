# SPDX-License-Identifier: Apache-2.0
"""TDD contract for MemoryGateway — the OpenAI-compatible memory middleware.

It wraps ANY upstream (CLIProxyAPI for external models, vLLM for internal) with
augment -> forward -> capture, so every model gains memory without being
modified. The HTTP/streaming shell is layered on top; this core is provider- and
transport-agnostic (the upstream call is an injected `forward`), so it is fully
unit-tested here.
"""
from __future__ import annotations

from sndr.memory.embedder import HashEmbedder
from sndr.memory.engine import MemoryEngine
from sndr.memory.gateway import MemoryGateway, extract_assistant_text
from sndr.memory.inmemory import InMemoryStore
from sndr.memory.middleware import ConversationMemory


def _gateway_with_capture():
    eng = MemoryEngine(store=InMemoryStore(), embedder=HashEmbedder(dim=64))
    cm = ConversationMemory(engine=eng)
    captured: dict = {}

    def forward(body):
        captured["body"] = body
        return {"choices": [{"message": {"role": "assistant", "content": "It is alpha-7."}}]}

    return MemoryGateway(memory=cm, forward=forward), eng, captured


class TestGatewayHandle:
    def test_augments_forwards_and_captures(self):
        gw, eng, captured = _gateway_with_capture()
        eng.remember(owner_id=1, text="the secret code is alpha-7")
        req = {"model": "gpt", "messages": [{"role": "user", "content": "what is the secret code"}]}
        resp = gw.handle(owner_id=1, request_body=req)

        # upstream received memory injected as a leading system message
        upstream_msgs = captured["body"]["messages"]
        assert upstream_msgs[0]["role"] == "system"
        assert "alpha-7" in upstream_msgs[0]["content"]
        # other request fields pass through untouched
        assert captured["body"]["model"] == "gpt"
        # caller's request object is not mutated
        assert req["messages"][0]["content"] == "what is the secret code"
        assert len(req["messages"]) == 1
        # response passes through
        assert resp["choices"][0]["message"]["content"] == "It is alpha-7."
        # the assistant reply was captured into memory
        hits = eng.recall(owner_id=1, query="alpha-7 secret", limit=10, reinforce=False)
        assert any("alpha-7" in h.node.content for h in hits)

    def test_no_memory_no_injection_but_still_forwards(self):
        gw, _eng, captured = _gateway_with_capture()
        req = {"model": "gpt", "messages": [{"role": "user", "content": "hello there"}]}
        resp = gw.handle(owner_id=1, request_body=req)
        assert captured["body"]["messages"] == req["messages"]  # nothing to inject
        assert resp["choices"][0]["message"]["content"] == "It is alpha-7."


class TestExtractAssistant:
    def test_openai_shape(self):
        resp = {"choices": [{"message": {"content": "hi"}}]}
        assert extract_assistant_text(resp) == "hi"

    def test_missing_is_empty(self):
        assert extract_assistant_text({}) == ""
        assert extract_assistant_text({"choices": []}) == ""
