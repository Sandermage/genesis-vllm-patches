# SPDX-License-Identifier: Apache-2.0
"""TDD contract for ConversationMemory — the universal capture/inject middleware.

This is the model-agnostic glue: before a chat request, recall the owner's
relevant memories and inject them as plain text (a system block) so ANY model
(Gemma, Qwen, external) benefits; after the exchange, capture the salient turn.
It lives in our core so the proxy adopts it without us touching that repo.

Duck-typed on the engine (recall + remember), so it is verified here against
MemoryEngine(InMemoryStore + HashEmbedder).
"""
from __future__ import annotations

from sndr.memory.embedder import HashEmbedder
from sndr.memory.engine import MemoryEngine
from sndr.memory.inmemory import InMemoryStore
from sndr.memory.middleware import ConversationMemory

OWNER = 1


def _mw() -> tuple[ConversationMemory, MemoryEngine]:
    eng = MemoryEngine(store=InMemoryStore(), embedder=HashEmbedder(dim=128))
    return ConversationMemory(engine=eng), eng


class TestAugment:
    def test_injects_relevant_memory_as_system_block(self):
        mw, eng = _mw()
        eng.remember(owner_id=OWNER, text="the deploy server is 192.168.1.10")
        msgs = [{"role": "user", "content": "what is the deploy server address"}]
        out = mw.augment(owner_id=OWNER, messages=msgs)
        assert out[0]["role"] == "system"
        assert "192.168.1.10" in out[0]["content"]
        # original user message is preserved, after the injected block
        assert out[-1] == msgs[0]

    def test_no_injection_when_memory_empty(self):
        mw, _ = _mw()
        msgs = [{"role": "user", "content": "anything"}]
        out = mw.augment(owner_id=OWNER, messages=msgs)
        assert out == msgs  # unchanged — nothing to inject

    def test_merges_into_existing_system_message(self):
        mw, eng = _mw()
        eng.remember(owner_id=OWNER, text="postgres lives on port 55432")
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "which postgres port do we use"},
        ]
        out = mw.augment(owner_id=OWNER, messages=msgs)
        # still exactly one system message, original instruction retained
        assert sum(1 for m in out if m["role"] == "system") == 1
        assert "You are helpful." in out[0]["content"]
        assert "55432" in out[0]["content"]

    def test_does_not_mutate_input_list(self):
        mw, eng = _mw()
        eng.remember(owner_id=OWNER, text="alpha beta gamma fact")
        msgs = [{"role": "user", "content": "alpha beta gamma"}]
        before = [dict(m) for m in msgs]
        mw.augment(owner_id=OWNER, messages=msgs)
        assert msgs == before  # caller's list untouched


class TestCapture:
    def test_capture_remembers_last_user_message(self):
        mw, eng = _mw()
        msgs = [{"role": "user", "content": "remember that the api owner header is X-Owner-Id"}]
        mw.capture(owner_id=OWNER, messages=msgs)
        hits = eng.recall(owner_id=OWNER, query="api owner header", limit=5, reinforce=False)
        assert any("X-Owner-Id" in h.node.content for h in hits)

    def test_capture_is_noop_without_user_message(self):
        mw, eng = _mw()
        mw.capture(owner_id=OWNER, messages=[{"role": "system", "content": "x"}])
        assert eng.store.count_nodes(owner_id=OWNER) == 0
