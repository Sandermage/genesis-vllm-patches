# SPDX-License-Identifier: Apache-2.0
"""Persistent neural-graph memory engine.

A brain-like, leak-bounded knowledge store: memory atoms (`MemoryNode`) wired
by automatically-formed edges (`MemoryEdge`) that cluster into communities and
are recalled by vector similarity + bounded graph spreading-activation.

The public surface is the storage interface `MemoryStore` (`store.py`). Two
backends implement it: `InMemoryStore` (reference + unit-test double, pure
stdlib) and — behind the `integration` marker — a Postgres + pgvector backend.

Design: docs/design/memory-engine-production-design.md
"""
from __future__ import annotations

from sndr.memory.client import MemoryHTTPClient
from sndr.memory.embedder import Embedder, HashEmbedder, Model2VecEmbedder
from sndr.memory.engine import MemoryEngine
from sndr.memory.inmemory import InMemoryStore
from sndr.memory.middleware import ConversationMemory
from sndr.memory.model import (
    HEBBIAN_ETA,
    HEBBIAN_LAMBDA,
    MemoryEdge,
    MemoryNode,
    SearchHit,
)
from sndr.memory.obsidian import import_vault
from sndr.memory.store import MemoryStore

__all__ = [
    "HEBBIAN_ETA",
    "HEBBIAN_LAMBDA",
    "ConversationMemory",
    "Embedder",
    "HashEmbedder",
    "InMemoryStore",
    "MemoryHTTPClient",
    "Model2VecEmbedder",
    "MemoryEdge",
    "MemoryEngine",
    "MemoryNode",
    "MemoryStore",
    "SearchHit",
    "import_vault",
]
