# SPDX-License-Identifier: Apache-2.0
"""Data model + tuned constants for the memory engine.

These dataclasses mirror the `mem_node` / `mem_edge` schema in
docs/design/memory-engine-production-design.md. They are backend-agnostic:
the in-memory backend stores them directly; the Postgres backend maps them to
rows. Keeping the brain-mechanic constants here (one source of truth) means the
in-memory reference and the SQL backend stay numerically identical.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── Hebbian co-access (HeLa-Mem):  w <- (1 - lambda) * w + eta * [co-accessed]
# Verified tuning from the survey literature; see design doc section 2.
HEBBIAN_ETA: float = 0.02     # learning rate per co-access
HEBBIAN_LAMBDA: float = 0.995  # retention (1 - decay) per co-access step

# ── Ebbinghaus retention:  R = exp(-age / (S * importance_boost))
# S is the base memory-strength time-constant in seconds; importance scales it.
EBBINGHAUS_S: float = 86_400.0  # 1 day base half-life scale

# ── Spreading activation along the graph during expand (design section 3).
SPREAD_DAMPING: float = 0.5   # beta: score multiplier per hop
MAX_EXPAND_DEPTH: int = 3     # bounded traversal (cycle-safe)

# Semantic auto-edge threshold (kNN cosine) — used by the batch linker.
SEMANTIC_EDGE_TAU: float = 0.8

CO_ACCESS_REL: str = "co_access"


@dataclass
class MemoryNode:
    """One memory atom (note / fact / entity / summary)."""

    id: int
    owner_id: int
    kind: str
    content: str
    embedding: list[float] = field(default_factory=list)
    importance: float = 0.0       # Generative-Agents importance (LLM-rated, batch)
    strength: float = 1.0         # Ebbinghaus retention base
    access_count: int = 0
    community_id: int | None = None  # Leiden cluster ("cloud"); set by batch
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0       # epoch seconds
    accessed_at: float = 0.0      # epoch seconds (updated on retrieval)


@dataclass
class MemoryEdge:
    """A relationship between two nodes. `weight` carries the Hebbian strength."""

    src_id: int
    dst_id: int
    rel: str
    weight: float = 0.0
    properties: dict[str, Any] = field(default_factory=dict)
    valid_at: float = 0.0
    invalid_at: float | None = None  # bi-temporal: invalidated, not deleted


@dataclass
class SearchHit:
    """A node returned from a similarity / activation query, with its score."""

    node: MemoryNode
    score: float

    @property
    def id(self) -> int:
        return self.node.id
