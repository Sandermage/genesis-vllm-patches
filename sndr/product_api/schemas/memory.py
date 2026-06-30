# SPDX-License-Identifier: Apache-2.0
"""Pydantic request/response schemas for the /api/v1/memory routes."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RememberIn(BaseModel):
    text: str = Field(min_length=1)
    kind: str = "note"
    importance: float = 0.0
    properties: dict[str, Any] = Field(default_factory=dict)


class RememberOut(BaseModel):
    id: int


class RecallIn(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=200)
    expand_depth: int = Field(default=2, ge=0, le=5)
    reinforce: bool = True


class LinkIn(BaseModel):
    tau: float = Field(default=0.8, ge=0.0, le=1.0)
    k: int = Field(default=10, ge=1, le=100)


class LinkOut(BaseModel):
    created: int


class ConsolidateOut(BaseModel):
    linked: int
    communities: int
    nodes: int


class ObsidianImportIn(BaseModel):
    path: str = Field(min_length=1, description="vault dir, relative to the allowed root")


class ObsidianImportOut(BaseModel):
    notes: int
    links: int
    missing: int


class HitOut(BaseModel):
    id: int
    content: str
    kind: str
    score: float


class NodeOut(BaseModel):
    id: int
    owner_id: int
    kind: str
    content: str
    importance: float
    strength: float
    access_count: int
    community_id: int | None
    properties: dict[str, Any]
    created_at: float
    accessed_at: float


class NeighborOut(BaseModel):
    id: int
    rel: str
    weight: float


class StatsOut(BaseModel):
    nodes: int
    edges: int


class GraphNodeOut(BaseModel):
    id: int
    content: str
    kind: str
    community_id: int | None
    importance: float
    access_count: int


class GraphEdgeOut(BaseModel):
    src: int
    dst: int
    rel: str
    weight: float


class GraphOut(BaseModel):
    nodes: list[GraphNodeOut]
    edges: list[GraphEdgeOut]


__all__ = [
    "ConsolidateOut", "GraphEdgeOut", "GraphNodeOut", "GraphOut", "HitOut",
    "LinkIn", "LinkOut", "NeighborOut", "NodeOut", "ObsidianImportIn",
    "ObsidianImportOut", "RecallIn", "RememberIn", "RememberOut", "StatsOut",
]
