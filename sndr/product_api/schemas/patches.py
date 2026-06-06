# SPDX-License-Identifier: Apache-2.0
"""Pydantic schemas for patch resources.

The patch resources expose the contents of ``sndr.dispatcher.registry`` —
the single source of truth for every patch (community + commercial) known to
this install. They power the GUI Patches view and the ``sndr patches.list``
CLI command.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PatchLifecycle = Literal[
    "experimental",
    "active",
    "deprecated",
    "retired",
]

PatchTier = Literal["community", "engine"]


class PatchSummary(BaseModel):
    """Lightweight per-patch row for the inventory listing.

    Optimized for table rendering — heavy fields (``applies_to``,
    ``conflicts_with``, full docstring) are deferred to ``PatchDetail``.
    """

    id: str = Field(description="Stable patch identifier, e.g. ``PN119``")
    title: str = Field(description="One-line human-readable summary")
    family: str = Field(
        description="Subsystem family (attention, dispatcher, kv_cache, ...)",
    )
    tier: PatchTier = Field(description="Community Apache or engine commercial")
    lifecycle: PatchLifecycle
    default_on: bool = Field(description="Applied by default in this build")
    enabled_now: bool = Field(
        description="True if currently activated for the live engine + config",
    )


class PatchDetail(PatchSummary):
    """Full patch metadata for the detail pane."""

    description: str = Field(default="", description="Multi-paragraph rationale")
    apply_module: str | None = Field(
        default=None,
        description="Dotted path to the apply module that wires this patch",
    )
    upstream_pr: str | None = Field(
        default=None,
        description="Linked upstream PR or commit",
    )
    vllm_version_range: str | None = Field(
        default=None,
        description="Pin range this patch is valid for",
    )
    conflicts_with: list[str] = Field(default_factory=list)
    superseded_by: list[str] = Field(default_factory=list)
    applies_to: dict[str, object] = Field(default_factory=dict)


class PatchInventoryReport(BaseModel):
    """Aggregate report — used by the GUI to populate the patch summary card."""

    total: int = Field(ge=0)
    active: int = Field(ge=0)
    retired: int = Field(ge=0)
    enabled_now: int = Field(ge=0)
    by_family: dict[str, int] = Field(default_factory=dict)
    by_lifecycle: dict[str, int] = Field(default_factory=dict)
    by_tier: dict[str, int] = Field(default_factory=dict)


__all__ = [
    "PatchDetail",
    "PatchInventoryReport",
    "PatchLifecycle",
    "PatchSummary",
    "PatchTier",
]
