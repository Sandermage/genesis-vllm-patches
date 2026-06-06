# SPDX-License-Identifier: Apache-2.0
"""Pydantic schemas for container resources.

A container is a single Docker container running an inference workload.
The GUI uses this for the Containers view + per-container action surface
(start/stop/restart, view logs, see live patches apply matrix).
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


ContainerState = Literal[
    "running",
    "paused",
    "exited",
    "restarting",
    "created",
    "dead",
    "unknown",
]


class ContainerPort(BaseModel):
    """Single port mapping."""

    container_port: int = Field(ge=1, le=65535)
    host_port: int | None = Field(default=None, ge=1, le=65535)
    protocol: Literal["tcp", "udp"] = "tcp"


class ContainerSummary(BaseModel):
    """Light per-container row for the listing."""

    name: str
    container_id: str = Field(description="Short ID (12 chars)")
    image: str
    image_digest: str | None = Field(default=None, description="sha256:... of the image")
    state: ContainerState
    status: str = Field(description="Human-readable status string (e.g. 'Up 9 hours')")
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    served_model_name: str | None = None
    engine: str | None = None
    engine_pin: str | None = None
    ports: list[ContainerPort] = Field(default_factory=list)


class ContainerDetail(ContainerSummary):
    """Full per-container metadata for the detail pane."""

    cmd: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    mounts: dict[str, str] = Field(
        default_factory=dict,
        description="host_path → container_path",
    )
    labels: dict[str, str] = Field(default_factory=dict)
    sndr_apply_summary: dict[str, int] = Field(
        default_factory=dict,
        description="applied/skipped/failed/unresolved counts from container logs",
    )


class ContainerInventoryReport(BaseModel):
    """Aggregate counts useful for the GUI summary card."""

    total: int = Field(ge=0)
    by_state: dict[str, int] = Field(default_factory=dict)
    by_engine: dict[str, int] = Field(default_factory=dict)


__all__ = [
    "ContainerDetail",
    "ContainerInventoryReport",
    "ContainerPort",
    "ContainerState",
    "ContainerSummary",
]
