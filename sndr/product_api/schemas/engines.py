# SPDX-License-Identifier: Apache-2.0
"""Pydantic schemas for engine resources."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EngineSummary(BaseModel):
    """Summary of an engine adapter."""

    name: str = Field(description="Engine identifier (e.g. 'vllm', 'sglang')")
    display_name: str = Field(description="Human-readable name (e.g. 'vLLM')")
    active: bool = Field(description="True if a working adapter is registered")
    version: str | None = Field(
        default=None,
        description="Engine package version if installed",
    )
    pin: str | None = Field(
        default=None,
        description="Normalized pin identifier matching a manifest directory",
    )
    container_count: int = Field(
        default=0,
        description="Number of running containers managed by this engine",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Operator-visible notes (deprecations, warnings, etc.)",
    )


class EngineDetail(EngineSummary):
    """Full engine details including manifest support and patch counts."""

    supported_pins: list[str] = Field(
        default_factory=list,
        description="All pins with valid manifests",
    )
    patch_count_community: int = Field(default=0)
    patch_count_engine: int = Field(default=0)
    install_root: str | None = Field(default=None)
    capabilities: dict[str, bool] = Field(
        default_factory=dict,
        description="Feature flags (e.g. 'multi_pin', 'drift_detection')",
    )


EngineStatus = Literal["ok", "degraded", "missing", "error"]


__all__ = ["EngineDetail", "EngineStatus", "EngineSummary"]
