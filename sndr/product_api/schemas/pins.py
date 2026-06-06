# SPDX-License-Identifier: Apache-2.0
"""Pydantic schemas for pin resources."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

PinStatus = Literal["current", "previous", "staging", "deprecated", "retired"]


class PinSummary(BaseModel):
    """Summary of a pin in this engine."""

    pin: str = Field(description="Normalized pin identifier")
    status: PinStatus
    full_version: str = Field(description="Full engine version string")
    upstream_sha: str | None = Field(default=None, description="Full upstream commit SHA")
    generated_at: datetime | None = Field(
        default=None,
        description="When the manifest was generated",
    )
    has_manifest: bool = Field(description="True if a manifest YAML exists")
    has_drift: bool = Field(default=False, description="True if drift has been detected")
    bench_tps_last: float | None = Field(
        default=None,
        description="Last sustained TPS measurement",
    )


class PinManifestSummary(BaseModel):
    """Summary of a pin's manifest."""

    pin: str
    file_count: int = Field(ge=0, description="Files tracked by the manifest")
    anchor_count: int = Field(ge=0, description="Anchors tracked")
    patch_count: int = Field(ge=0, description="Patches that reference this manifest")


class PinUpgradeRequest(BaseModel):
    """Request to initiate a pin upgrade pipeline."""

    target_pin: str
    auto_promote_if_clean: bool = Field(default=False)


class PinUpgradeReport(BaseModel):
    """Report from a pin upgrade attempt."""

    target_pin: str
    started_at: datetime
    completed_at: datetime | None
    steps_completed: list[str]
    drift_detected: bool
    bench_within_budget: bool
    eligible_for_promotion: bool
    notes: list[str]


__all__ = [
    "PinManifestSummary",
    "PinStatus",
    "PinSummary",
    "PinUpgradeReport",
    "PinUpgradeRequest",
]
