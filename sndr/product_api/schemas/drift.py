# SPDX-License-Identifier: Apache-2.0
"""Pydantic schemas for drift detection resources."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

DriftSeverity = Literal["ok", "benign", "drift", "blocked"]


class AnchorDriftReport(BaseModel):
    """Drift status of a single anchor."""

    anchor_name: str
    severity: DriftSeverity
    expected_md5: str | None = None
    actual_md5: str | None = None
    notes: str | None = None


class FileDriftReport(BaseModel):
    """Drift status of one file (potentially multiple anchors)."""

    file_path: str
    severity: DriftSeverity
    anchors: list[AnchorDriftReport]
    affected_patches: list[str] = Field(default_factory=list)


class DriftSummary(BaseModel):
    """Per-pin drift summary."""

    engine: str
    pin: str
    checked_at: datetime
    overall_severity: DriftSeverity
    files_ok: int = Field(ge=0)
    files_benign: int = Field(ge=0)
    files_drift: int = Field(ge=0)
    files_blocked: int = Field(ge=0)
    affected_patches: list[str] = Field(default_factory=list)


class DriftReport(DriftSummary):
    """Full drift report with per-file details."""

    files: list[FileDriftReport]


__all__ = [
    "AnchorDriftReport",
    "DriftReport",
    "DriftSeverity",
    "DriftSummary",
    "FileDriftReport",
]
