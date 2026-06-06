# SPDX-License-Identifier: Apache-2.0
"""Pydantic schemas for the observability surfaces.

Covers four resources used by the GUI's monitoring tabs:
  - Bench runs (history of bench results)
  - Doctor report (health / readiness gates)
  - Models registry (V2 model catalog snapshot)
  - Configs registry (V2 config catalog snapshot)
  - Evidence gates (release-readiness gates)
  - Jobs (async background work — installs, bench runs)
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ── Bench ──────────────────────────────────────────────────────────────────

BenchOutcome = Literal["pending", "running", "success", "regression", "failed"]


class BenchSummary(BaseModel):
    """One bench run summary row."""

    id: str = Field(description="Unique run id (uuid hex)")
    timestamp: datetime
    model: str
    pin: str
    wall_tps: float = Field(ge=0)
    decode_tpot_ms: float = Field(ge=0)
    ttft_ms: float = Field(ge=0)
    accept_rate: float | None = Field(default=None, ge=0, le=1)
    cv: float = Field(ge=0)
    n: int = Field(ge=0, description="Sample size")
    outcome: BenchOutcome
    delta_tps_vs_baseline: float | None = Field(
        default=None,
        description="Percent change vs. the model's documented baseline; "
                    "negative = regression",
    )


# ── Doctor ─────────────────────────────────────────────────────────────────

class DoctorFinding(BaseModel):
    """One doctor finding."""

    category: str = Field(description="e.g. 'gpu', 'docker', 'patches'")
    severity: Literal["info", "warning", "error", "critical"]
    title: str
    detail: str | None = None
    remediation: str | None = None


class DoctorReport(BaseModel):
    """Aggregate doctor output."""

    checked_at: datetime
    ok: bool = Field(description="True if no errors or criticals")
    findings: list[DoctorFinding] = Field(default_factory=list)
    counts: dict[str, int] = Field(
        default_factory=dict,
        description="severity → count",
    )


# ── Models / Configs ───────────────────────────────────────────────────────

class ModelSummary(BaseModel):
    """V2 model registry row."""

    id: str
    title: str | None = None
    served_model_name: str | None = None
    family: str | None = None
    quant_format: str | None = None
    kv_cache_dtype: str | None = None
    spec_method: str | None = None
    parameter_count: str | None = Field(
        default=None,
        description="e.g. '27B' / '35B-A3B'",
    )
    deployable_targets: list[str] = Field(
        default_factory=list,
        description="hardware ids this model can run on",
    )


class HardwareSummary(BaseModel):
    """V2 hardware registry row."""

    id: str
    title: str | None = None
    gpu: str | None = None
    gpu_count: int | None = None
    vram_per_gpu_gib: int | None = None
    cpu_cores: int | None = None
    ram_gib: int | None = None


class ProfileSummary(BaseModel):
    """V2 profile registry row."""

    id: str
    title: str | None = None
    parent_model: str | None = None
    role: str | None = None


class PresetSummary(BaseModel):
    """V2 preset registry row."""

    id: str
    title: str | None = None
    composed_key: str | None = None
    parent_model: str | None = None


class ConfigCatalog(BaseModel):
    """Aggregate of V2 catalog."""

    models: list[ModelSummary] = Field(default_factory=list)
    hardware: list[HardwareSummary] = Field(default_factory=list)
    profiles: list[ProfileSummary] = Field(default_factory=list)
    presets: list[PresetSummary] = Field(default_factory=list)


# ── Evidence ───────────────────────────────────────────────────────────────

EvidenceStatus = Literal["ok", "warning", "fail", "skipped"]


class EvidenceGate(BaseModel):
    """One release-readiness gate."""

    id: str
    name: str
    status: EvidenceStatus
    summary: str | None = None
    last_run: datetime | None = None


class EvidenceReport(BaseModel):
    """Aggregate evidence pass."""

    gates_total: int = Field(ge=0)
    gates_ok: int = Field(ge=0)
    gates_warning: int = Field(ge=0)
    gates_fail: int = Field(ge=0)
    gates_skipped: int = Field(ge=0)
    gates: list[EvidenceGate] = Field(default_factory=list)


# ── Jobs ───────────────────────────────────────────────────────────────────

JobState = Literal["queued", "running", "succeeded", "failed", "canceled"]


class JobSummary(BaseModel):
    """One async-job entry."""

    id: str
    kind: str = Field(description="e.g. 'bench', 'install', 'manifest_gen'")
    state: JobState
    progress_pct: int = Field(ge=0, le=100, default=0)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    operator: str | None = None
    summary: str | None = None


__all__ = [
    "BenchOutcome",
    "BenchSummary",
    "ConfigCatalog",
    "DoctorFinding",
    "DoctorReport",
    "EvidenceGate",
    "EvidenceReport",
    "EvidenceStatus",
    "HardwareSummary",
    "JobState",
    "JobSummary",
    "ModelSummary",
    "PresetSummary",
    "ProfileSummary",
]
