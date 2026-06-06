# SPDX-License-Identifier: Apache-2.0
"""Pydantic schemas for host inventory resources.

A *host* is a server in the operator's fleet (or just the local box).
Each host exposes:
  - hardware: GPU model, count, VRAM, CPU, RAM
  - software: OS, kernel, NVIDIA driver, CUDA, docker version
  - sndr install: version, install root, config root
  - active engine + pin (if any) — derived from the live container.

The GUI uses this for the Hosts view + Fleet aggregate dashboard.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


HostStatus = Literal["online", "degraded", "offline", "unknown"]


class GpuInfo(BaseModel):
    """Single GPU summary."""

    index: int = Field(ge=0)
    name: str = Field(description="Vendor product name, e.g. NVIDIA RTX A5000")
    sm_capability: str | None = Field(default=None, description="e.g. '8.6'")
    vram_total_mib: int = Field(ge=0)
    vram_used_mib: int = Field(ge=0)
    utilization_pct: int = Field(ge=0, le=100, default=0)
    temperature_c: int | None = Field(default=None, ge=0, le=150)
    power_draw_w: int | None = Field(default=None, ge=0)


class HostHardware(BaseModel):
    """Hardware summary for a host."""

    cpu_model: str = Field(description="e.g. Intel Xeon E5-2680 v4")
    cpu_cores: int = Field(ge=1)
    ram_total_gib: int = Field(ge=1)
    ram_available_gib: int = Field(ge=0)
    gpus: list[GpuInfo] = Field(default_factory=list)


class HostSoftware(BaseModel):
    """Software stack summary."""

    os_id: str = Field(description="e.g. 'ubuntu'")
    os_version: str
    kernel: str
    docker_version: str | None = None
    nvidia_driver: str | None = None
    cuda_version: str | None = None


class HostSummary(BaseModel):
    """Single host inventory entry."""

    hostname: str
    status: HostStatus
    last_seen_at: datetime
    sndr_version: str | None = None
    sndr_install_root: str | None = None
    active_engine: str | None = None
    active_engine_pin: str | None = None
    hardware: HostHardware | None = None
    software: HostSoftware | None = None
    notes: str | None = None


class FleetReport(BaseModel):
    """Aggregate over all known hosts."""

    total_hosts: int = Field(ge=0)
    online: int = Field(ge=0)
    degraded: int = Field(ge=0)
    offline: int = Field(ge=0)
    unknown: int = Field(ge=0)
    total_gpus: int = Field(ge=0)
    total_vram_gib: int = Field(ge=0)


__all__ = [
    "FleetReport",
    "GpuInfo",
    "HostHardware",
    "HostSoftware",
    "HostStatus",
    "HostSummary",
]
