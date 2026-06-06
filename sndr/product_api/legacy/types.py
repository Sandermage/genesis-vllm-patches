# SPDX-License-Identifier: Apache-2.0
"""Shared frozen dataclasses for top-level SNDR Product API snapshots.

The GUI, future web daemon, CLI JSON renderers, and tests need the same
stable response shapes. These types intentionally avoid framework-specific
objects so callers can serialize them with ``dataclasses.asdict``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


CapabilityStatus = Literal[
    "available",
    "partial",
    "render_only",
    "deferred",
    "missing",
]


@dataclass(frozen=True)
class PlatformSnapshot:
    """Host-independent identity fields for a Product API snapshot."""

    public_brand: str
    package_name: str
    sndr_core_version: str
    os_name: str
    machine: str
    python_version: str
    engine_installed: bool


@dataclass(frozen=True)
class ProductCapability:
    """One GUI-visible capability or runtime target."""

    id: str
    title: str
    kind: str
    status: CapabilityStatus
    detail: str = ""
    required_tools: tuple[str, ...] = ()
    present_tools: tuple[str, ...] = ()
    module: Optional[str] = None


@dataclass(frozen=True)
class ProductCapabilities:
    """Capability inventory backing the GUI settings/status surfaces."""

    platform: PlatformSnapshot
    runtime_targets: tuple[ProductCapability, ...]
    features: tuple[ProductCapability, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CatalogSummary:
    """Compact operator catalog summary for dashboard and launch screens."""

    models_count: int
    hardware_count: int
    profiles_count: int
    presets_count: int
    preset_cards_count: int
    unannotated_presets_count: int
    preset_load_error_count: int
    status_counts: dict[str, int] = field(default_factory=dict)
    workload_counts: dict[str, int] = field(default_factory=dict)
    family_counts: dict[str, int] = field(default_factory=dict)
    default_presets: tuple[str, ...] = ()
    preset_load_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProductOverview:
    """Single snapshot suitable for the first GUI dashboard panel."""

    capabilities: ProductCapabilities
    catalog: CatalogSummary
