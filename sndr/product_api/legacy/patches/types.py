# SPDX-License-Identifier: Apache-2.0
"""Frozen dataclass result types for ``product_api.patches`` queries.

Each dataclass mirrors the JSON output shape of the corresponding
``sndr patches`` CLI command so renderers can convert via
``dataclasses.asdict`` without manual field plumbing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


# ─── Listing ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PatchRow:
    """One flat record describing a registry entry — list / table view."""

    patch_id: str
    tier: str
    lifecycle: str
    family: str
    default_on: bool
    production_default: str
    implementation_status: str
    env_flag: str
    upstream_pr: Optional[int]
    title: str
    apply_module: str


# ─── Explain ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExplainView:
    """Full metadata for one patch, plus a live-decision probe result.

    ``live_decision`` is ``None`` when the dispatcher's ``should_apply``
    probe could not run on this host (e.g. no vllm installed on a Mac).
    When the probe raised, ``live_decision_error`` carries the exception
    class name so the renderer can mirror the historical CLI output
    ``"(unavailable: <ExceptionType>)"`` byte-identically.
    """

    patch_id: str  # canonical-cased registry key
    meta: dict[str, Any]
    spec: Any  # vllm.sndr_core.dispatcher.spec.PatchSpec (typed lazily)
    live_decision: Optional[tuple[bool, str]] = None
    live_decision_error: Optional[str] = None


# ─── Doctor ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DoctorReport:
    """Registry-level validator + apply_module coverage snapshot."""

    registry_size: int
    issues: tuple[Any, ...]  # tuple of dispatcher.audit.ValidationIssue
    coverage: Any  # dispatcher.spec.CoverageReport


# ─── Diff-upstream ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DiffReport:
    """Two-bucket upstream-drift triage report."""

    merged_upstream: tuple[dict[str, Any], ...]
    has_upstream_pr: tuple[dict[str, Any], ...]


# ─── Bundles ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BundleSpec:
    """Metadata for one bundle entry.

    ``has_apply`` is ``None`` when not probed (cheap-list path) and
    ``True``/``False`` when explicitly resolved (``explain_bundle``).
    """

    name: str
    umbrella_flag: str
    tier: str
    description: str
    module_path: str = ""
    has_apply: Optional[bool] = None
    import_error: Optional[str] = None
