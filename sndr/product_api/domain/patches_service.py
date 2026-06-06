# SPDX-License-Identifier: Apache-2.0
"""Patch service — read-only views over ``sndr.dispatcher.registry``.

The dispatcher registry is the canonical source of truth for every patch
known to the platform. This module never mutates it; it only renders it
into resource models that the GUI / CLI / OpenAPI consumers can render.
"""
from __future__ import annotations

import os
from collections import Counter
from typing import Any

from sndr.dispatcher.registry import PATCH_REGISTRY
from sndr.product_api.schemas.patches import (
    PatchDetail,
    PatchInventoryReport,
    PatchLifecycle,
    PatchSummary,
    PatchTier,
)


def _normalize_lifecycle(value: str | None) -> PatchLifecycle:
    """Map registry lifecycle to the schema literal."""
    if value in ("experimental", "active", "deprecated", "retired"):
        return value  # type: ignore[return-value]
    if value == "stable":
        return "active"
    return "experimental"


def _normalize_tier(value: str | None) -> PatchTier:
    if value == "engine":
        return "engine"
    return "community"


def _is_enabled_now(entry: dict[str, Any]) -> bool:
    """Probe the live env for the patch's enable flag.

    Mirrors how the dispatcher decides whether to apply: ``default_on`` AND
    not explicitly disabled, OR explicitly enabled. We read from
    ``os.environ`` so the live container state is reflected.
    """
    env_flag = entry.get("env_flag")
    if not env_flag:
        return bool(entry.get("default_on", False))
    env_value = os.environ.get(env_flag)
    if env_value is None:
        return bool(entry.get("default_on", False))
    return env_value.strip() in ("1", "true", "True", "yes", "on")


def _to_summary(patch_id: str, entry: dict[str, Any]) -> PatchSummary:
    return PatchSummary(
        id=patch_id,
        title=str(entry.get("title", patch_id)),
        family=str(entry.get("family", "unknown")),
        tier=_normalize_tier(entry.get("tier")),
        lifecycle=_normalize_lifecycle(entry.get("lifecycle")),
        default_on=bool(entry.get("default_on", False)),
        enabled_now=_is_enabled_now(entry),
    )


def _to_detail(patch_id: str, entry: dict[str, Any]) -> PatchDetail:
    summary = _to_summary(patch_id, entry)
    return PatchDetail(
        **summary.model_dump(),
        description=str(entry.get("description", "")),
        apply_module=entry.get("apply_module"),
        upstream_pr=_format_pr(entry.get("upstream_pr")),
        vllm_version_range=entry.get("vllm_version_range"),
        conflicts_with=list(entry.get("conflicts_with", []) or []),
        superseded_by=list(entry.get("superseded_by", []) or []),
        applies_to=dict(entry.get("applies_to", {}) or {}),
    )


def _format_pr(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, int):
        return f"vllm#{value}"
    return str(value)


def list_patches(
    *,
    family: str | None = None,
    tier: str | None = None,
    lifecycle: str | None = None,
    enabled_only: bool = False,
) -> list[PatchSummary]:
    """Return a filtered list of patches.

    Filters compose with AND semantics. ``None`` means do-not-filter.
    """
    out: list[PatchSummary] = []
    for patch_id, entry in PATCH_REGISTRY.items():
        summary = _to_summary(patch_id, entry)
        if family is not None and summary.family != family:
            continue
        if tier is not None and summary.tier != tier:
            continue
        if lifecycle is not None and summary.lifecycle != lifecycle:
            continue
        if enabled_only and not summary.enabled_now:
            continue
        out.append(summary)
    return sorted(out, key=lambda p: p.id)


def get_patch(patch_id: str) -> PatchDetail | None:
    """Return the detail for one patch by id, or ``None`` if unknown."""
    entry = PATCH_REGISTRY.get(patch_id)
    if entry is None:
        return None
    return _to_detail(patch_id, entry)


def inventory_report() -> PatchInventoryReport:
    """Aggregate counts useful for the GUI summary card."""
    summaries = list_patches()
    by_family: Counter[str] = Counter()
    by_lifecycle: Counter[str] = Counter()
    by_tier: Counter[str] = Counter()
    active = 0
    retired = 0
    enabled_now = 0
    for s in summaries:
        by_family[s.family] += 1
        by_lifecycle[s.lifecycle] += 1
        by_tier[s.tier] += 1
        if s.lifecycle == "active":
            active += 1
        elif s.lifecycle == "retired":
            retired += 1
        if s.enabled_now:
            enabled_now += 1
    return PatchInventoryReport(
        total=len(summaries),
        active=active,
        retired=retired,
        enabled_now=enabled_now,
        by_family=dict(by_family),
        by_lifecycle=dict(by_lifecycle),
        by_tier=dict(by_tier),
    )


__all__ = ["get_patch", "inventory_report", "list_patches"]
