# SPDX-License-Identifier: Apache-2.0
"""Pure-data query layer for ``sndr patches diff-upstream`` (M.6.1)."""
from __future__ import annotations

from typing import Any

from .types import DiffReport


def diff_upstream() -> DiffReport:
    """Surface patches likely retiring because upstream merged the fix.

    Two signals:

      1. ``lifecycle == "merged_upstream"`` — operator already flipped.
      2. ``upstream_pr`` set AND active — heuristic candidate for audit.

    Mirrors the bucket shape ``cli.patches._run_diff_upstream`` emitted
    pre-M.6.1.
    """
    from vllm.sndr_core.dispatcher import PATCH_REGISTRY
    from vllm.sndr_core.dispatcher.spec import iter_patch_specs

    merged_upstream: list[dict[str, Any]] = []
    has_upstream_pr: list[dict[str, Any]] = []

    for spec in iter_patch_specs():
        meta = PATCH_REGISTRY.get(spec.patch_id) or {}
        if spec.lifecycle == "merged_upstream":
            merged_upstream.append({
                "patch_id": spec.patch_id,
                "title": (spec.title or "")[:80],
                "upstream_pr": spec.upstream_pr,
                "credit": meta.get("credit", ""),
            })
            continue
        if spec.upstream_pr:
            has_upstream_pr.append({
                "patch_id": spec.patch_id,
                "title": (spec.title or "")[:80],
                "upstream_pr": spec.upstream_pr,
                "lifecycle": spec.lifecycle,
                "default_on": spec.default_on,
            })

    return DiffReport(
        merged_upstream=tuple(merged_upstream),
        has_upstream_pr=tuple(has_upstream_pr),
    )
