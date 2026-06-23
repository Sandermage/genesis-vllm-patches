# SPDX-License-Identifier: Apache-2.0
"""Pure-API layer for ``sndr patches release-check`` — M.6.2.

Thin wrapper around
:func:`sndr.proof.release_check.evaluate_release` that builds
the :class:`ReleasePolicy` from explicit parameters and, when scope =
``"production-subset"``, derives the canonical hardened-release patch
filter via :func:`sndr.proof.production_subset.get_production_subset`.

The result is the raw report dict (already a stable JSON shape consumed
by ``audit-release-check`` + CLI ``--json`` output) wrapped in a small
dataclass for typed access. ``ReleaseCheckError`` is re-raised so the
CLI keeps its single ``except ReleaseCheckError`` clause.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class ReleaseCheckResult:
    """Outcome of a release-check evaluation.

    ``raw`` is the dict returned by ``evaluate_release`` — stable
    contract consumed by the CLI ``--json`` path and downstream audit
    tooling (``audit-release-check``). Convenience properties below
    surface the common fields without duplicating the schema.
    """

    raw: dict[str, Any]

    @property
    def release_blocked(self) -> bool:
        return bool(self.raw.get("release_blocked", False))

    @property
    def considered(self) -> int:
        return int(self.raw.get("considered", 0))

    @property
    def total(self) -> int:
        return int(self.raw.get("total", 0))

    @property
    def policy(self) -> dict[str, Any]:
        return dict(self.raw.get("policy", {}))

    @property
    def verdicts(self) -> list[dict[str, Any]]:
        return list(self.raw.get("verdicts", []))


def release_check(
    *,
    mode: str,
    out_dir: Optional[Path] = None,
    max_regression_pct: Optional[float] = None,
    patch_filter: Optional[Iterable[str]] = None,
    tier_filter: Optional[Iterable[str]] = None,
    scope: str = "all",
) -> ReleaseCheckResult:
    """Evaluate the release-check policy.

    Builds a :class:`sndr.proof.release_check.ReleasePolicy`
    from explicit args. When ``patch_filter`` is unset and
    ``scope == "production-subset"``, the patch filter is widened to
    the canonical production scope.

    Raises :class:`sndr.proof.release_check.ReleaseCheckError`
    on invalid policy inputs.
    """
    from sndr.proof import DEFAULT_PROOF_DIR
    from sndr.proof.release_check import (
        ReleasePolicy,
        evaluate_release,
    )

    target_dir = Path(out_dir) if out_dir is not None else DEFAULT_PROOF_DIR

    resolved_patch_filter: Optional[frozenset[str]] = (
        frozenset(patch_filter) if patch_filter is not None else None
    )
    if resolved_patch_filter is None and scope == "production-subset":
        from sndr.proof.production_subset import get_production_subset

        resolved_patch_filter = get_production_subset()

    resolved_tier_filter: Optional[frozenset[str]] = (
        frozenset(tier_filter) if tier_filter is not None else None
    )

    policy = ReleasePolicy(
        mode=mode,
        max_regression_pct=max_regression_pct,
        patch_filter=resolved_patch_filter,
        tier_filter=resolved_tier_filter,
    )
    report = evaluate_release(policy, out_dir=target_dir)
    return ReleaseCheckResult(raw=report)
