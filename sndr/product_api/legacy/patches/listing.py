# SPDX-License-Identifier: Apache-2.0
"""Pure-data query layer for ``sndr patches list`` (M.6.1 / M.6.4).

Canonical implementation of the list / filter / spec-to-row helpers
that previously lived as private functions inside
``vllm.sndr_core.cli.patches``. The CLI's M.6.4 cleanup removed the
back-compat shims (``_matches_filters``, ``_spec_to_row``,
``_coerce_iter``); callers reach for ``matches_filters`` /
``spec_to_row`` / ``spec_to_row_dict`` here directly.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

from .types import PatchRow


def _coerce_iter(value: Any) -> tuple:
    """Loosely turn a registry tuple/list/string into an iterable for display."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(value)
    if isinstance(value, str):
        return (value,)
    return (value,)


def matches_filters(
    spec,
    *,
    tier: Optional[str] = None,
    lifecycle: Optional[str] = None,
    family: Optional[str] = None,
    default_on: Optional[bool] = None,
    has_upstream: Optional[bool] = None,
) -> bool:
    """Return True if ``spec`` matches every non-None filter."""
    if tier is not None and spec.tier != tier:
        return False
    if lifecycle is not None and spec.lifecycle != lifecycle:
        return False
    if family is not None and family not in (spec.family or ""):
        return False
    if default_on is True and not spec.default_on:
        return False
    if default_on is False and spec.default_on:
        return False
    if has_upstream is True and not spec.upstream_pr:
        return False
    if has_upstream is False and spec.upstream_pr:
        return False
    return True


def spec_to_row(spec) -> PatchRow:
    """Convert a ``PatchSpec`` into a typed flat row.

    The derived ``production_default`` field flags entries where
    ``default_on=True`` but the patch is a marker-only stub
    (``implementation_status='marker_only'``) — these have no apply
    module and the operator-facing label "Default-on" would mislead.
    Honest values:

      ``applied``  default_on + full apply_module
      ``marker``   default_on + marker_only (no runtime effect)
      ``opt-in``   default_on=False
      ``blocked``  implementation_status in {partial, placeholder} or
                   lifecycle in {retired, research}
    """
    impl = getattr(spec, "implementation_status", "full") or "full"
    if impl in ("partial", "placeholder"):
        prod_default = "blocked"
    elif spec.lifecycle in ("retired", "research"):
        prod_default = "blocked"
    elif spec.default_on and impl == "marker_only":
        prod_default = "marker"
    elif spec.default_on:
        prod_default = "applied"
    else:
        prod_default = "opt-in"
    return PatchRow(
        patch_id=spec.patch_id,
        tier=spec.tier,
        lifecycle=spec.lifecycle,
        family=spec.family,
        default_on=spec.default_on,
        production_default=prod_default,
        implementation_status=impl,
        env_flag=spec.env_flag or "",
        upstream_pr=spec.upstream_pr,
        title=(spec.title or "")[:80],
        apply_module=spec.apply_module or "",
    )


def spec_to_row_dict(spec) -> dict[str, Any]:
    """``spec_to_row`` rendered as a plain dict.

    Used by the CLI's JSON renderer which serialises rows verbatim;
    keeps dict-style access for downstream consumers that prefer
    plain-data over the frozen :class:`PatchRow` dataclass.
    """
    return asdict(spec_to_row(spec))


def list_patches(
    *,
    tier: Optional[str] = None,
    lifecycle: Optional[str] = None,
    family: Optional[str] = None,
    default_on: Optional[bool] = None,
    has_upstream: Optional[bool] = None,
) -> list[PatchRow]:
    """Return the filtered, sorted list of patch rows.

    Filter semantics match :func:`matches_filters`. Result is sorted by
    ``patch_id`` so the CLI's ASCII table + JSON ``patches`` array are
    deterministic.
    """
    from sndr.dispatcher.spec import iter_patch_specs

    rows: list[PatchRow] = []
    for spec in iter_patch_specs():
        if not matches_filters(
            spec,
            tier=tier,
            lifecycle=lifecycle,
            family=family,
            default_on=default_on,
            has_upstream=has_upstream,
        ):
            continue
        rows.append(spec_to_row(spec))
    rows.sort(key=lambda r: r.patch_id)
    return rows
