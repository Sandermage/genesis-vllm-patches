# SPDX-License-Identifier: Apache-2.0
"""Read-only apply-order shadow status for the Product API.

Surfaces ``sndr.apply.shadow.compare_apply_orders`` — the static comparison of the
legacy per-patch apply loop vs the spec-driven loop (SNDR_APPLY_VIA_SPECS=1). The
operator-critical field is ``spec_boot_unsafe``: patches that the legacy loop
applies but that would silently DROP under the spec-driven loop (a boot that looks
healthy while quietly missing patches). Also reports coverage and the
known/unexpected spec-only split.

Fail-safe: any analysis error yields an empty report with an ``error`` tag, never
a 500 — same contract as the other read-only patches endpoints.
"""
from __future__ import annotations

import dataclasses
from typing import Any

_EMPTY: dict[str, Any] = {
    "legacy_count": 0,
    "spec_count": 0,
    "legacy_only": [],
    "spec_only": [],
    "spec_only_known": [],
    "spec_only_unexpected": [],
    "spec_boot_unsafe": [],
    "legacy_unparseable": [],
    "spec_with_apply_module": [],
    "spec_without_apply_module": [],
}


def shadow_status() -> dict[str, Any]:
    """Return the apply-order diff as a JSON-serialisable dict.

    Keys: ``legacy_count``/``spec_count`` (coverage), ``spec_boot_unsafe`` (would
    silently drop under spec-driven apply), ``spec_only_unexpected`` (spec-only
    and NOT on the known allowlist), ``legacy_unparseable``, plus the full
    legacy/spec id splits.
    """
    try:
        from sndr.apply.shadow import compare_apply_orders

        return dataclasses.asdict(compare_apply_orders())
    except Exception as exc:  # noqa: BLE001 - best-effort; never break the API
        return {**_EMPTY, "error": type(exc).__name__}
