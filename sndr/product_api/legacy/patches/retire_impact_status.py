# SPDX-License-Identifier: Apache-2.0
"""Read-only retire-impact status for the Product API.

Surfaces the anchor-SoT retire-impact detector — which active dependents a
retired patch would break — as a structured payload for the GUI. These are the
HIGH/MEDIUM dependency-break edges the pin-bump preflight gate exists to catch
(the dev301-class silent perf regression: a retired patch whose anchor a
perf-bearing dependent still targets). HIGH = dependent is perf-bearing AND
anchor-breaks; MEDIUM = registry edge only.

Fail-safe: a missing vllm runtime (Mac dev host) or any analysis error yields an
empty report with an ``error`` tag, never a 500 — same contract as the other
read-only patches endpoints.
"""
from __future__ import annotations

from typing import Any


def retire_impact_status() -> dict[str, Any]:
    """Return ``{high_count, medium_count, edges[]}`` for the live registry.

    Each edge is ``{retired, retired_reason, dependent, severity, via[],
    dependent_category, dependent_lifecycle, dependent_default_on, detail}``.
    """
    try:
        from sndr.engines.vllm.retire_impact import detect_on_live_registry

        return detect_on_live_registry().to_dict()
    except Exception as exc:  # noqa: BLE001 - best-effort; never break the API
        return {"high_count": 0, "medium_count": 0, "edges": [], "error": type(exc).__name__}
