# SPDX-License-Identifier: Apache-2.0
"""Genesis observability — per-patch instrumentation utilities.

Public API:
  • ``measure_patch_apply(name)`` — context manager / decorator that
    captures elapsed time + memory delta around a patch's apply()
    invocation, emits one structured log line, and stores the metric in
    a process-local registry queryable via ``get_apply_metrics()``.
  • ``get_apply_metrics()`` → list of ``PatchApplyMetric`` ordered by
    apply sequence.
  • ``reset_apply_metrics()`` — for tests.

The instrumentation is opt-in via env ``GENESIS_OBSERVABILITY=1`` so
the default-OFF posture preserves apply-loop performance for operators
who don't need the data.
"""
from __future__ import annotations

from .patch_metrics import (
    PatchApplyMetric,
    get_apply_metrics,
    measure_patch_apply,
    reset_apply_metrics,
)
from .cudagraph_dispatch import (
    CudagraphDispatchSummary,
    emit_summary as emit_cudagraph_summary,
    get_summary as get_cudagraph_summary,
    record_dispatch as record_cudagraph_dispatch,
    reset_summary as reset_cudagraph_summary,
)
from .spec_decode_metrics import (
    get_profile_label as get_spec_decode_profile_label,
    is_enabled as is_spec_decode_metric_enabled,
    record_acceptance as record_spec_decode_acceptance,
)

__all__ = [
    # Per-patch apply timing (Wave 7)
    "PatchApplyMetric",
    "get_apply_metrics",
    "measure_patch_apply",
    "reset_apply_metrics",
    # CUDA graph dispatch hit-rate (Sprint 2.6)
    "CudagraphDispatchSummary",
    "emit_cudagraph_summary",
    "get_cudagraph_summary",
    "record_cudagraph_dispatch",
    "reset_cudagraph_summary",
    # Spec-decode acceptance proxy metric (PN282 / 2026-05-20)
    "get_spec_decode_profile_label",
    "is_spec_decode_metric_enabled",
    "record_spec_decode_acceptance",
]
