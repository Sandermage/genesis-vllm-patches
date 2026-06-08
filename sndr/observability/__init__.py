# SPDX-License-Identifier: Apache-2.0
"""Engine-agnostic observability primitives + per-patch instrumentation.

Two flavors of API live here:

  1. **Structured logging** (engine-agnostic, always on):
       SndrJsonFormatter, configure_logging, current_trace_id, set_trace_id

  2. **Per-patch instrumentation** (opt-in via GENESIS_OBSERVABILITY=1):
       measure_patch_apply, get_apply_metrics, reset_apply_metrics,
       CudagraphDispatchSummary, record_spec_decode_acceptance, ...

The instrumentation is opt-in via env ``GENESIS_OBSERVABILITY=1`` so the
default-OFF posture preserves apply-loop performance for operators who
don't need the data.
"""
from sndr.observability.logging import (  # noqa: F401
    SndrJsonFormatter,
    configure_logging,
    current_trace_id,
    set_trace_id,
)

# Per-patch instrumentation (migrated from sndr.observability in
# Phase 5/8). Re-exported here so existing call sites continue to work.
from sndr.observability.patch_metrics import (  # noqa: F401
    PatchApplyMetric,
    get_apply_metrics,
    measure_patch_apply,
    reset_apply_metrics,
)
from sndr.observability.cudagraph_dispatch import (  # noqa: F401
    CudagraphDispatchSummary,
    emit_summary as emit_cudagraph_summary,
    get_summary as get_cudagraph_summary,
    record_dispatch as record_cudagraph_dispatch,
    reset_summary as reset_cudagraph_summary,
)
from sndr.observability.spec_decode_metrics import (  # noqa: F401
    get_profile_label as get_spec_decode_profile_label,
    is_enabled as is_spec_decode_metric_enabled,
    record_acceptance as record_spec_decode_acceptance,
)
from sndr.observability.multiproc_bootstrap import (  # noqa: F401
    is_initialised as is_prometheus_multiproc_initialised,
    setup_prometheus_multiproc_dir,
)

__all__ = [
    # Structured logging (engine-agnostic)
    "SndrJsonFormatter",
    "configure_logging",
    "current_trace_id",
    "set_trace_id",
    # Per-patch apply timing
    "PatchApplyMetric",
    "get_apply_metrics",
    "measure_patch_apply",
    "reset_apply_metrics",
    # CUDA graph dispatch hit-rate
    "CudagraphDispatchSummary",
    "emit_cudagraph_summary",
    "get_cudagraph_summary",
    "record_cudagraph_dispatch",
    "reset_cudagraph_summary",
    # Spec-decode acceptance proxy
    "get_spec_decode_profile_label",
    "is_spec_decode_metric_enabled",
    "record_spec_decode_acceptance",
    # Multiprocess Prometheus
    "is_prometheus_multiproc_initialised",
    "setup_prometheus_multiproc_dir",
]
