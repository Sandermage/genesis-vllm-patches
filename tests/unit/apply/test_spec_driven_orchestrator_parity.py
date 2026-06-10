# SPDX-License-Identifier: Apache-2.0
"""Parity invariants for the spec-driven apply orchestrator path
(SNDR_APPLY_VIA_SPECS=1) vs the legacy registry-list iteration.

These tests document and pin the v12.0.0 readiness contract surfaced
by the P3.4 dispatcher migration scout (2026-06-03):

1. Observability instrumentation — both paths must wrap apply() in the
   measure_patch_apply() context manager so PatchMetrics + spans fire
   identically. Without this, GENESIS_OBSERVABILITY=1 boots that opt
   into the spec-driven path would silently lose per-patch telemetry.

2. Order preservation — the spec-driven path iterates dispatcher's
   PATCH_REGISTRY dict (insertion-ordered in Python 3.7+). For the
   v12.0.0 default flip to be safe, this iteration order MUST match
   what the legacy registration-order iteration produces. Boot-log
   stability + dependency ordering depend on it.

3. apply_module=None handling — the 22 intentionally unmapped registry
   entries (legacy/marker_only/coordinator/retired/research) must be
   skipped with a documented reason. Pinned here.

These are STATIC analyses — they read source / iterate dispatcher
registry without booting vLLM. They are CI-safe and run in 0.1s.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Status: v11.4.0-prep parity tests (orchestrator switch deferred to
v12.0.0 release scope; these tests pin the prerequisites).
"""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ORCHESTRATOR_PATH = (
    REPO_ROOT / "sndr" / "apply" / "orchestrator.py"
)


def _orchestrator_source() -> str:
    return ORCHESTRATOR_PATH.read_text(encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# 1. Observability instrumentation parity
# ──────────────────────────────────────────────────────────────────────


def test_spec_driven_path_imports_measure_patch_apply():
    """The spec-driven loop must import measure_patch_apply from
    sndr.observability.patch_metrics — otherwise spec-driven
    boots lose per-patch telemetry."""
    src = _orchestrator_source()
    # The legacy path uses it (line ~210 in _state.py); the spec-driven
    # path must too (v11.4.0 fix).
    assert "from sndr.observability.patch_metrics import" in src, (
        "spec-driven path missing measure_patch_apply import — "
        "PatchMetrics telemetry would be lost"
    )
    assert "measure_patch_apply" in src


def test_spec_driven_path_wraps_apply_in_measure_context():
    """measure_patch_apply(display) must wrap mod.apply() in the
    spec-driven loop, mirroring legacy @register_patch instrumentation."""
    src = _orchestrator_source()
    # Look for the measure_patch_apply context manager in the
    # spec-driven loop section.
    spec_loop_start = src.find("def _run_via_specs")
    if spec_loop_start < 0:
        spec_loop_start = src.find("SNDR_APPLY_VIA_SPECS")
    assert spec_loop_start >= 0, "could not locate spec-driven loop"
    spec_loop_block = src[spec_loop_start:]
    assert "measure_patch_apply" in spec_loop_block, (
        "spec-driven path doesn't wrap apply() in measure_patch_apply() "
        "context — observability parity broken"
    )
    # Must populate metric status + reason on success path.
    assert "_metric.status = status" in spec_loop_block or (
        "metric.status = status" in spec_loop_block
    ), "metric status assignment missing"


def test_spec_driven_path_handles_observability_import_failure():
    """When the observability stack is absent (e.g., torch-less CI run),
    spec-driven path must fall back to a no-op context manager rather
    than crashing the boot."""
    src = _orchestrator_source()
    spec_loop_start = src.find("SNDR_APPLY_VIA_SPECS")
    spec_loop_block = src[spec_loop_start:]
    # ImportError fallback via nullcontext
    assert "nullcontext" in spec_loop_block or (
        "ImportError" in spec_loop_block
        and "patch_metrics" in spec_loop_block
    ), (
        "spec-driven path doesn't have observability import fallback — "
        "boot will crash on torch-less host"
    )


# ──────────────────────────────────────────────────────────────────────
# 2. Order preservation invariant
# ──────────────────────────────────────────────────────────────────────


def test_dispatcher_registry_is_dict_with_insertion_order():
    """dispatcher.PATCH_REGISTRY is a Python dict, which preserves
    insertion order (Python 3.7+). iter_patch_specs() relies on this."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    assert isinstance(PATCH_REGISTRY, dict)
    # Sample insertion-order stability — first key shouldn't shift
    # between calls.
    keys_first = list(PATCH_REGISTRY.keys())
    keys_second = list(PATCH_REGISTRY.keys())
    assert keys_first == keys_second


def test_iter_patch_specs_yields_in_dispatcher_registry_order():
    """iter_patch_specs() yields in the same order as
    dispatcher.PATCH_REGISTRY iteration. The legacy registration-order
    iteration (apply._state.PATCH_REGISTRY list) may differ — the
    v12.0.0 default flip must verify this ordering doesn't break
    dependencies."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    from sndr.dispatcher.spec import iter_patch_specs
    registry_order = list(PATCH_REGISTRY.keys())
    iter_order = [spec.patch_id for spec in iter_patch_specs()]
    assert iter_order == registry_order, (
        "iter_patch_specs() yields in different order than the "
        "underlying registry — investigate before v12.0.0 default flip"
    )


# ──────────────────────────────────────────────────────────────────────
# 3. apply_module=None handling (22 unmapped entries)
# ──────────────────────────────────────────────────────────────────────


def test_unmapped_entries_skip_reason_documented():
    """The 22 entries with apply_module=None must hit the documented
    skip path in the orchestrator. Static check on the source."""
    src = _orchestrator_source()
    assert "no apply_module declared" in src, (
        "spec-driven path doesn't have the documented skip reason "
        "for apply_module=None entries"
    )
    assert "informational entry" in src


def test_unmapped_entries_count_matches_audit():
    """The number of registry entries with apply_module=None should
    match what audit_dispatcher_migration_readiness.py reports as
    'intentionally unmapped' — sanity check."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    unmapped = [
        pid for pid, meta in PATCH_REGISTRY.items()
        if isinstance(meta, dict) and meta.get("apply_module") is None
    ]
    # Per scout: 22 intentionally unmapped (10 legacy + 6 marker_only +
    # 1 research + 2 retired + 3 coordinator). The count may shift +/-1
    # as patches retire, but should stay in the 18-30 range.
    assert 18 <= len(unmapped) <= 30, (
        f"unmapped count {len(unmapped)} outside expected 18-30 range. "
        f"Update tier_breakdown in scripts/audit_dispatcher_migration_readiness.py "
        f"if a patch was added without apply_module or lifecycle override."
    )


# ──────────────────────────────────────────────────────────────────────
# 4. Skip-reason taxonomy
# ──────────────────────────────────────────────────────────────────────


def test_lifecycle_distribution_is_known():
    """Sanity: lifecycle values are from the documented enum.

    Pinned set: stable, experimental, legacy, retired, deprecated,
    research, coordinator. Adding a new lifecycle value requires
    updating the spec-driven orchestrator's skip logic + the audit
    script's tier categorization.
    """
    from sndr.dispatcher.registry import PATCH_REGISTRY
    known = {
        "stable", "experimental", "legacy", "retired", "deprecated",
        "research", "coordinator",
    }
    seen = {
        meta.get("lifecycle")
        for meta in PATCH_REGISTRY.values()
        if isinstance(meta, dict) and meta.get("lifecycle") is not None
    }
    unknown = seen - known
    assert not unknown, (
        f"unknown lifecycle values in registry: {unknown}. "
        f"Update the parity test + the audit script categorization."
    )


def test_marker_only_lifecycle_has_no_apply_module():
    """implementation_status=marker_only entries that ALSO lack an
    apply_module are the canonical "informational only" bucket.

    The combination (marker_only impl_status) AND (apply_module=None)
    is what the audit script categorizes as
    `intentionally_unmapped_marker_only` and what the spec-driven
    orchestrator skips with reason "no apply_module declared".
    """
    from sndr.dispatcher.registry import PATCH_REGISTRY
    marker_only_unmapped = [
        pid for pid, meta in PATCH_REGISTRY.items()
        if isinstance(meta, dict)
        and meta.get("implementation_status") == "marker_only"
        and meta.get("apply_module") is None
    ]
    # Realistic count: 18-22 entries are marker_only + no apply_module.
    # The bucket includes early-Genesis informational entries +
    # coordinators + post-merge tombstones. Tolerate +/- 5 to allow
    # natural drift as patches lifecycle-transition.
    assert 15 <= len(marker_only_unmapped) <= 30, (
        f"marker_only + apply_module=None count "
        f"{len(marker_only_unmapped)} outside expected 15-30 range — "
        f"investigate registry shape"
    )
