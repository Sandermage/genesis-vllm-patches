# SPDX-License-Identifier: Apache-2.0
"""Unit test for the legacy vs spec-driven apply-matrix comparison
script.

This test runs the audit and verifies it produces a well-formed
structured output. The expected divergence is documented + pinned —
this is NOT a "v12_0_safe must be True" test (which would block CI
because the structural divergence is real and intentional in v11.3.0;
v12.0.0 implementation closes it via unified iteration).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "audit_legacy_vs_spec_driven_apply_matrix.py"


def test_script_exists_and_is_executable():
    assert SCRIPT.is_file()


def test_audit_runs_and_exits_zero():
    """Default mode (no flags) produces human report + returns 0."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0
    assert "Apply-matrix comparison" in proc.stdout
    assert "Legacy path total" in proc.stdout
    assert "Spec-driven path total" in proc.stdout


def test_audit_json_mode_returns_structured_output():
    """--json emits a parseable structured payload."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert "legacy_matrix" in data
    assert "spec_driven_matrix" in data
    assert "diff" in data
    diff = data["diff"]
    assert "legacy_total" in diff
    assert "spec_driven_total" in diff
    assert "common_count" in diff
    assert "v12_0_safe" in diff


def test_audit_reports_meaningful_totals():
    """Both matrices should have plausible patch counts (registry is at
    least 200 entries)."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    data = json.loads(proc.stdout)
    diff = data["diff"]
    # Plausible bounds — sanity range, NOT an exact count. The registry
    # was 241 in v11.3.0; spec_driven_total excludes apply_module=None
    # (informational entries).
    # Reconciled 2026-06-19: registry has grown to 319 patches, so
    # legacy_total=242 and spec_driven_total=303 (live). Upper bound
    # raised 300 -> 400 to give headroom for legitimate fleet growth
    # while still catching a runaway (e.g. a duplicated/exploded
    # registry). This is a plausibility range, not a tripwire on an
    # exact count.
    assert 100 <= diff["legacy_total"] <= 400, (
        f"legacy_total {diff['legacy_total']} outside plausible range"
    )
    assert 100 <= diff["spec_driven_total"] <= 400, (
        f"spec_driven_total {diff['spec_driven_total']} outside plausible"
    )
    # Common patches should be the majority of both
    assert diff["common_count"] >= 100, (
        f"common_count {diff['common_count']} too low"
    )


def test_audit_legacy_matrix_entries_have_patch_id():
    """Every legacy matrix entry has a patch_id (non-empty)."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    data = json.loads(proc.stdout)
    for entry in data["legacy_matrix"]:
        assert "patch_id" in entry, f"missing patch_id: {entry}"
        assert entry["patch_id"], f"empty patch_id: {entry}"


def test_audit_spec_driven_matrix_entries_have_apply_module():
    """Every spec-driven matrix entry has apply_module set (informational
    entries were filtered out in the matrix builder)."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    data = json.loads(proc.stdout)
    for entry in data["spec_driven_matrix"]:
        assert entry.get("apply_module") is not None, (
            f"spec-driven entry has apply_module=None — should have been "
            f"filtered out: {entry}"
        )


def test_audit_strict_mode_propagates_v12_safe_verdict():
    """--strict returns non-zero when v12_0_safe is False.

    In v11.3.0 the matrices structurally diverge (documented +
    intentional — see Phase 6 P3.4 master plan), so --strict should
    return 1. v12.0.0 implementation eliminates the divergence and
    --strict will return 0."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--strict", "--json"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    data = json.loads(proc.stdout)
    v12_safe = data["diff"]["v12_0_safe"]
    if v12_safe:
        assert proc.returncode == 0, (
            "--strict returned non-zero despite v12_0_safe=True"
        )
    else:
        assert proc.returncode == 1, (
            f"--strict expected exit 1 (v12_0_safe=False) but got "
            f"{proc.returncode}"
        )
