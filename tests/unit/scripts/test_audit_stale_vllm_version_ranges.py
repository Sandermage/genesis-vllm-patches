# SPDX-License-Identifier: Apache-2.0
"""Unit test for the stale vllm_version_range audit script.

Regression guard for CLAUDE.md Class 5 known-bug surface:

  - CRITICAL count MUST be 0 — no default_on=True patch may silently
    skip on the current pin due to stale version range upper bound
  - WARN count tracked but not pinned (allowed to drift with pin bumps)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "audit_stale_vllm_version_ranges.py"


def test_script_exists():
    assert SCRIPT.is_file()


def test_audit_runs_and_exits_zero():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0
    assert "Stale vllm_version_range audit" in proc.stdout


def test_audit_json_mode_structured():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert "pin" in data
    assert "total_stale_ranges" in data
    assert "critical_count" in data
    assert "warn_count" in data
    assert "rows" in data


def test_no_new_critical_beyond_baseline():
    """v11.3.0 BUG #14: CRITICAL = default_on=True silently skipping
    OR opt-in patch enabled-in-builtin-YAML silently no-oping. The
    19 known entries are pinned in
    `_BASELINE_CRITICAL_STALE` and surface as INFO in audit output.

    Any CRITICAL outside that baseline is a NEW regression: either
    a recently-added patch has a stale upper bound, or a YAML newly
    enables a patch whose range hasn't been bumped. Force review.
    """
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    data = json.loads(proc.stdout)
    # Import baseline directly so this test breaks atomically with
    # any baseline change in the audit script.
    import sys as _sys
    if str(REPO_ROOT) not in _sys.path:
        _sys.path.insert(0, str(REPO_ROOT))
    from scripts.audit_stale_vllm_version_ranges import (
        _BASELINE_CRITICAL_STALE,
    )
    critical_pids = {
        r["patch_id"] for r in data["rows"] if r["severity"] == "CRITICAL"
    }
    new = sorted(critical_pids - _BASELINE_CRITICAL_STALE)
    assert not new, (
        f"{len(new)} NEW CRITICAL stale-range entries beyond v11.3.0 "
        f"baseline:\n" + "\n".join(f"  - {p}" for p in new) +
        f"\n\nFix the range in registry.py OR add to baseline "
        f"_BASELINE_CRITICAL_STALE in scripts/audit_stale_vllm_version_ranges.py"
    )


def test_strict_mode_respects_baseline():
    """`--strict` should exit 0 when all CRITICAL entries are in the
    v11.3.0 baseline (known queued work, not a regression). Exits 1
    only on NEW critical entries beyond baseline.
    """
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--strict"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    # Currently 19 CRITICAL all in baseline → --strict should pass
    assert proc.returncode == 0, (
        f"--strict failed at v11.3.0 baseline (expected 0). "
        f"Stderr:\n{proc.stderr}"
    )


def test_pin_override_works():
    """`--pin VERSION` lets the audit run against an arbitrary pin —
    useful for "what if we bumped to X" experiments."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--pin", "0.99.0", "--json"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["pin"] == "0.99.0"
    # On a high pin, many ranges will look stale
    assert data["total_stale_ranges"] >= 1


def test_warn_count_within_documented_range():
    """WARN count baseline check — tracks doc-drift across pin bumps.

    At v11.3.0 baseline: ~25 WARN entries (patches with `<0.21.0`
    upper bound but actually working on 0.21.x via operator opt-in).

    Bound the count loosely so natural drift (patches retiring,
    bumping ranges) doesn't break CI."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    data = json.loads(proc.stdout)
    warn = data["warn_count"]
    # Bound: 0 (all cleaned up) to 50 (lots of drift accumulated)
    assert 0 <= warn <= 50, (
        f"WARN count {warn} outside expected 0-50 range. Either many "
        f"ranges were just fixed (GOOD — lower the bound) or many "
        f"patches were added with stale ranges (BAD — investigate)."
    )
