# SPDX-License-Identifier: Apache-2.0
"""CONFIG-UX.4.1 — tests for `scripts/audit_v1_migration.py`.

Covers:
  - Stage 0 default exits 0 on current corpus (acceptance gate 1)
  - --strict exits 1 (acceptance gate 2)
  - Env override SNDR_V1_ROLLOUT_STAGE works (acceptance gate 3)
  - --stage CLI flag overrides env
  - JSON output shape
  - All 12 V1 keys present in migration table (acceptance gate 6)
  - Each V1 key resolves to its declared bucket
  - Per-stage severity matrix end-to-end
  - audit_no_new_v1.py behavior unchanged (acceptance gate 4 cross-check)
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "audit_v1_migration.py"


def _import_audit():
    spec = importlib.util.spec_from_file_location(
        "audit_v1_migration", SCRIPT,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audit_v1_migration"] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_cli(*args, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
        env=env,
    )


# ─── Live corpus ────────────────────────────────────────────────────────────


class TestLiveCorpus:
    def test_default_returns_zero(self):
        """Acceptance gate 1: Stage 0 default exits 0 on current corpus."""
        result = _run_cli()
        assert result.returncode == 0, (
            f"default mode should exit 0; got rc={result.returncode}\n"
            f"stdout={result.stdout[:500]}"
        )

    def test_strict_returns_one(self):
        """Acceptance gate 2: --strict exits 1 (current corpus has 9
        non-transparent V1 keys → 9 warnings → exit 1)."""
        result = _run_cli("--strict")
        assert result.returncode == 1

    def test_stage_3_returns_one(self):
        """Stage 3 default escalates non-transparent buckets to error → exit 1."""
        import os
        env = dict(os.environ)
        env["SNDR_V1_ROLLOUT_STAGE"] = "3"
        result = _run_cli(env=env)
        assert result.returncode == 1

    def test_stage_3_with_disable_env_still_returns_one(self):
        """Stage 3 ERROR severity is NOT silenced by
        GENESIS_DISABLE_V1_DEPRECATION_WARNING (operator escape hatch
        only silences WARN severity)."""
        import os
        env = dict(os.environ)
        env["SNDR_V1_ROLLOUT_STAGE"] = "3"
        env["GENESIS_DISABLE_V1_DEPRECATION_WARNING"] = "1"
        result = _run_cli(env=env)
        assert result.returncode == 1

    def test_explicit_stage_flag_overrides_env(self):
        """`--stage N` overrides env-driven stage."""
        import os
        env = dict(os.environ)
        env["SNDR_V1_ROLLOUT_STAGE"] = "3"  # would otherwise be error
        result = _run_cli("--stage", "0", env=env)
        assert result.returncode == 0  # stage 0 → warn only


# ─── Migration table contract ───────────────────────────────────────────────


class TestMigrationTable:
    def test_all_12_v1_keys_in_table(self):
        """Acceptance gate 6: all 12 V1 keys present in migration table."""
        mod = _import_audit()
        table = mod.load_migration_table()
        on_disk = set(mod.list_v1_keys_on_disk())
        assert len(on_disk) == 12
        missing = on_disk - set(table.keys())
        assert not missing, (
            f"V1 keys on disk but missing from migration table: {sorted(missing)}"
        )

    def test_table_entries_have_valid_bucket(self):
        mod = _import_audit()
        table = mod.load_migration_table()
        v1_buckets = {"transparent", "needs_operator_choice", "deprecated", "tombstone"}
        for key, entry in table.items():
            assert entry.bucket in v1_buckets, (
                f"entry {key!r}: bucket={entry.bucket!r} not in {v1_buckets}"
            )

    def test_transparent_entries_have_v2_preset(self):
        """transparent bucket means there IS a 1:1 V2 alias."""
        mod = _import_audit()
        table = mod.load_migration_table()
        transparent = [k for k, v in table.items() if v.bucket == "transparent"]
        for key in transparent:
            assert table[key].v2_preset, (
                f"transparent entry {key!r}: missing v2_preset"
            )

    def test_tombstone_bucket_empty(self):
        """Stage 1 ship: tombstone bucket starts empty per
        CONFIG-UX.4.R §10.5 operator decision."""
        mod = _import_audit()
        table = mod.load_migration_table()
        tombstones = [k for k, v in table.items() if v.bucket == "tombstone"]
        assert tombstones == [], (
            f"tombstone bucket should be empty at CONFIG-UX.4.1 ship; "
            f"got {tombstones}"
        )

    def test_no_table_entry_without_yaml_on_disk(self):
        """Defensive: every table entry corresponds to an actual V1 YAML
        (no stale entries). audit reports stale entries as INFO."""
        mod = _import_audit()
        table = mod.load_migration_table()
        on_disk = set(mod.list_v1_keys_on_disk())
        stale = set(table.keys()) - on_disk
        assert not stale, (
            f"migration table has stale entries with no YAML on disk: "
            f"{sorted(stale)}"
        )


# ─── Bucket distribution sanity ─────────────────────────────────────────────


class TestBucketDistribution:
    def test_expected_bucket_counts_at_stage_1_ship(self):
        """Locked bucket distribution. Snapshot of the 12-entry table:

          - 2026-05-24 (CONFIG-UX.4.R §3 ship): 3 transparent, 5 needs-choice,
            4 deprecated, 0 tombstone.
          - 2026-05-26 (V1-SUNSET-DFLASH-ALIAS.1): a5000-2x-27b-int4-tq-k8v4-dflash
            moved from needs_operator_choice → deprecated (+1 to deprecated).
            New distribution: 3 transparent, 4 needs-choice, 5 deprecated.
          - 2026-05-26 (V1-SUNSET-PENDING-4.1): last 4 needs_operator_choice
            entries reclassified to deprecated after .R confirmed V2 sizing
            parity (qa-27b-tq-1x, long-ctx-27b, qa-27b-tested, prod-35b-dflash).
            Stage 5 audit gate unblocked. Final distribution:
            3 transparent, 0 needs-choice, 9 deprecated, 0 tombstone.
        """
        mod = _import_audit()
        report = mod.run_audit(stage=0)
        counts = report.count_by_bucket()
        assert counts.get("transparent", 0) == 3
        assert counts.get("needs_operator_choice", 0) == 0
        assert counts.get("deprecated", 0) == 9
        assert counts.get("tombstone", 0) == 0


# ─── Per-stage severity ────────────────────────────────────────────────────


class TestSeverityPerStage:
    def test_stage_0_all_warn(self):
        """At Stage 0, non-tombstone buckets all emit warn (regardless of strict)."""
        mod = _import_audit()
        report = mod.run_audit(stage=0)
        # 12 warnings (all V1 keys at Stage 0 are warn).
        # Tombstone bucket is empty; no errors.
        counts = report.count_by_severity()
        assert counts.get("error", 0) == 0
        assert counts.get("warn", 0) == 12

    def test_stage_2_default_all_warn(self):
        mod = _import_audit()
        report = mod.run_audit(stage=2, strict_mode=False)
        counts = report.count_by_severity()
        assert counts.get("error", 0) == 0
        assert counts.get("warn", 0) == 12

    def test_stage_2_strict_non_transparent_error(self):
        """Stage 2 + strict: non-transparent buckets emit ERROR;
        transparent stays WARN."""
        mod = _import_audit()
        report = mod.run_audit(stage=2, strict_mode=True)
        counts = report.count_by_severity()
        # transparent (3) stay warn; needs_choice (5) + deprecated (4) → error
        assert counts.get("error", 0) == 9
        assert counts.get("warn", 0) == 3

    def test_stage_3_non_transparent_error(self):
        mod = _import_audit()
        report = mod.run_audit(stage=3)
        counts = report.count_by_severity()
        assert counts.get("error", 0) == 9
        assert counts.get("warn", 0) == 3


# ─── JSON output ────────────────────────────────────────────────────────────


class TestJSONOutput:
    def test_json_structure(self):
        from vllm.sndr_core.model_configs._rollout import DEFAULT_STAGE
        result = _run_cli("--json")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        for key in ("stage", "strict", "v1_keys_on_disk", "table_entries",
                    "counts", "bucket_distribution", "findings",
                    "has_errors", "has_warnings"):
            assert key in data, f"missing JSON key {key!r}"
        # CONFIG-UX.4.2 (2026-05-24): DEFAULT_STAGE flipped 0 → 1.
        # Operators reverting with SNDR_V1_ROLLOUT_STAGE=0 still see
        # functionally identical behavior for non-tombstone buckets.
        assert data["stage"] == DEFAULT_STAGE
        assert data["v1_keys_on_disk"] == 12
        assert data["table_entries"] == 12

    def test_json_finding_shape(self):
        result = _run_cli("--json")
        data = json.loads(result.stdout)
        for f in data["findings"]:
            assert set(f.keys()) >= {
                "v1_key", "bucket", "severity", "v2_preset", "rationale",
            }

    def test_json_stage_override(self):
        result = _run_cli("--json", "--stage", "3")
        data = json.loads(result.stdout)
        assert data["stage"] == 3
        assert data["has_errors"] is True


# ─── Backward-compat: audit_no_new_v1.py unchanged ──────────────────────────


class TestAuditNoNewV1Unchanged:
    def test_no_new_v1_still_clean(self):
        """Acceptance gate 4: audit_no_new_v1.py behavior unchanged."""
        result = subprocess.run(
            [sys.executable, "scripts/audit_no_new_v1.py"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, (
            f"audit_no_new_v1.py regressed: rc={result.returncode}\n"
            f"{result.stdout}"
        )
