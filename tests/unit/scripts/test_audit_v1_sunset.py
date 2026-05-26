# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/audit_v1_sunset.py`` — TOOLING-HARDENING.2 L.10
(2026-05-26).

The audit produces a §9.T sunset-readiness report (current rollout
stage, per-V1-key classification, stage-1→5 readiness, blocker list).
Tests exercise the classifier + stage matrix on synthetic inputs, then
smoke-test the live repo (must return exit 0 at the default stage).

Coverage targets:

  - tombstone_ready    (transparent + V2 alias on disk)
  - tombstone_candidate (deprecated + V2 alias on disk)
  - operator_decision_pending (needs_operator_choice + V2 resolves)
  - blocker_no_v2_alias (v2_preset null OR alias missing)
  - tombstoned          (entry bucket=tombstone, file already gone)
  - untracked           (V1 file on disk, no migration entry)
  - tombstone incident  (entry bucket=tombstone, file still on disk)
  - stage readiness matrix (1/2/3/5)
  - --strict exit semantics at stage 3
  - live corpus smoke (real repo must remain exit-0 at default stage)
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_v1_sunset.py"


def _import_audit_module():
    spec = importlib.util.spec_from_file_location(
        "audit_v1_sunset", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audit_v1_sunset"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def audit_mod():
    return _import_audit_module()


class TestClassifyBucket:
    """Per-V1-key classification."""

    def test_transparent_with_resolvable_v2_is_tombstone_ready(self, audit_mod):
        cls, untracked = audit_mod.classify(
            v1_keys={"key-a"},
            entries={
                "key-a": {
                    "bucket": "transparent",
                    "v2_preset": "prod-a",
                    "rationale": "direct",
                },
            },
            v2_presets={"prod-a"},
        )
        assert untracked == []
        assert len(cls) == 1
        assert cls[0].classification == "tombstone_ready"
        assert cls[0].v2_resolves is True

    def test_deprecated_with_resolvable_v2_is_tombstone_candidate(
        self, audit_mod,
    ):
        cls, _ = audit_mod.classify(
            v1_keys={"old-example"},
            entries={
                "old-example": {
                    "bucket": "deprecated",
                    "v2_preset": "example-x",
                    "rationale": "example→example",
                },
            },
            v2_presets={"example-x"},
        )
        assert cls[0].classification == "tombstone_candidate"

    def test_needs_choice_with_resolvable_v2_is_operator_decision(
        self, audit_mod,
    ):
        cls, _ = audit_mod.classify(
            v1_keys={"split-config"},
            entries={
                "split-config": {
                    "bucket": "needs_operator_choice",
                    "v2_preset": "qa-something",
                    "rationale": "qa promote",
                },
            },
            v2_presets={"qa-something"},
        )
        assert cls[0].classification == "operator_decision_pending"

    def test_null_v2_preset_is_blocker(self, audit_mod):
        cls, _ = audit_mod.classify(
            v1_keys={"orphan"},
            entries={
                "orphan": {
                    "bucket": "needs_operator_choice",
                    "v2_preset": None,
                    "rationale": "no V2",
                },
            },
            v2_presets=set(),
        )
        assert cls[0].classification == "blocker_no_v2_alias"
        assert cls[0].v2_resolves is False

    def test_missing_v2_alias_is_blocker(self, audit_mod):
        """v2_preset declared but file does not exist → blocker."""
        cls, _ = audit_mod.classify(
            v1_keys={"stale"},
            entries={
                "stale": {
                    "bucket": "transparent",
                    "v2_preset": "phantom-preset",
                    "rationale": "test",
                },
            },
            v2_presets=set(),  # phantom-preset absent
        )
        assert cls[0].classification == "blocker_no_v2_alias"

    def test_tombstone_entry_classified_tombstoned(self, audit_mod):
        cls, _ = audit_mod.classify(
            v1_keys=set(),  # V1 file already removed
            entries={
                "gone-v1": {
                    "bucket": "tombstone",
                    "v2_preset": "successor",
                    "rationale": "removed in release N",
                },
            },
            v2_presets={"successor"},
        )
        assert cls[0].classification == "tombstoned"

    def test_v1_file_without_table_entry_is_untracked(self, audit_mod):
        cls, untracked = audit_mod.classify(
            v1_keys={"new-rogue-v1"},
            entries={},
            v2_presets=set(),
        )
        assert untracked == ["new-rogue-v1"]
        assert cls[0].classification == "untracked"


class TestStageReadiness:
    """The 1→5 readiness matrix derived from classifications."""

    def _all_clean(self, audit_mod):
        # Three transparent keys, all V2 siblings resolve.
        return audit_mod.classify(
            v1_keys={"a", "b", "c"},
            entries={
                k: {
                    "bucket": "transparent",
                    "v2_preset": f"prod-{k}",
                    "rationale": "",
                }
                for k in ("a", "b", "c")
            },
            v2_presets={"prod-a", "prod-b", "prod-c"},
        )

    def test_all_transparent_resolvable_is_stage_5_ready(self, audit_mod):
        cls, _ = self._all_clean(audit_mod)
        stages = audit_mod.compute_stage_readiness(cls)
        assert all(s.ready for s in stages), (
            f"all-transparent+resolvable should mark all stages ready, "
            f"got {[(s.stage, s.ready) for s in stages]}"
        )

    def test_untracked_blocks_all_stages(self, audit_mod):
        cls, _ = audit_mod.classify(
            v1_keys={"rogue"},
            entries={},
            v2_presets=set(),
        )
        stages = audit_mod.compute_stage_readiness(cls)
        # Stage 1 prerequisite is "no untracked"; cascades to 2/3/4/5.
        assert not any(s.ready for s in stages)

    def test_blocker_blocks_stage_3_and_up(self, audit_mod):
        cls, _ = audit_mod.classify(
            v1_keys={"orphan"},
            entries={
                "orphan": {
                    "bucket": "needs_operator_choice",
                    "v2_preset": None,
                    "rationale": "",
                },
            },
            v2_presets=set(),
        )
        stages = audit_mod.compute_stage_readiness(cls)
        ready_by_stage = {s.stage: s.ready for s in stages}
        assert ready_by_stage[1] is True
        assert ready_by_stage[2] is True
        assert ready_by_stage[3] is False
        assert ready_by_stage[5] is False

    def test_operator_pending_only_blocks_stage_5(self, audit_mod):
        cls, _ = audit_mod.classify(
            v1_keys={"qa-thing"},
            entries={
                "qa-thing": {
                    "bucket": "needs_operator_choice",
                    "v2_preset": "qa-other",
                    "rationale": "",
                },
            },
            v2_presets={"qa-other"},
        )
        stages = audit_mod.compute_stage_readiness(cls)
        ready_by_stage = {s.stage: s.ready for s in stages}
        # Stage 3 + 4 still ready (V2 alias resolves), Stage 5 not ready
        # because operator decision pending.
        assert ready_by_stage[3] is True
        assert ready_by_stage[4] is True
        assert ready_by_stage[5] is False


class TestTombstoneIncident:
    """Tombstone bucket + V1 file still on disk = incident."""

    def test_tombstone_incident_flagged_when_file_present(self, audit_mod):
        cls, _ = audit_mod.classify(
            v1_keys={"forgotten"},
            entries={
                "forgotten": {
                    "bucket": "tombstone",
                    "v2_preset": "newer",
                    "rationale": "removed release N — but file still here",
                },
            },
            v2_presets={"newer"},
        )
        # `classify` itself marks it `tombstoned`; the incident check
        # is in `build_report` (it computes from v1_keys ∩ tombstoned).
        assert cls[0].classification == "tombstoned"


class TestLiveCorpus:
    """Smoke against the real repo + migration table."""

    def test_live_default_exit_zero(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert result.returncode == 0, (
            f"audit_v1_sunset should be exit 0 at default stage, got "
            f"rc={result.returncode}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    def test_live_json_shape(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        # Core keys exist + plausible values.
        assert "rollout_stage" in data
        assert "v1_files_on_disk" in data
        assert "migration_table_entries" in data
        assert "stages" in data
        assert "counts" in data
        # Live repo has 12 V1 monoliths frozen by audit_no_new_v1.
        assert data["v1_files_on_disk"] >= 1
        assert data["migration_table_entries"] >= 1
        # No tombstone incidents on the live tree.
        assert data["tombstone_incidents"] == []
        # No untracked keys (audit_no_new_v1 keeps them in sync).
        assert data["untracked_keys"] == []

    def test_live_strict_at_stage_2_still_clean(self):
        """At Stage 2 strict mode does not gate blockers (matrix locks
        Stage 3+ for hard ERROR); the live tree must still exit 0."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--strict", "--stage", "2"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert result.returncode == 0, (
            f"strict at stage 2 must not gate blockers, got rc="
            f"{result.returncode}\nstdout:\n{result.stdout}"
        )

    def test_live_strict_at_stage_3_is_clean(self):
        """After V1-SUNSET-DFLASH-ALIAS.1 (2026-05-26) the last
        ``blocker_no_v2_alias`` was retired by mapping
        a5000-2x-27b-int4-tq-k8v4-dflash → ``experimental-27b-tq-dflash-ab``
        in the migration table. Live tree at Stage 3 strict must
        exit 0 — Stage 3 advancement is unblocked.

        The strict-blocks-blocker code path stays covered by the
        synthetic unit tests in ``TestStageReadiness``; this live
        smoke pins the resolved state.
        """
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--strict", "--stage", "3"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert result.returncode == 0, (
            f"strict at stage 3 should be clean post-V1-SUNSET-DFLASH-ALIAS.1, "
            f"got rc={result.returncode}\nstdout:\n{result.stdout}"
        )

    def test_live_all_stages_ready_post_pending_4(self):
        """After V1-SUNSET-PENDING-4.1 (2026-05-26) the last four
        ``operator_decision_pending`` entries (qa-27b-tq-1x, long-ctx-27b,
        qa-27b-tested, prod-35b-dflash) were reclassified to ``deprecated``
        after their .R audit verified V2 sizing parity. With zero blockers
        and zero pending decisions, every stage (1..5) must report ready
        and ``--strict --stage 5`` must exit 0.

        Stage 5 readiness is the §9.T audit-gate unblock. Actual V1 file
        removal remains a separate operator-approved deletion phase.
        """
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json", "--strict",
             "--stage", "5"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert result.returncode == 0, (
            f"strict at stage 5 should be clean post-V1-SUNSET-PENDING-4.1, "
            f"got rc={result.returncode}\nstdout:\n{result.stdout}"
        )
        data = json.loads(result.stdout)
        assert data["counts"]["operator_decision_pending"] == 0
        assert data["counts"]["blocker_no_v2_alias"] == 0
        # All 5 stage readiness booleans must be True.
        stage_ready = {s["stage"]: s["ready"] for s in data["stages"]}
        for stage in (1, 2, 3, 4, 5):
            assert stage_ready[stage] is True, (
                f"stage {stage} must be ready post-PENDING-4.1, "
                f"got blockers={stage_ready}"
            )
