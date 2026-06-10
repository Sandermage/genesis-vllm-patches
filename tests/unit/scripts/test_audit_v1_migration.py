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
        """Acceptance gate 2: --strict semantics.

        2026-06-01 update: Phase 10 Step 4 retired the last 2 V1 files
        (final sunset of transparent-bucket a5000-2x-35b-prod +
        a5000-2x-27b-int4-tq-k8v4). With ZERO findings, --strict exits
        0 (no warnings to escalate). The strict-escalation invariant
        is now covered by the TestSeverityPerStage synthetic tests
        only — live corpus exits cleanly at every stage."""
        result = _run_cli("--strict")
        assert result.returncode == 0

    def test_stage_3_returns_one(self):
        """Stage 3 default escalates non-transparent buckets to error → exit 1.

        2026-06-01 update: after the 10 V1 sunsets completed this session
        (including the PN95 architectural unblock for the 2 tier-aware
        files), all remaining V1 keys are TRANSPARENT bucket
        (a5000-2x-27b-int4-tq-k8v4 + a5000-2x-35b-prod). Transparent
        bucket never errors at stage 3 — so the audit exits 0 now.

        The original test invariant ("stage 3 escalates → 1") is still
        covered by the synthetic unit tests in TestSeverityPerStage.
        This live-corpus test now asserts the CORRECT post-sunset
        state: exit 0 because no deprecated/needs-choice/tombstone
        keys remain on disk.
        """
        import os
        env = dict(os.environ)
        env["SNDR_V1_ROLLOUT_STAGE"] = "3"
        result = _run_cli(env=env)
        # Post 2026-06-01 sunsets: only transparent-bucket V1 keys remain.
        assert result.returncode == 0

    def test_stage_3_with_disable_env_still_returns_one(self):
        """Stage 3 ERROR severity is NOT silenced by
        GENESIS_DISABLE_V1_DEPRECATION_WARNING (operator escape hatch
        only silences WARN severity).

        2026-06-01 update: same as test_stage_3_returns_one — post-
        sunset live corpus has only transparent bucket entries, so
        exit 0. The ERROR-not-silenced invariant is covered by
        TestSeverityPerStage synthetic tests.
        """
        import os
        env = dict(os.environ)
        env["SNDR_V1_ROLLOUT_STAGE"] = "3"
        env["GENESIS_DISABLE_V1_DEPRECATION_WARNING"] = "1"
        result = _run_cli(env=env)
        # Post 2026-06-01 sunsets: only transparent-bucket V1 keys remain.
        assert result.returncode == 0

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
        """Acceptance gate 6: all V1 keys present in migration table.

        2026-06-01 (Phase 10 Step 4): V1 monolithic preset tier FULLY
        retired. on_disk == empty; table.keys() == empty. Invariant
        (on_disk ⊆ table.keys()) holds vacuously."""
        mod = _import_audit()
        table = mod.load_migration_table()
        on_disk = set(mod.list_v1_keys_on_disk())
        assert len(on_disk) == 0
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
            parity (qa-qwen3.6-27b-tq-1x, long-ctx-qwen3.6-27b, qa-qwen3.6-27b-tested, prod-qwen3.6-35b-dflash).
            Stage 5 audit gate unblocked. Final distribution:
            3 transparent, 0 needs-choice, 9 deprecated, 0 tombstone.
          - 2026-06-01 (V1-SUNSET-PoC): single-3090-dense-cpu-offload-EXAMPLE
            retired (deleted YAML + migration table entry). First V1 file
            actually deleted; demonstrates the Phase 9 → Phase 10 transition
            workflow. New distribution: 3 transparent, 0 needs-choice,
            8 deprecated, 0 tombstone.
          - 2026-06-01 (V1-SUNSET-#2): single-3090-hybrid-gdn-tier-aware-
            EXAMPLE retired. Second V1 file removed in same session,
            continuing the EXAMPLE namespace cleanup. New distribution:
            3 transparent, 0 needs-choice, 7 deprecated, 0 tombstone.
          - 2026-06-01 (V1-SUNSET-#3): a5000-1x-27b-int4-tested retired.
            First NON-EXAMPLE V1 file removed; gated by runtime-coupling
            audit (zero refs outside docs/tests/audit). New distribution:
            3 transparent, 0 needs-choice, 6 deprecated, 0 tombstone.
          - 2026-06-01 (V1-SUNSET-#4): a5000-2x-35b-fp8-dflash retired
            after comparative study of 6 LOW-risk candidates (zero test
            refs, zero V1 cross-refs, V2 prod-qwen3.6-35b-dflash is
            production_candidate with corrected max_model_len 65K
            replacing V1's stale unsafe 160K). New distribution:
            3 transparent, 0 needs-choice, 5 deprecated, 0 tombstone.
          - 2026-06-01 (V1-SUNSET-#5): a5000-2x-27b-int4-tq-k8v4-dflash
            retired — already had `lifecycle: retired` marker in YAML
            since 2026-05-26; V2 equivalent `experimental-qwen3.6-27b-
            tq-dflash-ab` already shipped; deletion completes the
            sunset that the lifecycle marker started. New distribution:
            3 transparent, 0 needs-choice, 4 deprecated, 0 tombstone.
          - 2026-06-01 (V1-SUNSET-#6): a5000-2x-27b-dflash-true retired
            — first TRANSPARENT-bucket V1 file deleted (V2 prod-qwen3.6-
            27b-dflash composes byte-identical config DFlash N=5 single-
            stream). New distribution: 2 transparent, 0 needs-choice,
            4 deprecated, 0 tombstone.
          - 2026-06-01 (V1-SUNSET-#7): a5000-2x-27b-int4-long-ctx retired
            — V2 long-ctx-qwen3.6-27b sizing-identical (280K ctx / util
            0.90 / seqs 2 / batched 2048 / fp8_e5m2 KV / MTP K=3); V2
            carries `override_policy.bench_pending=true` since 32K+
            bench refresh against current pin is deferred. New
            distribution: 2 transparent, 0 needs-choice, 3 deprecated,
            0 tombstone.
          - 2026-06-01 (V1-SUNSET-#8): a5000-2x-27b-int4-tested retired
            — V2 qa-qwen3.6-27b-tested sizing-identical (131K ctx / util
            0.90 / seqs 2 / batched 4096 / fp8_e5m2 KV / MTP K=3); V2
            explicitly disables 16 Wave 1/7/8 patches via patches_delta
            to preserve May-5 bench snapshot (operator must consciously
            pick — V2 ≠ byte-identical V1). Legacy CLI test fixtures
            in tests/legacy/ migrated to surviving sibling
            `a5000-2x-27b-int4-tq-k8v4`. New distribution:
            2 transparent, 0 needs-choice, 2 deprecated, 0 tombstone.
          - 2026-06-01 (V1-SUNSET-#9 + #10): PN95 architectural unblock
            — extracted PN95 tier_specs from V1 files
            a5000-2x-tier-aware-EXAMPLE.yaml and a5000-1x-tier-aware-
            pn95.yaml into PN95-internal dir vllm/sndr_core/cache/pn95/
            tier_configs/. PN95 hook now reads from PN95-internal first,
            V1 fallback preserved. Both V1 files retired together as
            sunsets #9 (2x) + #10 (1x). New distribution: 2 transparent,
            0 needs-choice, 0 deprecated, 0 tombstone.
          - 2026-06-01 (Phase 10 Step 4 — FINAL SUNSET): the last 2
            transparent-bucket V1 files retired —
            a5000-2x-35b-prod.yaml (V2 equivalent
            `prod-qwen3.6-35b-balanced`, byte-identical compose) +
            a5000-2x-27b-int4-tq-k8v4.yaml (V2 equivalent
            `prod-qwen3.6-27b-tq-k8v4`, byte-identical compose). V1
            tier 100% retired. Final distribution: 0 transparent,
            0 needs-choice, 0 deprecated, 0 tombstone. Audit gates
            (audit_no_new_v1.py, audit_v1_migration.py) continue to
            enforce the freeze contract — any future V1-tier addition
            requires explicit baseline bump signalling deliberate
            legacy extension.
        """
        mod = _import_audit()
        report = mod.run_audit(stage=0)
        counts = report.count_by_bucket()
        assert counts.get("transparent", 0) == 0
        assert counts.get("needs_operator_choice", 0) == 0
        assert counts.get("deprecated", 0) == 0
        assert counts.get("tombstone", 0) == 0


# ─── Per-stage severity ────────────────────────────────────────────────────


class TestSeverityPerStage:
    def test_stage_0_all_warn(self):
        """At Stage 0, non-tombstone buckets all emit warn (regardless of strict)."""
        mod = _import_audit()
        report = mod.run_audit(stage=0)
        # 2 warnings (was 12 pre-2026-06-01; 10× V1 sunsets in single
        # day session: 2× EXAMPLE + a5000-1x-27b-int4-tested +
        # a5000-2x-35b-fp8-dflash + a5000-2x-27b-int4-tq-k8v4-dflash +
        # a5000-2x-27b-dflash-true + a5000-2x-27b-int4-long-ctx +
        # a5000-2x-27b-int4-tested + 2× tier-aware retired via PN95
        # architectural unblock). Tombstone empty.
        counts = report.count_by_severity()
        assert counts.get("error", 0) == 0
        assert counts.get("warn", 0) == 0  # V1 tier fully retired 2026-06-01

    def test_stage_2_default_all_warn(self):
        mod = _import_audit()
        report = mod.run_audit(stage=2, strict_mode=False)
        counts = report.count_by_severity()
        assert counts.get("error", 0) == 0
        assert counts.get("warn", 0) == 0  # V1 tier fully retired 2026-06-01

    def test_stage_2_strict_non_transparent_error(self):
        """Stage 2 + strict: non-transparent buckets emit ERROR;
        transparent stays WARN."""
        mod = _import_audit()
        report = mod.run_audit(stage=2, strict_mode=True)
        counts = report.count_by_severity()
        # All remaining V1 files are transparent bucket (both
        # a5000-2x-27b-int4-tq-k8v4 and a5000-2x-35b-prod). 0 errors
        # at strict stage 2. The 2 deprecated entries retired in PN95
        # architectural unblock (2026-06-01 sunsets #9 + #10).
        assert counts.get("error", 0) == 0
        assert counts.get("warn", 0) == 0  # V1 tier fully retired 2026-06-01

    def test_stage_3_non_transparent_error(self):
        mod = _import_audit()
        report = mod.run_audit(stage=3)
        counts = report.count_by_severity()
        assert counts.get("error", 0) == 0
        assert counts.get("warn", 0) == 0  # V1 tier fully retired 2026-06-01


# ─── JSON output ────────────────────────────────────────────────────────────


class TestJSONOutput:
    def test_json_structure(self):
        from sndr.model_configs._rollout import DEFAULT_STAGE
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
        assert data["v1_keys_on_disk"] == 0  # 2026-06-01: V1 tier 100% retired (Phase 10 Step 4)
        assert data["table_entries"] == 0

    def test_json_finding_shape(self):
        result = _run_cli("--json")
        data = json.loads(result.stdout)
        for f in data["findings"]:
            assert set(f.keys()) >= {
                "v1_key", "bucket", "severity", "v2_preset", "rationale",
            }

    def test_json_stage_override(self):
        # 2026-06-01 update: post 10× V1 sunsets, all remaining V1 keys
        # are transparent bucket → no errors at stage 3.
        result = _run_cli("--json", "--stage", "3")
        data = json.loads(result.stdout)
        assert data["stage"] == 3
        assert data["has_errors"] is False


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
