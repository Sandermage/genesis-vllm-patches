# SPDX-License-Identifier: Apache-2.0
"""CONFIG-UX.5.1 — tests for `scripts/generate_config_catalog.py`."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "generate_config_catalog.py"
BUILTIN = REPO_ROOT / "sndr" / "model_configs" / "builtin"
BASELINES = REPO_ROOT / "tests" / "integration" / "baselines"


def _expected_row_counts() -> dict:
    """Derive expected per-type row counts from live source filesystem
    so the test does not freeze on counter drift (e.g. when a new
    preset/baseline lands)."""
    return {
        "preset": len(list((BUILTIN / "presets").glob("*.yaml"))),
        "profile": len(list((BUILTIN / "profile").glob("*.yaml"))),
        "model": len(list((BUILTIN / "model").glob("*.yaml"))),
        "hardware": len(list((BUILTIN / "hardware").glob("*.yaml"))),
        "baseline": len(list(BASELINES.glob("*.json"))),
    }


def _import_gen():
    spec = importlib.util.spec_from_file_location("generate_config_catalog", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["generate_config_catalog"] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )


# ─── Determinism ───────────────────────────────────────────────────────────


class TestDeterminism:
    def test_stdout_runs_produce_same_drift_stripped_output(self):
        """Two consecutive --stdout runs must produce identical
        drift-stripped output."""
        mod = _import_gen()
        r1 = mod.build_catalog()
        r2 = mod.build_catalog()
        s1 = mod.serialise_for_drift(r1)
        s2 = mod.serialise_for_drift(r2)
        assert s1 == s2, "non-deterministic output between two consecutive runs"

    def test_check_mode_exits_zero(self):
        """`--check` runs the generator twice and exits 0 on determinism."""
        result = _run_cli("--check")
        assert result.returncode == 0, (
            f"--check should exit 0; got rc={result.returncode}\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )

    def test_row_count_stable(self):
        """Row counts per type match live source filesystem; total is
        sum of types. Parametric so it survives addition/removal of
        baseline files, presets, etc."""
        mod = _import_gen()
        rows = mod.build_catalog()
        from collections import Counter
        types = Counter(r["row_type"] for r in rows)
        expected = _expected_row_counts()
        for row_type, count in expected.items():
            assert types[row_type] == count, (
                f"row_type={row_type!r}: catalog has {types[row_type]}, "
                f"filesystem has {count}"
            )
        assert len(rows) == sum(expected.values())


# ─── Redaction enforcement ─────────────────────────────────────────────────


class TestRedactionEnforcement:
    def test_no_sndr_private_in_output(self):
        result = _run_cli("--stdout")
        assert result.returncode == 0
        assert "sndr_private/" not in result.stdout, (
            "redaction leak: 'sndr_private/' appears in generated catalog"
        )

    def test_no_local_absolute_paths_in_output(self):
        result = _run_cli("--stdout")
        for banned in ("/Users/", "/home/", "/tmp/", "/var/"):
            assert banned not in result.stdout, (
                f"redaction leak: {banned!r} appears in generated catalog"
            )

    def test_private_evidence_refs_redacted_with_marker(self):
        """Profiles/presets carrying private evidence get
        `{redacted: true, ...}` markers in their card_evidence_refs."""
        mod = _import_gen()
        rows = mod.build_catalog()
        target = next(
            (r for r in rows
             if r["row_type"] == "preset"
             and r["id"] == "prod-gemma4-26b-multiconc"),
            None,
        )
        assert target is not None, "expected prod-gemma4-26b-multiconc in catalog"
        ev_refs = target["card_evidence_refs"]
        assert len(ev_refs) > 0
        for ref in ev_refs:
            # The fixture preset has private bench refs → must be redacted
            assert ref.get("redacted") is True, (
                f"private evidence_ref not redacted: {ref}"
            )
            assert "path" not in ref or "sndr_private/" not in str(ref.get("path", "")), (
                f"redacted ref still leaks path: {ref}"
            )

    def test_public_evidence_refs_kept_verbatim(self):
        """Public refs in repo (e.g. `tests/integration/baselines/...`)
        appear in the catalog unredacted."""
        mod = _import_gen()
        rows = mod.build_catalog()
        target = next(
            (r for r in rows
             if r["row_type"] == "preset" and r["id"] == "prod-qwen3.6-35b-balanced"),
            None,
        )
        assert target is not None
        ev_refs = target["card_evidence_refs"]
        # prod-qwen3.6-35b-balanced cites tests/integration/baselines/35b_v11_wave9.json (public)
        assert any(
            ref.get("path", "").startswith("tests/integration/baselines/")
            for ref in ev_refs
        )


# ─── Schema completeness ────────────────────────────────────────────────────


class TestSchemaCompleteness:
    def test_preset_row_required_fields(self):
        """Operator-locked PresetRow fields per CONFIG-UX.5.R §2.3."""
        mod = _import_gen()
        rows = mod.build_catalog()
        preset_rows = [r for r in rows if r["row_type"] == "preset"]
        assert preset_rows
        for r in preset_rows:
            for required in (
                "schema_version", "row_type", "id", "source_path",
                "source_sha256", "status", "family", "tags",
                "updated_from_git_commit", "generated_at",
                "model_id", "hardware_id", "profile_id",
                "composed_key", "composed_sha256", "has_card",
                "card_workload_allow", "card_workload_deny",
                "card_primary_metric_kind", "card_primary_metric_value",
                "card_fallback_preset", "card_default_for_family",
            ):
                assert required in r, (
                    f"preset row {r['id']!r} missing required field {required!r}"
                )

    def test_profile_row_required_fields(self):
        mod = _import_gen()
        rows = mod.build_catalog()
        profile_rows = [r for r in rows if r["row_type"] == "profile"]
        assert profile_rows
        for r in profile_rows:
            for required in (
                "has_override_policy", "override_class", "override_reason",
                "override_expires_at", "override_allowed_to_exceed_hardware_default",
                "class4_clean",
                "sizing_max_model_len", "sizing_max_num_seqs",
                "sizing_gpu_memory_utilization",
            ):
                assert required in r, (
                    f"profile row {r['id']!r} missing required field {required!r}"
                )

    def test_baseline_row_match_quality(self):
        mod = _import_gen()
        rows = mod.build_catalog()
        baseline_rows = [r for r in rows if r["row_type"] == "baseline"]
        assert baseline_rows
        for r in baseline_rows:
            assert r["match_quality"] in (
                "exact_preset", "model_only", "family_only", "none",
            )


# ─── Class-4 clean field accuracy ───────────────────────────────────────────


class TestClass4CleanField:
    def test_all_profiles_class4_clean_post_debt_closure(self):
        """Post CONFIG-UX.4.DEBT closure, every profile in catalog
        should report `class4_clean: true`."""
        mod = _import_gen()
        rows = mod.build_catalog()
        profile_rows = [r for r in rows if r["row_type"] == "profile"]
        unclean = [r["id"] for r in profile_rows if not r["class4_clean"]]
        assert unclean == [], (
            f"profiles with class4_clean=false (Class-4 violations): {unclean}. "
            f"Should be 0 post-DEBT closure."
        )


# ─── JSON output shape ──────────────────────────────────────────────────────


class TestJSONOutput:
    def test_stdout_is_valid_json(self):
        result = _run_cli("--stdout")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["schema_version"] == 1
        assert isinstance(data["rows"], list)
        assert data["row_count"] == len(data["rows"])
        assert data["row_count"] == sum(_expected_row_counts().values())

    def test_rows_sorted_by_type_then_id(self):
        result = _run_cli("--stdout")
        data = json.loads(result.stdout)
        keys = [(r["row_type"], r["id"]) for r in data["rows"]]
        assert keys == sorted(keys)


# ─── Generator API ──────────────────────────────────────────────────────────


class TestGeneratorAPI:
    def test_build_catalog_accepts_pinned_generated_at(self):
        """Tests can pin generated_at to avoid timestamp churn."""
        mod = _import_gen()
        rows = mod.build_catalog(generated_at="2026-05-24T00:00:00Z")
        assert all(r["generated_at"] == "2026-05-24T00:00:00Z" for r in rows)
