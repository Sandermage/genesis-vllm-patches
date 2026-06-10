# SPDX-License-Identifier: Apache-2.0
"""CONFIG-UX.audit — tests for scripts/audit_config_catalog.py.

Covers:
  - Default severity (warnings non-fatal) vs --strict (warnings fatal)
  - Card-less presets → warning at Stage 1
  - Production-grade card validation → error
  - Non-production card validation → permissive (no error)
  - fallback_preset resolution
  - default_for_family uniqueness (collision detection)
  - evidence_refs[].path existence (filesystem + external://)
  - JSON output shape
  - production_candidate + private-only evidence → warning
  - Live corpus: 21 builtin presets must not produce errors at Stage 1
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import warnings
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "audit_config_catalog.py"


def _import_audit():
    spec = importlib.util.spec_from_file_location(
        "audit_config_catalog", SCRIPT,
    )
    mod = importlib.util.module_from_spec(spec)
    # Dataclass decoration on python 3.13 reads sys.modules[cls.__module__]
    # during _is_type resolution; register before exec to avoid
    # AttributeError on NoneType.__dict__.
    sys.modules["audit_config_catalog"] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
    )


# ─── Live corpus (21 builtin presets) ───────────────────────────────────────


class TestLiveCorpus:
    def test_default_severity_returns_zero_on_live_corpus(self):
        """All 21 builtin presets must produce zero errors at Stage 1.

        Card-less presets emit warnings (expected), so default mode
        must exit 0.
        """
        result = _run_cli()
        assert result.returncode == 0, (
            f"default mode should exit 0 on Stage 1 warnings; got "
            f"rc={result.returncode}\nstdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )

    def test_strict_severity_returns_one_on_live_corpus(self):
        """--strict elevates Stage 1 warnings to fatal."""
        result = _run_cli("--strict")
        assert result.returncode == 1, (
            f"--strict should exit 1 on warnings; got rc={result.returncode}\n"
            f"stdout={result.stdout}"
        )

    def test_no_errors_on_live_corpus(self):
        """Stage 1 acceptance: zero errors on the live builtin preset corpus.

        Errors mean schema/path/fallback/family violations; those must
        be zero before CONFIG-UX.2 starts mass annotation.
        """
        mod = _import_audit()
        report = mod.run_audit()
        errors = [f for f in report.findings if f.severity == "error"]
        assert errors == [], (
            f"Stage 1 acceptance violated — {len(errors)} error(s) on "
            f"builtin presets:\n"
            + "\n".join(f"  [{f.rule}] {f.preset_id}: {f.message}" for f in errors)
        )

    def test_warnings_expected_at_stage_1(self):
        """Stage 1 (post-CONFIG-UX.2): two warning rules expected.

        - `missing_card` — for the 7 non-prod-* presets that CONFIG-UX.2b
          will annotate (example-*, qa-*, experimental-*, long-ctx-*).
        - `production_candidate_public_evidence` — for annotated prod-*
          presets whose evidence is currently private-only (gemma4 and
          qwen3.6-35b-dflash variants without public baselines).

        Any other warning rule is unexpected and should be investigated.
        """
        mod = _import_audit()
        report = mod.run_audit()
        warnings_list = [f for f in report.findings if f.severity == "warning"]
        assert warnings_list, "expected at least some warnings"
        allowed = {"missing_card", "production_candidate_public_evidence"}
        for f in warnings_list:
            assert f.rule in allowed, (
                f"unexpected warning rule {f.rule!r} on {f.preset_id!r}; "
                f"only {sorted(allowed)} expected at Stage 1"
            )


# ─── JSON output shape ──────────────────────────────────────────────────────


class TestJSONOutput:
    def test_json_structure(self):
        result = _run_cli("--json")
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert "scanned" in payload
        assert "counts" in payload
        assert "findings" in payload
        assert "has_errors" in payload
        assert "has_warnings" in payload
        assert isinstance(payload["findings"], list)
        assert payload["scanned"] >= 21

    def test_json_finding_shape(self):
        result = _run_cli("--json")
        payload = json.loads(result.stdout)
        for f in payload["findings"]:
            assert set(f.keys()) >= {"preset_id", "severity", "rule", "message"}
            assert f["severity"] in {"info", "warning", "error"}


# ─── Programmatic API: synthesised fixtures ─────────────────────────────────


class TestSynthesisedFindings:
    """Test the audit logic by patching the loader to return crafted
    PresetDef objects with specific card states."""

    def test_card_less_preset_emits_missing_card_warning(self, monkeypatch):
        mod = _import_audit()
        from sndr.model_configs.preset_schema import PresetDef

        def fake_load(alias):
            return PresetDef(id=alias, model="m", hardware="h")

        monkeypatch.setattr(
            "sndr.model_configs.registry_v2.load_preset_def",
            fake_load,
        )
        report = mod.run_audit(preset_ids=["fake-preset"])
        warnings_list = [f for f in report.findings if f.severity == "warning"]
        assert any(f.rule == "missing_card" for f in warnings_list)
        assert not report.has_errors()

    def test_non_production_card_validates_permissively(self, monkeypatch):
        mod = _import_audit()
        from sndr.model_configs.preset_schema import (
            PresetCard, PresetDef,
        )

        def fake_load(alias):
            return PresetDef(
                id=alias, model="m", hardware="h",
                card=PresetCard(
                    title="x", summary="y", status="experimental",
                ),
            )

        monkeypatch.setattr(
            "sndr.model_configs.registry_v2.load_preset_def",
            fake_load,
        )
        report = mod.run_audit(preset_ids=["fake-experimental"])
        # No errors for experimental, no missing_card warning (card present).
        assert not report.has_errors()
        assert not any(f.rule == "missing_card" for f in report.findings)

    def test_production_card_missing_required_fields_errors(self, monkeypatch):
        mod = _import_audit()
        from sndr.model_configs.preset_schema import (
            PresetCard, PresetDef,
        )

        def fake_load(alias):
            return PresetDef(
                id=alias, model="m", hardware="h",
                card=PresetCard(
                    title="x", summary="y", status="production",
                    # all production-required fields missing
                ),
            )

        monkeypatch.setattr(
            "sndr.model_configs.registry_v2.load_preset_def",
            fake_load,
        )
        report = mod.run_audit(preset_ids=["fake-production"])
        errors = [f for f in report.findings if f.severity == "error"]
        assert errors, "production card missing required fields should error"
        assert any(
            f.rule == "card_strict_validation" for f in errors
        ), "strict validation errors should carry rule=card_strict_validation"

    def test_fallback_preset_missing_errors(self, monkeypatch):
        mod = _import_audit()
        from sndr.model_configs.preset_schema import (
            PresetCard, PresetDef,
        )

        def fake_load(alias):
            return PresetDef(
                id=alias, model="m", hardware="h",
                card=PresetCard(
                    title="x", summary="y", status="experimental",
                    fallback_preset="nonexistent-preset",
                ),
            )

        monkeypatch.setattr(
            "sndr.model_configs.registry_v2.load_preset_def",
            fake_load,
        )
        report = mod.run_audit(preset_ids=["fake-bad-fallback"])
        errors = [f for f in report.findings if f.severity == "error"]
        assert any(f.rule == "fallback_resolution" for f in errors)

    def test_fallback_preset_resolves_no_error(self, monkeypatch):
        mod = _import_audit()
        from sndr.model_configs.preset_schema import (
            PresetCard, PresetDef,
        )

        def fake_load(alias):
            return PresetDef(
                id=alias, model="m", hardware="h",
                card=PresetCard(
                    title="x", summary="y", status="experimental",
                    fallback_preset="other-preset",
                ),
            )

        monkeypatch.setattr(
            "sndr.model_configs.registry_v2.load_preset_def",
            fake_load,
        )
        # Pass both preset_ids so the fallback resolves.
        report = mod.run_audit(preset_ids=["fake-good", "other-preset"])
        errors = [
            f for f in report.findings
            if f.severity == "error" and f.rule == "fallback_resolution"
        ]
        assert not errors

    def test_default_for_family_collision_errors_both(self, monkeypatch):
        mod = _import_audit()
        from sndr.model_configs.preset_schema import (
            PresetCard, PresetDef,
        )

        cards = {
            "preset-a": PresetDef(
                id="preset-a", model="m", hardware="h",
                card=PresetCard(
                    title="A", summary="x", status="experimental",
                    routing_family="family-x", default_for_family=True,
                ),
            ),
            "preset-b": PresetDef(
                id="preset-b", model="m", hardware="h",
                card=PresetCard(
                    title="B", summary="y", status="experimental",
                    routing_family="family-x", default_for_family=True,
                ),
            ),
        }

        def fake_load(alias):
            return cards[alias]

        monkeypatch.setattr(
            "sndr.model_configs.registry_v2.load_preset_def",
            fake_load,
        )
        report = mod.run_audit(preset_ids=["preset-a", "preset-b"])
        coll = [f for f in report.findings if f.rule == "default_for_family_collision"]
        # Both presets receive the collision finding (so operator sees
        # the conflict from either entry's perspective).
        assert len(coll) == 2

    def test_default_for_family_single_no_error(self, monkeypatch):
        mod = _import_audit()
        from sndr.model_configs.preset_schema import (
            PresetCard, PresetDef,
        )

        def fake_load(alias):
            return PresetDef(
                id=alias, model="m", hardware="h",
                card=PresetCard(
                    title=alias, summary="x", status="experimental",
                    routing_family="family-x", default_for_family=True,
                ),
            )

        monkeypatch.setattr(
            "sndr.model_configs.registry_v2.load_preset_def",
            fake_load,
        )
        report = mod.run_audit(preset_ids=["solo-default"])
        assert not any(
            f.rule == "default_for_family_collision" for f in report.findings
        )

    def test_evidence_external_url_accepted(self, monkeypatch):
        mod = _import_audit()
        from sndr.model_configs.preset_schema import (
            EvidenceRef, PresetCard, PresetDef,
        )

        def fake_load(alias):
            return PresetDef(
                id=alias, model="m", hardware="h",
                card=PresetCard(
                    title="x", summary="y", status="experimental",
                    evidence_refs=[
                        EvidenceRef(type="bench", path="external://x.example/y"),
                    ],
                ),
            )

        monkeypatch.setattr(
            "sndr.model_configs.registry_v2.load_preset_def",
            fake_load,
        )
        report = mod.run_audit(preset_ids=["fake-external"])
        assert not any(f.rule == "evidence_path" for f in report.findings)

    def test_evidence_relative_path_missing_errors(self, monkeypatch):
        mod = _import_audit()
        from sndr.model_configs.preset_schema import (
            EvidenceRef, PresetCard, PresetDef,
        )

        def fake_load(alias):
            return PresetDef(
                id=alias, model="m", hardware="h",
                card=PresetCard(
                    title="x", summary="y", status="experimental",
                    evidence_refs=[
                        EvidenceRef(type="bench", path="does/not/exist.json"),
                    ],
                ),
            )

        monkeypatch.setattr(
            "sndr.model_configs.registry_v2.load_preset_def",
            fake_load,
        )
        report = mod.run_audit(preset_ids=["fake-missing-path"])
        assert any(
            f.rule == "evidence_path" and f.severity == "error"
            for f in report.findings
        )

    def test_evidence_relative_path_present_passes(self, monkeypatch, tmp_path):
        """Use an existing repo file as the evidence path."""
        mod = _import_audit()
        from sndr.model_configs.preset_schema import (
            EvidenceRef, PresetCard, PresetDef,
        )

        # README.md is guaranteed to exist at repo root
        def fake_load(alias):
            return PresetDef(
                id=alias, model="m", hardware="h",
                card=PresetCard(
                    title="x", summary="y", status="experimental",
                    evidence_refs=[
                        EvidenceRef(type="bench", path="README.md"),
                    ],
                ),
            )

        monkeypatch.setattr(
            "sndr.model_configs.registry_v2.load_preset_def",
            fake_load,
        )
        report = mod.run_audit(preset_ids=["fake-good-path"])
        assert not any(f.rule == "evidence_path" for f in report.findings)

    def test_production_candidate_private_only_evidence_warns(self, monkeypatch):
        mod = _import_audit()
        from sndr.model_configs.preset_schema import (
            EvidenceRef, PresetCard, PresetDef, ConcurrencyEnvelope,
            PrimaryMetric,
        )

        def fake_load(alias):
            return PresetDef(
                id=alias, model="m", hardware="h",
                card=PresetCard(
                    title="x", summary="y",
                    status="production_candidate",
                    audience="operator",
                    mode="throughput",
                    workload_allow=["w1"], workload_deny=["w2"],
                    K=1,
                    routing_family="rf",
                    concurrency=ConcurrencyEnvelope(min=1, canonical=1, max=1),
                    primary_metric=PrimaryMetric(
                        kind="agg_TPS", value=1.0, source="x",
                    ),
                    evidence_refs=[
                        EvidenceRef(
                            type="bench", path="external://x/y",
                            visibility="private",
                        ),
                    ],
                    evidence_visibility="private",
                ),
            )

        monkeypatch.setattr(
            "sndr.model_configs.registry_v2.load_preset_def",
            fake_load,
        )
        report = mod.run_audit(preset_ids=["fake-prod-cand"])
        warns = [
            f for f in report.findings
            if f.rule == "production_candidate_public_evidence"
        ]
        assert warns, "expected warning for prod-candidate without public evidence"
        # Production_candidate-with-public-evidence-missing is warning only,
        # not error (per audit phase scope).
        assert all(f.severity == "warning" for f in warns)


# ─── Severity gate behavior ─────────────────────────────────────────────────


class TestSeverityGate:
    def test_default_zero_with_warnings_only(self, monkeypatch):
        mod = _import_audit()
        from sndr.model_configs.preset_schema import PresetDef

        def fake_load(alias):
            return PresetDef(id=alias, model="m", hardware="h")

        monkeypatch.setattr(
            "sndr.model_configs.registry_v2.load_preset_def",
            fake_load,
        )
        report = mod.run_audit(preset_ids=["a", "b"])
        assert report.has_warnings()
        assert not report.has_errors()

    def test_invalid_preset_id_returns_two(self):
        """Out-of-corpus preset id → schema_load error, but rc=1 (error),
        not 2 (internal). Internal error rc=2 reserved for usage/IO."""
        result = _run_cli("--preset", "does-not-exist-anywhere")
        # `load_preset_def` will raise SchemaError → schema_load error finding
        assert result.returncode == 1, (
            f"missing preset id should be error (rc=1), got rc={result.returncode}\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
