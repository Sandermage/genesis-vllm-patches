# SPDX-License-Identifier: Apache-2.0
"""CONFIG-UX.audit — tests for scripts/audit_override_policy.py.

Covers:
  - Default severity (warnings non-fatal) vs --strict (warnings fatal)
  - sizing_override without override_policy → warning at Stage 1
  - Invalid override_policy shape → error
  - effective_class=production requires reason + evidence
  - Non-production class requires reason
  - JSON output shape
  - FORBIDDEN_OVERRIDES placeholder is empty (no-op at Stage 1)
  - Live corpus: 21 builtin profiles must not produce errors at Stage 1
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "audit_override_policy.py"


def _import_audit():
    spec = importlib.util.spec_from_file_location(
        "audit_override_policy", SCRIPT,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audit_override_policy"] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
    )


# ─── Live corpus ────────────────────────────────────────────────────────────


class TestLiveCorpus:
    def test_default_severity_zero_on_live_corpus(self):
        """Stage 1: all 21 builtin profiles produce zero errors.
        Missing override_policy → warning, not error."""
        result = _run_cli()
        assert result.returncode == 0, (
            f"default mode should exit 0 at Stage 1; got rc={result.returncode}\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )

    def test_strict_returns_one_on_live_corpus(self):
        """--strict elevates Stage 1 warnings to fatal."""
        result = _run_cli("--strict")
        assert result.returncode == 1, (
            f"--strict should exit 1 on warnings; got rc={result.returncode}"
        )

    def test_no_errors_on_live_corpus(self):
        mod = _import_audit()
        report = mod.run_audit()
        errors = [f for f in report.findings if f.severity == "error"]
        assert errors == [], (
            f"Stage 1 acceptance violated — {len(errors)} error(s) on "
            f"builtin profiles:\n"
            + "\n".join(
                f"  [{f.rule}] {f.profile_id}: {f.message}" for f in errors
            )
        )

    def test_warnings_are_missing_policy_only_at_stage_1(self):
        """At Stage 1 we expect only `missing_override_policy` warnings
        on live corpus (existing profiles haven't been retrofitted yet)."""
        mod = _import_audit()
        report = mod.run_audit()
        warns = [f for f in report.findings if f.severity == "warning"]
        assert warns, "expected warnings for profiles with sizing_override but no policy"
        for f in warns:
            assert f.rule == "missing_override_policy", (
                f"unexpected warning rule {f.rule!r} on {f.profile_id!r}"
            )


# ─── JSON output shape ──────────────────────────────────────────────────────


class TestJSONOutput:
    def test_json_structure(self):
        result = _run_cli("--json")
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        for key in ("scanned", "counts", "findings", "has_errors", "has_warnings"):
            assert key in payload, f"missing key {key!r} in JSON output"
        assert isinstance(payload["findings"], list)
        assert payload["scanned"] >= 21

    def test_json_finding_shape(self):
        result = _run_cli("--json")
        payload = json.loads(result.stdout)
        for f in payload["findings"]:
            assert set(f.keys()) >= {"profile_id", "severity", "rule", "message"}
            assert f["severity"] in {"info", "warning", "error"}


# ─── Programmatic API: synthesised fixtures ─────────────────────────────────


def _make_profile(
    profile_id="test-profile",
    *,
    role=None,
    sizing_override=None,
    override_policy=None,
):
    """Construct a minimal ProfileDef for fixture purposes."""
    from vllm.sndr_core.model_configs.schema_v2 import (
        ProfileDef, PatchesDelta,
    )
    return ProfileDef(
        schema_version=2,
        kind="profile",
        id=profile_id,
        parent_model="some-model",
        maintainer="sander",
        patches_delta=PatchesDelta(),
        sizing_override=sizing_override,
        role=role,
        override_policy=override_policy,
    )


def _make_sizing(**kwargs):
    from vllm.sndr_core.model_configs.schema_v2 import HardwareSizing
    return HardwareSizing(**kwargs)


def _make_policy(**kwargs):
    from vllm.sndr_core.model_configs.schema_v2 import OverridePolicy
    return OverridePolicy(**kwargs)


class TestSynthesisedFindings:
    def test_sizing_without_policy_warns(self, monkeypatch):
        mod = _import_audit()
        profile = _make_profile(
            sizing_override=_make_sizing(max_num_seqs=8),
            override_policy=None,
        )

        def fake_load(pid):
            return profile

        monkeypatch.setattr(
            "vllm.sndr_core.model_configs.registry_v2.load_profile",
            fake_load,
        )
        report = mod.run_audit(profile_ids=["test-profile"])
        warns = [f for f in report.findings if f.severity == "warning"]
        assert any(f.rule == "missing_override_policy" for f in warns)
        assert not report.has_errors()

    def test_clean_profile_no_findings(self, monkeypatch):
        """No sizing_override and no override_policy → no findings."""
        mod = _import_audit()
        profile = _make_profile()

        def fake_load(pid):
            return profile

        monkeypatch.setattr(
            "vllm.sndr_core.model_configs.registry_v2.load_profile",
            fake_load,
        )
        report = mod.run_audit(profile_ids=["test-clean"])
        assert not report.findings

    def test_policy_without_sizing_warns(self, monkeypatch):
        mod = _import_audit()
        profile = _make_profile(
            sizing_override=None,
            override_policy=_make_policy(
                override_class="bench", reason="some reason",
            ),
        )

        def fake_load(pid):
            return profile

        monkeypatch.setattr(
            "vllm.sndr_core.model_configs.registry_v2.load_profile",
            fake_load,
        )
        report = mod.run_audit(profile_ids=["test-policy-only"])
        assert any(
            f.rule == "policy_without_sizing" and f.severity == "warning"
            for f in report.findings
        )

    def test_production_class_requires_reason(self, monkeypatch):
        """Production effective class derived from role=default → require reason."""
        mod = _import_audit()
        profile = _make_profile(
            role="default",
            sizing_override=_make_sizing(max_num_seqs=8),
            override_policy=_make_policy(reason=None, evidence_refs=[]),
        )

        def fake_load(pid):
            return profile

        monkeypatch.setattr(
            "vllm.sndr_core.model_configs.registry_v2.load_profile",
            fake_load,
        )
        report = mod.run_audit(profile_ids=["test-prod-no-reason"])
        errors = [f for f in report.findings if f.severity == "error"]
        assert any(f.rule == "missing_reason" for f in errors)

    def test_production_class_requires_evidence(self, monkeypatch):
        mod = _import_audit()
        profile = _make_profile(
            role="default",
            sizing_override=_make_sizing(max_num_seqs=8),
            override_policy=_make_policy(
                reason="bench-driven", evidence_refs=[],
            ),
        )

        def fake_load(pid):
            return profile

        monkeypatch.setattr(
            "vllm.sndr_core.model_configs.registry_v2.load_profile",
            fake_load,
        )
        report = mod.run_audit(profile_ids=["test-prod-no-evidence"])
        errors = [f for f in report.findings if f.severity == "error"]
        assert any(f.rule == "production_missing_evidence" for f in errors)

    def test_production_class_complete_clean(self, monkeypatch):
        mod = _import_audit()
        profile = _make_profile(
            role="default",
            sizing_override=_make_sizing(max_num_seqs=8),
            override_policy=_make_policy(
                override_class="production",
                reason="bench-validated +1.7% TPS at max_num_seqs=8",
                evidence_refs=["tests/integration/baselines/some.json"],
                validated_by="sander",
                validated_at="2026-05-24",
            ),
        )

        def fake_load(pid):
            return profile

        monkeypatch.setattr(
            "vllm.sndr_core.model_configs.registry_v2.load_profile",
            fake_load,
        )
        report = mod.run_audit(profile_ids=["test-prod-complete"])
        assert not report.has_errors()

    def test_bench_class_requires_reason(self, monkeypatch):
        mod = _import_audit()
        profile = _make_profile(
            role="bench",
            sizing_override=_make_sizing(max_num_seqs=8),
            override_policy=_make_policy(reason=None),
        )

        def fake_load(pid):
            return profile

        monkeypatch.setattr(
            "vllm.sndr_core.model_configs.registry_v2.load_profile",
            fake_load,
        )
        report = mod.run_audit(profile_ids=["test-bench-no-reason"])
        errors = [f for f in report.findings if f.severity == "error"]
        assert any(f.rule == "missing_reason" for f in errors)

    def test_bench_class_no_evidence_required(self, monkeypatch):
        """Non-production classes only need `reason`; evidence_refs not
        mandatory at Stage 1."""
        mod = _import_audit()
        profile = _make_profile(
            role="bench",
            sizing_override=_make_sizing(max_num_seqs=8),
            override_policy=_make_policy(
                reason="A/B comparator setup", evidence_refs=[],
            ),
        )

        def fake_load(pid):
            return profile

        monkeypatch.setattr(
            "vllm.sndr_core.model_configs.registry_v2.load_profile",
            fake_load,
        )
        report = mod.run_audit(profile_ids=["test-bench-no-evidence"])
        assert not report.has_errors()

    def test_safe_per_launch_no_reason_required(self, monkeypatch):
        """Class=safe_per_launch is explicit no-policy-needed; reason
        optional."""
        mod = _import_audit()
        profile = _make_profile(
            role="default",
            sizing_override=_make_sizing(max_num_seqs=8),
            override_policy=_make_policy(
                override_class="safe_per_launch", reason=None,
            ),
        )

        def fake_load(pid):
            return profile

        monkeypatch.setattr(
            "vllm.sndr_core.model_configs.registry_v2.load_profile",
            fake_load,
        )
        report = mod.run_audit(profile_ids=["test-safe-per-launch"])
        # Safe-per-launch should not require reason. May still have
        # other warnings (none expected here).
        assert not any(f.rule == "missing_reason" for f in report.findings)

    def test_role_none_derives_production_class(self, monkeypatch):
        """role=None → effective_class='production' → requires reason+evidence."""
        mod = _import_audit()
        profile = _make_profile(
            role=None,
            sizing_override=_make_sizing(max_num_seqs=8),
            override_policy=_make_policy(reason=None),
        )

        def fake_load(pid):
            return profile

        monkeypatch.setattr(
            "vllm.sndr_core.model_configs.registry_v2.load_profile",
            fake_load,
        )
        report = mod.run_audit(profile_ids=["test-role-none"])
        errors = [f for f in report.findings if f.severity == "error"]
        # missing_reason fires for non-safe class, regardless of explicit
        # class. Role=None derives to "production".
        assert any(f.rule == "missing_reason" for f in errors)

    def test_explicit_class_overrides_role_derivation(self, monkeypatch):
        """Explicit override_class wins over role-based derivation."""
        mod = _import_audit()
        profile = _make_profile(
            role="default",  # would derive to "production"
            sizing_override=_make_sizing(max_num_seqs=8),
            override_policy=_make_policy(
                override_class="bench",  # explicit
                reason="A/B comparator",
                evidence_refs=[],  # bench class doesn't need evidence
            ),
        )

        def fake_load(pid):
            return profile

        monkeypatch.setattr(
            "vllm.sndr_core.model_configs.registry_v2.load_profile",
            fake_load,
        )
        report = mod.run_audit(profile_ids=["test-explicit-bench"])
        assert not report.has_errors()


# ─── Module-level constants ─────────────────────────────────────────────────


class TestForbiddenPlaceholder:
    def test_forbidden_overrides_populated_at_stage_2(self):
        """CONFIG-UX.4.2 populates FORBIDDEN_OVERRIDES with 4 rules.

        Detailed per-rule coverage lives in
        `test_audit_override_policy_class4.py`; this test just verifies
        the registration shape (right count, right rule ids).
        """
        mod = _import_audit()
        assert len(mod.FORBIDDEN_OVERRIDES) == 4
        rule_ids = {r.rule_id for r in mod.FORBIDDEN_OVERRIDES}
        assert rule_ids == {
            "gpu_memory_utilization_over_1",
            "tensor_parallel_size_over_hw_gpus",
            "kv_cache_dtype_downgrade",
            "spec_decode_method_change",
        }


# ─── Severity gate behavior ─────────────────────────────────────────────────


class TestSeverityGate:
    def test_default_zero_with_warnings_only(self, monkeypatch):
        mod = _import_audit()
        profile = _make_profile(
            sizing_override=_make_sizing(max_num_seqs=8),
            override_policy=None,
        )

        def fake_load(pid):
            return profile

        monkeypatch.setattr(
            "vllm.sndr_core.model_configs.registry_v2.load_profile",
            fake_load,
        )
        report = mod.run_audit(profile_ids=["a", "b"])
        assert report.has_warnings()
        assert not report.has_errors()

    def test_invalid_profile_id_returns_one(self):
        """Out-of-corpus profile id → schema_load error → rc=1."""
        result = _run_cli("--profile", "does-not-exist-anywhere")
        assert result.returncode == 1
