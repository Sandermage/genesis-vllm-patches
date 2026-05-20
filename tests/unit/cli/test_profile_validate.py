# SPDX-License-Identifier: Apache-2.0
"""P1.4 unit tests for `sndr profile validate`.

11 checks the validator runs:
  01  schema load + ProfileDef.validate
  02  parent_model exists + loads
  03  role enum
  04  spec_decode_override valid
  05  compression_plan kv_cache_dtype compatible with parent
  06  validation artifact JSON exists + parses
  07  artifact config_hash matches validation.config_hash
  08  intended_workloads ⊆ artifact.allowed_workloads (warning on diff)
  09  structured-role profile has non-empty effective_workloads
  10  role=default carries NO runtime-role blocks
  11  artifact decision not 'denied' / KERNEL_STORAGE_DTYPE_MISMATCH

Severity rules:
  ERROR    → bad enough to fail the profile under --strict
  WARNING  → operator should look but profile is functional
  INFO     → informational (none currently emitted)
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from vllm.sndr_core.cli.profile import (
    _SEV_ERROR,
    _SEV_WARNING,
    validate_profile,
)


# ─── Positive cases ─────────────────────────────────────────────────────


class TestPositiveCases:
    def test_gemma4_tq_default_validates_clean(self):
        issues, status = validate_profile("gemma4-tq-default")
        assert status == "ok", f"unexpected issues: {issues}"
        assert issues == []

    def test_gemma4_structured_k4_validates_clean(self):
        issues, status = validate_profile("gemma4-tq-mtp-structured-k4")
        assert status == "ok", f"unexpected issues: {issues}"
        assert issues == []

    def test_all_builtin_profiles_validate_clean(self):
        """Smoke: every builtin profile passes validate. Catches drift
        between schema additions and the validator."""
        from vllm.sndr_core.model_configs.registry_v2 import list_profiles
        for pid in list_profiles():
            issues, status = validate_profile(pid)
            errors = [i for i in issues if i["severity"] == _SEV_ERROR]
            assert not errors, (
                f"profile {pid!r} has ERRORs: {errors}"
            )


# ─── Negative case: artifact missing ────────────────────────────────────


class TestArtifactMissing:
    def test_structured_with_bad_artifact_id_fails(self, tmp_path, monkeypatch):
        """If a structured profile references a non-existent artifact,
        validate must emit ERROR on check 06."""
        # Patch _read_artifact to simulate artifact-missing for a specific
        # artifact id.
        from vllm.sndr_core.cli import profile as profile_cli

        original_read = profile_cli._read_artifact

        def fake_read(artifact_id):
            if artifact_id == "ghost-artifact":
                return None, "/synthetic does not exist"
            return original_read(artifact_id)

        # Build a synthetic profile referencing ghost-artifact
        from vllm.sndr_core.model_configs.schema_v2 import (
            ProfileDef, PatchesDelta, ValidationArtifactRef,
        )
        synthetic = ProfileDef(
            schema_version=2, kind="profile",
            id="gemma4-tq-mtp-structured-k4-ghost",
            parent_model="gemma-4-31b-it-awq",
            maintainer="tests",
            status="experimental",
            patches_delta=PatchesDelta(),
            role="structured",
            validation=ValidationArtifactRef(
                artifact_id="ghost-artifact",
                config_hash="71c874d7ffedae04",
            ),
        )
        synthetic.validate()  # schema-level passes

        from vllm.sndr_core.model_configs import registry_v2

        def fake_load(pid):
            if pid == "gemma4-tq-mtp-structured-k4-ghost":
                return synthetic
            return registry_v2.load_profile.__wrapped__(pid) if hasattr(
                registry_v2.load_profile, "__wrapped__"
            ) else registry_v2.load_profile(pid)

        monkeypatch.setattr(profile_cli, "_read_artifact", fake_read)
        monkeypatch.setattr(
            "vllm.sndr_core.model_configs.registry_v2.load_profile",
            fake_load,
        )

        issues, status = validate_profile("gemma4-tq-mtp-structured-k4-ghost")
        errors = [i for i in issues if i["severity"] == _SEV_ERROR]
        check_06 = [i for i in errors if i["check"] == "06_artifact_present"]
        assert check_06, (
            f"expected check 06_artifact_present ERROR; got {issues}"
        )
        assert status == "failed"


# ─── Negative case: config_hash mismatch ────────────────────────────────


class TestConfigHashMismatch:
    def test_hash_mismatch_fails(self, monkeypatch):
        """When validation.config_hash differs from the artifact's
        config_hash, validate must emit ERROR on check 07."""
        from vllm.sndr_core.cli import profile as profile_cli
        from vllm.sndr_core.model_configs.schema_v2 import (
            ProfileDef, PatchesDelta, ValidationArtifactRef,
        )

        synthetic = ProfileDef(
            schema_version=2, kind="profile",
            id="gemma4-tq-mtp-structured-k4-badhash",
            parent_model="gemma-4-31b-it-awq",
            maintainer="tests",
            status="experimental",
            patches_delta=PatchesDelta(),
            role="structured",
            validation=ValidationArtifactRef(
                artifact_id="gemma4-tq-mtp-structured-k4",
                config_hash="deadbeefcafebabe",  # wrong hash
            ),
        )

        def fake_load(pid):
            if pid == "gemma4-tq-mtp-structured-k4-badhash":
                return synthetic
            from vllm.sndr_core.model_configs import registry_v2 as real
            return real.load_profile(pid)

        monkeypatch.setattr(
            "vllm.sndr_core.model_configs.registry_v2.load_profile",
            fake_load,
        )

        issues, status = validate_profile("gemma4-tq-mtp-structured-k4-badhash")
        errors = [i for i in issues if i["severity"] == _SEV_ERROR]
        check_07 = [i for i in errors if i["check"] == "07_config_hash"]
        assert check_07, f"expected 07_config_hash ERROR; got {issues}"
        assert status == "failed"


# ─── Negative case: structured profile with denied intended workload ───


class TestWorkloadIntersection:
    def test_denied_intended_warns(self, monkeypatch):
        """When intended_workloads includes a class NOT in the
        artifact's allowed_workloads, validate emits WARNING on check
        08 (router will deny those classes; not fatal)."""
        from vllm.sndr_core.cli import profile as profile_cli
        from vllm.sndr_core.model_configs.schema_v2 import (
            ProfileDef, PatchesDelta, RoutingConfig, ValidationArtifactRef,
        )

        synthetic = ProfileDef(
            schema_version=2, kind="profile",
            id="gemma4-tq-mtp-structured-k4-denied-intent",
            parent_model="gemma-4-31b-it-awq",
            maintainer="tests",
            status="experimental",
            patches_delta=PatchesDelta(),
            role="structured",
            routing=RoutingConfig(
                intended_workloads=["tool_json", "code_gen"],
                # code_gen is in artifact.denied_workloads; tool_json allowed
            ),
            validation=ValidationArtifactRef(
                artifact_id="gemma4-tq-mtp-structured-k4",
                config_hash="71c874d7ffedae04",
            ),
        )

        def fake_load(pid):
            if pid == synthetic.id:
                return synthetic
            from vllm.sndr_core.model_configs import registry_v2 as real
            return real.load_profile(pid)

        monkeypatch.setattr(
            "vllm.sndr_core.model_configs.registry_v2.load_profile",
            fake_load,
        )

        issues, status = validate_profile(synthetic.id)
        warnings = [i for i in issues if i["severity"] == _SEV_WARNING]
        check_08 = [i for i in warnings if i["check"] == "08_intended_workloads"]
        assert check_08, f"expected 08 WARNING; got {issues}"
        # status should be warn (warnings only, no errors)
        assert status == "warn"

    def test_structured_with_empty_intersection_fails(self, monkeypatch):
        """structured role with intended_workloads ∩ allowed_workloads = {}
        must emit ERROR on check 09 (the router would route nothing
        through the structured upstream)."""
        from vllm.sndr_core.model_configs.schema_v2 import (
            ProfileDef, PatchesDelta, RoutingConfig, ValidationArtifactRef,
        )

        synthetic = ProfileDef(
            schema_version=2, kind="profile",
            id="gemma4-tq-mtp-structured-k4-empty-intersect",
            parent_model="gemma-4-31b-it-awq",
            maintainer="tests",
            status="experimental",
            patches_delta=PatchesDelta(),
            role="structured",
            routing=RoutingConfig(
                intended_workloads=["code_gen"],  # NOT in artifact.allowed
            ),
            validation=ValidationArtifactRef(
                artifact_id="gemma4-tq-mtp-structured-k4",
                config_hash="71c874d7ffedae04",
            ),
        )

        def fake_load(pid):
            if pid == synthetic.id:
                return synthetic
            from vllm.sndr_core.model_configs import registry_v2 as real
            return real.load_profile(pid)

        monkeypatch.setattr(
            "vllm.sndr_core.model_configs.registry_v2.load_profile",
            fake_load,
        )

        issues, status = validate_profile(synthetic.id)
        errors = [i for i in issues if i["severity"] == _SEV_ERROR]
        check_09 = [i for i in errors if i["check"] == "09_structured_effective_nonempty"]
        assert check_09, f"expected 09 ERROR; got {issues}"
        assert status == "failed"


# ─── Negative case: role=default carrying runtime-role blocks ──────────


class TestDefaultCleanContract:
    def test_default_with_spec_decode_override_fails(self, monkeypatch):
        from vllm.sndr_core.model_configs.schema import SpecDecodeConfig
        from vllm.sndr_core.model_configs.schema_v2 import (
            ProfileDef, PatchesDelta,
        )

        synthetic = ProfileDef(
            schema_version=2, kind="profile",
            id="gemma4-tq-default-dirty",
            parent_model="gemma-4-31b-it-awq",
            maintainer="tests",
            status="experimental",
            patches_delta=PatchesDelta(),
            role="default",
            spec_decode_override=SpecDecodeConfig(
                method="mtp", num_speculative_tokens=4,
            ),
        )

        def fake_load(pid):
            if pid == synthetic.id:
                return synthetic
            from vllm.sndr_core.model_configs import registry_v2 as real
            return real.load_profile(pid)

        monkeypatch.setattr(
            "vllm.sndr_core.model_configs.registry_v2.load_profile",
            fake_load,
        )

        issues, status = validate_profile(synthetic.id)
        errors = [i for i in issues if i["severity"] == _SEV_ERROR]
        check_10 = [i for i in errors if i["check"] == "10_default_clean"]
        assert check_10, f"expected 10 ERROR; got {issues}"
        assert status == "failed"


# ─── CLI exit code semantics ────────────────────────────────────────────


class TestExitCodes:
    def test_strict_zero_when_all_clean(self):
        """`sndr profile validate --strict` exits 0 when all profiles
        validate clean."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "vllm.sndr_core.cli",
             "profile", "validate", "--strict"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"expected exit 0; got {result.returncode}\nstderr:\n{result.stderr}"
        )

    def test_json_emits_valid_object(self):
        """--json output must be parseable + contain summary keys."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "vllm.sndr_core.cli",
             "profile", "validate", "gemma4-tq-mtp-structured-k4",
             "--json"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "profiles_checked" in data
        assert "ok" in data
        assert "errors" in data
        assert "warnings" in data
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["profile_id"] == "gemma4-tq-mtp-structured-k4"
        assert data["results"][0]["status"] == "ok"
