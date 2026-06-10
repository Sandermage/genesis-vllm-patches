# SPDX-License-Identifier: Apache-2.0
"""Tests for the M.5.3 ``audit_model_config`` extraction.

Pins the operator-facing invariants:

  * ``audit_model_config(cfg)`` returns the same list as ``cfg.audit()``.
  * Wording of the soft-warning strings is preserved byte-for-byte
    relative to the pre-M.5.3 monolithic method.
  * Hard validation still raises :class:`SchemaError` with the same
    messages for the representative invalid-config cases that drove
    the validate() restructure into named helpers.
"""
from __future__ import annotations

import pytest

from sndr.model_configs import (
    HardwareSpec,
    ModelConfig,
    SchemaError,
)
from sndr.model_configs.model_config_audit import audit_model_config


def _minimal_cfg(**overrides) -> ModelConfig:
    """Construct a minimal-valid ModelConfig that ``validate()`` accepts.

    Override fields via kwargs to construct each test scenario without
    relying on YAML / builtin presets.
    """
    base = dict(
        key="test-cfg",
        title="t",
        description="d",
        schema_version=1,
        maintainer="m",
        model_path="/models/test",
        hardware=HardwareSpec(
            gpu_match_keys=["test"], n_gpus=1, min_vram_per_gpu_mib=1,
        ),
        lifecycle="stable",
    )
    base.update(overrides)
    return ModelConfig(**base)


# ─── audit() ↔ audit_model_config() parity ─────────────────────────────


class TestAuditParity:
    def test_method_equals_function_on_clean_config(self):
        """``stable`` lifecycle with reference_metrics absent → both
        method and standalone function surface the same warning."""
        cfg = _minimal_cfg()
        assert audit_model_config(cfg) == cfg.audit()
        # Sanity: the warning text contains the stable-lifecycle hint.
        assert any("reference_metrics" in w for w in cfg.audit())

    def test_method_equals_function_on_clean_experimental(self):
        """experimental lifecycle → no reference_metrics hint."""
        cfg = _minimal_cfg(lifecycle="experimental")
        assert audit_model_config(cfg) == cfg.audit()

    def test_p98_warning_text_byte_identical(self):
        """The TQ k8v4 + hybrid-GDN warning string is operator-visible
        and must remain stable across M.5.3."""
        cfg = _minimal_cfg(
            kv_cache_dtype="turboquant_k8v4",
            genesis_env={"GENESIS_ENABLE_PN59_STREAMING_GDN": "1"},
        )
        warnings = audit_model_config(cfg)
        # Expected message preserved verbatim from the pre-M.5.3
        # monolithic ModelConfig.audit() body.
        expected = (
            "P98 should be enabled for TQ k8v4 + hybrid GDN model "
            "(WorkspaceManager fix vs vllm#40941). "
            "Add GENESIS_ENABLE_P98=1 to genesis_env."
        )
        assert expected in warnings
        assert warnings == cfg.audit()

    def test_p98_warning_skipped_when_genesis_env_has_p98(self):
        cfg = _minimal_cfg(
            kv_cache_dtype="turboquant_k8v4",
            genesis_env={
                "GENESIS_ENABLE_PN59_STREAMING_GDN": "1",
                "GENESIS_ENABLE_P98": "1",
            },
        )
        warnings = audit_model_config(cfg)
        assert not any("P98 should be enabled" in w for w in warnings)


# ─── validate() restructure: error messages preserved ─────────────────


class TestValidateMessagesPreserved:
    def test_missing_key(self):
        cfg = _minimal_cfg(key="")
        with pytest.raises(SchemaError) as excinfo:
            cfg.validate()
        assert str(excinfo.value) == "ModelConfig.key required"

    def test_non_kebab_key(self):
        cfg = _minimal_cfg(key="Bad_Key")
        with pytest.raises(SchemaError) as excinfo:
            cfg.validate()
        assert (
            "ModelConfig.key must be kebab-case "
            "(lowercase letters/digits/hyphens), got 'Bad_Key'"
            in str(excinfo.value)
        )

    def test_wrong_schema_version(self):
        cfg = _minimal_cfg(schema_version=42)
        with pytest.raises(SchemaError) as excinfo:
            cfg.validate()
        assert "ModelConfig.schema_version must be 1" in str(excinfo.value)
        assert "(got 42)" in str(excinfo.value)

    def test_missing_title_description_maintainer(self):
        cfg = _minimal_cfg(title="")
        with pytest.raises(SchemaError) as excinfo:
            cfg.validate()
        assert str(excinfo.value) == (
            "ModelConfig requires title, description, maintainer"
        )

    def test_empty_model_path(self):
        cfg = _minimal_cfg(model_path="")
        with pytest.raises(SchemaError) as excinfo:
            cfg.validate()
        assert str(excinfo.value) == "ModelConfig.model_path required"

    def test_invalid_lifecycle(self):
        cfg = _minimal_cfg(lifecycle="bogus")
        with pytest.raises(SchemaError) as excinfo:
            cfg.validate()
        assert "ModelConfig.lifecycle must be one of" in str(excinfo.value)
        assert "(got 'bogus')" in str(excinfo.value)

    def test_invalid_cudagraph_mode(self):
        cfg = _minimal_cfg(cudagraph_mode="WRONG")
        with pytest.raises(SchemaError) as excinfo:
            cfg.validate()
        assert "ModelConfig.cudagraph_mode must be one of" in str(excinfo.value)
        assert "(got 'WRONG')" in str(excinfo.value)

    def test_community_submitted_requires_community_lifecycle(self):
        cfg = _minimal_cfg(community_submitted=True, lifecycle="stable")
        with pytest.raises(SchemaError) as excinfo:
            cfg.validate()
        assert "community_submitted=True requires lifecycle" in str(excinfo.value)


class TestValidateHelpersExposed:
    """The named ``_validate_*`` helpers are private but documented:
    asserting they exist + can be invoked directly catches accidental
    rename-without-test in future hygiene passes."""

    def test_helpers_callable(self):
        cfg = _minimal_cfg()
        # Every helper returns None on a valid cfg.
        assert cfg._validate_identity() is None
        assert cfg._validate_community_lifecycle() is None
        assert cfg._validate_cudagraph_mode() is None
        assert cfg._validate_compatibility_matrix() is None
