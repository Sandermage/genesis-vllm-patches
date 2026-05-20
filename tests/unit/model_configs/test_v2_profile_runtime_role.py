# SPDX-License-Identifier: Apache-2.0
"""P1.1 unit tests for the runtime-role extension on ProfileDef.

Six new optional fields on `ProfileDef`:

  role, spec_decode_override, compression_plan, backend_plan,
  routing, validation

Backward-compatibility contract: every existing builtin profile YAML
must continue to load+validate unchanged with the new fields defaulting
to None.

Forward-compatibility contract: when set, each new dataclass must
validate independently and reject obvious malformed inputs.

See: docs/_internal/SNDR_RUNTIME_PROFILES_DESIGN_DECISIONS_2026-05-20.md
"""
from __future__ import annotations

import pytest

from vllm.sndr_core.model_configs.registry_v2 import (
    list_profiles,
    load_profile,
)
from vllm.sndr_core.model_configs.schema import SchemaError, SpecDecodeConfig
from vllm.sndr_core.model_configs.schema_v2 import (
    BackendPlanConfig,
    CompressionPlanConfig,
    PROFILE_ROLES,
    ProfileDef,
    RoutingConfig,
    ValidationArtifactRef,
)


# ─── Backward compatibility: existing 17 (currently 15) profiles ────────


class TestExistingProfilesLoadUnchanged:
    # P1.3 (2026-05-20) added two builtin profiles that intentionally
    # set the new runtime-role fields. They are exempted here; any
    # OTHER profile setting role-related fields fails this test
    # (the per-field assertions below catch the leak).
    _RUNTIME_ROLE_EXEMPTIONS = frozenset({
        "gemma4-tq-default",                # role=default, no spec/compression/etc.
        "gemma4-tq-mtp-structured-k4",      # role=structured, full structured config
    })

    def test_all_builtin_profiles_load_with_new_fields_default_none(self):
        """Every builtin ProfileDef YAML must load + validate. Profiles
        with role=None (the tuning-preset majority) must have all six
        new optional fields = None. Profiles in
        ``_RUNTIME_ROLE_EXEMPTIONS`` may set role-related fields and
        are checked separately by the dedicated P1.3 / Variant A tests.

        If this fails after adding a new builtin YAML, the new YAML
        is using one of the runtime-role fields without being in the
        exemption list — verify intentionally, then add the new
        profile id to ``_RUNTIME_ROLE_EXEMPTIONS`` above.
        """
        ids = list_profiles()
        assert ids, "no profiles discovered — registry_v2 broken?"
        for pid in ids:
            p = load_profile(pid)
            p.validate()
            if pid in self._RUNTIME_ROLE_EXEMPTIONS:
                # Exempt: runtime-role profile; field shape verified
                # by dedicated P1.3 / Variant A tests in this file
                # (TestProfileDefRuntimeRole::test_full_structured_profile_validates,
                # test_default_profile_validates).
                continue
            assert p.role is None, (
                f"{pid}: role={p.role!r} — only profiles in "
                f"_RUNTIME_ROLE_EXEMPTIONS may set role. Add this id "
                f"to the exemption list if intentional."
            )
            assert p.spec_decode_override is None, (
                f"{pid}: spec_decode_override set"
            )
            assert p.compression_plan is None, f"{pid}: compression_plan set"
            assert p.backend_plan is None, f"{pid}: backend_plan set"
            assert p.routing is None, f"{pid}: routing set"
            assert p.validation is None, f"{pid}: validation set"

    def test_p1_3_builtin_gemma4_tq_default_loads(self):
        """gemma4-tq-default (P1.3): role=default, no runtime-role
        sub-blocks. Inherits ModelDef canonical patches as-is."""
        p = load_profile("gemma4-tq-default")
        p.validate()
        assert p.role == "default"
        assert p.parent_model == "gemma-4-31b-it-awq"
        assert p.spec_decode_override is None
        assert p.compression_plan is None
        assert p.backend_plan is None
        assert p.routing is None
        assert p.validation is None
        assert p.patches_delta.enable == {}
        assert p.patches_delta.disable == []
        assert p.patches_delta.override == {}

    def test_p1_3_builtin_gemma4_structured_k4_loads(self):
        """gemma4-tq-mtp-structured-k4 (P1.3): full structured config
        with MTP K=4, skip-list 58,59, artifact validation reference."""
        p = load_profile("gemma4-tq-mtp-structured-k4")
        p.validate()
        assert p.role == "structured"
        assert p.parent_model == "gemma-4-31b-it-awq"
        # Spec-decode K=4
        assert p.spec_decode_override is not None
        assert p.spec_decode_override.method == "mtp"
        assert p.spec_decode_override.num_speculative_tokens == 4
        # Compression plan
        assert p.compression_plan is not None
        assert p.compression_plan.native_source_layers == [58, 59]
        assert p.compression_plan.default_kv_dtype == "turboquant_4bit_nc"
        assert p.compression_plan.strategy == "per_layer"
        # Backend plan
        assert p.backend_plan is not None
        assert p.backend_plan.target_default == "TURBOQUANT"
        assert p.backend_plan.target_native_layers == "TRITON_ATTN"
        assert p.backend_plan.drafter_sliding == "TRITON_ATTN"
        assert p.backend_plan.drafter_full == "TRITON_ATTN"
        # Routing
        assert p.routing is not None
        assert set(p.routing.intended_workloads) == {
            "structured_count", "tool_json",
        }
        # Validation
        assert p.validation is not None
        assert p.validation.artifact_id == "gemma4-tq-mtp-structured-k4"
        assert p.validation.config_hash == "71c874d7ffedae04"
        # patches_delta.enable populated (P1.3 transitional — moves to
        # backend_plan in P1.5)
        assert "GENESIS_ENABLE_G4_71B_DRAFTER_SLIDING_TRITON" in p.patches_delta.enable
        assert "GENESIS_ENABLE_G4_75_DRAFTER_HEAD512_TRITON" in p.patches_delta.enable
        assert "SNDR_ALLOW_SPEC_DECODE_KV_ADAPTER" in p.patches_delta.enable

    def test_profile_roles_enum_unchanged(self):
        assert PROFILE_ROLES == ("default", "structured", "gateway")


# ─── CompressionPlanConfig ───────────────────────────────────────────────


class TestCompressionPlanConfig:
    def test_empty_is_valid(self):
        CompressionPlanConfig().validate()  # no raise

    def test_populated_is_valid(self):
        CompressionPlanConfig(
            native_source_layers=[58, 59],
            default_kv_dtype="turboquant_4bit_nc",
            strategy="per_layer",
        ).validate()

    def test_negative_layer_rejected(self):
        with pytest.raises(SchemaError, match="non-negative int"):
            CompressionPlanConfig(native_source_layers=[-1]).validate()

    def test_non_int_layer_rejected(self):
        with pytest.raises(SchemaError, match="non-negative int"):
            CompressionPlanConfig(native_source_layers=["58"]).validate()  # type: ignore[list-item]

    def test_duplicate_layer_rejected(self):
        with pytest.raises(SchemaError, match="duplicate"):
            CompressionPlanConfig(native_source_layers=[58, 58]).validate()

    def test_unsupported_strategy_rejected(self):
        with pytest.raises(SchemaError, match="per_layer"):
            CompressionPlanConfig(strategy="global").validate()  # type: ignore[arg-type]

    def test_non_str_dtype_rejected(self):
        with pytest.raises(SchemaError, match="str"):
            CompressionPlanConfig(default_kv_dtype=42).validate()  # type: ignore[arg-type]


# ─── BackendPlanConfig ────────────────────────────────────────────────────


class TestBackendPlanConfig:
    def test_empty_is_valid(self):
        BackendPlanConfig().validate()

    def test_populated_is_valid(self):
        BackendPlanConfig(
            target_default="TURBOQUANT",
            target_native_layers="TRITON_ATTN",
            drafter_sliding="TRITON_ATTN",
            drafter_full="TRITON_ATTN",
        ).validate()

    @pytest.mark.parametrize(
        "field_name",
        ["target_default", "target_native_layers", "drafter_sliding", "drafter_full"],
    )
    def test_empty_string_rejected(self, field_name):
        kwargs = {field_name: ""}
        with pytest.raises(SchemaError, match="non-empty str"):
            BackendPlanConfig(**kwargs).validate()


# ─── RoutingConfig ────────────────────────────────────────────────────────


class TestRoutingConfig:
    def test_empty_is_valid(self):
        RoutingConfig().validate()

    def test_populated_is_valid(self):
        RoutingConfig(
            intended_workloads=["structured_count", "tool_json"],
        ).validate()

    def test_empty_workload_rejected(self):
        with pytest.raises(SchemaError, match="non-empty str"):
            RoutingConfig(intended_workloads=[""]).validate()

    def test_duplicate_workload_rejected(self):
        with pytest.raises(SchemaError, match="duplicate"):
            RoutingConfig(intended_workloads=["a", "a"]).validate()


# ─── ValidationArtifactRef ───────────────────────────────────────────────


class TestValidationArtifactRef:
    def test_valid_short_hash(self):
        ValidationArtifactRef("g4-test", "71c874d7ffedae04").validate()

    def test_valid_long_hash(self):
        ValidationArtifactRef(
            "g4-test", "0123456789abcdef0123456789abcdef",
        ).validate()

    def test_non_hex_rejected(self):
        with pytest.raises(SchemaError, match="hex"):
            ValidationArtifactRef("g4-test", "not-hex!").validate()

    def test_empty_hash_rejected(self):
        with pytest.raises(SchemaError, match="non-empty"):
            ValidationArtifactRef("g4-test", "").validate()

    def test_id_format_validated(self):
        # _check_id reuses the ID regex — should reject uppercase
        with pytest.raises(SchemaError):
            ValidationArtifactRef("INVALID_ID", "00").validate()


# ─── ProfileDef integration with new fields ─────────────────────────────


def _bare_profile(**overrides) -> ProfileDef:
    """Helper: build a minimal valid ProfileDef with explicit overrides."""
    base = dict(
        schema_version=2,
        kind="profile",
        id="test-profile",
        parent_model="test-model",
        maintainer="tests",
        status="experimental",
    )
    base.update(overrides)
    return ProfileDef(**base)  # type: ignore[arg-type]


class TestProfileDefRuntimeRole:
    def test_role_none_default_validates(self):
        _bare_profile().validate()

    @pytest.mark.parametrize("role", ["default", "structured", "gateway"])
    def test_valid_role_accepted(self, role):
        _bare_profile(role=role).validate()

    def test_invalid_role_rejected(self):
        with pytest.raises(SchemaError, match="role"):
            _bare_profile(role="cutover").validate()  # type: ignore[arg-type]

    def test_spec_decode_override_reuses_v1_type(self):
        # The decision: spec_decode_override is the V1 SpecDecodeConfig
        # type, not a new wrapper. Verify that.
        p = _bare_profile(
            role="structured",
            spec_decode_override=SpecDecodeConfig(
                method="mtp", num_speculative_tokens=4,
            ),
        )
        p.validate()
        assert isinstance(p.spec_decode_override, SpecDecodeConfig)

    def test_full_structured_profile_validates(self):
        """End-to-end shape of what the gemma4-tq-mtp-structured-k4
        profile (P1.3) will look like."""
        p = _bare_profile(
            id="gemma4-tq-mtp-structured-k4",
            parent_model="gemma-4-31b-it-awq",
            status="validated",
            role="structured",
            spec_decode_override=SpecDecodeConfig(
                method="mtp", num_speculative_tokens=4,
            ),
            compression_plan=CompressionPlanConfig(
                native_source_layers=[58, 59],
                default_kv_dtype="turboquant_4bit_nc",
            ),
            backend_plan=BackendPlanConfig(
                target_default="TURBOQUANT",
                target_native_layers="TRITON_ATTN",
                drafter_sliding="TRITON_ATTN",
                drafter_full="TRITON_ATTN",
            ),
            routing=RoutingConfig(
                intended_workloads=["structured_count", "tool_json"],
            ),
            validation=ValidationArtifactRef(
                artifact_id="gemma4-tq-mtp-structured-k4",
                config_hash="71c874d7ffedae04",
            ),
        )
        p.validate()

    def test_default_profile_validates(self):
        """End-to-end shape of the default-role profile (P1.3)."""
        p = _bare_profile(
            id="gemma4-tq-default",
            parent_model="gemma-4-31b-it-awq",
            status="validated",
            role="default",
            # default role: no spec_decode, no compression_plan, no validation
        )
        p.validate()

    def test_existing_tuning_profile_still_validates(self):
        """Profiles with role=None continue to work as pure tuning
        presets (the existing 17 builtin shape)."""
        from vllm.sndr_core.model_configs.schema_v2 import HardwareSizing
        p = _bare_profile(
            id="35b-balanced",
            sizing_override=HardwareSizing(max_num_seqs=2),
        )
        p.validate()
        assert p.role is None
