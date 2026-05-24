# SPDX-License-Identifier: Apache-2.0
"""P1.2 unit tests for compose() handling of ProfileDef runtime-role fields.

Gates from the operator GO message:

  G1 — structured profile compose emits skip-list env (BOTH
       SNDR_G4_TQ_FORCE_SKIP_LAYERS canonical and
       GENESIS_G4_TQ_FORCE_SKIP_LAYERS legacy alias, until the
       reader migrates)
  G2 — default profile (no compression_plan, no spec_decode_override)
       emits no MTP/spec env additions
  G3 — existing 15 builtin profiles compose identically before/after
       the P1.2 extension (no behavior change on the prior tuning
       presets)

Plus arithmetic + edge case coverage.
"""
from __future__ import annotations

import pytest

from vllm.sndr_core.model_configs.compose import (
    compose,
    render_compression_env,
)
from vllm.sndr_core.model_configs.registry_v2 import (
    list_profiles,
    load_hardware,
    load_model,
    load_profile,
)
from vllm.sndr_core.model_configs.schema import (
    HardwareSpec,
    SchemaError,
    SpecDecodeConfig,
)
from vllm.sndr_core.model_configs.schema_v2 import (
    BackendPlanConfig,
    CompressionPlanConfig,
    HardwareDef,
    HardwareSizing,
    ModelCapabilities,
    ModelDef,
    ModelRequires,
    ModelVersions,
    PatchesDelta,
    ProfileDef,
    RoutingConfig,
    RuntimeBlock,
    RuntimeBareMetalBlock,
    ValidationArtifactRef,
)


# ─── Fixtures: synthetic ModelDef + HardwareDef for tests ───────────────


def _make_model(
    *,
    model_id: str = "test-model",
    kv_dtype: str = "turboquant_4bit_nc",
    spec_decode: SpecDecodeConfig | None = None,
) -> ModelDef:
    return ModelDef(
        schema_version=2,
        kind="model",
        id=model_id,
        title=f"{model_id} (test)",
        maintainer="tests",
        last_validated="2026-05-20",
        license="apache-2.0",
        model_path=f"/models/{model_id}",
        served_model_name=model_id,
        quantization=None,
        dtype="bfloat16",
        trust_remote_code=True,
        chat_template=None,
        capabilities=ModelCapabilities(
            attention_arch="dense",
            tool_call_parser=None,
            reasoning_parser=None,
            enable_auto_tool_choice=False,
            spec_decode=spec_decode,
            kv_cache_dtype=kv_dtype,
        ),
        requires=ModelRequires(
            min_total_vram_mib=20000,
            min_gpu_count=1,
            min_cuda_capability="8.6",
            rig_arch_blocklist=[],
        ),
        versions=ModelVersions(
            genesis_pin_min="11.0.0",
            vllm_pin_required="0.20.2",
        ),
        patches={"GENESIS_ENABLE_TEST_BASE": "1"},
    )


def _make_hardware() -> HardwareDef:
    return HardwareDef(
        schema_version=2,
        kind="hardware",
        id="test-rig",
        title="test rig 2xA5000",
        maintainer="tests",
        hardware=HardwareSpec(
            gpu_match_keys=("A5000",),
            n_gpus=2,
            min_vram_per_gpu_mib=24000,
            cuda_capability_min="8.6",
        ),
        sizing=HardwareSizing(
            max_model_len=4096,
            gpu_memory_utilization=0.9,
            max_num_seqs=1,
            max_num_batched_tokens=8192,
            enable_chunked_prefill=True,
            enforce_eager=False,
            disable_custom_all_reduce=True,
        ),
        runtime=RuntimeBlock(
            default="bare-metal",
            supported=["bare-metal"],
            docker=None,
            bare_metal=RuntimeBareMetalBlock(),
        ),
        system_env={},
    )


def _make_profile(
    *,
    parent_model: str = "test-model",
    role: str | None = None,
    spec_decode_override: SpecDecodeConfig | None = None,
    compression_plan: CompressionPlanConfig | None = None,
    backend_plan: BackendPlanConfig | None = None,
    routing: RoutingConfig | None = None,
    validation: ValidationArtifactRef | None = None,
    patches_delta_enable: dict[str, str] | None = None,
) -> ProfileDef:
    return ProfileDef(
        schema_version=2,
        kind="profile",
        id="test-profile",
        parent_model=parent_model,
        maintainer="tests",
        status="experimental",
        patches_delta=PatchesDelta(enable=patches_delta_enable or {}),
        role=role,  # type: ignore[arg-type]
        spec_decode_override=spec_decode_override,
        compression_plan=compression_plan,
        backend_plan=backend_plan,
        routing=routing,
        validation=validation,
    )


# ─── render_compression_env helper ───────────────────────────────────────


class TestRenderCompressionEnv:
    def test_none_profile_returns_empty(self):
        assert render_compression_env(None) == {}

    def test_no_compression_plan_returns_empty(self):
        assert render_compression_env(_make_profile()) == {}

    def test_empty_layers_returns_empty(self):
        prof = _make_profile(compression_plan=CompressionPlanConfig())
        assert render_compression_env(prof) == {}

    def test_populated_layers_emits_both_envs(self):
        prof = _make_profile(
            compression_plan=CompressionPlanConfig(
                native_source_layers=[58, 59],
            ),
        )
        env = render_compression_env(prof)
        assert env == {
            "SNDR_G4_TQ_FORCE_SKIP_LAYERS": "58,59",
            "GENESIS_G4_TQ_FORCE_SKIP_LAYERS": "58,59",
        }

    def test_single_layer_csv_format(self):
        prof = _make_profile(
            compression_plan=CompressionPlanConfig(
                native_source_layers=[42],
            ),
        )
        env = render_compression_env(prof)
        assert env["SNDR_G4_TQ_FORCE_SKIP_LAYERS"] == "42"

    def test_layer_order_preserved(self):
        # Order matters for diagnostic clarity even though the reader
        # treats it as a set.
        prof = _make_profile(
            compression_plan=CompressionPlanConfig(
                native_source_layers=[59, 58, 7],
            ),
        )
        env = render_compression_env(prof)
        assert env["SNDR_G4_TQ_FORCE_SKIP_LAYERS"] == "59,58,7"


# ─── Gate G1 — structured profile emits skip-list env ───────────────────


class TestStructuredProfileEmitsSkipList:
    def test_structured_role_with_compression_plan(self):
        model = _make_model()
        hardware = _make_hardware()
        profile = _make_profile(
            role="structured",
            spec_decode_override=SpecDecodeConfig(
                method="mtp", num_speculative_tokens=4,
            ),
            compression_plan=CompressionPlanConfig(
                native_source_layers=[58, 59],
                default_kv_dtype="turboquant_4bit_nc",
            ),
        )
        cfg = compose(model, hardware, profile)
        env = cfg.genesis_env
        assert env["SNDR_G4_TQ_FORCE_SKIP_LAYERS"] == "58,59"
        assert env["GENESIS_G4_TQ_FORCE_SKIP_LAYERS"] == "58,59"
        # Model's canonical patches survive
        assert env["GENESIS_ENABLE_TEST_BASE"] == "1"

    def test_spec_decode_override_propagates(self):
        model = _make_model(spec_decode=None)
        hardware = _make_hardware()
        spec = SpecDecodeConfig(method="mtp", num_speculative_tokens=4)
        profile = _make_profile(
            role="structured",
            spec_decode_override=spec,
        )
        cfg = compose(model, hardware, profile)
        assert cfg.spec_decode is spec  # same instance
        assert cfg.spec_decode.num_speculative_tokens == 4  # type: ignore[union-attr]

    def test_operator_patches_delta_wins_over_compression_env(self):
        """If the operator explicitly sets SNDR_G4_TQ_FORCE_SKIP_LAYERS
        via patches_delta.enable, it must NOT be silently overwritten
        by compose(). Operator intent wins."""
        model = _make_model()
        hardware = _make_hardware()
        profile = _make_profile(
            role="structured",
            patches_delta_enable={
                "GENESIS_G4_TQ_FORCE_SKIP_LAYERS": "1,2,3",
                # Note: SNDR_ canonical is NOT set; only the legacy one.
            },
            compression_plan=CompressionPlanConfig(
                native_source_layers=[58, 59],
            ),
        )
        cfg = compose(model, hardware, profile)
        # Operator's GENESIS value wins (preserved)
        assert cfg.genesis_env["GENESIS_G4_TQ_FORCE_SKIP_LAYERS"] == "1,2,3"
        # SNDR_ canonical from compression_plan fills in since unset
        assert cfg.genesis_env["SNDR_G4_TQ_FORCE_SKIP_LAYERS"] == "58,59"


# ─── Gate G2 — default profile emits no MTP/spec env ───────────────────


class TestDefaultProfileEmitsNoMTPEnv:
    def test_default_role_no_spec_decode_override(self):
        """A role=default profile with no spec_decode_override and no
        compression_plan must not add any spec-decode / skip-layer envs."""
        model = _make_model(spec_decode=None)
        hardware = _make_hardware()
        profile = _make_profile(role="default")
        cfg = compose(model, hardware, profile)
        assert cfg.spec_decode is None  # inherits model's None
        assert "SNDR_G4_TQ_FORCE_SKIP_LAYERS" not in cfg.genesis_env
        assert "GENESIS_G4_TQ_FORCE_SKIP_LAYERS" not in cfg.genesis_env
        # Model's canonical patches still survive
        assert cfg.genesis_env["GENESIS_ENABLE_TEST_BASE"] == "1"


# ─── Gate G3 — existing profiles compose unchanged ─────────────────────


class TestExistingProfilesComposeUnchanged:
    """Every builtin profile with role=None must compose to the same
    V1 ModelConfig before and after P1.2 (no behavior change on the
    prior 15 tuning-only presets).

    Operationally enforced by: all 15 profile YAMLs set role=None
    (verified in test_v2_profile_runtime_role.py), and compose's new
    code paths only fire when role / spec_decode_override / compression_plan
    are non-None. This test serves as the integration guard.
    """

    def test_all_existing_profiles_compose_without_compression_env(self):
        """For every builtin profile compose with a synthetic matching
        model + hardware; confirm no new compression env is added to
        the composed genesis_env. The structured-role envs must NOT
        appear unless the profile explicitly opts in via compression_plan."""
        for pid in list_profiles():
            profile = load_profile(pid)
            if profile.role is not None:
                continue  # P1.3 profiles can have role set, skip
            # Use the synthetic test model+hardware. compose's
            # check_compat will only care that parent_model matches.
            # We use the profile's parent_model with a synthetic model
            # that matches.
            model = _make_model(model_id=profile.parent_model)
            hardware = _make_hardware()
            cfg = compose(model, hardware, profile)
            assert "SNDR_G4_TQ_FORCE_SKIP_LAYERS" not in cfg.genesis_env, (
                f"{pid}: SNDR_G4_TQ_FORCE_SKIP_LAYERS leaked"
            )
            assert "GENESIS_G4_TQ_FORCE_SKIP_LAYERS" not in cfg.genesis_env, (
                f"{pid}: GENESIS_G4_TQ_FORCE_SKIP_LAYERS leaked"
            )


# ─── KV dtype compatibility check ──────────────────────────────────────


class TestCompressionKvDtypeCompat:
    def test_matching_dtype_passes(self):
        model = _make_model(kv_dtype="turboquant_4bit_nc")
        hardware = _make_hardware()
        profile = _make_profile(
            role="structured",
            compression_plan=CompressionPlanConfig(
                native_source_layers=[58],
                default_kv_dtype="turboquant_4bit_nc",
            ),
        )
        compose(model, hardware, profile)  # no raise

    def test_diverging_dtype_rejected(self):
        model = _make_model(kv_dtype="turboquant_4bit_nc")
        hardware = _make_hardware()
        profile = _make_profile(
            role="structured",
            compression_plan=CompressionPlanConfig(
                native_source_layers=[58],
                default_kv_dtype="fp8_e5m2",  # disagrees with model
            ),
        )
        with pytest.raises(SchemaError, match="default_kv_dtype"):
            compose(model, hardware, profile)

    def test_unset_profile_dtype_inherits_silently(self):
        """If profile leaves default_kv_dtype=None, no check fires;
        the model's kv_cache_dtype carries through (already happens
        in compose step 6)."""
        model = _make_model(kv_dtype="turboquant_4bit_nc")
        hardware = _make_hardware()
        profile = _make_profile(
            role="structured",
            compression_plan=CompressionPlanConfig(
                native_source_layers=[58],
                # default_kv_dtype omitted
            ),
        )
        cfg = compose(model, hardware, profile)
        assert cfg.kv_cache_dtype == "turboquant_4bit_nc"

    # ─── P1.2b — neutral parent dtype semantics ────────────────────────

    def test_p1_2b_auto_parent_allows_concrete_profile_dtype(self):
        """ModelDef kv_cache_dtype='auto' means "workload/profile decides".
        A profile may set a concrete compression_plan.default_kv_dtype
        without conflicting.

        This is the key P1.2b unblocker: Gemma 4 31B ModelDef declares
        kv_cache_dtype='auto' in production. The structured β'-A K=4
        profile needs default_kv_dtype='turboquant_4bit_nc'. Pre-P1.2b
        this raised SchemaError (false positive). After P1.2b it
        passes.
        """
        model = _make_model(kv_dtype="auto")
        hardware = _make_hardware()
        profile = _make_profile(
            role="structured",
            compression_plan=CompressionPlanConfig(
                native_source_layers=[58, 59],
                default_kv_dtype="turboquant_4bit_nc",
            ),
        )
        compose(model, hardware, profile)  # no raise

    def test_p1_2b_none_parent_allows_concrete_profile_dtype(self):
        """Same neutral semantics for kv_cache_dtype=None (rare; community
        models that defer entirely to vLLM defaults)."""
        model = _make_model(kv_dtype=None)
        hardware = _make_hardware()
        profile = _make_profile(
            role="structured",
            compression_plan=CompressionPlanConfig(
                native_source_layers=[58],
                default_kv_dtype="fp8_e5m2",
            ),
        )
        compose(model, hardware, profile)  # no raise

    def test_p1_2b_concrete_parent_still_enforced(self):
        """Regression guard: P1.2b only relaxes None/auto. A concrete
        parent dtype that disagrees with a concrete profile dtype must
        still raise."""
        model = _make_model(kv_dtype="fp8_e5m2")
        hardware = _make_hardware()
        profile = _make_profile(
            role="structured",
            compression_plan=CompressionPlanConfig(
                native_source_layers=[58],
                default_kv_dtype="turboquant_4bit_nc",
            ),
        )
        with pytest.raises(SchemaError, match="default_kv_dtype"):
            compose(model, hardware, profile)

    def test_p1_7a_auto_parent_promotes_profile_dtype(self):
        """P1.7a: with parent kv_cache_dtype='auto' and profile
        compression_plan.default_kv_dtype='turboquant_4bit_nc', the
        composed cfg.kv_cache_dtype MUST be 'turboquant_4bit_nc' — NOT
        'auto'.

        Prior to P1.7a this test passed `cfg.kv_cache_dtype == "auto"`
        which was the silent-misrender bug found during the opt-in
        rehearsal: the rendered launcher said `--kv-cache-dtype auto`,
        vLLM picked something other than TQ, and the validated path
        was broken.
        """
        model = _make_model(kv_dtype="auto")
        hardware = _make_hardware()
        profile = _make_profile(
            role="structured",
            compression_plan=CompressionPlanConfig(
                native_source_layers=[58, 59],
                default_kv_dtype="turboquant_4bit_nc",
            ),
        )
        cfg = compose(model, hardware, profile)
        assert cfg.kv_cache_dtype == "turboquant_4bit_nc"

    def test_p1_7a_none_parent_promotes_profile_dtype(self):
        """Mirror for the None-parent (rare; community models)."""
        model = _make_model(kv_dtype=None)
        hardware = _make_hardware()
        profile = _make_profile(
            role="structured",
            compression_plan=CompressionPlanConfig(
                native_source_layers=[58],
                default_kv_dtype="fp8_e5m2",
            ),
        )
        cfg = compose(model, hardware, profile)
        assert cfg.kv_cache_dtype == "fp8_e5m2"

    def test_p1_7a_no_profile_dtype_falls_back_to_parent(self):
        """When profile leaves default_kv_dtype unset, the parent's
        kv_cache_dtype carries through unchanged. Default profile path."""
        model = _make_model(kv_dtype="auto")
        hardware = _make_hardware()
        profile = _make_profile(
            role="default",
            compression_plan=None,
        )
        cfg = compose(model, hardware, profile)
        assert cfg.kv_cache_dtype == "auto"

    def test_p1_7a_concrete_parent_concrete_profile_match(self):
        """Regression guard for the concrete-vs-concrete agreed path:
        composed dtype equals the agreed value, no SchemaError."""
        model = _make_model(kv_dtype="turboquant_4bit_nc")
        hardware = _make_hardware()
        profile = _make_profile(
            role="structured",
            compression_plan=CompressionPlanConfig(
                native_source_layers=[58],
                default_kv_dtype="turboquant_4bit_nc",
            ),
        )
        cfg = compose(model, hardware, profile)
        assert cfg.kv_cache_dtype == "turboquant_4bit_nc"


# ─── Phase 7.G4.31B.K4-BACKEND-FIX: --attention-backend emission ────────


class TestAttentionBackendEmission:
    """Phase 7.G4.31B.K4-BACKEND-FIX (2026-05-23): compose() must emit
    ``--attention-backend <target_default>`` in vllm_extra_args when
    profile.backend_plan.target_default is set. Prior to this fix the
    flag was dropped (BACKEND_PLAN_EMISSION_MAP returned None with a
    "vLLM CLI flag --attention-backend; no env needed" comment but the
    CLI emission was never wired), so vllm auto-picked an attention
    backend (TRITON_ATTN) incompatible with profile-declared
    kv_cache_dtype (turboquant_4bit_nc) → engine init ValueError at
    attention.py:299.
    """

    def test_target_default_turboquant_emits_attention_backend(self):
        """backend_plan.target_default=TURBOQUANT → CLI flag emitted."""
        model = _make_model(kv_dtype="turboquant_4bit_nc")
        hardware = _make_hardware()
        profile = _make_profile(
            role="structured",
            backend_plan=BackendPlanConfig(target_default="TURBOQUANT"),
        )
        cfg = compose(model, hardware, profile)
        assert "--attention-backend" in cfg.vllm_extra_args, (
            "target_default must emit --attention-backend CLI flag; "
            f"got vllm_extra_args={cfg.vllm_extra_args}"
        )
        idx = cfg.vllm_extra_args.index("--attention-backend")
        assert cfg.vllm_extra_args[idx + 1] == "TURBOQUANT"

    def test_target_default_triton_attn_emits_flag(self):
        """Same shape for the non-TURBOQUANT enum values to confirm
        the emission is generic, not hard-coded to TURBOQUANT."""
        model = _make_model(kv_dtype="auto")
        hardware = _make_hardware()
        profile = _make_profile(
            role="structured",
            backend_plan=BackendPlanConfig(target_default="TRITON_ATTN"),
        )
        cfg = compose(model, hardware, profile)
        idx = cfg.vllm_extra_args.index("--attention-backend")
        assert cfg.vllm_extra_args[idx + 1] == "TRITON_ATTN"

    def test_target_default_flash_attn_emits_flag(self):
        model = _make_model(kv_dtype="auto")
        hardware = _make_hardware()
        profile = _make_profile(
            role="structured",
            backend_plan=BackendPlanConfig(target_default="FLASH_ATTN"),
        )
        cfg = compose(model, hardware, profile)
        idx = cfg.vllm_extra_args.index("--attention-backend")
        assert cfg.vllm_extra_args[idx + 1] == "FLASH_ATTN"

    def test_backend_plan_null_skips_emission(self):
        """Profiles with backend_plan=null (the majority) must NOT
        receive a --attention-backend flag — preserves vllm auto-pick."""
        model = _make_model(kv_dtype="auto")
        hardware = _make_hardware()
        profile = _make_profile(role="default", backend_plan=None)
        cfg = compose(model, hardware, profile)
        assert "--attention-backend" not in cfg.vllm_extra_args, (
            "backend_plan=null must NOT emit --attention-backend; "
            f"got vllm_extra_args={cfg.vllm_extra_args}"
        )

    def test_backend_plan_present_but_target_default_none_skips(self):
        """A profile with backend_plan present but target_default=None
        (drafter-only routing — hypothetical future profile) must NOT
        receive --attention-backend; the target follows vllm auto-pick."""
        model = _make_model(kv_dtype="auto")
        hardware = _make_hardware()
        profile = _make_profile(
            role="structured",
            backend_plan=BackendPlanConfig(
                target_default=None,           # explicitly unset
                drafter_sliding="TRITON_ATTN",
            ),
        )
        cfg = compose(model, hardware, profile)
        assert "--attention-backend" not in cfg.vllm_extra_args

    def test_no_profile_no_attention_backend(self):
        """compose(model, hardware, profile=None) must not emit
        --attention-backend — backend_plan only lives on profiles."""
        model = _make_model(kv_dtype="auto")
        hardware = _make_hardware()
        cfg = compose(model, hardware, None)
        assert "--attention-backend" not in cfg.vllm_extra_args

    def test_real_structured_k4_profile_emits_turboquant(self):
        """End-to-end check against the actual builtin profile that
        motivated this fix: gemma4-tq-mtp-structured-k4 has
        backend_plan.target_default=TURBOQUANT, kv_cache_dtype=
        turboquant_4bit_nc. Composing it must yield --attention-backend
        TURBOQUANT in vllm_extra_args."""
        model = load_model("gemma-4-31b-it-awq")
        hardware = load_hardware("a5000-2x-24gbvram-16cpu-128gbram")
        profile = load_profile("gemma4-tq-mtp-structured-k4")
        cfg = compose(model, hardware, profile)
        assert "--attention-backend" in cfg.vllm_extra_args
        idx = cfg.vllm_extra_args.index("--attention-backend")
        assert cfg.vllm_extra_args[idx + 1] == "TURBOQUANT"
        # Cross-check: kv_cache_dtype is consistent with the target
        # backend choice.
        assert cfg.kv_cache_dtype == "turboquant_4bit_nc"

    def test_real_default_31b_profile_does_not_emit(self):
        """The other 31B profile (gemma4-tq-default, backend_plan=null)
        must NOT receive --attention-backend — regression guard against
        the fix leaking into unaffected presets."""
        model = load_model("gemma-4-31b-it-awq")
        hardware = load_hardware("a5000-2x-24gbvram-16cpu-128gbram")
        profile = load_profile("gemma4-tq-default")
        cfg = compose(model, hardware, profile)
        assert "--attention-backend" not in cfg.vllm_extra_args
