# SPDX-License-Identifier: Apache-2.0
"""Phase 1 V2 composer tests — `compose(model, hardware, profile)` produces
a valid V1 ModelConfig; merge precedence + conflict detection + runtime
resolution.

See PROJECT_ROADMAP_V2_2026-05-12_RU.md § 4.7 for the composition spec.
"""
from __future__ import annotations

import pytest

from sndr.model_configs.schema import (
    HardwareSpec, ModelConfig, SchemaError, SpecDecodeConfig,
)
from sndr.model_configs.schema_v2 import (
    HardwareDef,
    HardwareSizing,
    ModelCapabilities,
    ModelDef,
    ModelRequires,
    ModelVersions,
    PatchesDelta,
    ProfileDef,
    ProfileVersionsOverride,
    RuntimeBlock,
    RuntimeDockerBlock,
)
from sndr.model_configs.compose import (
    apply_patches_delta,
    check_compat,
    compose,
)


# ─── Builders ───────────────────────────────────────────────────────────


def _model(**kw) -> ModelDef:
    base = dict(
        schema_version=2, kind="model", id="qwen3.6-fp8",
        title="Q", maintainer="x", last_validated="2026-05-12",
        license="apache-2.0", model_path="/models/q",
        served_model_name="q",
        capabilities=ModelCapabilities(attention_arch="hybrid_gdn_moe"),
        requires=ModelRequires(min_total_vram_mib=44000, min_gpu_count=2),
        versions=ModelVersions(vllm_pin_required="0.20.2"),
        patches={"GENESIS_ENABLE_P67": "1"},
    )
    base.update(kw)
    m = ModelDef(**base)
    m.validate()
    return m


def _hardware(**kw) -> HardwareDef:
    base = dict(
        schema_version=2, kind="hardware", id="a5000-2x",
        title="H", maintainer="x",
        hardware=HardwareSpec(
            gpu_match_keys=["rtx a5000"], n_gpus=2,
            min_vram_per_gpu_mib=24000,
            cuda_capability_min=(8, 6),
        ),
        sizing=HardwareSizing(max_model_len=8192),
        runtime=RuntimeBlock(
            default="docker", supported=["docker"],
            docker=RuntimeDockerBlock(image="vllm:nightly"),
        ),
        system_env={"NCCL_P2P_DISABLE": "1"},
    )
    base.update(kw)
    h = HardwareDef(**base)
    h.validate()
    return h


def _profile(**kw) -> ProfileDef:
    base = dict(
        schema_version=2, kind="profile", id="wave9-test",
        parent_model="qwen3.6-fp8", maintainer="x",
        patches_delta=PatchesDelta(),
    )
    base.update(kw)
    p = ProfileDef(**base)
    p.validate()
    return p


# ─── apply_patches_delta ────────────────────────────────────────────────


class TestApplyPatchesDelta:
    def test_enable_adds_keys(self):
        out = apply_patches_delta(
            {"A": "1"}, PatchesDelta(enable={"B": "1"}),
        )
        assert out == {"A": "1", "B": "1"}

    def test_disable_removes_keys(self):
        out = apply_patches_delta(
            {"A": "1", "B": "1"}, PatchesDelta(disable=["A"]),
        )
        assert out == {"B": "1"}

    def test_override_replaces_value(self):
        out = apply_patches_delta(
            {"A": "1"}, PatchesDelta(override={"A": "42"}),
        )
        assert out == {"A": "42"}

    def test_order_enable_disable_override(self):
        """If enable adds K and override replaces it later, override wins."""
        out = apply_patches_delta(
            {"OLD": "1"},
            PatchesDelta(
                enable={"K": "1"},
                disable=["OLD"],
                override={"K": "99"},
            ),
        )
        assert out == {"K": "99"}

    def test_disable_nonexistent_is_noop(self):
        out = apply_patches_delta(
            {"A": "1"}, PatchesDelta(disable=["NEVER_EXISTED"]),
        )
        assert out == {"A": "1"}

    def test_does_not_mutate_input(self):
        canonical = {"A": "1"}
        apply_patches_delta(canonical, PatchesDelta(enable={"B": "2"}))
        assert canonical == {"A": "1"}


# ─── check_compat ───────────────────────────────────────────────────────


class TestCheckCompat:
    def test_matching_hw_ok(self):
        assert check_compat(_model(), _hardware()) is None

    def test_too_few_gpus_rejected(self):
        m = _model(requires=ModelRequires(min_gpu_count=4))
        err = check_compat(m, _hardware())
        assert err is not None and "min_gpu_count" in err

    def test_insufficient_vram_rejected(self):
        m = _model(requires=ModelRequires(min_total_vram_mib=100000))
        err = check_compat(m, _hardware())
        assert err is not None and "min_total_vram_mib" in err

    def test_low_cuda_capability_rejected(self):
        m = _model(requires=ModelRequires(min_cuda_capability=(9, 0)))
        err = check_compat(m, _hardware())
        assert err is not None and "CUDA capability" in err


# ─── compose() end-to-end ──────────────────────────────────────────────


class TestComposeBasic:
    def test_composed_key_double_dash(self):
        cfg = compose(_model(), _hardware())
        # V1 key regex forbids dots: compose() sanitizes `qwen3.6` → `qwen3-6`.
        assert cfg.key == "qwen3-6-fp8--a5000-2x"

    def test_composed_key_with_profile(self):
        cfg = compose(_model(), _hardware(), _profile())
        assert cfg.key == "qwen3-6-fp8--a5000-2x--wave9-test"

    def test_returns_v1_modelconfig(self):
        cfg = compose(_model(), _hardware())
        assert isinstance(cfg, ModelConfig)
        cfg.validate()  # downstream gates accept it

    def test_identity_from_model(self):
        cfg = compose(_model(), _hardware())
        assert cfg.model_path == "/models/q"
        assert cfg.served_model_name == "q"

    def test_sizing_from_hardware(self):
        cfg = compose(
            _model(), _hardware(sizing=HardwareSizing(max_model_len=131072)),
        )
        assert cfg.max_model_len == 131072

    def test_system_env_from_hardware(self):
        cfg = compose(_model(), _hardware())
        assert cfg.system_env == {"NCCL_P2P_DISABLE": "1"}

    def test_docker_rendered(self):
        cfg = compose(_model(), _hardware())
        assert cfg.docker is not None
        assert cfg.docker.image == "vllm:nightly"
        # container_name_template "vllm-{model_id}" substituted
        assert cfg.docker.container_name == "vllm-qwen3.6-fp8"

    def test_capabilities_owned_by_model(self):
        m = _model(capabilities=ModelCapabilities(
            attention_arch="dense",
            tool_call_parser="qwen3_coder",
            spec_decode=SpecDecodeConfig(method="mtp", num_speculative_tokens=3),
            kv_cache_dtype="turboquant_k8v4",
        ))
        cfg = compose(m, _hardware())
        assert cfg.tool_call_parser == "qwen3_coder"
        assert cfg.spec_decode.method == "mtp"
        assert cfg.kv_cache_dtype == "turboquant_k8v4"


class TestComposeProfileDelta:
    def test_enable_adds_to_patches(self):
        cfg = compose(_model(), _hardware(),
                       _profile(patches_delta=PatchesDelta(
                           enable={"GENESIS_ENABLE_PN90": "1"})))
        assert "GENESIS_ENABLE_PN90" in cfg.genesis_env
        assert "GENESIS_ENABLE_P67" in cfg.genesis_env  # canonical preserved

    def test_disable_removes_from_patches(self):
        cfg = compose(_model(), _hardware(),
                       _profile(patches_delta=PatchesDelta(
                           disable=["GENESIS_ENABLE_P67"])))
        assert "GENESIS_ENABLE_P67" not in cfg.genesis_env

    def test_override_changes_value(self):
        cfg = compose(_model(), _hardware(),
                       _profile(patches_delta=PatchesDelta(
                           override={"GENESIS_ENABLE_P67": "42"})))
        assert cfg.genesis_env["GENESIS_ENABLE_P67"] == "42"

    def test_vllm_pin_override_wins(self):
        p = _profile(versions_override=ProfileVersionsOverride(
            vllm_pin_required="0.20.99.dev0",
        ))
        cfg = compose(_model(), _hardware(), p)
        assert cfg.vllm_pin_required == "0.20.99.dev0"

    def test_pin_falls_back_to_model_when_no_override(self):
        cfg = compose(_model(), _hardware(), _profile())
        assert cfg.vllm_pin_required == "0.20.2"


# ─── Compat error paths ─────────────────────────────────────────────────


class TestComposeRejections:
    def test_profile_targets_wrong_model_rejected(self):
        m = _model(id="qwen3.6-other")
        p = _profile(parent_model="qwen3.6-fp8")
        with pytest.raises(SchemaError, match="parent_model"):
            compose(m, _hardware(), p)

    def test_unsupported_runtime_rejected(self):
        with pytest.raises(SchemaError, match="not in supported"):
            compose(_model(), _hardware(), runtime_override="k8s")

    def test_incompatible_hw_rejected(self):
        m = _model(requires=ModelRequires(min_gpu_count=8))
        with pytest.raises(SchemaError, match="min_gpu_count"):
            compose(m, _hardware())

    def test_runtime_block_missing_for_chosen_runtime(self):
        rt = RuntimeBlock(
            default="podman", supported=["podman"],
            docker=None, podman=None,
        )
        h = _hardware(runtime=rt)
        with pytest.raises(SchemaError, match="block missing"):
            compose(_model(), h)


# ─── Runtime override ──────────────────────────────────────────────────


class TestRuntimeOverride:
    def test_default_used_when_no_override(self):
        rt = RuntimeBlock(
            default="docker", supported=["docker", "podman"],
            docker=RuntimeDockerBlock(image="vllm:docker"),
            podman=RuntimeDockerBlock(image="vllm:podman"),
        )
        cfg = compose(_model(), _hardware(runtime=rt))
        assert cfg.docker.image == "vllm:docker"

    def test_explicit_override_to_podman(self):
        rt = RuntimeBlock(
            default="docker", supported=["docker", "podman"],
            docker=RuntimeDockerBlock(image="vllm:docker"),
            podman=RuntimeDockerBlock(image="vllm:podman"),
        )
        cfg = compose(_model(), _hardware(runtime=rt),
                       runtime_override="podman")
        assert cfg.docker.image == "vllm:podman"

    def test_bare_metal_yields_no_docker_config(self):
        from sndr.model_configs.schema_v2 import RuntimeBareMetalBlock
        rt = RuntimeBlock(
            default="bare-metal", supported=["bare-metal"],
            bare_metal=RuntimeBareMetalBlock(venv_path="/opt/venv"),
        )
        cfg = compose(_model(), _hardware(runtime=rt))
        assert cfg.docker is None
