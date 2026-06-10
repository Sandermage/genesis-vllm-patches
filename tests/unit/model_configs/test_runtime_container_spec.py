# SPDX-License-Identifier: Apache-2.0
"""Phase 4.5 acceptance tests — RuntimeContainerSpec canonical IR.

Goal: prove that ONE `RuntimeContainerSpec` drives every emitter to produce
semantically equivalent output. Format diffs are OK; semantic diffs (different
mounts, different env, different ports) are bugs.

Today the test covers:
  - Composer correctness from V1 ModelConfig (V2 alias path).
  - Image-digest precedence over image tag.
  - V1 mount string parsing (source:target[:mode]) → typed MountSpec.
  - Env merge (system_env + genesis_env, with genesis_env winning on
    collision — matches existing emitter behaviour).
  - SELinux label-disable extraction from extra_run_flags.

Cross-emitter diff (docker/compose/quadlet/k8s) ships as separate test
files once each emitter is refactored to consume the spec.
"""
from __future__ import annotations

import pytest

from sndr.model_configs.registry_v2 import load_alias
from sndr.model_configs.runtime_container import (
    DeviceSpec,
    MountSpec,
    PortSpec,
    RuntimeContainerSpec,
    SecuritySpec,
    _parse_v1_mount_string,
    build_runtime_container_spec,
)


# ─── Sub-types ─────────────────────────────────────────────────────────


class TestMountSpec:
    def test_to_docker_arg_ro(self):
        m = MountSpec(source="/models", target="/models", mode="ro")
        assert m.to_docker_arg() == "/models:/models:ro"

    def test_to_docker_arg_rw(self):
        m = MountSpec(source="/cache", target="/root/.cache", mode="rw")
        assert m.to_docker_arg() == "/cache:/root/.cache:rw"


class TestPortSpec:
    def test_default_tcp(self):
        p = PortSpec(host_port=8000, container_port=8000)
        assert p.to_docker_arg() == "8000:8000/tcp"

    def test_explicit_udp(self):
        p = PortSpec(host_port=9000, container_port=9000, protocol="udp")
        assert p.to_docker_arg() == "9000:9000/udp"


class TestV1MountStringParsing:
    def test_two_part_default_rw(self):
        m = _parse_v1_mount_string("/host:/container")
        assert m.source == "/host"
        assert m.target == "/container"
        assert m.mode == "rw"

    def test_three_part_ro(self):
        m = _parse_v1_mount_string("/host:/container:ro")
        assert m.mode == "ro"

    def test_three_part_rw_explicit(self):
        m = _parse_v1_mount_string("/host:/container:rw")
        assert m.mode == "rw"

    def test_three_part_invalid_mode_falls_back_to_rw(self):
        m = _parse_v1_mount_string("/host:/container:weird")
        assert m.mode == "rw"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="cannot parse V1 mount"):
            _parse_v1_mount_string("/just-one")


# ─── Composer from V2 alias path ───────────────────────────────────────


class TestComposeFromV2Alias:
    """The V2 alias resolver produces a V1 ModelConfig; that ModelConfig
    must compose into a complete RuntimeContainerSpec."""

    def test_prod_35b_alias_produces_complete_spec(self):
        cfg = load_alias("prod-qwen3.6-35b-balanced")
        spec = build_runtime_container_spec(cfg, runtime="docker")
        assert isinstance(spec, RuntimeContainerSpec)
        assert spec.runtime == "docker"
        # Container name from V2 hardware template expansion.
        assert "qwen3.6-35b-a3b-fp8" in spec.container_name
        # Image + digest both present (V2 hardware uses digest-pinned image).
        assert spec.image is not None
        assert spec.image_digest is not None
        assert spec.image_digest.startswith("vllm/vllm-openai@sha256:")
        # Single port mapping at 8000.
        assert len(spec.ports) == 1
        assert spec.ports[0].host_port == 8000
        assert spec.ports[0].container_port == 8000

    def test_image_digest_wins_over_tag(self):
        cfg = load_alias("prod-qwen3.6-35b-balanced")
        spec = build_runtime_container_spec(cfg)
        # effective_image_ref MUST return the digest when both are set.
        assert spec.effective_image_ref() == spec.image_digest

    def test_env_merged_from_genesis_plus_system(self):
        cfg = load_alias("prod-qwen3.6-35b-balanced")
        spec = build_runtime_container_spec(cfg)
        # genesis_env keys (P-codes) present.
        assert "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL" in spec.env
        # system_env keys (NCCL/PYTORCH/VLLM) present.
        assert "NCCL_P2P_DISABLE" in spec.env
        assert "PYTORCH_CUDA_ALLOC_CONF" in spec.env

    def test_mounts_parsed_from_v1_strings(self):
        cfg = load_alias("prod-qwen3.6-35b-balanced")
        spec = build_runtime_container_spec(cfg)
        assert len(spec.mounts) >= 1
        # At least one models mount, read-only.
        models_mounts = [m for m in spec.mounts if "models" in m.target.lower()]
        assert len(models_mounts) >= 1
        assert models_mounts[0].mode == "ro"

    def test_argv_invariant_via_runtime_command_spec(self):
        cfg = load_alias("prod-qwen3.6-35b-balanced")
        spec = build_runtime_container_spec(cfg)
        argv = spec.command.argv
        # Argv invariants from runtime_command.py:
        assert argv[0:2] == ["vllm", "serve"]
        assert "--model" in argv
        assert "/models/Qwen3.6-35B-A3B-FP8" in argv
        # Tensor parallel size = hardware.n_gpus = 2.
        tp_idx = argv.index("--tensor-parallel-size")
        assert argv[tp_idx + 1] == "2"


class TestComposeAcrossAllAliases:
    """Phase 3 ships 11 V2 aliases. Each must compose into a valid spec."""

    @pytest.mark.parametrize("alias", [
        "prod-qwen3.6-35b-balanced",
        "prod-qwen3.6-27b-tq-k8v4",
        "prod-qwen3.6-35b-dflash",
        "long-ctx-qwen3.6-27b",
        "qa-qwen3.6-27b-tested",
        "qa-qwen3.6-27b-tq-1x",
        "prod-qwen3.6-27b-dflash",
        "experimental-qwen3.6-27b-tq-dflash-ab",
        "example-2x-tier-aware",
        "example-3090-dense-cpu-offload",
        "example-3090-tier-aware",
    ])
    def test_alias_composes_to_spec(self, alias):
        cfg = load_alias(alias)
        spec = build_runtime_container_spec(cfg, runtime="docker")
        # Invariants for every spec:
        assert spec.runtime == "docker"
        assert spec.container_name
        assert spec.effective_image_ref()
        assert len(spec.command.argv) >= 4
        assert spec.command.argv[0:2] == ["vllm", "serve"]
        # Single port mapping.
        assert len(spec.ports) == 1


# ─── Runtime parameter ────────────────────────────────────────────────


class TestRuntimeParameter:
    """Composer honors the requested runtime backend."""

    def test_explicit_compose_runtime(self):
        cfg = load_alias("prod-qwen3.6-35b-balanced")
        spec = build_runtime_container_spec(cfg, runtime="compose")
        assert spec.runtime == "compose"

    def test_explicit_quadlet_runtime(self):
        cfg = load_alias("prod-qwen3.6-35b-balanced")
        spec = build_runtime_container_spec(cfg, runtime="quadlet")
        assert spec.runtime == "quadlet"

    def test_default_runtime_is_docker(self):
        cfg = load_alias("prod-qwen3.6-35b-balanced")
        spec = build_runtime_container_spec(cfg)
        assert spec.runtime == "docker"


# ─── Cross-runtime semantic equivalence ────────────────────────────────


class TestCrossRuntimeSemanticEquivalence:
    """For a given alias, switching runtime backend must NOT change the
    container-semantic fields (image, env, mounts, ports). Only the
    `runtime` discriminator field changes."""

    @pytest.mark.parametrize("alias", [
        "prod-qwen3.6-35b-balanced",
        "prod-qwen3.6-27b-tq-k8v4",
        "long-ctx-qwen3.6-27b",
    ])
    def test_docker_compose_quadlet_semantic_equality(self, alias):
        cfg = load_alias(alias)
        sd = build_runtime_container_spec(cfg, runtime="docker")
        sc = build_runtime_container_spec(cfg, runtime="compose")
        sq = build_runtime_container_spec(cfg, runtime="quadlet")
        # Container-semantic fields identical across all three.
        for other in (sc, sq):
            assert other.container_name == sd.container_name
            assert other.image == sd.image
            assert other.image_digest == sd.image_digest
            assert other.env == sd.env
            assert other.mounts == sd.mounts
            assert other.ports == sd.ports
            assert other.devices == sd.devices
            assert other.shm_size == sd.shm_size
            assert other.memory_limit == sd.memory_limit
            assert other.network_mode == sd.network_mode
            assert other.security == sd.security
            assert other.extra_run_flags == sd.extra_run_flags
            # Argv invariant — list equality on the underlying argv.
            assert other.command.argv == sd.command.argv


# ─── Security spec extraction ──────────────────────────────────────────


class TestSecurityExtraction:
    def test_selinux_label_disable_extracted(self):
        """If V1 docker.extra_run_flags contains the SELinux opt, spec
        sets the structured field AND strips it from extra_run_flags."""
        from sndr.model_configs.schema import DockerConfig, ModelConfig

        # Build a minimal ModelConfig + DockerConfig with SELinux flag.
        cfg = load_alias("prod-qwen3.6-35b-balanced")
        # Patch extra_run_flags to include the SELinux opt.
        from dataclasses import replace
        docker = replace(
            cfg.docker,
            extra_run_flags=["--security-opt label=disable"],
        )
        # Re-attach the modified docker block.
        cfg_with_flag = replace(cfg, docker=docker)
        spec = build_runtime_container_spec(cfg_with_flag)
        # Structured field set.
        assert spec.security.selinux_label_disable is True
        # Flag stripped from extra_run_flags so emitter doesn't re-emit it.
        assert "--security-opt label=disable" not in spec.extra_run_flags

    def test_no_security_flag_default_off(self):
        cfg = load_alias("prod-qwen3.6-35b-balanced")
        spec = build_runtime_container_spec(cfg)
        # prod-qwen3.6-35b-balanced V2 hardware default has no SELinux flag.
        assert spec.security.selinux_label_disable is False
