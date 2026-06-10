# SPDX-License-Identifier: Apache-2.0
"""TDD for multi-runtime support — DeploymentConfig + symbolic mounts.

Three new schema concepts:
  1. DeploymentConfig — per-runtime boolean flags (docker/podman/kubernetes/
     lxc_proxmox/bare_metal) + a `default` runtime to pick when launcher
     called without --runtime override.
  2. Mounts as symbolic references (`models_dir`, `hf_cache`, `triton_cache`,
     `compile_cache`, `genesis_src`, `plugin_src`) instead of absolute paths.
     Each user has different paths — community configs MUST be portable.
  3. Host config (`~/.genesis/host.yaml`) — written at install/first-run by
     auto-detection (scans common locations: /nfs/*, /opt/*, ~/.cache/*,
     /var/lib/genesis/*). Render time resolves symbolic mounts via host.yaml.

Reference: noonghunna/club-3090 docs/CONTAINER_RUNTIMES.md documents
microk8s + Proxmox LXC + podman environments + bare-metal venv workaround.
"""
from __future__ import annotations

import pytest

from sndr.model_configs.schema import (
    ModelConfig,
    HardwareSpec,
    SchemaError,
)


def _minimal(**kwargs) -> ModelConfig:
    base = dict(
        key="test-config",
        title="Test config",
        description="Minimal test",
        schema_version=1,
        maintainer="testuser",
        model_path="${models_dir}/Qwen3.6-27B-int4-AutoRound",  # symbolic
        hardware=HardwareSpec(
            gpu_match_keys=["rtx a5000"], n_gpus=2, min_vram_per_gpu_mib=20000,
        ),
    )
    base.update(kwargs)
    return ModelConfig(**base)


class TestDeploymentConfig:
    """deploy: block with per-runtime booleans."""

    def test_default_deployment_docker_only(self):
        """Without explicit deploy: block, defaults to docker=True."""
        cfg = _minimal()
        # default deployment must allow docker — backward-compat with all
        # current builtin configs that have no deploy: block
        assert cfg.deploy.docker is True
        assert cfg.deploy.default == "docker"

    def test_explicit_deployment_block_accepted(self):
        from sndr.model_configs.schema import DeploymentConfig
        cfg = _minimal(
            deploy=DeploymentConfig(
                docker=True,
                podman=True,
                kubernetes=False,
                lxc_proxmox=False,
                bare_metal=True,
                default="bare_metal",
            ),
        )
        cfg.validate()  # should not raise

    def test_default_runtime_must_be_supported(self):
        """deploy.default must be a runtime where deploy.<runtime>=True."""
        from sndr.model_configs.schema import DeploymentConfig
        cfg = _minimal(
            deploy=DeploymentConfig(
                docker=True,
                podman=False,
                kubernetes=True,
                lxc_proxmox=False,
                bare_metal=False,
                default="bare_metal",  # mismatch — bare_metal=False
            ),
        )
        with pytest.raises(SchemaError, match="default.*bare_metal.*not supported"):
            cfg.validate()

    def test_at_least_one_runtime_must_be_true(self):
        """All runtimes False = config can't run anywhere → reject."""
        from sndr.model_configs.schema import DeploymentConfig
        cfg = _minimal(
            deploy=DeploymentConfig(
                docker=False, podman=False, kubernetes=False,
                lxc_proxmox=False, bare_metal=False,
                default="docker",  # default itself is False
            ),
        )
        with pytest.raises(SchemaError, match="at least one"):
            cfg.validate()

    def test_unknown_default_runtime_rejected(self):
        from sndr.model_configs.schema import DeploymentConfig
        cfg = _minimal(
            deploy=DeploymentConfig(
                docker=True, podman=False, kubernetes=False,
                lxc_proxmox=False, bare_metal=False,
                default="bogus",  # not a known runtime
            ),
        )
        with pytest.raises(SchemaError, match="default.*bogus"):
            cfg.validate()


class TestSymbolicMounts:
    """Mounts use ${var} references resolved at render-time via host.yaml."""

    def test_mount_string_with_symbolic_var_accepted(self):
        from sndr.model_configs.schema import DockerConfig
        cfg = _minimal(
            docker=DockerConfig(
                image="vllm/vllm-openai:nightly",
                container_name="vllm-server",
                port=8000,
                mounts=[
                    "${models_dir}:/models:ro",
                    "${hf_cache}:/root/.cache/huggingface:ro",
                    "${genesis_src}:/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro",
                ],
            ),
        )
        cfg.validate()  # symbolic mounts must validate cleanly

    def test_resolve_mounts_via_host_config(self):
        """resolve_symbolic_mounts() expands ${var} per host.yaml."""
        from sndr.model_configs.schema import (
            DockerConfig, resolve_symbolic_mounts,
        )
        host_paths = {
            "models_dir": "/data/models",
            "hf_cache": "/home/alice/.cache/huggingface",
            "genesis_src": "/opt/genesis-vllm-patches/vllm/_genesis",
        }
        mounts = [
            "${models_dir}:/models:ro",
            "${hf_cache}:/root/.cache/huggingface:ro",
            "${genesis_src}:/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro",
        ]
        resolved = resolve_symbolic_mounts(mounts, host_paths)
        assert resolved == [
            "/data/models:/models:ro",
            "/home/alice/.cache/huggingface:/root/.cache/huggingface:ro",
            "/opt/genesis-vllm-patches/vllm/_genesis:/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro",
        ]

    def test_unresolved_var_raises(self):
        """Missing host_paths entry raises with clear error."""
        from sndr.model_configs.schema import resolve_symbolic_mounts
        with pytest.raises(SchemaError, match="unknown_path"):
            resolve_symbolic_mounts(
                ["${unknown_path}:/x"],
                {"models_dir": "/data/models"},
            )

    def test_absolute_path_passes_through(self):
        """Absolute paths (no ${var}) work as before — backward compat."""
        from sndr.model_configs.schema import resolve_symbolic_mounts
        resolved = resolve_symbolic_mounts(
            ["/data/models:/models:ro", "/etc/foo.conf:/etc/foo.conf:ro"],
            {},
        )
        assert resolved == [
            "/data/models:/models:ro",
            "/etc/foo.conf:/etc/foo.conf:ro",
        ]


class TestHostConfig:
    """Auto-detection writes ~/.genesis/host.yaml at install."""

    def test_load_host_config_from_path(self, tmp_path):
        from sndr.model_configs.host import (
            load_host_config, HostConfig,
        )
        host_yaml = tmp_path / "host.yaml"
        host_yaml.write_text("""
paths:
  models_dir: /data/models
  hf_cache: /home/alice/.cache/huggingface
  triton_cache: /var/cache/triton
  compile_cache: /var/cache/vllm-compile
  genesis_src: /opt/genesis/vllm/_genesis
  plugin_src: /opt/genesis/tools/genesis_vllm_plugin
""")
        hc = load_host_config(host_yaml)
        assert hc.paths["models_dir"] == "/data/models"
        assert hc.paths["hf_cache"] == "/home/alice/.cache/huggingface"

    def test_detect_paths_finds_common_locations(self, tmp_path, monkeypatch):
        from sndr.model_configs.host import detect_paths
        # Synthesize fake host filesystem
        nfs_models = tmp_path / "nfs" / "models"
        nfs_models.mkdir(parents=True)
        hf_cache = tmp_path / "fake_home" / ".cache" / "huggingface"
        hf_cache.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(tmp_path / "fake_home"))
        # detect_paths takes optional candidates list
        detected = detect_paths(
            models_candidates=[str(nfs_models), "/nonexistent/place"],
            hf_cache_candidates=None,  # default falls back to $HOME/.cache/huggingface
        )
        assert detected["models_dir"] == str(nfs_models)
        assert detected["hf_cache"] == str(hf_cache)

    def test_save_host_config_round_trip(self, tmp_path):
        from sndr.model_configs.host import (
            save_host_config, load_host_config, HostConfig,
        )
        hc = HostConfig(paths={
            "models_dir": "/data/models",
            "hf_cache": "/home/bob/.cache/huggingface",
        })
        path = tmp_path / "host.yaml"
        save_host_config(hc, path)
        loaded = load_host_config(path)
        assert loaded.paths == hc.paths


class TestKubernetesRender:
    """Phase A 2026-05-06: --runtime kubernetes emits valid k8s YAML."""

    def test_kubernetes_render_emits_valid_yaml(self):
        """Output must be parseable as YAML stream (3 documents).

        Phase 10 (2026-06-01): migrated from V1 registry.get() to V2
        load_alias() — V1 file scheduled for sunset, V2 alias composes
        byte-identical ModelConfig. _render_kubernetes() consumes the
        ModelConfig shape unchanged.
        """
        from sndr.compat.model_config_cli import _render_kubernetes
        from sndr.model_configs.registry_v2 import load_alias
        cfg = load_alias("prod-qwen3.6-35b-balanced")
        assert cfg is not None
        output = _render_kubernetes(cfg)
        # Strip header comments (everything before first `---`)
        idx = output.find("---")
        assert idx >= 0, "k8s render must contain at least one YAML document separator"
        yaml_stream = output[idx:]
        # Parse all documents
        import yaml
        docs = list(yaml.safe_load_all(yaml_stream))
        # Filter out None (trailing separator with no content)
        docs = [d for d in docs if d is not None]
        kinds = [d.get("kind") for d in docs]
        # Should produce ConfigMap + Deployment + Service
        assert "ConfigMap" in kinds, f"missing ConfigMap, got {kinds}"
        assert "Deployment" in kinds, f"missing Deployment, got {kinds}"
        assert "Service" in kinds, f"missing Service, got {kinds}"

    def test_kubernetes_render_includes_gpu_request(self):
        """Deployment must request nvidia.com/gpu: <n_gpus>.

        Phase 10 migration: V2 load_alias() replaces V1 get() (same
        rationale as test_kubernetes_render_emits_valid_yaml above).
        """
        from sndr.compat.model_config_cli import _render_kubernetes
        from sndr.model_configs.registry_v2 import load_alias
        cfg = load_alias("prod-qwen3.6-35b-balanced")
        output = _render_kubernetes(cfg)
        assert f"nvidia.com/gpu: {cfg.hardware.n_gpus}" in output

    def test_kubernetes_render_no_docker_block_emits_error_comment(self):
        """Configs without docker block should emit a clear error comment."""
        from sndr.compat.model_config_cli import _render_kubernetes
        cfg = _minimal()  # no docker block in the minimal helper
        output = _render_kubernetes(cfg)
        assert "ERROR" in output
        assert "docker" in output.lower()
