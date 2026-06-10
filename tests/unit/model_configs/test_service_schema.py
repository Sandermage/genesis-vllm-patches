# SPDX-License-Identifier: Apache-2.0
"""Y10 (UNIFIED_CONFIG plan 2026-05-09) — ServiceConfig schema tests."""
from __future__ import annotations

import pytest

from sndr.model_configs.schema import (
    ServiceConfig, ModelConfig, HardwareSpec, DockerConfig,
    SchemaError, dump_yaml, load_yaml,
)


def test_service_default_validates():
    s = ServiceConfig()
    s.validate()
    assert s.backend == "docker_compose"
    assert s.restart == "on-failure"


def test_service_all_known_backends():
    for backend in ("systemd", "docker_compose", "podman_quadlet",
                    "kubernetes", "proxmox", "bare_metal"):
        ServiceConfig(backend=backend, service_name="x").validate()


def test_service_rejects_unknown_backend():
    with pytest.raises(SchemaError, match="backend must be one of"):
        ServiceConfig(backend="initd").validate()


def test_service_all_known_restart_policies():
    for restart in ("always", "on-failure", "no", "unless-stopped"):
        ServiceConfig(restart=restart).validate()


def test_service_rejects_unknown_restart():
    with pytest.raises(SchemaError, match="restart must be one of"):
        ServiceConfig(restart="reboot-machine").validate()


def test_service_yaml_roundtrip():
    cfg = ModelConfig(
        key="test-svc",
        title="x", description="x",
        schema_version=1, maintainer="sandermage",
        model_path="/m",
        hardware=HardwareSpec(gpu_match_keys=["a5000"], n_gpus=1,
                              min_vram_per_gpu_mib=1),
        docker=DockerConfig(image="i", container_name="c", port=8000),
        service=ServiceConfig(
            backend="systemd",
            service_name="genesis-35b-prod",
            user="sander",
            working_dir="/home/sander/genesis-vllm-patches",
            env_file="/etc/genesis/35b.env",
            logs_dir="/var/log/genesis",
            restart="always",
            notes="systemd unit; bench rig deployment",
        ),
    )
    yaml_str = dump_yaml(cfg)
    cfg2 = load_yaml(yaml_str)
    assert cfg2.service is not None
    assert cfg2.service.backend == "systemd"
    assert cfg2.service.service_name == "genesis-35b-prod"
    assert cfg2.service.restart == "always"
