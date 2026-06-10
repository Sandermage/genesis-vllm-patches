# SPDX-License-Identifier: Apache-2.0
"""Y4 (UNIFIED_CONFIG plan 2026-05-09) — host_port/container_port split.

Backward-compat: configs declaring only `port` keep the same render.
New behavior: configs declaring `host_port` and/or `container_port`
honor the split for both `docker -p HOST:CONTAINER` and the
`vllm serve --port CONTAINER` flag.
"""
from __future__ import annotations

import pytest

from sndr.model_configs.schema import (
    DockerConfig, ModelConfig, HardwareSpec, SchemaError,
)


def _cfg(docker: DockerConfig) -> ModelConfig:
    return ModelConfig(
        key="test-port",
        title="Test port-split config",
        description="Minimal docker config for Y4 tests.",
        schema_version=1,
        maintainer="sandermage",
        model_path="/models/dummy",
        hardware=HardwareSpec(
            gpu_match_keys=["rtx a5000"], n_gpus=1,
            min_vram_per_gpu_mib=1,
        ),
        docker=docker,
    )


# ─── DockerConfig fall-back behavior

def test_docker_legacy_port_only_back_compat():
    d = DockerConfig(image="img", container_name="c", port=8000)
    d.validate()
    assert d.effective_host_port() == 8000
    assert d.effective_container_port() == 8000


def test_docker_explicit_split_overrides_port():
    d = DockerConfig(image="img", container_name="c",
                     port=8000, host_port=18000, container_port=8000)
    d.validate()
    assert d.effective_host_port() == 18000
    assert d.effective_container_port() == 8000


def test_docker_partial_split_keeps_legacy_for_unset_side():
    """Setting only host_port leaves container_port falling back to `port`."""
    d = DockerConfig(image="img", container_name="c",
                     port=8000, host_port=19000)
    d.validate()
    assert d.effective_host_port() == 19000
    assert d.effective_container_port() == 8000


def test_docker_validate_rejects_out_of_range_port():
    for bad in (0, -1, 65536, 99999):
        with pytest.raises(SchemaError, match="1..65535"):
            DockerConfig(image="img", container_name="c",
                         port=bad).validate()


def test_docker_validate_rejects_non_int_port():
    with pytest.raises(SchemaError, match="must be int"):
        DockerConfig(image="img", container_name="c",
                     port=8000, host_port="bad").validate()  # type: ignore


# ─── Renderer respects the split

def test_renderer_legacy_port_uses_same_value_both_sides():
    cfg = _cfg(DockerConfig(image="i", container_name="c", port=8000))
    script = cfg.to_launch_script()
    assert "-p 8000:8000" in script
    assert "--port 8000" in script


def test_renderer_split_emits_host_to_container_mapping():
    cfg = _cfg(DockerConfig(
        image="i", container_name="c",
        port=8000,
        host_port=18000,
        container_port=8000,
    ))
    script = cfg.to_launch_script()
    # Docker mapping reflects the split
    assert "-p 18000:8000" in script
    # vllm serve listens on the container-side port
    assert "--port 8000" in script
    # No leftover same-port mapping
    assert "-p 8000:8000" not in script


def test_renderer_split_with_different_container_port():
    """Operator wants container to listen on 9000 (e.g. avoid clash with sidecar)."""
    cfg = _cfg(DockerConfig(
        image="i", container_name="c",
        port=8000,
        host_port=18000,
        container_port=9000,
    ))
    script = cfg.to_launch_script()
    assert "-p 18000:9000" in script
    assert "--port 9000" in script
    assert "--port 8000" not in script


# ─── YAML round-trip preserves the split

def test_port_split_yaml_roundtrip():
    from sndr.model_configs.schema import dump_yaml, load_yaml
    cfg = _cfg(DockerConfig(
        image="i", container_name="c",
        port=8000,
        host_port=18000,
        container_port=9000,
    ))
    yaml_str = dump_yaml(cfg)
    cfg2 = load_yaml(yaml_str)
    assert cfg2.docker.host_port == 18000
    assert cfg2.docker.container_port == 9000
    assert cfg2.docker.effective_host_port() == 18000
    assert cfg2.docker.effective_container_port() == 9000
