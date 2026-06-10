# SPDX-License-Identifier: Apache-2.0
"""Y2 (UNIFIED_CONFIG plan 2026-05-09) — PackageSources schema tests."""
from __future__ import annotations

import pytest

from sndr.model_configs.schema import (
    PackageSource, PackageSources, ModelConfig, HardwareSpec,
    DockerConfig, SchemaError, dump_yaml, load_yaml,
)


def test_package_source_minimal_valid():
    s = PackageSource(name="docker", kind="distro_repo")
    s.validate()
    assert s.channel == "stable"
    assert s.allow_third_party is False


def test_package_source_all_known_kinds():
    for kind in ("distro_repo", "pip", "docker_image", "nvidia_repo",
                 "github_release", "source_build"):
        PackageSource(name="x", kind=kind).validate()


def test_package_source_curl_pipe_bash_requires_opt_in():
    """SAFETY: curl|bash MUST require explicit allow_third_party=True."""
    with pytest.raises(SchemaError, match="curl_pipe_bash"):
        PackageSource(name="docker", kind="curl_pipe_bash",
                      allow_third_party=False).validate()
    PackageSource(name="docker", kind="curl_pipe_bash",
                  allow_third_party=True).validate()


def test_package_source_rejects_unknown_kind():
    with pytest.raises(SchemaError, match="kind must be one of"):
        PackageSource(name="x", kind="conda").validate()


def test_package_source_rejects_empty_name():
    with pytest.raises(SchemaError, match="name"):
        PackageSource(name="", kind="pip").validate()


def test_package_sources_default_empty():
    ps = PackageSources()
    ps.validate()
    assert ps.sources == []
    assert ps.get("anything") is None


def test_package_sources_validates_each_member():
    ps = PackageSources(sources=[
        PackageSource(name="docker", kind="distro_repo"),
        PackageSource(name="vllm", kind="pip", channel="nightly"),
        PackageSource(name="model-weights", kind="docker_image"),
    ])
    ps.validate()
    assert ps.get("docker").kind == "distro_repo"
    assert ps.get("vllm").channel == "nightly"


def test_package_sources_rejects_duplicate_names():
    with pytest.raises(SchemaError, match="duplicate name"):
        PackageSources(sources=[
            PackageSource(name="docker", kind="distro_repo"),
            PackageSource(name="docker", kind="curl_pipe_bash",
                          allow_third_party=True),
        ]).validate()


def test_package_sources_yaml_roundtrip():
    cfg = ModelConfig(
        key="test-pkg-src",
        title="x", description="x",
        schema_version=1, maintainer="sandermage",
        model_path="/m",
        hardware=HardwareSpec(gpu_match_keys=["a5000"], n_gpus=1,
                              min_vram_per_gpu_mib=1),
        docker=DockerConfig(image="i", container_name="c", port=8000),
        package_sources=PackageSources(sources=[
            PackageSource(name="docker", kind="distro_repo",
                          channel="stable",
                          notes="apt install docker.io"),
            PackageSource(name="nvidia_container_toolkit",
                          kind="nvidia_repo",
                          channel="stable",
                          notes="prefer official NVIDIA repo"),
            PackageSource(name="vllm", kind="pip",
                          channel="nightly"),
        ]),
    )
    yaml_str = dump_yaml(cfg)
    cfg2 = load_yaml(yaml_str)
    assert cfg2.package_sources is not None
    assert len(cfg2.package_sources.sources) == 3
    assert cfg2.package_sources.get("vllm").channel == "nightly"
