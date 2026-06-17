# SPDX-License-Identifier: Apache-2.0
"""Y1 + B6 (UNIFIED_CONFIG plan 2026-05-09) tests.

Covers:
  - PackageVersions dataclass validation
  - to_pip_args() rendering
  - Renderer integration: when present, package_versions.python_packages
    wins over the legacy hardcoded baseline; when absent/None, legacy
    baseline still appears in the rendered script
  - Loaded YAML round-trip preserves the block
  - Both 35B PROD and 27B INT4 builtin configs declare the block
"""
from __future__ import annotations

import pytest

from sndr.model_configs.schema import (
    PackageVersions, ModelConfig, HardwareSpec, DockerConfig, SchemaError,
    load_yaml, dump_yaml,
)


def _minimal_docker_cfg(pv: PackageVersions | None = None) -> ModelConfig:
    """Smallest valid ModelConfig that exercises the docker renderer."""
    return ModelConfig(
        key="test-pv-cfg",
        title="Test PackageVersions config",
        description="Minimal docker config for renderer integration tests.",
        schema_version=1,
        maintainer="sandermage",
        model_path="/models/dummy",
        hardware=HardwareSpec(
            gpu_match_keys=["rtx a5000"], n_gpus=1,
            min_vram_per_gpu_mib=1,
        ),
        docker=DockerConfig(
            image="vllm/vllm-openai:nightly",
            container_name="test",
            port=8000,
        ),
        package_versions=pv,
    )


# ─── PackageVersions dataclass

def test_package_versions_default_empty():
    pv = PackageVersions()
    assert pv.python_packages == {}
    assert pv.to_pip_args() == ""
    pv.validate()


def test_package_versions_validates_pinned_versions():
    pv = PackageVersions(python_packages={
        "pandas": "2.2.3",
        "scipy": "1.14.1",
    })
    pv.validate()
    out = pv.to_pip_args()
    assert "pandas==2.2.3" in out
    assert "scipy==1.14.1" in out


def test_package_versions_accepts_explicit_eq_prefix():
    pv = PackageVersions(python_packages={"xxhash": "==3.5.0"})
    pv.validate()
    assert pv.to_pip_args() == "xxhash==3.5.0"


def test_package_versions_rejects_version_ranges():
    """Bare ranges (>=, <, ~=) must be rejected — supply-chain integrity."""
    for bad in (">=2.0", "<3.0", ">1.0", "<=2.5", "~=2.2"):
        pv = PackageVersions(python_packages={"pandas": bad})
        with pytest.raises(SchemaError, match="exact pin"):
            pv.validate()


def test_package_versions_rejects_empty_name_or_version():
    with pytest.raises(SchemaError):
        PackageVersions(python_packages={"": "1.0"}).validate()
    with pytest.raises(SchemaError):
        PackageVersions(python_packages={"pkg": ""}).validate()


# ─── Renderer integration (B6)

def test_renderer_uses_package_versions_when_set():
    """When config declares package_versions, its pins appear in the script."""
    pv = PackageVersions(python_packages={
        "pandas": "9.9.9",
        "scipy": "8.8.8",
        "xxhash": "7.7.7",
    })
    cfg = _minimal_docker_cfg(pv)
    script = cfg.to_launch_script()
    # New pins must appear
    assert "pandas==9.9.9" in script
    assert "scipy==8.8.8" in script
    assert "xxhash==7.7.7" in script
    # Legacy hardcoded pins must NOT appear
    assert "pandas==2.2.3" not in script
    assert "scipy==1.14.1" not in script


def test_rendered_launcher_uses_sndr_apply_not_legacy_sndr_core():
    """docker_cmd emitter must invoke `python3 -m sndr.apply`, NOT the
    removed `python3 -m vllm.sndr_core.apply` shim.

    v12.0 dropped the `vllm.sndr_core` mirror (pyproject ships only
    `sndr*`). A freshly-rendered launcher that no longer bind-mounts the
    legacy mirror would otherwise fail apply with ModuleNotFoundError,
    silently leaving Genesis patches unapplied. The surviving entrypoint
    is `sndr/apply/__main__.py` → `python3 -m sndr.apply`.
    """
    cfg = _minimal_docker_cfg(None)
    script = cfg.to_launch_script()
    assert "python3 -m sndr.apply" in script, (
        "rendered launcher must call `python3 -m sndr.apply` (the "
        "surviving v12 entrypoint sndr/apply/__main__.py)"
    )
    assert "python3 -m vllm.sndr_core.apply" not in script, (
        "rendered launcher must NOT call the removed "
        "`python3 -m vllm.sndr_core.apply` shim — patches would silently "
        "fail to apply with ModuleNotFoundError"
    )


def test_renderer_falls_back_to_legacy_when_unset():
    """When package_versions is None, legacy hardcoded baseline kicks in.

    Backwards-compat: existing YAML configs that don't declare the block
    must keep working without surprise dependency drops.
    """
    cfg = _minimal_docker_cfg(None)
    script = cfg.to_launch_script()
    assert "pandas==2.2.3" in script
    assert "scipy==1.14.1" in script
    assert "xxhash==3.5.0" in script


def test_renderer_falls_back_when_python_packages_empty_dict():
    """Empty python_packages also falls through to legacy."""
    cfg = _minimal_docker_cfg(PackageVersions(python_packages={}))
    script = cfg.to_launch_script()
    assert "pandas==2.2.3" in script


# ─── YAML round-trip

def test_package_versions_yaml_roundtrip():
    cfg_in = _minimal_docker_cfg(PackageVersions(python_packages={
        "pandas": "2.2.3",
    }, notes="round-trip test"))
    yaml_str = dump_yaml(cfg_in)
    cfg_out = load_yaml(yaml_str)
    assert cfg_out.package_versions is not None
    assert cfg_out.package_versions.python_packages == {"pandas": "2.2.3"}
    assert cfg_out.package_versions.notes == "round-trip test"


# ─── Builtin configs declare the block (post-Y1+B6 acceptance criteria)

def test_builtin_35b_prod_declares_package_versions():
    # Phase 10 (2026-06-01): test fixture migrated from V1 get() →
    # V2 load_alias() via the ModelDef.package_versions schema
    # extension. Both V1 PROD files (a5000-2x-35b-prod.yaml,
    # a5000-2x-27b-int4-tq-k8v4.yaml) carried the same package pins
    # block; that data now lives in the V2 ModelDef YAMLs (qwen3.6-
    # 35b-a3b-fp8.yaml, qwen3.6-27b-int4-autoround-tq-k8v4.yaml) and
    # is forwarded into the composed cfg via compose.py:603.
    # V1 file source-of-truth was retired in Phase 10 V1 sunset.
    from sndr.model_configs.registry_v2 import load_alias
    cfg = load_alias("prod-qwen3.6-35b-balanced")
    assert cfg is not None
    assert cfg.package_versions is not None
    pkgs = cfg.package_versions.python_packages
    assert pkgs.get("pandas") == "2.2.3"
    assert pkgs.get("scipy") == "1.14.1"
    assert pkgs.get("xxhash") == "3.5.0"


def test_builtin_27b_tq_k8v4_declares_package_versions():
    # Phase 10 migration: V2 alias replaces V1 key (see comment on the
    # 35B test above for full rationale).
    from sndr.model_configs.registry_v2 import load_alias
    cfg = load_alias("prod-qwen3.6-27b-tq-k8v4")
    assert cfg is not None
    assert cfg.package_versions is not None
    pkgs = cfg.package_versions.python_packages
    assert pkgs.get("pandas") == "2.2.3"
    assert pkgs.get("scipy") == "1.14.1"
    assert pkgs.get("xxhash") == "3.5.0"
