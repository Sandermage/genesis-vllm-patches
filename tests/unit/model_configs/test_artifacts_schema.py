# SPDX-License-Identifier: Apache-2.0
"""Y3 (UNIFIED_CONFIG plan 2026-05-09) — Artifacts schema tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from sndr.model_configs.schema import (
    ArtifactModel, ArtifactCache, Artifacts, ModelConfig, HardwareSpec,
    DockerConfig, SchemaError, dump_yaml, load_yaml,
)


# ─── ArtifactModel

def test_artifact_model_minimal_valid():
    m = ArtifactModel(
        hf_id="Qwen/Qwen3.6-27B-int4-AutoRound",
        local_dir="/models/Qwen3.6-27B-int4-AutoRound",
    )
    m.validate()
    assert m.revision == "main"
    assert m.gated is False
    assert m.required_files == ["config.json"]


def test_artifact_model_rejects_bad_hf_id():
    with pytest.raises(SchemaError, match="org/repo"):
        ArtifactModel(hf_id="just-a-name", local_dir="/m").validate()
    with pytest.raises(SchemaError, match="org/repo"):
        ArtifactModel(hf_id="", local_dir="/m").validate()


def test_artifact_model_rejects_empty_local_dir():
    with pytest.raises(SchemaError, match="local_dir"):
        ArtifactModel(hf_id="org/repo", local_dir="").validate()


def test_artifact_model_rejects_negative_min_size():
    with pytest.raises(SchemaError, match="min_total_gib"):
        ArtifactModel(hf_id="o/r", local_dir="/m",
                      min_total_gib=-1.0).validate()


def test_artifact_model_verify_local_dir_missing(tmp_path):
    m = ArtifactModel(hf_id="o/r", local_dir=str(tmp_path / "nope"))
    problems = m.verify()
    assert len(problems) == 1
    assert "does not exist" in problems[0]


def test_artifact_model_verify_required_file_missing(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    m = ArtifactModel(
        hf_id="o/r", local_dir=str(tmp_path),
        required_files=["config.json", "model.safetensors"],
    )
    problems = m.verify()
    assert len(problems) == 1
    assert "model.safetensors" in problems[0]


def test_artifact_model_verify_required_glob_match(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model-001.safetensors").write_bytes(b"x" * 100)
    m = ArtifactModel(
        hf_id="o/r", local_dir=str(tmp_path),
        required_files=["config.json", "*.safetensors"],
    )
    assert m.verify() == []


def test_artifact_model_verify_min_size(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tiny").write_bytes(b"x" * 100)
    m = ArtifactModel(
        hf_id="o/r", local_dir=str(tmp_path),
        required_files=["config.json"],
        min_total_gib=1.0,
    )
    problems = m.verify()
    assert any("min_total_gib" in p for p in problems)


# ─── ArtifactCache

def test_artifact_cache_known_kinds():
    for kind in ("huggingface_hub", "triton", "torch_compile",
                 "compile_cache", "safetensors", "other"):
        c = ArtifactCache(kind=kind, path="/tmp")
        c.validate()


def test_artifact_cache_rejects_unknown_kind():
    with pytest.raises(SchemaError, match="kind must be one of"):
        ArtifactCache(kind="cuda_graph_cache", path="/tmp").validate()


def test_artifact_cache_rejects_empty_path():
    with pytest.raises(SchemaError, match="path"):
        ArtifactCache(kind="triton", path="").validate()


# ─── Artifacts container

def test_artifacts_default_empty():
    a = Artifacts()
    a.validate()
    assert a.models == []
    assert a.caches == []


def test_artifacts_validates_each_member():
    a = Artifacts(
        models=[ArtifactModel(hf_id="a/b", local_dir="/m")],
        caches=[
            ArtifactCache(kind="triton", path="/tmp/triton"),
            ArtifactCache(kind="huggingface_hub", path="~/.cache/huggingface"),
        ],
    )
    a.validate()


def test_artifacts_rejects_non_list_members():
    with pytest.raises(SchemaError):
        Artifacts(models="not-a-list").validate()  # type: ignore
    with pytest.raises(SchemaError):
        Artifacts(caches="not-a-list").validate()  # type: ignore


# ─── YAML round-trip

def _cfg_with_artifacts(a: Artifacts) -> ModelConfig:
    return ModelConfig(
        key="test-artifacts",
        title="x", description="x",
        schema_version=1, maintainer="sandermage",
        model_path="/m",
        hardware=HardwareSpec(gpu_match_keys=["a5000"], n_gpus=1,
                              min_vram_per_gpu_mib=1),
        docker=DockerConfig(image="i", container_name="c", port=8000),
        artifacts=a,
    )


def test_artifacts_yaml_roundtrip():
    a = Artifacts(
        models=[
            ArtifactModel(
                hf_id="Qwen/Qwen3.6-27B-int4-AutoRound",
                local_dir="/models/Qwen3.6-27B-int4-AutoRound",
                revision="main",
                gated=False,
                required_files=["config.json", "*.safetensors"],
                min_total_gib=14.0,
            ),
        ],
        caches=[
            ArtifactCache(kind="huggingface_hub",
                          path="~/.cache/huggingface", persistent=True),
            ArtifactCache(kind="triton",
                          path="/home/sander/.cache/triton-v11"),
        ],
    )
    cfg = _cfg_with_artifacts(a)
    yaml_str = dump_yaml(cfg)
    cfg2 = load_yaml(yaml_str)
    assert cfg2.artifacts is not None
    assert len(cfg2.artifacts.models) == 1
    assert cfg2.artifacts.models[0].hf_id == "Qwen/Qwen3.6-27B-int4-AutoRound"
    assert cfg2.artifacts.models[0].min_total_gib == 14.0
    assert len(cfg2.artifacts.caches) == 2
    assert cfg2.artifacts.caches[0].kind == "huggingface_hub"
    assert cfg2.artifacts.caches[1].kind == "triton"
