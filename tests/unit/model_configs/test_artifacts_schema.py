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


# ─── A5: single-file GGUF artifact (kind='gguf-file') ──────────────────────


def _gguf_art(local_dir, **kw):
    base = dict(
        hf_id="unsloth/Qwen3.6-27B-MTP-GGUF",
        kind="gguf-file",
        filename="Qwen3.6-27B-Q4_K_M.gguf",
        local_dir=str(local_dir),
    )
    base.update(kw)
    return ArtifactModel(**base)


def test_gguf_artifact_validates():
    _gguf_art("/models/x").validate()  # no raise


def test_gguf_artifact_requires_filename():
    with pytest.raises(SchemaError, match="filename"):
        ArtifactModel(hf_id="o/r", local_dir="/m", kind="gguf-file").validate()


def test_gguf_artifact_filename_must_end_gguf():
    with pytest.raises(SchemaError, match=r"\.gguf"):
        ArtifactModel(hf_id="o/r", local_dir="/m", kind="gguf-file",
                      filename="model.safetensors").validate()


def test_artifact_rejects_unknown_kind():
    with pytest.raises(SchemaError, match="kind"):
        ArtifactModel(hf_id="o/r", local_dir="/m", kind="zip-blob").validate()


def test_gguf_verify_file_missing(tmp_path):
    problems = _gguf_art(tmp_path).verify()
    assert len(problems) == 1
    assert "does not exist" in problems[0]


def test_gguf_verify_file_present_nonzero(tmp_path):
    (tmp_path / "Qwen3.6-27B-Q4_K_M.gguf").write_bytes(b"\x00" * 4096)
    assert _gguf_art(tmp_path).verify() == []


def test_gguf_verify_rejects_empty_file(tmp_path):
    (tmp_path / "Qwen3.6-27B-Q4_K_M.gguf").write_bytes(b"")
    problems = _gguf_art(tmp_path).verify()
    assert any("empty" in p for p in problems)


def test_gguf_verify_does_not_use_is_dir(tmp_path):
    # The bug this fixes: hf-dir verify() hard-requires is_dir(), which always
    # fails for a .gguf FILE. The gguf-file branch must verify the FILE itself,
    # never treat local_dir/filename as a directory.
    (tmp_path / "Qwen3.6-27B-Q4_K_M.gguf").write_bytes(b"\x00" * 4096)
    art = _gguf_art(tmp_path)
    # the gguf path is a file, not a dir — and verify still passes.
    assert (tmp_path / art.filename).is_file()
    assert not (tmp_path / art.filename).is_dir()
    assert art.verify() == []


def test_gguf_verify_min_size_floor(tmp_path):
    (tmp_path / "Qwen3.6-27B-Q4_K_M.gguf").write_bytes(b"\x00" * 4096)
    problems = _gguf_art(tmp_path, min_total_gib=15.0).verify()
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
