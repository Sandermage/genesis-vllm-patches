# SPDX-License-Identifier: Apache-2.0
"""B4 + Y3 wire-in (UNIFIED_CONFIG plan 2026-05-09) — pull_via_artifacts tests."""
from __future__ import annotations

from pathlib import Path

import pytest


# Phase 10 (2026-06-01): V1 sunset — pull_via_artifacts resolves cfg_key
# via V1 registry; tests using V1 keys skip when V1 files retire. The
# synthetic-fixture test (test_pull_via_artifacts_skips_when_locally_
# complete) injects directly into registry and keeps running.
_V1_DIR_PULL = (Path(__file__).resolve().parents[2] / "vllm" / "sndr_core"
                / "model_configs" / "builtin")
_skip_if_no_v1_35b_pull = pytest.mark.skipif(
    not (_V1_DIR_PULL / "a5000-2x-35b-prod.yaml").is_file(),
    reason="V1 fixture a5000-2x-35b-prod.yaml retired (Phase 10 sunset)",
)
_skip_if_no_v1_27b_pull = pytest.mark.skipif(
    not (_V1_DIR_PULL / "a5000-2x-27b-int4-tq-k8v4.yaml").is_file(),
    reason="V1 fixture a5000-2x-27b-int4-tq-k8v4.yaml retired (Phase 10 sunset)",
)


def test_pull_via_artifacts_unknown_config_returns_2(capsys):
    from sndr.compat.models.pull import pull_via_artifacts
    rc = pull_via_artifacts(cfg_key="nonexistent-xyz", dry_run=True)
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown preset" in err


@pytest.mark.skip(
    reason="Fixture `single-3090-hybrid-gdn-tier-aware-example` retired in "
           "V1 sunset #2 (2026-06-01); pull_via_artifacts now returns "
           "'unknown preset' before reaching the artifacts.models check. "
           "Test intent (config-exists-but-no-artifacts-block) needs to "
           "pivot to synthetic ModelConfig injection — same pattern as "
           "test_pull_via_artifacts_skips_when_locally_complete."
)
def test_pull_via_artifacts_config_without_artifacts_returns_2(capsys):
    """Path C EXAMPLE has no artifacts.models block → clean error."""
    from sndr.compat.models.pull import pull_via_artifacts
    rc = pull_via_artifacts(
        cfg_key="single-3090-hybrid-gdn-tier-aware-example",
        dry_run=True,
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "artifacts.models" in err


@_skip_if_no_v1_35b_pull
def test_pull_via_artifacts_35b_prod_dry_run(capsys):
    """35B PROD has Y3 artifacts.models declared → dry-run should succeed."""
    from sndr.compat.models.pull import pull_via_artifacts
    rc = pull_via_artifacts(cfg_key="a5000-2x-35b-prod", dry_run=True)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Y3 artifacts" in out
    assert "Qwen3.6-35B-A3B-FP8" in out
    assert "[dry-run]" in out


@_skip_if_no_v1_27b_pull
def test_pull_via_artifacts_27b_prod_dry_run(capsys):
    """27B PROD has Y3 artifacts.models declared → dry-run should succeed."""
    from sndr.compat.models.pull import pull_via_artifacts
    rc = pull_via_artifacts(cfg_key="a5000-2x-27b-int4-tq-k8v4",
                              dry_run=True)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Lorbus/Qwen3.6-27B-int4-AutoRound" in out


def test_pull_via_artifacts_skips_when_locally_complete(tmp_path, capsys):
    """If local_dir exists and verify() returns no problems → skip pull."""
    from sndr.model_configs.schema import (
        ArtifactModel, Artifacts, ModelConfig, HardwareSpec, DockerConfig,
    )
    # Build a synthetic complete artifact on tmp_path
    art_dir = tmp_path / "fake-model"
    art_dir.mkdir()
    (art_dir / "config.json").write_text("{}")
    (art_dir / "model-001.safetensors").write_bytes(b"x" * 1024)

    cfg = ModelConfig(
        key="test-pull-skip",
        title="x", description="x",
        schema_version=1, maintainer="sandermage",
        model_path=str(art_dir),
        hardware=HardwareSpec(gpu_match_keys=["a5000"], n_gpus=1,
                              min_vram_per_gpu_mib=1),
        docker=DockerConfig(image="i", container_name="c", port=8000),
        artifacts=Artifacts(models=[ArtifactModel(
            hf_id="org/repo",
            local_dir=str(art_dir),
            required_files=["config.json", "*.safetensors"],
        )]),
    )
    # Inject into registry temporarily
    from sndr.model_configs import registry as R
    original = R.get
    R.get = lambda key: cfg if key == "test-pull-skip" else original(key)
    try:
        from sndr.compat.models.pull import pull_via_artifacts
        rc = pull_via_artifacts(cfg_key="test-pull-skip")
        assert rc == 0
        out = capsys.readouterr().out
        assert "already complete" in out
    finally:
        R.get = original


@_skip_if_no_v1_35b_pull
def test_cli_argparser_accepts_config_flag():
    """CLI parses --config without requiring positional model_key."""
    from sndr.compat.models.pull import _parse_args
    args = _parse_args(["--config", "a5000-2x-35b-prod", "--dry-run"])
    assert args.config == "a5000-2x-35b-prod"
    assert args.dry_run is True
    assert args.model_key is None


@_skip_if_no_v1_35b_pull
def test_cli_main_dispatches_to_pull_via_artifacts(capsys):
    """`pull --config <key> --dry-run` reaches pull_via_artifacts."""
    from sndr.compat.models.pull import main
    rc = main(["--config", "a5000-2x-35b-prod", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Y3 artifacts" in out


def test_cli_main_legacy_path_still_works():
    """Without --config, legacy registry-based pull still triggers
    (returns 2 on unknown key — that's the registry handling)."""
    from sndr.compat.models.pull import main
    rc = main(["never-a-registered-key-xyz"])
    assert rc == 2  # legacy path: unknown key


def test_cli_main_no_key_no_config_returns_2(capsys):
    """Neither model_key nor --config → friendly error."""
    from sndr.compat.models.pull import main
    rc = main([])
    assert rc == 2
    err = capsys.readouterr().err
    assert "model_key" in err or "--config" in err
