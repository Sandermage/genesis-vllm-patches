# SPDX-License-Identifier: Apache-2.0
"""S3.2 closure (audit P3-2, 2026-05-12): tests for `sndr quadlet render`.

Cover:

  • All required sections (`[Unit]`, `[Container]`, `[Service]`,
    `[Install]`).
  • Image / ContainerName / PublishPort are correct.
  • Environment line per env var (system + genesis).
  • Volume substitution via host_paths.
  • Exec= a single line with shlex-quoted args.
  • GPU device line is present.
  • Idempotency.
"""
from __future__ import annotations

import pytest

from sndr.cli.legacy.quadlet import render_quadlet
from sndr.model_configs.schema import (
    DockerConfig, HardwareSpec, ModelConfig,
)


def _make_cfg(**overrides) -> ModelConfig:
    base = dict(
        key="test-quad", title="Test Quadlet",
        description="d", schema_version=1, maintainer="x",
        model_path="/models/Test-7B",
        hardware=HardwareSpec(
            gpu_match_keys=["test"], n_gpus=2,
            min_vram_per_gpu_mib=24576,
        ),
        max_model_len=8192,
        gpu_memory_utilization=0.92,
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        genesis_env={"GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL": "1"},
        system_env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        docker=DockerConfig(
            image="vllm/vllm-openai:nightly",
            container_name="vllm-quad-test",
            port=8000,
            shm_size="8g",
            # Default: absolute path (no placeholder substitution required).
            # Mount-substitution test below uses a separate cfg.
            mounts=[
                "/srv/models:/models:ro",
            ],
        ),
    )
    base.update(overrides)
    return ModelConfig(**base)


def _make_cfg_with_placeholder_mount(**overrides) -> ModelConfig:
    """Helper for tests that exercise placeholder substitution flow."""
    cfg = _make_cfg(**overrides)
    cfg.docker.mounts = ["${models_dir}:/models:ro"]
    return cfg


class TestRenderQuadlet:
    def test_all_sections_present(self):
        out = render_quadlet(_make_cfg(), host_paths={})
        for section in ("[Unit]", "[Container]", "[Service]", "[Install]"):
            assert section in out

    def test_image_and_container_lines(self):
        out = render_quadlet(_make_cfg(), host_paths={})
        assert "Image=vllm/vllm-openai:nightly" in out
        assert "ContainerName=vllm-quad-test" in out
        assert "PublishPort=8000:8000" in out

    def test_env_lines_one_per_var(self):
        out = render_quadlet(_make_cfg(), host_paths={})
        assert (
            "Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            in out
        )
        assert (
            "Environment=GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1"
            in out
        )

    def test_volume_substituted(self):
        out = render_quadlet(
            _make_cfg_with_placeholder_mount(),
            host_paths={"models_dir": "/srv/m"},
        )
        assert "Volume=/srv/m:/models:ro" in out

    def test_exec_single_line_with_vllm_serve(self):
        out = render_quadlet(_make_cfg(), host_paths={})
        exec_lines = [
            line for line in out.splitlines() if line.startswith("Exec=")
        ]
        assert len(exec_lines) == 1
        assert exec_lines[0].startswith("Exec=vllm serve ")

    def test_gpu_device_line(self):
        out = render_quadlet(_make_cfg(), host_paths={})
        assert "AddDevice=nvidia.com/gpu=all" in out

    def test_shm_size_line(self):
        out = render_quadlet(_make_cfg(), host_paths={})
        assert "ShmSize=8g" in out

    def test_no_docker_raises(self):
        with pytest.raises(ValueError, match="no docker block"):
            render_quadlet(_make_cfg(docker=None), host_paths={})

    def test_idempotent(self):
        a = render_quadlet(_make_cfg_with_placeholder_mount(),
                            host_paths={"models_dir": "/srv/m"})
        b = render_quadlet(_make_cfg_with_placeholder_mount(),
                            host_paths={"models_dir": "/srv/m"})
        assert a == b

    def test_restart_on_failure(self):
        out = render_quadlet(_make_cfg(), host_paths={})
        assert "Restart=on-failure" in out

    def test_wanted_by_default_target(self):
        out = render_quadlet(_make_cfg(), host_paths={})
        assert "WantedBy=default.target" in out


# ─── Etap 2.3 — systemd escaping safety ─────────────────────────────────


class TestEnvEscaping:
    """Etap 2.3 (audit 2026-05-12): Environment= values are escaped so
    newlines / quotes / spaces don't break the unit file. Invalid env
    keys are rejected early instead of silently producing a unit that
    systemd refuses to load."""

    def test_value_with_space_gets_quoted(self):
        cfg = _make_cfg()
        cfg.system_env = {"PATH": "/usr/local/bin /opt/bin"}
        out = render_quadlet(cfg, host_paths={})
        assert 'Environment=PATH="/usr/local/bin /opt/bin"' in out

    def test_value_with_newline_escaped(self):
        cfg = _make_cfg()
        cfg.system_env = {"NOTE": "line1\nline2"}
        out = render_quadlet(cfg, host_paths={})
        # Single physical line containing literal `\n`
        env_lines = [l for l in out.splitlines() if l.startswith("Environment=NOTE=")]
        assert len(env_lines) == 1
        assert "\\n" in env_lines[0]
        # Actual newline must not split the value across two lines
        assert "\nline2" not in env_lines[0]

    def test_value_with_dollar_quoted(self):
        cfg = _make_cfg()
        cfg.system_env = {"V": "use $HOME"}
        out = render_quadlet(cfg, host_paths={})
        assert 'Environment=V="use $HOME"' in out

    def test_value_with_double_quote_escaped(self):
        cfg = _make_cfg()
        cfg.system_env = {"V": 'has "quote"'}
        out = render_quadlet(cfg, host_paths={})
        # Wrapped in outer quotes, inner double-quote backslash-escaped
        assert r'Environment=V="has \"quote\""' in out

    def test_empty_value_emits_paired_quotes(self):
        cfg = _make_cfg()
        cfg.system_env = {"V": ""}
        out = render_quadlet(cfg, host_paths={})
        assert 'Environment=V=""' in out

    def test_simple_value_not_quoted(self):
        cfg = _make_cfg()
        cfg.system_env = {"V": "simple-value-123"}
        out = render_quadlet(cfg, host_paths={})
        assert "Environment=V=simple-value-123" in out

    def test_invalid_env_key_raises(self):
        cfg = _make_cfg()
        cfg.system_env = {"bad-key!": "x"}
        with pytest.raises(ValueError, match="systemd-invalid env key"):
            render_quadlet(cfg, host_paths={})

    def test_key_starting_with_digit_rejected(self):
        cfg = _make_cfg()
        cfg.system_env = {"1ST": "x"}
        with pytest.raises(ValueError, match="systemd-invalid env key"):
            render_quadlet(cfg, host_paths={})


class TestExecEscaping:
    """Etap 2.3: argv with newline cannot be represented on a single
    systemd Exec= line — reject explicitly rather than silently truncate."""

    def test_argv_with_newline_rejected(self):
        from sndr.cli.legacy.quadlet import _argv_for_exec
        with pytest.raises(ValueError, match="newline"):
            _argv_for_exec(["vllm", "serve", "--model", "bad\nname"])

    def test_argv_with_spaces_quoted(self):
        from sndr.cli.legacy.quadlet import _argv_for_exec
        out = _argv_for_exec(["vllm", "serve", "--model", "/path with space"])
        assert "'/path with space'" in out
