# SPDX-License-Identifier: Apache-2.0
"""Etap 2.1 closure (audit 2026-05-12): parity tests for
`sndr.model_configs.runtime_command`.

Before unifying compose/quadlet/k8s, each had its own
`_container_command` that diverged from the canonical
`ModelConfig._build_vllm_cmd` (bare-metal). Example divergence:
  • compose: `vllm serve <path>` (positional) — no `--model`
  • bare-metal: `vllm serve --model <path>` (named)
  • compose did not add `--language-model-only`
  • compose did not honor offload

They now all go through `build_runtime_command(cfg).argv`. These tests
guarantee parity: one preset → identical argv across every deployment
adapter.
"""
from __future__ import annotations

import pytest

from sndr.model_configs.runtime_command import (
    RuntimeCommandSpec,
    argv_to_shell,
    build_runtime_command,
)
from sndr.model_configs.schema import (
    DockerConfig, HardwareSpec, ModelConfig, SpecDecodeConfig,
)


def _make_cfg(**overrides) -> ModelConfig:
    base = dict(
        key="test-parity", title="Test Parity",
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
        served_model_name="test-7b",
        tool_call_parser="qwen3_coder",
        reasoning_parser="qwen3",
        docker=DockerConfig(
            image="vllm/vllm-openai:nightly",
            container_name="vllm-test",
            port=8000,
        ),
    )
    base.update(overrides)
    return ModelConfig(**base)


# ─── Canonical argv contract ────────────────────────────────────────────


class TestCanonicalArgv:
    def test_starts_with_vllm_serve(self):
        spec = build_runtime_command(_make_cfg())
        assert spec.argv[0:2] == ["vllm", "serve"]

    def test_model_is_named_flag(self):
        """`--model <path>` form, not positional."""
        spec = build_runtime_command(_make_cfg())
        assert "--model" in spec.argv
        i = spec.argv.index("--model")
        assert spec.argv[i + 1] == "/models/Test-7B"
        # And NOT positional — the third element must be '--model', not the path
        assert spec.argv[2] == "--model"

    def test_language_model_only_present(self):
        """Before Etap 2.1, compose skipped this; now it must be present."""
        spec = build_runtime_command(_make_cfg(language_model_only=True))
        assert "--language-model-only" in spec.argv

    def test_language_model_only_absent_when_false(self):
        spec = build_runtime_command(_make_cfg(language_model_only=False))
        assert "--language-model-only" not in spec.argv

    def test_api_key_not_in_argv(self):
        """Etap 0.4 contract: VLLM_API_KEY goes via env, not CLI."""
        cfg = _make_cfg()
        cfg.api_key = "secret-token-XYZ"
        spec = build_runtime_command(cfg)
        assert "--api-key" not in spec.argv
        assert "secret-token-XYZ" not in spec.argv

    def test_offload_args_included(self):
        from sndr.model_configs.schema import OffloadConfig
        cfg = _make_cfg()
        cfg.offload = OffloadConfig(cpu_offload_gib=8)
        spec = build_runtime_command(cfg)
        # OffloadConfig.to_vllm_args() returns `--cpu-offload-gb 8`
        assert any("cpu-offload" in a for a in spec.argv)

    def test_spec_decode_serialized(self):
        cfg = _make_cfg(spec_decode=SpecDecodeConfig(
            method="mtp", num_speculative_tokens=3,
        ))
        spec = build_runtime_command(cfg)
        assert "--speculative-config" in spec.argv
        i = spec.argv.index("--speculative-config")
        # JSON form
        assert '"method": "mtp"' in spec.argv[i + 1]

    def test_vllm_extra_args_appended_last(self):
        cfg = _make_cfg()
        cfg.vllm_extra_args = ["--no-scheduler-reserve-full-isl"]
        spec = build_runtime_command(cfg)
        assert spec.argv[-1] == "--no-scheduler-reserve-full-isl"

    def test_port_from_docker(self):
        cfg = _make_cfg()
        cfg.docker.port = 8101
        spec = build_runtime_command(cfg)
        i = spec.argv.index("--port")
        assert spec.argv[i + 1] == "8101"

    def test_port_default_without_docker(self):
        cfg = _make_cfg(docker=None)
        spec = build_runtime_command(cfg)
        i = spec.argv.index("--port")
        assert spec.argv[i + 1] == "8000"


# ─── Parity across deployment adapters ──────────────────────────────────


class TestDeploymentParity:
    """Etap 2.1: compose / quadlet emit identical argv via the canonical builder."""

    def test_compose_command_matches_canonical(self):
        from sndr.cli.legacy.compose import _container_command
        cfg = _make_cfg()
        compose_argv = _container_command(cfg)
        canonical_argv = build_runtime_command(cfg).argv
        assert compose_argv == canonical_argv

    def test_quadlet_uses_compose_command(self):
        """Quadlet imports `_container_command` from compose — so it
        should emit identical argv."""
        from sndr.cli.legacy.compose import _container_command
        from sndr.cli.legacy import quadlet as Q
        cfg = _make_cfg()
        # Quadlet's render fetches argv via the same _container_command
        assert Q._container_command is _container_command


# ─── argv_to_shell ──────────────────────────────────────────────────────


class TestArgvToShell:
    def test_quotes_spaces(self):
        out = argv_to_shell(["vllm", "serve", "--model", "/path with space"])
        # shlex.quote wraps in single quotes
        assert any("'/path with space'" in part for part in out)

    def test_no_quote_for_simple(self):
        out = argv_to_shell(["--port", "8000"])
        # Simple tokens without special chars — no quotes
        assert out == ["--port", "8000"]

    def test_quotes_dollar_sign(self):
        out = argv_to_shell(["--env", "${VAR}"])
        # $ must be quoted so the shell does not interpret it
        assert "'${VAR}'" in out or '"${VAR}"' in out
