# SPDX-License-Identifier: Apache-2.0
"""S3.1 closure (audit P3-1, 2026-05-12): tests for `sndr compose render`.

Cover:

  • Header contains preset key + maintainer + usage hint.
  • yaml.safe_load result — dict with the correct topology
    (`services.vllm-server.{image,container_name,ports,...}`).
  • genesis_env + system_env are merged into the `environment` block.
  • GPU reservation is present with the correct count.
  • Mount substitution from host_paths is applied.
  • Hermetic — does not require docker / kubectl / a real registry,
    uses a synthetic ModelConfig.
"""
from __future__ import annotations

import pytest

yaml = pytest.importorskip("yaml")

from sndr.cli.legacy.compose import render_compose_yaml
from sndr.model_configs.schema import (
    DockerConfig, HardwareSpec, ModelConfig,
)


def _make_cfg(**overrides) -> ModelConfig:
    base = dict(
        key="test-compose", title="Test Compose",
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
            container_name="vllm-test",
            port=8000,
            shm_size="8g",
            network="genesis-net",
            # Default: absolute paths (no placeholder substitution required).
            # Substitution tests use a separate cfg with a placeholder.
            mounts=[
                "/srv/models:/models:ro",
                "/etc/hosts:/etc/hosts:ro",
            ],
        ),
    )
    base.update(overrides)
    return ModelConfig(**base)


def _make_cfg_with_placeholder_mount(**overrides) -> ModelConfig:
    """Helper for tests that exercise the placeholder substitution flow."""
    cfg = _make_cfg(**overrides)
    cfg.docker.mounts = ["${models_dir}:/models:ro", "/etc/hosts:/etc/hosts:ro"]
    return cfg


class TestRenderCompose:
    def test_header_mentions_preset_key(self):
        cfg = _make_cfg()
        out = render_compose_yaml(cfg, host_paths={})
        assert "sndr compose render test-compose" in out
        assert "Maintainer: x" in out

    def test_yaml_parses(self):
        cfg = _make_cfg()
        out = render_compose_yaml(cfg, host_paths={})
        # safe_load skips header comments — must return a dict.
        parsed = yaml.safe_load(out)
        assert "services" in parsed
        assert "vllm-server" in parsed["services"]

    def test_image_and_container_name(self):
        cfg = _make_cfg()
        parsed = yaml.safe_load(render_compose_yaml(cfg, host_paths={}))
        svc = parsed["services"]["vllm-server"]
        assert svc["image"] == "vllm/vllm-openai:nightly"
        assert svc["container_name"] == "vllm-test"

    def test_port_mapping(self):
        cfg = _make_cfg()
        parsed = yaml.safe_load(render_compose_yaml(cfg, host_paths={}))
        ports = parsed["services"]["vllm-server"]["ports"]
        assert any("8000:8000" in str(p) for p in ports)

    def test_environment_combines_system_and_genesis(self):
        cfg = _make_cfg()
        parsed = yaml.safe_load(render_compose_yaml(cfg, host_paths={}))
        env = parsed["services"]["vllm-server"]["environment"]
        assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
        assert env["GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL"] == "1"

    def test_volumes_substituted(self):
        cfg = _make_cfg_with_placeholder_mount()
        out = render_compose_yaml(cfg, host_paths={"models_dir": "/srv/models"})
        parsed = yaml.safe_load(out)
        vols = parsed["services"]["vllm-server"]["volumes"]
        joined = "\n".join(vols)
        assert "/srv/models:/models:ro" in joined
        # second mount without substitution — must remain as-is
        assert "/etc/hosts:/etc/hosts:ro" in joined

    def test_command_starts_with_vllm_serve(self):
        cfg = _make_cfg()
        parsed = yaml.safe_load(render_compose_yaml(cfg, host_paths={}))
        cmd = parsed["services"]["vllm-server"]["command"]
        # Etap 2.1: canonical form — `vllm serve --model <path>` (named flag),
        # not positional. This aligns compose/quadlet/k8s with bare-metal.
        assert cmd[0:2] == ["vllm", "serve"]
        assert "--model" in cmd
        i_model = cmd.index("--model")
        assert cmd[i_model + 1] == "/models/Test-7B"
        # tensor-parallel-size is taken from hardware.n_gpus
        i = cmd.index("--tensor-parallel-size")
        assert cmd[i + 1] == "2"
        # Etap 2.1: --language-model-only must be present (was missing
        # in compose before the unification — divergence vs bare-metal)
        assert "--language-model-only" in cmd

    def test_gpu_reservation_in_deploy(self):
        cfg = _make_cfg()
        parsed = yaml.safe_load(render_compose_yaml(cfg, host_paths={}))
        deploy = parsed["services"]["vllm-server"]["deploy"]
        devices = deploy["resources"]["reservations"]["devices"]
        assert devices[0]["driver"] == "nvidia"
        assert devices[0]["count"] == 2
        assert "gpu" in devices[0]["capabilities"]

    def test_network_block_external(self):
        cfg = _make_cfg()
        parsed = yaml.safe_load(render_compose_yaml(cfg, host_paths={}))
        assert parsed["networks"]["genesis-net"]["external"] is True
        assert "genesis-net" in parsed["services"]["vllm-server"]["networks"]

    def test_no_docker_block_raises(self):
        cfg = _make_cfg(docker=None)
        with pytest.raises(ValueError, match="no docker block"):
            render_compose_yaml(cfg, host_paths={})

    def test_idempotent_render(self):
        cfg = _make_cfg_with_placeholder_mount()
        a = render_compose_yaml(cfg, host_paths={"models_dir": "/srv/m"})
        b = render_compose_yaml(cfg, host_paths={"models_dir": "/srv/m"})
        assert a == b


# ─── Etap 0.4 — secret-leak prevention ──────────────────────────────────


class TestApiKeyNotLeaked:
    """Etap 0.4 (audit 2026-05-12): the API key must never end up in
    the rendered compose as a literal value. Compose interpolation
    `${VLLM_API_KEY:?...}` pulls it from shell env / .env at
    `docker compose up` time.

    Previously `cfg.api_key` went to two places:
      • `service.environment.VLLM_API_KEY=<literal>` — now interpolation.
      • `service.command: ["--api-key", <literal>]` — now removed
        (vLLM picks it up from the env var).
    """

    SECRET = "super-secret-token-NEVER-LEAK-ME"

    def _cfg_with_key(self):
        cfg = _make_cfg()
        cfg.api_key = self.SECRET
        return cfg

    def test_literal_secret_not_in_yaml(self):
        out = render_compose_yaml(self._cfg_with_key(), host_paths={})
        assert self.SECRET not in out, (
            "API key literal found in rendered YAML — Etap 0.4 fix regression"
        )

    def test_env_uses_compose_interpolation(self):
        parsed = yaml.safe_load(
            render_compose_yaml(self._cfg_with_key(), host_paths={})
        )
        env = parsed["services"]["vllm-server"]["environment"]
        assert "VLLM_API_KEY" in env
        # Value is an interpolation reference, not a literal
        assert env["VLLM_API_KEY"].startswith("${VLLM_API_KEY")
        assert self.SECRET not in env["VLLM_API_KEY"]

    def test_api_key_not_in_command(self):
        parsed = yaml.safe_load(
            render_compose_yaml(self._cfg_with_key(), host_paths={})
        )
        cmd = parsed["services"]["vllm-server"]["command"]
        assert "--api-key" not in cmd
        assert self.SECRET not in " ".join(cmd)

    def test_header_documents_secret_flow(self):
        out = render_compose_yaml(self._cfg_with_key(), host_paths={})
        # Header should hint at how the operator can pass the key
        assert "VLLM_API_KEY" in out.split("services:")[0]
        # Mention `.env` or shell env
        header = out.split("services:")[0]
        assert ".env" in header or "shell env" in header.lower()


class TestMountResolverStrict:
    """Etap 2.2 (audit 2026-05-12): unresolved `${var}` mount placeholders
    raise ValueError instead of silent pass-through (previously Docker
    received literal `${unknown}` → cryptic boot failure)."""

    def test_resolved_mount_passes(self):
        from sndr.cli.legacy.compose import _resolve_mount
        result = _resolve_mount("${models_dir}:/models:ro", {"models_dir": "/srv/m"})
        assert result == "/srv/m:/models:ro"

    def test_unresolved_placeholder_raises(self):
        from sndr.cli.legacy.compose import _resolve_mount
        with pytest.raises(ValueError, match="unresolved mount placeholder"):
            _resolve_mount(
                "${undeclared_var}:/models:ro", {"models_dir": "/srv/m"},
            )

    def test_no_placeholders_pass_through(self):
        from sndr.cli.legacy.compose import _resolve_mount
        assert _resolve_mount("/abs:/path:ro", {}) == "/abs:/path:ro"

    def test_multiple_placeholders_all_resolved(self):
        from sndr.cli.legacy.compose import _resolve_mount
        result = _resolve_mount(
            "${a}/${b}:/x:rw",
            {"a": "/srv", "b": "models"},
        )
        assert result == "/srv/models:/x:rw"

    def test_render_compose_yaml_raises_on_unresolved(self):
        """Top-level render must propagate the error, not silently emit
        a broken compose."""
        cfg = _make_cfg()
        # mount with an unknown var
        cfg.docker.mounts = ["${some_unknown_var}:/models:ro"]
        with pytest.raises(ValueError, match="unresolved mount placeholder"):
            render_compose_yaml(cfg, host_paths={})


class TestTempdirPermissions:
    """Etap 0.4: defense-in-depth — `/tmp/sndr-compose/` is always `0o700`,
    the rendered YAML is `0o600`, even if the directory already existed."""

    def test_tempdir_chmod_0o700(self, monkeypatch, tmp_path):
        from sndr.cli.legacy import compose as C
        # Swap tempfile.gettempdir so the test does not touch the real /tmp
        monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
        # Pre-create dir with a broad mode — the fix must change it to 0o700
        precreated = tmp_path / "sndr-compose"
        precreated.mkdir(mode=0o755)
        cfg = _make_cfg()
        cfg.api_key = "leak-test"
        path = C._write_temp_compose(cfg)
        assert path.is_file()
        assert (precreated.stat().st_mode & 0o777) == 0o700
        assert (path.stat().st_mode & 0o777) == 0o600
        # File must not contain the literal API key
        assert "leak-test" not in path.read_text()
