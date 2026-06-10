# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr service` per-backend behaviour.

The service CLI now delegates compose-backed install/start/stop to
the compose layer instead of the old "render manually" output. These
tests cover the delegation contract + the backend dispatch table so
service surfaces don't silently regress when compose changes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sndr.cli.legacy import service as service_cli
from sndr.model_configs.registry_v2 import load_alias
from sndr.model_configs.schema import ServiceConfig


@pytest.fixture
def cfg_with_docker():
    """Preset with a synthesised ServiceConfig. prod-qwen3.6-35b-balanced's V2 alias
    leaves cfg.service=None by default, so each test gets a fresh
    ServiceConfig it can mutate independently."""
    cfg = load_alias("prod-qwen3.6-35b-balanced")
    cfg.service = ServiceConfig(backend="docker_compose")
    return cfg


# ─── install: docker_compose backend writes a compose YAML ───────────────


class TestInstallDockerCompose:
    def test_install_renders_compose_to_home_sndr_compose(
        self, cfg_with_docker, tmp_path, monkeypatch,
    ):
        """install with backend=docker_compose must produce a real
        compose YAML at ~/.sndr/compose/<key>.yml — not just print
        a "render manually" hint."""
        # Redirect $HOME so we don't touch the operator's real ~/.sndr.
        # Path.home() ignores HOME env on macOS (uses pwd entry), so
        # we patch _compose_file_path directly to route writes/lookups
        # under tmp_path.
        def _stub_path(cfg):
            return tmp_path / ".sndr" / "compose" / f"{cfg.key}.yml"
        monkeypatch.setattr(
            "sndr.cli.legacy.service._compose_file_path", _stub_path,
        )
        monkeypatch.setattr(
            "pathlib.Path.home", lambda: tmp_path,
        )

        # Force backend=docker_compose on the loaded cfg (some presets
        # default to systemd via ServiceConfig).
        cfg_with_docker.service.backend = "docker_compose"

        with patch(
            "sndr.cli.legacy.service._resolve",
            return_value=cfg_with_docker,
        ):
            with patch(
                "sndr.cli.legacy.compose.render_compose_yaml",
                return_value="# stub compose\nservices:\n  vllm-server:\n    image: x\n",
            ):
                args = argparse.Namespace(
                    config="prod-qwen3.6-35b-balanced", yes=True, system=False,
                )
                rc = service_cli.run_install(args)

        assert rc == 0
        target = tmp_path / ".sndr" / "compose" / f"{cfg_with_docker.key}.yml"
        assert target.is_file(), f"compose YAML not written to {target}"
        body = target.read_text()
        assert "services:" in body


# ─── start/stop: docker_compose uses `docker compose -f … up/down` ───────


class TestStartStopDockerCompose:
    def test_start_uses_compose_when_file_exists(
        self, cfg_with_docker, tmp_path, monkeypatch,
    ):
        """When the compose file exists at the canonical path, start
        invokes `docker compose -f <file> up -d`."""
        # Path.home() ignores HOME env on macOS (uses pwd entry), so
        # we patch _compose_file_path directly to route writes/lookups
        # under tmp_path.
        def _stub_path(cfg):
            return tmp_path / ".sndr" / "compose" / f"{cfg.key}.yml"
        monkeypatch.setattr(
            "sndr.cli.legacy.service._compose_file_path", _stub_path,
        )
        monkeypatch.setattr(
            "pathlib.Path.home", lambda: tmp_path,
        )
        compose_dir = tmp_path / ".sndr" / "compose"
        compose_dir.mkdir(parents=True)
        compose_file = compose_dir / f"{cfg_with_docker.key}.yml"
        compose_file.write_text("services: {}\n")

        cfg_with_docker.service.backend = "docker_compose"

        captured: dict = {}

        def _fake_docker_cmd(*args, **kwargs):
            captured["args"] = args
            captured["dry_run"] = kwargs.get("dry_run", False)
            return 0

        with patch("sndr.cli.legacy.service._resolve",
                   return_value=cfg_with_docker):
            with patch("sndr.cli.legacy.service._docker_cmd",
                       side_effect=_fake_docker_cmd):
                args = argparse.Namespace(
                    config="prod-qwen3.6-35b-balanced", yes=True, system=False,
                )
                rc = service_cli.run_start(args)

        assert rc == 0
        # First args should be "compose" "-f" "<path>" "up" "-d"
        assert captured["args"][0] == "compose"
        assert captured["args"][1] == "-f"
        # cfg.key for V2 aliases is the composed triplet, not the short alias.
        assert captured["args"][2].endswith(f"{cfg_with_docker.key}.yml")
        assert captured["args"][3:5] == ("up", "-d")

    def test_start_falls_back_to_docker_start_without_compose_file(
        self, cfg_with_docker, tmp_path, monkeypatch,
    ):
        """Operator may have moved the compose file. Fall back to
        raw `docker start <container>` so the lifecycle still works."""
        # Path.home() ignores HOME env on macOS (uses pwd entry), so
        # we patch _compose_file_path directly to route writes/lookups
        # under tmp_path.
        def _stub_path(cfg):
            return tmp_path / ".sndr" / "compose" / f"{cfg.key}.yml"
        monkeypatch.setattr(
            "sndr.cli.legacy.service._compose_file_path", _stub_path,
        )
        monkeypatch.setattr(
            "pathlib.Path.home", lambda: tmp_path,
        )
        cfg_with_docker.service.backend = "docker_compose"

        captured: dict = {}

        def _fake_docker_cmd(*args, **kwargs):
            captured["args"] = args
            return 0

        with patch("sndr.cli.legacy.service._resolve",
                   return_value=cfg_with_docker):
            with patch("sndr.cli.legacy.service._docker_cmd",
                       side_effect=_fake_docker_cmd):
                args = argparse.Namespace(
                    config="prod-qwen3.6-35b-balanced", yes=True, system=False,
                )
                rc = service_cli.run_start(args)

        assert rc == 0
        # Fallback path: "start" "<container>"
        assert captured["args"][0] == "start"

    def test_stop_uses_compose_down_when_file_exists(
        self, cfg_with_docker, tmp_path, monkeypatch,
    ):
        # Path.home() ignores HOME env on macOS (uses pwd entry), so
        # we patch _compose_file_path directly to route writes/lookups
        # under tmp_path.
        def _stub_path(cfg):
            return tmp_path / ".sndr" / "compose" / f"{cfg.key}.yml"
        monkeypatch.setattr(
            "sndr.cli.legacy.service._compose_file_path", _stub_path,
        )
        monkeypatch.setattr(
            "pathlib.Path.home", lambda: tmp_path,
        )
        compose_dir = tmp_path / ".sndr" / "compose"
        compose_dir.mkdir(parents=True)
        (compose_dir / f"{cfg_with_docker.key}.yml").write_text("services: {}\n")

        cfg_with_docker.service.backend = "docker_compose"

        captured: dict = {}

        def _fake_docker_cmd(*args, **kwargs):
            captured["args"] = args
            return 0

        with patch("sndr.cli.legacy.service._resolve",
                   return_value=cfg_with_docker):
            with patch("sndr.cli.legacy.service._docker_cmd",
                       side_effect=_fake_docker_cmd):
                args = argparse.Namespace(
                    config="prod-qwen3.6-35b-balanced", yes=True, system=False,
                )
                rc = service_cli.run_stop(args)

        assert rc == 0
        assert captured["args"][0] == "compose"
        assert captured["args"][3] == "down"


# ─── podman_quadlet now routes through systemd ───────────────────────────


class TestPodmanQuadletViaSystemd:
    def test_start_routes_through_systemctl(
        self, cfg_with_docker, monkeypatch,
    ):
        cfg_with_docker.service.backend = "podman_quadlet"
        called: dict = {}

        def _fake_systemctl(action, unit, **kwargs):
            called["action"] = action
            called["unit"] = unit
            return 0

        with patch("sndr.cli.legacy.service._resolve",
                   return_value=cfg_with_docker):
            with patch("sndr.cli.legacy.service._systemctl",
                       side_effect=_fake_systemctl):
                args = argparse.Namespace(
                    config="prod-qwen3.6-35b-balanced", yes=True, system=False,
                )
                rc = service_cli.run_start(args)

        assert rc == 0
        assert called["action"] == "start"
        assert called["unit"].startswith("sndr-")
