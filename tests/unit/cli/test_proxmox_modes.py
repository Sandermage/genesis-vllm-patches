# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr proxmox render` mode validation.

The render path now fails fast on unknown modes (exit 2) instead of
warning and proceeding. This guards against operator overrides that
bypass the ProxmoxConfig.validate() loader and reach the renderer
with a mode the renderer can't handle.
"""
from __future__ import annotations

import argparse
import io
from contextlib import redirect_stdout
from unittest.mock import patch

import pytest

from sndr.cli.legacy import proxmox as proxmox_cli
from sndr.model_configs.registry_v2 import load_alias
from sndr.model_configs.schema import ProxmoxConfig


@pytest.fixture
def cfg_lxc():
    """A preset with an explicit proxmox block — prod-qwen3.6-35b-balanced's cfg
    has proxmox=None by default, so we synthesise one for the test."""
    cfg = load_alias("prod-qwen3.6-35b-balanced")
    cfg.proxmox = ProxmoxConfig(
        mode="lxc", runtime="venv", container_id_or_vmid=100,
    )
    return cfg


class TestKnownModesRender:
    @pytest.mark.parametrize("mode", ["lxc", "vm", "host"])
    def test_known_mode_renders_without_error(self, cfg_lxc, mode):
        cfg_lxc.proxmox.mode = mode
        buf = io.StringIO()
        with patch("sndr.cli.legacy.proxmox._resolve",
                   return_value=cfg_lxc):
            with redirect_stdout(buf):
                rc = proxmox_cli.run_render(
                    argparse.Namespace(config="prod-qwen3.6-35b-balanced")
                )
        assert rc == 0
        out = buf.getvalue()
        assert f"mode={mode}" in out


class TestUnknownModeFailsFast:
    def test_unknown_mode_returns_exit_2(self, cfg_lxc):
        cfg_lxc.proxmox.mode = "kubernetes-on-pve"   # not in valid set
        with patch("sndr.cli.legacy.proxmox._resolve",
                   return_value=cfg_lxc):
            rc = proxmox_cli.run_render(
                argparse.Namespace(config="prod-qwen3.6-35b-balanced")
            )
        # Fail-fast: exit 2 means "operator action required", not
        # silently continuing past a misconfiguration.
        assert rc == 2

    def test_empty_mode_returns_exit_2(self, cfg_lxc):
        cfg_lxc.proxmox.mode = ""
        with patch("sndr.cli.legacy.proxmox._resolve",
                   return_value=cfg_lxc):
            rc = proxmox_cli.run_render(
                argparse.Namespace(config="prod-qwen3.6-35b-balanced")
            )
        assert rc == 2
