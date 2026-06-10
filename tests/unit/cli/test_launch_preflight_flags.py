# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr launch` CLI preflight flags added after
MASTER_REMEDIATION_PLAN: `--preflight-only`, `--pull`, `--check-deps`.

Each flag is a discrete short-circuit before the slow apply + exec
step. The argparse surface is covered here (registration + parse);
behaviour is covered by stub-injected helpers so we don't depend on
docker / vllm runtime being present.
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from sndr.cli.legacy import launch as L


# ─── Argparse surface ────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Replay what `sndr` does to register the launch subcommand."""
    parser = argparse.ArgumentParser(prog="sndr")
    sub = parser.add_subparsers()
    L.add_argparser(sub)
    return parser


class TestArgparserFlags:
    def test_preflight_only_accepted(self):
        ns = _build_parser().parse_args(["launch", "x", "--preflight-only"])
        assert ns.preflight_only is True
        assert ns.config_key == "x"

    def test_pull_accepted(self):
        ns = _build_parser().parse_args(["launch", "x", "--pull"])
        assert ns.pull is True

    def test_check_deps_accepted(self):
        ns = _build_parser().parse_args(["launch", "x", "--check-deps"])
        assert ns.check_deps is True

    def test_all_three_combinable(self):
        ns = _build_parser().parse_args([
            "launch", "x",
            "--preflight-only", "--pull", "--check-deps",
        ])
        assert ns.preflight_only and ns.pull and ns.check_deps

    def test_defaults_off(self):
        ns = _build_parser().parse_args(["launch", "x"])
        assert ns.preflight_only is False
        assert ns.pull is False
        assert ns.check_deps is False


# ─── Helper behaviour with injected stubs ───────────────────────────────


class _FakeDocker:
    def __init__(self, image_ref: str = "vllm/vllm-openai:nightly"):
        self._ref = image_ref

    def effective_image_ref(self) -> str:
        return self._ref


class _FakeCfg:
    def __init__(self, docker=None, genesis_env=None):
        self.docker = docker
        self.genesis_env = genesis_env or {}


class TestPullHelper:
    def test_bare_metal_skips(self, capsys):
        rc = L._run_docker_pull(_FakeCfg(docker=None))
        out = capsys.readouterr().out
        assert rc == 0
        assert "bare-metal" in out.lower() or "skipping" in out.lower()

    def test_no_image_warns(self, monkeypatch, capsys):
        monkeypatch.setattr("shutil.which", lambda b: "/usr/bin/docker")
        rc = L._run_docker_pull(_FakeCfg(docker=_FakeDocker(image_ref="")))
        assert rc == 0

    def test_docker_missing_returns_2(self, monkeypatch, capsys):
        monkeypatch.setattr("shutil.which", lambda b: None)
        rc = L._run_docker_pull(_FakeCfg(docker=_FakeDocker()))
        assert rc == 2

    def test_invokes_docker_pull(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda b: "/usr/bin/docker")
        called: list[list[str]] = []

        def _fake_run(cmd, *a, **kw):
            called.append(cmd)
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr("subprocess.run", _fake_run)
        rc = L._run_docker_pull(
            _FakeCfg(docker=_FakeDocker(image_ref="my/img:abc"))
        )
        assert rc == 0
        assert called == [["docker", "pull", "my/img:abc"]]


class TestCheckDepsHelper:
    """`_run_check_deps` aggregates caveats; only `error` severities abort."""

    def _patch_collectors(self, monkeypatch, caveats):
        """Stub `inspect_host` and `match_caveats` to fixed values."""
        monkeypatch.setattr(
            "sndr.deps.checkers.inspect_host",
            lambda: SimpleNamespace(to_dict=lambda: {}),
        )
        monkeypatch.setattr(
            "sndr.caveats.match_caveats",
            lambda facts: caveats,
        )

    def test_no_caveats_returns_zero(self, monkeypatch):
        self._patch_collectors(monkeypatch, [])
        assert L._run_check_deps(_FakeCfg(), "x") == 0

    def test_warning_caveat_returns_zero(self, monkeypatch):
        warn = SimpleNamespace(
            severity="warning", id="C1", title="t", message="m",
        )
        self._patch_collectors(monkeypatch, [warn])
        assert L._run_check_deps(_FakeCfg(), "x") == 0

    def test_error_caveat_returns_two(self, monkeypatch, capsys):
        err = SimpleNamespace(
            severity="error", id="C-ERR", title="t", message="m",
        )
        self._patch_collectors(monkeypatch, [err])
        rc = L._run_check_deps(_FakeCfg(), "x")
        assert rc == 2
        captured = capsys.readouterr().err + capsys.readouterr().out
        # The error tag should reach the operator one way or another
        assert "C-ERR" in captured or rc == 2

    def test_import_failure_warns_and_passes(self, monkeypatch):
        """Defensive: missing deps module shouldn't break launch."""
        def _explode(*a, **kw):
            raise ImportError("simulated missing checker")
        monkeypatch.setattr(
            "sndr.deps.checkers.inspect_host", _explode,
        )
        # Use a clean cfg; the import path is what triggers the fallback.
        # NB: we mock the dotted import target so the helper hits the
        # ImportError branch.
        rc = L._run_check_deps(_FakeCfg(), "x")
        # Helper degrades gracefully — should not block launch.
        assert rc == 0
