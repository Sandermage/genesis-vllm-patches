# SPDX-License-Identifier: Apache-2.0
"""Tests for the gated service-action executor.

Uses only harmless local commands; never touches real containers or the
network. Real server verification is done manually (see the handoff log).
"""
from __future__ import annotations

import pytest

from sndr.product_api.legacy import runtime_exec as rx


def test_apply_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SNDR_ENABLE_APPLY", raising=False)
    assert rx.apply_enabled() is False
    monkeypatch.setenv("SNDR_ENABLE_APPLY", "1")
    assert rx.apply_enabled() is True
    monkeypatch.setenv("SNDR_ENABLE_APPLY", "0")
    assert rx.apply_enabled() is False


def test_exec_command_strips_follow():
    # Follow flags would block forever during execution; they must be stripped.
    assert "-f" not in rx.exec_safe_command("docker logs -f --tail 200 c").split()
    assert rx.exec_safe_command("docker ps --filter name=c") == "docker ps --filter name=c"
    assert "-f" not in rx.exec_safe_command("docker compose -p x logs -f --tail=200")


def test_wrap_ssh_vs_local():
    local = rx.wrap_command("docker ps", transport="local", ssh_target="")
    assert local == "docker ps"
    ssh = rx.wrap_command("docker ps", transport="ssh", ssh_target="user@host")
    assert ssh.startswith("ssh ") and "user@host" in ssh and "docker ps" in ssh


def test_run_steps_executes_local_safe_command():
    results = rx.run_steps(
        [("Status", "true"), ("Echo", "echo sndr-ok")],
        transport="local",
        ssh_target="",
        timeout=10,
    )
    assert all(r.exit_code == 0 for r in results)
    assert any("sndr-ok" in r.stdout for r in results)
    assert all(r.status == "ok" for r in results)


def test_run_steps_marks_failure():
    results = rx.run_steps([("Fail", "false")], transport="local", ssh_target="", timeout=10)
    assert results[0].exit_code != 0
    assert results[0].status == "failed"


def test_execute_blocked_when_disabled(monkeypatch):
    monkeypatch.delenv("SNDR_ENABLE_APPLY", raising=False)
    with pytest.raises(rx.ApplyDisabledError):
        rx.execute_service_action(preset_id="prod-qwen3.6-35b-balanced", action="status")


def test_mutating_requires_confirm(monkeypatch):
    monkeypatch.setenv("SNDR_ENABLE_APPLY", "1")
    with pytest.raises(rx.ConfirmationRequiredError):
        rx.execute_service_action(preset_id="prod-qwen3.6-35b-balanced", action="restart", confirm=False)
