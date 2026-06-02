# SPDX-License-Identifier: Apache-2.0
"""Tests for scoped container management (whitelist, gating, both backends).

No real docker / SSH is touched: the SSH backend runs through an injected
``runner`` that records the docker argv, and the socket backend runs through an
injected ``transport`` that returns canned Docker Engine API payloads.
"""
from __future__ import annotations

import json

import pytest

from vllm.sndr_core.product_api import container_ops as co


# ─── whitelist ─────────────────────────────────────────────────────────


def test_is_managed_name_accepts_engine_and_daemon():
    assert co.is_managed_name("vllm-pn95-2xa5000")
    assert co.is_managed_name("vllm-35b-prod")
    assert co.is_managed_name("sndr-daemon")


def test_is_managed_name_rejects_foreign_and_empty():
    assert not co.is_managed_name("postgres")
    assert not co.is_managed_name("nginx-proxy")
    assert not co.is_managed_name("")
    assert not co.is_managed_name(None)  # type: ignore[arg-type]


def test_ensure_managed_blocks_foreign_and_malformed():
    with pytest.raises(co.NotManagedError):
        co.ensure_managed("postgres")
    with pytest.raises(co.NotManagedError):
        co.ensure_managed("vllm; rm -rf /")  # injection attempt → rejected by name regex
    co.ensure_managed("vllm-35b-prod")  # managed → no raise


# ─── gating (pure, HTTP layer consumes these) ─────────────────────────


def test_gate_lifecycle_requires_apply_and_confirm():
    assert not co.gate_lifecycle(apply_on=False, confirm=True).allowed
    assert co.gate_lifecycle(apply_on=False, confirm=True).status == 403
    assert not co.gate_lifecycle(apply_on=True, confirm=False).allowed
    assert co.gate_lifecycle(apply_on=True, confirm=False).status == 400
    assert co.gate_lifecycle(apply_on=True, confirm=True).allowed


def test_gate_exec_requires_separate_exec_flag():
    # apply on + confirm but EXEC off → still blocked, 403
    blocked = co.gate_exec(apply_on=True, exec_on=False, confirm=True)
    assert not blocked.allowed and blocked.status == 403
    assert "EXEC" in blocked.reason or "exec" in blocked.reason
    assert co.gate_exec(apply_on=True, exec_on=True, confirm=True).allowed
    # exec inherits the lifecycle gate too
    assert not co.gate_exec(apply_on=False, exec_on=True, confirm=True).allowed


# ─── SSH backend (injected runner) ────────────────────────────────────


class _RecordingRunner:
    """Captures the docker argv lists and returns canned (rc, out, err)."""

    def __init__(self, responses=None):
        self.calls: list[list[str]] = []
        self._responses = responses or {}

    def __call__(self, argv):
        self.calls.append(list(argv))
        key = argv[1] if len(argv) > 1 else ""
        return self._responses.get(key, (0, "", ""))


def _ssh(responses=None):
    runner = _RecordingRunner(responses)
    return co.SshContainerControl(runner=runner), runner


def test_ssh_list_parses_and_scopes():
    line_engine = json.dumps({
        "Names": "vllm-35b-prod", "Image": "vllm/vllm-openai:nightly",
        "State": "running", "Status": "Up 2 hours", "Ports": "0.0.0.0:8101->8101/tcp",
        "ID": "abc123", "CreatedAt": "2026-06-01 10:00:00 +0000", "Labels": "com.x=y",
    })
    line_foreign = json.dumps({
        "Names": "postgres", "Image": "postgres:16", "State": "running",
        "Status": "Up 3 days", "Ports": "", "ID": "def456", "CreatedAt": "x", "Labels": "",
    })
    ctrl, runner = _ssh({"ps": (0, line_engine + "\n" + line_foreign + "\n", "")})
    items = ctrl.list_managed()
    names = {c.name for c in items}
    assert "vllm-35b-prod" in names
    assert "postgres" not in names  # foreign container filtered out
    assert runner.calls[0][:2] == ["docker", "ps"]


def test_ssh_lifecycle_builds_correct_argv_and_is_scoped():
    ctrl, runner = _ssh()
    ctrl.restart("vllm-35b-prod")
    assert ["docker", "restart", "vllm-35b-prod"] == runner.calls[-1]
    # foreign target never reaches the runner
    with pytest.raises(co.NotManagedError):
        ctrl.stop("postgres")
    assert all("postgres" not in c for c in runner.calls)


def test_ssh_exec_quotes_argv_and_scopes():
    ctrl, runner = _ssh({"exec": (7, "hello\n", "")})
    res = ctrl.exec("vllm-35b-prod", ["python3", "-c", "print(1)"])
    assert res.exit_code == 7
    assert res.stdout.strip() == "hello"
    assert runner.calls[-1][:3] == ["docker", "exec", "vllm-35b-prod"]
    assert runner.calls[-1][-3:] == ["python3", "-c", "print(1)"]


# ─── socket backend (injected transport) ──────────────────────────────


class _FakeTransport:
    """Stands in for the unix-socket Docker Engine API."""

    def __init__(self, routes):
        self.routes = routes
        self.calls: list[tuple[str, str]] = []

    def __call__(self, method, path, body=None):
        self.calls.append((method, path))
        for (m, prefix), resp in self.routes.items():
            if m == method and path.startswith(prefix):
                return resp
        return (404, b"{}")


def test_socket_list_parses_and_scopes():
    payload = json.dumps([
        {"Names": ["/vllm-35b-prod"], "Image": "vllm/vllm-openai:nightly",
         "State": "running", "Status": "Up 2 hours", "Id": "abc", "Created": 1000,
         "Ports": [{"PublicPort": 8101, "PrivatePort": 8101, "Type": "tcp"}], "Labels": {}},
        {"Names": ["/redis"], "Image": "redis:7", "State": "running",
         "Status": "Up", "Id": "zzz", "Created": 2000, "Ports": [], "Labels": {}},
    ]).encode()
    tr = _FakeTransport({("GET", "/containers/json"): (200, payload)})
    ctrl = co.SocketContainerControl(transport=tr)
    items = ctrl.list_managed()
    names = {c.name for c in items}
    assert names == {"vllm-35b-prod"}


def test_socket_lifecycle_posts_and_scopes():
    tr = _FakeTransport({("POST", "/containers/vllm-35b-prod/restart"): (204, b"")})
    ctrl = co.SocketContainerControl(transport=tr)
    ctrl.restart("vllm-35b-prod")
    assert ("POST", "/containers/vllm-35b-prod/restart") in tr.calls
    with pytest.raises(co.NotManagedError):
        ctrl.start("redis")


def test_socket_label_managed_even_with_foreign_name():
    payload = json.dumps([
        {"Names": ["/my-custom-engine"], "Image": "x", "State": "running",
         "Status": "Up", "Id": "id1", "Created": 1, "Ports": [],
         "Labels": {"sndr.managed": "true"}},
    ]).encode()
    tr = _FakeTransport({("GET", "/containers/json"): (200, payload)})
    ctrl = co.SocketContainerControl(transport=tr)
    assert {c.name for c in ctrl.list_managed()} == {"my-custom-engine"}


# ─── log stream demux (socket multiplexed frames) ─────────────────────


def test_demux_docker_stream_strips_frame_headers():
    # Two frames: stdout "ok\n", stderr "err\n" (8-byte header: type, 0,0,0, len32)
    import struct
    frame1 = struct.pack(">BxxxL", 1, 3) + b"ok\n"
    frame2 = struct.pack(">BxxxL", 2, 4) + b"err\n"
    out = co.demux_docker_stream(frame1 + frame2)
    assert out == "ok\nerr\n"
