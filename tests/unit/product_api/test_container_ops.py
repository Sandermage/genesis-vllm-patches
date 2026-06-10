# SPDX-License-Identifier: Apache-2.0
"""Tests for scoped container management (whitelist, gating, both backends).

No real docker / SSH is touched: the SSH backend runs through an injected
``runner`` that records the docker argv, and the socket backend runs through an
injected ``transport`` that returns canned Docker Engine API payloads.
"""
from __future__ import annotations

import json

import pytest

from sndr.product_api.legacy import container_ops as co


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


def test_ssh_stats_normalized_to_socket_shape():
    """SSH and socket backends must return the SAME stats keys (regression: the
    CLI shape leaked CPUPerc/MemUsage and crashed the frontend's .toFixed)."""
    cli = json.dumps({"CPUPerc": "12.50%", "MemUsage": "1.19GiB / 24GiB", "MemPerc": "4.97%",
                      "NetIO": "1.2kB / 3.4kB", "BlockIO": "0B / 8MB", "PIDs": "30"})
    ctrl, _ = _ssh({"stats": (0, cli + "\n", "")})
    s = ctrl.stats("vllm-35b-prod")
    # Same compact contract the socket backend returns (regression: .toFixed crash).
    assert {"cpu_pct", "mem_usage", "mem_limit", "mem_pct", "net_rx", "net_tx",
            "blk_read", "blk_write", "pids"} <= set(s)
    assert s["cpu_pct"] == 12.5
    assert s["mem_pct"] == 4.97
    assert s["mem_limit"] == 24 * 1024 ** 3
    assert abs(s["mem_usage"] - 1.19 * 1024 ** 3) < 1024 ** 3  # ~1.19 GiB
    assert s["pids"] == 30
    assert s["blk_write"] == 8 * 1000 ** 2


def test_parse_size_handles_units():
    assert co._parse_size("0B") == 0
    assert co._parse_size("512MiB") == 512 * 1024 ** 2
    assert co._parse_size("2GB") == 2 * 1000 ** 3
    assert co._parse_size("garbage") == 0


def test_ssh_exec_quotes_argv_and_scopes():
    ctrl, runner = _ssh({"exec": (7, "hello\n", "")})
    res = ctrl.exec("vllm-35b-prod", ["python3", "-c", "print(1)"])
    assert res.exit_code == 7
    assert res.stdout.strip() == "hello"
    assert runner.calls[-1][:3] == ["docker", "exec", "vllm-35b-prod"]
    assert runner.calls[-1][-3:] == ["python3", "-c", "print(1)"]


def test_ssh_top_parses_processes():
    ctrl, _ = _ssh({"top": (0, "PID USER CMD\n117 root VLLM::EngineCore\n118 root VLLM::Worker_TP0\n", "")})
    t = ctrl.top("vllm-35b-prod")
    assert t["titles"] == ["PID", "USER", "CMD"]
    assert t["processes"][0] == ["117", "root", "VLLM::EngineCore"]
    assert len(t["processes"]) == 2


def test_ssh_changes_parses_diff():
    ctrl, _ = _ssh({"diff": (0, "C /tmp\nA /tmp/new\nD /gone\n", "")})
    ch = ctrl.changes("vllm-35b-prod")
    assert {"kind": "added", "path": "/tmp/new"} in ch
    assert {"kind": "deleted", "path": "/gone"} in ch
    assert {"kind": "modified", "path": "/tmp"} in ch


def test_ssh_pull_uses_image_from_inspect():
    inspect = json.dumps([{"Config": {"Image": "vllm/vllm-openai:nightly"}}])
    ctrl, runner = _ssh({"inspect": (0, inspect, ""), "pull": (0, "Pulling…\nDigest: sha256:x\n", "")})
    r = ctrl.pull("vllm-35b-prod")
    assert r["image"] == "vllm/vllm-openai:nightly"
    assert ["docker", "pull", "vllm/vllm-openai:nightly"] in runner.calls


def test_list_dir_parses_ls_and_requires_abs_path():
    ls = ("total 12\n"
          "drwxr-xr-x 2 root root 4096 2026-06-03 10:00 etc\n"
          "-rw-r--r-- 1 root root 1234 2026-06-03 11:00 config.json\n"
          "lrwxrwxrwx 1 root root    7 2026-06-03 11:00 link -> config.json\n")
    ctrl, _ = _ssh({"exec": (0, ls, "")})
    res = ctrl.list_dir("vllm-35b-prod", "/app")
    names = [e["name"] for e in res["entries"]]
    assert names[0] == "etc" and res["entries"][0]["is_dir"]  # dirs first
    cfg = next(e for e in res["entries"] if e["name"] == "config.json")
    assert cfg["size"] == 1234 and not cfg["is_dir"]
    link = next(e for e in res["entries"] if e["name"] == "link")
    assert link["is_link"] and link["link_target"] == "config.json"
    with pytest.raises(ValueError):
        ctrl.list_dir("vllm-35b-prod", "relative/path")


def test_read_file_returns_content():
    ctrl, runner = _ssh({"exec": (0, '{"k": 1}', "")})
    res = ctrl.read_file("vllm-35b-prod", "/app/config.json")
    assert res["content"] == '{"k": 1}'
    assert runner.calls[-1][:3] == ["docker", "exec", "vllm-35b-prod"]
    assert "head" in runner.calls[-1] and "/app/config.json" in runner.calls[-1]


def test_ssh_pool_reuses_warm_client():
    connects = []
    pool = co._SshPool(
        connect=lambda t, to: connects.append(object()) or connects[-1],
        run_cmd=lambda c, cmd, to: (0, cmd, ""),
    )
    target = {"host": "h1", "port": 22, "user": "u"}
    pool.run(target, 5, "docker ps")
    pool.run(target, 5, "docker stats")
    assert len(connects) == 1  # second call reused the warm connection


def test_ssh_pool_separates_hosts():
    connects = []
    pool = co._SshPool(connect=lambda t, to: connects.append(object()) or connects[-1],
                       run_cmd=lambda c, cmd, to: (0, "", ""))
    pool.run({"host": "a"}, 5, "x")
    pool.run({"host": "b"}, 5, "x")
    assert len(connects) == 2  # distinct hosts → distinct clients


def test_ssh_pool_reconnects_on_broken_connection():
    state = {"runs": 0, "connects": 0}

    def run_cmd(c, cmd, to):
        state["runs"] += 1
        if state["runs"] == 1:
            raise OSError("broken pipe")
        return (0, "recovered", "")

    pool = co._SshPool(connect=lambda t, to: state.__setitem__("connects", state["connects"] + 1) or object(),
                       run_cmd=run_cmd)
    rc, out, _ = pool.run({"host": "h"}, 5, "docker ps")
    assert out == "recovered" and state["connects"] == 2  # initial + reconnect


def test_ssh_list_stats_batches_and_scopes():
    lines = "\n".join(json.dumps(r) for r in [
        {"Name": "vllm-35b-prod", "CPUPerc": "5.0%", "MemUsage": "1GiB / 24GiB", "MemPerc": "4%"},
        {"Name": "postgres", "CPUPerc": "1%", "MemUsage": "100MiB / 24GiB", "MemPerc": "0.4%"},
    ])
    ctrl, runner = _ssh({"stats": (0, lines + "\n", "")})
    s = ctrl.list_stats()
    assert "vllm-35b-prod" in s and "postgres" not in s  # foreign filtered
    assert s["vllm-35b-prod"]["cpu_pct"] == 5.0
    assert runner.calls[-1][:3] == ["docker", "stats", "--no-stream"]  # one call, no name


def test_ssh_update_settings_builds_docker_update():
    ctrl, runner = _ssh()
    ctrl.update_settings("vllm-35b-prod", cpus=2.0, memory=2147483648, restart_policy="always")
    call = runner.calls[-1]
    assert call[:2] == ["docker", "update"]
    assert "--cpus" in call and "2.0" in call
    assert "--memory" in call and "2147483648" in call
    assert "--restart" in call and "always" in call
    assert call[-1] == "vllm-35b-prod"


def test_update_settings_validates():
    ctrl, _ = _ssh()
    with pytest.raises(ValueError):
        ctrl.update_settings("vllm-35b-prod", restart_policy="bogus")
    with pytest.raises(co.NotManagedError):
        ctrl.update_settings("postgres", cpus=1)


def test_ssh_network_connect_disconnect_and_scope():
    ctrl, runner = _ssh()
    ctrl.connect_network("vllm-35b-prod", "frontends")
    assert runner.calls[-1] == ["docker", "network", "connect", "frontends", "vllm-35b-prod"]
    ctrl.disconnect_network("vllm-35b-prod", "frontends")
    assert runner.calls[-1] == ["docker", "network", "disconnect", "frontends", "vllm-35b-prod"]
    with pytest.raises(ValueError):
        ctrl.connect_network("vllm-35b-prod", "bad net; rm -rf")   # injection rejected


def test_ssh_list_networks_parses():
    lines = "\n".join(json.dumps(n) for n in [
        {"Name": "bridge", "Driver": "bridge", "Scope": "local"},
        {"Name": "host", "Driver": "host", "Scope": "local"},
    ])
    ctrl, _ = _ssh({"network": (0, lines + "\n", "")})
    nets = ctrl.list_networks()
    assert {n["name"] for n in nets} == {"bridge", "host"}


def test_socket_update_settings_posts_body():
    captured = {}

    class _T:
        def __call__(self, method, path, body=None):
            captured["method"] = method; captured["path"] = path
            captured["body"] = json.loads(body) if body else None
            return (200, b"{}")
    ctrl = co.SocketContainerControl(transport=_T())
    ctrl.update_settings("vllm-35b-prod", cpus=1.5, memory=1000, restart_policy="unless-stopped")
    assert captured["path"] == "/containers/vllm-35b-prod/update"
    assert captured["body"]["NanoCPUs"] == int(1.5e9)
    assert captured["body"]["Memory"] == 1000
    assert captured["body"]["RestartPolicy"] == {"Name": "unless-stopped"}


def test_engine_port_extraction():
    insp = {"NetworkSettings": {"Ports": {"9000/udp": [{"HostPort": "9000"}], "8101/tcp": [{"HostPort": "8101"}]}}}
    assert co._engine_port(insp) == 8101
    insp2 = {"HostConfig": {"PortBindings": {"8102/tcp": [{"HostPort": "18102"}]}}}
    assert co._engine_port(insp2) == 18102
    assert co._engine_port({}) is None


def test_ssh_engine_health_probes_published_port():
    inspect = json.dumps([{"NetworkSettings": {"Ports": {"8101/tcp": [{"HostPort": "8101"}]}}}])
    ctrl, runner = _ssh({"inspect": (0, inspect, ""), "-sf": (0, "200", "")})
    h = ctrl.engine_health("vllm-35b-prod")
    assert h["reachable"] is True and h["port"] == 8101 and h["status_code"] == 200
    # the probe curls the host-local engine port
    assert any("http://127.0.0.1:8101/health" in c for c in runner.calls[-1])
    with pytest.raises(co.NotManagedError):
        ctrl.engine_health("postgres")


def test_ssh_engine_health_no_port():
    ctrl, _ = _ssh({"inspect": (0, json.dumps([{"NetworkSettings": {"Ports": {}}}]), "")})
    h = ctrl.engine_health("vllm-35b-prod")
    assert h["reachable"] is False and h["port"] is None


def test_ssh_system_df_normalizes():
    rows = "\n".join(json.dumps(r) for r in [
        {"Type": "Images", "TotalCount": "5", "Active": "3", "Size": "2.5GB", "Reclaimable": "1GB (40%)"},
        {"Type": "Containers", "TotalCount": "3", "Active": "2", "Size": "100MB", "Reclaimable": "0B"},
    ])
    ctrl, _ = _ssh({"system": (0, rows + "\n", "")})
    df = ctrl.system_df()
    images = next(t for t in df["types"] if t["type"] == "Images")
    assert images["total_count"] == 5 and images["active"] == 3
    assert images["size"] == int(2.5 * 1000 ** 3)
    assert images["reclaimable"] == 1000 ** 3
    assert df["total_size"] > 0


def test_ssh_scan_image_grype_counts_by_severity():
    inspect = json.dumps([{"Config": {"Image": "vllm/vllm-openai:nightly"}}])
    grype = json.dumps({"matches": [
        {"vulnerability": {"severity": "High"}},
        {"vulnerability": {"severity": "Critical"}},
        {"vulnerability": {"severity": "High"}},
    ]})
    ctrl, _ = _ssh({
        "inspect": (0, inspect, ""),
        "-c": (0, "grype\n", ""),                       # scanner detection
        "vllm/vllm-openai:nightly": (0, grype, ""),     # grype <image> …
    })
    res = ctrl.scan_image("vllm-35b-prod")
    assert res["available"] and res["scanner"] == "grype"
    assert res["counts"]["high"] == 2 and res["counts"]["critical"] == 1
    assert res["total"] == 3


def test_ssh_scan_image_no_scanner():
    inspect = json.dumps([{"Config": {"Image": "img:latest"}}])
    ctrl, _ = _ssh({"inspect": (0, inspect, ""), "-c": (0, "none\n", "")})
    res = ctrl.scan_image("vllm-35b-prod")
    assert res["available"] is False and "scanner" in res["reason"]


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


def test_socket_top_and_changes():
    top = json.dumps({"Titles": ["PID", "CMD"], "Processes": [["1", "python3"]]}).encode()
    diff = json.dumps([{"Path": "/a", "Kind": 1}, {"Path": "/b", "Kind": 2}]).encode()
    tr = _FakeTransport({
        ("GET", "/containers/vllm-35b-prod/top"): (200, top),
        ("GET", "/containers/vllm-35b-prod/changes"): (200, diff),
    })
    ctrl = co.SocketContainerControl(transport=tr)
    assert ctrl.top("vllm-35b-prod")["titles"] == ["PID", "CMD"]
    ch = ctrl.changes("vllm-35b-prod")
    assert {"kind": "added", "path": "/a"} in ch and {"kind": "deleted", "path": "/b"} in ch
    with pytest.raises(co.NotManagedError):
        ctrl.top("redis")


def test_socket_system_df_and_scan_unavailable():
    df = json.dumps({
        "Images": [{"Size": 1000, "SharedSize": 400, "Containers": 1}],
        "Containers": [{"SizeRw": 50, "State": "running"}],
        "Volumes": [{"UsageData": {"Size": 200}}],
        "BuildCache": [{"Size": 10}],
    }).encode()
    tr = _FakeTransport({("GET", "/system/df"): (200, df)})
    ctrl = co.SocketContainerControl(transport=tr)
    out = ctrl.system_df()
    images = next(t for t in out["types"] if t["type"] == "Images")
    assert images["size"] == 1000 and images["reclaimable"] == 600
    assert out["total_size"] == 1260
    # scanning over the socket is honestly reported as unavailable
    scan = ctrl.scan_image("vllm-35b-prod")
    assert scan["available"] is False


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


def test_frame_demux_handles_split_frames_and_tty():
    import struct
    d = co._FrameDemux()
    frame = struct.pack(">BxxxL", 1, 5) + b"hello"
    assert d.feed(frame[:3]) == ""        # header not yet complete
    assert d.feed(frame[3:]) == "hello"   # completes across the boundary
    # A second frame split mid-payload.
    f2 = struct.pack(">BxxxL", 2, 4) + b"errs"
    assert d.feed(f2[:10]) == ""          # 8-byte header + 2 payload bytes
    assert d.feed(f2[10:]) == "errs"
    # TTY/raw (no frame header) passes straight through.
    assert co._FrameDemux().feed(b"plain log line\n") == "plain log line\n"


def test_chunked_decoder_reassembles_and_passthrough():
    d = co._ChunkedDecoder(enabled=True)
    assert d.feed(b"5\r\nhel") == b""          # waits for full chunk + CRLF
    assert d.feed(b"lo\r\n") == b"hello"
    assert d.feed(b"3\r\nabc\r\n0\r\n\r\n") == b"abc"  # then terminal chunk
    assert co._ChunkedDecoder(enabled=False).feed(b"raw bytes") == b"raw bytes"


def test_stream_logs_is_whitelist_scoped():
    ctrl = co.SocketContainerControl(transport=lambda *a, **k: (404, b""))
    with pytest.raises(co.NotManagedError):
        ctrl.stream_logs("redis")  # rejected before any connection attempt


def test_demux_docker_stream_strips_frame_headers():
    # Two frames: stdout "ok\n", stderr "err\n" (8-byte header: type, 0,0,0, len32)
    import struct
    frame1 = struct.pack(">BxxxL", 1, 3) + b"ok\n"
    frame2 = struct.pack(">BxxxL", 2, 4) + b"err\n"
    out = co.demux_docker_stream(frame1 + frame2)
    assert out == "ok\nerr\n"
