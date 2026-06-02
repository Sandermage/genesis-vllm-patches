# SPDX-License-Identifier: Apache-2.0
"""Scoped container management for the SNDR Control Center.

One contract (:class:`ContainerControl`), two transports:

* :class:`SshContainerControl` — runs ``docker …`` on a host over the existing
  SSH channel (used by the central GUI for a Host card; no node change needed).
* :class:`SocketContainerControl` — talks the Docker Engine API directly over a
  mounted ``/var/run/docker.sock`` (used natively by a node's management daemon;
  no third-party SDK, just stdlib).

Two safety layers, enforced regardless of transport:

* **Whitelist (security boundary).** Every operation goes through
  :func:`ensure_managed`; only vLLM/engine containers (name prefix or the
  ``sndr.managed`` label) are reachable. A foreign or malformed name never
  reaches a shell or the socket. This lives in the BASE class so neither backend
  can skip it.
* **Gating (defense in depth).** Read ops are ungated. Lifecycle requires
  ``SNDR_ENABLE_APPLY`` + an explicit confirm; ``exec`` additionally requires
  ``SNDR_ENABLE_EXEC`` (off by default — arbitrary in-container execution is
  strictly more dangerous). The pure :func:`gate_lifecycle` / :func:`gate_exec`
  helpers are consumed by the HTTP layer so gating is unit-testable without HTTP.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# A docker container/object name: same shape ssh_client validates, so an SSH
# target can never carry a shell metacharacter.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_DEFAULT_MANAGED_PREFIXES = ("vllm", "sndr-daemon")
_MANAGED_LABEL = "sndr.managed"
_TRUTHY = {"1", "true", "yes", "on"}


# ─── env helpers ───────────────────────────────────────────────────────


def _env_true(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in _TRUTHY


def exec_enabled() -> bool:
    """Whether in-container ``exec`` is unlocked (separate from apply)."""
    return _env_true("SNDR_ENABLE_EXEC")


def managed_prefixes() -> tuple[str, ...]:
    """Container-name prefixes considered SNDR-managed (env-overridable)."""
    raw = os.environ.get("SNDR_MANAGED_PREFIXES", "").strip()
    if not raw:
        return _DEFAULT_MANAGED_PREFIXES
    parts = tuple(p.strip().lower() for p in raw.split(",") if p.strip())
    return parts or _DEFAULT_MANAGED_PREFIXES


# ─── whitelist ─────────────────────────────────────────────────────────


class NotManagedError(Exception):
    """Raised when an operation targets a non-managed or malformed container.

    Maps to HTTP 403 at the API layer."""


class ContainerOpError(Exception):
    """Transport/engine failure. Carries an HTTP-ish status for the API layer."""

    def __init__(self, message: str, *, status: int = 502) -> None:
        super().__init__(message)
        self.status = status


def is_managed_name(name: Optional[str], *, prefixes: Optional[tuple[str, ...]] = None) -> bool:
    """True when ``name`` matches a managed prefix (engine/daemon container)."""
    n = (name or "").strip().lower()
    if not n:
        return False
    for p in (prefixes or managed_prefixes()):
        if n == p or n.startswith(p):
            return True
    return False


def ensure_managed(name: str, *, prefixes: Optional[tuple[str, ...]] = None) -> None:
    """Guard: raise :class:`NotManagedError` unless ``name`` is valid AND managed."""
    if not _NAME_RE.match(name or ""):
        raise NotManagedError(f"invalid container name: {name!r}")
    if not is_managed_name(name, prefixes=prefixes):
        raise NotManagedError(f"container not managed by SNDR: {name!r}")


# ─── gating ────────────────────────────────────────────────────────────


@dataclass
class GateResult:
    allowed: bool
    status: int = 200
    reason: str = ""


def gate_lifecycle(*, apply_on: bool, confirm: bool) -> GateResult:
    if not apply_on:
        return GateResult(False, 403, "apply is disabled — start the daemon with SNDR_ENABLE_APPLY=1")
    if not confirm:
        return GateResult(False, 400, "explicit confirm:true is required")
    return GateResult(True)


def gate_exec(*, apply_on: bool, exec_on: bool, confirm: bool) -> GateResult:
    base = gate_lifecycle(apply_on=apply_on, confirm=confirm)
    if not base.allowed:
        return base
    if not exec_on:
        return GateResult(False, 403, "container exec is disabled — start the daemon with SNDR_ENABLE_EXEC=1")
    return GateResult(True)


# ─── data ──────────────────────────────────────────────────────────────


@dataclass
class ManagedContainer:
    name: str
    id: str
    image: str
    state: str
    status: str
    ports: str
    created: str
    labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "id": self.id, "image": self.image,
            "state": self.state, "status": self.status, "ports": self.ports,
            "created": self.created, "labels": self.labels,
        }


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, Any]:
        return {"exit_code": self.exit_code, "stdout": self.stdout, "stderr": self.stderr}


def _parse_label_string(raw: str) -> dict[str, str]:
    """Parse docker CLI's comma-joined ``k=v,k2=v2`` label string into a dict."""
    out: dict[str, str] = {}
    for piece in (raw or "").split(","):
        piece = piece.strip()
        if "=" in piece:
            k, v = piece.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def demux_docker_stream(raw: bytes) -> str:
    """Strip Docker's 8-byte multiplexing frame headers from a log/exec stream.

    Non-TTY streams are framed: ``[type:1][000][len:uint32 BE][payload]``. TTY
    streams are raw. We detect frames and fall back to a plain decode when the
    bytes don't look framed (so both shapes decode cleanly)."""
    if not raw:
        return ""
    chunks: list[bytes] = []
    i, n = 0, len(raw)
    while i + 8 <= n:
        stream_type = raw[i]
        if stream_type not in (0, 1, 2):  # not a frame header → treat the rest as raw
            return raw.decode("utf-8", "replace")
        (length,) = struct.unpack(">L", raw[i + 4:i + 8])
        start = i + 8
        end = start + length
        if end > n:  # truncated frame → take what's left
            chunks.append(raw[start:n])
            i = n
            break
        chunks.append(raw[start:end])
        i = end
    if i < n:  # trailing bytes that weren't a full frame
        chunks.append(raw[i:n])
    return b"".join(chunks).decode("utf-8", "replace")


# ─── contract ──────────────────────────────────────────────────────────


class ContainerControl(ABC):
    """Backend-agnostic, whitelist-enforcing container control surface."""

    def __init__(self, *, prefixes: Optional[tuple[str, ...]] = None) -> None:
        self._prefixes = prefixes or managed_prefixes()

    # public, guarded API ------------------------------------------------
    def list_managed(self) -> list[ManagedContainer]:
        return [c for c in self._raw_list() if self._is_managed(c)]

    def inspect(self, name: str) -> dict[str, Any]:
        self._ensure(name)
        return self._raw_inspect(name)

    def logs(self, name: str, *, tail: int = 200) -> str:
        self._ensure(name)
        return self._raw_logs(name, tail=tail)

    def stats(self, name: str) -> dict[str, Any]:
        self._ensure(name)
        return self._raw_stats(name)

    def start(self, name: str) -> None:
        self._ensure(name)
        self._raw_lifecycle(name, "start")

    def stop(self, name: str) -> None:
        self._ensure(name)
        self._raw_lifecycle(name, "stop")

    def restart(self, name: str) -> None:
        self._ensure(name)
        self._raw_lifecycle(name, "restart")

    def exec(self, name: str, argv: list[str], *, timeout: float = 30.0) -> ExecResult:
        self._ensure(name)
        if not argv:
            raise ValueError("exec requires a non-empty argv")
        return self._raw_exec(name, list(argv), timeout=timeout)

    # helpers ------------------------------------------------------------
    def _ensure(self, name: str) -> None:
        ensure_managed(name, prefixes=self._prefixes)

    def _is_managed(self, c: ManagedContainer) -> bool:
        if str(c.labels.get(_MANAGED_LABEL, "")).strip().lower() in _TRUTHY:
            return True
        return is_managed_name(c.name, prefixes=self._prefixes)

    # raw transport ops (no guard — base class already guarded) -----------
    @abstractmethod
    def _raw_list(self) -> list[ManagedContainer]: ...
    @abstractmethod
    def _raw_inspect(self, name: str) -> dict[str, Any]: ...
    @abstractmethod
    def _raw_logs(self, name: str, *, tail: int) -> str: ...
    @abstractmethod
    def _raw_stats(self, name: str) -> dict[str, Any]: ...
    @abstractmethod
    def _raw_lifecycle(self, name: str, action: str) -> None: ...
    @abstractmethod
    def _raw_exec(self, name: str, argv: list[str], *, timeout: float) -> ExecResult: ...


# ─── SSH backend ───────────────────────────────────────────────────────

# runner(argv: list[str]) -> (rc, stdout, stderr). Injectable for tests; the
# default opens an SSH client to the target and runs the (safely quoted) command.
SshRunner = Callable[[list[str]], "tuple[int, str, str]"]


def _default_ssh_runner(target: dict[str, Any], timeout: float) -> SshRunner:
    from . import ssh_client

    def run(argv: list[str]) -> tuple[int, str, str]:
        client = ssh_client._open_client(target, timeout)  # noqa: SLF001 — intra-package reuse
        try:
            command = " ".join(shlex.quote(tok) for tok in argv)
            return ssh_client._exec(client, command, timeout)  # noqa: SLF001
        finally:
            try:
                client.close()
            except Exception:
                pass

    return run


class SshContainerControl(ContainerControl):
    """Control containers on a host by running ``docker`` over SSH."""

    def __init__(self, *, target: Optional[dict[str, Any]] = None, runner: Optional[SshRunner] = None,
                 timeout: float = 15.0, prefixes: Optional[tuple[str, ...]] = None) -> None:
        super().__init__(prefixes=prefixes)
        if runner is None:
            if target is None:
                raise ValueError("SshContainerControl needs a target or a runner")
            runner = _default_ssh_runner(target, timeout)
        self._run = runner

    def _docker(self, *args: str) -> tuple[int, str, str]:
        return self._run(["docker", *args])

    def _raw_list(self) -> list[ManagedContainer]:
        rc, out, err = self._docker("ps", "-a", "--format", "{{json .}}")
        if rc != 0:
            raise ContainerOpError(err.strip() or "docker ps failed")
        items: list[ManagedContainer] = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            items.append(ManagedContainer(
                name=str(d.get("Names", "")).split(",")[0],
                id=str(d.get("ID", "")), image=str(d.get("Image", "")),
                state=str(d.get("State", "")), status=str(d.get("Status", "")),
                ports=str(d.get("Ports", "")), created=str(d.get("CreatedAt", "")),
                labels=_parse_label_string(str(d.get("Labels", ""))),
            ))
        return items

    def _raw_inspect(self, name: str) -> dict[str, Any]:
        rc, out, err = self._docker("inspect", name)
        if rc != 0:
            raise ContainerOpError(err.strip() or f"docker inspect {name} failed", status=404)
        try:
            data = json.loads(out)
        except json.JSONDecodeError as exc:
            raise ContainerOpError(f"could not parse docker inspect output: {exc}") from exc
        return data[0] if isinstance(data, list) and data else {}

    def _raw_logs(self, name: str, *, tail: int) -> str:
        rc, out, err = self._docker("logs", "--tail", str(int(tail)), name)
        # docker logs writes app output to BOTH streams; concatenate for the view.
        return out + err

    def _raw_stats(self, name: str) -> dict[str, Any]:
        rc, out, err = self._docker("stats", "--no-stream", "--format", "{{json .}}", name)
        if rc != 0:
            raise ContainerOpError(err.strip() or f"docker stats {name} failed")
        try:
            return json.loads(out.strip().splitlines()[0]) if out.strip() else {}
        except (json.JSONDecodeError, IndexError):
            return {}

    def _raw_lifecycle(self, name: str, action: str) -> None:
        rc, out, err = self._docker(action, name)
        if rc != 0:
            raise ContainerOpError(err.strip() or f"docker {action} {name} failed")

    def _raw_exec(self, name: str, argv: list[str], *, timeout: float) -> ExecResult:
        rc, out, err = self._docker("exec", name, *argv)
        return ExecResult(exit_code=rc, stdout=out, stderr=err)


# ─── socket backend ────────────────────────────────────────────────────

# transport(method, path, body) -> (status, bytes). Injectable for tests; the
# default speaks HTTP over the unix docker socket.
SocketTransport = Callable[[str, str, Optional[bytes]], "tuple[int, bytes]"]


def _default_socket_transport(sock_path: str, timeout: float) -> SocketTransport:
    import http.client
    import socket

    class _UnixHTTPConnection(http.client.HTTPConnection):
        def __init__(self) -> None:
            super().__init__("localhost", timeout=timeout)

        def connect(self) -> None:  # type: ignore[override]
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect(sock_path)
            self.sock = s

    def transport(method: str, path: str, body: Optional[bytes] = None) -> tuple[int, bytes]:
        conn = _UnixHTTPConnection()
        try:
            headers = {"Host": "localhost"}
            if body is not None:
                headers["Content-Type"] = "application/json"
            conn.request(method, path, body=body, headers=headers)
            resp = conn.getresponse()
            return resp.status, resp.read()
        finally:
            try:
                conn.close()
            except Exception:
                pass

    return transport


class SocketContainerControl(ContainerControl):
    """Control containers via the Docker Engine API over the unix socket."""

    def __init__(self, *, transport: Optional[SocketTransport] = None,
                 sock_path: str = "/var/run/docker.sock", timeout: float = 15.0,
                 prefixes: Optional[tuple[str, ...]] = None) -> None:
        super().__init__(prefixes=prefixes)
        self._transport = transport or _default_socket_transport(sock_path, timeout)

    def _json(self, method: str, path: str, body: Optional[dict[str, Any]] = None) -> tuple[int, Any]:
        payload = json.dumps(body).encode() if body is not None else None
        status, raw = self._transport(method, path, payload)
        if not raw:
            return status, None
        try:
            return status, json.loads(raw.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            return status, None

    def _raw_list(self) -> list[ManagedContainer]:
        status, data = self._json("GET", "/containers/json?all=1")
        if status != 200 or not isinstance(data, list):
            raise ContainerOpError(f"docker API /containers/json returned {status}")
        items: list[ManagedContainer] = []
        for d in data:
            names = d.get("Names") or []
            name = (names[0] if names else "").lstrip("/")
            ports = ", ".join(
                f"{p.get('PublicPort')}->{p.get('PrivatePort')}/{p.get('Type')}"
                if p.get("PublicPort") else f"{p.get('PrivatePort')}/{p.get('Type')}"
                for p in (d.get("Ports") or [])
            )
            items.append(ManagedContainer(
                name=name, id=str(d.get("Id", ""))[:12], image=str(d.get("Image", "")),
                state=str(d.get("State", "")), status=str(d.get("Status", "")),
                ports=ports, created=str(d.get("Created", "")),
                labels={str(k): str(v) for k, v in (d.get("Labels") or {}).items()},
            ))
        return items

    def _raw_inspect(self, name: str) -> dict[str, Any]:
        status, data = self._json("GET", f"/containers/{name}/json")
        if status == 404:
            raise ContainerOpError(f"no such container: {name}", status=404)
        if status != 200 or not isinstance(data, dict):
            raise ContainerOpError(f"docker API inspect returned {status}")
        return data

    def _raw_logs(self, name: str, *, tail: int) -> str:
        status, raw = self._transport(
            "GET", f"/containers/{name}/logs?stdout=1&stderr=1&tail={int(tail)}", None)
        if status != 200:
            raise ContainerOpError(f"docker API logs returned {status}")
        return demux_docker_stream(raw)

    def _raw_stats(self, name: str) -> dict[str, Any]:
        status, data = self._json("GET", f"/containers/{name}/stats?stream=0")
        if status != 200 or not isinstance(data, dict):
            raise ContainerOpError(f"docker API stats returned {status}")
        return _summarize_stats(data)

    def _raw_lifecycle(self, name: str, action: str) -> None:
        status, raw = self._transport("POST", f"/containers/{name}/{action}", None)
        # 204 = done, 304 = already in that state (treat as success).
        if status not in (204, 304):
            raise ContainerOpError(
                f"docker API {action} returned {status}: {raw.decode('utf-8', 'replace')[:200]}")

    def _raw_exec(self, name: str, argv: list[str], *, timeout: float) -> ExecResult:
        status, created = self._json("POST", f"/containers/{name}/exec", {
            "AttachStdout": True, "AttachStderr": True, "Tty": False, "Cmd": argv,
        })
        if status not in (200, 201) or not isinstance(created, dict) or not created.get("Id"):
            raise ContainerOpError(f"docker API exec create returned {status}")
        exec_id = str(created["Id"])
        s2, raw = self._transport("POST", f"/exec/{exec_id}/start",
                                  json.dumps({"Detach": False, "Tty": False}).encode())
        if s2 not in (200, 201):
            raise ContainerOpError(f"docker API exec start returned {s2}")
        output = demux_docker_stream(raw)
        s3, info = self._json("GET", f"/exec/{exec_id}/json")
        exit_code = int(info.get("ExitCode") or 0) if isinstance(info, dict) else 0
        return ExecResult(exit_code=exit_code, stdout=output, stderr="")


def _summarize_stats(d: dict[str, Any]) -> dict[str, Any]:
    """Reduce the verbose Docker stats payload to a compact GUI summary."""
    cpu_pct = 0.0
    try:
        cpu = d["cpu_stats"]
        pre = d["precpu_stats"]
        cpu_delta = cpu["cpu_usage"]["total_usage"] - pre["cpu_usage"]["total_usage"]
        sys_delta = cpu["system_cpu_usage"] - pre.get("system_cpu_usage", 0)
        ncpu = cpu.get("online_cpus") or len(cpu["cpu_usage"].get("percpu_usage") or [1])
        if sys_delta > 0 and cpu_delta > 0:
            cpu_pct = (cpu_delta / sys_delta) * ncpu * 100.0
    except (KeyError, TypeError, ZeroDivisionError):
        pass
    mem = d.get("memory_stats") or {}
    mem_usage = int(mem.get("usage") or 0)
    mem_limit = int(mem.get("limit") or 0)
    return {
        "cpu_pct": round(cpu_pct, 2),
        "mem_usage": mem_usage,
        "mem_limit": mem_limit,
        "mem_pct": round(mem_usage / mem_limit * 100.0, 2) if mem_limit else 0.0,
    }


__all__ = [
    "ManagedContainer", "ExecResult", "GateResult",
    "ContainerControl", "SshContainerControl", "SocketContainerControl",
    "NotManagedError", "ContainerOpError",
    "is_managed_name", "ensure_managed", "managed_prefixes", "exec_enabled",
    "gate_lifecycle", "gate_exec", "demux_docker_stream",
]
