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
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# A docker container/object name: same shape ssh_client validates, so an SSH
# target can never carry a shell metacharacter.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_DEFAULT_MANAGED_PREFIXES = ("vllm", "sndr-daemon")
_MANAGED_LABEL = "sndr.managed"
_TRUTHY = {"1", "true", "yes", "on"}
_RESTART_POLICIES = {"no", "always", "unless-stopped", "on-failure"}
_NET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _validate_net(network: str) -> None:
    if not _NET_RE.match(network or ""):
        raise ValueError(f"invalid network name: {network!r}")


def _engine_port(inspect: dict[str, Any]) -> Optional[int]:
    """First published host TCP port from a container inspect — the engine's
    externally reachable port. Checks NetworkSettings.Ports then PortBindings."""
    sources = [
        (inspect.get("NetworkSettings", {}) or {}).get("Ports", {}) or {},
        (inspect.get("HostConfig", {}) or {}).get("PortBindings", {}) or {},
    ]
    for ports in sources:
        for key, binds in ports.items():
            if str(key).endswith("/tcp") and binds:
                try:
                    return int(binds[0].get("HostPort"))
                except (ValueError, TypeError, AttributeError):
                    continue
    return None


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
    networks: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "id": self.id, "image": self.image,
            "state": self.state, "status": self.status, "ports": self.ports,
            "created": self.created, "labels": self.labels, "networks": self.networks,
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


def _validate_abs_path(path: str) -> None:
    """Require an absolute, control-char-free path (passed to exec as argv)."""
    if not path or not path.startswith("/"):
        raise ValueError("path must be absolute (start with /)")
    if "\x00" in path or any(ord(c) < 32 for c in path):
        raise ValueError("path contains control characters")
    if len(path) > 4096:
        raise ValueError("path too long")


def _parse_ls(text: str) -> list[dict[str, Any]]:
    """Parse ``ls -la --time-style=long-iso`` output into structured entries."""
    entries: list[dict[str, Any]] = []
    for line in (text or "").splitlines():
        line = line.rstrip()
        if not line or line.startswith("total "):
            continue
        parts = line.split(None, 7)
        if len(parts) < 8:
            continue
        perms, _links, owner, group, size, date, time_, name = parts
        is_link = perms.startswith("l")
        target = None
        if is_link and " -> " in name:
            name, target = name.split(" -> ", 1)
        if name in (".", ".."):
            continue
        try:
            size_i = int(size)
        except ValueError:
            size_i = 0
        entries.append({
            "name": name, "is_dir": perms.startswith("d"), "is_link": is_link,
            "link_target": target, "perms": perms, "owner": owner, "group": group,
            "size": size_i, "mtime": f"{date} {time_}",
        })
    entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))
    return entries


class _FrameDemux:
    """Incrementally strip Docker's 8-byte multiplex frame headers across chunk
    boundaries (for a streaming logs/attach socket). TTY streams (no headers)
    pass through unchanged."""

    def __init__(self) -> None:
        self._buf = b""

    def feed(self, chunk: bytes) -> str:
        self._buf += chunk
        out: list[str] = []
        while self._buf:
            if self._buf[0] not in (0, 1, 2):  # not a frame header → TTY/raw, flush
                out.append(self._buf.decode("utf-8", "replace"))
                self._buf = b""
                break
            if len(self._buf) < 8:
                break  # wait for the full header
            (length,) = struct.unpack(">L", self._buf[4:8])
            if len(self._buf) < 8 + length:
                break  # wait for the full payload
            out.append(self._buf[8:8 + length].decode("utf-8", "replace"))
            self._buf = self._buf[8 + length:]
        return "".join(out)


class _ChunkedDecoder:
    """Incrementally decode HTTP/1.1 ``Transfer-Encoding: chunked`` bodies. When
    the stream isn't chunked (``enabled=False``), it's a transparent pass-through."""

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._buf = b""

    def feed(self, chunk: bytes) -> bytes:
        if not self._enabled:
            return chunk
        self._buf += chunk
        out: list[bytes] = []
        while True:
            nl = self._buf.find(b"\r\n")
            if nl < 0:
                break
            size_line = self._buf[:nl]
            try:
                size = int(size_line.split(b";")[0].strip(), 16)
            except ValueError:
                break  # incomplete size line
            if size == 0:
                self._buf = b""  # terminal chunk
                break
            if len(self._buf) < nl + 2 + size + 2:
                break  # wait for the full chunk + trailing CRLF
            out.append(self._buf[nl + 2:nl + 2 + size])
            self._buf = self._buf[nl + 2 + size + 2:]
        return b"".join(out)


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

    def top(self, name: str) -> dict[str, Any]:
        """Running processes inside the container (docker top)."""
        self._ensure(name)
        return self._raw_top(name)

    def changes(self, name: str) -> list[dict[str, Any]]:
        """Filesystem changes vs the image (docker diff): A/C/D per path."""
        self._ensure(name)
        return self._raw_changes(name)

    def pull(self, name: str) -> dict[str, Any]:
        """Pull the latest image for the container's tag (does not recreate)."""
        self._ensure(name)
        return self._raw_pull(name)

    def system_df(self) -> dict[str, Any]:
        """Host-level disk usage (docker system df). Not container-scoped."""
        return self._raw_system_df()

    def scan_image(self, name: str) -> dict[str, Any]:
        """Scan the container's image for CVEs (grype/trivy) — safe-pull check."""
        self._ensure(name)
        return self._raw_scan_image(name)

    def stream_logs(self, name: str, *, tail: int = 200):
        """Yield live log text chunks (docker logs --follow). Caller closes the
        generator to tear down the underlying SSH channel / socket."""
        self._ensure(name)
        return self._raw_stream_logs(name, tail=tail)

    def engine_health(self, name: str) -> dict[str, Any]:
        """Is the vLLM engine INSIDE this container actually serving? Probes its
        published port's /health — distinguishes 'container running' from 'engine
        crashed/loading' (the real readiness signal)."""
        self._ensure(name)
        port = _engine_port(self._raw_inspect(name))
        if not port:
            return {"reachable": False, "port": None, "reason": "no published TCP port"}
        return self._raw_engine_probe(port)

    def list_stats(self) -> dict[str, dict[str, Any]]:
        """Live stats for ALL managed containers in one shot — name → summary.

        Over SSH this is a SINGLE connection + one ``docker stats`` for the whole
        set, instead of one SSH handshake per container per poll (the big
        responsiveness win for remote hosts)."""
        return self._raw_list_stats()

    # Editable settings — live `docker update` (no recreate) + network attach.
    def update_settings(self, name: str, *, cpus: Optional[float] = None,
                        memory: Optional[int] = None, restart_policy: Optional[str] = None) -> dict[str, Any]:
        self._ensure(name)
        if restart_policy is not None and restart_policy not in _RESTART_POLICIES:
            raise ValueError(f"invalid restart policy: {restart_policy!r} (allowed: {', '.join(sorted(_RESTART_POLICIES))})")
        if cpus is not None and (cpus < 0 or cpus > 1024):
            raise ValueError("cpus out of range")
        if memory is not None and memory < 0:
            raise ValueError("memory must be >= 0")
        return self._raw_update_settings(name, cpus=cpus, memory=memory, restart_policy=restart_policy)

    def connect_network(self, name: str, network: str) -> dict[str, Any]:
        self._ensure(name)
        _validate_net(network)
        return self._raw_network(name, network, connect=True)

    def disconnect_network(self, name: str, network: str) -> dict[str, Any]:
        self._ensure(name)
        _validate_net(network)
        return self._raw_network(name, network, connect=False)

    def list_networks(self) -> list[dict[str, Any]]:
        return self._raw_list_networks()

    # File browsing rides the exec transport (so it needs the SAME exec gate at
    # the HTTP layer). Paths are passed as argv (never a shell string), so there
    # is no metacharacter injection; we only require an absolute path.
    def list_dir(self, name: str, path: str) -> dict[str, Any]:
        self._ensure(name)
        _validate_abs_path(path)
        res = self._raw_exec(name, ["ls", "-la", "--time-style=long-iso", "--", path], timeout=15.0)
        if res.exit_code != 0:
            raise ContainerOpError(res.stderr.strip() or res.stdout.strip() or f"cannot list {path}", status=404)
        return {"path": path, "entries": _parse_ls(res.stdout)}

    def read_file(self, name: str, path: str, *, max_bytes: int = 65536) -> dict[str, Any]:
        self._ensure(name)
        _validate_abs_path(path)
        res = self._raw_exec(name, ["head", "-c", str(int(max_bytes)), "--", path], timeout=15.0)
        if res.exit_code != 0:
            raise ContainerOpError(res.stderr.strip() or f"cannot read {path}", status=404)
        return {"path": path, "content": res.stdout, "truncated": len(res.stdout.encode("utf-8", "replace")) >= max_bytes}

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
    @abstractmethod
    def _raw_top(self, name: str) -> dict[str, Any]: ...
    @abstractmethod
    def _raw_changes(self, name: str) -> list[dict[str, Any]]: ...
    @abstractmethod
    def _raw_pull(self, name: str) -> dict[str, Any]: ...
    @abstractmethod
    def _raw_system_df(self) -> dict[str, Any]: ...
    @abstractmethod
    def _raw_scan_image(self, name: str) -> dict[str, Any]: ...
    @abstractmethod
    def _raw_stream_logs(self, name: str, *, tail: int): ...
    @abstractmethod
    def _raw_list_stats(self) -> dict[str, dict[str, Any]]: ...
    @abstractmethod
    def _raw_engine_probe(self, port: int) -> dict[str, Any]: ...
    @abstractmethod
    def _raw_update_settings(self, name: str, *, cpus, memory, restart_policy) -> dict[str, Any]: ...
    @abstractmethod
    def _raw_network(self, name: str, network: str, *, connect: bool) -> dict[str, Any]: ...
    @abstractmethod
    def _raw_list_networks(self) -> list[dict[str, Any]]: ...


# ─── SSH backend ───────────────────────────────────────────────────────

# runner(argv: list[str]) -> (rc, stdout, stderr). Injectable for tests; the
# default opens an SSH client to the target and runs the (safely quoted) command.
SshRunner = Callable[[list[str]], "tuple[int, str, str]"]


class _SshPool:
    """Keep ONE warm SSH client per host and reuse it across requests, so short
    container ops (list/stats/inspect/top…) don't pay a fresh TCP+auth handshake
    every time — the main responsiveness win for remote hosts. Access per host is
    serialized by a lock (paramiko channels are cheap; the handshake is what's
    expensive). A broken connection is transparently reconnected once.

    ``connect`` and ``run_cmd`` are injectable so the pool is unit-testable
    without a real SSH server."""

    def __init__(self, connect: Callable[[dict[str, Any], float], Any],
                 run_cmd: Callable[[Any, str, float], "tuple[int, str, str]"]) -> None:
        self._connect = connect
        self._run_cmd = run_cmd
        self._clients: dict[Any, Any] = {}
        self._locks: dict[Any, "threading.Lock"] = {}
        self._guard = threading.Lock()

    @staticmethod
    def _key(target: dict[str, Any]) -> tuple:
        return (target.get("host"), int(target.get("port") or 22), str(target.get("user")))

    def _lock_for(self, key: tuple) -> "threading.Lock":
        with self._guard:
            return self._locks.setdefault(key, threading.Lock())

    def run(self, target: dict[str, Any], timeout: float, command: str) -> tuple[int, str, str]:
        key = self._key(target)
        with self._lock_for(key):
            client = self._clients.get(key)
            if client is None:
                client = self._clients[key] = self._connect(target, timeout)
            try:
                return self._run_cmd(client, command, timeout)
            except Exception:
                # Stale/broken connection → reconnect once and retry.
                try:
                    client.close()
                except Exception:
                    pass
                client = self._clients[key] = self._connect(target, timeout)
                return self._run_cmd(client, command, timeout)

    def close(self) -> None:
        with self._guard:
            for client in self._clients.values():
                try:
                    client.close()
                except Exception:
                    pass
            self._clients.clear()


def _make_default_pool() -> _SshPool:
    from . import ssh_client
    return _SshPool(
        connect=lambda t, to: ssh_client._open_client(t, to),  # noqa: SLF001
        run_cmd=lambda c, cmd, to: ssh_client._exec(c, cmd, to),  # noqa: SLF001
    )


_POOL: Optional[_SshPool] = None
_POOL_LOCK = threading.Lock()


def _pool() -> _SshPool:
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                _POOL = _make_default_pool()
    return _POOL


def _default_ssh_runner(target: dict[str, Any], timeout: float) -> SshRunner:
    """Pooled runner for short ops — reuses the host's warm connection."""
    def run(argv: list[str]) -> tuple[int, str, str]:
        command = " ".join(shlex.quote(tok) for tok in argv)
        return _pool().run(target, timeout, command)

    return run


def _direct_ssh_run(target: dict[str, Any], timeout: float, argv: list[str]) -> tuple[int, str, str]:
    """One-shot, NON-pooled connection for long ops (image scan) so they don't
    hold the per-host pool lock for minutes and stall stat polls."""
    from . import ssh_client
    client = ssh_client._open_client(target, timeout)  # noqa: SLF001
    try:
        return ssh_client._exec(client, " ".join(shlex.quote(t) for t in argv), timeout)  # noqa: SLF001
    finally:
        try:
            client.close()
        except Exception:
            pass


class SshContainerControl(ContainerControl):
    """Control containers on a host by running ``docker`` over SSH."""

    def __init__(self, *, target: Optional[dict[str, Any]] = None, runner: Optional[SshRunner] = None,
                 timeout: float = 15.0, prefixes: Optional[tuple[str, ...]] = None) -> None:
        super().__init__(prefixes=prefixes)
        self._target = target
        self._timeout = timeout
        if runner is None:
            if target is None:
                raise ValueError("SshContainerControl needs a target or a runner")
            runner = _default_ssh_runner(target, timeout)
        self._run = runner

    def _docker(self, *args: str) -> tuple[int, str, str]:
        return self._run(["docker", *args])

    def _run_slow(self, argv: list[str], timeout: float) -> tuple[int, str, str]:
        """Run a long command (e.g. an image scan) on a dedicated, NON-pooled
        connection so it doesn't hold the per-host pool lock. Falls back to the
        injected runner in tests."""
        if self._target is not None:
            return _direct_ssh_run(self._target, timeout, argv)
        return self._run(argv)

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
                networks=str(d.get("Networks", "")),
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
            raw = json.loads(out.strip().splitlines()[0]) if out.strip() else {}
        except (json.JSONDecodeError, IndexError):
            raw = {}
        # Normalize the CLI shape (CPUPerc/MemUsage strings) to the SAME compact
        # summary the socket backend returns, so the frontend has one contract.
        return _summarize_cli_stats(raw)

    def _raw_engine_probe(self, port: int) -> dict[str, Any]:
        # Probe the engine on the HOST (where the container publishes its port).
        rc, out, _ = self._run(["curl", "-sf", "-m", "3", "-o", "/dev/null",
                                "-w", "%{http_code}", f"http://127.0.0.1:{port}/health"])
        code = int(out.strip()) if out.strip().isdigit() else None
        return {"reachable": rc == 0 and code == 200, "port": port, "status_code": code}

    def _raw_list_stats(self) -> dict[str, dict[str, Any]]:
        rc, out, err = self._docker("stats", "--no-stream", "--format", "{{json .}}")
        if rc != 0:
            raise ContainerOpError(err.strip() or "docker stats failed")
        result: dict[str, dict[str, Any]] = {}
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = str(raw.get("Name") or raw.get("Container") or "").split(",")[0]
            if name and is_managed_name(name, prefixes=self._prefixes):
                result[name] = _summarize_cli_stats(raw)
        return result

    def _raw_lifecycle(self, name: str, action: str) -> None:
        rc, out, err = self._docker(action, name)
        if rc != 0:
            raise ContainerOpError(err.strip() or f"docker {action} {name} failed")

    def _raw_update_settings(self, name: str, *, cpus, memory, restart_policy) -> dict[str, Any]:
        args = ["update"]
        if cpus is not None:
            args += ["--cpus", str(float(cpus))]
        if memory is not None:
            args += ["--memory", str(int(memory))]
        if restart_policy is not None:
            args += ["--restart", restart_policy]
        if len(args) == 1:
            return {"updated": False, "reason": "nothing to update"}
        rc, out, err = self._docker(*args, name)
        if rc != 0:
            raise ContainerOpError(err.strip() or f"docker update {name} failed")
        return {"updated": True}

    def _raw_network(self, name: str, network: str, *, connect: bool) -> dict[str, Any]:
        verb = "connect" if connect else "disconnect"
        rc, out, err = self._docker("network", verb, network, name)
        if rc != 0:
            raise ContainerOpError(err.strip() or f"docker network {verb} failed")
        return {"ok": True, "network": network, "action": verb}

    def _raw_list_networks(self) -> list[dict[str, Any]]:
        rc, out, err = self._docker("network", "ls", "--format", "{{json .}}")
        if rc != 0:
            raise ContainerOpError(err.strip() or "docker network ls failed")
        nets: list[dict[str, Any]] = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            nets.append({"name": d.get("Name", ""), "driver": d.get("Driver", ""), "scope": d.get("Scope", "")})
        return nets

    def _raw_exec(self, name: str, argv: list[str], *, timeout: float) -> ExecResult:
        rc, out, err = self._docker("exec", name, *argv)
        return ExecResult(exit_code=rc, stdout=out, stderr=err)

    def _raw_top(self, name: str) -> dict[str, Any]:
        rc, out, err = self._docker("top", name)
        if rc != 0:
            raise ContainerOpError(err.strip() or f"docker top {name} failed")
        lines = [ln for ln in out.splitlines() if ln.strip()]
        if not lines:
            return {"titles": [], "processes": []}
        titles = lines[0].split()
        rows = [ln.split(None, len(titles) - 1) for ln in lines[1:]]
        return {"titles": titles, "processes": rows}

    def _raw_changes(self, name: str) -> list[dict[str, Any]]:
        rc, out, err = self._docker("diff", name)
        if rc != 0:
            raise ContainerOpError(err.strip() or f"docker diff {name} failed")
        kinds = {"C": "modified", "A": "added", "D": "deleted"}
        changes: list[dict[str, Any]] = []
        for ln in out.splitlines():
            if len(ln) >= 2 and ln[0] in kinds:
                changes.append({"kind": kinds[ln[0]], "path": ln[2:]})
        return changes

    def _raw_pull(self, name: str) -> dict[str, Any]:
        image = self._raw_inspect(name).get("Config", {}).get("Image") or ""
        if not image:
            raise ContainerOpError(f"could not determine image for {name}")
        rc, out, err = self._docker("pull", image)
        if rc != 0:
            raise ContainerOpError(err.strip() or f"docker pull {image} failed")
        return {"image": image, "output": (out + err).strip()[-2000:]}

    def _raw_system_df(self) -> dict[str, Any]:
        rc, out, err = self._docker("system", "df", "--format", "{{json .}}")
        if rc != 0:
            raise ContainerOpError(err.strip() or "docker system df failed")
        rows = []
        for line in out.splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return _summarize_df_cli(rows)

    def _raw_scan_image(self, name: str) -> dict[str, Any]:
        image = str(self._raw_inspect(name).get("Config", {}).get("Image") or "")
        if not image:
            raise ContainerOpError(f"could not determine image for {name}")
        rc, which, _ = self._run(["sh", "-c",
            "command -v grype >/dev/null 2>&1 && echo grype || "
            "(command -v trivy >/dev/null 2>&1 && echo trivy || echo none)"])
        scanner = which.strip()
        if scanner == "grype":
            rc, out, err = self._run_slow(["grype", image, "-o", "json", "-q"], 300.0)
            if rc != 0 and not out.strip():
                raise ContainerOpError(err.strip() or "grype failed")
            return _summarize_grype(out, image)
        if scanner == "trivy":
            rc, out, err = self._run_slow(["trivy", "image", "-f", "json", "-q", image], 300.0)
            if rc != 0 and not out.strip():
                raise ContainerOpError(err.strip() or "trivy failed")
            return _summarize_trivy(out, image)
        return {"available": False, "image": image,
                "reason": "no scanner found — install grype or trivy on the host"}

    def _raw_stream_logs(self, name: str, *, tail: int):
        # `docker logs --follow … 2>&1` over a paramiko channel: merged stderr so
        # a single stdout stream carries everything (no multiplex framing on the
        # CLI). The channel + client are closed when the generator is torn down.
        if self._target is None:
            raise ContainerOpError("log streaming requires a live SSH target")
        import socket as _socket

        from . import ssh_client
        client = ssh_client._open_client(self._target, self._timeout)  # noqa: SLF001
        chan = client.get_transport().open_session()
        chan.settimeout(1.0)
        chan.exec_command(
            f"docker logs --follow --tail {int(tail)} {shlex.quote(name)} 2>&1")
        try:
            while True:
                try:
                    data = chan.recv(4096)
                except _socket.timeout:
                    if chan.exit_status_ready() and not chan.recv_ready():
                        break
                    yield ""  # heartbeat → lets the caller notice a disconnect
                    continue
                if not data:
                    break
                yield data.decode("utf-8", "replace")
        finally:
            try:
                chan.close()
            except Exception:
                pass
            try:
                client.close()
            except Exception:
                pass


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
        self._sock_path = sock_path
        self._timeout = timeout
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
            nets = ",".join((d.get("NetworkSettings", {}) or {}).get("Networks", {}) or {})
            items.append(ManagedContainer(
                name=name, id=str(d.get("Id", ""))[:12], image=str(d.get("Image", "")),
                state=str(d.get("State", "")), status=str(d.get("Status", "")),
                ports=ports, created=str(d.get("Created", "")),
                labels={str(k): str(v) for k, v in (d.get("Labels") or {}).items()},
                networks=nets,
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

    def _raw_engine_probe(self, port: int) -> dict[str, Any]:
        # The node daemon runs --network host, so 127.0.0.1:<port> reaches the
        # engine on the same host. stdlib HTTP, short timeout, never raises.
        import http.client
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            conn.request("GET", "/health")
            code = conn.getresponse().status
            conn.close()
            return {"reachable": 200 <= code < 300, "port": port, "status_code": code}
        except Exception:
            return {"reachable": False, "port": port, "status_code": None}

    def _raw_list_stats(self) -> dict[str, dict[str, Any]]:
        # The socket is local (no handshake cost), so per-container is cheap; we
        # only sample the running, managed containers.
        result: dict[str, dict[str, Any]] = {}
        for c in self._raw_list():
            if not self._is_managed(c) or str(c.state).lower() != "running":
                continue
            try:
                result[c.name] = self._raw_stats(c.name)
            except ContainerOpError:
                pass
        return result

    def _raw_lifecycle(self, name: str, action: str) -> None:
        status, raw = self._transport("POST", f"/containers/{name}/{action}", None)
        # 204 = done, 304 = already in that state (treat as success).
        if status not in (204, 304):
            raise ContainerOpError(
                f"docker API {action} returned {status}: {raw.decode('utf-8', 'replace')[:200]}")

    def _raw_update_settings(self, name: str, *, cpus, memory, restart_policy) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if cpus is not None:
            body["NanoCPUs"] = int(float(cpus) * 1e9)
        if memory is not None:
            body["Memory"] = int(memory)
        if restart_policy is not None:
            body["RestartPolicy"] = {"Name": restart_policy}
        if not body:
            return {"updated": False, "reason": "nothing to update"}
        status, _ = self._json("POST", f"/containers/{name}/update", body)
        if status != 200:
            raise ContainerOpError(f"docker API update returned {status}")
        return {"updated": True}

    def _raw_network(self, name: str, network: str, *, connect: bool) -> dict[str, Any]:
        verb = "connect" if connect else "disconnect"
        status, raw = self._transport("POST", f"/networks/{network}/{verb}",
                                      json.dumps({"Container": name}).encode())
        if status not in (200, 204):
            raise ContainerOpError(f"docker API network {verb} returned {status}: "
                                   f"{raw.decode('utf-8', 'replace')[:200]}")
        return {"ok": True, "network": network, "action": verb}

    def _raw_list_networks(self) -> list[dict[str, Any]]:
        status, data = self._json("GET", "/networks")
        if status != 200 or not isinstance(data, list):
            raise ContainerOpError(f"docker API /networks returned {status}")
        return [{"name": n.get("Name", ""), "driver": n.get("Driver", ""), "scope": n.get("Scope", "")}
                for n in data]

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

    def _raw_top(self, name: str) -> dict[str, Any]:
        status, data = self._json("GET", f"/containers/{name}/top")
        if status != 200 or not isinstance(data, dict):
            raise ContainerOpError(f"docker API top returned {status}")
        return {"titles": data.get("Titles") or [], "processes": data.get("Processes") or []}

    def _raw_changes(self, name: str) -> list[dict[str, Any]]:
        status, data = self._json("GET", f"/containers/{name}/changes")
        if status != 200:
            raise ContainerOpError(f"docker API changes returned {status}")
        kinds = {0: "modified", 1: "added", 2: "deleted"}
        return [{"kind": kinds.get(int(c.get("Kind", 0)), "modified"), "path": c.get("Path", "")}
                for c in (data or [])]

    def _raw_pull(self, name: str) -> dict[str, Any]:
        image = str(self._raw_inspect(name).get("Config", {}).get("Image") or "")
        if not image:
            raise ContainerOpError(f"could not determine image for {name}")
        repo, _, tag = image.partition(":")
        from urllib.parse import quote
        q = f"/images/create?fromImage={quote(repo, safe='/')}&tag={quote(tag or 'latest')}"
        status, raw = self._transport("POST", q, None)
        if status not in (200, 201):
            raise ContainerOpError(f"docker API image pull returned {status}")
        return {"image": image, "output": raw.decode("utf-8", "replace")[-2000:]}

    def _raw_system_df(self) -> dict[str, Any]:
        status, data = self._json("GET", "/system/df")
        if status != 200 or not isinstance(data, dict):
            raise ContainerOpError(f"docker API /system/df returned {status}")
        return _summarize_df_api(data)

    def _raw_scan_image(self, name: str) -> dict[str, Any]:
        try:
            image = str(self._raw_inspect(name).get("Config", {}).get("Image") or "")
        except ContainerOpError:
            image = ""
        # The docker socket can't run a host-side scanner. Be honest rather than fake.
        return {"available": False, "image": image,
                "reason": "image scanning needs grype/trivy on the host — manage this host "
                          "over SSH, or install a scanner"}

    def _raw_stream_logs(self, name: str, *, tail: int):
        # Stream GET /containers/{id}/logs?follow=1 over a raw unix socket. The
        # body may be HTTP/1.1 chunked AND Docker-multiplex framed → two stateful
        # decoders (both unit-tested) feed each other. Socket closed on teardown.
        import socket as _socket
        sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        sock.settimeout(self._timeout)
        try:
            sock.connect(self._sock_path)
            req = (f"GET /containers/{name}/logs?follow=1&stdout=1&stderr=1&tail={int(tail)} "
                   "HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            sock.sendall(req.encode())
            head = b""
            while b"\r\n\r\n" not in head:
                part = sock.recv(4096)
                if not part:
                    return
                head += part
            header_blob, _, rest = head.partition(b"\r\n\r\n")
            chunked = b"transfer-encoding: chunked" in header_blob.lower()
            dechunk = _ChunkedDecoder(enabled=chunked)
            demux = _FrameDemux()

            def _decode(raw: bytes) -> str:
                return demux.feed(dechunk.feed(raw))

            first = _decode(rest)
            if first:
                yield first
            sock.settimeout(1.0)
            while True:
                try:
                    part = sock.recv(4096)
                except _socket.timeout:
                    yield ""
                    continue
                if not part:
                    break
                text = _decode(part)
                if text:
                    yield text
        finally:
            try:
                sock.close()
            except Exception:
                pass


_SIZE_UNITS = {
    "b": 1, "kb": 1000, "mb": 1000 ** 2, "gb": 1000 ** 3, "tb": 1000 ** 4,
    "kib": 1024, "mib": 1024 ** 2, "gib": 1024 ** 3, "tib": 1024 ** 4,
}


def _parse_size(text: str) -> int:
    """Parse a docker size string (e.g. ``1.19GiB``, ``512MiB``, ``0B``) to bytes."""
    m = re.match(r"\s*([0-9.]+)\s*([A-Za-z]+)\s*$", text or "")
    if not m:
        return 0
    try:
        value = float(m.group(1))
    except ValueError:
        return 0
    return int(value * _SIZE_UNITS.get(m.group(2).lower(), 1))


def _summarize_df_cli(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Normalize ``docker system df --format json`` rows."""
    types: list[dict[str, Any]] = []
    total = 0
    for r in rows:
        size = _parse_size(str(r.get("Size", "")))
        recl = _parse_size(str(r.get("Reclaimable", "")).split("(")[0])
        try:
            count = int(str(r.get("TotalCount", "0")).strip() or 0)
        except ValueError:
            count = 0
        try:
            active = int(str(r.get("Active", "0")).strip() or 0)
        except ValueError:
            active = 0
        types.append({"type": str(r.get("Type", "")), "total_count": count,
                      "active": active, "size": size, "reclaimable": recl})
        total += size
    return {"types": types, "total_size": total}


def _summarize_df_api(d: dict[str, Any]) -> dict[str, Any]:
    """Normalize the Docker Engine API ``/system/df`` payload to the same shape."""
    images = d.get("Images") or []
    containers = d.get("Containers") or []
    volumes = d.get("Volumes") or []
    cache = d.get("BuildCache") or []
    img_size = sum(int(i.get("Size") or 0) for i in images)
    img_shared = sum(int(i.get("SharedSize") or 0) for i in images)
    ctr_size = sum(int(c.get("SizeRw") or 0) for c in containers)
    vol_size = sum(int((v.get("UsageData") or {}).get("Size") or 0) for v in volumes)
    cache_size = sum(int(c.get("Size") or 0) for c in cache)
    types = [
        {"type": "Images", "total_count": len(images),
         "active": sum(1 for i in images if i.get("Containers", 0) not in (0, -1)),
         "size": img_size, "reclaimable": max(0, img_size - img_shared)},
        {"type": "Containers", "total_count": len(containers),
         "active": sum(1 for c in containers if str(c.get("State", "")).lower() == "running"),
         "size": ctr_size, "reclaimable": ctr_size},
        {"type": "Local Volumes", "total_count": len(volumes), "active": 0,
         "size": vol_size, "reclaimable": vol_size},
        {"type": "Build Cache", "total_count": len(cache), "active": 0,
         "size": cache_size, "reclaimable": cache_size},
    ]
    return {"types": types, "total_size": img_size + ctr_size + vol_size + cache_size}


_SEVERITIES = ("critical", "high", "medium", "low", "negligible", "unknown")


def _empty_counts() -> dict[str, int]:
    return {s: 0 for s in _SEVERITIES}


def _summarize_grype(out: str, image: str) -> dict[str, Any]:
    counts = _empty_counts()
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return {"available": True, "scanner": "grype", "image": image, "counts": counts, "total": 0,
                "error": "could not parse grype output"}
    for m in (data.get("matches") or []):
        sev = str((m.get("vulnerability") or {}).get("severity", "unknown")).lower()
        counts[sev if sev in counts else "unknown"] += 1
    return {"available": True, "scanner": "grype", "image": image,
            "counts": counts, "total": sum(counts.values())}


def _summarize_trivy(out: str, image: str) -> dict[str, Any]:
    counts = _empty_counts()
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return {"available": True, "scanner": "trivy", "image": image, "counts": counts, "total": 0,
                "error": "could not parse trivy output"}
    for res in (data.get("Results") or []):
        for v in (res.get("Vulnerabilities") or []):
            sev = str(v.get("Severity", "unknown")).lower()
            counts[sev if sev in counts else "unknown"] += 1
    return {"available": True, "scanner": "trivy", "image": image,
            "counts": counts, "total": sum(counts.values())}


def _summarize_cli_stats(d: dict[str, Any]) -> dict[str, Any]:
    """Reduce ``docker stats --format json`` (CLI) to the compact GUI summary —
    the SAME shape :func:`_summarize_stats` produces for the socket backend."""
    def _pct(v: str) -> float:
        try:
            return round(float(str(v).strip().rstrip("%")), 2)
        except ValueError:
            return 0.0

    def _pair(text: str) -> tuple[int, int]:
        if "/" in (text or ""):
            a, b = text.split("/", 1)
            return _parse_size(a), _parse_size(b)
        return 0, 0

    used, limit = _pair(str(d.get("MemUsage", "")))
    net_rx, net_tx = _pair(str(d.get("NetIO", "")))
    blk_read, blk_write = _pair(str(d.get("BlockIO", "")))
    try:
        pids = int(str(d.get("PIDs", "0")).strip() or 0)
    except ValueError:
        pids = 0
    return {
        "cpu_pct": _pct(d.get("CPUPerc", "0")),
        "mem_usage": used,
        "mem_limit": limit,
        "mem_pct": _pct(d.get("MemPerc", "0")),
        "net_rx": net_rx, "net_tx": net_tx,
        "blk_read": blk_read, "blk_write": blk_write,
        "pids": pids,
    }


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
    net_rx = net_tx = 0
    for iface in (d.get("networks") or {}).values():
        net_rx += int(iface.get("rx_bytes") or 0)
        net_tx += int(iface.get("tx_bytes") or 0)
    blk_read = blk_write = 0
    for entry in ((d.get("blkio_stats") or {}).get("io_service_bytes_recursive") or []):
        op = str(entry.get("op", "")).lower()
        if op == "read":
            blk_read += int(entry.get("value") or 0)
        elif op == "write":
            blk_write += int(entry.get("value") or 0)
    pids = int((d.get("pids_stats") or {}).get("current") or 0)
    return {
        "cpu_pct": round(cpu_pct, 2),
        "mem_usage": mem_usage,
        "mem_limit": mem_limit,
        "mem_pct": round(mem_usage / mem_limit * 100.0, 2) if mem_limit else 0.0,
        "net_rx": net_rx, "net_tx": net_tx,
        "blk_read": blk_read, "blk_write": blk_write,
        "pids": pids,
    }


__all__ = [
    "ManagedContainer", "ExecResult", "GateResult",
    "ContainerControl", "SshContainerControl", "SocketContainerControl",
    "NotManagedError", "ContainerOpError",
    "is_managed_name", "ensure_managed", "managed_prefixes", "exec_enabled",
    "gate_lifecycle", "gate_exec", "demux_docker_stream",
]
