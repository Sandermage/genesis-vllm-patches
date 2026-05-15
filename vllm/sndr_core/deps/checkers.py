# SPDX-License-Identifier: Apache-2.0
"""Pure host inventory checkers — no install side effects.

Each checker is a small function that probes one slice of the host
(docker daemon, nvidia driver, python interpreter, vllm install, etc.)
and returns a typed dataclass. None of these functions install
anything, modify config, or hit the network beyond local CLI probes.

This module is the foundation of `sndr deps check` (C2) and the
preflight wider than `model_configs/preflight.py` already covers.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Optional


# ─── Subprocess helper ─────────────────────────────────────────────────


def _run(cmd: list[str], *, timeout: float = 5.0) -> tuple[int, str, str]:
    """Run a command and capture (rc, stdout, stderr).

    Returns (-1, "", str(exc)) on FileNotFoundError or subprocess error.
    Never raises. Always returns within `timeout` seconds.
    """
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except FileNotFoundError as e:
        return -1, "", str(e)
    except subprocess.TimeoutExpired:
        return -1, "", f"timeout after {timeout}s"
    except Exception as e:  # pragma: no cover — defensive
        return -1, "", str(e)


# ─── Docker ────────────────────────────────────────────────────────────


@dataclass
class DockerInfo:
    """Result of probing the local Docker daemon."""
    installed: bool
    binary_path: Optional[str] = None
    version: Optional[str] = None       # eg. "27.2.0"
    daemon_running: bool = False
    server_version: Optional[str] = None
    nvidia_runtime_present: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "installed": self.installed,
            "binary_path": self.binary_path,
            "version": self.version,
            "daemon_running": self.daemon_running,
            "server_version": self.server_version,
            "nvidia_runtime_present": self.nvidia_runtime_present,
            "notes": self.notes,
        }


def check_docker() -> DockerInfo:
    """Probe Docker presence + daemon liveness + nvidia runtime."""
    bin_ = shutil.which("docker")
    if bin_ is None:
        return DockerInfo(installed=False, notes="`docker` not on PATH")

    info = DockerInfo(installed=True, binary_path=bin_)
    rc, out, err = _run([bin_, "--version"])
    if rc == 0:
        # eg. "Docker version 27.2.0, build 3ab4256"
        m = re.search(r"version\s+(\S+?)[,\s]", out + " ")
        if m:
            info.version = m.group(1)

    rc, out, err = _run([bin_, "info", "--format", "{{.ServerVersion}}"])
    info.daemon_running = rc == 0 and bool(out)
    if info.daemon_running:
        info.server_version = out

    # nvidia runtime: `docker info` lists `Runtimes: io.containerd ... nvidia`
    if info.daemon_running:
        rc, out, err = _run([bin_, "info"])
        if rc == 0 and "nvidia" in out.lower():
            info.nvidia_runtime_present = True
        else:
            info.notes = (info.notes + "; " if info.notes else "") + (
                "nvidia runtime not detected — install nvidia-container-toolkit"
            )

    return info


# ─── NVIDIA driver / CUDA ──────────────────────────────────────────────


@dataclass
class NvidiaInfo:
    """Result of probing nvidia-smi."""
    installed: bool
    binary_path: Optional[str] = None
    driver_version: Optional[str] = None    # eg. "550.54.15"
    cuda_version: Optional[str] = None      # eg. "12.4"
    n_gpus: int = 0
    gpu_names: list[str] = field(default_factory=list)
    gpu_total_vram_mib: list[int] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "installed": self.installed,
            "binary_path": self.binary_path,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "n_gpus": self.n_gpus,
            "gpu_names": self.gpu_names,
            "gpu_total_vram_mib": self.gpu_total_vram_mib,
            "notes": self.notes,
        }


def check_nvidia() -> NvidiaInfo:
    """Probe nvidia-smi presence + driver/CUDA versions + GPU list."""
    bin_ = shutil.which("nvidia-smi")
    if bin_ is None:
        return NvidiaInfo(installed=False,
                          notes="`nvidia-smi` not on PATH (no driver?)")

    info = NvidiaInfo(installed=True, binary_path=bin_)

    # Driver + CUDA
    rc, out, err = _run([
        bin_, "--query-gpu=driver_version", "--format=csv,noheader",
    ])
    if rc == 0 and out:
        info.driver_version = out.splitlines()[0].strip()

    rc, out, err = _run([bin_])  # plain nvidia-smi prints CUDA Version: X.Y
    if rc == 0 and out:
        m = re.search(r"CUDA Version:\s*(\S+)", out)
        if m:
            info.cuda_version = m.group(1)

    # GPUs
    rc, out, err = _run([
        bin_, "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if rc == 0 and out:
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                info.gpu_names.append(parts[0])
                try:
                    info.gpu_total_vram_mib.append(int(parts[1]))
                except ValueError:
                    info.gpu_total_vram_mib.append(0)
        info.n_gpus = len(info.gpu_names)

    return info


# ─── Python ────────────────────────────────────────────────────────────


@dataclass
class PythonInfo:
    """Local Python interpreter snapshot."""
    binary_path: str
    version: str                          # eg. "3.12.4"
    implementation: str                   # eg. "CPython"
    venv_active: bool                     # VIRTUAL_ENV / sys.prefix mismatch
    pip_present: bool
    pip_version: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "binary_path": self.binary_path,
            "version": self.version,
            "implementation": self.implementation,
            "venv_active": self.venv_active,
            "pip_present": self.pip_present,
            "pip_version": self.pip_version,
        }


def check_python() -> PythonInfo:
    """Snapshot the Python interpreter currently running this process."""
    import sys
    info = PythonInfo(
        binary_path=sys.executable,
        version=platform.python_version(),
        implementation=platform.python_implementation(),
        venv_active=bool(os.environ.get("VIRTUAL_ENV"))
                    or sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        pip_present=False,
    )

    rc, out, err = _run([sys.executable, "-m", "pip", "--version"])
    if rc == 0 and out:
        info.pip_present = True
        m = re.match(r"pip\s+(\S+)", out)
        if m:
            info.pip_version = m.group(1)

    return info


# ─── vLLM ──────────────────────────────────────────────────────────────


@dataclass
class VLLMInfo:
    """Whether vllm is installed in the current Python env."""
    installed: bool
    version: Optional[str] = None         # full pin from vllm.__version__
    location: Optional[str] = None        # site-packages path

    def to_dict(self) -> dict:
        return {
            "installed": self.installed,
            "version": self.version,
            "location": self.location,
        }


def check_vllm() -> VLLMInfo:
    """Detect vllm presence in the running Python (no import side effects)."""
    import sys
    rc, out, err = _run([
        sys.executable, "-c",
        "import vllm, os; print(vllm.__version__);"
        "print(os.path.dirname(vllm.__file__))",
    ], timeout=10.0)
    if rc != 0:
        return VLLMInfo(installed=False)
    lines = out.splitlines()
    return VLLMInfo(
        installed=True,
        version=lines[0].strip() if len(lines) >= 1 else None,
        location=lines[1].strip() if len(lines) >= 2 else None,
    )


# ─── OS ────────────────────────────────────────────────────────────────


@dataclass
class OSInfo:
    system: str       # 'Linux' / 'Darwin' / 'Windows'
    release: str      # kernel version
    distro: str       # /etc/os-release PRETTY_NAME (Linux only)
    arch: str         # 'x86_64' / 'aarch64'

    def to_dict(self) -> dict:
        return {
            "system": self.system,
            "release": self.release,
            "distro": self.distro,
            "arch": self.arch,
        }


def check_os() -> OSInfo:
    """Snapshot OS basics."""
    distro = ""
    try:
        with open("/etc/os-release", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("PRETTY_NAME="):
                    distro = line.split("=", 1)[1].strip().strip('"')
                    break
    except FileNotFoundError:
        pass
    return OSInfo(
        system=platform.system(),
        release=platform.release(),
        distro=distro,
        arch=platform.machine(),
    )


# ─── Aggregator ────────────────────────────────────────────────────────


@dataclass
class HostInventory:
    """Composite snapshot of the host's runtime stack."""
    os: OSInfo
    python: PythonInfo
    docker: DockerInfo
    nvidia: NvidiaInfo
    vllm: VLLMInfo

    def to_dict(self) -> dict:
        return {
            "os": self.os.to_dict(),
            "python": self.python.to_dict(),
            "docker": self.docker.to_dict(),
            "nvidia": self.nvidia.to_dict(),
            "vllm": self.vllm.to_dict(),
        }


def inspect_host() -> HostInventory:
    """Snapshot the host: OS, Python, Docker, NVIDIA driver, vllm.

    All probes are pure (no install side effects). Subprocess timeouts
    are bounded — total wall time of `inspect_host()` is ≤ ~30s even on
    slow VMs.
    """
    return HostInventory(
        os=check_os(),
        python=check_python(),
        docker=check_docker(),
        nvidia=check_nvidia(),
        vllm=check_vllm(),
    )
