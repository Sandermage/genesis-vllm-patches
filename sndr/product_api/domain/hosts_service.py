# SPDX-License-Identifier: Apache-2.0
"""Host inventory service.

Implementation strategy v12.0: the in-process service can ALWAYS report
the local host (the box running ``sndr serve``). Remote hosts come from a
``~/.sndr/fleet.yaml`` registry that the operator maintains. Each remote
entry is just (hostname, ssh_target, optional notes); polling happens
on-demand at the route layer.

This module is intentionally read-only — no SSH calls from the API; the
GUI's Hosts view shows the registry + the local host. Polling remotes is
a separate operator action via ``sndr fleet refresh``.
"""
from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psutil  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover — operator-side dependency
    psutil = None  # type: ignore[assignment]

import yaml

from sndr.product_api.schemas.hosts import (
    FleetReport,
    GpuInfo,
    HostHardware,
    HostSoftware,
    HostSummary,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError:
        return {}


def _detect_gpus() -> list[GpuInfo]:
    """Run nvidia-smi if present; return GPU summaries."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return []
        gpus = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                gpus.append(GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    vram_total_mib=int(parts[2]),
                    vram_used_mib=int(parts[3]),
                    utilization_pct=int(parts[4]),
                    temperature_c=int(parts[5]) if len(parts) > 5 and parts[5] not in ("[N/A]", "") else None,
                    power_draw_w=int(float(parts[6])) if len(parts) > 6 and parts[6] not in ("[N/A]", "") else None,
                ))
            except (ValueError, IndexError):
                continue
        return gpus
    except subprocess.TimeoutExpired:
        return []


def _detect_software() -> HostSoftware:
    """Discover OS + driver versions."""
    uname = platform.uname()
    docker_v = None
    try:
        out = subprocess.run(["docker", "--version"],
                              capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            docker_v = out.stdout.strip().split("version")[1].strip().split(",")[0].strip() if "version" in out.stdout else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    nvidia_driver = None
    cuda_version = None
    if shutil.which("nvidia-smi") is not None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0:
                nvidia_driver = out.stdout.strip().split("\n")[0].strip() or None
        except subprocess.TimeoutExpired:
            pass
        try:
            out = subprocess.run(
                ["nvidia-smi", "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0:
                for line in out.stdout.splitlines():
                    if "CUDA Version" in line:
                        cuda_version = line.split(":", 1)[1].strip()
                        break
        except subprocess.TimeoutExpired:
            pass

    # Resolve OS id + version from /etc/os-release on Linux
    os_id = "unknown"
    os_version = ""
    osr = Path("/etc/os-release")
    if osr.is_file():
        for line in osr.read_text().splitlines():
            if line.startswith("ID="):
                os_id = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("VERSION_ID="):
                os_version = line.split("=", 1)[1].strip().strip('"')
    else:
        os_id = uname.system.lower()
        os_version = uname.release

    return HostSoftware(
        os_id=os_id,
        os_version=os_version,
        kernel=uname.release,
        docker_version=docker_v,
        nvidia_driver=nvidia_driver,
        cuda_version=cuda_version,
    )


def _detect_hardware() -> HostHardware:
    """Discover hardware via psutil + nvidia-smi.

    psutil is an optional dep. When unavailable (e.g. minimal install),
    we fall back to OS-API best effort: ``os.cpu_count()`` and reading
    ``/proc/meminfo``.
    """
    if psutil is not None:
        mem = psutil.virtual_memory()
        ram_total = int(mem.total / (1024 ** 3))
        ram_available = int(mem.available / (1024 ** 3))
        cores = psutil.cpu_count(logical=False) or 1
    else:
        meminfo = Path("/proc/meminfo")
        ram_total = 0
        ram_available = 0
        if meminfo.is_file():
            for line in meminfo.read_text().splitlines():
                if line.startswith("MemTotal:"):
                    ram_total = int(line.split()[1]) // (1024 ** 2)
                elif line.startswith("MemAvailable:"):
                    ram_available = int(line.split()[1]) // (1024 ** 2)
        cores = os.cpu_count() or 1

    cpu_model = platform.processor() or "unknown"
    # Linux /proc/cpuinfo if processor() is empty
    if cpu_model in ("", "unknown"):
        ci = Path("/proc/cpuinfo")
        if ci.is_file():
            for line in ci.read_text().splitlines():
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break

    return HostHardware(
        cpu_model=cpu_model or "unknown",
        cpu_cores=cores,
        ram_total_gib=ram_total,
        ram_available_gib=ram_available,
        gpus=_detect_gpus(),
    )


def get_local_host() -> HostSummary:
    """Build a HostSummary for the box running this process."""
    from sndr.version import __version__

    hostname = socket.gethostname()
    try:
        hardware = _detect_hardware()
    except Exception:
        hardware = None
    try:
        software = _detect_software()
    except Exception:
        software = None

    return HostSummary(
        hostname=hostname,
        status="online",
        last_seen_at=datetime.now(timezone.utc),
        sndr_version=__version__,
        sndr_install_root=str(Path(__file__).resolve().parents[3]),
        active_engine=os.environ.get("SNDR_ENGINE"),
        active_engine_pin=os.environ.get("SNDR_ENGINE_PIN"),
        hardware=hardware,
        software=software,
    )


def list_hosts() -> list[HostSummary]:
    """List local host + fleet entries from ``~/.sndr/fleet.yaml``."""
    out = [get_local_host()]

    fleet_yaml = Path(os.environ.get("SNDR_HOME", "~/.sndr")).expanduser() / "fleet.yaml"
    data = _read_yaml(fleet_yaml)
    for entry in data.get("hosts", []) or []:
        try:
            out.append(HostSummary(
                hostname=entry["hostname"],
                status=entry.get("status", "unknown"),
                last_seen_at=entry.get("last_seen_at") or datetime.now(timezone.utc),
                sndr_version=entry.get("sndr_version"),
                sndr_install_root=entry.get("sndr_install_root"),
                active_engine=entry.get("active_engine"),
                active_engine_pin=entry.get("active_engine_pin"),
                notes=entry.get("notes"),
            ))
        except Exception:
            continue
    return out


def fleet_report() -> FleetReport:
    """Aggregate over all known hosts."""
    hosts = list_hosts()
    by_status = {"online": 0, "degraded": 0, "offline": 0, "unknown": 0}
    total_gpus = 0
    total_vram_gib = 0
    for h in hosts:
        by_status[h.status] = by_status.get(h.status, 0) + 1
        if h.hardware:
            total_gpus += len(h.hardware.gpus)
            total_vram_gib += sum(g.vram_total_mib // 1024 for g in h.hardware.gpus)

    return FleetReport(
        total_hosts=len(hosts),
        online=by_status["online"],
        degraded=by_status["degraded"],
        offline=by_status["offline"],
        unknown=by_status["unknown"],
        total_gpus=total_gpus,
        total_vram_gib=total_vram_gib,
    )


__all__ = ["get_local_host", "list_hosts", "fleet_report"]
