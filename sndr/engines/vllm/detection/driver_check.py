# SPDX-License-Identifier: Apache-2.0
"""NVIDIA driver version sanity check (S-05 install.sh refactor, 2026-05-08).

The Genesis pin (vllm 0.20.2rc1.dev9+g01d4d1ad3) targets CUDA 13.0,
which requires NVIDIA driver ≥ 580.x. Older drivers silently fall back
to CUDA 12.x compatibility — works but ~3× slower on FP8 paths, breaks
Blackwell SM 12.0 codegen, and disables some FlashInfer kernels.

This module is the canonical Python version of the install.sh check.
It does NOT block — only reports — because some hosts have legitimate
reasons to run older drivers (locked-down systems, vendor support
windows, etc.).

Usage:
    from vllm.sndr_core.detection.driver_check import probe_driver
    info = probe_driver()
    if info.below_recommended:
        warn(info.recommendation)
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

# Genesis-pinned vllm dev build targets CUDA 13.0 → NVIDIA driver 580+
_RECOMMENDED_DRIVER_MAJOR = 580


@dataclass(frozen=True)
class DriverInfo:
    """Outcome of `probe_driver()`."""
    nvidia_smi_present: bool
    raw_driver_version: Optional[str]   # full string like "535.183.06"
    driver_major: Optional[int]
    cuda_runtime_reported: Optional[str]  # from `nvidia-smi` header
    below_recommended: bool
    recommendation: str


def _read_driver_version() -> Optional[str]:
    """Run `nvidia-smi --query-gpu=driver_version` and return the first
    GPU's version string. None if nvidia-smi missing/failed."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    first = out.stdout.strip().splitlines()[0:1]
    if not first:
        return None
    return first[0].strip() or None


def _read_cuda_header() -> Optional[str]:
    """Read the `CUDA Version: X.Y` header from plain `nvidia-smi`."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    for line in out.stdout.splitlines():
        idx = line.find("CUDA Version:")
        if idx < 0:
            continue
        rest = line[idx + len("CUDA Version:"):].strip()
        # Take first whitespace-delimited token, drop trailing chars
        token = rest.split()[0] if rest else ""
        return token or None
    return None


def _major_int(raw: str) -> Optional[int]:
    """Extract leading integer from `NNN.MMM.PPP` style version. None
    if unparseable."""
    if not raw:
        return None
    head = raw.split(".", 1)[0]
    digits = "".join(ch for ch in head if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def probe_driver() -> DriverInfo:
    """Inspect the host's NVIDIA driver. Never raises."""
    if not shutil.which("nvidia-smi"):
        return DriverInfo(
            nvidia_smi_present=False,
            raw_driver_version=None,
            driver_major=None,
            cuda_runtime_reported=None,
            below_recommended=False,
            recommendation=(
                "nvidia-smi not found — driver check skipped. "
                "Install NVIDIA driver before running Genesis on a real GPU."
            ),
        )

    raw = _read_driver_version()
    cuda = _read_cuda_header()
    major = _major_int(raw) if raw else None
    below = major is not None and major < _RECOMMENDED_DRIVER_MAJOR

    if raw is None:
        recommendation = (
            "could not read NVIDIA driver version from nvidia-smi — "
            "driver check inconclusive"
        )
    elif major is None:
        recommendation = (
            f"driver version {raw!r} unparseable — driver check "
            "inconclusive"
        )
    elif below:
        recommendation = (
            f"NVIDIA driver {raw} detected — Genesis pin recommends "
            f"≥{_RECOMMENDED_DRIVER_MAJOR}.x for CUDA 13.0 compat. "
            "Symptoms on older drivers: FP8 paths ~3× slower, "
            "Blackwell SM 12.0 codegen unavailable, some FlashInfer "
            "kernels disabled. Install will continue — most paths "
            "still work but expect lower throughput than published "
            "Genesis benchmarks."
        )
    else:
        recommendation = (
            f"driver {raw} (≥{_RECOMMENDED_DRIVER_MAJOR}.x — CUDA 13.0 "
            "compatible)"
        )

    return DriverInfo(
        nvidia_smi_present=True,
        raw_driver_version=raw,
        driver_major=major,
        cuda_runtime_reported=cuda,
        below_recommended=below,
        recommendation=recommendation,
    )


__all__ = ["DriverInfo", "probe_driver"]
