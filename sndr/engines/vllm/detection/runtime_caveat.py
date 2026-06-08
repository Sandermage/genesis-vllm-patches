# SPDX-License-Identifier: Apache-2.0
"""Container-runtime caveat detection (S-05 install.sh refactor, 2026-05-08).

Some host environments have known incompatibilities with the official
vllm container image. The shell installer used to detect Proxmox VE
hosts via `uname -r` and `/etc/pve` to auto-suggest bare-metal mode.
This module is the canonical Python version.

Background:
  noonghunna/club-3090#49 (lexhoefsloot 2026-05-04) — the official
  `vllm/vllm-openai:nightly-…` image hits a uvloop "event loop already
  running" crash on Proxmox VE 8.x kernel 6.17.x hosts during
  `vllm serve`, BEFORE Genesis runs. Bare docker repros it.

  Workaround: native `pip install vllm==0.20.x` venv on the same host
  launches cleanly past the failure point.

Usage:
    from sndr.engines.vllm.detection.runtime_caveat import probe_caveats
    cav = probe_caveats()
    if cav.proxmox_detected:
        # auto-flip to bare-metal in the wizard
        ...
"""
from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeCaveats:
    """Outcome of `probe_caveats()`."""
    proxmox_detected: bool
    kernel_release: str
    reason: str


def _kernel_looks_pve(kernel_release: str) -> bool:
    """Proxmox builds their own kernel branded with `pve` or `proxmox`."""
    if not kernel_release:
        return False
    k = kernel_release.lower()
    return "pve" in k or "proxmox" in k


def _has_pve_etc() -> bool:
    """Proxmox host has `/etc/pve/` (cluster config dir)."""
    return Path("/etc/pve").is_dir()


def probe_caveats() -> RuntimeCaveats:
    """Inspect the running host for known container-runtime caveats.

    Returns a `RuntimeCaveats` describing what was detected and a short
    operator-facing reason string.
    """
    try:
        kernel = platform.release()
    except Exception:
        kernel = ""

    pve_kernel = _kernel_looks_pve(kernel)
    pve_etc = _has_pve_etc()
    proxmox = pve_kernel or pve_etc

    if not proxmox:
        return RuntimeCaveats(
            proxmox_detected=False,
            kernel_release=kernel,
            reason="no Proxmox VE markers — docker path should be safe",
        )

    markers = []
    if pve_kernel:
        markers.append(f"kernel '{kernel}' matches pve|proxmox")
    if pve_etc:
        markers.append("/etc/pve/ present")

    return RuntimeCaveats(
        proxmox_detected=True,
        kernel_release=kernel,
        reason=(
            f"Proxmox VE host detected ({'; '.join(markers)}). "
            "club-3090#49: official vllm/vllm-openai:nightly image hits "
            "uvloop crash on PVE 8.x + kernel 6.17.x during vllm serve. "
            "Suggested workaround: native venv `pip install vllm==0.20.x`."
        ),
    )


__all__ = ["RuntimeCaveats", "probe_caveats"]
