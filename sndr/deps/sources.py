# SPDX-License-Identifier: Apache-2.0
"""P3 (UNIFIED_CONFIG plan 2026-05-09) — package source channel logic.

Pure-functional module that decides WHICH source channel to use for
each package. Honors:
  - Y2 PackageSources block from the model_config (operator override)
  - Per-system defaults (apt for Ubuntu, dnf for Fedora, brew for macOS)
  - Safety policy: refuse `curl_pipe_bash` unless explicitly opted in
    via `PackageSource.allow_third_party=True`

No subprocess calls; no installs. Tests can drive every branch
synthetically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SourceDecision:
    """One package's resolved source decision (output of resolve_source)."""
    name: str          # 'docker' | 'nvidia_container_toolkit' | 'vllm' | ...
    kind: str          # 'distro_repo' | 'pip' | 'docker_image' | ...
    channel: str       # 'stable' | 'nightly' | 'main' | ...
    safe: bool         # True if approved for auto-install
    suggested_command: Optional[str] = None
    rationale: str = ""


# Default channel per package kind on common distros. Conservative —
# prefer official distro repos.
_DEFAULT_DOCKER_INSTALL = (
    "follow https://docs.docker.com/engine/install/<distro>/ "
    "(use distro package manager; do NOT pipe curl|bash)"
)
_DEFAULT_NVIDIA_TOOLKIT = (
    "https://docs.nvidia.com/datacenter/cloud-native/"
    "container-toolkit/latest/install-guide.html"
)


def resolve_source(
    package_name: str,
    cfg_sources=None,
    *,
    distro: str = "",
) -> SourceDecision:
    """Decide which source channel to use for `package_name`.

    Order:
      1. Operator's Y2 `cfg.package_sources.get(package_name)` if set
      2. Built-in conservative default (distro_repo where possible)
      3. Fallback: documented manual instruction (no auto-install)

    Returns a `SourceDecision` describing the resolved channel.
    """
    if cfg_sources is not None:
        declared = cfg_sources.get(package_name)
        if declared is not None:
            # Honor operator's choice — but enforce safety
            safe = (declared.kind != "curl_pipe_bash"
                     or declared.allow_third_party)
            return SourceDecision(
                name=declared.name,
                kind=declared.kind,
                channel=declared.channel,
                safe=safe,
                suggested_command=None,
                rationale=(
                    "from cfg.package_sources (operator-declared)"
                    + ("" if safe else " — UNSAFE without --yes opt-in")
                ),
            )

    # Built-in defaults
    if package_name == "docker":
        return SourceDecision(
            name="docker", kind="distro_repo", channel="stable",
            safe=True,
            suggested_command=_distro_docker_cmd(distro),
            rationale="distro repo (safe; conservative default)",
        )
    if package_name == "nvidia_container_toolkit":
        return SourceDecision(
            name="nvidia_container_toolkit", kind="nvidia_repo",
            channel="stable", safe=True,
            suggested_command=_DEFAULT_NVIDIA_TOOLKIT,
            rationale="official NVIDIA repo (safe)",
        )
    if package_name == "vllm":
        return SourceDecision(
            name="vllm", kind="pip", channel="stable",
            safe=True,
            suggested_command="pip install vllm",
            rationale="pip stable index",
        )
    if package_name in ("python", "python3", "python3.12"):
        return SourceDecision(
            name=package_name, kind="distro_repo", channel="stable",
            safe=True,
            suggested_command=_distro_python_cmd(distro),
            rationale="distro python (3.10+)",
        )
    # Unknown package — no auto-action, return manual hint
    return SourceDecision(
        name=package_name, kind="distro_repo", channel="stable",
        safe=False,
        suggested_command=f"install {package_name} via your distro package manager",
        rationale="unknown package — operator must declare a Y2 source explicitly",
    )


def _distro_docker_cmd(distro: str) -> str:
    distro_lower = (distro or "").lower()
    if "ubuntu" in distro_lower or "debian" in distro_lower:
        return "sudo apt-get install -y docker.io  # then add user to docker group"
    if "fedora" in distro_lower or "rocky" in distro_lower or "rhel" in distro_lower:
        return "sudo dnf install -y docker"
    if "arch" in distro_lower:
        return "sudo pacman -S docker"
    if "alpine" in distro_lower:
        return "sudo apk add docker"
    return _DEFAULT_DOCKER_INSTALL


def _distro_python_cmd(distro: str) -> str:
    distro_lower = (distro or "").lower()
    if "ubuntu" in distro_lower or "debian" in distro_lower:
        return "sudo apt-get install -y python3.12 python3.12-venv"
    if "fedora" in distro_lower:
        return "sudo dnf install -y python3.12"
    return "install python 3.12 via your distro package manager"


def list_safe_channels() -> tuple[str, ...]:
    """Channels that auto-install respects without a `--yes` flag."""
    return ("distro_repo", "pip", "nvidia_repo", "docker_image", "github_release")
