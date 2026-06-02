# SPDX-License-Identifier: Apache-2.0
"""Runtime environment + version report for the admin dashboard.

Reports the SNDR Core project version, the vLLM engine version, the dependency
stack and runtime-tool availability. Versions come from installed package
metadata (``importlib.metadata``) — we never import vLLM/torch and never run a
subprocess, so this stays cheap and torch-free.
"""
from __future__ import annotations

import importlib.metadata as _md
import importlib.util
import platform as _platform
import shutil
from dataclasses import dataclass, field
from typing import Optional

_PYTHON_DEPS = ("vllm", "torch", "transformers", "fastapi", "uvicorn", "pydantic", "numpy")
_RUNTIME_TOOLS = ("docker", "podman", "kubectl", "systemctl", "nvidia-smi", "git", "curl")


@dataclass(frozen=True)
class DependencyInfo:
    name: str
    version: Optional[str]
    present: bool
    kind: str  # "python" | "tool"


@dataclass(frozen=True)
class EnvironmentReport:
    brand: str
    package_name: str
    sndr_core_version: str
    engine_name: str
    engine_version: Optional[str]
    engine_installed: bool
    python_version: str
    os_name: str
    machine: str
    dependencies: tuple[DependencyInfo, ...] = field(default_factory=tuple)
    tools: tuple[DependencyInfo, ...] = field(default_factory=tuple)


def _pkg_version(name: str) -> Optional[str]:
    try:
        return _md.version(name)
    except Exception:
        return None


def collect_environment_report() -> EnvironmentReport:
    """Build the read-only environment / version report."""
    from vllm.sndr_core.brand import PKG_NAME_CORE, PUBLIC_BRAND_COMMUNITY
    from vllm.sndr_core.version import SNDR_CORE_VERSION

    engine_version = _pkg_version("vllm")
    # "engine" in THIS report means the vLLM runtime (see engine_name/engine_version
    # below) — NOT the optional commercial vllm.sndr_engine tier, which is surfaced
    # separately via the capabilities/platform report. vLLM counts as installed if
    # its dist metadata is present OR it's importable (source/editable installs and
    # the in-image mount expose no metadata but are perfectly importable).
    engine_installed = engine_version is not None or importlib.util.find_spec("vllm") is not None

    dependencies = tuple(
        DependencyInfo(name=name, version=_pkg_version(name), present=_pkg_version(name) is not None, kind="python")
        for name in _PYTHON_DEPS
    )
    tools = tuple(
        DependencyInfo(name=tool, version=None, present=shutil.which(tool) is not None, kind="tool")
        for tool in _RUNTIME_TOOLS
    )

    return EnvironmentReport(
        brand=PUBLIC_BRAND_COMMUNITY,
        package_name=PKG_NAME_CORE,
        sndr_core_version=SNDR_CORE_VERSION,
        engine_name="vLLM",
        engine_version=engine_version,
        engine_installed=engine_installed,
        python_version=_platform.python_version(),
        os_name=_platform.system(),
        machine=_platform.machine(),
        dependencies=dependencies,
        tools=tools,
    )


__all__ = ["DependencyInfo", "EnvironmentReport", "collect_environment_report"]
