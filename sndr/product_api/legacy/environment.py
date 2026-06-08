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
import sys as _sys
from dataclasses import dataclass, field
from pathlib import Path as _Path
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
    # Self-locating launch context so the GUI can show a restart command that
    # ACTUALLY works on whatever node this daemon runs on — instead of a static
    # `python3 -m sndr.cli ...` that fails with ModuleNotFoundError when the
    # operator runs it from the wrong directory (the package isn't pip-installed).
    python_executable: str = "python3"
    install_root: Optional[str] = None
    sndr_importable_globally: bool = False
    restart_command: str = "python3 -m sndr.cli gui-api --enable-apply"


def _pkg_version(name: str) -> Optional[str]:
    try:
        return _md.version(name)
    except Exception:
        return None


def _daemon_launch_context() -> tuple[str, Optional[str], bool, str]:
    """Compute (python_executable, install_root, importable_globally, restart_cmd).

    The restart command is the single most useful thing the GUI can hand an
    operator who needs to flip the daemon into apply mode. It must work as-is:

    * If a ``sndr`` console script is on PATH (a real pip install), use it
      directly — it resolves from any working directory.
    * Otherwise the daemon is running from a source checkout / in-image mount,
      so emit ``cd <dir-containing-sndr> && <python> -m sndr.cli ...``. The cd is
      what makes the bare top-level ``sndr`` package importable without a pip
      install (the exact failure the operator hit running it from ``~``).
    """
    py = _sys.executable or "python3"
    # environment.py = <root>/sndr/product_api/legacy/environment.py ->
    # parents[3] is the directory that CONTAINS the importable `sndr/` package.
    root = _Path(__file__).resolve().parents[3]
    has_sndr_here = (root / "sndr").is_dir()
    # `sndr` importable globally == its dist metadata exists (a real install),
    # as opposed to being importable only because cwd happens to be the repo.
    importable_globally = _pkg_version("vllm-sndr-core") is not None
    console = shutil.which("sndr")
    if console:
        cmd = "sndr gui-api --enable-apply"
    elif has_sndr_here:
        cmd = f"cd '{root}' && {py} -m sndr.cli gui-api --enable-apply"
    else:
        cmd = f"{py} -m sndr.cli gui-api --enable-apply"
    return py, (str(root) if has_sndr_here else None), importable_globally, cmd


def collect_environment_report() -> EnvironmentReport:
    """Build the read-only environment / version report."""
    from sndr.brand import PKG_NAME_CORE, PUBLIC_BRAND_COMMUNITY
    from sndr.version import SNDR_CORE_VERSION

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

    py, install_root, importable_globally, restart_cmd = _daemon_launch_context()

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
        python_executable=py,
        install_root=install_root,
        sndr_importable_globally=importable_globally,
        restart_command=restart_cmd,
    )


__all__ = ["DependencyInfo", "EnvironmentReport", "collect_environment_report"]
