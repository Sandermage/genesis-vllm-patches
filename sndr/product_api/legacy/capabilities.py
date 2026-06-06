# SPDX-License-Identifier: Apache-2.0
"""Read-only Product API capability inventory for GUI and web callers.

This module answers "what can this checkout expose to an operator UI?"
without launching containers, probing GPUs, importing SNDR Engine, or
printing. Runtime tool checks use ``shutil.which`` only, so the function is
safe on macOS/Linux/Windows development hosts.
"""
from __future__ import annotations

import importlib.util
import platform
import shutil
from collections.abc import Callable
from typing import Optional

from vllm.sndr_core.brand import PUBLIC_BRAND_COMMUNITY, PKG_NAME_CORE
from vllm.sndr_core.version import SNDR_CORE_VERSION

from .types import (
    PlatformSnapshot,
    ProductCapabilities,
    ProductCapability,
)


WhichFn = Callable[[str], Optional[str]]


def _safe_find_spec(module: str) -> bool:
    """Return True when ``module`` can be resolved without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def is_engine_installed() -> bool:
    """Detect optional SNDR Engine package without importing it."""
    return _safe_find_spec("vllm.sndr_engine")


def _present_tools(tools: tuple[str, ...], which: WhichFn) -> tuple[str, ...]:
    return tuple(tool for tool in tools if which(tool) is not None)


def _runtime_capability(
    *,
    id: str,
    title: str,
    required_tools: tuple[str, ...],
    which: WhichFn,
    detail_available: str,
    detail_render_only: str,
) -> ProductCapability:
    present = _present_tools(required_tools, which)
    if len(present) == len(required_tools):
        return ProductCapability(
            id=id,
            title=title,
            kind="runtime_target",
            status="available",
            detail=detail_available,
            required_tools=required_tools,
            present_tools=present,
        )
    return ProductCapability(
        id=id,
        title=title,
        kind="runtime_target",
        status="render_only",
        detail=detail_render_only,
        required_tools=required_tools,
        present_tools=present,
    )


def collect_platform_snapshot(
    *,
    engine_installed: Optional[bool] = None,
) -> PlatformSnapshot:
    """Return stable platform/package identity for UI status bars."""
    return PlatformSnapshot(
        public_brand=PUBLIC_BRAND_COMMUNITY,
        package_name=PKG_NAME_CORE,
        sndr_core_version=SNDR_CORE_VERSION,
        os_name=platform.system() or "unknown",
        machine=platform.machine() or "unknown",
        python_version=platform.python_version(),
        engine_installed=(
            is_engine_installed()
            if engine_installed is None
            else bool(engine_installed)
        ),
    )


def collect_capabilities(
    *,
    which: WhichFn = shutil.which,
    engine_installed: Optional[bool] = None,
) -> ProductCapabilities:
    """Return the first top-level capability snapshot for GUI clients.

    The status vocabulary is intentionally conservative:

    - ``available`` means a typed Product API surface exists or the local
      runtime toolchain needed for direct action is present.
    - ``partial`` means CLI/core support exists but a dedicated Product API
      contract is still pending.
    - ``render_only`` means SNDR can render/plan config, but the local host
      lacks the tool needed to execute that target.
    - ``deferred`` means planned GUI functionality with no stable API yet.
    """
    platform_snapshot = collect_platform_snapshot(
        engine_installed=engine_installed,
    )

    runtime_targets = (
        ProductCapability(
            id="local_bare_metal",
            title="Local Python/vLLM process",
            kind="runtime_target",
            status="available",
            detail="Local dry-run, compose, patch planning and direct launch primitives are part of SNDR Core.",
        ),
        _runtime_capability(
            id="docker_compose",
            title="Docker Compose",
            required_tools=("docker",),
            which=which,
            detail_available="Docker CLI is present; GUI can graduate from render/plan into local container actions.",
            detail_render_only="Compose files and launch commands can be rendered, but local execution needs docker.",
        ),
        _runtime_capability(
            id="podman_quadlet",
            title="Podman Quadlet",
            required_tools=("podman", "systemctl"),
            which=which,
            detail_available="Podman and systemctl are present for local Quadlet/service workflows.",
            detail_render_only="Quadlet config can be rendered, but local service control needs podman and systemctl.",
        ),
        _runtime_capability(
            id="kubernetes",
            title="Kubernetes",
            required_tools=("kubectl",),
            which=which,
            detail_available="kubectl is present for cluster apply/status workflows.",
            detail_render_only="Kubernetes manifests can be rendered, but cluster actions need kubectl.",
        ),
        _runtime_capability(
            id="proxmox_lxc",
            title="Proxmox LXC",
            required_tools=("pct",),
            which=which,
            detail_available="pct is present for Proxmox LXC lifecycle workflows.",
            detail_render_only="Proxmox plans can be represented, but local LXC actions need pct on a Proxmox host.",
        ),
        _runtime_capability(
            id="remote_ssh",
            title="Remote over SSH",
            required_tools=("ssh",),
            which=which,
            detail_available="ssh is present; desktop remote mode can use an SSH/tunnel transport once implemented.",
            detail_render_only="Remote mode is planned, but this host does not expose ssh in PATH.",
        ),
    )

    features = (
        ProductCapability(
            id="catalog_overview",
            title="Catalog overview",
            kind="feature",
            status="available",
            detail="Read-only Product API summary for models, hardware, profiles and presets.",
            module="vllm.sndr_core.product_api.overview",
        ),
        ProductCapability(
            id="preset_catalog",
            title="Preset catalog and cards",
            kind="feature",
            status="available",
            detail="Typed Product API exposes V2 preset records, explain payloads and recommendations.",
            module="vllm.sndr_core.product_api.presets",
        ),
        ProductCapability(
            id="patch_inventory",
            title="Patch inventory",
            kind="feature",
            status="available",
            detail="Typed Product API exists for patch list/explain/doctor/diff/bundle surfaces.",
            module="vllm.sndr_core.product_api.patches",
        ),
        ProductCapability(
            id="patch_plan",
            title="Patch plan simulation",
            kind="feature",
            status="available",
            detail="Typed Product API can simulate preset patch decisions with environment restoration.",
            module="vllm.sndr_core.product_api.patches.plan",
        ),
        ProductCapability(
            id="service_lifecycle",
            title="Service lifecycle",
            kind="feature",
            status="available",
            detail=(
                "Plan/apply lifecycle Product API is registered. Read-only "
                "status/logs and dry-run apply are always available; real "
                "start/stop/restart execution is gated behind --enable-apply "
                "and requires an explicit confirm."
            ),
            module="vllm.sndr_core.product_api.runtime_exec",
        ),
        ProductCapability(
            id="benchmark_runs",
            title="Benchmark runs and evidence",
            kind="feature",
            status="partial",
            detail=(
                "Benchmark/evidence actions queue as dry-run jobs and report "
                "bundles generate locally; full GPU benchmark execution stays "
                "an operator/rig action."
            ),
        ),
        ProductCapability(
            id="web_daemon",
            title="Local web daemon",
            kind="feature",
            status="available",
            detail="Read-only FastAPI/OpenAPI daemon exposes Product API snapshots for GUI/web clients.",
            module="vllm.sndr_core.product_api.http_app",
        ),
        ProductCapability(
            id="desktop_remote",
            title="Desktop remote mode",
            kind="feature",
            status="deferred",
            detail="Tauri desktop shell and SSH tunnel manager are planned after the local API daemon.",
        ),
        ProductCapability(
            id="engine_fleet",
            title="SNDR Engine fleet features",
            kind="feature",
            status="available" if platform_snapshot.engine_installed else "deferred",
            detail=(
                "SNDR Engine package is installed."
                if platform_snapshot.engine_installed
                else "Commercial/fleet features remain outside the community SNDR Core package."
            ),
            module="vllm.sndr_engine",
        ),
    )

    return ProductCapabilities(
        platform=platform_snapshot,
        runtime_targets=runtime_targets,
        features=features,
    )
