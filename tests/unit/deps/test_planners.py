# SPDX-License-Identifier: Apache-2.0
"""Tier 2 P3 — deps.planners unit tests.

Plans are pure functions of (cfg, inventory). We craft synthetic
inventories to drive each branch deterministically — no live host
state needed.
"""
from __future__ import annotations

import pytest

from sndr.deps.checkers import (
    HostInventory, OSInfo, PythonInfo, DockerInfo, NvidiaInfo, VLLMInfo,
)
from sndr.deps.planners import plan_changes, DepsPlan, PlanItem
from sndr.model_configs.schema import (
    ModelConfig, HardwareSpec, DockerConfig, UpstreamPinPolicy,
)


# ─── fixtures ──────────────────────────────────────────────────────────


def _good_inventory() -> HostInventory:
    """Inventory of a fully-provisioned 2× A5000 host."""
    return HostInventory(
        os=OSInfo(system="Linux", release="6.8.0", distro="Ubuntu 24.04",
                  arch="x86_64"),
        python=PythonInfo(binary_path="/usr/bin/python3.12",
                          version="3.12.4", implementation="CPython",
                          venv_active=False, pip_present=True,
                          pip_version="24.0"),
        docker=DockerInfo(installed=True, binary_path="/usr/bin/docker",
                          version="27.2.0", daemon_running=True,
                          server_version="27.2.0",
                          nvidia_runtime_present=True),
        nvidia=NvidiaInfo(installed=True, binary_path="/usr/bin/nvidia-smi",
                          driver_version="550.54.15", cuda_version="12.4",
                          n_gpus=2,
                          gpu_names=["NVIDIA RTX A5000", "NVIDIA RTX A5000"],
                          gpu_total_vram_mib=[24564, 24564]),
        vllm=VLLMInfo(installed=True,
                      version="0.20.2rc1.dev93+g51f22dcfd",
                      location="/usr/local/lib/python3.12/dist-packages/vllm"),
    )


def _docker_cfg(*, n_gpus: int = 2,
                vllm_pin: str | None = None,
                upstream: UpstreamPinPolicy | None = None) -> ModelConfig:
    return ModelConfig(
        key="t",
        title="Test plan config",
        description="Unit-test config for planners.",
        schema_version=1,
        maintainer="sandermage",
        model_path="/models/dummy",
        hardware=HardwareSpec(
            gpu_match_keys=["rtx a5000"], n_gpus=n_gpus,
            min_vram_per_gpu_mib=22000,
        ),
        docker=DockerConfig(image="i", container_name="c", port=8000),
        vllm_pin_required=vllm_pin,
        upstream=upstream,
    )


# ─── happy path

def test_plan_clean_when_host_matches_config():
    plan = plan_changes(
        _docker_cfg(vllm_pin="0.20.2rc1.dev93+g51f22dcfd"),
        _good_inventory(),
    )
    assert plan.is_ready()
    assert plan.blockers() == []
    assert any("ready" in n.lower() for n in plan.notes)


# ─── docker missing

def test_plan_blocks_when_docker_missing():
    inv = _good_inventory()
    inv.docker = DockerInfo(installed=False, notes="not on PATH")
    plan = plan_changes(_docker_cfg(), inv)
    blockers = plan.blockers()
    assert any(i.scope == "docker" and i.action == "install" for i in blockers)


def test_plan_blocks_when_docker_daemon_down():
    inv = _good_inventory()
    inv.docker.daemon_running = False
    plan = plan_changes(_docker_cfg(), inv)
    blockers = plan.blockers()
    assert any(i.scope == "docker" and i.action == "configure" for i in blockers)


def test_plan_blocks_when_nvidia_runtime_missing():
    inv = _good_inventory()
    inv.docker.nvidia_runtime_present = False
    plan = plan_changes(_docker_cfg(), inv)
    blockers = plan.blockers()
    assert any(
        i.scope == "docker" and "nvidia-container-toolkit" in i.target
        for i in blockers
    )


# ─── nvidia driver

def test_plan_blocks_when_no_gpu_visible():
    inv = _good_inventory()
    inv.nvidia = NvidiaInfo(installed=False)
    plan = plan_changes(_docker_cfg(n_gpus=2), inv)
    assert any(i.scope == "nvidia" for i in plan.blockers())


def test_plan_blocks_when_too_few_gpus():
    inv = _good_inventory()
    inv.nvidia.n_gpus = 1
    inv.nvidia.gpu_total_vram_mib = [24564]
    inv.nvidia.gpu_names = ["NVIDIA RTX A5000"]
    plan = plan_changes(_docker_cfg(n_gpus=2), inv)
    assert any(
        i.scope == "nvidia" and "≥ 2 GPU" in i.target
        for i in plan.blockers()
    )


def test_plan_blocks_when_gpu_vram_too_small():
    inv = _good_inventory()
    inv.nvidia.gpu_total_vram_mib = [12000, 12000]  # half of required 22000
    plan = plan_changes(_docker_cfg(n_gpus=2), inv)
    blockers = plan.blockers()
    # Both GPUs fail VRAM check
    vram_blockers = [b for b in blockers if "VRAM" in b.target]
    assert len(vram_blockers) == 2


# ─── vllm pin (Y11 honored)

def test_plan_warns_on_vllm_pin_drift():
    inv = _good_inventory()
    inv.vllm.version = "0.20.1rc1.dev16+g7a1eb8ac2"
    plan = plan_changes(
        _docker_cfg(vllm_pin="0.20.2rc1.dev93+g51f22dcfd"),
        inv,
    )
    items = [i for i in plan.items if i.scope == "vllm" and i.action == "upgrade"]
    assert len(items) == 1
    assert items[0].severity == "warning"


def test_plan_blocks_when_upstream_blocks_pin():
    """Y11 upstream.blocked_pins drives a blocker."""
    inv = _good_inventory()
    inv.vllm.version = "0.20.2rc1.dev99+gbroken"
    cfg = _docker_cfg(
        vllm_pin="0.20.2rc1.dev93+g51f22dcfd",
        upstream=UpstreamPinPolicy(
            blocked_pins=["0.20.2rc1.dev99+gbroken"],
            notes="dev99 hybrid GDN crash",
        ),
    )
    plan = plan_changes(cfg, inv)
    blockers = plan.blockers()
    vllm_blockers = [b for b in blockers if b.scope == "vllm"]
    assert len(vllm_blockers) == 1
    assert "blocked_pins" in vllm_blockers[0].reason


def test_plan_blocks_when_upstream_allowed_list_excludes_pin():
    inv = _good_inventory()
    inv.vllm.version = "0.20.2rc1.dev9+g01d4d1ad3"
    cfg = _docker_cfg(
        upstream=UpstreamPinPolicy(
            allowed_pins=["0.20.2rc1.dev93+g51f22dcfd"],
        ),
    )
    plan = plan_changes(cfg, inv)
    assert any(b.scope == "vllm" and "allowed_pins" in b.reason
               for b in plan.blockers())


# ─── python

def test_plan_blocks_on_python_too_old():
    inv = _good_inventory()
    inv.python.version = "3.9.2"
    plan = plan_changes(_docker_cfg(), inv)
    assert any(b.scope == "python" and b.action == "upgrade"
               for b in plan.blockers())


def test_plan_blocks_when_pip_missing():
    inv = _good_inventory()
    inv.python.pip_present = False
    inv.python.pip_version = None
    plan = plan_changes(_docker_cfg(), inv)
    assert any(b.scope == "python" and b.target == "pip"
               for b in plan.blockers())


# ─── DepsPlan dataclass

def test_plan_to_dict_serializes():
    import json
    plan = plan_changes(_docker_cfg(), _good_inventory())
    s = json.dumps(plan.to_dict())
    parsed = json.loads(s)
    assert "items" in parsed
    assert "is_ready" in parsed
    assert parsed["is_ready"] is True
