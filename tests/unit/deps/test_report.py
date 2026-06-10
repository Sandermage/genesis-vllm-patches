# SPDX-License-Identifier: Apache-2.0
"""Tier 2 P3 — deps.report unit tests.

Verifies report writers produce JSON+MD with the right structure and
land in the requested destination (or `~/.sndr/reports/` by default).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sndr.deps.checkers import (
    HostInventory, OSInfo, PythonInfo, DockerInfo, NvidiaInfo, VLLMInfo,
)
from sndr.deps.planners import DepsPlan, PlanItem
from sndr.deps.report import report_inventory, report_plan


def _inv() -> HostInventory:
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
                          gpu_names=["RTX A5000", "RTX A5000"],
                          gpu_total_vram_mib=[24564, 24564]),
        vllm=VLLMInfo(installed=True,
                      version="0.20.2rc1.dev93+g51f22dcfd",
                      location="/usr/local/lib/python3.12/dist-packages/vllm"),
    )


def test_report_inventory_writes_json_and_md(tmp_path):
    json_path, md_path = report_inventory(_inv(), dest=tmp_path)
    assert json_path.exists()
    assert md_path.exists()
    body = json.loads(json_path.read_text())
    assert body["kind"] == "host_inventory"
    assert "inventory" in body
    md = md_path.read_text()
    assert "# SNDR Host Inventory" in md
    assert "Ubuntu 24.04" in md
    assert "RTX A5000" in md
    assert "0.20.2rc1.dev93" in md


def test_report_plan_writes_json_and_md(tmp_path):
    plan = DepsPlan(config_key="t",
                    items=[
                        PlanItem(scope="docker", action="install",
                                 target="Docker", severity="blocker",
                                 reason="not on PATH",
                                 suggested_command="apt install docker-ce"),
                        PlanItem(scope="vllm", action="upgrade",
                                 target="vllm==0.20.2rc1.dev93+g51f22dcfd",
                                 severity="warning",
                                 reason="pin drift"),
                    ])
    json_path, md_path = report_plan(plan, dest=tmp_path)
    body = json.loads(json_path.read_text())
    assert body["kind"] == "deps_plan"
    assert body["plan"]["n_blockers"] == 1
    assert body["plan"]["n_warnings"] == 1
    assert body["plan"]["is_ready"] is False
    md = md_path.read_text()
    assert "Blockers" in md
    assert "Warnings" in md
    assert "apt install docker-ce" in md


def test_report_plan_clean_when_no_items(tmp_path):
    plan = DepsPlan(config_key="t", notes=["Host is ready: no changes required."])
    json_path, md_path = report_plan(plan, dest=tmp_path)
    body = json.loads(json_path.read_text())
    assert body["plan"]["is_ready"] is True
    md = md_path.read_text()
    assert "No changes required" in md


def test_report_uses_sndr_home_when_dest_omitted(monkeypatch, tmp_path):
    """SNDR_HOME → reports go under $SNDR_HOME/reports/."""
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    json_path, md_path = report_inventory(_inv())
    assert str(tmp_path / "reports") in str(json_path)
    assert str(tmp_path / "reports") in str(md_path)
