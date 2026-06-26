# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the remote-install planner (read-only / dry-run)."""
from __future__ import annotations

from sndr.product_api.legacy import deployment, installer


def _fake_dep(preset, target, host_paths=None, image_override=None, with_daemon=False):
    return {
        "artifact": {"filename": "provision-vm.sh", "kind": "bash", "content": "qm create 9200 ..."},
        "target_label": "Proxmox VM (KVM)", "parameters": {"container_name": "vllm"}, "dependencies": {"docker": "required"},
        "image_override": image_override,
        "with_daemon": with_daemon,
        "commands": ["chmod +x provision-vm.sh", "VMID=9200 STORAGE=local-lvm PCI=01:00 ./provision-vm.sh", 'qm terminal "$VMID"'],
    }


def test_plan_lays_out_ordered_steps(monkeypatch):
    monkeypatch.setattr(deployment, "build_deployment", _fake_dep)
    plan = installer.build_install_plan(host={"label": "Prod", "host": "192.0.2.10"}, preset_id="p", target="proxmox_vm")
    kinds = [s["kind"] for s in plan["steps"]]
    assert kinds[0] == "preflight"          # SSH/SFTP check first
    assert "sftp" in kinds                  # push the artifact
    assert "remote-exec" in kinds           # run the commands
    assert kinds[-1] == "verify"            # probe the engine last
    # steps are contiguously ordered.
    assert [s["order"] for s in plan["steps"]] == list(range(1, len(plan["steps"]) + 1))


def test_plan_flags_dangerous_provisioning_steps(monkeypatch):
    monkeypatch.setattr(deployment, "build_deployment", _fake_dep)
    plan = installer.build_install_plan(host={"label": "Prod", "host": "192.0.2.10"}, preset_id="p", target="proxmox_vm")
    danger = [s for s in plan["steps"] if s["danger"]]
    assert danger and any("provision-vm.sh" in s.get("cmd", "") for s in danger)
    assert plan["danger_count"] == len(danger)


def test_plan_is_dry_run_and_cannot_apply_yet(monkeypatch):
    monkeypatch.setattr(deployment, "build_deployment", _fake_dep)
    plan = installer.build_install_plan(host={"label": "h", "host": "x"}, preset_id="p", target="compose")
    assert plan["dry_run"] is True
    assert plan["can_apply"] is False  # the SSH executor is a later, gated phase
    assert plan["artifact"]["filename"] == "provision-vm.sh"


def test_plan_includes_targets_passthrough(monkeypatch):
    monkeypatch.setattr(deployment, "build_deployment", _fake_dep)
    plan = installer.build_install_plan(host={"label": "h", "host": "x"}, preset_id="p", target="compose")
    assert plan["dependencies"] == {"docker": "required"}
    assert plan["host"]["host"] == "x"


def test_compose_up_command_is_flagged():
    assert installer._is_danger("docker compose -f docker-compose.yml up -d")
    assert installer._is_danger("pct create 9100 ...")
    assert installer._is_danger("qm create 9200 --name x")
    assert not installer._is_danger("docker compose logs -f")


def test_apply_is_double_gated_by_flag_and_confirm(monkeypatch):
    monkeypatch.setattr(deployment, "build_deployment", _fake_dep)
    calls = []

    def run_apply(ssh_target, **kw):
        calls.append(kw)
        return {"ok": True, "steps": [{"cmd": "upload x", "rc": 0, "output": ""}]}

    common = dict(host={"label": "h", "host": "x"}, preset_id="p", target="proxmox_vm",
                  ssh_target={"host": "x"}, run_apply=run_apply)
    # Gate 1: apply disabled → nothing runs.
    r = installer.apply_install_plan(**common, apply_enabled=False, confirm=True)
    assert r["applied"] is False and "disabled" in r["error"] and not calls
    # Gate 2: no confirm → nothing runs.
    r = installer.apply_install_plan(**common, apply_enabled=True, confirm=False)
    assert r["applied"] is False and "confirm" in r["error"] and not calls
    # Both gates open → executes, forwarding the artifact + remote-exec commands.
    r = installer.apply_install_plan(**common, apply_enabled=True, confirm=True)
    assert r["applied"] is True and r["ok"] is True
    assert calls and calls[0]["artifact_name"] == "provision-vm.sh"
    assert calls[0]["commands"] == [
        "chmod +x provision-vm.sh",
        "VMID=9200 STORAGE=local-lvm PCI=01:00 ./provision-vm.sh",
        'qm terminal "$VMID"',
    ]
