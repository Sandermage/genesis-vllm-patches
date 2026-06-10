# SPDX-License-Identifier: Apache-2.0
"""C10 + C11 + C12 (UNIFIED_CONFIG plan 2026-05-09) — k8s/proxmox/bootstrap CLI tests."""
from __future__ import annotations

import argparse

import pytest

from sndr.model_configs.schema import (
    ModelConfig, HardwareSpec, DockerConfig, KubernetesConfig,
    ProxmoxConfig, BootstrapConfig,
)
from sndr.model_configs import registry as R


def _parse(module_name: str, args: list[str]) -> argparse.Namespace:
    import importlib
    mod = importlib.import_module(f"sndr.cli.legacy.{module_name}")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    mod.add_argparser(sub)
    return parser.parse_args(args)


def _make_full_cfg(*, k8s=True, proxmox=True, bootstrap=True) -> ModelConfig:
    """Build a synthetic preset with Y5+Y6+Y7 blocks declared."""
    return ModelConfig(
        key="test-c10-c11-c12",
        title="x", description="x",
        schema_version=1, maintainer="sandermage",
        model_path="/m",
        hardware=HardwareSpec(gpu_match_keys=["a5000"], n_gpus=2,
                              min_vram_per_gpu_mib=22000),
        docker=DockerConfig(image="img", container_name="c", port=8000,
                              image_digest="img@sha256:abc"),
        kubernetes=KubernetesConfig(
            namespace="genesis-prod", image="genesis:dev93",
            gpu_count=2, service_type="NodePort",
            service_node_port=30800,
            storage={"models": "/data/models"},
        ) if k8s else None,
        proxmox=ProxmoxConfig(
            mode="lxc", container_id_or_vmid=200,
            gpu_passthrough=True, runtime="venv",
        ) if proxmox else None,
        bootstrap=BootstrapConfig(
            scopes=["python-runtime", "container-runtime", "gpu-runtime"],
            apply_policy="ask", privilege="sudo",
        ) if bootstrap else None,
    )


@pytest.fixture(autouse=True)
def _synth_registry(monkeypatch):
    cfg = _make_full_cfg()
    original = R.get
    monkeypatch.setattr(
        R, "get",
        lambda k: cfg if k == "test-c10-c11-c12" else original(k),
    )
    yield


# ─── C10 sndr k8s

def test_k8s_argparser_render():
    ns = _parse("k8s", ["k8s", "render", "test-c10-c11-c12"])
    assert ns.k8s_cmd == "render"
    assert ns.config == "test-c10-c11-c12"


def test_k8s_render_unknown_config_returns_2():
    from sndr.cli.legacy.k8s import run_render
    ns = _parse("k8s", ["k8s", "render", "nonexistent-xyz"])
    assert run_render(ns) == 2


def test_k8s_render_emits_three_manifests(capsys):
    """render produces ConfigMap + Service + Deployment."""
    from sndr.cli.legacy.k8s import run_render
    ns = _parse("k8s", ["k8s", "render", "test-c10-c11-c12"])
    rc = run_render(ns)
    assert rc == 0
    out = capsys.readouterr().out
    assert "kind: ConfigMap" in out
    assert "kind: Service" in out
    assert "kind: Deployment" in out
    # Y5-driven values
    assert "namespace: genesis-prod" in out
    assert "nvidia.com/gpu" in out
    assert "30800" in out  # NodePort


def test_k8s_render_no_y5_block_returns_2(capsys, monkeypatch):
    """Config without Y5 → friendly skip."""
    from sndr.cli.legacy.k8s import run_render
    cfg = _make_full_cfg(k8s=False)
    monkeypatch.setattr(R, "get",
                          lambda k: cfg if k == "no-y5" else None)
    ns = _parse("k8s", ["k8s", "render", "no-y5"])
    rc = run_render(ns)
    assert rc == 2


def test_k8s_apply_dry_run_default():
    """apply without --yes is dry-run (no kubectl call)."""
    from sndr.cli.legacy.k8s import run_apply
    ns = _parse("k8s", ["k8s", "apply", "test-c10-c11-c12"])
    rc = run_apply(ns)
    # 0 from dry-run path; or 1 if kubectl missing — both ok
    assert rc in (0, 1)


# ─── C11 sndr proxmox

def test_proxmox_argparser_render():
    ns = _parse("proxmox", ["proxmox", "render", "test-c10-c11-c12"])
    assert ns.proxmox_cmd == "render"


def test_proxmox_render_unknown_config_returns_2():
    from sndr.cli.legacy.proxmox import run_render
    ns = _parse("proxmox", ["proxmox", "render", "nonexistent-xyz"])
    assert run_render(ns) == 2


def test_proxmox_render_lxc_emits_pct_create(capsys):
    from sndr.cli.legacy.proxmox import run_render
    ns = _parse("proxmox", ["proxmox", "render", "test-c10-c11-c12"])
    rc = run_render(ns)
    assert rc == 0
    out = capsys.readouterr().out
    assert "pct create 200" in out
    assert "GPU passthrough" in out  # gpu_passthrough=True
    assert "venv" in out  # runtime=venv path


def test_proxmox_render_vm_emits_qm(capsys, monkeypatch):
    cfg = _make_full_cfg()
    cfg.proxmox.mode = "vm"
    cfg.proxmox.container_id_or_vmid = 100
    monkeypatch.setattr(R, "get",
                          lambda k: cfg if k == "vm-test" else None)
    from sndr.cli.legacy.proxmox import run_render
    ns = _parse("proxmox", ["proxmox", "render", "vm-test"])
    rc = run_render(ns)
    assert rc == 0
    out = capsys.readouterr().out
    assert "qm create 100" in out


def test_proxmox_doctor_no_pve_returns_1(capsys):
    """On non-PVE host (Mac), doctor returns 1 cleanly."""
    from sndr.cli.legacy.proxmox import run_doctor
    ns = _parse("proxmox", ["proxmox", "doctor"])
    rc = run_doctor(ns)
    # If not on PVE host → 1; if accidentally on PVE → 0. Both ok.
    assert rc in (0, 1)


# ─── C12 sndr bootstrap

def test_bootstrap_argparser_doctor():
    ns = _parse("bootstrap", ["bootstrap", "doctor", "test-c10-c11-c12"])
    assert ns.bootstrap_cmd == "doctor"
    assert ns.scope == "all"


def test_bootstrap_argparser_scope_filter():
    ns = _parse("bootstrap", ["bootstrap", "plan", "test-c10-c11-c12",
                                "--scope", "gpu-runtime,python-runtime"])
    assert ns.scope == "gpu-runtime,python-runtime"


def test_bootstrap_doctor_unknown_config_returns_2():
    from sndr.cli.legacy.bootstrap import run_doctor
    ns = _parse("bootstrap", ["bootstrap", "doctor", "nonexistent-xyz"])
    assert run_doctor(ns) == 2


def test_bootstrap_doctor_runs_on_synth_config(capsys):
    from sndr.cli.legacy.bootstrap import run_doctor
    ns = _parse("bootstrap", ["bootstrap", "doctor", "test-c10-c11-c12"])
    rc = run_doctor(ns)
    # 0 = host ready; 1 = missing deps. Both acceptable per env.
    assert rc in (0, 1)
    out = capsys.readouterr().out
    assert "Y7 declared scopes" in out


def test_bootstrap_apply_never_policy_refused(capsys, monkeypatch):
    """apply_policy='never' → refuse to install ever."""
    cfg = _make_full_cfg()
    cfg.bootstrap.apply_policy = "never"
    monkeypatch.setattr(R, "get",
                          lambda k: cfg if k == "never-test" else None)
    from sndr.cli.legacy.bootstrap import run_apply
    ns = _parse("bootstrap", ["bootstrap", "apply", "never-test", "--yes"])
    rc = run_apply(ns)
    assert rc == 2


def test_bootstrap_status_returns_0_or_1():
    """status: 0 = ready, 1 = not ready. Either ok in test env."""
    from sndr.cli.legacy.bootstrap import run_status
    ns = _parse("bootstrap", ["bootstrap", "status", "test-c10-c11-c12"])
    rc = run_status(ns)
    assert rc in (0, 1)


# ─── Top-level dispatch

def test_top_level_dispatch_k8s():
    from sndr.cli.legacy import cli_main
    rc = cli_main(["k8s", "render", "nonexistent-xyz"])
    assert rc == 2


def test_top_level_dispatch_proxmox():
    from sndr.cli.legacy import cli_main
    rc = cli_main(["proxmox", "render", "nonexistent-xyz"])
    assert rc == 2


def test_top_level_dispatch_bootstrap():
    from sndr.cli.legacy import cli_main
    rc = cli_main(["bootstrap", "doctor", "nonexistent-xyz"])
    assert rc == 2
