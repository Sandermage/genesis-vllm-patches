# SPDX-License-Identifier: Apache-2.0
"""Y5 + Y6 + Y7 (UNIFIED_CONFIG plan 2026-05-09) — k8s/proxmox/bootstrap schema tests."""
from __future__ import annotations

import pytest

from sndr.model_configs.schema import (
    KubernetesConfig, ProxmoxConfig, BootstrapConfig,
    ModelConfig, HardwareSpec, DockerConfig,
    SchemaError, dump_yaml, load_yaml,
)


# ─── Y5 KubernetesConfig

def test_k8s_default_validates():
    KubernetesConfig().validate()


def test_k8s_all_known_flavors():
    for flavor in ("microk8s-single-node", "generic-single-node",
                    "generic-multinode"):
        KubernetesConfig(flavor=flavor).validate()


def test_k8s_rejects_unknown_flavor():
    with pytest.raises(SchemaError, match="flavor must be one of"):
        KubernetesConfig(flavor="ranchhand").validate()


def test_k8s_rejects_unknown_pull_policy():
    with pytest.raises(SchemaError, match="image_pull_policy"):
        KubernetesConfig(image_pull_policy="Maybe").validate()


def test_k8s_rejects_unknown_service_type():
    with pytest.raises(SchemaError, match="service_type"):
        KubernetesConfig(service_type="Internal").validate()


def test_k8s_nodeport_range_check():
    KubernetesConfig(service_type="NodePort", service_node_port=30050).validate()
    with pytest.raises(SchemaError, match="service_node_port"):
        KubernetesConfig(service_type="NodePort", service_node_port=80).validate()
    with pytest.raises(SchemaError, match="service_node_port"):
        KubernetesConfig(service_type="NodePort", service_node_port=33000).validate()


# ─── Y6 ProxmoxConfig

def test_proxmox_default_validates():
    ProxmoxConfig().validate()


def test_proxmox_all_known_modes():
    for mode in ("lxc", "vm", "host"):
        ProxmoxConfig(mode=mode).validate()


def test_proxmox_rejects_unknown_mode():
    with pytest.raises(SchemaError, match="mode must be one of"):
        ProxmoxConfig(mode="snap").validate()


def test_proxmox_lxc_docker_combo_requires_acknowledgement():
    """LXC + docker is risky → require explicit notes ack."""
    with pytest.raises(SchemaError, match="docker-inside-lxc"):
        ProxmoxConfig(mode="lxc", runtime="docker").validate()
    # With ack — passes
    ProxmoxConfig(mode="lxc", runtime="docker",
                   notes="we accept docker-inside-lxc risk").validate()


def test_proxmox_vm_docker_no_acknowledgement_needed():
    """VM + docker is safe (docker-inside-VM) — no ack required."""
    ProxmoxConfig(mode="vm", runtime="docker").validate()


def test_proxmox_vmid_range_check():
    ProxmoxConfig(container_id_or_vmid=100).validate()
    ProxmoxConfig(container_id_or_vmid=999_900).validate()
    with pytest.raises(SchemaError, match="container_id_or_vmid"):
        ProxmoxConfig(container_id_or_vmid=99).validate()
    with pytest.raises(SchemaError, match="container_id_or_vmid"):
        ProxmoxConfig(container_id_or_vmid=999_901).validate()


# ─── Y7 BootstrapConfig

def test_bootstrap_default_validates():
    BootstrapConfig().validate()


def test_bootstrap_all_known_scopes():
    BootstrapConfig(scopes=[
        "os-packages", "gpu-runtime", "python-runtime",
        "container-runtime", "model-artifacts", "service",
    ]).validate()


def test_bootstrap_all_scope():
    BootstrapConfig(scopes=["all"]).validate()


def test_bootstrap_rejects_unknown_scope():
    with pytest.raises(SchemaError, match="invalid scope"):
        BootstrapConfig(scopes=["nuke-everything"]).validate()


def test_bootstrap_apply_policy_validation():
    for policy in ("ask", "auto-yes", "never"):
        BootstrapConfig(apply_policy=policy).validate()
    with pytest.raises(SchemaError, match="apply_policy"):
        BootstrapConfig(apply_policy="yolo").validate()


def test_bootstrap_privilege_validation():
    for priv in ("sudo", "root", "user"):
        BootstrapConfig(privilege=priv).validate()
    with pytest.raises(SchemaError, match="privilege"):
        BootstrapConfig(privilege="superadmin").validate()


# ─── YAML round-trip

def test_y5_y6_y7_yaml_roundtrip():
    cfg = ModelConfig(
        key="test-y5-y6-y7",
        title="x", description="x",
        schema_version=1, maintainer="sandermage",
        model_path="/m",
        hardware=HardwareSpec(gpu_match_keys=["a5000"], n_gpus=2,
                              min_vram_per_gpu_mib=22000),
        docker=DockerConfig(image="i", container_name="c", port=8000),
        kubernetes=KubernetesConfig(
            flavor="microk8s-single-node",
            namespace="genesis-prod",
            image="vllm-genesis:dev93",
            gpu_count=2,
            service_type="NodePort",
            service_node_port=30800,
            storage={"models": "/data/models", "hf_cache": "/data/hf"},
        ),
        proxmox=ProxmoxConfig(
            mode="lxc",
            api_endpoint="https://192.168.1.33:8006",
            target_node="ms-a2",
            container_id_or_vmid=200,
            gpu_passthrough=True,
            runtime="venv",
        ),
        bootstrap=BootstrapConfig(
            scopes=["gpu-runtime", "python-runtime", "container-runtime"],
            apply_policy="ask",
            privilege="sudo",
        ),
    )
    yaml_str = dump_yaml(cfg)
    cfg2 = load_yaml(yaml_str)
    # Y5
    assert cfg2.kubernetes is not None
    assert cfg2.kubernetes.flavor == "microk8s-single-node"
    assert cfg2.kubernetes.service_node_port == 30800
    assert cfg2.kubernetes.storage.get("models") == "/data/models"
    # Y6
    assert cfg2.proxmox is not None
    assert cfg2.proxmox.mode == "lxc"
    assert cfg2.proxmox.container_id_or_vmid == 200
    # Y7
    assert cfg2.bootstrap is not None
    assert "gpu-runtime" in cfg2.bootstrap.scopes
    assert cfg2.bootstrap.apply_policy == "ask"
