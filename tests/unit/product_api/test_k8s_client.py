# SPDX-License-Identifier: Apache-2.0
"""Tests for the read-only Kubernetes client — the pure shaping functions are
verified with lightweight mocks, so no live cluster (or kubernetes package) is
needed. Graceful degradation is asserted too."""
from __future__ import annotations

from types import SimpleNamespace as NS

from vllm.sndr_core.product_api import k8s_client as k8s


def _node(name, *, ready=True, gpu_cap=None, gpu_alloc=None, labels=None,
          taints=None, unschedulable=False, kubelet="v1.29.2", pressures=None):
    conds = [NS(type="Ready", status="True" if ready else "False")]
    for p in (pressures or []):
        conds.append(NS(type=p, status="True"))
    cap = {"cpu": "32", "memory": "131072Ki"}
    alloc = {"cpu": "31", "memory": "130000Ki"}
    if gpu_cap is not None:
        cap[k8s.GPU_RESOURCE] = str(gpu_cap)
    if gpu_alloc is not None:
        alloc[k8s.GPU_RESOURCE] = str(gpu_alloc)
    return NS(
        metadata=NS(name=name, labels=labels or {}),
        status=NS(conditions=conds, capacity=cap, allocatable=alloc,
                  node_info=NS(kubelet_version=kubelet, os_image="Ubuntu 22.04")),
        spec=NS(taints=taints or [], unschedulable=unschedulable),
    )


def test_shape_node_basic_ready_and_versions():
    s = k8s.shape_node(_node("gpu-a"))
    assert s["name"] == "gpu-a"
    assert s["ready"] is True and s["schedulable"] is True
    assert s["kubelet_version"] == "v1.29.2"
    assert s["gpu_capacity"] is None and s["gpu_allocatable"] is None


def test_shape_node_extracts_gpu_capacity_and_allocatable():
    s = k8s.shape_node(_node("gpu-a", gpu_cap=8, gpu_alloc=8))
    assert s["gpu_capacity"] == 8
    assert s["gpu_allocatable"] == 8


def test_shape_node_roles_from_labels():
    s = k8s.shape_node(_node("cp", labels={"node-role.kubernetes.io/control-plane": ""}))
    assert "control-plane" in s["roles"]


def test_shape_node_surfaces_pressure_and_unschedulable():
    s = k8s.shape_node(_node("hot", ready=True, pressures=["MemoryPressure"], unschedulable=True))
    assert "MemoryPressure" in s["pressures"]
    assert s["schedulable"] is False


def test_shape_node_taints_and_gpu_labels():
    taint = NS(key="nvidia.com/gpu", value="present", effect="NoSchedule")
    s = k8s.shape_node(_node("gpu-a", taints=[taint],
                             labels={"nvidia.com/gpu.product": "NVIDIA-RTX-A5000", "zone": "rack1"}))
    assert s["taints"] == [{"key": "nvidia.com/gpu", "value": "present", "effect": "NoSchedule"}]
    assert s["gpu_labels"] == {"nvidia.com/gpu.product": "NVIDIA-RTX-A5000"}  # zone excluded


def test_gpu_requested_sums_per_node_and_skips_terminal():
    pods = NS(items=[
        NS(spec=NS(node_name="gpu-a", containers=[NS(resources=NS(requests={k8s.GPU_RESOURCE: "1"}))]),
           status=NS(phase="Running")),
        NS(spec=NS(node_name="gpu-a", containers=[NS(resources=NS(requests={k8s.GPU_RESOURCE: "2"}))]),
           status=NS(phase="Running")),
        # terminal pod must NOT count toward live GPU usage
        NS(spec=NS(node_name="gpu-a", containers=[NS(resources=NS(requests={k8s.GPU_RESOURCE: "4"}))]),
           status=NS(phase="Succeeded")),
        # pod with no GPU request is ignored
        NS(spec=NS(node_name="gpu-b", containers=[NS(resources=NS(requests={}))]),
           status=NS(phase="Running")),
    ])
    req = k8s.gpu_requested_by_node(pods)
    assert req == {"gpu-a": 3}


def test_quantity_to_int_handles_garbage():
    assert k8s._quantity_to_int("2") == 2
    assert k8s._quantity_to_int(None) is None
    assert k8s._quantity_to_int("250m") is None  # millicpu isn't an integer GPU count


def test_availability_degrades_without_kubernetes(monkeypatch):
    monkeypatch.setattr(k8s, "_kubernetes", lambda: None)
    a = k8s.availability()
    assert a["available"] is False and "not installed" in a["error"]


def test_cluster_status_and_list_nodes_degrade_gracefully(monkeypatch):
    # No client installed -> structured unavailable, never raises.
    monkeypatch.setattr(k8s, "_kubernetes", lambda: None)
    cs = k8s.cluster_status()
    ls = k8s.list_nodes()
    assert cs["available"] is False and cs["error"]
    assert ls["available"] is False and ls["nodes"] == []
