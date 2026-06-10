# SPDX-License-Identifier: Apache-2.0
"""Tests for the read-only Kubernetes client — the pure shaping functions are
verified with lightweight mocks, so no live cluster (or kubernetes package) is
needed. Graceful degradation is asserted too."""
from __future__ import annotations

from types import SimpleNamespace as NS

from sndr.product_api.legacy import k8s_client as k8s


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


def _pod(name, *, ns="default", node="gpu-a", phase="Running", ready=1, total=1,
         restarts=0, gpu=0, reason=None, labels=None, annotations=None):
    cstat = [NS(ready=(i < ready), restart_count=restarts) for i in range(total)]
    ctrs = [NS(image="vllm/vllm-openai:nightly",
               resources=NS(requests={k8s.GPU_RESOURCE: str(gpu)} if gpu else {})) for _ in range(total)]
    return NS(metadata=NS(name=name, namespace=ns, labels=labels or {}, annotations=annotations or {}),
              spec=NS(node_name=node, containers=ctrs),
              status=NS(phase=phase, container_statuses=cstat, reason=reason))


def test_shape_pod_ready_restarts_and_gpu():
    s = k8s.shape_pod(_pod("vllm-0", ready=1, total=1, restarts=3, gpu=2))
    assert s["ready"] == "1/1" and s["ready_ok"] is True
    assert s["restarts"] == 3 and s["gpu_request"] == 2
    assert s["node"] == "gpu-a" and s["namespace"] == "default"


def test_shape_pod_not_fully_ready():
    s = k8s.shape_pod(_pod("vllm-0", ready=0, total=2))
    assert s["ready"] == "0/2" and s["ready_ok"] is False


def test_shape_pod_pending_reason_surfaced():
    s = k8s.shape_pod(_pod("vllm-0", phase="Pending", node=None, ready=0, total=1, reason="Unschedulable"))
    assert s["phase"] == "Pending" and s["reason"] == "Unschedulable" and s["node"] is None


def test_shape_pod_surfaces_sndr_identity():
    # A pod rendered by `sndr k8s` carries its preset/pin/patches identity —
    # shape_pod must surface it so the panel maps the pod back to its preset.
    s = k8s.shape_pod(_pod(
        "sndr-a5000-2x-35b-0",
        labels={"app.kubernetes.io/managed-by": "sndr", "sndr.io/preset": "a5000-2x-35b", "sndr.io/patch-count": "14"},
        annotations={"sndr.io/pin": "nightly-abc123", "sndr.io/patches": "P82,PN90,PN95"},
    ))
    assert s["sndr_managed"] is True
    assert s["sndr_preset"] == "a5000-2x-35b"
    assert s["sndr_patch_count"] == 14
    assert s["sndr_pin"] == "nightly-abc123"
    assert s["sndr_patches"] == ["P82", "PN90", "PN95"]


def test_shape_pod_without_sndr_identity_is_neutral():
    # A foreign pod (not rendered by sndr) reports managed=False, no identity.
    s = k8s.shape_pod(_pod("some-other-app-0"))
    assert s["sndr_managed"] is False
    assert s["sndr_preset"] is None and s["sndr_pin"] is None
    assert s["sndr_patches"] == [] and s["sndr_patch_count"] is None


def test_shape_event_warning_with_object():
    ev = NS(type="Warning", reason="FailedScheduling",
            message="0/3 nodes are available: 3 Insufficient nvidia.com/gpu.",
            involved_object=NS(kind="Pod", name="vllm-0"),
            metadata=NS(namespace="default"), count=5)
    s = k8s.shape_event(ev)
    assert s["type"] == "Warning" and s["reason"] == "FailedScheduling"
    assert s["object"] == "Pod/vllm-0" and "nvidia.com/gpu" in s["message"] and s["count"] == 5


def test_list_pods_and_events_degrade_gracefully(monkeypatch):
    monkeypatch.setattr(k8s, "_kubernetes", lambda: None)
    assert k8s.list_pods()["available"] is False and k8s.list_pods()["pods"] == []
    assert k8s.list_events()["available"] is False and k8s.list_events()["events"] == []


def test_shape_kubevirt_vmi():
    vmi = {
        "metadata": {"name": "vllm-vm", "namespace": "default", "labels": {"sndr.io/preset": "qwen3.6-35b-balanced"}},
        "status": {"phase": "Running", "nodeName": "pve1",
                   "interfaces": [{"ipAddress": "10.0.0.9"}],
                   "conditions": [{"type": "Ready", "status": "True"}]},
        "spec": {"domain": {"cpu": {"cores": 8}, "resources": {"requests": {"memory": "48Gi"}},
                            "devices": {"gpus": [{"name": "gpu0"}, {"name": "gpu1"}]}}},
    }
    s = k8s.shape_kubevirt(vmi)
    assert s["name"] == "vllm-vm" and s["kind"] == "kubevirt" and s["running"] is True
    assert s["cpu_cores"] == 8 and s["memory"] == "48Gi" and s["ip"] == "10.0.0.9"
    assert s["gpu_count"] == 2 and s["ready"] is True and s["sndr_preset"] == "qwen3.6-35b-balanced"


def test_list_kubevirt_degrades_without_kubernetes(monkeypatch):
    monkeypatch.setattr(k8s, "_kubernetes", lambda: None)
    r = k8s.list_kubevirt_vms()
    assert r["available"] is False and r["vms"] == []
