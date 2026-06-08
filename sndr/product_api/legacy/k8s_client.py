# SPDX-License-Identifier: Apache-2.0
"""Read-only Kubernetes client for the admin panel (k8s mode, P1).

Honours the operator's kubeconfig/RBAC — acts as the user's credential, never a
god-mode service account (Headlamp's posture). Degrades gracefully: when the
``kubernetes`` client isn't installed, no kubeconfig is present, or the cluster
is unreachable, every call returns a structured ``{available: False, error}``
instead of raising, so the GUI can render "connect a cluster" cleanly.

The data-shaping functions (``shape_node`` etc.) are PURE — they take a node-like
object and return a plain dict — so they are unit-testable with lightweight
mocks, without a live cluster or the kubernetes package installed (the GPU-fleet
operator's #1 view, node GPU capacity, is shaped here and fully tested).
"""
from __future__ import annotations

import os
from typing import Any, Optional

GPU_RESOURCE = "nvidia.com/gpu"

# GFD / gpu-operator node labels worth surfacing (product, memory, count, driver).
_GPU_LABEL_PREFIXES = ("nvidia.com/gpu", "nvidia.com/cuda", "feature.node.kubernetes.io/pci-10de")


def _kubernetes():
    """Import the kubernetes client lazily; None if it isn't installed."""
    try:
        import kubernetes  # noqa: F401
        return kubernetes
    except Exception:
        return None


def availability() -> dict[str, Any]:
    """Why k8s mode is or isn't usable, without touching the network."""
    if _kubernetes() is None:
        return {"available": False,
                "error": "the 'kubernetes' Python client is not installed — pip install 'vllm-sndr-core[k8s]'"}
    cfg = _kubeconfig_path()
    in_cluster = os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/token")
    if not cfg and not in_cluster:
        return {"available": False,
                "error": "no kubeconfig found (set KUBECONFIG or ~/.kube/config), and not running in-cluster"}
    return {"available": True, "error": None, "kubeconfig": cfg, "in_cluster": in_cluster}


def _kubeconfig_path() -> Optional[str]:
    env = os.environ.get("KUBECONFIG", "").strip()
    if env:
        first = env.split(os.pathsep)[0]
        return first if os.path.exists(first) else None
    default = os.path.expanduser("~/.kube/config")
    return default if os.path.exists(default) else None


def _load(context: Optional[str] = None):
    """Load kube config (file or in-cluster) and return the kubernetes module, or
    raise a Containerless RuntimeError-equivalent via the caller's try/except."""
    k = _kubernetes()
    if k is None:
        raise RuntimeError("kubernetes client not installed")
    from kubernetes import config as _cfg
    if os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/token"):
        _cfg.load_incluster_config()
    else:
        _cfg.load_kube_config(context=context)
    return k


# ── pure shaping (unit-tested without a cluster) ─────────────────────────────

def _quantity_to_int(value: Any) -> Optional[int]:
    """k8s integer resource quantities (e.g. nvidia.com/gpu: '2') -> int."""
    if value is None:
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def shape_node(node: Any) -> dict[str, Any]:
    """Shape a V1Node-like object into the dict the GUI nodes table renders.
    Pure: only reads attributes, never calls the API."""
    meta = getattr(node, "metadata", None)
    status = getattr(node, "status", None)
    spec = getattr(node, "spec", None)
    labels = dict(getattr(meta, "labels", None) or {})
    capacity = dict(getattr(status, "capacity", None) or {})
    allocatable = dict(getattr(status, "allocatable", None) or {})

    conditions = getattr(status, "conditions", None) or []
    cond_map = {getattr(c, "type", None): getattr(c, "status", None) for c in conditions}
    ready = cond_map.get("Ready") == "True"
    # Non-Ready pressure conditions that are True are the actionable warnings.
    pressures = [t for t in ("MemoryPressure", "DiskPressure", "PIDPressure")
                 if cond_map.get(t) == "True"]

    roles = sorted(
        k.split("/", 1)[1] or "master"
        for k in labels
        if k.startswith("node-role.kubernetes.io/")
    ) or (["worker"] if labels else [])

    node_info = getattr(status, "node_info", None)
    taints = getattr(spec, "taints", None) or []
    gpu_labels = {k: v for k, v in labels.items()
                  if any(k.startswith(p) for p in _GPU_LABEL_PREFIXES)}

    return {
        "name": getattr(meta, "name", None),
        "ready": ready,
        "schedulable": not bool(getattr(spec, "unschedulable", False)),
        "roles": roles,
        "kubelet_version": getattr(node_info, "kubelet_version", None) if node_info else None,
        "os_image": getattr(node_info, "os_image", None) if node_info else None,
        "cpu_capacity": str(capacity.get("cpu")) if capacity.get("cpu") is not None else None,
        "mem_capacity": str(capacity.get("memory")) if capacity.get("memory") is not None else None,
        "gpu_capacity": _quantity_to_int(capacity.get(GPU_RESOURCE)),
        "gpu_allocatable": _quantity_to_int(allocatable.get(GPU_RESOURCE)),
        "pressures": pressures,
        "taints": [{"key": getattr(t, "key", None), "value": getattr(t, "value", None),
                    "effect": getattr(t, "effect", None)} for t in taints],
        "gpu_labels": gpu_labels,
        "label_count": len(labels),
    }


def shape_pod(pod: Any) -> dict[str, Any]:
    """Shape a V1Pod-like object into the dict the GUI pods table renders. Pure."""
    meta = getattr(pod, "metadata", None)
    spec = getattr(pod, "spec", None)
    status = getattr(pod, "status", None)
    containers = getattr(spec, "containers", None) or []
    cstatuses = getattr(status, "container_statuses", None) or []
    ready = sum(1 for cs in cstatuses if getattr(cs, "ready", False))
    restarts = sum(int(getattr(cs, "restart_count", 0) or 0) for cs in cstatuses)
    gpu = 0
    for ctr in containers:
        req = getattr(getattr(ctr, "resources", None), "requests", None) or {}
        gpu += _quantity_to_int(req.get(GPU_RESOURCE)) or 0
    # A pending pod's reason (e.g. "Unschedulable") is the actionable signal.
    reason = getattr(status, "reason", None)
    # SNDR identity (stamped by `sndr k8s render`): maps a live pod back to the
    # preset/pin/patches that produced it — the round-trip that makes this a
    # SNDR surface rather than a generic pod list.
    labels = dict(getattr(meta, "labels", None) or {})
    annotations = dict(getattr(meta, "annotations", None) or {})
    patch_count_raw = labels.get("sndr.io/patch-count")
    try:
        patch_count = int(patch_count_raw) if patch_count_raw is not None else None
    except (TypeError, ValueError):
        patch_count = None
    patches_raw = annotations.get("sndr.io/patches") or ""
    return {
        "name": getattr(meta, "name", None),
        "namespace": getattr(meta, "namespace", None),
        "node": getattr(spec, "node_name", None),
        "phase": getattr(status, "phase", None),
        "ready": f"{ready}/{len(containers)}",
        "ready_ok": ready == len(containers) and len(containers) > 0,
        "restarts": restarts,
        "gpu_request": gpu,
        "reason": reason,
        "images": [getattr(c, "image", None) for c in containers],
        "sndr_managed": labels.get("app.kubernetes.io/managed-by") == "sndr",
        "sndr_preset": labels.get("sndr.io/preset"),
        "sndr_patch_count": patch_count,
        "sndr_pin": annotations.get("sndr.io/pin"),
        "sndr_patches": [p for p in patches_raw.split(",") if p],
    }


def shape_event(ev: Any) -> dict[str, Any]:
    """Shape a CoreV1Event/EventsV1Event-like object. Pure. The GPU operator most
    cares about Warning events like FailedScheduling 'Insufficient nvidia.com/gpu'."""
    obj = getattr(ev, "involved_object", None) or getattr(ev, "regarding", None)
    return {
        "type": getattr(ev, "type", None),
        "reason": getattr(ev, "reason", None),
        "message": getattr(ev, "message", None) or getattr(ev, "note", None),
        "object": f"{getattr(obj, 'kind', '')}/{getattr(obj, 'name', '')}".strip("/") if obj else None,
        "namespace": getattr(getattr(ev, "metadata", None), "namespace", None),
        "count": getattr(ev, "count", None),
    }


def gpu_requested_by_node(pods: Any) -> dict[str, int]:
    """Sum nvidia.com/gpu requested across all (non-terminal) pods, keyed by the
    node they're scheduled on. Pure over a V1PodList-like ``.items``."""
    out: dict[str, int] = {}
    for pod in getattr(pods, "items", None) or []:
        spec = getattr(pod, "spec", None)
        node_name = getattr(spec, "node_name", None)
        phase = getattr(getattr(pod, "status", None), "phase", None)
        if not node_name or phase in ("Succeeded", "Failed"):
            continue
        total = 0
        for ctr in (getattr(spec, "containers", None) or []):
            res = getattr(ctr, "resources", None)
            req = getattr(res, "requests", None) or {}
            total += _quantity_to_int(req.get(GPU_RESOURCE)) or 0
        if total:
            out[node_name] = out.get(node_name, 0) + total
    return out


# ── live calls (graceful) ────────────────────────────────────────────────────

def cluster_status(context: Optional[str] = None) -> dict[str, Any]:
    avail = availability()
    if not avail["available"]:
        return {"available": False, "error": avail["error"]}
    try:
        k = _load(context)
        ver = k.client.VersionApi().get_code()
        core = k.client.CoreV1Api()
        nodes = core.list_node().items
        ns = core.list_namespace().items
        ready = sum(1 for n in nodes if shape_node(n)["ready"])
        gpu_nodes = sum(1 for n in nodes if (shape_node(n)["gpu_capacity"] or 0) > 0)
        return {
            "available": True, "error": None,
            "version": getattr(ver, "git_version", None),
            "platform": getattr(ver, "platform", None),
            "node_count": len(nodes), "nodes_ready": ready,
            "gpu_node_count": gpu_nodes, "namespace_count": len(ns),
        }
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"[:300]}


def list_nodes(context: Optional[str] = None) -> dict[str, Any]:
    avail = availability()
    if not avail["available"]:
        return {"available": False, "error": avail["error"], "nodes": []}
    try:
        k = _load(context)
        core = k.client.CoreV1Api()
        nodes = [shape_node(n) for n in core.list_node().items]
        # Annotate GPU pressure: requested vs allocatable per node.
        try:
            req = gpu_requested_by_node(core.list_pod_for_all_namespaces())
        except Exception:
            req = {}
        for n in nodes:
            n["gpu_requested"] = req.get(n["name"], 0)
            alloc = n.get("gpu_allocatable") or 0
            n["gpu_free"] = max(0, alloc - n["gpu_requested"]) if alloc else None
        return {"available": True, "error": None, "nodes": nodes}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"[:300], "nodes": []}


def list_pods(context: Optional[str] = None, namespace: Optional[str] = None) -> dict[str, Any]:
    avail = availability()
    if not avail["available"]:
        return {"available": False, "error": avail["error"], "pods": []}
    try:
        k = _load(context)
        core = k.client.CoreV1Api()
        raw = (core.list_namespaced_pod(namespace) if namespace
               else core.list_pod_for_all_namespaces())
        pods = [shape_pod(p) for p in raw.items]
        # GPU pods + non-running first — that's what an operator scans for.
        pods.sort(key=lambda p: (p["phase"] == "Running", -(p["gpu_request"] or 0), p["name"] or ""))
        return {"available": True, "error": None, "pods": pods}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"[:300], "pods": []}


def list_events(context: Optional[str] = None, *, warnings_only: bool = False, limit: int = 100) -> dict[str, Any]:
    avail = availability()
    if not avail["available"]:
        return {"available": False, "error": avail["error"], "events": []}
    try:
        k = _load(context)
        core = k.client.CoreV1Api()
        raw = core.list_event_for_all_namespaces()
        events = [shape_event(e) for e in raw.items]
        if warnings_only:
            events = [e for e in events if e["type"] == "Warning"]
        # Most recent / highest-count first; cap.
        events.sort(key=lambda e: (e["type"] != "Warning", -(e["count"] or 0)))
        return {"available": True, "error": None, "events": events[:limit]}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"[:300], "events": []}


def shape_kubevirt(vm: dict[str, Any]) -> dict[str, Any]:
    """Shape a KubeVirt VirtualMachineInstance (a CustomObjectsApi dict). Pure.

    KubeVirt runs VMs as first-class k8s objects; this surfaces the same vitals
    as a Proxmox guest so the unified Virtualization view reads consistently."""
    meta = vm.get("metadata") or {}
    status = vm.get("status") or {}
    domain = (vm.get("spec") or {}).get("domain") or {}
    ifaces = status.get("interfaces") or []
    ip = next((i.get("ipAddress") for i in ifaces if i.get("ipAddress")), None)
    conds = status.get("conditions") or []
    gpus = (domain.get("devices") or {}).get("gpus") or []
    labels = meta.get("labels") or {}
    return {
        "name": meta.get("name"), "namespace": meta.get("namespace"),
        "kind": "kubevirt",
        "phase": status.get("phase"),
        "running": status.get("phase") == "Running",
        "node": status.get("nodeName"),
        "cpu_cores": (domain.get("cpu") or {}).get("cores"),
        "memory": ((domain.get("resources") or {}).get("requests") or {}).get("memory"),
        "ip": ip,
        "gpu_count": len(gpus),
        "ready": any(c.get("type") == "Ready" and c.get("status") == "True" for c in conds),
        "sndr_preset": labels.get("sndr.io/preset"),
    }


def list_kubevirt_vms(context: Optional[str] = None) -> dict[str, Any]:
    """KubeVirt VMs (VirtualMachineInstances). Degrades cleanly when the KubeVirt
    CRD isn't installed (the common case on a plain cluster) — ``installed: False``
    rather than an error, so the UI says "KubeVirt not installed" not "broken"."""
    avail = availability()
    if not avail["available"]:
        return {"available": False, "error": avail["error"], "vms": []}
    try:
        k = _load(context)
        obj = k.client.CustomObjectsApi().list_cluster_custom_object(
            "kubevirt.io", "v1", "virtualmachineinstances")
        return {"available": True, "installed": True, "error": None,
                "vms": [shape_kubevirt(v) for v in (obj.get("items") or [])]}
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        if "404" in msg or "could not find" in msg.lower() or "NotFound" in msg:
            return {"available": True, "installed": False, "error": None, "vms": []}
        return {"available": False, "error": msg[:300], "vms": []}


__all__ = [
    "GPU_RESOURCE", "availability", "cluster_status", "list_nodes", "list_pods", "list_events",
    "list_kubevirt_vms", "shape_node", "shape_pod", "shape_event", "shape_kubevirt",
    "gpu_requested_by_node",
]
