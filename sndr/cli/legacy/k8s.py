# SPDX-License-Identifier: Apache-2.0
"""C10 (UNIFIED_CONFIG plan 2026-05-09) — `sndr k8s` Kubernetes CLI.

Reads a preset's Y5 `kubernetes` block + `docker` + `genesis_env` and
renders k8s manifests (Deployment + Service + ConfigMap) ready for
`kubectl apply -f`.

Subcommands:
  sndr k8s render <key>      — print manifests to stdout
  sndr k8s apply <key>       — `kubectl apply -f` (--yes required)
  sndr k8s status <key>      — `kubectl get pods/svc` for the namespace
  sndr k8s logs <key>        — `kubectl logs` of the deployment
  sndr k8s delete <key>      — tear down (--yes required)

Dry-run by default; `--yes` to actually shell out to kubectl.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from typing import Any, Optional

from . import _io


__all__ = ["add_argparser", "run_render", "run_apply", "run_status",
           "run_logs", "run_delete", "run_doctor"]


def add_argparser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "k8s",
        help="Kubernetes deployment wrapper around Y5 KubernetesConfig (UNIFIED_CONFIG C10).",
        description=(
            "Render + apply Genesis presets to a Kubernetes cluster. "
            "Backends: microk8s-single-node / generic-single-node / "
            "generic-multinode. Default --dry-run; --yes to mutate."
        ),
    )
    sub = p.add_subparsers(dest="k8s_cmd", required=True)

    for cmd, helper, fn in (
        ("render", "Print manifests (Deployment + Service + ConfigMap)", run_render),
        ("apply",  "kubectl apply -f the rendered manifests",            run_apply),
        ("status", "kubectl get pods/svc",                               run_status),
        ("logs",   "kubectl logs of the deployment",                     run_logs),
        ("delete", "Tear down the deployment + service + configmap",     run_delete),
    ):
        sp = sub.add_parser(cmd, help=helper)
        sp.add_argument("config", help="model_config preset key")
        sp.add_argument("--yes", action="store_true",
                          help="Actually call kubectl (default: dry-run).")
        sp.add_argument("--lines", type=int, default=50,
                          help="logs: number of recent lines (default 50).")
        # Etap 2.6 (audit 2026-05-12): explicit opt-in for PVC deletion.
        # PVCs hold state (model cache, hf-cache); preserving them across
        # `delete` is the safe default — operator must consciously decide
        # to drop the data with --delete-pvc.
        if cmd == "delete":
            sp.add_argument(
                "--delete-pvc", action="store_true",
                help="Also delete PersistentVolumeClaims created from "
                     "kubernetes.pvc (default: preserve PVC state).",
            )
        sp.set_defaults(func=fn)

    # `sndr k8s doctor` — verify cluster prerequisites for the preset
    p_doctor = sub.add_parser(
        "doctor",
        help="Verify cluster prerequisites for the preset.",
        description=(
            "Probes the current kubectl context for GPU operator / "
            "device-plugin presence, runtimeClass support, namespace "
            "existence, and basic NVIDIA driver visibility on nodes. "
            "Read-only — never mutates."
        ),
    )
    p_doctor.add_argument("config", nargs="?", default=None,
                            help="optional preset key (otherwise cluster-wide check).")
    p_doctor.add_argument("--json", action="store_true",
                            help="Emit JSON.")
    p_doctor.set_defaults(func=run_doctor)


def _resolve(key: str):
    from sndr.model_configs.registry import get
    cfg = get(key)
    if cfg is None:
        _io.warn(f"unknown preset key {key!r}")
        return None
    if cfg.kubernetes is None:
        _io.warn(f"preset {key!r} has no Y5 kubernetes block; "
                  f"add `kubernetes:` to the YAML to use this CLI.")
        return None
    return cfg


# Etap 2.4/2.5 (audit 2026-05-12): dict-based manifests + yaml.safe_dump_all.
# Previously YAML was built via f-string concatenation — that path silently
# admitted invalid DNS-1123 names, non-absolute mount paths and zero-sized
# PVCs (kubectl would reject them mid-apply with cryptic errors). Validators
# below catch these at render time.
_DNS_1123_RE = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
# k8s label key = [optional <prefix>/]<name> where prefix is a DNS
# subdomain (≤253 chars, dots/hyphens allowed) and name is ≤63 chars
# alphanumeric + `-_.`. Examples: `gpu-class`, `nvidia.com/gpu.present`.
_LABEL_NAME_RE = re.compile(
    r"^[A-Za-z0-9]([-A-Za-z0-9_.]{0,61}[A-Za-z0-9])?$"
)
_LABEL_PREFIX_RE = re.compile(
    r"^[a-z0-9]([-a-z0-9.]{0,251}[a-z0-9])?$"
)


def _check_label_key(key: str, field: str) -> None:
    """Validate a Kubernetes label key (optional `<prefix>/<name>`).

    Spec: https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/
    """
    if "/" in key:
        prefix, _, name = key.partition("/")
        if not _LABEL_PREFIX_RE.match(prefix):
            raise ValueError(
                f"{field}={key!r}: prefix {prefix!r} is not a valid "
                "DNS-1123 subdomain"
            )
    else:
        name = key
    if not _LABEL_NAME_RE.match(name):
        raise ValueError(
            f"{field}={key!r}: name segment {name!r} is not a valid "
            "label name (alphanumeric + `-_.`, ≤63 chars, "
            "must start/end alphanumeric)"
        )


def _check_dns_1123(name: str, field: str) -> None:
    """RFC-1123 DNS subdomain segment validation for k8s names."""
    if not name or len(name) > 63 or not _DNS_1123_RE.match(name):
        raise ValueError(
            f"{field}={name!r} is not a valid DNS-1123 name "
            "(lowercase alphanumerics + hyphens, ≤63 chars, "
            "must start/end with alphanumeric)"
        )


def _check_absolute_mount(path: str, field: str) -> None:
    if not path.startswith("/"):
        raise ValueError(
            f"{field}={path!r} must be an absolute path (starts with /)"
        )


def _name(cfg) -> str:
    return f"sndr-{cfg.key}"


def _image_ref(cfg) -> str:
    k = cfg.kubernetes
    if k.image:
        return k.image
    if cfg.docker:
        return cfg.docker.effective_image_ref()
    return ""


def _service_manifest(cfg) -> dict:
    k = cfg.kubernetes
    name = _name(cfg)
    _check_dns_1123(name, "service.metadata.name")
    _check_dns_1123(k.namespace, "service.metadata.namespace")
    port = (cfg.docker.effective_container_port()
            if cfg.docker else 8000)
    port_entry: dict = {
        "port": port,
        "targetPort": port,
        "protocol": "TCP",
    }
    if k.service_type == "NodePort" and k.service_node_port:
        port_entry["nodePort"] = k.service_node_port
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": name, "namespace": k.namespace},
        "spec": {
            "type": k.service_type,
            "selector": {"app": name},
            "ports": [port_entry],
        },
    }


def _configmap_manifest(cfg) -> dict:
    """Genesis env vars as a ConfigMap (simpler than Secret for typical knobs)."""
    name = f"{_name(cfg)}-env"
    k = cfg.kubernetes
    _check_dns_1123(name, "configmap.metadata.name")
    # ConfigMap values must be strings (k8s rejects non-string scalars).
    data = {str(key): str(val) for key, val in
            {**cfg.system_env, **cfg.genesis_env}.items()}
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": name, "namespace": k.namespace},
        "data": data,
    }


def _pvc_manifests(cfg) -> list[dict]:
    """S3.3 audit P3-3: one PersistentVolumeClaim per declared claim_name.

    Empty list when `kubernetes.pvc` is empty (operator uses hostPath
    storage). Etap 2.5: validates claim names + sizes.
    """
    k = cfg.kubernetes
    out: list[dict] = []
    for claim_name in k.pvc:
        _check_dns_1123(claim_name, "pvc.metadata.name")
        size_gib = k.pvc_size_gib.get(claim_name, 100)
        if size_gib <= 0:
            raise ValueError(
                f"pvc_size_gib[{claim_name!r}]={size_gib} must be > 0"
            )
        spec: dict = {
            "accessModes": ["ReadWriteOnce"],
            "resources": {"requests": {"storage": f"{size_gib}Gi"}},
        }
        if k.pvc_storage_class:
            spec["storageClassName"] = k.pvc_storage_class
        out.append({
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {"name": claim_name, "namespace": k.namespace},
            "spec": spec,
        })
    return out


def _build_volumes_and_mounts(
    cfg,
) -> tuple[list[dict], list[dict]]:
    """Merge hostPath / PVC / Secret sources into a single volumes/mounts pair.

    Etap 2.5: each volume name validated as DNS-1123; mount paths must
    be absolute. Misconfigurations raise ValueError at render time.
    """
    k = cfg.kubernetes
    volumes: list[dict] = []
    mounts: list[dict] = []
    seen_paths: set[str] = set()

    def _record(name: str, mount_path: str, *, field_prefix: str,
                 read_only: bool = False) -> None:
        _check_dns_1123(name, f"{field_prefix}.name")
        _check_absolute_mount(mount_path, f"{field_prefix}.mountPath")
        if mount_path in seen_paths:
            raise ValueError(
                f"{field_prefix}.mountPath={mount_path!r} collides with "
                "another volume mount (mount paths must be unique)"
            )
        seen_paths.add(mount_path)
        m: dict = {"name": name, "mountPath": mount_path}
        if read_only:
            m["readOnly"] = True
        mounts.append(m)

    # 1) hostPath storage
    for vol_name, vol_path in k.storage.items():
        _check_absolute_mount(vol_path, "storage.hostPath.path")
        volumes.append({
            "name": vol_name,
            "hostPath": {"path": vol_path},
        })
        _record(vol_name, f"/{vol_name}", field_prefix="storage",
                read_only=True)
    # 2) PVC
    for claim_name, mount_path in k.pvc.items():
        volumes.append({
            "name": claim_name,
            "persistentVolumeClaim": {"claimName": claim_name},
        })
        _record(claim_name, mount_path, field_prefix="pvc")
    # 3) Secret mounts
    for secret_name, mount_path in k.secret_mounts.items():
        volumes.append({
            "name": secret_name,
            "secret": {"secretName": secret_name},
        })
        _record(secret_name, mount_path, field_prefix="secret_mounts",
                read_only=True)
    return volumes, mounts


def _sndr_identity(cfg) -> tuple[dict, dict]:
    """SNDR identity for a rendered Deployment: labels + annotations that tie a
    live pod back to the preset/pin/patches that produced it. The keystone of
    the k8s↔SNDR integration — every downstream feature (pod→preset mapping,
    drift badge, rolling pin upgrade, autoscale bounds) keys off these.

    Labels carry the short, label-safe identity (managed-by, preset, count).
    Annotations carry the longer/free-form values (pin tag, patch-name list)
    that would violate the 63-char label-value rule.
    """
    enabled = [key[len("GENESIS_ENABLE_"):]
               for key, val in sorted(cfg.genesis_env.items())
               if key.startswith("GENESIS_ENABLE_")
               and str(val).strip().lower() not in ("", "0", "false", "no")]
    pin = _image_ref(cfg).rsplit(":", 1)[-1] if ":" in _image_ref(cfg) else ""
    labels = {
        "app.kubernetes.io/managed-by": "sndr",
        "app.kubernetes.io/name": cfg.key,
        "sndr.io/preset": cfg.key,
        "sndr.io/patch-count": str(len(enabled)),
    }
    annotations: dict = {}
    if pin:
        annotations["sndr.io/pin"] = pin
    if enabled:
        annotations["sndr.io/patches"] = ",".join(enabled)
    return labels, annotations


def _deployment_manifest(cfg) -> dict:
    k = cfg.kubernetes
    name = _name(cfg)
    _check_dns_1123(name, "deployment.metadata.name")
    image = _image_ref(cfg)
    port = (cfg.docker.effective_container_port()
            if cfg.docker else 8000)
    volumes, volume_mounts = _build_volumes_and_mounts(cfg)
    for label_key in k.node_selector:
        _check_label_key(label_key, "nodeSelector key")
    pod_spec: dict = {
        "runtimeClassName": k.runtime_class_name,
        "containers": [{
            "name": name,
            "image": image,
            "imagePullPolicy": k.image_pull_policy,
            "ports": [{"containerPort": port}],
            "envFrom": [{"configMapRef": {"name": f"{name}-env"}}],
            "resources": {"limits": {k.gpu_resource_name: k.gpu_count}},
            "readinessProbe": {
                "httpGet": {"path": "/health", "port": port},
                "initialDelaySeconds": k.readiness_initial_delay,
            },
            "livenessProbe": {
                "httpGet": {"path": "/health", "port": port},
                "initialDelaySeconds": k.liveness_initial_delay,
            },
            "volumeMounts": volume_mounts,
        }],
        "volumes": volumes,
    }
    if k.node_selector:
        pod_spec["nodeSelector"] = dict(k.node_selector)
    id_labels, id_annotations = _sndr_identity(cfg)
    # `app` stays the selector key (immutable); SNDR identity rides alongside.
    # Distinct dict copies per metadata block so safe_dump emits plain maps
    # (not YAML anchors/aliases) — the Deploy tab shows this YAML verbatim.
    deploy_meta: dict = {"name": name, "namespace": k.namespace, "labels": {"app": name, **id_labels}}
    tmpl_meta: dict = {"labels": {"app": name, **id_labels}}
    if id_annotations:
        deploy_meta["annotations"] = dict(id_annotations)
        tmpl_meta["annotations"] = dict(id_annotations)
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": deploy_meta,
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": name}},
            "template": {
                "metadata": tmpl_meta,
                "spec": pod_spec,
            },
        },
    }


def _all_yaml(cfg) -> str:
    """Render the full k8s manifest set as a single YAML stream.

    Etap 2.4 (audit 2026-05-12): uses `yaml.safe_dump_all` over dict
    manifests instead of f-string concatenation. Validators raise at
    render time so misconfigurations don't reach `kubectl apply`.
    """
    import yaml
    manifests: list[dict] = [
        _configmap_manifest(cfg),
        _service_manifest(cfg),
        *_pvc_manifests(cfg),
        _deployment_manifest(cfg),
    ]
    header = (
        f"# Generated by sndr k8s render — preset {cfg.key!r}\n"
        f"# Do NOT edit by hand; re-run sndr k8s render to refresh.\n"
    )
    body = yaml.safe_dump_all(
        manifests, default_flow_style=False, sort_keys=False,
    )
    return header + body


def _kubectl(*args, dry_run: bool, stdin: Optional[str] = None) -> int:
    if dry_run:
        cmd_str = " ".join(["kubectl"] + list(args))
        if stdin:
            cmd_str = f"<manifests> | {cmd_str}"
        _io.info(f"[dry-run] would: {cmd_str}")
        return 0
    if shutil.which("kubectl") is None:
        _io.error("kubectl not on PATH")
        return 1
    r = subprocess.run(["kubectl"] + list(args),
                        input=stdin, capture_output=True, text=True,
                        timeout=60)
    if r.stdout:
        print(r.stdout.rstrip())
    if r.stderr:
        print(r.stderr.rstrip())
    return r.returncode


# ─── render

def run_render(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    print(_all_yaml(cfg))
    return 0


# ─── apply

def run_apply(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    body = _all_yaml(cfg)
    return _kubectl("apply", "-f", "-", dry_run=not args.yes, stdin=body)


# ─── status

def run_status(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    namespace = cfg.kubernetes.namespace
    name = f"sndr-{cfg.key}"
    rc1 = _kubectl("get", "deploy", "-n", namespace, name, dry_run=False)
    rc2 = _kubectl("get", "svc", "-n", namespace, name, dry_run=False)
    rc3 = _kubectl("get", "pods", "-n", namespace, "-l", f"app={name}",
                    dry_run=False)
    return max(rc1, rc2, rc3)


# ─── logs

def run_logs(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    namespace = cfg.kubernetes.namespace
    name = f"sndr-{cfg.key}"
    return _kubectl("logs", "-n", namespace, "-l", f"app={name}",
                     "--tail", str(args.lines), dry_run=False)


# ─── delete

def run_delete(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    namespace = cfg.kubernetes.namespace
    name = f"sndr-{cfg.key}"
    dry_run = not args.yes
    rcs = [
        _kubectl("delete", "deploy", "-n", namespace, name,
                  "--ignore-not-found", dry_run=dry_run),
        _kubectl("delete", "svc", "-n", namespace, name,
                  "--ignore-not-found", dry_run=dry_run),
        _kubectl("delete", "configmap", "-n", namespace, f"{name}-env",
                  "--ignore-not-found", dry_run=dry_run),
    ]
    # Etap 2.6: PVCs are preserved by default (they hold model/cache state).
    # Operator must opt in with --delete-pvc to drop the volumes.
    if getattr(args, "delete_pvc", False):
        for claim_name in cfg.kubernetes.pvc:
            rcs.append(_kubectl(
                "delete", "pvc", "-n", namespace, claim_name,
                "--ignore-not-found", dry_run=dry_run,
            ))
    elif cfg.kubernetes.pvc:
        _io.info(
            f"preserving {len(cfg.kubernetes.pvc)} PVC(s); pass "
            "--delete-pvc to drop them as well"
        )
    return max(rcs)


# ─── doctor: cluster prerequisite checks
def run_doctor(args: argparse.Namespace) -> int:
    """Cluster readiness probe.

    Walks a small set of checks and emits a structured report. Never
    mutates. Returns 0 when all checks pass, 1 when any FAIL, 2 when
    only WARN-level findings.
    """
    import json
    if not shutil.which("kubectl"):
        out = {
            "kubectl_present": False,
            "checks": [],
            "summary": "kubectl not on PATH — install it before using `sndr k8s`.",
        }
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            _io.error(out["summary"])
        return 1

    checks: list[dict] = []

    def _check(name: str, cmd: list[str], pass_if_zero: bool = True,
               warn_only: bool = False) -> dict:
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10,
            )
            ok = (r.returncode == 0) if pass_if_zero else (r.returncode != 0)
            severity = "PASS" if ok else ("WARN" if warn_only else "FAIL")
            return {
                "name": name,
                "severity": severity,
                "message": (r.stdout or r.stderr or "").strip()[:200],
            }
        except FileNotFoundError:
            return {"name": name, "severity": "FAIL", "message": "binary missing"}
        except subprocess.TimeoutExpired:
            return {"name": name, "severity": "WARN", "message": "command timed out"}
        except Exception as e:
            return {"name": name, "severity": "WARN", "message": f"{type(e).__name__}: {e}"}

    # 1. kubectl context reachable
    checks.append(_check(
        "cluster_reachable",
        ["kubectl", "cluster-info", "--request-timeout=5s"],
    ))

    # 2. node count + readiness
    r_nodes = _check(
        "nodes_ready",
        ["kubectl", "get", "nodes", "--no-headers"],
    )
    checks.append(r_nodes)

    # 3. GPU device plugin DaemonSet (nvidia.com/gpu)
    checks.append(_check(
        "nvidia_device_plugin",
        ["kubectl", "get", "daemonset", "-A",
         "-l", "k8s-app=nvidia-device-plugin",
         "--no-headers"],
        warn_only=True,
    ))

    # 4. runtimeClass nvidia
    checks.append(_check(
        "runtime_class_nvidia",
        ["kubectl", "get", "runtimeclass", "nvidia"],
        warn_only=True,
    ))

    # 5. NVIDIA GPU operator (alt installation method)
    checks.append(_check(
        "gpu_operator",
        ["kubectl", "get", "ns", "gpu-operator", "--no-headers"],
        warn_only=True,
    ))

    # 6. Per-preset checks: namespace exists, can list secrets in it
    if args.config:
        try:
            cfg = _resolve(args.config)
            ns = getattr(getattr(cfg, "kubernetes", None), "namespace",
                          None) or "default"
            checks.append(_check(
                f"namespace_{ns}",
                ["kubectl", "get", "ns", ns, "--no-headers"],
            ))
        except Exception as e:
            checks.append({
                "name": f"preset_{args.config}",
                "severity": "FAIL",
                "message": f"preset resolve failed: {e}",
            })

    fails = sum(1 for c in checks if c["severity"] == "FAIL")
    warns = sum(1 for c in checks if c["severity"] == "WARN")

    if args.json:
        print(json.dumps({"kubectl_present": True, "checks": checks,
                           "fails": fails, "warns": warns}, indent=2))
    else:
        _io.banner("sndr k8s doctor",
                    f"{len(checks)} checks — {fails} fail · {warns} warn")
        for c in checks:
            mark = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}[c["severity"]]
            line = f"  {mark} [{c['severity']:<4}] {c['name']:<24} {c['message'][:80]}"
            if c["severity"] == "FAIL":
                _io.error(line)
            elif c["severity"] == "WARN":
                _io.warn(line)
            else:
                _io.info(line)

    if fails:
        return 1
    if warns:
        return 2
    return 0
