# SPDX-License-Identifier: Apache-2.0
"""Deployment-runtime sub-component dataclasses.

Hosts the Kubernetes / Proxmox / Bootstrap / GPU-tuning / Observability /
Service blocks. All classes relocated from ``model_configs/schema.py``
in M.5.1; bodies unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ._base import SchemaError


@dataclass
class KubernetesConfig:
    """Y5 (UNIFIED_CONFIG plan 2026-05-09): Kubernetes deployment contract.

    Drives `sndr k8s render/apply/status` (Tier 4 CLI). Declares the
    bare-minimum shape: namespace, image, GPU resource, volumes,
    service exposure, probes.

    Fields:
      - flavor: 'microk8s-single-node' | 'generic-single-node' | 'generic-multinode'
      - namespace: k8s namespace
      - image / image_pull_policy: container image
      - gpu_resource_name: 'nvidia.com/gpu' (default) or hostpath alt
      - gpu_count: requested GPU count
      - runtime_class_name: 'nvidia' typically (RuntimeClass)
      - storage: PVC declarations (model weights, hf cache, ...)
      - service_type: 'NodePort' | 'ClusterIP' | 'LoadBalancer'
      - service_node_port: when type=NodePort
      - liveness_initial_delay / readiness_initial_delay: probe timings
      - notes: free-form
    """
    flavor: str = "microk8s-single-node"
    namespace: str = "genesis"
    image: str = ""
    image_pull_policy: str = "IfNotPresent"
    gpu_resource_name: str = "nvidia.com/gpu"
    gpu_count: int = 1
    runtime_class_name: str = "nvidia"
    storage: dict[str, str] = field(default_factory=dict)
    service_type: str = "ClusterIP"
    service_node_port: Optional[int] = None
    liveness_initial_delay: int = 600
    readiness_initial_delay: int = 600
    notes: str = ""

    # S3.3 (audit P3-3, 2026-05-12): additional deployment knobs.
    # nodeSelector — labels on the pod so the scheduler picks the right
    # node (e.g. `gpu-class=a5000`). Empty dict → omit the nodeSelector
    # block from the render.
    node_selector: dict[str, str] = field(default_factory=dict)

    # PVC bindings — `pvc_name -> mount_path`. When provided, the
    # renderer emits PersistentVolumeClaim resources (operator must
    # have matching PVs or a storageClass) AND mounts via
    # `volumes[].persistentVolumeClaim.claimName`. To use hostPath
    # (legacy `storage`) leave pvc empty.
    pvc: dict[str, str] = field(default_factory=dict)

    # PVC sizing (GiB per claim). Default 100 GiB on each claim when
    # unset. Shape: `{claim_name: size_gib}`.
    pvc_size_gib: dict[str, int] = field(default_factory=dict)

    # PVC storageClassName. Empty → operator must have a default class.
    pvc_storage_class: str = ""

    # Secret references — `secret_name -> mount_path`. Useful for
    # bearer tokens / wandb keys. The secret content is NOT generated
    # by the renderer — operator creates it manually via `kubectl create
    # secret generic …`.
    secret_mounts: dict[str, str] = field(default_factory=dict)

    _VALID_FLAVORS = (
        "microk8s-single-node", "generic-single-node", "generic-multinode",
    )
    _VALID_PULL = ("Always", "IfNotPresent", "Never")
    _VALID_SVC = ("NodePort", "ClusterIP", "LoadBalancer")

    def validate(self) -> None:
        if self.flavor not in self._VALID_FLAVORS:
            raise SchemaError(
                f"KubernetesConfig.flavor must be one of "
                f"{self._VALID_FLAVORS} (got {self.flavor!r})"
            )
        if self.image_pull_policy not in self._VALID_PULL:
            raise SchemaError(
                f"KubernetesConfig.image_pull_policy must be one of "
                f"{self._VALID_PULL} (got {self.image_pull_policy!r})"
            )
        if self.service_type not in self._VALID_SVC:
            raise SchemaError(
                f"KubernetesConfig.service_type must be one of "
                f"{self._VALID_SVC} (got {self.service_type!r})"
            )
        if self.gpu_count < 0:
            raise SchemaError(
                f"KubernetesConfig.gpu_count must be >= 0 "
                f"(got {self.gpu_count})"
            )
        if (self.service_type == "NodePort"
                and self.service_node_port is not None
                and not (30000 <= self.service_node_port <= 32767)):
            raise SchemaError(
                f"KubernetesConfig.service_node_port must be in "
                f"30000..32767 for NodePort (got {self.service_node_port})"
            )
        if self.liveness_initial_delay < 0 or self.readiness_initial_delay < 0:
            raise SchemaError(
                "KubernetesConfig.*_initial_delay must be >= 0"
            )


@dataclass
class ProxmoxConfig:
    """Y6 (UNIFIED_CONFIG plan 2026-05-09): Proxmox deployment contract.

    Drives `sndr proxmox doctor/render/apply/status` (Tier 4 CLI).

    Three modes:
      - 'lxc': run inside an unprivileged LXC container (preferred for
               GPU stack — bare-metal venv inside LXC; Docker-inside-
               LXC is experimental/risky)
      - 'vm': run inside a Proxmox VM (Docker-on-VM safe)
      - 'host': bare-metal on the PVE host itself (least-isolated,
                expert-only — voids Proxmox upgrade safety)

    Fields:
      - mode: 'lxc' | 'vm' | 'host'
      - api_endpoint: PVE API URL
      - target_node: PVE node hostname
      - container_id_or_vmid: 100-999900
      - gpu_passthrough: True if GPU is passed through to LXC/VM
      - runtime: container runtime inside the LXC/VM ('docker' | 'podman'
                  | 'venv' | 'system_python')
      - notes: free-form
    """
    mode: str = "lxc"
    api_endpoint: Optional[str] = None
    target_node: Optional[str] = None
    container_id_or_vmid: Optional[int] = None
    gpu_passthrough: bool = True
    runtime: str = "venv"
    notes: str = ""

    _VALID_MODES = ("lxc", "vm", "host")
    _VALID_RUNTIMES = ("docker", "podman", "venv", "system_python")

    def validate(self) -> None:
        if self.mode not in self._VALID_MODES:
            raise SchemaError(
                f"ProxmoxConfig.mode must be one of {self._VALID_MODES} "
                f"(got {self.mode!r})"
            )
        if self.runtime not in self._VALID_RUNTIMES:
            raise SchemaError(
                f"ProxmoxConfig.runtime must be one of "
                f"{self._VALID_RUNTIMES} (got {self.runtime!r})"
            )
        # Safety: Docker-inside-LXC is risky; flag in notes
        if self.mode == "lxc" and self.runtime == "docker":
            # Not a hard error — let operators opt in — but the
            # validator should at least flag it. We use lazy
            # self.notes append (validate() must NOT mutate) so we
            # surface via raise only when notes don't acknowledge.
            if "docker-inside-lxc" not in (self.notes or "").lower():
                # Soft signal via SchemaError — operator can add the
                # acknowledgement string to notes to opt in.
                raise SchemaError(
                    "ProxmoxConfig.mode='lxc' + runtime='docker' is "
                    "experimental/risky (PVE LXC + nested Docker has "
                    "GPU passthrough quirks). Add the literal string "
                    "'docker-inside-lxc' to notes to acknowledge + opt in."
                )
        if (self.container_id_or_vmid is not None
                and not (100 <= self.container_id_or_vmid <= 999_900)):
            raise SchemaError(
                f"ProxmoxConfig.container_id_or_vmid must be 100..999900 "
                f"(got {self.container_id_or_vmid})"
            )


@dataclass
class BootstrapConfig:
    """Y7 (UNIFIED_CONFIG plan 2026-05-09): universal-installer driver.

    Operators run `sndr bootstrap apply --scope <X>` and this block
    declares which scopes are enabled + apply_policy.

    Fields:
      - scopes: list of {'os-packages', 'gpu-runtime', 'python-runtime',
                          'container-runtime', 'model-artifacts', 'service'}
                 — the operator's "what to bootstrap" matrix
      - apply_policy: 'ask' (default) | 'auto-yes' | 'never'
      - privilege: 'sudo' | 'root' | 'user' — what privilege the
                     bootstrapper is allowed to acquire
      - rollback_on_failure: True
      - notes: free-form
    """
    scopes: list[str] = field(default_factory=list)
    apply_policy: str = "ask"
    privilege: str = "sudo"
    rollback_on_failure: bool = True
    notes: str = ""

    _VALID_SCOPES = (
        "os-packages", "gpu-runtime", "python-runtime",
        "container-runtime", "model-artifacts", "service", "all",
    )
    _VALID_POLICY = ("ask", "auto-yes", "never")
    _VALID_PRIV = ("sudo", "root", "user")

    def validate(self) -> None:
        if not isinstance(self.scopes, list):
            raise SchemaError("BootstrapConfig.scopes must be list[str]")
        for s in self.scopes:
            if s not in self._VALID_SCOPES:
                raise SchemaError(
                    f"BootstrapConfig.scopes contains invalid scope "
                    f"{s!r}; valid: {self._VALID_SCOPES}"
                )
        if self.apply_policy not in self._VALID_POLICY:
            raise SchemaError(
                f"BootstrapConfig.apply_policy must be one of "
                f"{self._VALID_POLICY} (got {self.apply_policy!r})"
            )
        if self.privilege not in self._VALID_PRIV:
            raise SchemaError(
                f"BootstrapConfig.privilege must be one of "
                f"{self._VALID_PRIV} (got {self.privilege!r})"
            )


@dataclass
class GpuTuningConfig:
    """Y8 (UNIFIED_CONFIG plan 2026-05-09): GPU tuning policy.

    Declares which GPU-side knobs the launcher should apply at boot.
    Safety: ONLY `persistence_mode`, `ulimits`, `shm_size`, and
    `vllm_args` are safe-by-default. `power_limit` and `clocks`
    require explicit `unsafe_apply: true` because they can throttle
    cards or void warranties on consumer hardware.

    Fields:
      - persistence_mode: bool — `nvidia-smi -pm 1` (recommended on)
      - power_limit_watts: int — `nvidia-smi -pl` (UNSAFE — needs opt-in)
      - clocks_mhz: dict gfx_max/mem_max — `nvidia-smi -lgc/-lmc` (UNSAFE)
      - ulimits: dict — locked memory + open files
      - transparent_hugepages: 'never' | 'madvise' | 'always'
      - unsafe_apply: bool — gate for power_limit / clocks
    """
    persistence_mode: Optional[bool] = None
    power_limit_watts: Optional[int] = None
    clocks_gfx_mhz: Optional[int] = None
    clocks_mem_mhz: Optional[int] = None
    ulimits: dict[str, str] = field(default_factory=dict)
    transparent_hugepages: Optional[str] = None
    unsafe_apply: bool = False
    notes: str = ""

    _VALID_THP = (None, "never", "madvise", "always")

    def validate(self) -> None:
        if (self.power_limit_watts is not None
                or self.clocks_gfx_mhz is not None
                or self.clocks_mem_mhz is not None) and not self.unsafe_apply:
            raise SchemaError(
                "GpuTuningConfig: power_limit_watts/clocks_gfx_mhz/"
                "clocks_mem_mhz require explicit unsafe_apply=true. "
                "These can throttle consumer cards or void warranties."
            )
        if self.power_limit_watts is not None and self.power_limit_watts < 50:
            raise SchemaError(
                f"GpuTuningConfig.power_limit_watts must be >= 50W "
                f"(got {self.power_limit_watts})"
            )
        if (self.clocks_gfx_mhz is not None
                and not (100 <= self.clocks_gfx_mhz <= 4000)):
            raise SchemaError(
                f"GpuTuningConfig.clocks_gfx_mhz must be in 100..4000 "
                f"(got {self.clocks_gfx_mhz})"
            )
        if self.transparent_hugepages not in self._VALID_THP:
            raise SchemaError(
                f"GpuTuningConfig.transparent_hugepages must be one of "
                f"{self._VALID_THP[1:]} or unset"
            )


@dataclass
class ObservabilityConfig:
    """Y14 (UNIFIED_CONFIG plan 2026-05-09): observability declarations.

    Drives memory_trace + cudagraph dispatch tracking + per-patch
    apply telemetry — all already exist in `sndr/observability/`.
    This block makes them declarative per-config instead of pure env-var.

    Fields:
      - memory_trace.enabled: bool
      - memory_trace.csv_path: str
      - cudagraph_dispatch_trace: bool
      - per_patch_telemetry: bool
    """
    memory_trace_enabled: bool = False
    memory_trace_csv_path: Optional[str] = None
    cudagraph_dispatch_trace: bool = False
    per_patch_telemetry: bool = True
    notes: str = ""

    def validate(self) -> None:
        if (self.memory_trace_enabled and not self.memory_trace_csv_path):
            raise SchemaError(
                "ObservabilityConfig.memory_trace_enabled=True requires "
                "memory_trace_csv_path"
            )
        if self.memory_trace_csv_path is not None:
            if not isinstance(self.memory_trace_csv_path, str):
                raise SchemaError("memory_trace_csv_path must be string")


@dataclass
class ServiceConfig:
    """Y10 (UNIFIED_CONFIG plan 2026-05-09): service-management contract.

    Declares which service backend the launcher should target when
    operator runs `sndr service install`. Multiple backends are
    supported per-config; the launcher picks based on `--runtime` (W-runtime).

    Fields:
      - backend: 'systemd' | 'docker_compose' | 'podman_quadlet' |
                 'kubernetes' | 'proxmox' | 'bare_metal'
      - service_name: short-name for the unit (e.g. 'genesis-35b-prod')
      - user: OS user to run as (systemd User=, docker --user)
      - working_dir: cwd for the service
      - env_file: path to file with extra env vars
      - logs_dir: where stdout/stderr land
      - restart: 'always' | 'on-failure' | 'no'
      - notes: free-form
    """
    backend: str = "docker_compose"
    service_name: str = ""
    user: Optional[str] = None
    working_dir: Optional[str] = None
    env_file: Optional[str] = None
    logs_dir: Optional[str] = None
    restart: str = "on-failure"
    notes: str = ""

    _VALID_BACKENDS = (
        "systemd", "docker_compose", "podman_quadlet",
        "kubernetes", "proxmox", "bare_metal",
    )
    _VALID_RESTART = ("always", "on-failure", "no", "unless-stopped")

    def validate(self) -> None:
        if self.backend not in self._VALID_BACKENDS:
            raise SchemaError(
                f"ServiceConfig.backend must be one of "
                f"{self._VALID_BACKENDS} (got {self.backend!r})"
            )
        if self.restart not in self._VALID_RESTART:
            raise SchemaError(
                f"ServiceConfig.restart must be one of "
                f"{self._VALID_RESTART} (got {self.restart!r})"
            )
        if self.service_name and not isinstance(self.service_name, str):
            raise SchemaError("ServiceConfig.service_name must be string")
