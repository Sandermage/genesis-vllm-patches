# SPDX-License-Identifier: Apache-2.0
"""Phase 4.5 canonical IR — RuntimeContainerSpec for all container emitters.

`RuntimeCommandSpec` (sibling module `runtime_command.py`) is the canonical
vllm argv. `RuntimeContainerSpec` is the canonical CONTAINER-LEVEL invariant
that wraps it: image, mounts, ports, env, security, network. Every emitter
(docker run, docker compose, podman quadlet, kubernetes manifest, dry-run
report) takes a `RuntimeContainerSpec` as input.

Diff between any two emitter outputs for the SAME spec must be format-only
(YAML vs argv vs unit file) — never semantic (different mounts, different
env, different ports). That invariant is enforced by Phase 4.5 acceptance
tests.

Acceptance contract (Roadmap §4.5):

    sndr launch <alias> --dry-run --runtime docker
    sndr launch <alias> --dry-run --runtime compose
    sndr launch <alias> --dry-run --runtime quadlet
    sndr launch <alias> --dry-run --runtime kubernetes
    # All emit text/YAML/JSON derived from the SAME RuntimeContainerSpec.
    # Semantic diff between any two outputs = 0.

This module ships the IR + builder. Emitter refactor (compose.py, quadlet.py,
k8s.py) happens incrementally; the byte-equivalence test gates each refactor
against pre-refactor golden output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

from .runtime_command import RuntimeCommandSpec, build_runtime_command

if TYPE_CHECKING:
    from .schema import ModelConfig


__all__ = [
    "MountSpec",
    "PortSpec",
    "DeviceSpec",
    "SecuritySpec",
    "RuntimeContainerSpec",
    "build_runtime_container_spec",
]


# ─── Sub-types ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MountSpec:
    """One container mount. Source resolved from operator env vars
    (`${models_dir}`, `${hf_cache}`, etc.) at build time."""
    source: str                          # host path (resolved)
    target: str                          # container path
    mode: Literal["ro", "rw"] = "ro"

    def to_docker_arg(self) -> str:
        """Render as `-v <source>:<target>:<mode>` value (without -v)."""
        return f"{self.source}:{self.target}:{self.mode}"


@dataclass(frozen=True)
class PortSpec:
    """One published port mapping."""
    host_port: int
    container_port: int
    protocol: Literal["tcp", "udp"] = "tcp"

    def to_docker_arg(self) -> str:
        """Render as `-p <host>:<container>/<proto>` value (without -p)."""
        return f"{self.host_port}:{self.container_port}/{self.protocol}"


@dataclass(frozen=True)
class DeviceSpec:
    """GPU/device passthrough. Docker uses `--gpus`, k8s uses
    nvidia.com/gpu resources, quadlet uses `AddDevice=`."""
    selector: str                        # e.g. "all", "device=0", "device=0,1"
    capabilities: tuple[str, ...] = ()   # e.g. ("compute", "utility")


@dataclass(frozen=True)
class SecuritySpec:
    """Per-container security knobs. Currently sparse; expanded as
    emitters add stricter defaults (no-new-privileges, drop-caps, etc.)."""
    selinux_label_disable: bool = False
    cap_add: tuple[str, ...] = ()
    cap_drop: tuple[str, ...] = ()
    no_new_privileges: bool = True


# ─── Main spec ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RuntimeContainerSpec:
    """Canonical container-level IR consumed by all emitters.

    Invariants:
      - `image_digest` wins over `image` when both set (reproducibility).
      - `command.argv` is the vllm serve invocation, identical across
        emitters for the same (model, hw, profile) triplet.
      - `env` is the final merged env (model.patches + hardware.system_env);
        emitters render it as their format requires (--env, environment:,
        Environment=, env:) but DO NOT modify keys/values.
      - Emitters MAY reject a spec if their target format cannot express
        some field (e.g. quadlet has limited ulimits support); they MUST
        raise SchemaError rather than silently drop it.
    """
    # Identity / runtime backend
    runtime: Literal["docker", "podman", "compose", "quadlet",
                     "kubernetes", "proxmox", "bare-metal"]
    container_name: str

    # Image identity
    image: Optional[str]
    image_digest: Optional[str]          # pinned digest — wins over `image`

    # Vllm command (delegated to RuntimeCommandSpec for argv invariance)
    command: RuntimeCommandSpec

    # Container resources
    env: dict[str, str] = field(default_factory=dict)
    mounts: tuple[MountSpec, ...] = field(default_factory=tuple)
    ports: tuple[PortSpec, ...] = field(default_factory=tuple)
    devices: tuple[DeviceSpec, ...] = field(default_factory=tuple)
    shm_size: Optional[str] = None
    memory_limit: Optional[str] = None
    cpu_limit: Optional[str] = None

    # Networking
    network_mode: Optional[str] = None

    # Security
    security: SecuritySpec = field(default_factory=SecuritySpec)

    # Per-runtime extras (operator escape hatch — explicitly NOT structured)
    extra_run_flags: tuple[str, ...] = field(default_factory=tuple)

    # ─── Accessors ────────────────────────────────────────────────────

    def effective_image_ref(self) -> str:
        """Return the image reference an emitter must use (digest wins)."""
        if self.image_digest:
            return self.image_digest
        if self.image:
            return self.image
        raise ValueError(
            f"RuntimeContainerSpec for {self.container_name!r} has no image "
            f"or image_digest — cannot resolve container image reference"
        )


# ─── Builder ────────────────────────────────────────────────────────────


_MOUNT_RO_HINTS = (":ro",)  # legacy `source:target:ro` strings


def _parse_v1_mount_string(raw: str) -> MountSpec:
    """V1 stored mounts as 'source:target' or 'source:target:ro' strings.
    Convert to typed MountSpec."""
    parts = raw.split(":")
    if len(parts) == 2:
        return MountSpec(source=parts[0], target=parts[1], mode="rw")
    if len(parts) == 3:
        mode = parts[2]
        if mode not in ("ro", "rw"):
            mode = "rw"
        return MountSpec(source=parts[0], target=parts[1], mode=mode)
    raise ValueError(
        f"cannot parse V1 mount string {raw!r} — expected "
        f"'source:target' or 'source:target:ro'"
    )


def build_runtime_container_spec(
    cfg: "ModelConfig",
    *,
    runtime: str = "docker",
) -> RuntimeContainerSpec:
    """Compose a `RuntimeContainerSpec` from a V1 `ModelConfig`.

    Today V1 stores docker config under `cfg.docker`. After Phase 4.5 is
    consumed by compose/quadlet/k8s, the V2 composer can produce a
    `ModelConfig` plus `RuntimeContainerSpec` in one pass — but the V1
    bridge keeps existing callers working unchanged.
    """
    if cfg.docker is None:
        raise ValueError(
            f"ModelConfig {cfg.key!r} has no `docker` block — "
            f"build_runtime_container_spec requires container metadata"
        )
    docker = cfg.docker

    # Mount strings → typed MountSpec
    mounts: list[MountSpec] = []
    for raw in docker.mounts or []:
        try:
            mounts.append(_parse_v1_mount_string(raw))
        except ValueError:
            # Skip malformed entries; logging surface added in Phase 7 audit.
            continue

    # Ports — single port mapping for now (V1 schema only carries one).
    ports = (
        PortSpec(
            host_port=docker.effective_host_port(),
            container_port=docker.effective_container_port(),
        ),
    )

    # Devices — V1 only carries `--gpus all` semantics.
    devices = (DeviceSpec(selector=docker.gpus or "all"),)

    # Final env — merged genesis_env + system_env (model.patches +
    # hardware.system_env from V2 path; for V1 these come pre-merged).
    env: dict[str, str] = {}
    env.update(cfg.system_env or {})
    env.update(cfg.genesis_env or {})

    # Security — V1 `extra_run_flags` like `--security-opt label=disable`
    # are matched into structured fields; the rest carry through as-is.
    selinux_off = False
    extra_run: list[str] = []
    for flag in (docker.extra_run_flags or []):
        if flag.strip() in ("--security-opt label=disable",
                            "--security-opt=label=disable"):
            selinux_off = True
        else:
            extra_run.append(flag)
    security = SecuritySpec(selinux_label_disable=selinux_off)

    # Command argv — delegated to the established RuntimeCommandSpec
    # so argv parity tests keep passing.
    command = build_runtime_command(cfg)

    return RuntimeContainerSpec(
        runtime=runtime,                          # type: ignore[arg-type]
        container_name=docker.container_name,
        image=docker.image,
        image_digest=docker.image_digest,
        command=command,
        env=env,
        mounts=tuple(mounts),
        ports=ports,
        devices=devices,
        shm_size=docker.shm_size,
        memory_limit=docker.memory_limit,
        network_mode=docker.network,
        security=security,
        extra_run_flags=tuple(extra_run),
    )
