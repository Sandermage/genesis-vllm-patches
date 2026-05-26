# SPDX-License-Identifier: Apache-2.0
"""DockerConfig + DeploymentConfig + ``resolve_symbolic_mounts``.

Relocated from ``model_configs/schema.py`` in M.5.1. Bodies unchanged
relative to the pre-refactor version; only the import path for
:class:`SchemaError` is new.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar, Optional

from ._base import SchemaError


@dataclass
class DeploymentConfig:
    """Multi-runtime support — operator picks one at launch.

    2026-05-06 (W-runtime): Genesis configs were docker-only by design.
    Community feedback (noonghunna club-3090 docs/CONTAINER_RUNTIMES.md):
    operators run on Docker, Podman, microk8s, Proxmox LXC + bare-metal
    venv (Proxmox kernel 6.17.x footgun workaround). The schema captures
    which runtimes a config supports so the launcher can render the
    appropriate artifact and `--runtime` CLI flag picks one explicitly.
    """
    # Per-runtime support flags. Operator can render+launch any runtime
    # where the corresponding flag is True. Default = docker only (matches
    # all builtin configs as of 2026-05-06).
    docker: bool = True
    podman: bool = False           # mostly compat with docker (COMPOSE_BIN override)
    kubernetes: bool = False       # microk8s / k3s / k8s — manual translation per noonghunna disc#48
    lxc_proxmox: bool = False      # Proxmox VE LXC — see kernel 6.17.x footgun in noonghunna CONTAINER_RUNTIMES.md
    bare_metal: bool = False       # native venv (Proxmox workaround + minimal-deps option)

    # Default runtime to use when launcher called without --runtime override.
    # Must satisfy `getattr(self, default) is True` (validated below).
    default: str = "docker"

    # Known runtime names — single source of truth for valid `default` values
    # AND for `--runtime` CLI choices. Update if adding a runtime.
    # ClassVar = not a dataclass field, not serialized to YAML, not in __init__.
    KNOWN_RUNTIMES: ClassVar[tuple] = (
        "docker", "podman", "kubernetes", "lxc_proxmox", "bare_metal",
    )

    def supported_runtimes(self) -> list[str]:
        """List of runtimes this config can launch on."""
        return [r for r in self.KNOWN_RUNTIMES if getattr(self, r) is True]

    def validate(self) -> None:
        if self.default not in self.KNOWN_RUNTIMES:
            raise SchemaError(
                f"DeploymentConfig.default='{self.default}' not in known "
                f"runtimes {self.KNOWN_RUNTIMES}"
            )
        supported = self.supported_runtimes()
        if not supported:
            raise SchemaError(
                "DeploymentConfig: at least one runtime must be True. "
                "Got all runtimes False — config can't launch anywhere."
            )
        if not getattr(self, self.default):
            raise SchemaError(
                f"DeploymentConfig.default='{self.default}' not supported by "
                f"this config (deploy.{self.default}=False). "
                f"Supported runtimes: {supported}"
            )


def resolve_symbolic_mounts(
    mounts: list[str],
    host_paths: dict[str, str],
    strict: bool = True,
) -> list[str]:
    """Expand `${var}` references in mount strings via host_paths.

    Each user has different paths (`/mnt/models` vs `/data/models`,
    `~/.cache/huggingface` vs `/var/cache/huggingface`). Configs use
    symbolic refs so they're portable across rigs. Host-specific paths live
    in `~/.sndr/host.yaml` (auto-detected at install or first-run).

    Absolute paths (no `${var}`) pass through unchanged.

    Args:
        mounts: list of "<host_path>:<container_path>[:ro|rw]" strings,
            host_path may contain `${var}` references
        host_paths: dict mapping symbolic var names to absolute paths
        strict: if True (default), raise on unknown var. If False, leave
            `${var}` literal in the output — useful for `render` previews
            on machines that don't have a complete host config yet.

    Returns:
        list of mount strings with all `${var}` resolved (or left as
        literals when `strict=False` and the var is missing)

    Raises:
        SchemaError: if `strict=True` and a referenced var is missing
            from host_paths
    """
    out = []
    var_pat = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
    for m in mounts:
        def _sub(match):
            var_name = match.group(1)
            if var_name not in host_paths:
                if not strict:
                    return match.group(0)  # leave ${var} literal
                raise SchemaError(
                    f"resolve_symbolic_mounts: unknown variable "
                    f"'{var_name}' in mount '{m}'. "
                    f"Available: {sorted(host_paths.keys())}. "
                    f"Update host config (~/.sndr/host.yaml) or "
                    f"config YAML to use absolute path."
                )
            return host_paths[var_name]
        out.append(var_pat.sub(_sub, m))
    return out


@dataclass
class DockerConfig:
    """Docker container setup."""
    image: str
    container_name: str
    # Port contract — Y4 (UNIFIED_CONFIG plan 2026-05-09):
    #   - `port`             : legacy single-port shorthand. When set,
    #                          host_port and container_port both default
    #                          to it. Existing YAML configs keep working.
    #   - `host_port`        : optional explicit host-side port. Wins
    #                          over `port` when set.
    #   - `container_port`   : optional explicit container-side port
    #                          (the value passed to `vllm serve --port`).
    #                          Wins over `port` when set.
    # Use the new fields when host port differs from container port (eg.
    # multi-instance hosts running several models on the same image,
    # k8s pods with NodePort, RunPod with fixed external port).
    port: int = 8000
    host_port: Optional[int] = None
    container_port: Optional[int] = None
    shm_size: str = "8g"
    memory_limit: Optional[str] = None  # '64g'
    network: Optional[str] = None
    gpus: str = "all"
    mounts: list[str] = field(default_factory=list)
    extra_run_flags: list[str] = field(default_factory=list)

    # T1.6 (audit closure §7.4): pinned image digest. When set, the
    # launcher will refuse to launch unless the local image's
    # `RepoDigests` resolves to this exact digest, OR the operator
    # passes `--strict-image=off` to opt out. Empty/None means "no
    # digest pin" — same behavior as before T1.6.
    #
    # Format: full sha256 digest as printed by `docker inspect -f
    # '{{index .RepoDigests 0}}' <image>` (e.g.
    # "vllm/vllm-openai@sha256:abc123…"). Tag-only references are
    # rejected because tags are mutable and defeat the pin's purpose.
    image_digest: Optional[str] = None

    def effective_host_port(self) -> int:
        """Y4: returns host_port if explicitly set, else `port`."""
        return self.host_port if self.host_port is not None else self.port

    def effective_container_port(self) -> int:
        """Y4: returns container_port if explicitly set, else `port`."""
        return (self.container_port
                if self.container_port is not None else self.port)

    def effective_image_ref(self) -> str:
        """Return the immutable image reference when a digest pin exists."""
        return self.image_digest or self.image

    def validate(self) -> None:
        if not self.image:
            raise SchemaError("DockerConfig.image required")
        if not self.container_name:
            raise SchemaError("DockerConfig.container_name required")
        for name, val in (
            ("port", self.port),
            ("host_port", self.host_port),
            ("container_port", self.container_port),
        ):
            if val is None:
                continue
            if not isinstance(val, int):
                raise SchemaError(
                    f"DockerConfig.{name} must be int (got {type(val).__name__})"
                )
            if not (1 <= val <= 65535):
                raise SchemaError(
                    f"DockerConfig.{name} must be in 1..65535 (got {val})"
                )
        if self.image_digest is not None:
            if not isinstance(self.image_digest, str):
                raise SchemaError(
                    "DockerConfig.image_digest must be a string"
                )
            if "@sha256:" not in self.image_digest:
                raise SchemaError(
                    "DockerConfig.image_digest must include '@sha256:' "
                    "(use `docker inspect` RepoDigests, not a tag-only "
                    f"ref). Got: {self.image_digest!r}"
                )
