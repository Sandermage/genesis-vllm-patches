# SPDX-License-Identifier: Apache-2.0
"""Pure planning — given a config + host inventory, derive what to change.

No installs, no subprocess calls. Takes typed dataclasses in, returns a
typed `DepsPlan` out. Side-effecting `installers.apply(plan)` lives
elsewhere; this module is unit-testable in seconds without a privileged
shell.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .checkers import HostInventory


@dataclass
class PlanItem:
    """One actionable change."""
    scope: str           # 'docker' | 'nvidia' | 'python' | 'vllm' | 'config'
    action: str          # 'install' | 'upgrade' | 'configure' | 'verify'
    target: str          # human-readable target ("Docker Engine", "vllm 0.20.2rc1.dev93")
    severity: str        # 'blocker' | 'warning' | 'info'
    reason: str          # why this item exists
    suggested_command: Optional[str] = None  # one-line install hint

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "action": self.action,
            "target": self.target,
            "severity": self.severity,
            "reason": self.reason,
            "suggested_command": self.suggested_command,
        }


@dataclass
class DepsPlan:
    """Aggregate plan: all changes needed to make `cfg` runnable on `host`."""
    config_key: Optional[str]
    items: list[PlanItem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def blockers(self) -> list[PlanItem]:
        return [i for i in self.items if i.severity == "blocker"]

    def warnings(self) -> list[PlanItem]:
        return [i for i in self.items if i.severity == "warning"]

    def is_ready(self) -> bool:
        """True iff no blockers — host can run this config."""
        return not self.blockers()

    def to_dict(self) -> dict:
        return {
            "config_key": self.config_key,
            "items": [i.to_dict() for i in self.items],
            "notes": list(self.notes),
            "is_ready": self.is_ready(),
            "n_blockers": len(self.blockers()),
            "n_warnings": len(self.warnings()),
        }


# ─── Plan derivation ───────────────────────────────────────────────────


def plan_changes(cfg, inventory: HostInventory) -> DepsPlan:
    """Compute what changes the host needs to run `cfg`.

    `cfg` is a `ModelConfig` (kept as duck-typed Any to avoid an import
    cycle with the schema module). The plan covers:
      - Docker presence + daemon + nvidia runtime (if cfg.docker is set)
      - NVIDIA driver presence + GPU count
      - vLLM presence + pin allowlist (Y11 upstream block honored)
      - Python interpreter version

    The plan does NOT include in-container python deps — those are
    rendered into the bash bootstrap by the renderer (see Y1+B6).
    """
    plan = DepsPlan(config_key=getattr(cfg, "key", None))

    # ── OS
    os_ = inventory.os
    if os_.system not in ("Linux", "Darwin"):
        plan.items.append(PlanItem(
            scope="os", action="verify", target=os_.system,
            severity="warning",
            reason=(
                f"Genesis is validated on Linux + macOS dev. "
                f"{os_.system} support is best-effort."
            ),
        ))

    # ── Python
    py = inventory.python
    major, minor = (int(x) for x in py.version.split(".")[:2])
    if (major, minor) < (3, 10):
        plan.items.append(PlanItem(
            scope="python", action="upgrade",
            target=f"Python ≥ 3.10 (got {py.version})",
            severity="blocker",
            reason="vllm + Genesis stack require Python ≥ 3.10",
            suggested_command="install Python 3.12 via your distro",
        ))
    if not py.pip_present:
        plan.items.append(PlanItem(
            scope="python", action="install", target="pip",
            severity="blocker",
            reason="`pip` is needed to install vllm and runtime deps",
            suggested_command=(
                f"{py.binary_path} -m ensurepip --upgrade"
            ),
        ))

    # ── Docker (only if cfg uses docker)
    docker_cfg = getattr(cfg, "docker", None)
    if docker_cfg is not None:
        docker = inventory.docker
        if not docker.installed:
            plan.items.append(PlanItem(
                scope="docker", action="install", target="Docker Engine",
                severity="blocker",
                reason=(
                    f"Config '{cfg.key}' renders a Docker launch but "
                    f"docker is not on PATH"
                ),
                suggested_command=(
                    "follow https://docs.docker.com/engine/install/ "
                    "(do NOT pipe curl|bash; use your distro repo)"
                ),
            ))
        elif not docker.daemon_running:
            plan.items.append(PlanItem(
                scope="docker", action="configure",
                target="Docker daemon",
                severity="blocker",
                reason="`docker info` failed — daemon not running",
                suggested_command="sudo systemctl start docker",
            ))
        elif not docker.nvidia_runtime_present:
            plan.items.append(PlanItem(
                scope="docker", action="install",
                target="nvidia-container-toolkit",
                severity="blocker",
                reason=(
                    "Docker daemon runs but the nvidia runtime is not "
                    "registered — GPUs will not be visible inside the container"
                ),
                suggested_command=(
                    "https://docs.nvidia.com/datacenter/cloud-native/"
                    "container-toolkit/latest/install-guide.html"
                ),
            ))

    # ── NVIDIA driver (if config requires GPUs)
    hw = getattr(cfg, "hardware", None)
    if hw is not None and getattr(hw, "n_gpus", 0) >= 1:
        nv = inventory.nvidia
        if not nv.installed:
            plan.items.append(PlanItem(
                scope="nvidia", action="install",
                target="NVIDIA driver",
                severity="blocker",
                reason=(
                    f"Config requires {hw.n_gpus} GPU(s) but nvidia-smi "
                    f"is not on PATH"
                ),
                suggested_command="install NVIDIA datacenter driver",
            ))
        elif nv.n_gpus < hw.n_gpus:
            plan.items.append(PlanItem(
                scope="nvidia", action="verify",
                target=f"≥ {hw.n_gpus} GPU(s)",
                severity="blocker",
                reason=(
                    f"Config requires {hw.n_gpus} GPU(s); nvidia-smi "
                    f"sees {nv.n_gpus}"
                ),
            ))
        # VRAM per GPU
        min_vram = int(getattr(hw, "min_vram_per_gpu_mib", 0) or 0)
        if min_vram > 0 and nv.gpu_total_vram_mib:
            for idx, mib in enumerate(nv.gpu_total_vram_mib):
                if mib < min_vram:
                    plan.items.append(PlanItem(
                        scope="nvidia", action="verify",
                        target=f"GPU {idx} VRAM ≥ {min_vram} MiB",
                        severity="blocker",
                        reason=(
                            f"GPU {idx} has {mib} MiB; config needs "
                            f"≥ {min_vram} MiB"
                        ),
                    ))

    # ── vLLM
    vllm = inventory.vllm
    pin_required = getattr(cfg, "vllm_pin_required", None)
    upstream = getattr(cfg, "upstream", None)
    if not vllm.installed:
        plan.items.append(PlanItem(
            scope="vllm", action="install",
            target=f"vllm {pin_required or 'latest'}",
            severity="warning",   # docker-mode configs install inside image
            reason="vllm not in current Python; OK if running via Docker",
        ))
    else:
        # Y11 upstream check first (per-config policy wins)
        if upstream is not None and hasattr(upstream, "check"):
            msg = upstream.check(vllm.version)
            if msg is not None:
                plan.items.append(PlanItem(
                    scope="vllm", action="verify",
                    target=f"vllm {pin_required or vllm.version}",
                    severity="blocker",
                    reason=msg,
                ))
        # Top-level vllm_pin_required (legacy)
        elif pin_required and vllm.version != pin_required:
            plan.items.append(PlanItem(
                scope="vllm", action="upgrade",
                target=f"vllm == {pin_required}",
                severity="warning",
                reason=(
                    f"installed vllm {vllm.version} != "
                    f"vllm_pin_required {pin_required}"
                ),
                suggested_command=(
                    f"pip install --upgrade vllm=={pin_required}"
                ),
            ))

    # Model-artifacts planner: probe declared `artifacts.models`, mark
    # any missing weights / size-shortfall as blockers. The actual
    # download is done by `installers.apply_model_artifacts(plan)` which
    # shells out to `huggingface-cli` (or rsync from a local mirror).
    artifacts = getattr(cfg, "artifacts", None)
    if artifacts is not None:
        from pathlib import Path
        for spec in (getattr(artifacts, "models", None) or []):
            local_dir = Path(getattr(spec, "local_dir", "") or "")
            hf_id = getattr(spec, "hf_id", "") or ""
            min_gib = float(getattr(spec, "min_total_gib", 0) or 0)
            if not local_dir:
                continue
            if not local_dir.is_dir():
                plan.items.append(PlanItem(
                    scope="model", action="download",
                    target=f"{hf_id} → {local_dir}",
                    severity="blocker",
                    reason="model directory missing",
                    suggested_command=(
                        f"huggingface-cli download {hf_id} "
                        f"--local-dir {local_dir}"
                    ),
                ))
                continue
            # Size check (rough — operator chose `min_total_gib`)
            try:
                total = sum(
                    p.stat().st_size for p in local_dir.rglob("*")
                    if p.is_file()
                )
                gib = total / (1024 ** 3)
                if min_gib and gib < min_gib * 0.9:
                    plan.items.append(PlanItem(
                        scope="model", action="verify",
                        target=f"{hf_id} ({local_dir})",
                        severity="warning",
                        reason=(
                            f"on-disk {gib:.1f} GiB < declared min "
                            f"{min_gib} GiB — partial download?"
                        ),
                        suggested_command=(
                            f"huggingface-cli download {hf_id} "
                            f"--local-dir {local_dir}"
                        ),
                    ))
            except OSError:
                pass

    # Service planner: when cfg declares a Y10 `service` block, derive
    # the runtime backend (systemd / docker-compose / podman-quadlet)
    # and emit a `service:install` PlanItem so apply() can render the
    # unit file. Pure planning here — no writes.
    service = getattr(cfg, "service", None)
    if service is not None:
        backend = getattr(service, "backend", "") or "systemd"
        unit_name = getattr(service, "name",
                              f"sndr-{getattr(cfg, 'key', 'preset')}")
        if backend not in (
            "systemd", "docker_compose", "podman_quadlet", "bare_metal",
        ):
            plan.items.append(PlanItem(
                scope="service", action="configure",
                target=f"{unit_name}",
                severity="warning",
                reason=(
                    f"unknown service backend {backend!r} — "
                    f"supported: systemd|docker_compose|podman_quadlet"
                ),
                suggested_command=None,
            ))
        else:
            plan.items.append(PlanItem(
                scope="service", action="install",
                target=f"{unit_name} ({backend})",
                severity="info",
                reason=(
                    f"service unit not yet installed under {backend} — "
                    f"`sndr service install {getattr(cfg, 'key', '')}` "
                    f"renders + installs"
                ),
                suggested_command=(
                    f"sndr service install {getattr(cfg, 'key', '')}"
                ),
            ))

    if not plan.items:
        plan.notes.append("Host is ready: no changes required.")
    return plan
