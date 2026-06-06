# SPDX-License-Identifier: Apache-2.0
"""Remote-install planner — turns a deployment render into an ordered, dry-run
install plan for the Setup wizard.

This is the read-only planning layer that connects two existing pieces: the
deployment renderer (``deployment.build_deployment`` produces the artifact +
commands for a preset/target) and the SSH layer. It lays the render out as the
steps a remote install actually performs — preflight SSH/SFTP check, push the
artifact, run the commands, verify the engine — and flags the dangerous,
infrastructure-mutating ones (``pct create`` / ``qm create`` / ``docker … up``).

It executes nothing. ``can_apply`` is ``False`` here: the gated SSH executor that
runs the plan is a separate, ``SNDR_ENABLE_APPLY``-guarded phase.
"""
from __future__ import annotations

import re
from typing import Any, Optional

# Commands that create or mutate infrastructure — surfaced so the operator
# reviews them deliberately before any future apply.
_DANGER_RE = re.compile(
    r"(pct create|pct start|pct exec|qm create|qm set|qm start|qm importdisk|qm disk|"
    r"rm -rf|mkfs|dd if=|docker run|docker compose .*\bup\b|provision-\w+\.sh)",
    re.IGNORECASE,
)


def _is_danger(command: str) -> bool:
    return bool(_DANGER_RE.search(command or ""))


def build_install_plan(
    *,
    host: dict[str, Any],
    preset_id: str,
    target: str,
    host_paths: Optional[dict[str, str]] = None,
    image_override: Optional[str] = None,
    with_daemon: bool = False,
) -> dict[str, Any]:
    """Render a preset/target and lay it out as an ordered, dry-run install plan.

    ``image_override`` installs the engine at an explicit vLLM pin/image;
    ``with_daemon`` bundles the SNDR management daemon into the same install."""
    from . import deployment

    dep = deployment.build_deployment(preset_id, target, host_paths=host_paths, image_override=image_override, with_daemon=with_daemon)
    artifact = dep["artifact"]
    label = host.get("label") or host.get("host") or "the host"

    steps: list[dict[str, Any]] = []
    order = 1

    def add(kind: str, title: str, *, cmd: Optional[str] = None, file: Optional[str] = None, danger: bool = False) -> None:
        nonlocal order
        step: dict[str, Any] = {"order": order, "kind": kind, "title": title, "danger": danger}
        if cmd is not None:
            step["cmd"] = cmd
        if file is not None:
            step["file"] = file
        steps.append(step)
        order += 1

    add("preflight", f"Verify SSH + SFTP reachability to {label}")
    add("sftp", f"Copy {artifact['filename']} to the host working directory", file=artifact["filename"])
    for cmd in dep.get("commands", []) or []:
        add("remote-exec", (cmd.splitlines() or [cmd])[0][:90], cmd=cmd, danger=_is_danger(cmd))
    add("verify", "Probe the engine on its port once it reports ready")

    danger_count = sum(1 for s in steps if s["danger"])
    return {
        "host": {"label": host.get("label"), "host": host.get("host")},
        "preset_id": preset_id,
        "target": target,
        "target_label": dep.get("target_label"),
        "artifact": artifact,
        "parameters": dep.get("parameters"),
        "image_override": dep.get("image_override"),
        "with_daemon": dep.get("with_daemon", False),
        "dependencies": dep.get("dependencies"),
        "steps": steps,
        "danger_count": danger_count,
        "provisions_infra": target in ("proxmox", "proxmox_vm"),
        "dry_run": True,
        "can_apply": False,  # the gated SSH executor lands in a later phase
        "notes": (
            "Plan + dry-run only — nothing runs. Real SSH execution (SFTP the "
            "artifact, run the commands) is a later phase, gated by "
            "SNDR_ENABLE_APPLY + an explicit confirm."
        ),
    }


def apply_install_plan(
    *,
    host: dict[str, Any],
    preset_id: str,
    target: str,
    ssh_target: dict[str, Any],
    run_apply: Any,
    apply_enabled: bool,
    confirm: bool,
    host_paths: Optional[dict[str, str]] = None,
    image_override: Optional[str] = None,
    with_daemon: bool = False,
) -> dict[str, Any]:
    """Execute an install plan on a host over SSH — the gated apply phase.

    Two hard gates: ``apply_enabled`` (SNDR_ENABLE_APPLY) AND an explicit
    ``confirm``. Reuses :func:`build_install_plan` for the artifact + ordered
    commands, then hands them to ``run_apply`` (injected = ssh_client.run_apply)
    for SFTP + remote execution. Returns the per-step results.
    """
    if not apply_enabled:
        return {"ok": False, "applied": False,
                "error": "apply is disabled — start the daemon with SNDR_ENABLE_APPLY=1"}
    if not confirm:
        return {"ok": False, "applied": False, "error": "explicit confirm is required to run on a host"}

    plan = build_install_plan(host=host, preset_id=preset_id, target=target, host_paths=host_paths, image_override=image_override, with_daemon=with_daemon)
    artifact = plan["artifact"]
    commands = [s["cmd"] for s in plan["steps"] if s.get("kind") == "remote-exec" and s.get("cmd")]
    exec_result = run_apply(
        ssh_target, artifact_name=artifact["filename"], artifact_content=artifact["content"], commands=commands,
    )
    return {
        "ok": bool(exec_result.get("ok")),
        "applied": True,
        "target": target,
        "target_label": plan.get("target_label"),
        "host": plan.get("host"),
        "artifact": artifact["filename"],
        "steps": exec_result.get("steps", []),
        "error": exec_result.get("error"),
    }


__all__ = ["build_install_plan", "apply_install_plan"]
