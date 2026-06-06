# SPDX-License-Identifier: Apache-2.0
"""Read-only service lifecycle plan (plan-before-apply, no execution).

Answers "what would start/stop/restart/status/logs do for this preset on this
runtime target?" as a structured, copyable plan. It never runs a process,
opens SSH, or writes files — the apply side is a separate future GO.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

ACTIONS = ("start", "stop", "restart", "status", "logs")
MUTATING = {"start", "stop", "restart"}


@dataclass(frozen=True)
class ServiceStep:
    order: int
    title: str
    command: str


@dataclass(frozen=True)
class ServiceActionPlan:
    plan_id: str
    preset_id: str
    action: str
    runtime_target: str
    host: str
    container_name: str
    mutating: bool
    actionable: bool
    action_reason: str
    steps: tuple[ServiceStep, ...]
    side_effects: tuple[str, ...]
    gates: tuple[dict, ...]
    cli_mirror: tuple[str, ...]
    rollback: str


def _commands(action: str, runtime_target: str, container: str, preset_id: str) -> list[str]:
    rt = runtime_target.lower()
    if rt in ("docker_compose", "compose"):
        compose = f"docker compose -p sndr-{preset_id}"
        return {
            "start": [f"{compose} up -d"],
            "stop": [f"{compose} down"],
            "restart": [f"{compose} restart"],
            "status": [f"{compose} ps"],
            "logs": [f"{compose} logs -f --tail=200"],
        }[action]
    if rt == "systemd":
        unit = f"{container}.service"
        return {
            "start": [f"systemctl --user start {unit}"],
            "stop": [f"systemctl --user stop {unit}"],
            "restart": [f"systemctl --user restart {unit}"],
            "status": [f"systemctl --user status {unit}"],
            "logs": [f"journalctl --user -u {unit} -f -n 200"],
        }[action]
    if rt in ("podman", "quadlet"):
        return {
            "start": [f"podman start {container}"],
            "stop": [f"podman stop {container}"],
            "restart": [f"podman restart {container}"],
            "status": [f"podman ps --filter name={container}"],
            "logs": [f"podman logs -f --tail 200 {container}"],
        }[action]
    if rt == "kubernetes":
        return {
            "start": [f"kubectl apply -f sndr-{preset_id}.yaml"],
            "stop": [f"kubectl delete deploy sndr-{preset_id}"],
            "restart": [f"kubectl rollout restart deploy/sndr-{preset_id}"],
            "status": [f"kubectl get pods -l app=sndr-{preset_id}"],
            "logs": [f"kubectl logs -f deploy/sndr-{preset_id} --tail=200"],
        }[action]
    # docker / bare default
    return {
        "start": [f"docker start {container}"],
        "stop": [f"docker stop {container}"],
        "restart": [f"docker restart {container}"],
        "status": [f"docker ps --filter name={container}"],
        "logs": [f"docker logs -f --tail 200 {container}"],
    }[action]


def build_service_plan(
    *,
    preset_id: str,
    action: str = "status",
    runtime_target: str = "docker_compose",
    host: str = "127.0.0.1",
) -> ServiceActionPlan:
    """Build a read-only lifecycle plan for one preset and action."""
    normalized = action.lower()
    if normalized not in ACTIONS:
        raise ValueError(f"Unknown service action: {action}")

    from .presets import get_preset

    preset = get_preset(preset_id)  # raises PresetNotFoundError on miss
    container = f"vllm-{preset_id}"
    mutating = normalized in MUTATING

    commands = _commands(normalized, runtime_target, container, preset_id)
    preflight: list[str] = []
    if normalized == "start":
        preflight = [
            "docker image inspect ghcr.io/sndr/vllm-runtime:catalog",
            "nvidia-smi --query-gpu=memory.free --format=csv,noheader",
        ]
    steps = tuple(
        ServiceStep(order=index + 1, title=title, command=command)
        for index, (title, command) in enumerate(
            [("Preflight", cmd) for cmd in preflight]
            + [(normalized.capitalize(), cmd) for cmd in commands]
        )
    )

    side_effects: tuple[str, ...] = ()
    rollback = "No mutation — read-only command."
    if normalized == "start":
        side_effects = (
            f"Creates/starts container {container}",
            "Binds ports 8000 (OpenAI API) and 8001 (metrics)",
            "Loads the model into GPU memory",
        )
        rollback = f"Run the stop plan to remove {container}."
    elif normalized == "stop":
        side_effects = (f"Stops and removes container {container}", "Frees GPU memory and ports")
        rollback = "Run the start plan to bring the service back."
    elif normalized == "restart":
        side_effects = (f"Recreates container {container}", "Brief downtime during restart")
        rollback = "Re-run restart, or use start after a failed stop."

    gates = (
        {
            "id": "lifecycle_api",
            "title": "Service Lifecycle API",
            "status": "blocked" if mutating else "pass",
            "detail": (
                "Write-safe apply endpoint is required before the GUI can run this action."
                if mutating
                else "Read-only status/logs can be mirrored to the operator."
            ),
        },
        {
            "id": "preset_card",
            "title": "Preset",
            "status": "pass" if preset.has_card else "warning",
            "detail": (
                f"{preset.model} on {preset.hardware}"
                if preset.has_card
                else "Preset has no operator card; review before launch."
            ),
        },
    )

    actionable = not mutating and False  # status/logs still need a job/stream API
    action_reason = (
        "Lifecycle writes are gated until the apply API is enabled."
        if mutating
        else "Read-only preview — copy the commands to run them over SSH."
    )

    plan_id = "svcplan_" + hashlib.sha256(
        f"{preset_id}:{normalized}:{runtime_target}:{host}".encode("utf-8")
    ).hexdigest()[:12]

    cli_mirror = tuple(
        [f"sndr service {normalized} --preset {preset_id} --runtime-target {runtime_target}"]
        + list(commands)
    )

    return ServiceActionPlan(
        plan_id=plan_id,
        preset_id=preset_id,
        action=normalized,
        runtime_target=runtime_target,
        host=host,
        container_name=container,
        mutating=mutating,
        actionable=actionable,
        action_reason=action_reason,
        steps=steps,
        side_effects=side_effects,
        gates=gates,
        cli_mirror=cli_mirror,
        rollback=rollback,
    )


__all__ = ["ServiceActionPlan", "ServiceStep", "build_service_plan"]
