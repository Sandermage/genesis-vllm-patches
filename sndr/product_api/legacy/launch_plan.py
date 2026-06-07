# SPDX-License-Identifier: Apache-2.0
"""Read-only launch plan contract for SNDR GUI clients.

This module is deliberately narrower than the eventual lifecycle API. It
answers "what would the GUI launch, and why is Apply allowed or blocked?"
without writing files, starting services, opening SSH sessions, or importing
runtime-heavy vLLM modules beyond the existing preset compose path.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal, Optional

from .capabilities import collect_capabilities
from .overview import collect_catalog_summary
from .patches.doctor import run_doctor as run_patch_doctor
from .presets import explain_preset, get_preset


GateStatus = Literal["pass", "warning", "blocked", "planned"]
ArtifactKind = Literal["compose", "systemd", "commands", "env"]


@dataclass(frozen=True)
class LaunchPlanGate:
    """One GUI readiness gate attached to a launch plan."""

    id: str
    title: str
    status: GateStatus
    detail: str
    action: str


@dataclass(frozen=True)
class LaunchPlanEndpoint:
    """One runtime endpoint the GUI can display or copy."""

    label: str
    url: str


@dataclass(frozen=True)
class LaunchPlanArtifact:
    """A rendered preview artifact for the selected runtime target."""

    kind: ArtifactKind
    title: str
    content: str


@dataclass(frozen=True)
class LaunchPlanResult:
    """Single immutable launch plan snapshot for web/desktop clients."""

    plan_id: str
    preset_id: str
    runtime_target: str
    patch_policy: str
    mode: str
    host: str
    actionable: bool
    action_reason: str
    summary: dict[str, Any]
    gates: tuple[LaunchPlanGate, ...]
    endpoints: tuple[LaunchPlanEndpoint, ...]
    artifacts: tuple[LaunchPlanArtifact, ...]
    cli_mirror: tuple[str, ...]
    events: tuple[dict[str, str], ...]


def build_launch_plan(
    *,
    preset_id: str,
    runtime_target: str = "docker_compose",
    patch_policy: str = "safe",
    host: str = "127.0.0.1",
    mode: str = "remote",
) -> LaunchPlanResult:
    """Build a deterministic, read-only launch plan for one preset.

    The result is intentionally useful before lifecycle writes exist. The GUI
    can show the exact plan, gates, CLI mirror and generated artifacts, while
    Apply stays disabled unless every blocking gate is cleared by future
    Product API work.
    """
    preset = get_preset(preset_id)
    explain = explain_preset(preset_id)
    capabilities = collect_capabilities()
    catalog = collect_catalog_summary()
    doctor = run_patch_doctor()

    card = explain.card or preset.card or {}
    composed = explain.composed
    runtime = next(
        (
            target
            for target in capabilities.runtime_targets
            if target.id == runtime_target
        ),
        None,
    )
    service_lifecycle = next(
        (
            feature
            for feature in capabilities.features
            if feature.id == "service_lifecycle"
        ),
        None,
    )
    benchmark_runs = next(
        (
            feature
            for feature in capabilities.features
            if feature.id == "benchmark_runs"
        ),
        None,
    )

    gates = (
        LaunchPlanGate(
            id="catalog",
            title="Catalog Snapshot",
            detail=(
                "V2 registry loaded without preset errors"
                if catalog.preset_load_error_count == 0
                else f"{catalog.preset_load_error_count} preset load errors"
            ),
            status="pass" if catalog.preset_load_error_count == 0 else "blocked",
            action="Open catalog doctor",
        ),
        LaunchPlanGate(
            id="preset_card",
            title="Preset Card",
            detail=(
                f"{_card_text(card, 'title', preset_id)} is available"
                if preset.has_card
                else "Preset has no operator-facing product card yet"
            ),
            status="pass" if preset.has_card else "warning",
            action="Open preset card",
        ),
        LaunchPlanGate(
            id="runtime_target",
            title="Runtime Target",
            detail=runtime.detail if runtime else "Runtime target is unknown",
            status=_runtime_gate_status(runtime.status if runtime else None),
            action="Check runtime tools",
        ),
        LaunchPlanGate(
            id="engine_package",
            title="Engine Package",
            detail=(
                "SNDR/vLLM engine package is installed"
                if capabilities.platform.engine_installed
                else "Engine package was not detected in this Python shell"
            ),
            status="pass" if capabilities.platform.engine_installed else "warning",
            action="Run environment doctor",
        ),
        LaunchPlanGate(
            id="patch_doctor",
            title="Patch Registry Doctor",
            detail=(
                f"{doctor.coverage.mapped}/{doctor.coverage.total} apply modules mapped"
                if not doctor.issues
                else f"{len(doctor.issues)} registry validation issues"
            ),
            status="pass" if not doctor.issues else "blocked",
            action="Open patch doctor",
        ),
        LaunchPlanGate(
            id="service_lifecycle",
            title="Service Lifecycle API",
            detail=(
                service_lifecycle.detail
                if service_lifecycle
                else "Write-safe lifecycle Product API is not registered"
            ),
            status=(
                "pass"
                if service_lifecycle and service_lifecycle.status == "available"
                else "blocked"
            ),
            action="Implement lifecycle API",
        ),
        LaunchPlanGate(
            id="evidence_orchestration",
            title="Evidence Orchestration",
            detail=(
                benchmark_runs.detail
                if benchmark_runs
                else "Benchmark and proof job orchestration is pending"
            ),
            status=(
                "pass"
                if benchmark_runs and benchmark_runs.status == "available"
                else "warning"
            ),
            action="Open evidence plan",
        ),
        LaunchPlanGate(
            id="release_proof",
            title="Release Proof",
            detail=(
                "Generate a proof/report bundle (Reports) before a production "
                "launch — recommended, not required for a gated apply"
            ),
            status="warning",
            action="Generate proof bundle",
        ),
    )

    blocked = [gate for gate in gates if gate.status == "blocked"]
    actionable = not blocked
    action_reason = (
        "All launch gates passed"
        if actionable
        else "Blocked by: " + ", ".join(gate.title for gate in blocked)
    )
    endpoints = _endpoints(host)
    cli_mirror = _cli_mirror(
        preset_id=preset_id,
        runtime_target=runtime_target,
        patch_policy=patch_policy,
        host=host,
    )
    plan_id = _plan_id(
        preset_id=preset_id,
        runtime_target=runtime_target,
        patch_policy=patch_policy,
        mode=mode,
        host=host,
    )

    summary = {
        "model": preset.model,
        "hardware": preset.hardware,
        "profile": preset.profile,
        "runtime": preset.runtime,
        "title": _card_text(card, "title", preset_id),
        "status": _card_text(card, "status", "unannotated"),
        "routing_family": _card_text(card, "routing_family", "unknown"),
        "context": composed.get("max_model_len"),
        "max_num_seqs": composed.get("max_num_seqs"),
        "kv_cache_dtype": composed.get("kv_cache_dtype"),
        "spec_decode_method": composed.get("spec_decode_method"),
        "spec_decode_K": composed.get("spec_decode_K"),
        "enabled_patches_count": composed.get("enabled_patches_count", 0),
        "fallback_preset": _card_text(card, "fallback_preset", ""),
        "primary_metric_kind": _metric_value(card, "kind", "Metric"),
        "primary_metric_value": _metric_value(card, "value", None),
        "evidence_visibility": _card_text(card, "evidence_visibility", "unknown"),
        "patch_registry_size": doctor.registry_size,
        "patch_coverage_total": doctor.coverage.total,
        "patch_coverage_mapped": doctor.coverage.mapped,
    }

    return LaunchPlanResult(
        plan_id=plan_id,
        preset_id=preset_id,
        runtime_target=runtime_target,
        patch_policy=patch_policy,
        mode=mode,
        host=host,
        actionable=actionable,
        action_reason=action_reason,
        summary=summary,
        gates=gates,
        endpoints=endpoints,
        artifacts=_artifacts(
            preset_id=preset_id,
            runtime_target=runtime_target,
            patch_policy=patch_policy,
            host=host,
            mode=mode,
            plan_id=plan_id,
        ),
        cli_mirror=cli_mirror,
        events=_events(
            preset_id=preset_id,
            runtime_target=runtime_target,
            action_reason=action_reason,
        ),
    )


def _runtime_gate_status(status: Optional[str]) -> GateStatus:
    if status == "available":
        return "pass"
    if status in ("partial", "render_only"):
        return "warning"
    if status == "deferred":
        return "planned"
    return "blocked"


def _card_text(card: dict[str, Any], key: str, fallback: str) -> str:
    value = card.get(key)
    if isinstance(value, str) and value.strip():
        return value
    return fallback


def _metric_value(card: dict[str, Any], key: str, fallback: Any) -> Any:
    metric = card.get("primary_metric")
    if isinstance(metric, dict) and metric.get(key) is not None:
        return metric[key]
    return fallback


def _plan_id(
    *,
    preset_id: str,
    runtime_target: str,
    patch_policy: str,
    mode: str,
    host: str,
) -> str:
    raw = f"{preset_id}:{runtime_target}:{patch_policy}:{mode}:{host}"
    suffix = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    safe_preset = "".join(ch if ch.isalnum() else "_" for ch in preset_id)
    return f"plan_{safe_preset}_{suffix}"


def _endpoints(host: str) -> tuple[LaunchPlanEndpoint, ...]:
    return (
        LaunchPlanEndpoint("OpenAI API", f"http://{host}:8000/v1"),
        LaunchPlanEndpoint("Health", f"http://{host}:8000/health"),
        LaunchPlanEndpoint("Metrics", f"http://{host}:8001/metrics"),
        LaunchPlanEndpoint("Docs", f"http://{host}:8000/docs"),
    )


def _cli_mirror(
    *,
    preset_id: str,
    runtime_target: str,
    patch_policy: str,
    host: str,
) -> tuple[str, ...]:
    return (
        f"python -m sndr.cli preset explain {preset_id}",
        (
            "python -m sndr.cli launch plan "
            f"--preset {preset_id} "
            f"--runtime-target {runtime_target} "
            f"--patch-policy {patch_policy} --dry-run"
        ),
        (
            "python -m sndr.cli service render "
            f"--preset {preset_id} --target {runtime_target}"
        ),
        "python -m sndr.cli patches doctor",
        f"curl http://{host}:8000/health",
    )


def _artifacts(
    *,
    preset_id: str,
    runtime_target: str,
    patch_policy: str,
    host: str,
    mode: str,
    plan_id: str,
) -> tuple[LaunchPlanArtifact, ...]:
    return (
        LaunchPlanArtifact(
            kind="compose",
            title="Docker Compose Preview",
            content=_compose_yaml(
                preset_id=preset_id,
                runtime_target=runtime_target,
                patch_policy=patch_policy,
                host=host,
                plan_id=plan_id,
            ),
        ),
        LaunchPlanArtifact(
            kind="systemd",
            title="systemd Unit Preview",
            content=_systemd_unit(
                preset_id=preset_id,
                runtime_target=runtime_target,
                patch_policy=patch_policy,
                plan_id=plan_id,
            ),
        ),
        LaunchPlanArtifact(
            kind="commands",
            title="Operator Command Preview",
            content="\n".join(
                _cli_mirror(
                    preset_id=preset_id,
                    runtime_target=runtime_target,
                    patch_policy=patch_policy,
                    host=host,
                )
            ),
        ),
        LaunchPlanArtifact(
            kind="env",
            title="Environment Preview",
            content=_env_file(
                preset_id=preset_id,
                runtime_target=runtime_target,
                patch_policy=patch_policy,
                host=host,
                mode=mode,
                plan_id=plan_id,
            ),
        ),
    )


def _compose_yaml(
    *,
    preset_id: str,
    runtime_target: str,
    patch_policy: str,
    host: str,
    plan_id: str,
) -> str:
    return "\n".join(
        [
            'version: "3.8"',
            "services:",
            "  sndr-vllm:",
            "    image: ghcr.io/sndr/vllm-runtime:catalog",
            "    command:",
            "      - python",
            "      - -m",
            "      - sndr.cli",
            "      - launch",
            "      - plan",
            f"      - --preset={preset_id}",
            f"      - --runtime-target={runtime_target}",
            f"      - --patch-policy={patch_policy}",
            "      - --dry-run",
            "    ports:",
            '      - "8000:8000"',
            '      - "8001:8001"',
            "    environment:",
            f"      SNDR_PLAN_ID: {plan_id}",
            f"      SNDR_RUNTIME_HOST: {host}",
            f"      SNDR_PRESET: {preset_id}",
            f"      SNDR_PATCH_POLICY: {patch_policy}",
        ]
    )


def _systemd_unit(
    *,
    preset_id: str,
    runtime_target: str,
    patch_policy: str,
    plan_id: str,
) -> str:
    return "\n".join(
        [
            "[Unit]",
            f"Description=SNDR vLLM runtime preview for {preset_id}",
            "After=network-online.target",
            "",
            "[Service]",
            f"Environment=SNDR_PLAN_ID={plan_id}",
            f"Environment=SNDR_PRESET={preset_id}",
            f"Environment=SNDR_RUNTIME_TARGET={runtime_target}",
            f"Environment=SNDR_PATCH_POLICY={patch_policy}",
            (
                "ExecStart=/usr/bin/python -m sndr.cli "
                f"launch plan --preset {preset_id} "
                f"--runtime-target {runtime_target} "
                f"--patch-policy {patch_policy} --dry-run"
            ),
            "Restart=on-failure",
            "RestartSec=5s",
            "",
            "[Install]",
            "WantedBy=multi-user.target",
        ]
    )


def _env_file(
    *,
    preset_id: str,
    runtime_target: str,
    patch_policy: str,
    host: str,
    mode: str,
    plan_id: str,
) -> str:
    base = [
        f"SNDR_PLAN_ID={plan_id}",
        f"SNDR_PRESET={preset_id}",
        f"SNDR_RUNTIME_TARGET={runtime_target}",
        f"SNDR_PATCH_POLICY={patch_policy}",
        f"SNDR_RUNTIME_MODE={mode}",
        f"SNDR_OPENAI_BASE_URL=http://{host}:8000/v1",
        f"SNDR_METRICS_URL=http://{host}:8001/metrics",
        "SNDR_DRY_RUN=true",
        "SNDR_APPLY_ENABLED=false",
    ]
    # Operator patch overrides become explicit enable/disable env flags so the
    # launch reflects them (see patch_overrides.py).
    try:
        from .patch_overrides import env_lines

        overrides = env_lines()
        if overrides:
            base.append("# operator patch overrides")
            base.extend(overrides)
    except Exception:
        pass
    return "\n".join(base)


def _events(
    *,
    preset_id: str,
    runtime_target: str,
    action_reason: str,
) -> tuple[dict[str, str], ...]:
    return (
        {
            "level": "info",
            "message": f"Launch plan prepared for {preset_id}",
        },
        {
            "level": "info",
            "message": f"Runtime target selected: {runtime_target}",
        },
        {
            "level": "warning" if action_reason.startswith("Blocked") else "info",
            "message": action_reason,
        },
    )


__all__ = [
    "ArtifactKind",
    "GateStatus",
    "LaunchPlanArtifact",
    "LaunchPlanEndpoint",
    "LaunchPlanGate",
    "LaunchPlanResult",
    "build_launch_plan",
]
