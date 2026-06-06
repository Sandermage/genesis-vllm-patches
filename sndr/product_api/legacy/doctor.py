# SPDX-License-Identifier: Apache-2.0
"""Aggregated environment + readiness doctor for the GUI/CLI.

Combines the read-only Product API signals — platform/runtime capabilities,
catalog health, patch-registry doctor and proof-artifact status — into a flat
list of categorised findings with severity, evidence, suggested action and a
CLI mirror. Import-safe without vLLM/torch (heavy modules lazy-imported).
"""
from __future__ import annotations

from dataclasses import dataclass, field


SEVERITY_ORDER = ("blocked", "warning", "info", "ok")


@dataclass(frozen=True)
class DoctorFinding:
    """One diagnostic finding rendered as an actionable row."""

    category: str
    id: str
    title: str
    severity: str  # "ok" | "info" | "warning" | "blocked"
    detail: str = ""
    evidence: str = ""
    action: str = ""
    cli: str = ""


@dataclass(frozen=True)
class DoctorReport:
    """Aggregated diagnostics snapshot."""

    findings: tuple[DoctorFinding, ...]
    summary: dict[str, int]
    categories: tuple[str, ...]
    generated_for: str = "current host"
    warnings: tuple[str, ...] = field(default_factory=tuple)


_RUNTIME_SEVERITY = {
    "available": "ok",
    "partial": "warning",
    "render_only": "warning",
    "deferred": "info",
    "missing": "blocked",
}


def collect_doctor_report() -> DoctorReport:
    """Build the aggregated doctor report from read-only Product API sources."""
    from .capabilities import collect_capabilities
    from .overview import collect_catalog_summary

    findings: list[DoctorFinding] = []
    warnings: list[str] = []

    caps = collect_capabilities()
    platform = caps.platform

    # --- Environment -------------------------------------------------------
    findings.append(DoctorFinding(
        category="environment",
        id="engine",
        title="SNDR / vLLM engine",
        severity="ok" if platform.engine_installed else "warning",
        detail="Engine package is importable in this shell."
        if platform.engine_installed
        else "Engine package is not importable here — live runtime checks are limited.",
        evidence=f"engine_installed={platform.engine_installed}",
        action="" if platform.engine_installed else "Install the engine package on the GPU host to enable live checks.",
        cli="python -m vllm.sndr_core.cli doctor --all",
    ))
    findings.append(DoctorFinding(
        category="environment",
        id="python",
        title="Python runtime",
        severity="info",
        detail=f"Python {platform.python_version} on {platform.os_name} / {platform.machine}.",
        evidence=f"{platform.package_name} {platform.sndr_core_version}",
        action="Python 3.10+ is required for the Product API.",
    ))

    # --- Runtime targets ---------------------------------------------------
    for target in caps.runtime_targets:
        severity = _RUNTIME_SEVERITY.get(target.status, "info")
        present = ", ".join(target.present_tools) or "none"
        missing = [tool for tool in target.required_tools if tool not in target.present_tools]
        findings.append(DoctorFinding(
            category="runtime",
            id=target.id,
            title=target.title,
            severity=severity,
            detail=target.detail,
            evidence=f"status={target.status}; tools present: {present}",
            action="Install: " + ", ".join(missing) if missing else "Runtime tooling present.",
        ))

    # --- Lifecycle / write-safety features --------------------------------
    for feature in caps.features:
        if feature.id not in {"service_lifecycle", "benchmark_runs", "web_daemon"}:
            continue
        severity = "ok" if feature.status == "available" else (
            "blocked" if feature.id == "service_lifecycle" and feature.status != "available" else "info"
        )
        findings.append(DoctorFinding(
            category="lifecycle",
            id=feature.id,
            title=feature.title,
            severity=severity,
            detail=feature.detail,
            evidence=f"status={feature.status}",
            action="" if severity == "ok" else "Pending a write-safe Product API contract.",
        ))

    # --- Catalog health ----------------------------------------------------
    catalog = collect_catalog_summary()
    findings.append(DoctorFinding(
        category="catalog",
        id="load",
        title="V2 catalog load",
        severity="ok" if catalog.preset_load_error_count == 0 else "blocked",
        detail="Catalog composed without preset load errors."
        if catalog.preset_load_error_count == 0
        else f"{catalog.preset_load_error_count} preset load error(s).",
        evidence=(
            f"models={catalog.models_count}, hardware={catalog.hardware_count}, "
            f"profiles={catalog.profiles_count}, presets={catalog.presets_count}"
        ),
        action="" if catalog.preset_load_error_count == 0 else "Inspect failing preset YAML and re-validate.",
        cli="python -m vllm.sndr_core.cli preset list",
    ))
    if catalog.unannotated_presets_count:
        findings.append(DoctorFinding(
            category="catalog",
            id="cards",
            title="Preset product cards",
            severity="info",
            detail=f"{catalog.unannotated_presets_count} preset(s) without an operator card.",
            evidence=f"{catalog.preset_cards_count} cards / {catalog.presets_count} presets",
            action="Annotate presets with a card to expose them in the operator catalog.",
        ))

    # --- Patch registry doctor --------------------------------------------
    try:
        from .patches.doctor import run_doctor

        patch_doctor = run_doctor()
        coverage = patch_doctor.coverage
        total = getattr(coverage, "total", 0)
        mapped = getattr(coverage, "mapped", 0)
        unmapped = getattr(coverage, "unmapped", ()) or ()
        intentional = getattr(coverage, "intentionally_unmapped", ()) or ()
        issues = patch_doctor.issues or ()
        findings.append(DoctorFinding(
            category="patches",
            id="validation",
            title="Patch registry validation",
            severity="ok" if not issues else "blocked",
            detail="Registry passes validation." if not issues else f"{len(issues)} validation issue(s).",
            evidence=f"registry_size={patch_doctor.registry_size}",
            action="" if not issues else "Resolve registry validation issues before release.",
            cli="python -m vllm.sndr_core.cli patches doctor",
        ))
        findings.append(DoctorFinding(
            category="patches",
            id="coverage",
            title="Apply-module coverage",
            severity="ok" if not list(unmapped) else "warning",
            detail=f"{mapped}/{total} patches mapped to an apply module.",
            evidence=f"unmapped={len(list(unmapped))}, intentionally_unmapped={len(list(intentional))}",
            action="" if not list(unmapped) else "Map or waive the remaining patches.",
        ))
    except Exception as exc:  # pragma: no cover - environment dependent
        warnings.append(f"patch doctor unavailable: {type(exc).__name__}: {exc}")

    # --- Proof / evidence --------------------------------------------------
    try:
        from .patches.proof_status import proof_status

        proof = proof_status()
        counts = getattr(proof, "counts", {}) or {}
        dead = int(counts.get("dead", 0))
        total = getattr(proof, "total", 0)
        findings.append(DoctorFinding(
            category="evidence",
            id="proof",
            title="Proof artifacts",
            severity="warning" if dead else "ok",
            detail=f"{total} proof artifact(s) tracked." + (f" {dead} dead." if dead else ""),
            evidence=", ".join(f"{k}={v}" for k, v in counts.items()) or "no buckets",
            action="Re-prove dead patches." if dead else "",
            cli="python -m vllm.sndr_core.cli patches prove --all",
        ))
    except Exception as exc:  # pragma: no cover - environment dependent
        warnings.append(f"proof status unavailable: {type(exc).__name__}: {exc}")

    # --- Release readiness rollup -----------------------------------------
    blocking = [f for f in findings if f.severity == "blocked"]
    findings.append(DoctorFinding(
        category="release",
        id="readiness",
        title="Release readiness",
        severity="blocked" if blocking else "warning",
        detail=(
            f"{len(blocking)} blocking finding(s) must clear before a writable launch."
            if blocking
            else "No blocking findings; writable launch still gated by lifecycle API."
        ),
        evidence=", ".join(f"{f.category}:{f.id}" for f in blocking) or "no blockers",
        action="Implement the write-safe lifecycle + proof gates to enable Apply Launch.",
    ))

    summary: dict[str, int] = {key: 0 for key in SEVERITY_ORDER}
    for finding in findings:
        summary[finding.severity] = summary.get(finding.severity, 0) + 1

    categories = tuple(dict.fromkeys(finding.category for finding in findings))
    return DoctorReport(
        findings=tuple(findings),
        summary=summary,
        categories=categories,
        warnings=tuple(warnings),
    )


__all__ = ["DoctorFinding", "DoctorReport", "collect_doctor_report"]
