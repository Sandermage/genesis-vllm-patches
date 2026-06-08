# SPDX-License-Identifier: Apache-2.0
"""Observability service — bench / doctor / configs / evidence / jobs.

In-process implementation: bench history is read from
``~/.sndr/bench/*.json`` (one file per run), doctor walks a fixed list
of probes, configs come from the V2 registry, evidence reads
``~/.sndr/evidence/*.yaml``, and jobs is a small in-memory queue (will
be backed by SQLite in v12.1).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from sndr.product_api.schemas.observability import (
    BenchSummary,
    ConfigCatalog,
    DoctorFinding,
    DoctorReport,
    EvidenceGate,
    EvidenceReport,
    HardwareSummary,
    JobSummary,
    ModelSummary,
    PresetSummary,
    ProfileSummary,
)


# ── Bench ──────────────────────────────────────────────────────────────────


def _bench_dir() -> Path:
    return Path(os.environ.get("SNDR_HOME", "~/.sndr")).expanduser() / "bench"


def list_bench_runs(*, model: str | None = None) -> list[BenchSummary]:
    """List bench runs from ~/.sndr/bench/*.json."""
    bd = _bench_dir()
    if not bd.is_dir():
        return []
    out = []
    for f in sorted(bd.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        try:
            entry = BenchSummary(
                id=data.get("id", f.stem),
                timestamp=data.get("timestamp", datetime.now(timezone.utc)),
                model=data.get("model", ""),
                pin=data.get("pin", ""),
                wall_tps=float(data.get("wall_tps", 0)),
                decode_tpot_ms=float(data.get("decode_tpot_ms", 0)),
                ttft_ms=float(data.get("ttft_ms", 0)),
                accept_rate=data.get("accept_rate"),
                cv=float(data.get("cv", 0)),
                n=int(data.get("n", 0)),
                outcome=data.get("outcome", "success"),
                delta_tps_vs_baseline=data.get("delta_tps_vs_baseline"),
            )
        except Exception:
            continue
        if model and entry.model != model:
            continue
        out.append(entry)
    return out


# ── Doctor ─────────────────────────────────────────────────────────────────


def doctor_report() -> DoctorReport:
    """Run a fixed health-check sweep and return findings."""
    findings: list[DoctorFinding] = []

    # GPU probe
    if shutil.which("nvidia-smi") is None:
        findings.append(DoctorFinding(
            category="gpu", severity="warning",
            title="nvidia-smi not found",
            detail="CUDA toolchain may not be installed on this host.",
            remediation="Install the NVIDIA driver + CUDA toolkit, then re-run.",
        ))
    else:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode != 0:
                findings.append(DoctorFinding(
                    category="gpu", severity="warning",
                    title="nvidia-smi failed",
                    detail=out.stderr.strip(),
                ))
        except subprocess.TimeoutExpired:
            findings.append(DoctorFinding(
                category="gpu", severity="warning",
                title="nvidia-smi timeout",
            ))

    # Docker probe
    if shutil.which("docker") is None:
        findings.append(DoctorFinding(
            category="docker", severity="error",
            title="docker not installed",
            detail="The Control Center surface needs docker for container management.",
            remediation="Install docker-ce + nvidia-container-toolkit.",
        ))
    else:
        try:
            out = subprocess.run(["docker", "info"],
                                  capture_output=True, text=True, timeout=5)
            if out.returncode != 0:
                findings.append(DoctorFinding(
                    category="docker", severity="error",
                    title="docker daemon not reachable",
                    detail=out.stderr.strip(),
                ))
        except subprocess.TimeoutExpired:
            findings.append(DoctorFinding(
                category="docker", severity="error",
                title="docker info timeout",
            ))

    # Patches probe
    try:
        from sndr.dispatcher.registry import PATCH_REGISTRY
        if len(PATCH_REGISTRY) == 0:
            findings.append(DoctorFinding(
                category="patches", severity="error",
                title="PATCH_REGISTRY empty",
            ))
        else:
            findings.append(DoctorFinding(
                category="patches", severity="info",
                title=f"{len(PATCH_REGISTRY)} patches in registry",
            ))
    except Exception as e:
        findings.append(DoctorFinding(
            category="patches", severity="critical",
            title="PATCH_REGISTRY import failed",
            detail=str(e),
        ))

    # SNDR_HOME probe
    home = Path(os.environ.get("SNDR_HOME", "~/.sndr")).expanduser()
    if not home.is_dir():
        findings.append(DoctorFinding(
            category="install", severity="warning",
            title=f"SNDR_HOME {home} does not exist",
            remediation="Run `sndr install` to bootstrap operator-local config.",
        ))

    counts = {"info": 0, "warning": 0, "error": 0, "critical": 0}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1

    return DoctorReport(
        checked_at=datetime.now(timezone.utc),
        ok=counts["error"] == 0 and counts["critical"] == 0,
        findings=findings,
        counts=counts,
    )


# ── Configs catalog ────────────────────────────────────────────────────────


def config_catalog() -> ConfigCatalog:
    """Build a snapshot of the V2 catalog."""
    try:
        from sndr.model_configs.registry_v2 import (
            list_hardware, list_models, list_presets, list_profiles,
            load_hardware, load_model, load_preset_def, load_profile,
        )
    except ImportError:
        return ConfigCatalog()

    def _safe(loader: Any, ident: str) -> Any:
        try:
            return loader(ident)
        except Exception:
            return None

    models = []
    for m in list_models():
        d = _safe(load_model, m)
        if d is None:
            continue
        models.append(ModelSummary(
            id=m,
            title=getattr(d, "title", None),
            served_model_name=getattr(d, "served_model_name", None),
            family=getattr(d, "family", None),
            quant_format=getattr(d, "quant_format", None),
            kv_cache_dtype=getattr(d, "kv_cache_dtype", None),
            spec_method=getattr(d, "spec_method", None),
            parameter_count=getattr(d, "parameter_count", None),
        ))

    hardware = []
    for h in list_hardware():
        d = _safe(load_hardware, h)
        if d is None:
            continue
        hardware.append(HardwareSummary(
            id=h,
            title=getattr(d, "title", None),
            gpu=getattr(d, "gpu", None),
            gpu_count=getattr(d, "gpu_count", None),
            vram_per_gpu_gib=getattr(d, "vram_per_gpu_gib", None),
            cpu_cores=getattr(d, "cpu_cores", None),
            ram_gib=getattr(d, "ram_gib", None),
        ))

    profiles = []
    for p in list_profiles():
        d = _safe(load_profile, p)
        if d is None:
            continue
        profiles.append(ProfileSummary(
            id=p,
            title=getattr(d, "title", None),
            parent_model=getattr(d, "parent_model", None),
            role=getattr(d, "role", None),
        ))

    presets = []
    for p in list_presets():
        d = _safe(load_preset_def, p)
        if d is None:
            continue
        presets.append(PresetSummary(
            id=p,
            title=getattr(d, "title", None),
            composed_key=getattr(d, "composed_key", None),
            parent_model=getattr(d, "parent_model", None),
        ))

    return ConfigCatalog(
        models=models, hardware=hardware,
        profiles=profiles, presets=presets,
    )


# ── Evidence ───────────────────────────────────────────────────────────────


def _evidence_dir() -> Path:
    return Path(os.environ.get("SNDR_HOME", "~/.sndr")).expanduser() / "evidence"


def evidence_report() -> EvidenceReport:
    """Load evidence gate states from ~/.sndr/evidence/*.yaml."""
    ed = _evidence_dir()
    gates: list[EvidenceGate] = []
    counts = {"ok": 0, "warning": 0, "fail": 0, "skipped": 0}
    if ed.is_dir():
        for f in sorted(ed.glob("*.yaml")):
            try:
                data = yaml.safe_load(f.read_text()) or {}
            except yaml.YAMLError:
                continue
            try:
                gate = EvidenceGate(
                    id=data.get("id", f.stem),
                    name=data.get("name", f.stem),
                    status=data.get("status", "skipped"),
                    summary=data.get("summary"),
                    last_run=data.get("last_run"),
                )
                gates.append(gate)
                counts[gate.status] = counts.get(gate.status, 0) + 1
            except Exception:
                continue

    return EvidenceReport(
        gates_total=len(gates),
        gates_ok=counts["ok"],
        gates_warning=counts["warning"],
        gates_fail=counts["fail"],
        gates_skipped=counts["skipped"],
        gates=gates,
    )


# ── Jobs ───────────────────────────────────────────────────────────────────


_JOBS: dict[str, JobSummary] = {}


def list_jobs(*, state: str | None = None) -> list[JobSummary]:
    out = sorted(_JOBS.values(),
                 key=lambda j: j.started_at or datetime.now(timezone.utc),
                 reverse=True)
    if state:
        out = [j for j in out if j.state == state]
    return out


def get_job(job_id: str) -> JobSummary | None:
    return _JOBS.get(job_id)


def register_job(job: JobSummary) -> None:
    """Test-only / internal: register a job for the listing."""
    _JOBS[job.id] = job


__all__ = [
    "config_catalog",
    "doctor_report",
    "evidence_report",
    "get_job",
    "list_bench_runs",
    "list_jobs",
    "register_job",
]
