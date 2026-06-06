# SPDX-License-Identifier: Apache-2.0
"""HTTP routes for observability surfaces.

Combines bench, doctor, configs, evidence, and jobs into one module so
the GUI's monitoring tabs all share a single set of FastAPI routers.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query

from sndr.product_api.domain.observability_service import (
    config_catalog,
    doctor_report,
    evidence_report,
    get_job,
    list_bench_runs,
    list_jobs,
)
from sndr.product_api.schemas.common import Envelope, ResponseMeta
from sndr.product_api.schemas.observability import (
    BenchSummary,
    ConfigCatalog,
    DoctorReport,
    EvidenceReport,
    JobSummary,
)


def _meta() -> ResponseMeta:
    return ResponseMeta(
        request_id=uuid4().hex,
        engine=None,
        pin=None,
        timestamp=datetime.now(timezone.utc),
    )


# ── Bench ──────────────────────────────────────────────────────────────────

bench_router = APIRouter(prefix="/api/v1/bench", tags=["bench"])


@bench_router.get("/runs", response_model=Envelope[list[BenchSummary]],
                   summary="List bench runs")
async def list_bench_runs_endpoint(
    model: str | None = Query(default=None, description="Filter by model id"),
) -> Envelope[list[BenchSummary]]:
    return Envelope(data=list_bench_runs(model=model), meta=_meta())


# ── Doctor ─────────────────────────────────────────────────────────────────

doctor_router = APIRouter(prefix="/api/v1/doctor", tags=["doctor"])


@doctor_router.get("", response_model=Envelope[DoctorReport],
                    summary="Run a health-check sweep")
async def doctor_endpoint() -> Envelope[DoctorReport]:
    return Envelope(data=doctor_report(), meta=_meta())


# ── Configs ────────────────────────────────────────────────────────────────

configs_router = APIRouter(prefix="/api/v1/configs", tags=["configs"])


@configs_router.get("", response_model=Envelope[ConfigCatalog],
                     summary="V2 config catalog snapshot")
async def config_catalog_endpoint() -> Envelope[ConfigCatalog]:
    return Envelope(data=config_catalog(), meta=_meta())


# ── Evidence ───────────────────────────────────────────────────────────────

evidence_router = APIRouter(prefix="/api/v1/evidence", tags=["evidence"])


@evidence_router.get("", response_model=Envelope[EvidenceReport],
                      summary="Release-readiness gate report")
async def evidence_endpoint() -> Envelope[EvidenceReport]:
    return Envelope(data=evidence_report(), meta=_meta())


# ── Jobs ───────────────────────────────────────────────────────────────────

jobs_router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@jobs_router.get("", response_model=Envelope[list[JobSummary]],
                  summary="List async jobs")
async def list_jobs_endpoint(
    state: str | None = Query(default=None,
                              description="queued|running|succeeded|failed|canceled"),
) -> Envelope[list[JobSummary]]:
    return Envelope(data=list_jobs(state=state), meta=_meta())


@jobs_router.get("/{job_id}", response_model=Envelope[JobSummary],
                  summary="Get one job by id")
async def get_job_endpoint(job_id: str) -> Envelope[JobSummary]:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    return Envelope(data=job, meta=_meta())


__all__ = [
    "bench_router",
    "configs_router",
    "doctor_router",
    "evidence_router",
    "jobs_router",
]
