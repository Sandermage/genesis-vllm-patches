# SPDX-License-Identifier: Apache-2.0
"""HTTP routes for observability surfaces.

Combines bench, doctor, configs, evidence, and jobs into one module so
the GUI's monitoring tabs all share a single set of FastAPI routers.

``doctor_report`` shells out to ``nvidia-smi`` + ``docker info`` via
blocking ``subprocess.run`` (each up to 5 s), so the doctor handler
offloads it with ``await asyncio.to_thread(...)`` to keep the event loop
free. The other surfaces here (bench/configs/evidence) read files and do
not block, so they stay direct.

Migration status — jobs
=======================
The modular ``/api/v1/jobs`` routes are intentionally NOT wired. Their
backing store (``observability_service._JOBS``) has no producer: the only
writer, ``register_job()``, has zero callers. The live ``sndr gui-api``
daemon's real job system is the separate legacy
``sndr.product_api.legacy.jobs`` module (job persistence + service-apply).
Rather than serve a misleading always-empty ``200``, the modular jobs
handlers return ``501 Not Implemented`` until the legacy job system is
migrated here (Phase 11). Do not add a fake producer — wire the real one.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query

from sndr.product_api.domain.observability_service import (
    config_catalog,
    doctor_report,
    evidence_report,
    list_bench_runs,
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
    data = await asyncio.to_thread(doctor_report)
    return Envelope(data=data, meta=_meta())


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
#
# Intentionally unimplemented in the modular server (see the module
# docstring). The handlers return 501 rather than the former lying
# 200+empty-list, because the modular jobs store has no producer and the
# live daemon uses the separate legacy job system.

jobs_router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])

_JOBS_NOT_WIRED = (
    "modular jobs API not wired — the live server uses the legacy job "
    "system (sndr.product_api.legacy.jobs). This surface lands in Phase 11."
)


@jobs_router.get("", response_model=Envelope[list[JobSummary]],
                  summary="List async jobs (not implemented)")
async def list_jobs_endpoint(
    state: str | None = Query(default=None,
                              description="queued|running|succeeded|failed|canceled"),
) -> Envelope[list[JobSummary]]:
    raise HTTPException(status_code=501, detail=_JOBS_NOT_WIRED)


@jobs_router.get("/{job_id}", response_model=Envelope[JobSummary],
                  summary="Get one job by id (not implemented)")
async def get_job_endpoint(job_id: str) -> Envelope[JobSummary]:
    raise HTTPException(status_code=501, detail=_JOBS_NOT_WIRED)


__all__ = [
    "bench_router",
    "configs_router",
    "doctor_router",
    "evidence_router",
    "jobs_router",
]
