# SPDX-License-Identifier: Apache-2.0
"""HTTP routes for host inventory."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from sndr.product_api.domain.hosts_service import (
    fleet_report,
    get_local_host,
    list_hosts,
)
from sndr.product_api.schemas.common import Envelope, ResponseMeta
from sndr.product_api.schemas.hosts import FleetReport, HostSummary

router = APIRouter(prefix="/api/v1/hosts", tags=["hosts"])


def _meta() -> ResponseMeta:
    return ResponseMeta(
        request_id=uuid4().hex,
        engine=None,
        pin=None,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("", response_model=Envelope[list[HostSummary]],
            summary="List hosts (local + fleet registry)")
async def list_hosts_endpoint() -> Envelope[list[HostSummary]]:
    return Envelope(data=list_hosts(), meta=_meta())


@router.get("/local", response_model=Envelope[HostSummary],
            summary="Get the local host")
async def get_local_host_endpoint() -> Envelope[HostSummary]:
    return Envelope(data=get_local_host(), meta=_meta())


@router.get("/{hostname}", response_model=Envelope[HostSummary],
            summary="Get one host by hostname")
async def get_host_endpoint(hostname: str) -> Envelope[HostSummary]:
    for h in list_hosts():
        if h.hostname == hostname:
            return Envelope(data=h, meta=_meta())
    raise HTTPException(status_code=404, detail=f"host not found: {hostname}")


fleet_router = APIRouter(prefix="/api/v1/fleet", tags=["fleet"])


@fleet_router.get("", response_model=Envelope[FleetReport],
                  summary="Aggregate fleet report")
async def get_fleet_endpoint() -> Envelope[FleetReport]:
    return Envelope(data=fleet_report(), meta=_meta())


__all__ = ["fleet_router", "router"]
