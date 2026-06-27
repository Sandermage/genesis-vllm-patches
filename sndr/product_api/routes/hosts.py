# SPDX-License-Identifier: Apache-2.0
"""HTTP routes for host inventory.

The host-service functions probe the host with blocking ``subprocess.run``
calls (``nvidia-smi`` ×3 + ``docker --version``). Offload them with
``await asyncio.to_thread(...)`` so a slow probe cannot stall the event
loop and every concurrent request behind it.
"""
from __future__ import annotations

import asyncio
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
    data = await asyncio.to_thread(list_hosts)
    return Envelope(data=data, meta=_meta())


@router.get("/local", response_model=Envelope[HostSummary],
            summary="Get the local host")
async def get_local_host_endpoint() -> Envelope[HostSummary]:
    data = await asyncio.to_thread(get_local_host)
    return Envelope(data=data, meta=_meta())


@router.get("/{hostname}", response_model=Envelope[HostSummary],
            summary="Get one host by hostname")
async def get_host_endpoint(hostname: str) -> Envelope[HostSummary]:
    hosts = await asyncio.to_thread(list_hosts)
    for h in hosts:
        if h.hostname == hostname:
            return Envelope(data=h, meta=_meta())
    raise HTTPException(status_code=404, detail=f"host not found: {hostname}")


fleet_router = APIRouter(prefix="/api/v1/fleet", tags=["fleet"])


@fleet_router.get("", response_model=Envelope[FleetReport],
                  summary="Aggregate fleet report")
async def get_fleet_endpoint() -> Envelope[FleetReport]:
    data = await asyncio.to_thread(fleet_report)
    return Envelope(data=data, meta=_meta())


__all__ = ["fleet_router", "router"]
