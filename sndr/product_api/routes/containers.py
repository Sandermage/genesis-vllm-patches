# SPDX-License-Identifier: Apache-2.0
"""HTTP routes for container inventory.

The container-service functions shell out to ``docker ps`` / ``docker
inspect`` / ``docker logs`` via blocking ``subprocess.run`` (each up to a
10 s timeout). Calling them directly from an ``async def`` handler would
block the event loop for the whole subprocess, stalling every concurrent
request on the worker. Each blocking call is therefore offloaded with
``await asyncio.to_thread(...)`` — the same convention the legacy
``http_app`` uses for its blocking work (preflight, log streaming, SSH).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query

from sndr.product_api.domain.containers_service import (
    get_container_detail,
    inventory_report,
    list_containers,
)
from sndr.product_api.schemas.common import Envelope, ResponseMeta
from sndr.product_api.schemas.containers import (
    ContainerDetail,
    ContainerInventoryReport,
    ContainerSummary,
)

router = APIRouter(prefix="/api/v1/containers", tags=["containers"])


def _meta() -> ResponseMeta:
    return ResponseMeta(
        request_id=uuid4().hex,
        engine=None,
        pin=None,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("", response_model=Envelope[list[ContainerSummary]],
            summary="List containers")
async def list_containers_endpoint(
    engine: str | None = Query(default=None,
                                description="Filter by engine (e.g. vllm, sglang)"),
) -> Envelope[list[ContainerSummary]]:
    data = await asyncio.to_thread(list_containers, engine=engine)
    return Envelope(data=data, meta=_meta())


@router.get("/inventory", response_model=Envelope[ContainerInventoryReport],
            summary="Aggregate container inventory")
async def inventory_endpoint() -> Envelope[ContainerInventoryReport]:
    data = await asyncio.to_thread(inventory_report)
    return Envelope(data=data, meta=_meta())


@router.get("/{name}", response_model=Envelope[ContainerDetail],
            summary="Get container detail by name")
async def get_container_endpoint(name: str) -> Envelope[ContainerDetail]:
    detail = await asyncio.to_thread(get_container_detail, name)
    if detail is None:
        raise HTTPException(status_code=404,
                            detail=f"container not found: {name}")
    return Envelope(data=detail, meta=_meta())


__all__ = ["router"]
