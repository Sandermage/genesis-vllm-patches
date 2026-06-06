# SPDX-License-Identifier: Apache-2.0
"""HTTP routes for container inventory."""
from __future__ import annotations

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
    return Envelope(data=list_containers(engine=engine), meta=_meta())


@router.get("/inventory", response_model=Envelope[ContainerInventoryReport],
            summary="Aggregate container inventory")
async def inventory_endpoint() -> Envelope[ContainerInventoryReport]:
    return Envelope(data=inventory_report(), meta=_meta())


@router.get("/{name}", response_model=Envelope[ContainerDetail],
            summary="Get container detail by name")
async def get_container_endpoint(name: str) -> Envelope[ContainerDetail]:
    detail = get_container_detail(name)
    if detail is None:
        raise HTTPException(status_code=404,
                            detail=f"container not found: {name}")
    return Envelope(data=detail, meta=_meta())


__all__ = ["router"]
