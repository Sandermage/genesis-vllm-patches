# SPDX-License-Identifier: Apache-2.0
"""HTTP routes for patch resources.

Endpoints:
    GET /api/v1/patches                    — paginated patch inventory
    GET /api/v1/patches/inventory          — aggregate counts (cheap)
    GET /api/v1/patches/{patch_id}         — full detail for one patch
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query

from sndr.product_api.domain.patches_service import (
    get_patch,
    inventory_report,
    list_patches,
)
from sndr.product_api.schemas.common import Envelope, ResponseMeta
from sndr.product_api.schemas.patches import (
    PatchDetail,
    PatchInventoryReport,
    PatchSummary,
)

router = APIRouter(prefix="/api/v1/patches", tags=["patches"])


def _meta() -> ResponseMeta:
    return ResponseMeta(
        request_id=uuid4().hex,
        engine=None,
        pin=None,
        timestamp=datetime.now(timezone.utc),
    )


@router.get(
    "",
    response_model=Envelope[list[PatchSummary]],
    summary="List patches with optional filters",
)
async def list_patches_endpoint(
    family: str | None = Query(default=None, description="Filter by subsystem family"),
    tier: str | None = Query(default=None, description="community | engine"),
    lifecycle: str | None = Query(
        default=None,
        description="experimental | active | deprecated | retired",
    ),
    enabled_only: bool = Query(
        default=False,
        description="Only patches enabled in the live environment",
    ),
) -> Envelope[list[PatchSummary]]:
    """Return a filtered patch inventory."""
    patches = list_patches(
        family=family,
        tier=tier,
        lifecycle=lifecycle,
        enabled_only=enabled_only,
    )
    return Envelope(data=patches, meta=_meta())


@router.get(
    "/inventory",
    response_model=Envelope[PatchInventoryReport],
    summary="Aggregate patch inventory report",
)
async def inventory_endpoint() -> Envelope[PatchInventoryReport]:
    """Return aggregate counts (total, by family, by lifecycle, by tier)."""
    return Envelope(data=inventory_report(), meta=_meta())


@router.get(
    "/{patch_id}",
    response_model=Envelope[PatchDetail],
    summary="Get patch detail by id",
)
async def get_patch_endpoint(patch_id: str) -> Envelope[PatchDetail]:
    """Return full metadata for one patch."""
    detail = get_patch(patch_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"patch not found: {patch_id}")
    return Envelope(data=detail, meta=_meta())


__all__ = ["router"]
