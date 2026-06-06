# SPDX-License-Identifier: Apache-2.0
"""HTTP routes for pin resources."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from sndr.exceptions import PinManifestMissingError
from sndr.product_api.domain.pins_service import (
    get_pin_manifest_summary,
    list_pins,
)
from sndr.product_api.schemas.common import Envelope, ResponseMeta
from sndr.product_api.schemas.pins import PinManifestSummary, PinSummary

router = APIRouter(prefix="/api/v1/engines", tags=["pins"])


def _meta(engine: str | None = None, pin: str | None = None) -> ResponseMeta:
    return ResponseMeta(
        request_id=uuid4().hex,
        engine=engine,
        pin=pin,
        timestamp=datetime.now(timezone.utc),
    )


@router.get(
    "/{engine}/pins",
    response_model=Envelope[list[PinSummary]],
    summary="List pins for engine",
)
async def list_pins_endpoint(engine: str) -> Envelope[list[PinSummary]]:
    """Return all pins with manifests for the given engine."""
    pins = list_pins(engine)
    return Envelope(data=pins, meta=_meta(engine=engine))


@router.get(
    "/{engine}/pins/{pin}",
    response_model=Envelope[PinManifestSummary],
    summary="Get pin manifest summary",
)
async def get_pin_endpoint(engine: str, pin: str) -> Envelope[PinManifestSummary]:
    """Return summary of one pin's manifest (file count, anchor count, ...)."""
    try:
        summary = get_pin_manifest_summary(engine, pin)
    except PinManifestMissingError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    return Envelope(data=summary, meta=_meta(engine=engine, pin=pin))


__all__ = ["router"]
