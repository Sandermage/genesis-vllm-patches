# SPDX-License-Identifier: Apache-2.0
"""HTTP routes for licensing resources."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter

from sndr.product_api.domain.license_status import get_license_status
from sndr.product_api.schemas.common import Envelope, ResponseMeta
from sndr.product_api.schemas.licensing import LicenseStatus

router = APIRouter(prefix="/api/v1/licensing", tags=["licensing"])


def _meta() -> ResponseMeta:
    return ResponseMeta(
        request_id=uuid4().hex,
        timestamp=datetime.now(timezone.utc),
    )


@router.get(
    "/status",
    response_model=Envelope[LicenseStatus],
    summary="Current license status",
)
async def licensing_status() -> Envelope[LicenseStatus]:
    """Return the current license verification status.

    NEVER returns the license token itself. Only metadata (status,
    expiry, engine_major, package availability) is exposed.
    """
    status = get_license_status()
    return Envelope(data=status, meta=_meta())


__all__ = ["router"]
