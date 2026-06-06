# SPDX-License-Identifier: Apache-2.0
"""Health-check and version endpoints (always available, no auth required)."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter

from sndr.product_api.schemas.common import Envelope, ResponseMeta
from sndr.version import __version__

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", summary="Liveness probe")
async def health() -> Envelope[dict[str, str]]:
    """Always returns ``{"status": "ok"}``. Used by load balancers and
    container orchestrators to verify the process is alive.
    """
    return Envelope(
        data={"status": "ok"},
        meta=ResponseMeta(
            request_id=uuid4().hex,
            timestamp=datetime.now(timezone.utc),
        ),
    )


@router.get("/version", summary="Build version")
async def version_endpoint() -> Envelope[dict[str, str]]:
    """Return the running sndr-platform version and build info."""
    from sndr.version import __commit__
    return Envelope(
        data={"version": __version__, "commit": __commit__},
        meta=ResponseMeta(
            request_id=uuid4().hex,
            timestamp=datetime.now(timezone.utc),
        ),
    )


__all__ = ["router"]
