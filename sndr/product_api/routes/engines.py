# SPDX-License-Identifier: Apache-2.0
"""HTTP routes for engine resources.

Endpoints:
    GET /api/v1/engines
        List every registered engine with summary info.

    GET /api/v1/engines/{engine}
        Get detailed information about one engine, including supported pins
        and patch counts.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from sndr.exceptions import EngineNotInstalledError, EngineUnsupportedError
from sndr.product_api.domain.engines_service import (
    get_engine_detail,
    list_engine_summaries,
)
from sndr.product_api.schemas.common import Envelope, ResponseMeta
from sndr.product_api.schemas.engines import EngineDetail, EngineSummary

router = APIRouter(prefix="/api/v1/engines", tags=["engines"])


def _meta(engine: str | None = None) -> ResponseMeta:
    """Build a ResponseMeta with a fresh request id."""
    return ResponseMeta(
        request_id=uuid4().hex,
        engine=engine,
        timestamp=datetime.now(timezone.utc),
    )


@router.get(
    "",
    response_model=Envelope[list[EngineSummary]],
    summary="List registered engines",
)
async def list_engines_endpoint() -> Envelope[list[EngineSummary]]:
    """Return summary of every engine that has a registered adapter.

    Engines without an installed package appear with ``active=false``.
    """
    summaries = list_engine_summaries()
    return Envelope(data=summaries, meta=_meta())


@router.get(
    "/{engine}",
    response_model=Envelope[EngineDetail],
    summary="Get engine detail",
)
async def get_engine_endpoint(engine: str) -> Envelope[EngineDetail]:
    """Return detailed info about one engine.

    Includes supported pin list, patch counts, capabilities.
    """
    try:
        detail = get_engine_detail(engine)
    except EngineUnsupportedError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except EngineNotInstalledError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    return Envelope(data=detail, meta=_meta(engine=engine))


__all__ = ["router"]
