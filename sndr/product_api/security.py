# SPDX-License-Identifier: Apache-2.0
"""API-key guard for the memory + gateway routes (review finding C3).

The modular product-API has no session auth, so these routes derive the owner
from a client header — which means without a guard ANY caller could read/write
any owner's memory and drive the upstream LLM. This dependency closes that:

  * GENESIS_MEMORY_API_KEY unset  -> open (localhost / single-user dev)
  * GENESIS_MEMORY_API_KEY set     -> every guarded route requires a matching
                                      `Authorization: Bearer <key>` or `X-Api-Key`

Read at request time (not import time) so it is reconfigurable/testable.
Constant-time comparison; no key value is logged.
"""
from __future__ import annotations

import hmac
import os

from fastapi import HTTPException, Request


def _presented(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[len("Bearer "):].strip()
    return request.headers.get("X-Api-Key", "").strip()


def require_api_key(request: Request) -> None:
    """FastAPI dependency: enforce the API key when one is configured."""
    expected = os.environ.get("GENESIS_MEMORY_API_KEY", "").strip()
    if not expected:
        return  # auth disabled
    presented = _presented(request)
    if not presented or not hmac.compare_digest(presented, expected):
        raise HTTPException(status_code=401, detail="missing or invalid API key")


def owner_from_request(request: Request) -> int:
    """The owner scope for a request, from the `X-Owner-Id` header (default 1).
    Single source shared by the memory + gateway routes."""
    raw = request.headers.get("X-Owner-Id", "1")
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="invalid X-Owner-Id") from None
