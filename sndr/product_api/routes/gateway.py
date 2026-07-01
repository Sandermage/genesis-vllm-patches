# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible memory gateway (`POST /v1/chat/completions`).

Clients point here instead of directly at the upstream; the gateway transparently
adds memory to EVERY model (the единое-ядро layer):

    client -> /v1/chat/completions -> recall+inject -> upstream -> capture -> client

Upstream is set by env (the unmodified CLIProxyAPI for external models, or the
vLLM OpenAI server for the internal 35B):
  * GATEWAY_UPSTREAM_URL  e.g. http://cliproxy:8317/v1  (or http://127.0.0.1:8102/v1)
  * GATEWAY_UPSTREAM_KEY  bearer token the upstream expects (optional)

The actual upstream call lives on `app.state.gateway_forward` / `gateway_stream`
(set by create_app from env, overridable in tests), so this route is verifiable
without a live upstream. Owner scoping via the X-Owner-Id header.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from sndr.memory.gateway import assistant_text_from_sse, extract_assistant_text
from sndr.memory.middleware import ConversationMemory
from sndr.product_api.security import owner_from_request

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

log = logging.getLogger("sndr.product_api.gateway")
router = APIRouter(tags=["gateway"])



def _memory(request: Request) -> ConversationMemory:
    engine = getattr(request.app.state, "memory_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="memory engine not configured")
    return ConversationMemory(engine=engine)


def _resolve_upstream(request: Request) -> dict[str, Any]:
    """Pick the named upstream for this request: X-Memory-Upstream header, else
    the configured default. 503 if none configured, 400 if the name is unknown."""
    upstreams: dict[str, Any] = getattr(request.app.state, "gateway_upstreams", None) or {}
    if not upstreams:
        raise HTTPException(status_code=503, detail="gateway upstream not configured")
    name = request.headers.get("X-Memory-Upstream") or getattr(
        request.app.state, "gateway_default", None
    )
    if name is None:
        name = next(iter(upstreams))
    up = upstreams.get(name)
    if up is None:
        raise HTTPException(status_code=400, detail=f"unknown upstream: {name}")
    return up


@router.get("/v1/upstreams", summary="List configured gateway upstreams (for the GUI)")
async def list_upstreams(request: Request) -> Any:
    upstreams: dict[str, Any] = getattr(request.app.state, "gateway_upstreams", None) or {}
    return JSONResponse({
        "upstreams": sorted(upstreams),
        "default": getattr(request.app.state, "gateway_default", None),
    })


def _capture_best_effort(mem: ConversationMemory, *, owner: int, messages: list, assistant: str) -> None:
    """Persisting the turn is best-effort: a store/embedder failure (e.g. a DB
    blip) must NEVER destroy a successful upstream reply the operator paid for."""
    try:
        mem.capture(owner_id=owner, messages=messages, assistant=assistant)
    except Exception:  # noqa: BLE001
        log.warning("gateway memory capture failed (reply still delivered)", exc_info=True)


@router.post("/v1/chat/completions", summary="Memory-augmented OpenAI chat completions")
async def chat_completions(request: Request) -> Any:
    upstream = _resolve_upstream(request)
    forward = upstream["forward"]
    stream_fn = upstream.get("stream")

    body: dict[str, Any] = await request.json()
    owner = owner_from_request(request)
    mem = _memory(request)
    original = list(body.get("messages") or [])
    # augment = embed + recall — synchronous CPU/DB work; to_thread keeps the
    # event loop (which serves the whole unified daemon) responsive.
    augmented = await asyncio.to_thread(mem.augment, owner_id=owner, messages=original)
    body = {**body, "messages": augmented}

    if body.get("stream") and stream_fn is not None:
        async def tee() -> AsyncIterator[bytes]:
            buf: list[str] = []
            errored = False
            try:
                async for chunk in stream_fn(body):
                    buf.append(chunk.decode("utf-8", "ignore"))
                    yield chunk
            except Exception:  # noqa: BLE001 - upstream failed mid-stream
                errored = True
                log.warning("gateway stream upstream error", exc_info=True)
            # A truncated turn must not become a permanent "memory" that gets
            # re-injected into future prompts — capture only clean completions.
            # (A client disconnect raises GeneratorExit at the yield, which
            # skips this entirely — also correct: the turn didn't complete.)
            if errored:
                return
            assistant = assistant_text_from_sse("".join(buf))
            if assistant:
                await asyncio.to_thread(
                    _capture_best_effort, mem, owner=owner, messages=original, assistant=assistant
                )

        return StreamingResponse(tee(), media_type="text/event-stream")

    try:
        response = await forward(body)
    except Exception as exc:  # noqa: BLE001 - the gateway wraps an upstream
        # Map any upstream failure to a clean 502/504 instead of a raw 500, and
        # do NOT capture a failed turn. Propagate the upstream's own status/body
        # when present — an opaque "HTTPStatusError" hid legitimate 400s like
        # "model not found" behind a detail-free 502.
        status = 504 if "timeout" in type(exc).__name__.lower() else 502
        detail = f"upstream error: {type(exc).__name__}"
        up_resp = getattr(exc, "response", None)
        if up_resp is not None:
            up_status = getattr(up_resp, "status_code", None)
            up_text = ""
            with contextlib.suppress(Exception):
                up_text = (up_resp.text or "")[:300]
            detail = f"upstream {up_status}: {up_text}" if up_status else detail
        raise HTTPException(status_code=status, detail=detail) from exc
    assistant = extract_assistant_text(response)
    await asyncio.to_thread(
        _capture_best_effort, mem, owner=owner, messages=original, assistant=assistant
    )
    return JSONResponse(response)


__all__ = ["router"]
