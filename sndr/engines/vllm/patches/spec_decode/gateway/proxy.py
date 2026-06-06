# SPDX-License-Identifier: Apache-2.0
"""proxy — request forwarder with default-first fallback.

D2a:
  - non-streaming requests: full proxy to chosen upstream
  - streaming (stream=true): forced to DEFAULT upstream

D2b (this commit):
  - streaming requests route by the same router/artifact rules as
    non-streaming; SSE chunks are passed through with no buffering
  - structured upstream EXCEPTION/down still triggers fallback BEFORE
    the upstream connection is opened; once streaming starts, mid-
    stream errors are surfaced as-is (we cannot replay chunks the
    client already has)
  - upstream connection refused / timeout / 5xx → propagated to client
    (no retry); only PRE-stream structured-side failures trigger
    fallback to default
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from . import metrics
from .upstream import UpstreamState

log = logging.getLogger("genesis.spec_decode.gateway.proxy")


def _is_streaming(body: dict[str, Any]) -> bool:
    if not isinstance(body, dict):
        return False
    v = body.get("stream")
    return bool(v) if v is not None else False


def _decide_route(
    *,
    body: dict[str, Any],
    artifact,                 # FunctionalArtifact | None
    default_state: UpstreamState,
    structured_state: UpstreamState,
    force_default: bool,
) -> tuple[UpstreamState, str, str]:
    """Return (chosen_upstream, decision_label, reason).

    decision_label is one of:
      'structured', 'default', 'fallback_force', 'fallback_streaming',
      'fallback_no_artifact', 'fallback_router_error',
      'fallback_structured_down', 'fallback_router_denied'
    """
    # 1) Hard switch
    if force_default:
        return (default_state, "fallback_force",
                "admin force-default flag is active")

    # 2) Streaming: D2b routes streaming by the same rules as
    # non-streaming (no longer forced to default). Fall through to
    # the artifact + router checks.

    # 3) Artifact present?
    if artifact is None:
        return (default_state, "fallback_no_artifact",
                "no FunctionalArtifact loaded; router has no profile")

    # 4) Ask router
    try:
        from ..request_router import select_profile
        sel = select_profile(request=body, artifact=artifact)
    except Exception as e:  # noqa: BLE001
        log.warning("[gateway.proxy] router exception: %s", e)
        return (default_state, "fallback_router_error",
                f"router raised: {type(e).__name__}: {e}")

    # Update router decision metric
    try:
        metrics.ROUTER_DECISION.labels(
            profile=sel.profile, accepted=str(bool(sel.accepted)).lower(),
        ).inc()
    except Exception:
        pass

    if not sel.accepted:
        return (default_state, "fallback_router_denied",
                f"router denied: {sel.reason}")

    # 5) Structured health check
    if not structured_state.is_routable():
        return (default_state, "fallback_structured_down",
                f"structured upstream is {structured_state.state}: "
                f"{structured_state.last_error}")

    return (structured_state, "structured", sel.reason)


_HOP_BY_HOP_REQ_HEADERS = (
    "host", "content-length", "connection", "keep-alive",
    "transfer-encoding", "upgrade", "te", "trailer",
    "proxy-authenticate", "proxy-authorization",
)
_HOP_BY_HOP_RESP_HEADERS = (
    "content-length", "connection", "keep-alive",
    "transfer-encoding", "content-encoding",
)


def _strip_headers(headers: dict[str, str], drop: tuple[str, ...]) -> dict[str, str]:
    drop_lc = {d.lower() for d in drop}
    return {k: v for k, v in headers.items() if k.lower() not in drop_lc}


async def _open_upstream_stream(method: str, url: str, body_bytes: bytes,
                                headers: dict[str, str], timeout_s: float):
    """Open a streaming request to upstream.

    Returns (status, response_headers, async-byte-generator).
    The generator owns the lifecycle of the httpx client + response
    and closes both on completion / exception.
    """
    import httpx
    client = httpx.AsyncClient(timeout=timeout_s)
    try:
        req = client.build_request(
            method=method, url=url, content=body_bytes, headers=headers,
        )
        response = await client.send(req, stream=True)
    except Exception:
        await client.aclose()
        raise

    status = response.status_code
    resp_headers = dict(response.headers)

    async def _byte_stream():
        try:
            async for chunk in response.aiter_raw():
                yield chunk
        except Exception as e:  # noqa: BLE001
            log.warning(
                "[gateway.proxy] mid-stream upstream error: %s: %s",
                type(e).__name__, e,
            )
        finally:
            try:
                await response.aclose()
            except Exception:
                pass
            try:
                await client.aclose()
            except Exception:
                pass

    return status, resp_headers, _byte_stream()


async def proxy_request(
    *,
    method: str,
    path: str,
    headers: dict[str, str],
    body_bytes: bytes,
    body_json: dict[str, Any] | None,
    artifact,
    default_state: UpstreamState,
    structured_state: UpstreamState,
    force_default: bool,
    timeout_s: float = 120.0,
):
    """Forward a request to the chosen upstream and return a FastAPI
    Response (non-streaming) or StreamingResponse (stream=true).

    Routing decision is identical for streaming and non-streaming.
    """
    from fastapi import Response
    from fastapi.responses import StreamingResponse

    body = body_json if isinstance(body_json, dict) else {}
    is_streaming = _is_streaming(body)

    chosen, label, reason = _decide_route(
        body=body, artifact=artifact,
        default_state=default_state,
        structured_state=structured_state,
        force_default=force_default,
    )

    # The profile label for D2c metrics: artifact's profile when
    # accepted; else the fallback profile.
    profile_label = (
        artifact.profile if (label == "structured" and artifact is not None)
        else "gemma4-31b-tq-default"
    )

    log.info(
        "[gateway.proxy] decision=%s upstream=%s streaming=%s reason=%s",
        label, chosen.name, is_streaming, reason,
    )

    # Routing metrics
    try:
        if label == "structured":
            metrics.ROUTED_STRUCTURED.inc()
        else:
            metrics.ROUTED_DEFAULT.inc()
            if label.startswith("fallback_"):
                metrics.FALLBACK_TOTAL.labels(
                    reason=label[len("fallback_"):]).inc()
        if is_streaming:
            metrics.STREAMING_REQUEST_TOTAL.labels(
                upstream=chosen.name).inc()
    except Exception:
        pass

    url = chosen.base_url.rstrip("/") + path
    fwd_headers = _strip_headers(headers, _HOP_BY_HOP_REQ_HEADERS)

    t0 = time.perf_counter()

    # ---- streaming path -----------------------------------------
    if is_streaming:
        try:
            status, raw_resp_headers, body_gen = await _open_upstream_stream(
                method=method, url=url, body_bytes=body_bytes,
                headers=fwd_headers, timeout_s=timeout_s,
            )
        except Exception as e:  # noqa: BLE001
            latency = time.perf_counter() - t0
            try:
                metrics.UPSTREAM_ERROR.labels(
                    upstream=chosen.name, kind=type(e).__name__,
                ).inc()
                metrics.STREAMING_ERROR_TOTAL.labels(
                    upstream=chosen.name, reason="open_failed",
                ).inc()
                metrics.REQUEST_LATENCY.labels(
                    upstream=chosen.name).observe(latency)
                metrics.ROUTE_LATENCY.labels(
                    upstream=chosen.name, profile=profile_label,
                    stream="true",
                ).observe(latency)
            except Exception:
                pass
            log.warning(
                "[gateway.proxy] streaming open failed to %s: %s: %s",
                chosen.name, type(e).__name__, e,
            )
            return Response(
                content=json.dumps({
                    "error": {
                        "message": f"upstream {chosen.name} stream open failed",
                        "type": "upstream_error",
                        "code": "upstream_stream_failed",
                    }
                }).encode("utf-8"),
                status_code=502,
                media_type="application/json",
            )

        # Observe latency at "stream opened" time (TTFB-like)
        try:
            metrics.REQUEST_LATENCY.labels(
                upstream=chosen.name).observe(time.perf_counter() - t0)
            metrics.ROUTE_LATENCY.labels(
                upstream=chosen.name, profile=profile_label, stream="true",
            ).observe(time.perf_counter() - t0)
        except Exception:
            pass

        resp_headers = _strip_headers(
            raw_resp_headers, _HOP_BY_HOP_RESP_HEADERS)
        return StreamingResponse(
            body_gen,
            status_code=status,
            headers=resp_headers,
            media_type=raw_resp_headers.get(
                "content-type", "text/event-stream"),
        )

    # ---- non-streaming path -------------------------------------
    import httpx
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.request(
                method=method, url=url, content=body_bytes,
                headers=fwd_headers,
            )
        latency = time.perf_counter() - t0
        try:
            metrics.REQUEST_LATENCY.labels(
                upstream=chosen.name).observe(latency)
            metrics.ROUTE_LATENCY.labels(
                upstream=chosen.name, profile=profile_label,
                stream="false",
            ).observe(latency)
        except Exception:
            pass
        resp_headers = _strip_headers(
            dict(r.headers), _HOP_BY_HOP_RESP_HEADERS)
        return Response(
            content=r.content,
            status_code=r.status_code,
            headers=resp_headers,
            media_type=r.headers.get("content-type", "application/json"),
        )
    except Exception as e:  # noqa: BLE001
        latency = time.perf_counter() - t0
        try:
            metrics.UPSTREAM_ERROR.labels(
                upstream=chosen.name, kind=type(e).__name__,
            ).inc()
            metrics.REQUEST_LATENCY.labels(
                upstream=chosen.name).observe(latency)
            metrics.ROUTE_LATENCY.labels(
                upstream=chosen.name, profile=profile_label,
                stream="false",
            ).observe(latency)
        except Exception:
            pass
        log.warning(
            "[gateway.proxy] upstream error to %s: %s: %s",
            chosen.name, type(e).__name__, e,
        )
        return Response(
            content=json.dumps({
                "error": {
                    "message": f"upstream {chosen.name} unreachable",
                    "type": "upstream_error",
                    "code": "upstream_unreachable",
                }
            }).encode("utf-8"),
            status_code=502,
            media_type="application/json",
        )


__all__ = ["proxy_request", "_decide_route", "_is_streaming"]
