# SPDX-License-Identifier: Apache-2.0
"""app — FastAPI dispatcher entry point.

Run with:
    python -m vllm.sndr_core.integrations.spec_decode.gateway

Or imported and mounted into another app via ``create_app()``.

Env vars:
  GENESIS_GATEWAY_DEFAULT_URL      (default: http://localhost:8101)
  GENESIS_GATEWAY_STRUCTURED_URL   (default: http://localhost:8102)
  GENESIS_GATEWAY_PROFILE          (default: gemma4-31b-tq-mtp-structured-k4)
  GENESIS_GATEWAY_BIND_HOST        (default: 0.0.0.0)
  GENESIS_GATEWAY_BIND_PORT        (default: 8100)
  GENESIS_GATEWAY_HEALTH_INTERVAL  (seconds; default: 5)
  GENESIS_GATEWAY_TIMEOUT          (upstream timeout seconds; default: 120)
  GENESIS_GATEWAY_ADMIN_ALLOW_REMOTE  (default: off; 1 to allow non-local admin)
  GENESIS_SPEC_DECODE_ARTIFACTS_DIR   (extra artifacts dir; optional)

D2a scope:
  - non-streaming proxy for /v1/chat/completions
  - stream=true -> default upstream (no structured streaming yet)
  - default-first on any uncertainty
  - admin endpoints localhost-only
  - day-1 prometheus metrics
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request, Response

from ....env import get_sndr_env, get_sndr_env_float, get_sndr_env_int
from .upstream import UpstreamState, run_health_loop

log = logging.getLogger("genesis.spec_decode.gateway.app")
logging.basicConfig(
    level=(get_sndr_env("GATEWAY_LOG_LEVEL", "INFO") or "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)


@dataclass
class GatewayState:
    default_state: UpstreamState
    structured_state: UpstreamState
    artifact: Any = None
    force_default: bool = False
    profile_name: str = ""
    timeout_s: float = 120.0

    def reload_artifact(self) -> tuple[bool, str]:
        """Re-read artifact JSON. Returns (ok, message)."""
        try:
            from ..functional_artifact import _ARTIFACTS_DIR, read
            path = _ARTIFACTS_DIR / f"{self.profile_name}.json"
            if not path.exists():
                self.artifact = None
                return (False,
                        f"no artifact at {path} for profile "
                        f"{self.profile_name!r}; routing will fall back")
            self.artifact = read(path)
            return (True,
                    f"loaded artifact profile={self.artifact.profile!r} "
                    f"config_hash={self.artifact.config_hash!r} "
                    f"decision={self.artifact.decision!r}")
        except Exception as e:  # noqa: BLE001
            self.artifact = None
            return (False, f"{type(e).__name__}: {e}")


def _build_state() -> GatewayState:
    default_url = get_sndr_env(
        "GATEWAY_DEFAULT_URL", "http://localhost:8101")
    structured_url = get_sndr_env(
        "GATEWAY_STRUCTURED_URL", "http://localhost:8102")
    profile = get_sndr_env(
        "GATEWAY_PROFILE", "gemma4-31b-tq-mtp-structured-k4")
    timeout_s = get_sndr_env_float("GATEWAY_TIMEOUT", 120.0)

    state = GatewayState(
        default_state=UpstreamState(name="default", base_url=default_url),
        structured_state=UpstreamState(name="structured",
                                       base_url=structured_url),
        profile_name=profile,
        timeout_s=timeout_s,
    )
    ok, msg = state.reload_artifact()
    log.info("[gateway] artifact load on startup: ok=%s msg=%s", ok, msg)
    return state


def create_app(state: GatewayState | None = None):
    """Build the FastAPI app."""
    state = state or _build_state()
    app = FastAPI(title="genesis-spec-decode-gateway", version="0.1.0")
    app.state.gateway = state

    # ---- public endpoints ----

    @app.get("/healthz")
    async def _healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def _readyz(response: Response) -> dict[str, Any]:
        if not state.default_state.is_routable():
            response.status_code = 503
            return {
                "status": "not_ready",
                "default": state.default_state.state,
                "structured": state.structured_state.state,
            }
        return {
            "status": "ready",
            "default": state.default_state.state,
            "structured": state.structured_state.state,
        }

    @app.get("/metrics")
    async def _metrics(response: Response):
        from . import metrics as _m
        body, ctype = _m.render()
        return Response(content=body, media_type=ctype)

    @app.get("/v1/models")
    async def _models(request: Request):
        # Passthrough from default upstream (canonical source).
        from .proxy import proxy_request
        return await proxy_request(
            method="GET", path="/v1/models",
            headers=dict(request.headers),
            body_bytes=b"", body_json={},
            artifact=state.artifact,
            default_state=state.default_state,
            structured_state=state.structured_state,
            force_default=True,  # always go default for model list
            timeout_s=state.timeout_s,
        )

    @app.post("/v1/chat/completions")
    async def _chat(request: Request):
        from .proxy import proxy_request
        body_bytes = await request.body()
        try:
            body_json = json.loads(body_bytes) if body_bytes else {}
        except json.JSONDecodeError as e:
            return Response(
                content=json.dumps({"error": {
                    "message": f"invalid JSON: {e}",
                    "type": "invalid_request_error",
                    "code": "json_decode_error",
                }}).encode("utf-8"),
                status_code=400,
                media_type="application/json",
            )
        return await proxy_request(
            method="POST", path="/v1/chat/completions",
            headers=dict(request.headers),
            body_bytes=body_bytes, body_json=body_json,
            artifact=state.artifact,
            default_state=state.default_state,
            structured_state=state.structured_state,
            force_default=state.force_default,
            timeout_s=state.timeout_s,
        )

    @app.post("/v1/completions")
    async def _completions(request: Request):
        # Same router applies (response_format/tool_choice rare but handled)
        from .proxy import proxy_request
        body_bytes = await request.body()
        try:
            body_json = json.loads(body_bytes) if body_bytes else {}
        except json.JSONDecodeError as e:
            return Response(
                content=json.dumps({"error": {
                    "message": f"invalid JSON: {e}",
                    "type": "invalid_request_error",
                    "code": "json_decode_error",
                }}).encode("utf-8"),
                status_code=400,
                media_type="application/json",
            )
        return await proxy_request(
            method="POST", path="/v1/completions",
            headers=dict(request.headers),
            body_bytes=body_bytes, body_json=body_json,
            artifact=state.artifact,
            default_state=state.default_state,
            structured_state=state.structured_state,
            force_default=state.force_default,
            timeout_s=state.timeout_s,
        )

    # ---- admin endpoints (localhost-only) ----
    from .admin import install_admin_routes
    install_admin_routes(app, state)

    # ---- startup tasks ----

    @app.on_event("startup")
    async def _startup() -> None:
        interval = get_sndr_env_float("GATEWAY_HEALTH_INTERVAL", 5.0)
        app.state.health_task = asyncio.create_task(
            run_health_loop(
                [state.default_state, state.structured_state],
                interval_s=interval,
            )
        )
        # SIGHUP -> reload artifact (POSIX only)
        try:
            loop = asyncio.get_event_loop()
            loop.add_signal_handler(
                signal.SIGHUP,
                lambda: log.warning(
                    "[gateway] SIGHUP — artifact reload: %s",
                    state.reload_artifact(),
                ),
            )
        except (NotImplementedError, AttributeError):
            log.info("[gateway] SIGHUP handler not available on this platform")

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        t = getattr(app.state, "health_task", None)
        if t is not None:
            t.cancel()

    return app


def main() -> None:
    import uvicorn
    host = get_sndr_env("GATEWAY_BIND_HOST", "0.0.0.0")
    port = get_sndr_env_int("GATEWAY_BIND_PORT", 8100)
    log.info("[gateway] starting on %s:%s", host, port)
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()


__all__ = ["GatewayState", "create_app", "main"]
