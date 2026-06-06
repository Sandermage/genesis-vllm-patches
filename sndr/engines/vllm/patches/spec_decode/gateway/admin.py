# SPDX-License-Identifier: Apache-2.0
"""admin — localhost-only operator controls.

Endpoints (mounted at /admin):
  POST /admin/force-default        set force-default flag
  POST /admin/clear-force-default  clear force-default flag
  POST /admin/reload-artifacts     re-read artifact JSON from disk
  GET  /admin/state                full state dump

All admin endpoints reject requests whose client IP is not localhost
(127.0.0.1 / ::1) unless GENESIS_GATEWAY_ADMIN_ALLOW_REMOTE=1 is set.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import HTTPException, Request

log = logging.getLogger("genesis.spec_decode.gateway.admin")


_LOCAL_IPS = ("127.0.0.1", "::1", "localhost")


def _client_is_local(request) -> bool:
    from ....env import get_sndr_env_bool
    if get_sndr_env_bool("GATEWAY_ADMIN_ALLOW_REMOTE"):
        return True
    try:
        host = request.client.host if request.client else ""
        return host in _LOCAL_IPS or host.startswith("127.") or host == "::1"
    except Exception:
        return False


def install_admin_routes(app, state) -> None:
    """Mount admin endpoints onto the FastAPI app.

    ``state`` is the GatewayState (see app.py): has fields
    .force_default (bool), .default_state, .structured_state,
    .artifact (FunctionalArtifact | None), .reload_artifact() method.
    """
    def _require_local(request: Request) -> None:
        if not _client_is_local(request):
            raise HTTPException(
                status_code=403,
                detail=(
                    "admin endpoints are localhost-only. Set "
                    "GENESIS_GATEWAY_ADMIN_ALLOW_REMOTE=1 to override."
                ),
            )

    @app.post("/admin/force-default")
    async def _force_default(request: Request) -> dict[str, Any]:
        _require_local(request)
        state.force_default = True
        try:
            from . import metrics
            metrics.FORCE_DEFAULT_ACTIVE.set(1)
        except Exception:
            pass
        log.warning("[gateway.admin] force-default ACTIVATED")
        return {"force_default": True}

    @app.post("/admin/clear-force-default")
    async def _clear_force_default(request: Request) -> dict[str, Any]:
        _require_local(request)
        state.force_default = False
        try:
            from . import metrics
            metrics.FORCE_DEFAULT_ACTIVE.set(0)
        except Exception:
            pass
        log.warning("[gateway.admin] force-default CLEARED")
        return {"force_default": False}

    @app.post("/admin/reload-artifacts")
    async def _reload_artifacts(request: Request) -> dict[str, Any]:
        _require_local(request)
        ok, msg = state.reload_artifact()
        log.warning("[gateway.admin] reload-artifacts: ok=%s msg=%s",
                    ok, msg)
        return {"ok": ok, "message": msg,
                "artifact_loaded": state.artifact is not None,
                "artifact_profile": (
                    state.artifact.profile if state.artifact else None)}

    @app.get("/admin/state")
    async def _state(request: Request) -> dict[str, Any]:
        _require_local(request)
        return {
            "force_default": state.force_default,
            "upstreams": {
                "default": state.default_state.to_dict(),
                "structured": state.structured_state.to_dict(),
            },
            "artifact": (
                {
                    "profile": state.artifact.profile,
                    "config_hash": state.artifact.config_hash,
                    "decision": state.artifact.decision,
                    "allowed_workloads": state.artifact.allowed_workloads,
                    "denied_workloads": state.artifact.denied_workloads,
                    "vllm_pin": state.artifact.vllm_pin,
                    "created_at": state.artifact.created_at,
                } if state.artifact else None
            ),
        }


__all__ = ["install_admin_routes"]
