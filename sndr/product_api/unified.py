# SPDX-License-Identifier: Apache-2.0
"""Unified daemon: the full Control Center + persistent memory in one app.

During the v11→v12 migration the platform routes still live in the legacy
monolith (:mod:`sndr.product_api.legacy.http_app`, ~197 routes) while the
persistent-memory subsystem lives in the modular routers. Neither app alone is a
superset of what the GUI calls, so a single-daemon deployment 404s on one half.

This factory COMPOSES them — it builds the unchanged legacy app and mounts the
memory + gateway routers onto it — giving the "main Control Center daemon WITH
memory" the GUI needs, without editing the monolith (so it neither inherits the
legacy lint backlog nor risks the 197 working routes). Run it with::

    uvicorn sndr.product_api.unified:create_app --factory --host 0.0.0.0 --port 8800

Phase 11 proper will migrate the legacy routes into the modular ``server``; until
then this composition is the superset daemon.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

_SPA_MOUNT_NAME = "ui"  # the legacy factory's static-SPA catch-all (app.mount("/", ..., name="ui"))


def create_app() -> FastAPI:
    """Build the legacy Control Center app and mount the memory subsystem onto it.

    The legacy factory mounts the SPA catch-all at ``/`` LAST, and Starlette
    matches routes in registration order, so that Mount would shadow any API
    route added after it. We detach the SPA (absent without a built GUI), mount
    the memory + gateway API routes, then re-attach the SPA — so
    ``/api/v1/memory/*`` and ``/v1/*`` resolve before the static fallback.
    """
    from sndr.product_api.legacy.http_app import create_app as legacy_create_app
    from sndr.product_api.memory_wiring import mount_memory_routes

    app = legacy_create_app()

    routes = app.router.routes
    spa = None
    for i in range(len(routes) - 1, -1, -1):
        if getattr(routes[i], "name", None) == _SPA_MOUNT_NAME:
            spa = routes.pop(i)
            break

    mount_memory_routes(app)

    if spa is not None:
        routes.append(spa)
    return app


__all__ = ["create_app"]
