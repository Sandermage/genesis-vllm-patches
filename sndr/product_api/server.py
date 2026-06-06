# SPDX-License-Identifier: Apache-2.0
"""FastAPI application factory.

Usage::

    from sndr.product_api.server import create_app
    app = create_app()

Or with uvicorn::

    uvicorn sndr.product_api.server:create_app --factory --host 0.0.0.0 --port 8800

In v12.x, this new server runs alongside the legacy
``vllm.sndr_core.product_api`` server. New engine-aware routes
(``/api/v1/engines``, ``/api/v1/health``, ``/api/v1/version``) are served
here; legacy routes continue to come from the old monolith.

Phase 11 will fully migrate the legacy routes here.
"""
from __future__ import annotations

import logging

log = logging.getLogger("sndr.product_api.server")


def create_app() -> "FastAPI":  # type: ignore[name-defined]
    """Build a FastAPI application with all sndr routers mounted.

    FastAPI is imported lazily so that ``import sndr.product_api`` does not
    pull in the dependency unless the server is actually started.
    """
    from fastapi import FastAPI

    from sndr.version import __version__

    app = FastAPI(
        title="sndr-platform Control Center",
        description=(
            "Multi-engine inference patch orchestration platform. "
            "See https://github.com/sandermage/sndr-platform."
        ),
        version=__version__,
    )

    # Register all routers. Each route file declares its own prefix.
    from sndr.product_api.routes.containers import router as containers_router
    from sndr.product_api.routes.engines import router as engines_router
    from sndr.product_api.routes.health import router as health_router
    from sndr.product_api.routes.hosts import fleet_router, router as hosts_router
    from sndr.product_api.routes.licensing import router as licensing_router
    from sndr.product_api.routes.observability import (
        bench_router,
        configs_router,
        doctor_router,
        evidence_router,
        jobs_router,
    )
    from sndr.product_api.routes.patches import router as patches_router
    from sndr.product_api.routes.pins import router as pins_router

    app.include_router(health_router)
    app.include_router(engines_router)
    app.include_router(pins_router)
    app.include_router(patches_router)
    app.include_router(licensing_router)
    app.include_router(hosts_router)
    app.include_router(fleet_router)
    app.include_router(containers_router)
    app.include_router(bench_router)
    app.include_router(doctor_router)
    app.include_router(configs_router)
    app.include_router(evidence_router)
    app.include_router(jobs_router)

    log.info(
        "product_api.app.created",
        extra={
            "version": __version__,
            "routes": [r.path for r in app.routes],
        },
    )

    return app


__all__ = ["create_app"]
