# SPDX-License-Identifier: Apache-2.0
"""FastAPI application factory.

Usage::

    from sndr.product_api.server import create_app
    app = create_app()

Or with uvicorn::

    uvicorn sndr.product_api.server:create_app --factory --host 0.0.0.0 --port 8800

In v12.x, this new server runs alongside the legacy
``sndr.product_api.legacy.http_app`` monolith (the production daemon,
``sndr gui-api``). New engine-aware routes
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

    # Serve the built Carbon Control Center SPA from the daemon itself.
    # API routes are registered above, so they take precedence over the mount.
    # Absent a build, the daemon stays API-only (unchanged behavior).
    _mount_carbon_ui(app)

    log.info(
        "product_api.app.created",
        extra={
            "version": __version__,
            "routes": [r.path for r in app.routes],
        },
    )

    return app


def _mount_carbon_ui(app: "FastAPI") -> None:  # type: ignore[name-defined]
    """Mount the built Carbon GUI as a static SPA, if a build is present.

    Resolution order: ``SNDR_GUI_STATIC_CARBON`` env → packaged
    ``web_static_carbon`` beside this module. Requires an ``index.html``
    (the build's ``index.carbon.html`` is renamed on bundling — see the
    ``gui-build-carbon`` make target).

    The Carbon app uses a history-based router (BrowserRouter), so unknown
    paths (client-routed deep links like ``/fleet``) fall back to
    ``index.html`` instead of 404-ing.
    """
    static_dir = _resolve_carbon_static_dir()
    if static_dir is None:
        log.info("product_api.ui.carbon_static_absent")
        return

    from starlette.exceptions import HTTPException as StarletteHTTPException
    from starlette.staticfiles import StaticFiles

    class _CarbonUiStatic(StaticFiles):
        """Serve the Carbon SPA with correct caching and client-route
        fallback. HTML revalidates so a fresh deploy is picked up
        immediately; content-hashed assets are immutable for a year;
        unknown non-asset paths resolve to ``index.html`` for the router.

        Starlette's StaticFiles raises ``HTTPException(404)`` for a missing
        file rather than returning a 404 response, so the fallback has to
        catch the exception (and tolerate the returned-404 case too)."""

        async def _serve(self, path: str, scope):
            try:
                resp = await super().get_response(path, scope)
            except StarletteHTTPException as exc:
                if exc.status_code == 404 and not path.startswith("assets/"):
                    return await super().get_response("index.html", scope)
                raise
            if resp.status_code == 404 and not path.startswith("assets/"):
                return await super().get_response("index.html", scope)
            return resp

        async def get_response(self, path: str, scope):  # type: ignore[override]
            resp = await self._serve(path, scope)
            if path.endswith(".html") or path in ("", "."):
                resp.headers["Cache-Control"] = "no-cache"
            elif "assets/" in path:
                resp.headers["Cache-Control"] = (
                    "public, max-age=31536000, immutable"
                )
            return resp

    app.mount(
        "/", _CarbonUiStatic(directory=str(static_dir), html=True),
        name="carbon-ui",
    )
    log.info("product_api.ui.carbon_mounted", extra={"dir": str(static_dir)})


def _resolve_carbon_static_dir():
    """Locate the built Carbon GUI directory, or None if not present.

    Resolution order: ``SNDR_GUI_STATIC_CARBON`` env → packaged
    ``web_static_carbon`` beside this module. Requires an ``index.html``
    (the build emits ``index.carbon.html``, renamed on bundling — see the
    ``gui-build-carbon`` make target).
    """
    import os
    from pathlib import Path

    candidates = []
    env_dir = os.environ.get("SNDR_GUI_STATIC_CARBON", "").strip()
    if env_dir:
        candidates.append(Path(env_dir))
    here = Path(__file__).resolve()
    candidates.append(here.parent / "web_static_carbon")
    for candidate in candidates:
        try:
            if (candidate / "index.html").is_file():
                return candidate
        except OSError:
            continue
    return None


__all__ = ["create_app"]
