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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

log = logging.getLogger("sndr.product_api.server")


def create_app() -> FastAPI:
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
    from sndr.product_api.routes.gateway import router as gateway_router
    from sndr.product_api.routes.health import router as health_router
    from sndr.product_api.routes.hosts import fleet_router
    from sndr.product_api.routes.hosts import router as hosts_router
    from sndr.product_api.routes.licensing import router as licensing_router
    from sndr.product_api.routes.memory import router as memory_router
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
    app.include_router(memory_router)
    app.include_router(gateway_router)

    # Persistent neural-graph memory engine on app.state (Postgres if
    # GENESIS_MEMORY_DSN is set, else the in-memory reference backend).
    _init_memory_engine(app)
    # Memory-augmented OpenAI gateway (active only if GATEWAY_UPSTREAM_URL set).
    _init_gateway(app)
    # Background consolidation + prune (the wired leak-bound); off unless an
    # interval is configured.
    _init_maintenance(app)

    # Serve the built Carbon Control Center SPA from the daemon itself.
    # API routes are registered above, so they take precedence over the mount.
    # Absent a build, the daemon stays API-only (unchanged behavior).
    _mount_carbon_ui(app)

    log.info(
        "product_api.app.created",
        extra={
            "version": __version__,
            "routes": len(app.routes),  # version-robust (some route objects lack .path)
        },
    )

    return app


def _init_memory_engine(app: FastAPI) -> None:
    """Attach a MemoryEngine to ``app.state.memory_engine``.

    Backend by env: ``GENESIS_MEMORY_DSN`` -> Postgres+pgvector, else the
    in-memory reference backend (dev default). ``GENESIS_MEMORY_DIM`` sets the
    embedding dimension (must match the Postgres ``vector(dim)`` column).
    Embedder is the dependency-free HashEmbedder until a CPU embedder is wired.
    """
    import os

    from sndr.memory.engine import MemoryEngine

    # Embedder by env: model2vec (real CPU embedder) or the dep-free hash fallback.
    # The store's vector dim derives from the embedder, so they always match.
    if os.environ.get("GENESIS_MEMORY_EMBEDDER", "hash").lower() == "model2vec":
        from sndr.memory.embedder import Model2VecEmbedder

        embedder = Model2VecEmbedder(
            os.environ.get("GENESIS_MEMORY_MODEL", "minishlab/potion-base-8M")
        )
        log.info("product_api.memory.embedder", extra={"embedder": "model2vec"})
    else:
        from sndr.memory.embedder import HashEmbedder

        embedder = HashEmbedder(dim=int(os.environ.get("GENESIS_MEMORY_DIM", "256")))
        log.info("product_api.memory.embedder", extra={"embedder": "hash"})
    dim = embedder.dim

    dsn = os.environ.get("GENESIS_MEMORY_DSN")
    if dsn:
        from sndr.memory.postgres import PostgresStore

        store = PostgresStore(dsn, dim=dim)
        log.info("product_api.memory.backend", extra={"backend": "postgres"})
    else:
        from sndr.memory.inmemory import InMemoryStore

        store = InMemoryStore()
        log.info("product_api.memory.backend", extra={"backend": "inmemory"})
    app.state.memory_engine = MemoryEngine(store=store, embedder=embedder)


def _make_upstream(url: str, key: str | None) -> dict:
    """Build httpx forward/stream closures for one OpenAI-compatible upstream."""
    base = url.rstrip("/")

    def _headers() -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if key:
            h["Authorization"] = f"Bearer {key}"
        return h

    async def forward(body: dict) -> dict:
        import httpx

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{base}/chat/completions", json=body, headers=_headers()
            )
            resp.raise_for_status()
            return resp.json()

    async def stream(body: dict):
        import httpx

        async with httpx.AsyncClient(timeout=300.0) as client, client.stream(
            "POST", f"{base}/chat/completions", json=body, headers=_headers()
        ) as resp:
            async for chunk in resp.aiter_bytes():
                yield chunk

    return {"forward": forward, "stream": stream, "url": base}


def _init_gateway(app: FastAPI) -> None:
    """Configure the memory-gateway upstream registry from env. Each request
    picks an upstream by the `X-Memory-Upstream` header, else the default; the
    route stays dormant (503) until at least one upstream is configured.

    Two ways to configure (combinable), so you can run with your CLIProxyAPI,
    another proxy, and/or the internal vLLM and choose per request:

      GATEWAY_UPSTREAMS   JSON map, e.g.
        {"cliproxy":{"url":"http://cliproxyapi:8317/v1","key":"..."},
         "local":{"url":"http://vllm:8102/v1","key":"genesis-local"}}
      GATEWAY_DEFAULT_UPSTREAM   name used when no header is sent
      GATEWAY_UPSTREAM_URL / GATEWAY_UPSTREAM_KEY   single-upstream shortcut
        (registered as "default")
    """
    import json
    import os

    upstreams: dict[str, dict] = {}
    raw = os.environ.get("GATEWAY_UPSTREAMS")
    if raw:
        try:
            cfg = json.loads(raw)
        except (ValueError, TypeError):
            log.warning("product_api.gateway.bad_upstreams_json")
            cfg = {}
        for name, spec in (cfg or {}).items():
            if isinstance(spec, dict) and spec.get("url"):
                upstreams[name] = _make_upstream(spec["url"], spec.get("key"))

    url = os.environ.get("GATEWAY_UPSTREAM_URL")
    if url and "default" not in upstreams:
        upstreams["default"] = _make_upstream(url, os.environ.get("GATEWAY_UPSTREAM_KEY"))

    if not upstreams:
        return
    app.state.gateway_upstreams = upstreams
    app.state.gateway_default = (
        os.environ.get("GATEWAY_DEFAULT_UPSTREAM") or next(iter(upstreams))
    )
    log.info(
        "product_api.gateway.upstreams",
        extra={"names": sorted(upstreams), "default": app.state.gateway_default},
    )


def _init_maintenance(app: FastAPI) -> None:
    """Start the background maintenance loop (consolidate + prune every owner on
    a timer) — the wired leak-bound + auto-organize. Off unless
    GENESIS_MEMORY_MAINTENANCE_INTERVAL (seconds) > 0. Runs in a daemon thread so
    the blocking store calls never stall the event loop.

      GENESIS_MEMORY_MAINTENANCE_INTERVAL   seconds between passes (0 = off)
      GENESIS_MEMORY_MAX_NODES              per-owner cap (default 10000)
    """
    import os
    import threading
    import time

    interval = int(os.environ.get("GENESIS_MEMORY_MAINTENANCE_INTERVAL", "0"))
    if interval <= 0:
        return
    max_nodes = int(os.environ.get("GENESIS_MEMORY_MAX_NODES", "10000"))

    from sndr.memory.engine import run_maintenance

    def _loop() -> None:
        while True:
            time.sleep(interval)
            engine = getattr(app.state, "memory_engine", None)
            if engine is None:
                continue
            try:
                report = run_maintenance(engine, max_nodes=max_nodes)
                log.info("product_api.memory.maintained", extra=report)
            except Exception:  # noqa: BLE001 - a maintenance failure must not kill the loop
                log.exception("product_api.memory.maintenance_failed")

    threading.Thread(target=_loop, name="memory-maintenance", daemon=True).start()
    log.info(
        "product_api.memory.maintenance_started",
        extra={"interval": interval, "max_nodes": max_nodes},
    )


def _mount_carbon_ui(app: FastAPI) -> None:
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
