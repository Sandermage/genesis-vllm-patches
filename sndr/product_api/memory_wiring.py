# SPDX-License-Identifier: Apache-2.0
"""Shared wiring for the persistent neural-graph memory subsystem.

The modular ``server.create_app`` and the composed ``unified.create_app`` (the
full Control Center daemon + memory) mount the SAME memory + gateway routers and
attach the SAME engine/gateway/maintenance, so the GUI Memory panel behaves
identically wherever the GUI is served. This module is the single source of that
wiring, so the two factories never drift.

Everything here is env-gated and side-effect-free at import:
  * ``attach_memory_engine``   — always attaches an engine (Postgres or in-memory).
  * ``attach_gateway``         — only if an upstream is configured.
  * ``attach_maintenance``     — only if a maintenance interval is configured.
  * ``mount_memory_routes``    — routers (API-key guarded) + the three attaches.
"""
from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI

log = logging.getLogger("sndr.product_api.memory_wiring")


def attach_memory_engine(app: FastAPI) -> None:
    """Attach a MemoryEngine to ``app.state.memory_engine``.

    Backend by env: ``GENESIS_MEMORY_DSN`` -> Postgres+pgvector, else the
    in-memory reference backend (dev default). Embedder by
    ``GENESIS_MEMORY_EMBEDDER`` (``model2vec`` or the dep-free ``hash``); the
    store's vector dim derives from the embedder, so they always match.
    """
    import os

    from sndr.memory.engine import MemoryEngine

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

    from sndr.memory.inmemory import InMemoryStore

    dsn = os.environ.get("GENESIS_MEMORY_DSN")
    if dsn:
        try:
            from sndr.memory.postgres import PostgresStore

            store = PostgresStore(dsn, dim=dim)
            log.info("product_api.memory.backend", extra={"backend": "postgres"})
        except Exception:  # noqa: BLE001 - DB outage must not kill the whole app
            # Graceful degradation: a Postgres outage downgrades memory to the
            # ephemeral in-memory backend instead of crashing create_app (S3).
            log.exception("product_api.memory.postgres_unavailable_fallback_inmemory")
            store = InMemoryStore()
    else:
        store = InMemoryStore()
        log.info("product_api.memory.backend", extra={"backend": "inmemory"})
    app.state.memory_engine = MemoryEngine(store=store, embedder=embedder)


def make_upstream(url: str, key: str | None) -> dict:
    """Build httpx forward/stream closures for one OpenAI-compatible upstream."""
    base = url.rstrip("/")

    def _headers() -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if key:
            h["Authorization"] = f"Bearer {key}"
        return h

    # One long-lived pooled client per upstream (created lazily on the running
    # loop): a fresh AsyncClient per request paid a TCP(+TLS) handshake on
    # every chat AND every stream. Closed via the "close" hook on app shutdown.
    _client: dict[str, Any] = {"c": None}

    def _get_client():
        import httpx

        if _client["c"] is None or _client["c"].is_closed:
            _client["c"] = httpx.AsyncClient(
                timeout=httpx.Timeout(300.0, connect=10.0)
            )
        return _client["c"]

    async def forward(body: dict) -> dict:
        client = _get_client()
        resp = await client.post(
            f"{base}/chat/completions", json=body, headers=_headers(), timeout=120.0
        )
        resp.raise_for_status()
        return resp.json()

    async def stream(body: dict):
        client = _get_client()
        async with client.stream(
            "POST", f"{base}/chat/completions", json=body, headers=_headers()
        ) as resp:
            async for chunk in resp.aiter_bytes():
                yield chunk

    async def close() -> None:
        if _client["c"] is not None and not _client["c"].is_closed:
            await _client["c"].aclose()

    return {"forward": forward, "stream": stream, "url": base, "close": close}


def attach_gateway(app: FastAPI) -> None:
    """Configure the memory-gateway upstream registry from env. Each request
    picks an upstream by the ``X-Memory-Upstream`` header, else the default; the
    route stays dormant (503) until at least one upstream is configured.

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
                upstreams[name] = make_upstream(spec["url"], spec.get("key"))

    url = os.environ.get("GATEWAY_UPSTREAM_URL")
    if url and "default" not in upstreams:
        upstreams["default"] = make_upstream(url, os.environ.get("GATEWAY_UPSTREAM_KEY"))

    if not upstreams:
        return
    app.state.gateway_upstreams = upstreams
    app.state.gateway_default = (
        os.environ.get("GATEWAY_DEFAULT_UPSTREAM") or next(iter(upstreams))
    )

    @app.on_event("shutdown")
    async def _close_gateway_clients() -> None:  # pragma: no cover - shutdown path
        for up in upstreams.values():
            close = up.get("close")
            if close is not None:
                with contextlib.suppress(Exception):
                    await close()

    log.info(
        "product_api.gateway.upstreams",
        extra={"names": sorted(upstreams), "default": app.state.gateway_default},
    )


def attach_maintenance(app: FastAPI) -> None:
    """Start the background maintenance loop (consolidate + prune every owner on
    a timer) — the wired leak-bound + auto-organize. Off unless
    GENESIS_MEMORY_MAINTENANCE_INTERVAL (seconds) > 0. Runs in a daemon thread so
    the blocking store calls never stall the event loop.
    """
    import os
    import threading
    import time

    interval = int(os.environ.get("GENESIS_MEMORY_MAINTENANCE_INTERVAL", "0"))
    if interval <= 0:
        return
    max_nodes = int(os.environ.get("GENESIS_MEMORY_MAX_NODES", "10000"))

    from sndr.memory.engine import run_maintenance

    # Log via uvicorn's logger so maintenance is visible in `docker logs` (the
    # bare "sndr.*" logger has no handler under uvicorn).
    oplog = logging.getLogger("uvicorn.error")

    def _loop() -> None:
        while True:
            time.sleep(interval)
            engine = getattr(app.state, "memory_engine", None)
            if engine is None:
                continue
            try:
                report = run_maintenance(engine, max_nodes=max_nodes)
                oplog.info(
                    "memory maintenance: owners=%s pruned=%s",
                    report.get("owners"), report.get("pruned"),
                )
            except Exception:  # noqa: BLE001 - a maintenance failure must not kill the loop
                oplog.exception("memory maintenance failed")

    threading.Thread(target=_loop, name="memory-maintenance", daemon=True).start()
    oplog.info(
        "memory maintenance scheduler started (interval=%ss, max_nodes=%s)",
        interval, max_nodes,
    )


def mount_memory_routes(app: FastAPI, *, guard: bool = True) -> None:
    """Mount the memory + gateway routers and wire the engine, gateway upstreams,
    and maintenance loop. Shared by both app factories so the GUI Memory panel
    behaves identically everywhere.

    ``guard`` adds the per-route ``GENESIS_MEMORY_API_KEY`` dependency. The
    standalone modular ``server`` sets it (it has NO other auth). The composed
    ``unified`` daemon sets ``guard=False``: it already sits behind the legacy
    platform auth middleware, so the SAME session/token protects memory as every
    other Control Center route — no second, GUI-breaking key. (With the homelab
    default ``SNDR_AUTH=off`` everything, memory included, is open; with
    ``SNDR_AUTH=on`` the login protects everything, memory included.)
    """
    from sndr.product_api.routes.gateway import router as gateway_router
    from sndr.product_api.routes.memory import router as memory_router

    deps = None
    if guard:
        from fastapi import Depends

        from sndr.product_api.security import require_api_key
        deps = [Depends(require_api_key)]

    app.include_router(memory_router, dependencies=deps)
    app.include_router(gateway_router, dependencies=deps)
    attach_memory_engine(app)
    attach_gateway(app)
    attach_maintenance(app)


__all__ = [
    "attach_gateway",
    "attach_maintenance",
    "attach_memory_engine",
    "make_upstream",
    "mount_memory_routes",
]
