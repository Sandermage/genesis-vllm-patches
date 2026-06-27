# SPDX-License-Identifier: Apache-2.0
"""Async-handler offload contract for the blocking-subprocess routes.

The containers / hosts / observability(doctor) domain-service functions
shell out via blocking ``subprocess.run`` (docker / nvidia-smi). The route
handlers are ``async def``, so calling those functions directly would block
the event loop for the whole subprocess and stall every concurrent request.
The fix wraps each blocking call in ``await asyncio.to_thread(...)``.

This test pins that contract structurally: it asserts the offending route
handlers ``await asyncio.to_thread`` (so a future refactor that drops the
offload is caught), and dynamically proves the loop stays free by patching a
domain-service function with a *blocking* sleep and showing a second
coroutine still makes progress concurrently.
"""
from __future__ import annotations

import asyncio
import inspect
import time

import pytest

pytest.importorskip("fastapi")

from sndr.product_api.routes import containers as containers_routes  # noqa: E402
from sndr.product_api.routes import hosts as hosts_routes  # noqa: E402
from sndr.product_api.routes import observability as observability_routes  # noqa: E402


# Each (module, handler-attr) whose body must offload its blocking service
# call. The handlers are wrapped by FastAPI's router decorator, so reach the
# underlying coroutine via the route table.
def _handler_sources(router, *paths_methods) -> dict[str, str]:
    out: dict[str, str] = {}
    for route in router.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None:
            continue
        out[endpoint.__name__] = inspect.getsource(endpoint)
    return out


class TestHandlersOffloadBlockingCalls:
    def test_container_handlers_use_to_thread(self):
        srcs = _handler_sources(containers_routes.router)
        for name in ("list_containers_endpoint", "inventory_endpoint",
                     "get_container_endpoint"):
            assert "asyncio.to_thread" in srcs[name], (
                f"{name} must offload its blocking docker call"
            )

    def test_host_handlers_use_to_thread(self):
        srcs = _handler_sources(hosts_routes.router)
        for name in ("list_hosts_endpoint", "get_local_host_endpoint",
                     "get_host_endpoint"):
            assert "asyncio.to_thread" in srcs[name]
        fleet_srcs = _handler_sources(hosts_routes.fleet_router)
        assert "asyncio.to_thread" in fleet_srcs["get_fleet_endpoint"]

    def test_doctor_handler_uses_to_thread(self):
        srcs = _handler_sources(observability_routes.doctor_router)
        assert "asyncio.to_thread" in srcs["doctor_endpoint"]


@pytest.mark.timeout(10)
def test_event_loop_stays_free_during_blocking_service(monkeypatch):
    """Behavioural proof: a blocking domain-service call dispatched via the
    handler must NOT freeze the loop — a concurrent ticker keeps ticking."""

    BLOCK_S = 0.4

    def _slow_list(*args, **kwargs):  # simulates subprocess.run blocking
        time.sleep(BLOCK_S)
        return []

    monkeypatch.setattr(containers_routes, "list_containers", _slow_list)

    async def _drive() -> int:
        ticks = 0

        async def _ticker() -> None:
            nonlocal ticks
            # Tick every 20ms for the duration of the blocking call.
            for _ in range(int(BLOCK_S / 0.02) + 5):
                await asyncio.sleep(0.02)
                ticks += 1

        async def _call_handler() -> None:
            # The real handler offloads via asyncio.to_thread, so the loop
            # is free to run the ticker while the blocking sleep runs.
            await containers_routes.list_containers_endpoint(engine=None)

        ticker = asyncio.create_task(_ticker())
        await _call_handler()
        # If the loop had been blocked for BLOCK_S, the ticker would have had
        # no chance to advance during the call. With to_thread offload it does.
        await asyncio.sleep(0)  # let the ticker settle
        ticker.cancel()
        return ticks

    ticks = asyncio.run(_drive())
    # During a 0.4s blocking call, a free loop ticks ~20 times. Require a
    # generous lower bound (loop made real progress, i.e. was NOT blocked).
    assert ticks >= 5, (
        f"event loop appears blocked during the handler call (ticks={ticks}); "
        "the blocking service call was not offloaded via asyncio.to_thread"
    )
