# SPDX-License-Identifier: Apache-2.0
"""Tests for the GUI preflight fit-check route (/api/v1/preflight).

The route must mirror ``sndr preflight <preset>`` exactly (it reuses
``preflight_fit.evaluate_fit``), so the GUI pre-launch fit-check and the CLI
never diverge. These tests pin the single-card escape-hatch case from
docs/SINGLE_CARD.md (1× 24 GB rig vs a 2× preset -> gpu_count FAIL).
"""
from __future__ import annotations

import asyncio
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from sndr.model_configs import preflight_fit  # noqa: E402
from sndr.product_api.legacy.http_app import create_app  # noqa: E402

_PRESET = "prod-qwen3.6-35b-balanced"


def _client() -> TestClient:
    return TestClient(create_app(allowed_origins=()))


def test_single_card_fake_rig_fails_gpu_count():
    """A single 24 GB card vs a 2× preset: gpu_count FAILs, VRAM/SM pass,
    verdict CANNOT RUN — the exact docs/SINGLE_CARD.md output shape."""
    r = _client().get(
        "/api/v1/preflight",
        params={"preset_id": _PRESET, "fake_gpus": "RTX 3090:24576:8.6"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["preset"] == _PRESET
    assert body["can_run"] is False
    assert body["verdict"] == "CANNOT RUN"
    assert body["rig"]["gpu_count"] == 1
    assert body["rig"]["min_vram_gb"] == 24

    by_dim = {c["dimension"]: c for c in body["checks"]}
    assert by_dim["gpu_count"]["status"] == "fail"
    assert by_dim["vram"]["status"] == "pass"
    assert by_dim["cuda_capability"]["status"] == "pass"
    # The required envelope is surfaced so the GUI can render "needs 2× 24GB".
    assert body["required"]["min_gpu_count"] == 2
    assert body["required"]["tensor_parallel"] == 2


def test_builtin_rig_two_cards_can_run():
    """The 2× A5000 builtin rig clears the 2× preset (boots), offline.

    The byte-level projection row is additive: against the builtin rig's
    DECLARED min-VRAM floor (22000 MiB — a conservative lower bound, not the
    24564 MiB physical card) the 35B's fixed footprint is tight, so the
    projection adds a WARN (downgraded from FAIL because a declared floor must
    not hard-block a config that runs on the real card). ``can_run`` stays
    True; the verdict may carry the warning."""
    r = _client().get(
        "/api/v1/preflight",
        params={"preset_id": _PRESET, "rig": "a5000-2x-24gbvram-16cpu-128gbram"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["can_run"] is True
    assert body["verdict"] in ("CAN RUN", "RUNNABLE (with warnings)")
    assert body["rig_source"].startswith("rig:")
    # The new byte-level projection row is present in the route response.
    dims = {c["dimension"] for c in body["checks"]}
    assert "projected_vram" in dims


def test_unknown_preset_is_404():
    r = _client().get("/api/v1/preflight", params={"preset_id": "no-such-preset"})
    assert r.status_code == 404


def test_unknown_rig_is_404():
    r = _client().get(
        "/api/v1/preflight",
        params={"preset_id": _PRESET, "rig": "no-such-rig"},
    )
    assert r.status_code == 404


# ── Non-blocking + bounded behavior (BUG 1: the fit-check used to HANG) ───────
#
# The original handler called RigProbe().detect() (a SYNCHRONOUS nvidia-smi
# subprocess) directly inside the async route, blocking the whole event loop for
# the probe's duration and — with no client-visible timeout — letting the GUI
# spin on "Running fit check…" forever. The fix runs the probe off the event
# loop (asyncio.to_thread) under a hard ``SNDR_PREFLIGHT_DEADLINE_S`` cap. These
# tests pin both guarantees: the loop stays free while the probe runs, and a
# stalled probe returns a bounded 504 instead of hanging.


def _slow_detect(delay: float):
    """A RigProbe.detect that blocks for ``delay`` seconds (simulates a wedged
    nvidia-smi / a GPU busy loading a 35B engine), then returns an empty rig."""
    def _detect(self):  # noqa: ANN001, ANN202
        time.sleep(delay)
        return preflight_fit.Rig(gpus=[], source="nvidia-smi")
    return _detect


async def _asgi_client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def test_live_probe_does_not_block_event_loop(monkeypatch):
    """While a SLOW live probe runs, a concurrent coroutine must keep ticking —
    proving the probe is off the event loop (the bug: 0 ticks, loop frozen)."""
    monkeypatch.setattr(preflight_fit.RigProbe, "detect", _slow_detect(1.0))
    # Generous deadline so the request SUCCEEDS (we are testing non-block, not
    # the timeout path) — the live-rig path (no rig/fake_gpus) hits the probe.
    monkeypatch.setenv("SNDR_PREFLIGHT_DEADLINE_S", "10")
    app = create_app(allowed_origins=())

    async def _run():
        client = await _asgi_client(app)
        ticks = 0
        stop = asyncio.Event()

        async def _heartbeat():
            nonlocal ticks
            while not stop.is_set():
                ticks += 1
                await asyncio.sleep(0.05)

        hb = asyncio.create_task(_heartbeat())
        resp = await client.get("/api/v1/preflight", params={"preset_id": _PRESET})
        stop.set()
        await hb
        await client.aclose()
        return resp, ticks

    resp, ticks = asyncio.run(_run())
    assert resp.status_code == 200, resp.text
    # ~1s of probe at 50ms ticks => ~20 ticks if the loop was free. A blocked
    # loop yields ~0-1. Assert clearly more than a blocked loop could produce.
    assert ticks >= 8, f"event loop appears blocked during probe (only {ticks} ticks)"


def test_stalled_probe_times_out_bounded(monkeypatch):
    """A probe that never returns within the deadline yields a bounded 504, not
    an indefinite hang. Deadline is 0.3s; the probe takes 2s — so the route MUST
    return before the probe completes (the orphaned worker thread is short so
    the test itself doesn't stall)."""
    monkeypatch.setattr(preflight_fit.RigProbe, "detect", _slow_detect(2.0))
    monkeypatch.setenv("SNDR_PREFLIGHT_DEADLINE_S", "0.3")
    app = create_app(allowed_origins=())

    async def _run():
        client = await _asgi_client(app)
        t0 = time.monotonic()
        resp = await client.get("/api/v1/preflight", params={"preset_id": _PRESET})
        elapsed = time.monotonic() - t0
        await client.aclose()
        return resp, elapsed

    resp, elapsed = asyncio.run(_run())
    assert resp.status_code == 504, resp.text
    assert "timed out" in resp.text.lower() or "exceeded" in resp.text.lower()
    # Returned near the 0.3s deadline, well before the 2s probe — bounded.
    assert elapsed < 1.5, f"504 took {elapsed:.1f}s — deadline not enforced"


def test_offline_rig_path_never_touches_the_probe(monkeypatch):
    """The modeled-rig (--rig) and fake-gpus paths must resolve WITHOUT the live
    probe, so they stay instant even on a host with a wedged nvidia-smi."""
    def _boom(self):  # noqa: ANN001, ANN202
        raise AssertionError("live probe must not run for an offline rig")
    monkeypatch.setattr(preflight_fit.RigProbe, "detect", _boom)
    r = _client().get(
        "/api/v1/preflight",
        params={"preset_id": _PRESET, "fake_gpus": "RTX A5000:24564:8.6;RTX A5000:24564:8.6"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["can_run"] is True
