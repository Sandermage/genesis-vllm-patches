# SPDX-License-Identifier: Apache-2.0
"""TTL-cache contract for the live GPU telemetry route.

`/api/v1/host/gpu` forks `nvidia-smi` and reads `/proc` on every call. The
Hardware view polls it every 4 s, so a single client must always see fresh
data; the cache exists only to collapse *concurrent* polls from multiple
clients within a short (~2 s) window so they do not each fork a subprocess.

A stale cache here would freeze the operator's live telemetry, so these tests
pin the contract: a hit inside the TTL reuses the reading, an expired entry
re-collects, and TTL=0 disables the cache entirely (every call is live).
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from sndr.product_api.legacy import gpu_telemetry as G  # noqa: E402
from sndr.product_api.legacy.http_app import create_app  # noqa: E402


class _Counter:
    """Stand-in for `collect_local` that counts subprocess collections and
    stamps each reading so the test can tell a fresh one from a cached one."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> G.HardwareTelemetry:
        self.calls += 1
        return G.HardwareTelemetry(
            gpus=[{"name": "fake", "reading": self.calls}],
            system={},
            error=None,
        )


def _patch_collector(monkeypatch) -> _Counter:
    counter = _Counter()
    monkeypatch.setattr(G, "collect_local", counter)
    return counter


def test_concurrent_polls_within_ttl_collapse_to_one_collection(monkeypatch):
    """Two back-to-back polls inside the TTL fork nvidia-smi exactly once."""
    monkeypatch.setenv("SNDR_GPU_TELEMETRY_TTL_S", "30")  # long: never expires mid-test
    counter = _patch_collector(monkeypatch)
    client = TestClient(create_app(allowed_origins=()))

    first = client.get("/api/v1/host/gpu").json()
    second = client.get("/api/v1/host/gpu").json()

    assert counter.calls == 1, "second poll inside TTL must be served from cache"
    assert first["gpus"][0]["reading"] == 1
    assert second == first, "cache hit must return the identical reading"


def test_expired_ttl_recollects(monkeypatch):
    """Once the TTL lapses, the next poll forks a fresh collection."""
    monkeypatch.setenv("SNDR_GPU_TELEMETRY_TTL_S", "0.05")
    counter = _patch_collector(monkeypatch)
    client = TestClient(create_app(allowed_origins=()))

    first = client.get("/api/v1/host/gpu").json()
    assert first["gpus"][0]["reading"] == 1

    import time
    time.sleep(0.1)  # let the entry expire

    second = client.get("/api/v1/host/gpu").json()
    assert counter.calls == 2, "an expired entry must trigger a fresh collection"
    assert second["gpus"][0]["reading"] == 2, "operator must see the live reading"


def test_ttl_zero_disables_cache(monkeypatch):
    """TTL=0 is the kill switch — every poll is live (no caching at all)."""
    monkeypatch.setenv("SNDR_GPU_TELEMETRY_TTL_S", "0")
    counter = _patch_collector(monkeypatch)
    client = TestClient(create_app(allowed_origins=()))

    client.get("/api/v1/host/gpu")
    client.get("/api/v1/host/gpu")
    client.get("/api/v1/host/gpu")

    assert counter.calls == 3, "TTL=0 must collect on every request"


def test_each_app_instance_has_its_own_cache(monkeypatch):
    """The cache is per-daemon (per app instance), so a fresh app never
    serves a previous instance's stale reading."""
    monkeypatch.setenv("SNDR_GPU_TELEMETRY_TTL_S", "30")
    counter = _patch_collector(monkeypatch)

    client_a = TestClient(create_app(allowed_origins=()))
    client_a.get("/api/v1/host/gpu")
    assert counter.calls == 1

    client_b = TestClient(create_app(allowed_origins=()))
    client_b.get("/api/v1/host/gpu")
    assert counter.calls == 2, "a new app instance must not reuse the old cache"
