# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the live engine micro-benchmark (no real network)."""
from __future__ import annotations

import pytest

from sndr.product_api.legacy import engine_bench as eb
from sndr.product_api.legacy.engine_client import EngineError


def test_parse_sse_stream_timings_and_tokens():
    lines = [
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}',  # no content yet
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}',   # first token -> ttft
        b'data: {"choices":[{"delta":{"content":" world"}}]}',
        b'data: {"choices":[{"delta":{"content":"!"}}]}',
        b'data: {"choices":[],"usage":{"completion_tokens":3}}',
        b"data: [DONE]",
    ]
    ticks = iter([10.1, 10.2, 10.3])  # times for the 3 content chunks
    out = eb.parse_sse_stream(lines, t0=10.0, clock=lambda: next(ticks))
    assert out["ok"] is True
    assert out["tokens"] == 3                 # from usage
    assert out["ttft_s"] == pytest.approx(0.1)  # 10.1 - 10.0
    assert out["total_s"] == pytest.approx(0.3)  # last content at 10.3


def test_parse_sse_stream_counts_chunks_without_usage():
    lines = [
        b'data: {"choices":[{"delta":{"content":"a"}}]}',
        b'data: {"choices":[{"delta":{"content":"b"}}]}',
        b"data: [DONE]",
    ]
    ticks = iter([1.0, 1.05])
    out = eb.parse_sse_stream(lines, t0=0.9, clock=lambda: next(ticks))
    assert out["tokens"] == 2  # fell back to chunk count


def test_run_bench_aggregates(monkeypatch):
    # Deterministic per-request result: 0.05s ttft, 0.55s total, 50 tokens.
    def fake_runner():
        return {"ttft_s": 0.05, "total_s": 0.55, "tokens": 50, "ok": True}

    out = eb.run_bench({"num_requests": 4, "concurrency": 2, "max_tokens": 64}, _runner=fake_runner)
    assert out["ok"] is True
    m = out["metrics"]
    assert m["requests_ok"] == 4 and m["requests_failed"] == 0
    assert m["total_tokens"] == 200
    assert m["ttft_avg_ms"] == 50.0
    # tpot = (0.55-0.05)/(50-1) = 0.0102s -> ~10.2 ms
    assert m["tpot_avg_ms"] == pytest.approx(10.2, abs=0.1)
    assert m["cv_pct"] == 0.0  # identical requests
    assert out["params"]["num_requests"] == 4


def test_run_bench_guards_caps():
    captured = {"n": 0}

    def counting_runner():
        captured["n"] += 1
        return {"ttft_s": 0.01, "total_s": 0.1, "tokens": 5, "ok": True}

    out = eb.run_bench({"num_requests": 9999, "concurrency": 999, "max_tokens": 99999}, _runner=counting_runner)
    assert out["params"]["num_requests"] == 64        # capped
    assert out["params"]["max_tokens"] == 2048        # capped
    assert captured["n"] == 64


def test_run_bench_all_failed_raises():
    def failing_runner():
        return {"ttft_s": None, "total_s": 0.1, "tokens": 0, "ok": False, "error": "refused"}

    with pytest.raises(EngineError):
        eb.run_bench({"num_requests": 3}, _runner=failing_runner)


def test_run_bench_mixed_success_counts_failures():
    seq = iter([
        {"ttft_s": 0.05, "total_s": 0.5, "tokens": 40, "ok": True},
        {"ttft_s": None, "total_s": 0.1, "tokens": 0, "ok": False},
        {"ttft_s": 0.06, "total_s": 0.6, "tokens": 50, "ok": True},
    ])
    out = eb.run_bench({"num_requests": 3, "concurrency": 1}, _runner=lambda: next(seq))
    assert out["metrics"]["requests_ok"] == 2
    assert out["metrics"]["requests_failed"] == 1
    assert out["metrics"]["total_tokens"] == 90


def test_engine_bench_route(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from sndr.product_api.legacy.http_app import create_app

    monkeypatch.setattr(eb, "run_bench", lambda payload, host=None: {"ok": True, "metrics": {"throughput_tok_s": 142.0}})
    client = TestClient(create_app(allowed_origins=()))
    resp = client.post("/api/v1/engine/bench", json={"num_requests": 4})
    assert resp.status_code == 200 and resp.json()["metrics"]["throughput_tok_s"] == 142.0


def test_engine_bench_route_engine_down(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from sndr.product_api.legacy.http_app import create_app

    def boom(payload, host=None):
        raise OSError("Connection refused")
    monkeypatch.setattr(eb, "run_bench", boom)
    client = TestClient(create_app(allowed_origins=()))
    assert client.post("/api/v1/engine/bench", json={}).status_code == 503
