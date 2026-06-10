# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the live vLLM engine client (no real network)."""
from __future__ import annotations

import pytest

from sndr.product_api.legacy import engine_client as ec

# A trimmed but realistic vLLM Prometheus exposition, using this pin's names.
SAMPLE_METRICS = """
# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="qwen"} 3.0
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="qwen"} 5.0
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="qwen"} 0.42
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="qwen"} 12000.0
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="qwen"} 8000.0
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_sum{model_name="qwen"} 20.0
vllm:time_to_first_token_seconds_count{model_name="qwen"} 100.0
# TYPE vllm:time_per_output_token_seconds histogram
vllm:time_per_output_token_seconds_sum{model_name="qwen"} 50.0
vllm:time_per_output_token_seconds_count{model_name="qwen"} 5000.0
# TYPE vllm:spec_decode_acceptance_rate gauge
vllm:spec_decode_acceptance_rate{model_name="qwen"} 0.78
vllm:request_success_total{model_name="qwen",finished_reason="stop"} 90.0
vllm:request_success_total{model_name="qwen",finished_reason="length"} 10.0
"""


def test_safe_host_rejects_ssrf_payloads():
    assert ec._safe_host("gpu-build-01") == "gpu-build-01"
    assert ec._safe_host("10.0.0.5") == "10.0.0.5"
    # anything with scheme/path/port/space collapses to loopback
    assert ec._safe_host("http://evil/x") == "127.0.0.1"
    assert ec._safe_host("evil:9000/admin") == "127.0.0.1"
    assert ec._safe_host("a b") == "127.0.0.1"
    assert ec._safe_host(None) == "127.0.0.1"


def test_resolve_engine_defaults_and_env(monkeypatch):
    monkeypatch.delenv("SNDR_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("SNDR_METRICS_URL", raising=False)
    monkeypatch.delenv("SNDR_RUNTIME_HOST", raising=False)
    eng = ec.resolve_engine("gpu-host")
    assert eng["base_url"] == "http://gpu-host:8000/v1"
    assert eng["root_url"] == "http://gpu-host:8000"
    assert eng["metrics_url"] == "http://gpu-host:8001/metrics"
    monkeypatch.setenv("SNDR_OPENAI_BASE_URL", "https://api.example/v1")
    monkeypatch.setenv("SNDR_METRICS_URL", "https://api.example/metrics")
    eng2 = ec.resolve_engine()
    assert eng2["base_url"] == "https://api.example/v1"
    assert eng2["root_url"] == "https://api.example"


def test_parse_prometheus_basic():
    parsed = ec.parse_prometheus(SAMPLE_METRICS)
    assert parsed["vllm:num_requests_running"][0][1] == 3.0
    # two label sets summed by helper
    assert ec._sum(parsed, "vllm:request_success_total") == 100.0
    assert ec._first(parsed, "vllm:kv_cache_usage_perc") == 0.42
    labels = parsed["vllm:request_success_total"][0][0]
    assert labels["model_name"] == "qwen" and "finished_reason" in labels


def test_avg_from_histogram():
    parsed = ec.parse_prometheus(SAMPLE_METRICS)
    assert ec._avg_from_histogram(parsed, "vllm:time_to_first_token_seconds") == pytest.approx(0.2)
    assert ec._avg_from_histogram(parsed, "vllm:time_per_output_token_seconds") == pytest.approx(0.01)


def test_engine_metrics_kpis_and_throughput(monkeypatch):
    monkeypatch.delenv("SNDR_METRICS_URL", raising=False)
    monkeypatch.setattr(ec, "_get", lambda url, timeout=3.0: (200, SAMPLE_METRICS))
    ec._LAST_SCRAPE.clear()
    first = ec.engine_metrics("h", now=1000.0)
    assert first["reachable"] is True
    k = first["kpis"]
    assert k["requests_running"] == 3.0 and k["requests_waiting"] == 5.0
    assert k["kv_cache_usage"] == 0.42
    assert k["ttft_avg_s"] == pytest.approx(0.2)
    assert k["spec_decode_acceptance_rate"] == 0.78
    assert "generation_toks_per_s" not in k  # first scrape has no delta
    # second scrape 10s later with +2000 generation tokens -> 200 tok/s
    bumped = SAMPLE_METRICS.replace("vllm:generation_tokens_total{model_name=\"qwen\"} 8000.0",
                                    "vllm:generation_tokens_total{model_name=\"qwen\"} 10000.0")
    monkeypatch.setattr(ec, "_get", lambda url, timeout=3.0: (200, bumped))
    second = ec.engine_metrics("h", now=1010.0)
    assert second["kpis"]["generation_toks_per_s"] == 200.0


def test_engine_metrics_unreachable(monkeypatch):
    def boom(url, timeout=3.0):
        raise OSError("Connection refused")
    monkeypatch.setattr(ec, "_get", boom)
    out = ec.engine_metrics("h")
    assert out["reachable"] is False and "Connection refused" in out["error"]


def test_describe_dns_failure_is_operator_friendly():
    # urllib raises URLError(reason=socket.gaierror(8, "nodename nor servname
    # provided, or not known")) when the engine host can't be resolved (e.g. an
    # unconfigured placeholder host). The raw repr is opaque to operators; the
    # description must name the resolution problem clearly.
    import socket
    import urllib.error
    exc = urllib.error.URLError(socket.gaierror(8, "nodename nor servname provided, or not known"))
    msg = ec._describe(exc)
    assert "resolve" in msg.lower()
    assert "[Errno 8]" not in msg
    assert "nodename" not in msg.lower()


def test_describe_name_or_service_not_known_linux():
    # The Linux/glibc spelling of the same failure must be normalised too.
    import socket
    import urllib.error
    exc = urllib.error.URLError(socket.gaierror(-2, "Name or service not known"))
    assert "resolve" in ec._describe(exc).lower()


def test_engine_status_dns_failure_surfaces_clean_error(monkeypatch):
    import socket
    import urllib.error

    def boom(url, timeout=3.0, api_key=None):
        raise urllib.error.URLError(socket.gaierror(8, "nodename nor servname provided, or not known"))
    monkeypatch.setattr(ec, "_get", boom)
    out = ec.engine_status("gpu-build-01")
    assert out["reachable"] is False
    assert "resolve" in out["error"].lower()
    assert "nodename" not in out["error"].lower()


def test_engine_status(monkeypatch):
    def fake_get(url, timeout=3.0, api_key=None):
        if url.endswith("/health"):
            return 200, ""
        if url.endswith("/version"):
            return 200, '{"version": "0.20.2rc1.dev338"}'
        if url.endswith("/models"):
            return 200, '{"data": [{"id": "qwen3.6-35b"}]}'
        raise AssertionError(url)
    monkeypatch.setattr(ec, "_get", fake_get)
    out = ec.engine_status("h")
    assert out["reachable"] is True
    assert out["version"] == "0.20.2rc1.dev338"
    assert out["models"] == ["qwen3.6-35b"]


def test_engine_chat_proxy(monkeypatch):
    captured = {}
    def fake_post(url, payload, timeout=60.0, api_key=None):
        captured["url"] = url
        captured["payload"] = payload
        return 200, '{"model":"qwen","choices":[{"message":{"content":"hi there"},"finish_reason":"stop"}],"usage":{"total_tokens":12}}'
    monkeypatch.setattr(ec, "_post_json", fake_post)
    out = ec.engine_chat({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 16})
    assert out["reply"] == "hi there" and out["finish_reason"] == "stop"
    assert out["usage"]["total_tokens"] == 12 and out["latency_ms"] >= 0
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["payload"]["stream"] is False


def test_engine_chat_requires_messages():
    with pytest.raises(ValueError):
        ec.engine_chat({"messages": []})


def test_api_key_is_forwarded_as_bearer(monkeypatch):
    # A key-protected engine (e.g. 35B PROD :8102) needs Authorization: Bearer.
    captured = {}
    def fake_post(url, payload, timeout=60.0, api_key=None):
        captured["api_key"] = api_key
        return 200, '{"choices":[{"message":{"content":"ok"}}],"usage":{}}'
    monkeypatch.setattr(ec, "_post_json", fake_post)
    ec.engine_chat({"messages": [{"role": "user", "content": "hi"}]}, api_key="secret-key")
    assert captured["api_key"] == "secret-key"
    # The low-level header builder turns it into a Bearer header.
    assert ec._auth_headers("secret-key") == {"Authorization": "Bearer secret-key"}
    assert ec._auth_headers(None) == {}


def test_api_key_falls_back_to_env(monkeypatch):
    monkeypatch.delenv("SNDR_ENGINE_API_KEY", raising=False)
    monkeypatch.setenv("VLLM_API_KEY", "env-key")
    assert ec._resolve_api_key(None) == "env-key"
    assert ec._resolve_api_key("explicit") == "explicit"  # explicit wins


def test_probe_host_forwards_api_key_to_models(monkeypatch):
    # A host-card probe of a key-protected engine must list models, not 401.
    seen = {}
    def fake_get(url, timeout=3.0, api_key=None):
        seen[url.rsplit("/", 1)[-1] if "/" in url else url] = api_key
        if url.endswith("/health"):
            return 200, ""
        if url.endswith("/version"):
            return 200, '{"version":"x"}'
        if url.endswith("/models"):
            return 200, '{"data":[{"id":"qwen-35b"}]}'
        raise AssertionError(url)
    monkeypatch.setattr(ec, "_get", fake_get)
    out = ec.probe_host("192.168.1.10", 8102, api_key="genesis-local")
    assert out["reachable"] is True and out["models"] == ["qwen-35b"]
    assert seen["models"] == "genesis-local" and seen["version"] == "genesis-local"


def test_engine_chat_clamps_max_tokens(monkeypatch):
    captured = {}
    def fake_post(url, payload, timeout=60.0, api_key=None):
        captured["payload"] = payload
        return 200, '{"choices":[{"message":{"content":"x"}}],"usage":{}}'
    monkeypatch.setattr(ec, "_post_json", fake_post)
    ec.engine_chat({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10_000_000})
    assert captured["payload"]["max_tokens"] == 4096  # capped (anti-DoS)
    ec.engine_chat({"messages": [{"role": "user", "content": "hi"}], "max_tokens": "garbage"})
    assert captured["payload"]["max_tokens"] == 256   # invalid -> default


# ---- HTTP route wiring (FastAPI TestClient) ----

def test_engine_routes_wired(monkeypatch):
    fastapi = pytest.importorskip("fastapi")  # noqa: F841
    from fastapi.testclient import TestClient
    from sndr.product_api.legacy.http_app import create_app

    monkeypatch.setattr(ec, "engine_status", lambda host=None, port=None, api_key=None: {"reachable": True, "version": "x", "models": ["m"]})
    monkeypatch.setattr(ec, "engine_metrics", lambda host=None, port=None: {"reachable": True, "kpis": {"requests_running": 1}})
    monkeypatch.setattr(ec, "engine_chat", lambda payload, host=None, port=None, api_key=None: {"reply": "ok", "usage": {"total_tokens": 5}})
    client = TestClient(create_app(allowed_origins=()))

    assert client.get("/api/v1/engine/status").json()["version"] == "x"
    assert client.get("/api/v1/engine/metrics").json()["kpis"]["requests_running"] == 1
    chat = client.post("/api/v1/engine/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert chat.status_code == 200 and chat.json()["reply"] == "ok"


def test_engine_chat_route_engine_down(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from sndr.product_api.legacy.http_app import create_app

    def boom(payload, host=None, port=None, api_key=None):
        raise OSError("Connection refused")
    monkeypatch.setattr(ec, "engine_chat", boom)
    client = TestClient(create_app(allowed_origins=()))
    assert client.post("/api/v1/engine/chat", json={"messages": [{"role": "user", "content": "x"}]}).status_code == 503


# ---- streaming + history ----

class _FakeStreamResp:
    def __init__(self, lines): self._lines = lines
    def __enter__(self): return iter(self._lines)
    def __exit__(self, *a): return False


def test_stream_chat_yields_deltas_then_done(monkeypatch):
    lines = [
        b'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        b'data: {"choices":[{"delta":{"content":"lo"}}]}',
        b'data: {"choices":[],"usage":{"completion_tokens":2}}',
        b"data: [DONE]",
    ]
    monkeypatch.setattr(ec.urllib.request, "urlopen", lambda req, timeout=120.0: _FakeStreamResp(lines))
    import json
    out = [json.loads(c) for c in ec.stream_chat({"messages": [{"role": "user", "content": "hi"}]})]
    assert out[0] == {"delta": "Hel"} and out[1] == {"delta": "lo"}
    assert out[-1]["done"] is True and out[-1]["tokens"] == 2 and "latency_ms" in out[-1]


def test_stream_chat_error_line(monkeypatch):
    def boom(req, timeout=120.0):
        raise OSError("Connection refused")
    monkeypatch.setattr(ec.urllib.request, "urlopen", boom)
    import json
    out = [json.loads(c) for c in ec.stream_chat({"messages": [{"role": "user", "content": "hi"}]})]
    assert "error" in out[0] and "refused" in out[0]["error"]


def test_stream_chat_requires_messages():
    with pytest.raises(ValueError):
        list(ec.stream_chat({"messages": []}))


def test_metrics_history_accumulates(monkeypatch):
    monkeypatch.delenv("SNDR_METRICS_URL", raising=False)
    monkeypatch.setattr(ec, "_get", lambda url, timeout=3.0: (200, SAMPLE_METRICS))
    ec._LAST_SCRAPE.clear(); ec._HISTORY.clear()
    first = ec.engine_metrics("hist-host", now=2000.0)
    second = ec.engine_metrics("hist-host", now=2003.0)
    assert len(second["history"]) == 2
    assert second["history"][-1]["kv_cache"] == 0.42
    assert second["history"][0]["ts"] == 2000.0


def test_engine_stream_and_download_routes(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from sndr.product_api.legacy.http_app import create_app

    monkeypatch.setattr(ec, "stream_chat", lambda payload, host=None, port=None, api_key=None: iter(['{"delta":"hi"}', '{"done":true}']))
    client = TestClient(create_app(allowed_origins=()))
    streamed = client.post("/api/v1/engine/chat/stream", json={"messages": [{"role": "user", "content": "x"}]})
    assert streamed.status_code == 200
    assert '{"delta":"hi"}' in streamed.text and '{"done":true}' in streamed.text

    # download requires a known model id (full behaviour covered in test_jobs_exec)
    assert client.post("/api/v1/models/download", json={}).status_code == 400
