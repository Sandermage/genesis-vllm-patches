# SPDX-License-Identifier: Apache-2.0
"""Tests for Hugging Face Hub search + HF repo download (no network)."""
from __future__ import annotations

import io
import json

import pytest

from sndr.product_api.legacy import hub


class _FakeResp:
    def __init__(self, payload):
        self._buf = io.BytesIO(json.dumps(payload).encode())
    def __enter__(self):
        return self._buf
    def __exit__(self, *a):
        return False


def test_search_models_distills_results(monkeypatch):
    sample = [
        {"id": "Qwen/Qwen3-8B", "downloads": 12000000, "likes": 1100, "pipeline_tag": "text-generation", "gated": False, "tags": ["transformers", "qwen3", "x", "y", "z", "w", "extra"]},
        {"modelId": "meta-llama/Llama-3-8B", "downloads": 5000000, "likes": 900, "gated": True, "tags": ["llama"]},
        {"likes": 1},  # no id -> dropped
    ]
    captured = {}
    def fake_urlopen(req, timeout=8.0, context=None):
        captured["url"] = req.full_url
        return _FakeResp(sample)
    monkeypatch.setattr(hub.urllib.request, "urlopen", fake_urlopen)

    out = hub.search_models("qwen", limit=10)
    assert "search=qwen" in captured["url"] and "filter=text-generation" in captured["url"]
    assert [m["id"] for m in out] == ["Qwen/Qwen3-8B", "meta-llama/Llama-3-8B"]
    assert out[0]["downloads"] == 12000000 and out[0]["gated"] is False
    assert len(out[0]["tags"]) == 6  # capped
    assert out[1]["id"] == "meta-llama/Llama-3-8B" and out[1]["gated"] is True


def test_search_respects_limit(monkeypatch):
    monkeypatch.setattr(hub.urllib.request, "urlopen", lambda req, timeout=8.0, context=None: _FakeResp([{"id": f"m/{i}"} for i in range(50)]))
    assert len(hub.search_models("x", limit=5)) == 5


# ---- HF download route ----

def test_hub_search_and_hf_download_routes(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    from sndr.product_api.legacy import jobs
    jobs._reset_state()
    from fastapi.testclient import TestClient
    from sndr.product_api.legacy.http_app import create_app

    monkeypatch.setattr(hub, "search_models", lambda q, limit=20: [{"id": "Qwen/Qwen3-8B", "downloads": 1, "likes": 1, "pipeline_tag": "text-generation", "gated": False, "tags": []}])
    client = TestClient(create_app(enable_apply=False, allowed_origins=()))

    search = client.get("/api/v1/models/hub/search?query=qwen&limit=5")
    assert search.status_code == 200 and search.json()["results"][0]["id"] == "Qwen/Qwen3-8B"

    # HF repo download (dry-run since apply off)
    dl = client.post("/api/v1/models/download", json={"repo_id": "Qwen/Qwen3-8B"})
    assert dl.status_code == 200
    assert any("huggingface-cli download Qwen/Qwen3-8B" in s for s in dl.json()["cli_mirror"])
    # injection / malformed repo id rejected
    assert client.post("/api/v1/models/download", json={"repo_id": "evil; rm -rf /"}).status_code == 400
    assert client.post("/api/v1/models/download", json={"repo_id": "noslash"}).status_code == 400
    # neither id provided
    assert client.post("/api/v1/models/download", json={}).status_code == 400
