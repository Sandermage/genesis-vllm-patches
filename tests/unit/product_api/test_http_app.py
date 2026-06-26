# SPDX-License-Identifier: Apache-2.0
"""Tests for the read-only SNDR Product API FastAPI app."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from sndr.product_api.legacy.http_app import create_app  # noqa: E402


def _client() -> TestClient:
    return TestClient(create_app(allowed_origins=()))


def test_blocking_routes_are_threadpooled_not_event_loop_blocking():
    """The container / system / k8s / proxmox read handlers do blocking I/O
    (docker subprocess, SSH, Proxmox HTTP). They MUST be plain `def` so
    Starlette runs them in its threadpool — an `async def` with a blocking
    body stalls the whole event loop and serializes every concurrent request,
    which is exactly what made the Containers/Virtualization views crawl.
    """
    import inspect as _inspect
    app = create_app(allowed_origins=())
    # Path prefixes whose handlers are pure-sync blocking I/O. Streaming /
    # websocket paths are intentionally excluded (they are genuinely async).
    must_be_sync = (
        "/api/v1/containers", "/api/v1/system/df", "/api/v1/system/networks",
        "/api/v1/k8s/", "/api/v1/proxmox/",
    )
    streaming_exempt = ("/logs/stream", "/terminal")
    offenders = []
    for route in app.routes:
        path = getattr(route, "path", "")
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None or not any(path.startswith(p) for p in must_be_sync):
            continue
        if any(s in path for s in streaming_exempt):
            continue
        if _inspect.iscoroutinefunction(endpoint):
            offenders.append(path)
    assert not offenders, (
        "these blocking handlers are async def and will block the event loop: "
        + ", ".join(sorted(offenders))
    )


def test_security_headers_present():
    """Defense-in-depth headers must be on every response."""
    r = _client().get("/api/v1/health")
    h = r.headers
    assert h.get("X-Content-Type-Options") == "nosniff"
    assert h.get("X-Frame-Options") == "DENY"
    assert h.get("Referrer-Policy") == "no-referrer"
    assert "camera=()" in (h.get("Permissions-Policy") or "")
    assert "script-src 'self'" in (h.get("Content-Security-Policy") or "")
    # Clickjacking / cross-window isolation.
    assert (h.get("Cross-Origin-Opener-Policy") or "").startswith("same-origin")


def test_hsts_only_when_request_is_https():
    """HSTS must NOT be sent over plain http (it would pin a name a homelab
    serves on http), but MUST appear once the daemon is fronted by TLS —
    detected via the X-Forwarded-Proto a reverse proxy sets."""
    client = _client()
    assert "Strict-Transport-Security" not in client.get("/api/v1/health").headers
    https = client.get("/api/v1/health", headers={"X-Forwarded-Proto": "https"})
    assert "max-age=" in (https.headers.get("Strict-Transport-Security") or "")


def test_force_hsts_env(monkeypatch):
    monkeypatch.setenv("SNDR_FORCE_HSTS", "1")
    r = _client().get("/api/v1/health")
    assert "max-age=" in (r.headers.get("Strict-Transport-Security") or "")


def test_health_reflects_apply_state_when_enabled():
    from fastapi.testclient import TestClient
    from sndr.product_api.legacy.http_app import create_app as _ca
    client = TestClient(_ca(allowed_origins=(), enable_apply=True))
    body = client.get("/api/v1/health").json()
    # With apply on, health must NOT claim read-only (the misleading hardcoded
    # True masked that mutating endpoints were live).
    assert body["read_only"] is False
    assert body["apply_enabled"] is True


def test_health_and_openapi_are_available():
    client = _client()

    health = client.get("/api/v1/health")
    assert health.status_code == 200
    # Default app has no apply -> read_only true, apply_enabled false.
    assert health.json()["read_only"] is True
    assert health.json()["apply_enabled"] is False

    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    paths = openapi.json()["paths"]
    assert "/api/v1/overview" in paths
    assert "/api/v1/configs/v2/catalog" in paths
    assert "/api/v1/configs/v2/preview" in paths
    assert "/api/v1/configs/v2/plan" in paths
    assert "/api/v1/presets/recommend" in paths
    assert "/api/v1/launch/plan" in paths
    assert "/api/v1/patches" in paths
    assert "/api/v1/patches/doctor" in paths
    assert "/api/v1/patches/{patch_id}/explain" in paths
    assert "/api/v1/configs/v2/apply" in paths
    assert "/api/v1/configs/v2/user-presets" in paths
    assert "/api/v1/patches/bundles" in paths
    assert "/api/v1/patches/diff-upstream" in paths
    assert "/api/v1/proof/status" in paths
    assert "/api/v1/configs/v2/layer/{kind}/{layer_id}" in paths
    assert "/api/v1/configs/v2/layer/apply" in paths
    assert "/api/v1/doctor" in paths
    assert "/api/v1/services/plan" in paths
    assert "/api/v1/environment" in paths
    assert "/api/v1/services/apply" in paths
    assert "/api/v1/jobs" in paths
    assert "/api/v1/jobs/{job_id}" in paths
    assert "/api/v1/hosts" in paths
    assert "/api/v1/hosts/{host_id}" in paths
    assert "/api/v1/auth/status" in paths
    assert "/api/v1/memory/fit" in paths
    assert "/api/v1/models/cache" in paths
    assert "/api/v1/events" in paths
    assert "/api/v1/events/recent" in paths
    assert "/api/v1/reports/bundle" in paths
    assert "/api/v1/launch/apply" in paths
    assert "/api/v1/bench/run" in paths
    assert "/api/v1/evidence/attach" in paths


def test_chat_retrieve_grounds_in_project_knowledge():
    client = _client()
    resp = client.get("/api/v1/chat/retrieve", params={"query": "PN95 tiered kv cache", "k": 4})
    assert resp.status_code == 200
    body = resp.json()
    assert body["matched"] >= 1
    assert len(body["docs"]) <= 4
    assert any("pn95" in d["id"].lower() for d in body["docs"])
    top = body["docs"][0]
    assert set(top) >= {"id", "kind", "title", "ref", "snippet", "score"}


def test_chat_retrieve_empty_query_is_clean():
    client = _client()
    resp = client.get("/api/v1/chat/retrieve", params={"query": "   "})
    assert resp.status_code == 200
    assert resp.json() == {"query": "", "matched": 0, "docs": []}


def test_chat_retrieve_post_supports_notes_vault(tmp_path):
    from sndr.product_api.legacy import chat_rag

    chat_rag.reset_cache()
    (tmp_path / "notes.md").write_text(
        "# Homelab GPU notes\n\nThe A5000 idles at 25W; PN95 offload helps long context.\n",
        encoding="utf-8",
    )
    client = _client()
    resp = client.post(
        "/api/v1/chat/retrieve",
        json={"query": "A5000 idle watts homelab", "k": 5, "project": False, "vaults": [str(tmp_path)]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["matched"] >= 1
    assert all(d["kind"] == "note" for d in body["docs"])


def test_chat_rag_preview_validates_path(tmp_path):
    client = _client()
    (tmp_path / "a.md").write_text("# A\n\nbody\n", encoding="utf-8")
    ok = client.post("/api/v1/chat/rag/preview", json={"path": str(tmp_path)})
    assert ok.status_code == 200 and ok.json()["ok"] is True and ok.json()["files"] >= 1
    bad = client.post("/api/v1/chat/rag/preview", json={"path": str(tmp_path / "nope")})
    assert bad.status_code == 200 and bad.json()["ok"] is False


def test_hosts_ssh_check_persists_password_and_reports(monkeypatch, tmp_path):
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    monkeypatch.setenv("SNDR_SECRETS_BACKEND", "file")
    from sndr.product_api.legacy import secrets_store, ssh_client
    secrets_store.reset_backend_cache()

    captured = {}
    def fake_check(target, *, timeout=8.0):
        captured["target"] = target
        return {"available": True, "ssh_ok": True, "sftp_ok": True, "latency_ms": 5.0,
                "banner": "SSH-2.0", "uname": "Linux 6.2", "error": None}
    monkeypatch.setattr(ssh_client, "check_connectivity", fake_check)

    client = _client()
    resp = client.post("/api/v1/hosts/ssh-check", json={
        "host": "192.0.2.10", "host_id": "prod", "user": "operator",
        "auth_method": "password", "password": "s3cret", "ssh_port": 22,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["ssh_ok"] is True and body["sftp_ok"] is True
    # The password reached the connectivity check but is never echoed back.
    assert "password" not in body
    # And it was persisted (encrypted) for reuse, keyed by the host id.
    assert secrets_store.get_secret("ssh:prod") == "s3cret"
    # The host list now reports a stored password without leaking it.
    client.post("/api/v1/hosts", json={"label": "Prod", "host": "192.0.2.10", "id": "prod"})
    hosts = client.get("/api/v1/hosts").json()["hosts"]
    prod = next(h for h in hosts if h["id"] == "prod")
    assert prod["has_ssh_password"] is True
    assert "password" not in prod
    # forget clears it.
    forgot = client.post("/api/v1/hosts/ssh-check", json={"host_id": "prod", "forget_password": True})
    assert forgot.json()["forgot"] is True
    assert secrets_store.get_secret("ssh:prod") is None


def test_hosts_fetch_api_key_stores_on_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    from sndr.product_api.legacy import host_profiles, ssh_client

    monkeypatch.setattr(
        ssh_client, "discover_api_key",
        lambda target, **kw: {"available": True, "found": True, "key": "genesis-local", "source": "container:vllm-x", "error": None},
    )
    client = _client()
    client.post("/api/v1/hosts", json={"label": "Prod", "host": "192.0.2.10", "id": "prod", "ssh_user": "operator"})
    resp = client.post("/api/v1/hosts/fetch-api-key", json={"host_id": "prod"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is True and body["source"] == "container:vllm-x"
    assert "key" not in body and "key_masked" in body  # masked, not leaked
    # The discovered key is stored ENCRYPTED in the secrets store (never on the
    # profile / on disk) and surfaced only as a boolean presence flag.
    from sndr.product_api.legacy import secrets_store
    assert secrets_store.get_secret("apikey:prod") == "genesis-local"
    prof = next(p for p in host_profiles.list_host_profiles() if p.id == "prod")
    assert prof.api_key == ""  # not on the profile
    assert client.get("/api/v1/hosts").json()["hosts"][0]["has_api_key"] is True
    # Unknown host -> 404.
    assert client.post("/api/v1/hosts/fetch-api-key", json={"host_id": "nope"}).status_code == 404


def test_terminal_gate_enforces_apply_and_known_host():
    # The PTY terminal's security policy is unit-tested directly (the starlette
    # 1.0 TestClient cannot drive websockets); the transport is verified live.
    from sndr.product_api.legacy.http_app import terminal_gate

    # apply OFF -> refused regardless of host.
    off = terminal_gate(False, True, {"prod"}, "prod")
    assert off and off["type"] == "error" and "SNDR_ENABLE_APPLY" in off["data"]
    # apply ON but paramiko missing -> refused.
    no_pm = terminal_gate(True, False, {"prod"}, "prod")
    assert no_pm and "paramiko" in no_pm["data"]
    # apply ON, unknown host -> refused.
    unknown = terminal_gate(True, True, {"prod"}, "ghost")
    assert unknown and "unknown host" in unknown["data"]
    # apply ON, known host, paramiko present -> allowed (None).
    assert terminal_gate(True, True, {"prod"}, "prod") is None


def test_calc_kv_endpoint_breakdown_and_curve():
    client = _client()
    resp = client.post("/api/v1/calc/kv", json={
        "model_id": "qwen3.6-27b-int4", "context": 16384, "tp": 2, "gpu_count": 2,
        "gpu_vram_mib": 24564, "util": 0.9, "kv_dtype": "fp8",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"]["weights_per_gpu_mib"] > 0 and body["result"]["max_context"] > 0
    assert "fp8" in body["by_dtype"] and "fp16" in body["by_dtype"]
    # fp16 KV achieves less context than fp8.
    assert body["by_dtype"]["fp16"] < body["by_dtype"]["fp8"]
    assert len(body["curve"]) > 5
    # Max context scales with tensor-parallel width (more GPUs → more context).
    assert set(body["by_tp"]) == {"1", "2", "4", "8"}
    assert body["by_tp"]["8"] >= body["by_tp"]["1"]
    models = client.get("/api/v1/calc/models").json()
    assert "qwen3.6-27b-int4" in models["models"]


def test_calc_kv_rejects_bad_input_with_400():
    """Malformed numeric input is a clean 400, not an unhandled 500."""
    client = _client()
    bad = client.post("/api/v1/calc/kv", json={"model_id": "qwen3.6-27b-int4", "context": "abc"})
    assert bad.status_code == 400
    neg = client.post("/api/v1/calc/kv", json={"model_id": "qwen3.6-27b-int4", "context": -1, "tp": -2})
    assert neg.status_code == 400


def test_baselines_save_diff_delete(monkeypatch, tmp_path):
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    client = _client()
    base_result = {"label": "base", "scenarios": [{"name": "code", "metrics": {"tps": 120.0, "ttft_ms": 100.0}}]}
    saved = client.post("/api/v1/baselines", json={"result": base_result, "label": "prod-27b"}).json()
    assert saved["id"]
    assert any(b["id"] == saved["id"] for b in client.get("/api/v1/baselines").json()["baselines"])
    cur = {"label": "cur", "scenarios": [{"name": "code", "metrics": {"tps": 100.0, "ttft_ms": 100.0}}]}
    diff = client.post("/api/v1/baselines/diff", json={"current": cur, "baseline_id": saved["id"], "threshold_pct": 5}).json()
    assert diff["has_regression"] is True and diff["exit_code"] == 3
    assert client.delete(f"/api/v1/baselines/{saved['id']}").json()["deleted"] is True


def test_launch_bench_evidence_dry_run_by_default():
    client = _client()
    la = client.post("/api/v1/launch/apply", json={"preset_id": "prod-qwen3.6-35b-balanced", "runtime_target": "docker"})
    assert la.status_code == 200 and la.json()["dry_run"] is True
    br = client.post("/api/v1/bench/run", json={"preset_id": "prod-qwen3.6-35b-balanced"})
    assert br.status_code == 200 and br.json()["dry_run"] is True
    ev = client.post("/api/v1/evidence/attach", json={"preset_id": "prod-qwen3.6-35b-balanced"})
    assert ev.status_code == 200 and ev.json()["dry_run"] is True
    # Missing preset_id -> 400.
    assert client.post("/api/v1/bench/run", json={}).status_code == 400


def test_launch_apply_mutating_needs_confirm_when_enabled():
    from fastapi.testclient import TestClient as _TC

    from sndr.product_api.legacy.http_app import create_app as _ca

    client = _TC(_ca(allowed_origins=(), enable_apply=True))
    resp = client.post(
        "/api/v1/launch/apply",
        json={"preset_id": "prod-qwen3.6-35b-balanced", "runtime_target": "docker", "transport": "local"},
    )
    assert resp.status_code == 409  # launch is mutating; needs confirm


def test_reports_bundle_writes_operator_local(tmp_path, monkeypatch):
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    client = _client()
    resp = client.post("/api/v1/reports/bundle", json={"report_type": "catalog"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["report_type"] == "catalog"
    assert str(tmp_path) in body["bundle_dir"]
    assert "snapshot.json" in body["files"]
    # Bad report type rejected.
    bad = client.post("/api/v1/reports/bundle", json={"report_type": "nope"})
    assert bad.status_code == 400


def test_events_recent_and_stream():
    client = _client()
    # Generate an event via a dry-run apply, then poll the JSON feed.
    client.post(
        "/api/v1/services/apply",
        json={"preset_id": "prod-qwen3.6-35b-balanced", "action": "status", "runtime_target": "docker_compose"},
    )
    recent = client.get("/api/v1/events/recent")
    assert recent.status_code == 200
    body = recent.json()
    assert isinstance(body["events"], list)
    assert body["last_seq"] >= 1
    assert any(e["kind"] == "job" for e in body["events"])
    # since_seq filtering returns only newer events.
    newer = client.get("/api/v1/events/recent", params={"since_seq": body["last_seq"]})
    assert newer.json()["events"] == []
    # The live SSE stream (text/event-stream) is verified against the real
    # uvicorn daemon via curl; TestClient cannot tear down an infinite
    # generator cleanly, so it is not exercised here.


def test_models_cache_endpoint():
    client = _client()
    resp = client.get("/api/v1/models/cache")
    assert resp.status_code == 200
    body = resp.json()
    assert "host" in body
    assert body["total"] == len(body["models"])
    assert body["present_count"] == sum(1 for m in body["models"] if m["present"])


def test_memory_fit_endpoint_reports_compatibility():
    client = _client()
    resp = client.get(
        "/api/v1/memory/fit",
        params={
            "model_id": "qwen3.6-35b-a3b-fp8",
            "hardware_id": "a5000-2x-24gbvram-16cpu-128gbram",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["compatible"] is True
    assert any(c["id"] == "gpu_count" for c in body["checks"])
    assert body["vram"]["model_min_mib"] >= 1


def test_memory_fit_unknown_returns_404():
    client = _client()
    resp = client.get(
        "/api/v1/memory/fit",
        params={"model_id": "nope", "hardware_id": "nope"},
    )
    assert resp.status_code == 404


def test_daemon_serves_static_ui_when_present(tmp_path, monkeypatch):
    # API routes still win; non-API paths serve the built UI.
    (tmp_path / "index.html").write_text("<!doctype html><title>SNDR</title>", encoding="utf-8")
    monkeypatch.setenv("SNDR_GUI_STATIC", str(tmp_path))
    from fastapi.testclient import TestClient as _TC

    from sndr.product_api.legacy.http_app import create_app as _ca

    client = _TC(_ca(allowed_origins=()))
    # API route takes precedence over the static mount.
    assert client.get("/api/v1/health").status_code == 200
    # Root serves the UI index.
    root = client.get("/")
    assert root.status_code == 200
    assert "SNDR" in root.text


def test_daemon_api_only_without_static():
    # No SNDR_GUI_STATIC and no build dir in the test env -> API-only, root 404.
    client = _client()
    assert client.get("/api/v1/health").status_code == 200


def test_services_apply_dry_run_by_default():
    client = _client()
    resp = client.post(
        "/api/v1/services/apply",
        json={"preset_id": "prod-qwen3.6-35b-balanced", "action": "status", "runtime_target": "docker"},
    )
    assert resp.status_code == 200
    assert resp.json()["dry_run"] is True


def test_auth_status_reports_apply_disabled_by_default():
    client = _client()
    assert client.get("/api/v1/auth/status").json()["apply_enabled"] is False


def test_apply_enabled_gates_mutation_without_confirm():
    from fastapi.testclient import TestClient as _TC

    from sndr.product_api.legacy.http_app import create_app as _ca

    client = _TC(_ca(allowed_origins=(), enable_apply=True))
    assert client.get("/api/v1/auth/status").json()["apply_enabled"] is True
    # Mutating action without confirm -> 409 (no execution).
    resp = client.post(
        "/api/v1/services/apply",
        json={"preset_id": "prod-qwen3.6-35b-balanced", "action": "restart", "runtime_target": "docker", "transport": "local"},
    )
    assert resp.status_code == 409


def test_no_token_auth_by_default():
    client = _client()
    assert client.get("/api/v1/auth/status").json()["auth_required"] is False
    assert client.get("/api/v1/capabilities").status_code == 200


def test_token_auth_when_configured(monkeypatch):
    monkeypatch.setenv("SNDR_GUI_TOKEN", "s3cret")
    client = TestClient(create_app(allowed_origins=()))
    assert client.get("/api/v1/health").status_code == 200  # health stays open
    assert client.get("/api/v1/auth/status").json()["auth_required"] is True
    assert client.get("/api/v1/overview").status_code == 401  # gated
    assert client.get("/api/v1/overview", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.get("/api/v1/overview", headers={"X-SNDR-Token": "s3cret"}).status_code == 200
    assert client.get("/api/v1/overview", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_cors_allows_local_vite_fallback_port():
    client = _client()
    resp = client.get(
        "/api/v1/health",
        headers={"Origin": "http://127.0.0.1:5174"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://127.0.0.1:5174"


def test_overview_and_catalog_summary_endpoints(tmp_path, monkeypatch):
    # Isolate $SNDR_HOME so the count reflects only the builtin catalog
    # (deterministic regardless of operator-local presets / test order).
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    client = _client()

    overview = client.get("/api/v1/overview")
    assert overview.status_code == 200
    payload = overview.json()
    # chat-K3 promotion (2026-06-01): 21 → 23 preset aliases.
    # Gemma-31B kv-auto profile (2026-06-19, ea33b8e0): +1 → 24.
    # Canonical-config reorg (2026-06): 24 → 14 (archived 11 test/
    # experimental presets, added prod-diffusiongemma-tp2).
    assert payload["catalog"]["presets_count"] == 14
    assert payload["capabilities"]["platform"]["sndr_core_version"]

    summary = client.get("/api/v1/catalog/summary")
    assert summary.status_code == 200
    # Every builtin preset carries a card.
    # chat-K3 promotion (2026-06-01): 21 → 23 (both new presets ship
    # with operator cards from the start). Gemma-31B kv-auto (2026-06-19): +1 → 24.
    # Canonical-config reorg (2026-06): 24 → 14 (new diffusiongemma preset
    # also carries an operator card).
    assert summary.json()["preset_cards_count"] == 14


def test_presets_endpoints_are_read_only_json_views():
    client = _client()

    listed = client.get(
        "/api/v1/presets",
        params={"status": "production_candidate"},
    )
    assert listed.status_code == 200
    # chat-K3 promotion (2026-06-01): both new presets land as
    # production_candidate → +2 → 14 → 16.
    # Gemma-31B kv-auto preset (2026-06-19, ea33b8e0) also production_candidate → +1 → 17.
    # Canonical-config reorg (2026-06): archived presets dropped the
    # production_candidate-filter count to 8 (kept canonical+sibling prod
    # presets; the new diffusiongemma preset is experimental).
    assert listed.json()["matched"] == 8

    preset = client.get("/api/v1/presets/prod-qwen3.6-35b-balanced")
    assert preset.status_code == 200
    assert preset.json()["card"]["routing_family"] == "qwen3_6_35b_a3b_fp8"

    missing = client.get("/api/v1/presets/not-a-real-preset")
    assert missing.status_code == 404


def test_v2_config_catalog_and_preview_endpoints():
    client = _client()

    catalog = client.get("/api/v1/configs/v2/catalog")
    assert catalog.status_code == 200
    payload = catalog.json()
    assert any(item["id"] == "qwen3.6-35b-a3b-fp8" for item in payload["models"])
    assert any(item["id"] == "qwen3.6-35b-multiconc" for item in payload["profiles"])

    preview = client.get(
        "/api/v1/configs/v2/preview",
        params={
            "model_id": "qwen3.6-35b-a3b-fp8",
            "hardware_id": "a5000-2x-24gbvram-16cpu-128gbram",
            "profile_id": "qwen3.6-35b-multiconc",
            "runtime": "docker",
        },
    )
    assert preview.status_code == 200
    preview_payload = preview.json()
    assert preview_payload["compatible"] is True
    assert preview_payload["composed"]["max_num_seqs"] == 8
    assert "profile: qwen3.6-35b-multiconc" in preview_payload["draft_yaml"]

    plan = client.post(
        "/api/v1/configs/v2/plan",
        json={
            "preset_id": "gui-draft-qwen3.6-35b-multiconc",
            "model_id": "qwen3.6-35b-a3b-fp8",
            "hardware_id": "a5000-2x-24gbvram-16cpu-128gbram",
            "profile_id": "qwen3.6-35b-multiconc",
            "runtime": "docker",
        },
    )
    assert plan.status_code == 200
    plan_payload = plan.json()
    assert plan_payload["read_only"] is True
    assert plan_payload["apply_enabled"] is False
    assert plan_payload["valid"] is True
    assert any(line.startswith("+profile: qwen3.6-35b-multiconc") for line in plan_payload["diff_lines"])


def test_preset_recommend_and_explain_endpoints():
    client = _client()

    recommend = client.get(
        "/api/v1/presets/recommend",
        params={
            "workload": "free_chat",
            "hardware": "a5000-2x-24gbvram-16cpu-128gbram",
            "concurrency": 8,
            "top": 5,
        },
    )
    assert recommend.status_code == 200
    ids = [row["id"] for row in recommend.json()["results"]]
    assert "prod-qwen3.6-35b-multiconc" in ids
    assert "prod-gemma4-26b-mtp-k4" not in ids

    bad_workload = client.get(
        "/api/v1/presets/recommend",
        params={"workload": "freechat"},
    )
    assert bad_workload.status_code == 400

    explain = client.get("/api/v1/presets/prod-qwen3.6-35b-multiconc/explain")
    assert explain.status_code == 200
    assert explain.json()["composed"]["max_num_seqs"] == 8


def test_patch_inventory_and_doctor_endpoints():
    client = _client()

    listed = client.get("/api/v1/patches")
    assert listed.status_code == 200
    payload = listed.json()
    assert payload["total"] >= 200
    assert payload["matched"] == payload["total"]
    assert "lifecycle_counts" in payload["summary"]
    assert payload["patches"][0]["patch_id"]

    stable = client.get("/api/v1/patches", params={"lifecycle": "stable"})
    assert stable.status_code == 200
    assert stable.json()["matched"] >= 1

    doctor = client.get("/api/v1/patches/doctor")
    assert doctor.status_code == 200
    doctor_payload = doctor.json()
    assert doctor_payload["registry_size"] == payload["total"]
    assert "coverage" in doctor_payload

    explain = client.get(f"/api/v1/patches/{payload['patches'][0]['patch_id']}/explain")
    assert explain.status_code == 200
    explain_payload = explain.json()
    assert explain_payload["patch_id"] == payload["patches"][0]["patch_id"]
    assert "spec" in explain_payload

    missing = client.get("/api/v1/patches/not-real/explain")
    assert missing.status_code == 404


def test_config_apply_writes_operator_local_and_lists_user_presets(monkeypatch, tmp_path):
    monkeypatch.setenv("SNDR_MODEL_CONFIG_DIR", str(tmp_path))
    client = _client()

    body = {
        "preset_id": "gui-draft-qwen3.6-35b-multiconc",
        "model_id": "qwen3.6-35b-a3b-fp8",
        "hardware_id": "a5000-2x-24gbvram-16cpu-128gbram",
        "profile_id": "qwen3.6-35b-multiconc",
        "runtime": "docker",
    }
    applied = client.post("/api/v1/configs/v2/apply", json=body)
    assert applied.status_code == 200
    payload = applied.json()
    assert payload["status"] == "applied"
    assert payload["written"] is True

    user_presets = client.get("/api/v1/configs/v2/user-presets")
    assert user_presets.status_code == 200
    up = user_presets.json()
    assert up["count"] == 1
    assert up["presets"][0]["id"] == "gui-draft-qwen3.6-35b-multiconc"

    conflict = client.post(
        "/api/v1/configs/v2/apply",
        json={**body, "expected_plan_id": "cfgplan_stale000000"},
    )
    assert conflict.status_code == 409

    blocked = client.post(
        "/api/v1/configs/v2/apply",
        json={
            "preset_id": "gui-draft-bad",
            "model_id": "gemma-4-26b-a4b-it-awq",
            "hardware_id": "a5000-2x-24gbvram-16cpu-128gbram",
            "profile_id": "qwen3.6-35b-multiconc",
        },
    )
    assert blocked.status_code == 422


def test_host_profiles_crud(monkeypatch, tmp_path):
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    client = _client()
    assert client.get("/api/v1/hosts").json()["hosts"] == []
    created = client.post("/api/v1/hosts", json={"label": "GPU 01", "host": "gpu-01", "ssh_target": "u@gpu-01"})
    assert created.status_code == 200
    hid = created.json()["id"]
    assert any(h["id"] == hid for h in client.get("/api/v1/hosts").json()["hosts"])
    assert client.delete(f"/api/v1/hosts/{hid}").json()["deleted"] is True
    assert client.post("/api/v1/hosts", json={"notes": "x"}).status_code == 400


def test_service_apply_creates_dry_run_job_and_lists():
    client = _client()
    applied = client.post("/api/v1/services/apply", json={"preset_id": "prod-qwen3.6-35b-multiconc", "action": "start"})
    assert applied.status_code == 200
    job = applied.json()
    assert job["dry_run"] is True
    assert job["kind"] == "service.start"
    assert job["steps"]
    job_id = job["job_id"]

    listed = client.get("/api/v1/jobs")
    assert listed.status_code == 200
    assert any(j["job_id"] == job_id for j in listed.json()["jobs"])

    got = client.get(f"/api/v1/jobs/{job_id}")
    assert got.status_code == 200
    assert got.json()["job_id"] == job_id
    assert client.get("/api/v1/jobs/nope").status_code == 404

    assert client.post("/api/v1/services/apply", json={"preset_id": "nope", "action": "start"}).status_code == 404


def test_service_plan_endpoint_is_read_only():
    client = _client()
    resp = client.get("/api/v1/services/plan", params={"preset_id": "prod-qwen3.6-35b-multiconc", "action": "start"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["mutating"] is True
    assert payload["actionable"] is False
    assert payload["steps"]
    assert client.get("/api/v1/services/plan", params={"preset_id": "prod-qwen3.6-35b-multiconc", "action": "nope"}).status_code == 400
    assert client.get("/api/v1/services/plan", params={"preset_id": "nope", "action": "status"}).status_code == 404


def test_doctor_endpoint_returns_categorised_findings():
    client = _client()
    resp = client.get("/api/v1/doctor")
    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload["findings"]) >= 4
    assert "environment" in payload["categories"]
    assert sum(payload["summary"].values()) == len(payload["findings"])
    assert payload["findings"][0]["severity"] in {"ok", "info", "warning", "blocked"}


def test_v2_layer_apply_writes_operator_local(monkeypatch, tmp_path):
    monkeypatch.setenv("SNDR_MODEL_CONFIG_DIR", str(tmp_path))
    client = _client()
    ok = client.post("/api/v1/configs/v2/layer/apply", json={
        "kind": "model", "layer_id": "gui-edit-x",
        "yaml_text": "schema_version: 2\nkind: model\nid: gui-edit-x\n",
    })
    assert ok.status_code == 200
    assert ok.json()["status"] == "applied"
    assert (tmp_path / "model" / "gui-edit-x.yaml").is_file()

    bad = client.post("/api/v1/configs/v2/layer/apply", json={"kind": "widget", "layer_id": "x", "yaml_text": "a: 1"})
    assert bad.status_code == 422


def test_v2_layer_endpoint_returns_full_definition():
    client = _client()

    model = client.get("/api/v1/configs/v2/layer/model/qwen3.6-35b-a3b-fp8")
    assert model.status_code == 200
    payload = model.json()
    assert payload["kind"] == "model"
    assert payload["definition"]["capabilities"]["attention_arch"]
    assert isinstance(payload["definition"]["patches"], dict)

    assert client.get("/api/v1/configs/v2/layer/widget/x").status_code == 400
    assert client.get("/api/v1/configs/v2/layer/model/not-a-model").status_code == 404


def test_bundles_diff_upstream_and_proof_status_endpoints():
    client = _client()

    bundles = client.get("/api/v1/patches/bundles")
    assert bundles.status_code == 200
    names = [b["name"] for b in bundles.json()["bundles"]]
    assert "attention_tq_multi_query" in names

    one = client.get("/api/v1/patches/bundles/attention_tq_multi_query")
    assert one.status_code == 200
    assert one.json()["umbrella_flag"] == "BUNDLE_ATTENTION_TQ_MULTI_QUERY"
    assert client.get("/api/v1/patches/bundles/not-a-bundle").status_code == 404

    diff = client.get("/api/v1/patches/diff-upstream")
    assert diff.status_code == 200
    diff_payload = diff.json()
    assert "merged_upstream" in diff_payload
    assert "has_upstream_pr" in diff_payload

    proof = client.get("/api/v1/proof/status")
    assert proof.status_code == 200
    assert "available" in proof.json()


def test_launch_plan_endpoint_is_read_only_json_contract():
    client = _client()

    response = client.get(
        "/api/v1/launch/plan",
        params={
            "preset_id": "prod-qwen3.6-35b-multiconc",
            "runtime_target": "docker_compose",
            "patch_policy": "safe",
            "host": "gpu-build-01",
            "mode": "remote",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["preset_id"] == "prod-qwen3.6-35b-multiconc"
    # Lifecycle API implemented → plan is actionable (apply still gated by
    # --enable-apply + confirm at the apply endpoint).
    assert payload["actionable"] is True
    assert {artifact["kind"] for artifact in payload["artifacts"]} == {
        "compose",
        "systemd",
        "commands",
        "env",
    }
    assert payload["endpoints"][0]["url"] == "http://gpu-build-01:8000/v1"

    missing = client.get(
        "/api/v1/launch/plan",
        params={"preset_id": "not-a-real-preset"},
    )
    assert missing.status_code == 404


# ─── Container management endpoints ───────────────────────────────────

from sndr.product_api.legacy import container_ops as _co  # noqa: E402


class _FakeControl:
    """Stand-in for a ContainerControl backend — records mutating calls."""

    def __init__(self) -> None:
        self.calls: list = []

    def list_managed(self):
        return [_co.ManagedContainer(name="vllm-35b-prod", id="abc", image="img",
                                     state="running", status="Up 2h", ports="8101->8101/tcp",
                                     created="", labels={})]

    def inspect(self, name):
        return {"Name": name, "State": {"Running": True}, "Config": {"Image": "vllm/vllm-openai:nightly"}}

    def top(self, name):
        return {"titles": ["PID", "CMD"], "processes": [["1", "python3"]]}

    def changes(self, name):
        return [{"kind": "added", "path": "/tmp/x"}]

    def pull(self, name):
        self.calls.append(("pull", name))
        return {"image": "vllm/vllm-openai:nightly", "output": "Digest: sha256:x"}

    def list_dir(self, name, path):
        self.calls.append(("ls", name, path))
        return {"path": path, "entries": [{"name": "etc", "is_dir": True}]}

    def read_file(self, name, path, **kw):
        return {"path": path, "content": "data", "truncated": False}

    def stream_logs(self, name, *, tail=200):
        yield "hello\n"
        yield ""          # heartbeat tick
        yield "world\n"

    def system_df(self):
        return {"types": [{"type": "Images", "total_count": 2, "active": 1, "size": 1000, "reclaimable": 400}], "total_size": 1000}

    def scan_image(self, name):
        return {"available": True, "scanner": "grype", "image": "vllm/vllm-openai:nightly",
                "counts": {"critical": 1, "high": 2, "medium": 0, "low": 0, "negligible": 0, "unknown": 0}, "total": 3}

    def update_settings(self, name, *, cpus=None, memory=None, restart_policy=None):
        self.calls.append(("settings", name, cpus, memory, restart_policy))
        return {"updated": True}

    def connect_network(self, name, network):
        self.calls.append(("net-connect", name, network)); return {"ok": True, "network": network, "action": "connect"}

    def disconnect_network(self, name, network):
        self.calls.append(("net-disconnect", name, network)); return {"ok": True, "network": network, "action": "disconnect"}

    def list_networks(self):
        return [{"name": "bridge", "driver": "bridge", "scope": "local"}]

    def engine_health(self, name):
        return {"reachable": True, "port": 8101, "status_code": 200}

    def logs(self, name, *, tail=200):
        return f"log of {name} (tail={tail})"

    def stats(self, name):
        return {"cpu_pct": 12.5, "mem_usage": 100, "mem_limit": 1000, "mem_pct": 10.0}

    def list_stats(self):
        return {"vllm-35b-prod": {"cpu_pct": 12.5, "mem_usage": 100, "mem_limit": 1000, "mem_pct": 10.0}}

    def start(self, name): self.calls.append(("start", name))
    def stop(self, name): self.calls.append(("stop", name))
    def restart(self, name): self.calls.append(("restart", name))

    def exec(self, name, argv, **kw):
        self.calls.append(("exec", name, list(argv)))
        return _co.ExecResult(exit_code=0, stdout="done", stderr="")


def _patch_local(monkeypatch, control):
    """Pretend the docker socket is mounted and hand out a fake socket control."""
    monkeypatch.setattr("sndr.deps.checkers._docker_socket_present", lambda *a, **k: True)
    monkeypatch.setattr(_co, "SocketContainerControl", lambda **kw: control)


def test_containers_list_requires_mounted_socket(monkeypatch):
    monkeypatch.setattr("sndr.deps.checkers._docker_socket_present", lambda *a, **k: False)
    resp = _client().get("/api/v1/containers")
    assert resp.status_code == 503
    assert "socket" in resp.json()["detail"].lower()


def test_containers_list_scoped(monkeypatch):
    _patch_local(monkeypatch, _FakeControl())
    resp = _client().get("/api/v1/containers")
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "socket"
    assert [c["name"] for c in body["containers"]] == ["vllm-35b-prod"]


def test_containers_stats_batch_not_shadowed(monkeypatch):
    # /containers/stats must resolve to the batch endpoint, NOT /containers/{name="stats"}.
    _patch_local(monkeypatch, _FakeControl())
    r = _client().get("/api/v1/containers/stats")
    assert r.status_code == 200
    assert "vllm-35b-prod" in r.json()["stats"]


def test_container_logs_and_stats(monkeypatch):
    _patch_local(monkeypatch, _FakeControl())
    c = _client()
    logs = c.get("/api/v1/containers/vllm-35b-prod/logs", params={"tail": 50})
    assert logs.status_code == 200 and "tail=50" in logs.json()["logs"]
    stats = c.get("/api/v1/containers/vllm-35b-prod/stats")
    assert stats.status_code == 200 and stats.json()["stats"]["cpu_pct"] == 12.5


def test_container_action_blocked_when_apply_off(monkeypatch):
    fake = _FakeControl()
    _patch_local(monkeypatch, fake)
    # default app: apply OFF
    resp = _client().post("/api/v1/containers/vllm-35b-prod/action", json={"action": "restart", "confirm": True})
    assert resp.status_code == 403
    assert fake.calls == []  # never reached the backend


def test_container_action_runs_when_apply_on(monkeypatch):
    fake = _FakeControl()
    _patch_local(monkeypatch, fake)
    client = TestClient(create_app(allowed_origins=(), enable_apply=True))
    # confirm required
    assert client.post("/api/v1/containers/vllm-35b-prod/action", json={"action": "restart"}).status_code == 400
    ok = client.post("/api/v1/containers/vllm-35b-prod/action", json={"action": "restart", "confirm": True})
    assert ok.status_code == 200 and ok.json()["action"] == "restart"
    assert ("restart", "vllm-35b-prod") in fake.calls
    # Audit: the mutating op is recorded in the persisted event feed.
    events = client.get("/api/v1/events/recent").json()["events"]
    assert any(e["kind"] == "container.restart" and e["detail"].get("container") == "vllm-35b-prod" for e in events)


def test_container_action_rejects_foreign_container(monkeypatch):
    fake = _FakeControl()
    # real control would raise NotManagedError; simulate by using the real socket
    # control path is hard here, so assert the unsupported-name guard via a real
    # backend call: use a control that raises.
    class _Guarding(_FakeControl):
        def restart(self, name):
            raise _co.NotManagedError(f"container not managed by SNDR: {name!r}")
    _patch_local(monkeypatch, _Guarding())
    client = TestClient(create_app(allowed_origins=(), enable_apply=True))
    resp = client.post("/api/v1/containers/postgres/action", json={"action": "restart", "confirm": True})
    assert resp.status_code == 403


def test_container_exec_requires_exec_flag(monkeypatch):
    fake = _FakeControl()
    _patch_local(monkeypatch, fake)
    client = TestClient(create_app(allowed_origins=(), enable_apply=True))
    # apply on but EXEC off → blocked
    blocked = client.post("/api/v1/containers/vllm-35b-prod/exec", json={"argv": ["ls"], "confirm": True})
    assert blocked.status_code == 403
    assert fake.calls == []
    # turn EXEC on
    monkeypatch.setenv("SNDR_ENABLE_EXEC", "1")
    ok = client.post("/api/v1/containers/vllm-35b-prod/exec", json={"argv": ["echo", "hi"], "confirm": True})
    assert ok.status_code == 200 and ok.json()["exit_code"] == 0
    assert ("exec", "vllm-35b-prod", ["echo", "hi"]) in fake.calls


def test_container_exec_validates_argv(monkeypatch):
    _patch_local(monkeypatch, _FakeControl())
    monkeypatch.setenv("SNDR_ENABLE_EXEC", "1")
    client = TestClient(create_app(allowed_origins=(), enable_apply=True))
    bad = client.post("/api/v1/containers/vllm-35b-prod/exec", json={"argv": "ls", "confirm": True})
    assert bad.status_code == 400


def test_host_containers_unknown_host_404():
    resp = _client().get("/api/v1/hosts/does-not-exist/containers")
    assert resp.status_code == 404


def test_alerts_config_get_and_gated_set():
    c = _client()
    cfg = c.get("/api/v1/alerts/config")
    assert cfg.status_code == 200
    assert {"enabled", "chat_id", "has_token", "configured", "channel"} <= set(cfg.json())
    # set is gated by apply (stores a token / changes behavior)
    assert c.post("/api/v1/alerts/config", json={"enabled": True, "chat_id": "1"}).status_code == 403
    assert c.post("/api/v1/alerts/test", json={}).status_code == 403


def test_container_engine_health(monkeypatch):
    _patch_local(monkeypatch, _FakeControl())
    r = _client().get("/api/v1/containers/vllm-35b-prod/engine")
    assert r.status_code == 200
    assert r.json() == {"reachable": True, "port": 8101, "status_code": 200}


def test_container_source_report(monkeypatch):
    _patch_local(monkeypatch, _FakeControl())
    r = _client().get("/api/v1/containers/vllm-35b-prod/source")
    assert r.status_code == 200
    body = r.json()
    assert {"container", "preset_id", "linked_by", "drift", "drift_count"} <= set(body)
    assert body["container"] == "vllm-35b-prod"


def test_container_top_and_changes(monkeypatch):
    _patch_local(monkeypatch, _FakeControl())
    c = _client()
    top = c.get("/api/v1/containers/vllm-35b-prod/top")
    assert top.status_code == 200 and top.json()["titles"] == ["PID", "CMD"]
    ch = c.get("/api/v1/containers/vllm-35b-prod/changes")
    assert ch.status_code == 200 and ch.json()["changes"][0]["kind"] == "added"


def test_container_update_plan_engine_is_manual(monkeypatch):
    _patch_local(monkeypatch, _FakeControl())
    body = _client().get("/api/v1/containers/vllm-35b-prod/update-plan").json()
    assert body["is_engine"] is True
    assert body["guarded_update"] is False           # engine → manual pin policy
    assert any("docker pull vllm/vllm-openai" in c for c in body["commands"])


def test_container_pull_gated(monkeypatch):
    fake = _FakeControl()
    _patch_local(monkeypatch, fake)
    # apply OFF → blocked
    assert _client().post("/api/v1/containers/vllm-35b-prod/pull", json={"confirm": True}).status_code == 403
    assert ("pull", "vllm-35b-prod") not in fake.calls
    # apply ON + confirm → runs
    client = TestClient(create_app(allowed_origins=(), enable_apply=True))
    ok = client.post("/api/v1/containers/vllm-35b-prod/pull", json={"confirm": True})
    assert ok.status_code == 200 and ok.json()["image"] == "vllm/vllm-openai:nightly"
    assert ("pull", "vllm-35b-prod") in fake.calls


def test_container_logs_stream_ndjson(monkeypatch):
    import json as _json
    _patch_local(monkeypatch, _FakeControl())
    r = _client().get("/api/v1/containers/vllm-35b-prod/logs/stream")
    assert r.status_code == 200
    assert "ndjson" in r.headers["content-type"]
    objs = [_json.loads(ln) for ln in r.text.splitlines() if ln.strip()]
    assert {"line": "hello\n"} in objs and {"line": "world\n"} in objs
    assert {"hb": True} in objs  # heartbeat keeps the stream writable


def test_container_settings_gated_and_applied(monkeypatch):
    fake = _FakeControl()
    _patch_local(monkeypatch, fake)
    # apply OFF → blocked
    assert _client().post("/api/v1/containers/vllm-35b-prod/settings",
                          json={"cpus": 2, "confirm": True}).status_code == 403
    client = TestClient(create_app(allowed_origins=(), enable_apply=True))
    ok = client.post("/api/v1/containers/vllm-35b-prod/settings",
                     json={"cpus": 2, "memory": 1000, "restart_policy": "always", "confirm": True})
    assert ok.status_code == 200 and ok.json()["updated"] is True
    assert ("settings", "vllm-35b-prod", 2.0, 1000, "always") in fake.calls


def test_container_network_attach_and_list(monkeypatch):
    fake = _FakeControl()
    _patch_local(monkeypatch, fake)
    nets = _client().get("/api/v1/system/networks")
    assert nets.status_code == 200 and nets.json()["networks"][0]["name"] == "bridge"
    client = TestClient(create_app(allowed_origins=(), enable_apply=True))
    r = client.post("/api/v1/containers/vllm-35b-prod/network",
                    json={"network": "frontends", "action": "connect", "confirm": True})
    assert r.status_code == 200
    assert ("net-connect", "vllm-35b-prod", "frontends") in fake.calls


def test_system_df_and_scan(monkeypatch):
    _patch_local(monkeypatch, _FakeControl())
    c = _client()
    df = c.get("/api/v1/system/df")
    assert df.status_code == 200 and df.json()["total_size"] == 1000
    scan = c.get("/api/v1/containers/vllm-35b-prod/scan")
    assert scan.status_code == 200 and scan.json()["counts"]["critical"] == 1


def test_container_fs_requires_apply_not_exec(monkeypatch):
    _patch_local(monkeypatch, _FakeControl())
    # apply OFF → file browsing blocked
    assert _client().get("/api/v1/containers/vllm-35b-prod/fs", params={"path": "/app"}).status_code == 403
    # apply ON → works WITHOUT SNDR_ENABLE_EXEC (fixed read commands, read tier)
    client = TestClient(create_app(allowed_origins=(), enable_apply=True))
    ok = client.get("/api/v1/containers/vllm-35b-prod/fs", params={"path": "/app"})
    assert ok.status_code == 200 and ok.json()["entries"][0]["name"] == "etc"


def test_container_endpoints_registered():
    paths = _client().get("/openapi.json").json()["paths"]
    assert "/api/v1/containers" in paths
    assert "/api/v1/containers/{name}/action" in paths
    assert "/api/v1/containers/{name}/exec" in paths
    assert "/api/v1/hosts/{host_id}/containers" in paths
    assert "/api/v1/hosts/{host_id}/containers/{name}/exec" in paths
