# SPDX-License-Identifier: Apache-2.0
"""Tests for the read-only SNDR Product API FastAPI app."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from vllm.sndr_core.product_api.http_app import create_app  # noqa: E402


def _client() -> TestClient:
    return TestClient(create_app(allowed_origins=()))


def test_health_and_openapi_are_available():
    client = _client()

    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.json()["read_only"] is True

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
    from vllm.sndr_core.product_api import chat_rag

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

    from vllm.sndr_core.product_api.http_app import create_app as _ca

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

    from vllm.sndr_core.product_api.http_app import create_app as _ca

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

    from vllm.sndr_core.product_api.http_app import create_app as _ca

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
    assert payload["catalog"]["presets_count"] == 21
    assert payload["capabilities"]["platform"]["sndr_core_version"]

    summary = client.get("/api/v1/catalog/summary")
    assert summary.status_code == 200
    # Every builtin preset carries a card.
    assert summary.json()["preset_cards_count"] == 21


def test_presets_endpoints_are_read_only_json_views():
    client = _client()

    listed = client.get(
        "/api/v1/presets",
        params={"status": "production_candidate"},
    )
    assert listed.status_code == 200
    assert listed.json()["matched"] == 14

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
