# SPDX-License-Identifier: Apache-2.0
"""engine_model: bridge a live vLLM served model to the SNDR V2 catalog.

The GUI auto-detects the model a running engine serves (its ``/v1/models`` id).
These tests pin the *enrichment* half: given that served-model id, resolve it to
the catalog ModelDef (capabilities / requirements / pin) and the presets that run
it. Pure catalog logic — no live engine needed; the HTTP detail call is exercised
only for graceful degradation when nothing is listening."""
from __future__ import annotations


def test_match_catalog_model_bridges_a_real_served_name():
    from sndr.model_configs import registry_v2
    from sndr.product_api.legacy.engine_model import match_catalog_model

    ids = registry_v2.list_models()
    assert ids, "V2 catalog has no models to test against"
    md = registry_v2.load_model(ids[0])
    served = md.served_model_name or md.model_path.rstrip("/").split("/")[-1]

    matched = match_catalog_model(served)
    assert matched is not None, f"no catalog match for live served name {served!r}"
    # Robust to duplicate served names (several variants can share one): the model
    # we resolved to must actually serve this name.
    resolved = registry_v2.load_model(matched["model_id"])
    assert (resolved.served_model_name or resolved.model_path.rstrip("/").split("/")[-1]) == served

    caps = matched["capabilities"]
    for key in ("attention_arch", "tool_call_parser", "reasoning_parser", "spec_decode", "kv_cache_dtype"):
        assert key in caps, f"capability {key!r} missing from bridge payload"
    assert "min_total_vram_mib" in matched["requires"]
    assert isinstance(matched["presets"], list)
    assert matched["match_kind"] in {"served_model_name", "model_path", "id"}


def test_match_catalog_model_unknown_returns_none():
    from sndr.product_api.legacy.engine_model import match_catalog_model

    assert match_catalog_model("nonexistent-model-zzz-999") is None
    assert match_catalog_model("") is None


def test_recommended_sampling_surfaced_for_override_model():
    """A model that carries override_generation_config (the catalog's validated,
    club-3090-cross-referenced sampling) exposes it as `recommended_sampling`."""
    from sndr.model_configs import registry_v2
    from sndr.product_api.legacy.engine_model import match_catalog_model

    over = next((m for m in registry_v2.list_models()
                 if registry_v2.load_model(m).override_generation_config), None)
    if over is None:
        import pytest
        pytest.skip("no catalog model carries override_generation_config")

    matched = match_catalog_model(over)  # match by id (unique)
    assert matched is not None and matched["model_id"] == over
    rec = matched["recommended_sampling"]
    assert isinstance(rec, dict) and any(k in rec for k in ("temperature", "top_p", "top_k"))
    # only sampling keys are surfaced (no leaking of unrelated generation config)
    assert set(rec).issubset({"temperature", "top_p", "top_k", "min_p", "repetition_penalty"})


def test_recommended_sampling_none_when_absent():
    from sndr.model_configs import registry_v2
    from sndr.product_api.legacy.engine_model import match_catalog_model

    plain = next((m for m in registry_v2.list_models()
                  if not registry_v2.load_model(m).override_generation_config), None)
    if plain is None:
        import pytest
        pytest.skip("every catalog model carries override_generation_config")
    matched = match_catalog_model(plain)
    assert matched is not None and "recommended_sampling" in matched
    assert matched["recommended_sampling"] is None


def test_match_catalog_model_matches_by_model_path_basename():
    """vLLM that was launched with ``--model <path>`` and no ``--served-model-name``
    reports the path; the bridge must still resolve it via the model_path."""
    from sndr.model_configs import registry_v2
    from sndr.product_api.legacy.engine_model import match_catalog_model

    md = registry_v2.load_model(registry_v2.list_models()[0])
    basename = md.model_path.rstrip("/").split("/")[-1]
    matched = match_catalog_model(basename)
    assert matched is not None


def test_discover_engine_prefers_local_then_registered_host(monkeypatch):
    """When the local/configured engine is unreachable, discovery falls back to a
    registered host's declared engine endpoint (host + engine_port + stored key)."""
    import types
    from sndr.product_api.legacy import engine_model

    seen = []

    def fake_detail(host=None, *, port=None, timeout=3.0, api_key=None):
        seen.append((host, port, api_key))
        if host is None:  # local / configured — nothing running
            return {"reachable": False, "host": "127.0.0.1", "models": [], "error": "refused"}
        return {"reachable": True, "host": host, "version": "0.22",
                "models": [{"id": "qwen3.6-35b-a3b", "max_model_len": 262144, "root": None, "catalog": None}],
                "error": None}

    monkeypatch.setattr(engine_model, "engine_model_detail", fake_detail)
    prof = types.SimpleNamespace(id="prod-a5000", host="192.0.2.10", engine_port=8102)
    out = engine_model.discover_engine(timeout=0.2, profiles=[prof], key_for=lambda p: "stored-key")

    assert out["reachable"] is True and out["host"] == "192.0.2.10"
    assert out["host_id"] == "prod-a5000" and out["port"] == 8102
    assert out["models"][0]["id"] == "qwen3.6-35b-a3b"
    assert ("192.0.2.10", 8102, "stored-key") in seen  # probed with the host's key


def test_discover_engine_returns_local_when_it_already_serves_models(monkeypatch):
    import types
    from sndr.product_api.legacy import engine_model

    monkeypatch.setattr(engine_model, "engine_model_detail",
        lambda host=None, *, port=None, timeout=3.0, api_key=None: {
            "reachable": True, "host": "127.0.0.1", "base_url": "http://127.0.0.1:8102/v1",
            "models": [{"id": "x", "max_model_len": None, "root": None, "catalog": None}], "error": None})
    out = engine_model.discover_engine(
        timeout=0.2, profiles=[types.SimpleNamespace(id="p", host="h", engine_port=1)], key_for=lambda p: None)
    # The configured engine's real port (8102 from base_url) is surfaced so a chat
    # stuck on :8000 can adopt it — host_id stays None (it's the local/configured one).
    assert out["host"] == "127.0.0.1" and out["host_id"] is None and out["port"] == 8102


def test_engine_model_detail_unreachable_is_graceful():
    from sndr.product_api.legacy.engine_model import engine_model_detail

    detail = engine_model_detail(host="127.0.0.1", port=59999, timeout=0.5)
    assert detail["reachable"] is False
    assert detail["models"] == []
    assert "host" in detail and "error" in detail


def test_engine_model_detail_bridges_reachable_engine(monkeypatch):
    """Full path: a reachable engine serving a catalog model resolves to its
    catalog payload + the vLLM max_model_len/root metadata."""
    from sndr.model_configs import registry_v2
    from sndr.product_api.legacy import engine_client, engine_model

    md = registry_v2.load_model(registry_v2.list_models()[0])
    served = md.served_model_name or md.model_path.rstrip("/").split("/")[-1]

    monkeypatch.setattr(engine_client, "engine_status", lambda *a, **k: {
        "reachable": True, "host": "gpu-01", "base_url": "http://gpu-01:8000/v1",
        "version": "0.22.0", "models": [served], "error": None,
    })
    monkeypatch.setattr(engine_model, "_vllm_model_meta",
                        lambda *a, **k: {served: {"max_model_len": 131072, "root": "/models/x"}})

    detail = engine_model.engine_model_detail(host="gpu-01")
    assert detail["reachable"] is True and detail["version"] == "0.22.0"
    assert len(detail["models"]) == 1
    info = detail["models"][0]
    assert info["id"] == served and info["max_model_len"] == 131072 and info["root"] == "/models/x"
    assert info["catalog"] is not None and info["catalog"]["model_id"] == md.id
