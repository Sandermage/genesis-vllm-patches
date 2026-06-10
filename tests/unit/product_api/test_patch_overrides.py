# SPDX-License-Identifier: Apache-2.0
"""Tests for operator patch overrides: store, validation, launch consumption."""
from __future__ import annotations

import pytest

from sndr.product_api.legacy import patch_overrides as po


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))


def test_set_and_load_override():
    po.set_override("P67", "on", "GENESIS_ENABLE_P67")
    po.set_override("PN95", "off", "GENESIS_ENABLE_PN95_TIER_AWARE_CACHE")
    data = po.load()
    assert data["P67"] == {"state": "on", "env_flag": "GENESIS_ENABLE_P67"}
    assert data["PN95"]["state"] == "off"


def test_default_clears_override():
    po.set_override("P67", "on", "GENESIS_ENABLE_P67")
    po.set_override("P67", "default", "GENESIS_ENABLE_P67")
    assert "P67" not in po.load()


def test_env_lines_rendered_and_sorted():
    po.set_override("PN95", "off", "GENESIS_ENABLE_PN95")
    po.set_override("P67", "on", "GENESIS_ENABLE_P67")
    assert po.env_lines() == ["GENESIS_ENABLE_P67=1", "GENESIS_ENABLE_PN95=0"]


def test_validation_rejects_injection():
    with pytest.raises(ValueError):
        po.set_override("evil; rm -rf", "on", "GENESIS_ENABLE_X")
    with pytest.raises(ValueError):
        po.set_override("P67", "on", "X=1; rm -rf /")     # not a GENESIS_ flag
    with pytest.raises(ValueError):
        po.set_override("P67", "maybe", "GENESIS_ENABLE_X")  # bad state


def test_overrides_appear_in_launch_env():
    po.set_override("P67", "on", "GENESIS_ENABLE_P67")
    from sndr.product_api.legacy.launch_plan import build_launch_plan

    plan = build_launch_plan(preset_id="prod-qwen3.6-35b-multiconc")
    env_artifact = next((a for a in plan.artifacts if a.kind == "env"), None)
    assert env_artifact is not None
    assert "GENESIS_ENABLE_P67=1" in env_artifact.content


def test_override_routes(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from sndr.product_api.legacy.http_app import create_app

    client = TestClient(create_app(allowed_origins=()))
    assert client.get("/api/v1/patches/overrides").json() == {"overrides": {}}
    set_resp = client.post("/api/v1/patches/overrides", json={"patch_id": "P67", "state": "on", "env_flag": "GENESIS_ENABLE_P67"})
    assert set_resp.status_code == 200 and set_resp.json()["overrides"]["P67"]["state"] == "on"
    assert client.get("/api/v1/patches/overrides").json()["overrides"]["P67"]["state"] == "on"
    # bad input -> 400
    assert client.post("/api/v1/patches/overrides", json={"patch_id": "bad;id", "state": "on", "env_flag": "GENESIS_ENABLE_X"}).status_code == 400
