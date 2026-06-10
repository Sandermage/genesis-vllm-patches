# SPDX-License-Identifier: Apache-2.0
"""Tests for the daemon routing bridge (spec-decode router + artifacts)."""
import pytest

from sndr.product_api.legacy import routing as R

STRUCTURED = "gemma4-31b-tq-mtp-structured-k4"


def test_available_and_list():
    if not R.available():
        pytest.skip("spec_decode integration not importable in this env")
    out = R.list_artifacts()
    assert out["available"] is True
    profiles = {a["profile"] for a in out["artifacts"]}
    assert STRUCTURED in profiles
    art = next(a for a in out["artifacts"] if a["profile"] == STRUCTURED)
    assert art["k"] == 4
    assert "tool_json" in art["allowed_workloads"]
    assert "free_chat" in art["denied_workloads"]
    # bench economics carried through
    assert art["delta_tps_per_class"]["tool_json"] > 0
    assert art["delta_tps_per_class"]["free_chat"] < 0


def test_classify_accepts_structured_signal():
    if not R.available():
        pytest.skip("spec_decode integration not importable")
    # response_format=json_object → tool_json → allowed on the structured profile.
    res = R.classify(signals={"response_format": {"type": "json_object"}}, profile=STRUCTURED)
    assert res["accepted"] is True
    assert res["workload_class"] == "tool_json"
    assert res["profile"] == STRUCTURED
    assert res["expected_delta_tps"] is not None and res["expected_delta_tps"] > 0


def test_classify_denies_free_chat():
    if not R.available():
        pytest.skip("spec_decode integration not importable")
    res = R.classify(signals={"workload_class": "free_chat"}, profile=STRUCTURED)
    assert res["accepted"] is False           # denied → conservative fallback
    assert res["profile"] == "default (MTP off)"


def test_classify_no_signal_falls_back():
    if not R.available():
        pytest.skip("spec_decode integration not importable")
    res = R.classify(signals={}, profile=STRUCTURED)
    assert res["accepted"] is False
    assert res["signal"] == "no_signal"


def test_active_profile_env_override(monkeypatch):
    if not R.available():
        pytest.skip("spec_decode integration not importable")
    monkeypatch.setenv("SNDR_ACTIVE_PROFILE", STRUCTURED)
    act = R.active_profile()
    assert act["available"] is True
    assert act["profile"] == STRUCTURED and act["source"] == "env"
    assert act["artifact"]["profile"] == STRUCTURED
