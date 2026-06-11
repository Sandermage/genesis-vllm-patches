# SPDX-License-Identifier: Apache-2.0
"""Tests for container ↔ preset linkage + config drift detection (pure helpers)."""
from __future__ import annotations

from sndr.product_api.legacy import container_link as cl


def test_resolve_preset_prefers_label_then_name():
    index = {"vllm-pn95-2xa5000": "a5000-2x-27b"}
    # label wins
    pid, by = cl.resolve_preset("anything", {"sndr.preset": "prod-35b"}, index)
    assert pid == "prod-35b" and by == "label"
    # fall back to name convention
    pid, by = cl.resolve_preset("vllm-pn95-2xa5000", {}, index)
    assert pid == "a5000-2x-27b" and by == "name"
    # no link
    assert cl.resolve_preset("redis", {}, index) == (None, None)


def test_build_preset_index_skips_missing_names():
    idx = cl.build_preset_index(
        lambda: ["p1", "p2", "p3"],
        lambda pid: {"p1": "vllm-a", "p2": None, "p3": "vllm-c"}.get(pid),
    )
    assert idx == {"vllm-a": "p1", "vllm-c": "p3"}


def test_compute_drift_detects_image_and_env():
    inspect = {"Config": {
        "Image": "vllm/vllm-openai:OLD",
        "Env": ["GENESIS_ENABLE_P82=1", "PATH=/usr/bin", "GENESIS_ENABLE_PN90=0"],
    }}
    drift = cl.compute_drift(
        "vllm/vllm-openai:nightly",
        {"GENESIS_ENABLE_P82": "1", "GENESIS_ENABLE_PN90": "1", "GENESIS_ENABLE_P94": "1"},
        inspect,
    )
    fields = {d["field"]: d for d in drift}
    assert fields["image"]["kind"] == "image"
    assert fields["GENESIS_ENABLE_PN90"]["kind"] == "changed"     # 1 expected, 0 running
    assert fields["GENESIS_ENABLE_P94"]["kind"] == "missing"      # not set at all
    assert "GENESIS_ENABLE_P82" not in fields                     # matches → no drift


def test_compute_drift_clean_when_matching():
    inspect = {"Config": {"Image": "img:1", "Env": ["A=1"]}}
    assert cl.compute_drift("img:1", {"A": "1"}, inspect) == []


def test_parse_env():
    assert cl.parse_env(["A=1", "B=x=y", "NOEQ"]) == {"A": "1", "B": "x=y"}


def test_live_patches_extracts_on_genesis_flags():
    inspect = {"Config": {"Env": [
        "GENESIS_ENABLE_P82=1", "GENESIS_ENABLE_PN90=0", "PN95_TIER_AWARE=true",
        "PATH=/usr/bin", "CUDA_VISIBLE_DEVICES=0,1", "GENESIS_ENABLE_P94=on",
    ]}}
    live = cl.live_patches(inspect)
    flags = {p["flag"] for p in live}
    assert flags == {"GENESIS_ENABLE_P82", "PN95_TIER_AWARE", "GENESIS_ENABLE_P94"}  # on-only, Genesis-only
    assert "GENESIS_ENABLE_PN90" not in flags  # value 0 → off
    assert live == sorted(live, key=lambda p: p["flag"])  # stable order


def test_live_patches_includes_sndr_enable_patch_flags_but_not_daemon_gates():
    """A handful of patches use the SNDR_ENABLE_ prefix (PN282/PN283). They must
    be detected as live, while the same-prefix daemon gates SNDR_ENABLE_APPLY /
    SNDR_ENABLE_EXEC must NOT show up as patches (they aren't in the registry)."""
    inspect = {"Config": {"Env": [
        "SNDR_ENABLE_PN283_PROC_BRIDGE=1",                  # real registry patch flag
        "SNDR_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC=on",     # real registry patch flag (PN282)
        "SNDR_ENABLE_APPLY=1", "SNDR_ENABLE_EXEC=1",        # daemon gates — NOT patches
        "SNDR_BIND=0.0.0.0", "GENESIS_ENABLE_P82=1",
    ]}}
    flags = {p["flag"] for p in cl.live_patches(inspect)}
    assert "SNDR_ENABLE_PN283_PROC_BRIDGE" in flags
    assert "SNDR_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC" in flags
    assert "GENESIS_ENABLE_P82" in flags
    assert "SNDR_ENABLE_APPLY" not in flags
    assert "SNDR_ENABLE_EXEC" not in flags
    assert "SNDR_BIND" not in flags


def test_source_report_includes_live_patches():
    rep = cl.source_report("vllm-x", {"Config": {"Env": ["GENESIS_ENABLE_P82=1"], "Labels": {}}})
    assert rep["live_patch_count"] == 1 and rep["live_patches"][0]["flag"] == "GENESIS_ENABLE_P82"


def test_source_report_surfaces_launcher_identity_labels():
    # A launcher-stamped engine carries served-model / pin / role labels so the
    # GUI shows them without querying the engine's api-key-protected /v1/models.
    rep = cl.source_report("vllm-x", {"Config": {"Env": [], "Labels": {
        "sndr.served-model": "qwen3.6-35b-a3b",
        "sndr.pin": "nightly-626fa9bba",
        "sndr.role": "tuning",
    }}})
    assert rep["served_model"] == "qwen3.6-35b-a3b"
    assert rep["pin"] == "nightly-626fa9bba"
    assert rep["role"] == "tuning"


def test_source_report_identity_none_when_unlabelled():
    rep = cl.source_report("vllm-x", {"Config": {"Env": [], "Labels": {}}})
    assert rep["served_model"] is None and rep["pin"] is None and rep["role"] is None


def test_compute_drift_ignores_tag_vs_digest_same_repo():
    # The catalog may pin by digest while the launcher deploys the equivalent
    # tag — same repo, different ref scheme is NOT image drift.
    insp = {"Config": {"Image": "vllm/vllm-openai:nightly-626fa9b", "Env": []}}
    drift = cl.compute_drift("vllm/vllm-openai@sha256:674922aa", {}, insp)
    assert not any(d["field"] == "image" for d in drift)


def test_compute_drift_flags_real_image_change():
    # Different repo IS drift.
    insp = {"Config": {"Image": "vllm/other:nightly", "Env": []}}
    drift = cl.compute_drift("vllm/vllm-openai:nightly", {}, insp)
    assert any(d["field"] == "image" for d in drift)


def test_reconcile_patches_in_sync_missing_extra():
    expected = {"GENESIS_ENABLE_P82": "1", "GENESIS_ENABLE_PN90": "1", "GENESIS_ENABLE_OFF": "0"}
    inspect = {"Config": {"Env": ["GENESIS_ENABLE_P82=1", "PN95_EXTRA=true"]}}  # P82 on, PN90 missing, PN95 extra
    r = cl.reconcile_patches(expected, inspect)
    assert r["in_sync"] == ["GENESIS_ENABLE_P82"]
    assert r["missing"] == ["GENESIS_ENABLE_PN90"]   # config wants on, engine off
    assert r["extra"] == ["PN95_EXTRA"]              # on in engine, not declared
    assert "GENESIS_ENABLE_OFF" not in r["missing"]  # config has it off → not expected on
