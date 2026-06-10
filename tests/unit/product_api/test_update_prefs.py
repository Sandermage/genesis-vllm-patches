# SPDX-License-Identifier: Apache-2.0
"""Tests for per-container update-mode preferences."""
from __future__ import annotations

import pytest

from sndr.product_api.legacy import update_prefs as up


@pytest.fixture(autouse=True)
def _isolate_home(monkeypatch, tmp_path):
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))


def test_default_mode_is_manual():
    assert up.get_mode("local", "vllm-x") == "manual"


def test_normalize_rejects_unknown():
    assert up.normalize_mode("AUTO") == "auto"
    assert up.normalize_mode("bogus") == "manual"
    assert up.normalize_mode(None) == "manual"


def test_set_and_get_roundtrip():
    r = up.set_mode("local", "sndr-daemon", "semi")
    assert r["ok"] is True and r["mode"] == "semi"
    assert up.get_mode("local", "sndr-daemon") == "semi"


def test_auto_blocked_for_critical_containers():
    r = up.set_mode("local", "vllm-engine", "auto", is_critical=True)
    assert r["ok"] is False
    assert "blocked" in r["error"]
    # the mode must NOT have been persisted
    assert up.get_mode("local", "vllm-engine") == "manual"


def test_auto_allowed_for_non_critical():
    r = up.set_mode("host-a", "sidecar", "auto", is_critical=False)
    assert r["ok"] is True and r["mode"] == "auto"
    assert up.get_mode("host-a", "sidecar") == "auto"


def test_keys_do_not_collide_across_sources():
    up.set_mode("local", "c1", "semi")
    up.set_mode("host-a", "c1", "auto")
    assert up.get_mode("local", "c1") == "semi"
    assert up.get_mode("host-a", "c1") == "auto"


def test_previous_image_roundtrip_and_default():
    assert up.get_previous("local", "c2") is None
    up.set_previous("local", "c2", "sha256:abc123")
    assert up.get_previous("local", "c2") == "sha256:abc123"
    # empty is ignored (never clobbers a real previous)
    up.set_previous("local", "c2", "")
    assert up.get_previous("local", "c2") == "sha256:abc123"


def test_mode_and_previous_coexist():
    up.set_mode("local", "c3", "semi")
    up.set_previous("local", "c3", "sha256:deadbeef")
    assert up.get_mode("local", "c3") == "semi"
    assert up.get_previous("local", "c3") == "sha256:deadbeef"
