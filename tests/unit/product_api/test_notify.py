# SPDX-License-Identifier: Apache-2.0
"""Tests for operator notification config (transport not exercised — no network)."""
from __future__ import annotations

from sndr.product_api.legacy import notify


def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(notify, "_state_dir", lambda: tmp_path)
    for var in ("SNDR_TELEGRAM_BOT_TOKEN", "SNDR_TELEGRAM_CHAT_ID", "SNDR_ALERTS"):
        monkeypatch.delenv(var, raising=False)
    # Token resolves via env here (avoid touching the real secrets backend).
    monkeypatch.setattr(notify, "_token", lambda: None)


def test_config_roundtrip_and_not_configured(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    notify.set_config(enabled=True, chat_id="12345")
    cfg = notify.get_config()
    assert cfg["enabled"] is True and cfg["chat_id"] == "12345"
    assert cfg["has_token"] is False
    assert cfg["configured"] is False           # no token yet
    assert notify.alerts_enabled() is False      # enabled but not configured


def test_configured_when_token_and_chat_present(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(notify, "_token", lambda: "bot:TOKEN")
    notify.set_config(enabled=True, chat_id="999")
    assert notify.configured() is True
    assert notify.alerts_enabled() is True


def test_send_without_config_is_graceful(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    res = notify.send("hi")
    assert res["ok"] is False and "configured" in res["error"]


def test_notify_records_event_even_without_channel(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    res = notify.notify("engine down", kind="critical")
    assert res["ok"] is False and res.get("recorded") is True
