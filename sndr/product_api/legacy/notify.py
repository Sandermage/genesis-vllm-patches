# SPDX-License-Identifier: Apache-2.0
"""Operator notifications — currently Telegram, kept pluggable.

Inference-ops needs a push when the engine container dies at 3am. This module is
the transport + config: a bot token (stored encrypted via :mod:`secrets_store`)
plus a chat id and an enable flag (persisted alongside the GUI state). Env vars
``SNDR_TELEGRAM_BOT_TOKEN`` / ``SNDR_TELEGRAM_CHAT_ID`` are honoured as fallbacks
so a headless deploy can configure it without the API.

Sending is stdlib-only (urllib) so the daemon needs no extra dependency. The
:func:`notify` entrypoint also drops a structured event into the audit feed, so
an alert is recorded even when no channel is configured.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

_TOKEN_SECRET = "telegram_bot_token"


def _state_dir() -> Path:
    from sndr.engines.vllm.locations.project_paths import install_root
    return install_root() / "gui"


def _config_path() -> Path:
    return _state_dir() / "alerts.json"


def _read_config() -> dict[str, Any]:
    try:
        return json.loads(_config_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_config(data: dict[str, Any]) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _token() -> Optional[str]:
    env = os.environ.get("SNDR_TELEGRAM_BOT_TOKEN")
    if env:
        return env.strip()
    try:
        from . import secrets_store
        return secrets_store.get_secret(_TOKEN_SECRET)
    except Exception:
        return None


def _chat_id() -> Optional[str]:
    env = os.environ.get("SNDR_TELEGRAM_CHAT_ID")
    if env:
        return env.strip()
    cid = _read_config().get("chat_id")
    return str(cid) if cid else None


def configured() -> bool:
    """True when both a bot token and a chat id are available."""
    return bool(_token() and _chat_id())


def alerts_enabled() -> bool:
    """Whether the background health-watch should run + dispatch."""
    if os.environ.get("SNDR_ALERTS", "").strip().lower() in ("1", "true", "yes", "on"):
        return configured()
    return bool(_read_config().get("enabled")) and configured()


def get_config() -> dict[str, Any]:
    """Read-only config snapshot for the GUI (never returns the token value)."""
    cfg = _read_config()
    return {
        "enabled": bool(cfg.get("enabled")) or os.environ.get("SNDR_ALERTS", "").lower() in ("1", "true", "yes", "on"),
        "chat_id": _chat_id() or "",
        "has_token": bool(_token()),
        "configured": configured(),
        "channel": "telegram",
    }


def set_config(*, enabled: Optional[bool] = None, chat_id: Optional[str] = None,
               bot_token: Optional[str] = None) -> dict[str, Any]:
    """Persist alert config. A non-empty ``bot_token`` is stored encrypted; an
    empty string clears it. Returns the read-only snapshot."""
    cfg = _read_config()
    if enabled is not None:
        cfg["enabled"] = bool(enabled)
    if chat_id is not None:
        cfg["chat_id"] = str(chat_id).strip()
    _write_config(cfg)
    if bot_token is not None:
        from . import secrets_store
        if bot_token.strip():
            secrets_store.set_secret(_TOKEN_SECRET, bot_token.strip())
        else:
            try:
                secrets_store.delete_secret(_TOKEN_SECRET)
            except Exception:
                pass
    return get_config()


def send(text: str, *, timeout: float = 10.0) -> dict[str, Any]:
    """Send a Telegram message. Returns {"ok": bool, "error"?: str}."""
    token, chat = _token(), _chat_id()
    if not token or not chat:
        return {"ok": False, "error": "telegram not configured (need bot token + chat id)"}
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": chat, "text": text, "parse_mode": "HTML",
        "disable_web_page_preview": "true",
    }).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=timeout) as resp:
            return {"ok": 200 <= resp.status < 300}
    except Exception as exc:  # network / auth failure — surface, don't crash the watcher
        return {"ok": False, "error": str(exc)}


def notify(text: str, *, kind: str = "alert", detail: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Record an audit event AND dispatch to the configured channel (if any)."""
    try:
        from .jobs import record_event
        record_event(f"alert.{kind}", text, detail or {})
    except Exception:
        pass
    if configured():
        return send(text)
    return {"ok": False, "error": "no channel configured", "recorded": True}


__all__ = ["configured", "alerts_enabled", "get_config", "set_config", "send", "notify"]
