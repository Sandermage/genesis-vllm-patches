# SPDX-License-Identifier: Apache-2.0
"""Operator-managed prompt library (OpenWebUI-style prompts in the SNDR GUI).

Selectable system-prompt templates with a name + title, persisted operator-local
under ``SNDR_HOME/gui/prompts.json`` (same convention as :mod:`baselines`).
Built-in seeds (the ported crypto-analyst prompt + a few research templates) are
read-only; the operator adds / edits / deletes their own. The chat and copilot
load a prompt's ``content`` as the system prompt.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Optional

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _store_path() -> Path:
    from sndr.engines.vllm.locations.project_paths import install_root

    return install_root() / "gui" / "prompts.json"


def _seed_dir() -> Path:
    return Path(__file__).resolve().parent / "seed"


def _builtins() -> list[dict[str, Any]]:
    """Read-only seed prompts. The crypto-analyst prompt is the operator's ported
    OpenWebUI analyst (market-regime classes + news/info-field framework)."""
    items: list[dict[str, Any]] = []
    analyst = _seed_dir() / "crypto_analyst.txt"
    if analyst.exists():
        items.append({"id": "crypto-analyst", "name": "Crypto market analyst",
                      "title": "Senior crypto hedge-fund analyst — market regimes + news field",
                      "content": analyst.read_text(encoding="utf-8"), "builtin": True})
    items += [
        {"id": "general", "name": "General assistant", "title": "Plain helpful assistant",
         "content": "You are a helpful assistant.", "builtin": True},
        {"id": "web-research", "name": "Web research", "title": "Cited web research",
         "content": "You are a research assistant. For any real-time or factual claim, rely on web search "
                    "results in context and cite the source URLs. Be concise, structured, and flag uncertainty.",
         "builtin": True},
        {"id": "news-brief", "name": "News briefing", "title": "Dated, sourced news briefing",
         "content": "You are a news analyst. Gather current items via web search, then synthesize a dated, "
                    "sourced briefing: the key developments, why they matter, and what to watch — with citation URLs.",
         "builtin": True},
    ]
    return items


def _load_user() -> dict[str, Any]:
    p = _store_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_user(data: dict[str, Any]) -> None:
    p = _store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _slug(name: str, taken: set[str]) -> str:
    base = _SLUG_RE.sub("-", str(name).strip().lower()).strip("-") or "prompt"
    pid, n = base, 2
    while pid in taken:
        pid, n = f"{base}-{n}", n + 1
    return pid


# ── public CRUD ──────────────────────────────────────────────────────────────


def list_prompts() -> list[dict[str, Any]]:
    user = _load_user()
    return _builtins() + [user[k] for k in sorted(user, key=lambda k: user[k].get("created_at", 0))]


def get_prompt(pid: str) -> Optional[dict[str, Any]]:
    return next((p for p in list_prompts() if p["id"] == pid), None)


def create_prompt(name: str, content: str, *, title: str = "") -> dict[str, Any]:
    name = str(name or "").strip()
    content = str(content or "")
    if not name or not content.strip():
        raise ValueError("name and content are required")
    user = _load_user()
    taken = {p["id"] for p in _builtins()} | set(user)
    pid = _slug(name, taken)
    rec = {"id": pid, "name": name, "title": str(title or "").strip(),
           "content": content, "builtin": False, "created_at": time.time()}
    user[pid] = rec
    _save_user(user)
    return rec


def update_prompt(pid: str, *, name: Optional[str] = None, content: Optional[str] = None,
                  title: Optional[str] = None) -> dict[str, Any]:
    user = _load_user()
    if pid not in user:
        raise ValueError("unknown or read-only prompt" if get_prompt(pid) else "unknown prompt")
    rec = user[pid]
    if name is not None and str(name).strip():
        rec["name"] = str(name).strip()
    if title is not None:
        rec["title"] = str(title).strip()
    if content is not None and str(content).strip():
        rec["content"] = str(content)
    _save_user(user)
    return rec


def delete_prompt(pid: str) -> bool:
    user = _load_user()
    if pid not in user:
        return False
    del user[pid]
    _save_user(user)
    return True


__all__ = ["list_prompts", "get_prompt", "create_prompt", "update_prompt", "delete_prompt"]
