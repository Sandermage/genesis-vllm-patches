# SPDX-License-Identifier: Apache-2.0
"""Operator-managed declarative tools (the GUI tool manager).

OpenWebUI lets you add Python ``class Tools`` — powerful but it runs arbitrary
code. The operator chose the **safe** model instead: a tool is a *declaration*
(name, title, description, an HTTP method + URL template + typed parameters), not
code. The copilot exposes every enabled tool to the model and runs it through the
SSRF-safe :func:`run_tool` executor — no arbitrary Python in the daemon.

Safety: the **host is fixed by the operator** in the URL template; the model's
arguments only fill ``{param}`` placeholders and are URL-encoded, so a crafted
argument can never change the host/scheme (no SSRF via tool args). Only
``http``/``https`` GET/POST to the operator-declared endpoint. Persisted under
``SNDR_HOME/gui/managed_tools.json``.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,48}$")            # valid tool/function name
_TEMPLATE_HOST_RE = re.compile(r"^https?://[A-Za-z0-9._-]{1,253}(:\d{1,5})?(/|$|\?)")
_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
_RESULT_CAP = 6000  # trim a tool result before it enters the model context
_TYPES = {"string", "integer", "number", "boolean"}


def _store_path() -> Path:
    from sndr.engines.vllm.locations.project_paths import install_root

    return install_root() / "gui" / "managed_tools.json"


def _load() -> dict[str, Any]:
    p = _store_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save(data: dict[str, Any]) -> None:
    p = _store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _validate(name: str, url: str, method: str, params: list[dict[str, Any]]) -> None:
    if not _NAME_RE.match(name or ""):
        raise ValueError("name must be a lowercase identifier (a-z0-9_), 2-49 chars")
    if not _TEMPLATE_HOST_RE.match(url or ""):
        raise ValueError("url must be a fixed http(s)://host[:port]/path[...] template")
    if method.upper() not in ("GET", "POST"):
        raise ValueError("method must be GET or POST")
    declared = {p.get("name") for p in params}
    for ph in _PLACEHOLDER_RE.findall(url):
        if ph not in declared:
            raise ValueError(f"url placeholder {{{ph}}} has no matching parameter")
    for p in params:
        if not _NAME_RE.match(str(p.get("name") or "")):
            raise ValueError(f"bad parameter name: {p.get('name')!r}")
        if p.get("type", "string") not in _TYPES:
            raise ValueError(f"parameter type must be one of {sorted(_TYPES)}")


def _to_schema(params: list[dict[str, Any]]) -> dict[str, Any]:
    props = {p["name"]: {"type": p.get("type", "string"),
                         **({"description": p["description"]} if p.get("description") else {})}
             for p in params}
    required = [p["name"] for p in params if p.get("required")]
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


# ── public CRUD ──────────────────────────────────────────────────────────────


def list_tools() -> list[dict[str, Any]]:
    data = _load()
    return [data[k] for k in sorted(data, key=lambda k: data[k].get("created_at", 0))]


def get_tool(tid: str) -> Optional[dict[str, Any]]:
    return _load().get(tid)


def create_tool(name: str, url: str, *, description: str = "", title: str = "",
                method: str = "GET", params: Optional[list[dict[str, Any]]] = None,
                enabled: bool = True) -> dict[str, Any]:
    name = str(name or "").strip()
    params = [dict(p) for p in (params or [])]
    _validate(name, url, method, params)
    data = _load()
    if name in data:
        raise ValueError(f"a tool named {name!r} already exists")
    rec = {"id": name, "name": name, "title": str(title or "").strip(),
           "description": str(description or "").strip() or f"Operator tool {name}",
           "method": method.upper(), "url": url, "params": params,
           "enabled": bool(enabled), "created_at": time.time()}
    data[name] = rec
    _save(data)
    return rec


def update_tool(tid: str, **fields: Any) -> dict[str, Any]:
    data = _load()
    if tid not in data:
        raise ValueError("unknown tool")
    rec = dict(data[tid])
    for k in ("title", "description", "url", "method", "params", "enabled"):
        if k in fields and fields[k] is not None:
            rec[k] = fields[k]
    _validate(rec["name"], rec["url"], rec["method"], rec.get("params") or [])
    rec["method"] = str(rec["method"]).upper()
    rec["enabled"] = bool(rec["enabled"])
    data[tid] = rec
    _save(data)
    return rec


def delete_tool(tid: str) -> bool:
    data = _load()
    if tid not in data:
        return False
    del data[tid]
    _save(data)
    return True


def enabled_tool_specs() -> list[dict[str, Any]]:
    """For the copilot: {id, name, description, parameters} per enabled tool."""
    return [{"id": t["id"], "name": t["name"], "description": t["description"],
             "parameters": _to_schema(t.get("params") or [])}
            for t in list_tools() if t.get("enabled")]


# ── SSRF-safe executor ───────────────────────────────────────────────────────


def _fill(template: str, args: dict[str, Any]) -> str:
    """Substitute {param} with URL-encoded arg values. The host is part of the
    fixed template, never an argument, so this can't redirect the request."""
    def _sub(m: "re.Match[str]") -> str:
        return urllib.parse.quote(str(args.get(m.group(1), "")), safe="")
    return _PLACEHOLDER_RE.sub(_sub, template)


def run_tool(tid: str, args: dict[str, Any]) -> dict[str, Any]:
    tool = get_tool(tid)
    if not tool or not tool.get("enabled"):
        raise ValueError(f"unknown or disabled tool: {tid}")
    args = args if isinstance(args, dict) else {}
    missing = [p["name"] for p in (tool.get("params") or [])
               if p.get("required") and not str(args.get(p["name"], "")).strip()]
    if missing:
        raise ValueError(f"missing required parameter(s): {', '.join(missing)}")
    url = _fill(tool["url"], args)
    if not _TEMPLATE_HOST_RE.match(url):  # defensive: still a fixed http(s) host after fill
        raise ValueError("resolved url is not a valid http(s) endpoint")
    method = tool.get("method", "GET").upper()
    data = None
    if method == "POST":
        data = json.dumps({p["name"]: args.get(p["name"]) for p in (tool.get("params") or [])}).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Accept": "application/json", "User-Agent": "sndr-managed-tool/1.0",
                 **({"Content-Type": "application/json"} if data is not None else {})})
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:  # noqa: S310 - fixed operator host
            text = resp.read(2_000_000).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return {"tool": tid, "http_status": exc.code, "error": exc.read(2000).decode("utf-8", "replace")[:500]}
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return {"tool": tid, "error": f"unreachable: {getattr(exc, 'reason', exc)}"}
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        return {"tool": tid, "result_text": text[:_RESULT_CAP]}
    serialized = json.dumps(parsed)
    if len(serialized) > _RESULT_CAP:  # too big for the model context → send trimmed text
        return {"tool": tid, "result_text": serialized[:_RESULT_CAP], "truncated": True}
    return {"tool": tid, "result": parsed}


__all__ = ["list_tools", "get_tool", "create_tool", "update_tool", "delete_tool",
           "enabled_tool_specs", "run_tool"]
