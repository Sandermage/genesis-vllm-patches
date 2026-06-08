# SPDX-License-Identifier: Apache-2.0
"""Read-only Proxmox VE client for the admin panel (virtualization mode).

Mirrors :mod:`k8s_client`: it authenticates with an operator-provided API token
(never a stored password), degrades gracefully to ``{available: False, error}``
when Proxmox is not configured or unreachable, and keeps the data-shaping
functions PURE so the GPU-host operator's view — nodes, VMs and LXC with their
resources plus the SNDR preset they host — is unit-testable without a live PVE.

Configuration is by environment (no secrets in the catalog)::

    SNDR_PROXMOX_HOST          https://pve.local:8006  (or host / host:port)
    SNDR_PROXMOX_TOKEN_ID      root@pam!sndr
    SNDR_PROXMOX_TOKEN_SECRET  the token's secret UUID
    SNDR_PROXMOX_VERIFY_SSL    "0" to accept a self-signed PVE certificate

SNDR linkage: a guest is linked to the preset that defines it via a Proxmox tag
``sndr-preset-<id>`` (the lxc_proxmox renderer stamps it), so the panel maps a
running VM/LXC back to its preset exactly like the docker/k8s identity does.
"""
from __future__ import annotations

import json
import os
import re
import ssl
import urllib.error
import urllib.request
from typing import Any, Optional

_PRESET_TAG_PREFIX = "sndr-preset-"


def _config() -> dict[str, Any]:
    host = (os.environ.get("SNDR_PROXMOX_HOST") or "").strip()
    if host:
        if not host.startswith(("http://", "https://")):
            host = "https://" + host
        # Default to the PVE API port when none is given.
        rest = host.split("://", 1)[1]
        if ":" not in rest.split("/", 1)[0]:
            host = host.rstrip("/") + ":8006"
    return {
        "host": host or None,
        "token_id": (os.environ.get("SNDR_PROXMOX_TOKEN_ID") or "").strip() or None,
        "token_secret": (os.environ.get("SNDR_PROXMOX_TOKEN_SECRET") or "").strip() or None,
        "verify_ssl": (os.environ.get("SNDR_PROXMOX_VERIFY_SSL", "1").strip().lower()
                       not in ("0", "false", "no", "off")),
    }


def availability() -> dict[str, Any]:
    """Why Proxmox mode is or isn't usable — without touching the network."""
    c = _config()
    if not c["host"]:
        return {"available": False, "configured": False,
                "error": "Proxmox not configured — set SNDR_PROXMOX_HOST + SNDR_PROXMOX_TOKEN_ID + SNDR_PROXMOX_TOKEN_SECRET"}
    if not (c["token_id"] and c["token_secret"]):
        return {"available": False, "configured": False,
                "error": "Proxmox API token missing — set SNDR_PROXMOX_TOKEN_ID and SNDR_PROXMOX_TOKEN_SECRET"}
    return {"available": True, "configured": True, "error": None, "host": c["host"]}


def _describe(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 401:
            return "Proxmox rejected the API token (401) — check SNDR_PROXMOX_TOKEN_ID/SECRET and its privileges"
        return f"Proxmox API HTTP {exc.code}"
    if isinstance(exc, urllib.error.URLError):
        return f"Proxmox unreachable: {getattr(exc, 'reason', exc)}"
    return str(exc)[:200]


def _api_get(path: str, *, timeout: float = 6.0) -> Any:
    """GET ``/api2/json/<path>`` and return the parsed ``data``. Raises on error."""
    c = _config()
    if not (c["host"] and c["token_id"] and c["token_secret"]):
        raise RuntimeError("Proxmox not configured")
    url = f"{c['host']}/api2/json/{path.lstrip('/')}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"PVEAPIToken={c['token_id']}={c['token_secret']}",
        "Accept": "application/json",
    })
    ctx = None
    if c["host"].startswith("https") and not c["verify_ssl"]:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:  # noqa: S310 - fixed scheme + token
        body = resp.read().decode("utf-8", "replace")
    return (json.loads(body) or {}).get("data")


# ── pure shaping (unit-tested without a live Proxmox) ────────────────────────

def _pct(used: Any, total: Any) -> Optional[float]:
    try:
        u, t = float(used), float(total)
        return round(100.0 * u / t, 1) if t > 0 else None
    except (TypeError, ValueError):
        return None


def _preset_from_tags(tags: Any) -> Optional[str]:
    for tag in re.split(r"[;,\s]+", str(tags or "")):
        tag = tag.strip()
        if tag.startswith(_PRESET_TAG_PREFIX) and len(tag) > len(_PRESET_TAG_PREFIX):
            return tag[len(_PRESET_TAG_PREFIX):]
    return None


def _tag_list(tags: Any) -> list[str]:
    return [t.strip() for t in re.split(r"[;,\s]+", str(tags or "")) if t.strip()]


def shape_node(raw: dict[str, Any]) -> dict[str, Any]:
    """A Proxmox host node (from ``/cluster/resources?type=node``). Pure."""
    cpu = raw.get("cpu")
    return {
        "name": raw.get("node") or raw.get("name") or raw.get("id"),
        "status": raw.get("status") or ("online" if raw.get("uptime") else "unknown"),
        "online": (raw.get("status") == "online"),
        "cpu_pct": round(float(cpu) * 100, 1) if isinstance(cpu, (int, float)) else None,
        "cpu_cores": raw.get("maxcpu"),
        "mem_used": raw.get("mem"), "mem_total": raw.get("maxmem"),
        "mem_pct": _pct(raw.get("mem"), raw.get("maxmem")),
        "disk_used": raw.get("disk"), "disk_total": raw.get("maxdisk"),
        "disk_pct": _pct(raw.get("disk"), raw.get("maxdisk")),
        "uptime": raw.get("uptime"),
        "level": raw.get("level") or "",
    }


def shape_guest(raw: dict[str, Any]) -> dict[str, Any]:
    """A VM (qemu) or container (lxc) from ``/cluster/resources``. Pure."""
    typ = raw.get("type")
    cpu = raw.get("cpu")
    return {
        "vmid": raw.get("vmid"),
        "name": raw.get("name") or f"{typ}/{raw.get('vmid')}",
        "kind": "vm" if typ == "qemu" else ("lxc" if typ == "lxc" else str(typ)),
        "status": raw.get("status"),
        "running": (raw.get("status") == "running"),
        "node": raw.get("node"),
        "cpu_pct": round(float(cpu) * 100, 1) if isinstance(cpu, (int, float)) else None,
        "cpu_cores": raw.get("maxcpu"),
        "mem_used": raw.get("mem"), "mem_total": raw.get("maxmem"),
        "mem_pct": _pct(raw.get("mem"), raw.get("maxmem")),
        "disk_total": raw.get("maxdisk"),
        "uptime": raw.get("uptime"),
        "tags": _tag_list(raw.get("tags")),
        "sndr_preset": _preset_from_tags(raw.get("tags")),
        "template": bool(raw.get("template")),
    }


# ── live calls (graceful) ────────────────────────────────────────────────────

def _resources() -> list[dict[str, Any]]:
    """One call returns every node / VM / LXC / storage in the (single-node ok)
    cluster — the efficient PVE primitive."""
    return list(_api_get("cluster/resources") or [])


def cluster_status() -> dict[str, Any]:
    a = availability()
    if not a["available"]:
        return {"available": False, "configured": a.get("configured", False), "error": a["error"],
                "node_count": 0, "vm_count": 0, "lxc_count": 0}
    try:
        res = _resources()
        nodes = [r for r in res if r.get("type") == "node"]
        vms = [r for r in res if r.get("type") == "qemu" and not r.get("template")]
        lxc = [r for r in res if r.get("type") == "lxc" and not r.get("template")]
        managed = sum(1 for r in (vms + lxc) if _preset_from_tags(r.get("tags")))
        return {
            "available": True, "configured": True, "error": None, "host": a["host"],
            "node_count": len(nodes), "nodes_online": sum(1 for n in nodes if n.get("status") == "online"),
            "vm_count": len(vms), "vm_running": sum(1 for v in vms if v.get("status") == "running"),
            "lxc_count": len(lxc), "lxc_running": sum(1 for v in lxc if v.get("status") == "running"),
            "sndr_managed": managed,
        }
    except Exception as exc:
        return {"available": False, "configured": True, "error": _describe(exc),
                "node_count": 0, "vm_count": 0, "lxc_count": 0}


def list_nodes() -> dict[str, Any]:
    a = availability()
    if not a["available"]:
        return {"available": False, "error": a["error"], "nodes": []}
    try:
        res = _resources()
        return {"available": True, "error": None,
                "nodes": [shape_node(r) for r in res if r.get("type") == "node"]}
    except Exception as exc:
        return {"available": False, "error": _describe(exc), "nodes": []}


def list_guests() -> dict[str, Any]:
    a = availability()
    if not a["available"]:
        return {"available": False, "error": a["error"], "guests": []}
    try:
        res = _resources()
        guests = [shape_guest(r) for r in res if r.get("type") in ("qemu", "lxc")]
        guests.sort(key=lambda g: (g["kind"], g.get("vmid") or 0))
        return {"available": True, "error": None, "guests": guests}
    except Exception as exc:
        return {"available": False, "error": _describe(exc), "guests": []}


__all__ = [
    "availability", "cluster_status", "list_nodes", "list_guests",
    "shape_node", "shape_guest",
]
