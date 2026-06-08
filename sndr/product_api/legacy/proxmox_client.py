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
        "disk_used": raw.get("disk"),
        "net_in": raw.get("netin"), "net_out": raw.get("netout"),
        "disk_read": raw.get("diskread"), "disk_write": raw.get("diskwrite"),
        "uptime": raw.get("uptime"),
        "tags": _tag_list(raw.get("tags")),
        "sndr_preset": _preset_from_tags(raw.get("tags")),
        "template": bool(raw.get("template")),
    }


def _parse_pve_opts(value: Any) -> dict[str, str]:
    """Parse a Proxmox config value like ``virtio=AA:BB,bridge=vmbr0,size=8G``
    into a dict. A leading positional token (no ``=``) is stored as ``_first``."""
    out: dict[str, str] = {}
    for i, part in enumerate(str(value).split(",")):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
        elif i == 0:
            out["_first"] = part
    return out


def _pci_kind(cls: Any) -> str:
    """Coarse device class from a PCI class code like ``0x030000``."""
    c = str(cls or "")
    head = c[2:4] if c.startswith("0x") else c[:2]
    return {"03": "gpu", "04": "audio", "0c": "usb", "02": "net", "01": "storage"}.get(head, "pci")


def _pretty_pci(dev: dict[str, Any]) -> str:
    """A short human label, e.g. ``NVIDIA RTX A5000`` from a PCI hardware entry."""
    name = dev.get("device_name") or dev.get("id") or "device"
    m = re.search(r"\[([^\]]+)\]", name)
    short = m.group(1) if m else name
    vendor = dev.get("vendor_name") or ""
    if "NVIDIA" in vendor:
        v = "NVIDIA"
    elif "AMD" in vendor or "ATI" in vendor:
        v = "AMD"
    elif "Intel" in vendor:
        v = "Intel"
    else:
        v = vendor.split()[0] if vendor else ""
    return f"{v} {short}".strip()


def _pci_index(node: str) -> dict[str, dict[str, Any]]:
    """Map of PCI id → hardware entry for a node (best-effort, empty on error)."""
    try:
        return {str(d.get("id")): d for d in (_api_get(f"nodes/{node}/hardware/pci") or [])}
    except Exception:
        return {}


def guest_detail(node: str, kind: str, vmid: int) -> dict[str, Any]:
    """Rich per-guest detail: CPU topology, memory, OS, BIOS, boot, GPU
    passthrough, disks, networks and (via the guest agent) IPs. Read-only."""
    a = availability()
    if not a["available"]:
        return {"available": False, "error": a["error"]}
    typ = "qemu" if kind == "vm" else "lxc"
    try:
        cfg = _api_get(f"nodes/{node}/{typ}/{vmid}/config") or {}
        try:
            st = _api_get(f"nodes/{node}/{typ}/{vmid}/status/current") or {}
        except Exception:
            st = {}
        # Passthrough devices — resolve each address to a real name (e.g. an
        # "RTX A5000") via the node's PCI inventory, and classify it.
        pci = _pci_index(node)

        def _resolve(addr: str) -> dict[str, Any]:
            entry = pci.get(addr) or pci.get(addr + ".0") or next(
                (d for i, d in pci.items() if i.startswith(addr)), None)
            if entry:
                return {"address": addr, "name": _pretty_pci(entry), "kind": _pci_kind(entry.get("class"))}
            return {"address": addr, "name": addr, "kind": "pci"}

        devices: list[dict[str, Any]] = []
        for k, v in sorted(cfg.items()):
            if k.startswith("hostpci"):
                devices.append(_resolve(str(v).split(",")[0]))
            elif k.startswith("usb") and k[3:].isdigit():
                devices.append({"address": str(v).split(",")[0], "name": f"USB {str(v).split(',')[0]}", "kind": "usb"})
        if kind == "lxc":
            for k, v in sorted(cfg.items()):
                if k.startswith("dev") and "nvidia" in str(v).lower():
                    devices.append({"address": str(v), "name": "NVIDIA GPU device", "kind": "gpu"})
        # Disks: scsiN/virtioN/sataN/ideN/rootfs/mpN (skip cdrom 'none').
        disks = []
        for k in sorted(cfg):
            is_disk = (k == "rootfs") or any(
                k.startswith(p) and k[len(p):].isdigit() for p in ("scsi", "virtio", "sata", "ide", "mp"))
            if not is_disk:
                continue
            opts = _parse_pve_opts(cfg[k])
            vol = opts.get("_first", "")
            if vol == "none" or "media=cdrom" in str(cfg[k]):
                continue
            disks.append({"id": k, "volume": vol, "size": opts.get("size"),
                          "storage": vol.split(":")[0] if ":" in vol else None})
        # Networks: netN → model/mac/bridge/ip.
        nets = []
        for k in sorted(cfg):
            if not (k.startswith("net") and k[3:].isdigit()):
                continue
            opts = _parse_pve_opts(cfg[k])
            model = next((m for m in ("virtio", "e1000", "vmxnet3", "rtl8139") if m in opts), None)
            nets.append({"id": k, "model": model or ("veth" if kind == "lxc" else None),
                         "mac": opts.get(model) if model else opts.get("hwaddr"),
                         "bridge": opts.get("bridge"), "ip": opts.get("ip"),
                         "name": opts.get("name")})
        # IPs reported by the guest agent (qemu) — non-loopback IPv4.
        agent_ips: list[str] = []
        nics = st.get("nics")
        for nic in (nics.values() if isinstance(nics, dict) else (nics or [])):
            for addr in (nic.get("ip-addresses") or []):
                ip = addr.get("ip-address", "")
                if addr.get("ip-address-type") == "ipv4" and ip and not ip.startswith("127."):
                    agent_ips.append(ip)
        mem = cfg.get("memory")
        ha = st.get("ha")
        return {
            "available": True, "error": None, "vmid": vmid, "kind": kind, "node": node,
            "cores": cfg.get("cores"), "sockets": cfg.get("sockets"), "cpu_type": cfg.get("cpu"),
            "memory_mb": int(mem) if str(mem).isdigit() else None,
            "swap_mb": int(cfg["swap"]) if str(cfg.get("swap", "")).isdigit() else None,
            "balloon": cfg.get("balloon"), "bios": cfg.get("bios"), "machine": cfg.get("machine"),
            "ostype": cfg.get("ostype"),
            "onboot": str(cfg.get("onboot")) in ("1", "True"),
            "boot_order": cfg.get("boot"),
            "agent_enabled": str(cfg.get("agent", "")).startswith("1"),
            "qmpstatus": st.get("qmpstatus") or st.get("status"),
            "ha_managed": (ha.get("managed") == 1) if isinstance(ha, dict) else None,
            "unprivileged": (str(cfg.get("unprivileged")) == "1") if kind == "lxc" else None,
            "features": cfg.get("features"),
            "description": (str(cfg.get("description")).strip() or None) if cfg.get("description") else None,
            "tags": _tag_list(cfg.get("tags")),
            "devices": devices, "disks": disks, "networks": nets, "agent_ips": agent_ips,
        }
    except Exception as exc:
        return {"available": False, "error": _describe(exc)}


def node_detail(node: str) -> dict[str, Any]:
    """Rich detail for a Proxmox node: CPU model/topology, kernel, PVE version,
    load average, swap, root filesystem and the display GPUs present. Read-only."""
    a = availability()
    if not a["available"]:
        return {"available": False, "error": a["error"]}
    try:
        st = _api_get(f"nodes/{node}/status") or {}
        ci = st.get("cpuinfo") or {}
        ck = st.get("current-kernel") or {}
        rootfs = st.get("rootfs") or {}
        swap = st.get("swap") or {}
        gpus = sorted({_pretty_pci(d) for d in _pci_index(node).values()
                       if _pci_kind(d.get("class")) == "gpu"})
        return {
            "available": True, "error": None, "node": node,
            "cpu_model": ci.get("model"), "cpu_cores": ci.get("cores"), "cpu_threads": ci.get("cpus"),
            "cpu_sockets": ci.get("sockets"), "cpu_mhz": ci.get("mhz"), "cpu_vendor": ci.get("vendor"),
            "kernel": ck.get("release") or st.get("kversion"),
            "pve_version": st.get("pveversion"),
            "loadavg": st.get("loadavg") or [],
            "swap_total": swap.get("total"), "swap_used": swap.get("used"),
            "rootfs_total": rootfs.get("total"), "rootfs_used": rootfs.get("used"),
            "gpus": gpus, "uptime": st.get("uptime"),
        }
    except Exception as exc:
        return {"available": False, "error": _describe(exc)}


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
