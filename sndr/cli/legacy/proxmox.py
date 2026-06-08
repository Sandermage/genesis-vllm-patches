# SPDX-License-Identifier: Apache-2.0
"""C11 (UNIFIED_CONFIG plan 2026-05-09) — `sndr proxmox` Proxmox CLI.

Reads a preset's Y6 `proxmox` block and renders the appropriate
`pct` / `qm` commands for LXC / VM / host modes.

Subcommands:
  sndr proxmox doctor                — sanity-check PVE host (read-only)
  sndr proxmox render <key>          — print pct/qm commands per mode
  sndr proxmox inventory             — list LXCs + VMs on the target node
  sndr proxmox status <key>          — query container/VM state

Default --dry-run; --yes to execute pct/qm. Local execution only —
this CLI does NOT use the PVE API directly (use `pvesh` or the API
directly for cross-node operations).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from typing import Any

from . import _io


__all__ = ["add_argparser", "run_doctor", "run_render", "run_inventory",
           "run_status"]


def add_argparser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "proxmox",
        help="Proxmox deployment wrapper around Y6 ProxmoxConfig (UNIFIED_CONFIG C11).",
        description=(
            "Render + inspect Genesis presets for Proxmox deployment. "
            "Modes: lxc (preferred, bare-metal venv inside LXC) / "
            "vm (Docker-on-VM) / host (bare-metal on PVE host — expert)."
        ),
    )
    sub = p.add_subparsers(dest="proxmox_cmd", required=True)

    p_doc = sub.add_parser("doctor",
                              help="Sanity-check PVE host capabilities.")
    p_doc.set_defaults(func=run_doctor)

    p_inv = sub.add_parser("inventory",
                              help="List LXC containers + VMs on the host.")
    p_inv.add_argument("--json", action="store_true",
                          help="Emit JSON instead of table.")
    p_inv.set_defaults(func=run_inventory)

    for cmd, helper, fn in (
        ("render", "Print pct/qm commands for the preset", run_render),
        ("status", "pct status / qm status for the preset", run_status),
    ):
        sp = sub.add_parser(cmd, help=helper)
        sp.add_argument("config", help="model_config preset key")
        sp.set_defaults(func=fn)


def _resolve(key: str):
    from sndr.model_configs.registry import get
    cfg = get(key)
    if cfg is None:
        _io.warn(f"unknown preset key {key!r}")
        return None
    if cfg.proxmox is None:
        _io.warn(f"preset {key!r} has no Y6 proxmox block; "
                  f"add `proxmox:` to the YAML to use this CLI.")
        return None
    return cfg


def _has_pve() -> bool:
    return (shutil.which("pveversion") is not None
            or shutil.which("pct") is not None)


# ─── doctor

def run_doctor(args: argparse.Namespace) -> int:
    print("sndr proxmox doctor")
    print("─" * 60)
    if not _has_pve():
        _io.error("not a PVE host — `pveversion`/`pct` not on PATH")
        return 1
    # PVE version
    if shutil.which("pveversion"):
        r = subprocess.run(["pveversion"], capture_output=True, text=True,
                            timeout=5)
        if r.returncode == 0:
            print(f"  pveversion: {r.stdout.strip()}")
    # Kernel
    r = subprocess.run(["uname", "-r"], capture_output=True, text=True,
                        timeout=2)
    if r.returncode == 0:
        kernel = r.stdout.strip()
        print(f"  kernel:     {kernel}")
        # Caveat check via genesis caveats registry
        try:
            from sndr.caveats import match_caveats
            facts = {"virtualization": "pve",
                      "os": {"system": "Linux", "release": kernel}}
            triggered = match_caveats(facts)
            for c in triggered:
                if "proxmox" in c.id:
                    _io.warn(f"  caveat: {c.title}")
        except Exception:
            pass
    # GPU passthrough sanity
    if shutil.which("lspci"):
        r = subprocess.run(["lspci"], capture_output=True, text=True,
                            timeout=5)
        if r.returncode == 0:
            n_nv = sum(1 for line in r.stdout.splitlines()
                        if "nvidia" in line.lower())
            print(f"  NVIDIA PCI devices: {n_nv}")
    return 0


# ─── inventory

def run_inventory(args: argparse.Namespace) -> int:
    if not _has_pve():
        _io.error("not a PVE host")
        return 1
    out: dict = {"lxc": [], "vm": []}
    if shutil.which("pct"):
        r = subprocess.run(["pct", "list"], capture_output=True, text=True,
                            timeout=5)
        if r.returncode == 0:
            out["lxc"] = [
                line.strip() for line in r.stdout.splitlines()[1:]
                if line.strip()
            ]
    if shutil.which("qm"):
        r = subprocess.run(["qm", "list"], capture_output=True, text=True,
                            timeout=5)
        if r.returncode == 0:
            out["vm"] = [
                line.strip() for line in r.stdout.splitlines()[1:]
                if line.strip()
            ]
    if args.json:
        import json
        print(json.dumps(out, indent=2))
    else:
        print("sndr proxmox inventory")
        print("─" * 60)
        print(f"  LXC containers ({len(out['lxc'])}):")
        for line in out["lxc"]:
            print(f"    {line}")
        print(f"  VMs ({len(out['vm'])}):")
        for line in out["vm"]:
            print(f"    {line}")
    return 0


# ─── render

def run_render(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    p = cfg.proxmox
    # Fail-fast on unknown mode. ProxmoxConfig.validate() already
    # restricts mode to {lxc, vm, host} at schema load time, but a
    # defensive guard here catches operator overrides that bypass
    # the loader (e.g. constructing the dataclass programmatically).
    _VALID_RENDER_MODES = ("lxc", "vm", "host")
    if p.mode not in _VALID_RENDER_MODES:
        _io.error(
            f"sndr proxmox render: unsupported mode={p.mode!r}. "
            f"Valid modes: {_VALID_RENDER_MODES}. Edit the preset's "
            f"proxmox.mode field."
        )
        return 2
    # All Proxmox modes (lxc / vm / host) emit operator-runnable command
    # *guides*, not auto-executed deployments. Storage class, network
    # bridge, GPU PCI ID, and (in vm mode) the disk image are
    # operator-specific and the renderer leaves them as placeholders.
    # Banner explicitly states "preview / manual-guide" so the operator
    # knows the output is not turn-key.
    print(f"# sndr proxmox render — preset {cfg.key!r}")
    print(f"# mode={p.mode} runtime={p.runtime}")
    print("# STATUS: preview / manual-guide — the lines below are operator")
    print("#         command templates with site-specific placeholders")
    print("#         (storage class, network bridge, GPU PCI ID, etc).")
    print("#         REVIEW + customise every line before executing.")
    print()

    if p.mode == "lxc":
        vmid = p.container_id_or_vmid or 200
        print("# LXC creation (operator must adjust storage / network):")
        print(f"pct create {vmid} \\")
        print("  /var/lib/vz/template/cache/ubuntu-24.04-standard.tar.gz \\")
        print(f"  --hostname sndr-{cfg.key} \\")
        print("  --memory 65536 --cores 8 \\")
        print("  --rootfs local-lvm:64 \\")
        print("  --net0 name=eth0,bridge=vmbr0,ip=dhcp \\")
        print("  --features nesting=1")
        if p.gpu_passthrough:
            print()
            print("# GPU passthrough (replace with `lspci | grep NVIDIA` IDs):")
            print(f"# Edit /etc/pve/lxc/{vmid}.conf and add:")
            print("#   lxc.cgroup2.devices.allow: c 195:* rwm")
            print("#   lxc.cgroup2.devices.allow: c 510:* rwm")
            print("#   lxc.mount.entry: /dev/nvidia0 dev/nvidia0 none bind,optional,create=file")
            print("#   lxc.mount.entry: /dev/nvidiactl dev/nvidiactl none bind,optional,create=file")
            print("#   lxc.mount.entry: /dev/nvidia-uvm dev/nvidia-uvm none bind,optional,create=file")
        print()
        print(f"pct start {vmid}")
        print()
        if p.runtime == "docker":
            print("# Inside the LXC: install Docker via apt repo + keyring")
            print("# (no curl|sh — vetted package, pinning supported, reboot safe).")
            print("#")
            print("# Steps the operator runs (or wraps in a config-management script):")
            print(f"pct exec {vmid} -- bash -c 'install -m 0755 -d /etc/apt/keyrings'")
            print(f"pct exec {vmid} -- bash -c 'curl -fsSL https://download.docker.com/linux/debian/gpg \\")
            print("    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg'")
            print(f"pct exec {vmid} -- bash -c 'chmod a+r /etc/apt/keyrings/docker.gpg'")
            print(f"pct exec {vmid} -- bash -c 'echo \"deb [arch=$(dpkg --print-architecture) \\")
            print("    signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \\")
            print("    $(. /etc/os-release && echo \\\"$VERSION_CODENAME\\\") stable\" \\")
            print("    > /etc/apt/sources.list.d/docker.list'")
            print(f"pct exec {vmid} -- bash -c 'apt-get update && apt-get install -y \\")
            print("    docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin'")
            print("#")
            print("# (Note: LXC docker still requires nesting + manual GPU device passthrough;")
            print("# see `proxmox doctor` output for the cgroup/lxc.mount entries.)")
        elif p.runtime == "venv":
            print("# Inside the LXC: bootstrap venv + install sndr:")
            print(f"pct exec {vmid} -- bash -c 'apt update && apt install -y python3.12 python3.12-venv git'")
            print(f"pct exec {vmid} -- bash -c 'python3.12 -m venv /opt/sndr-venv && /opt/sndr-venv/bin/pip install vllm-sndr-core'")

    elif p.mode == "vm":
        vmid = p.container_id_or_vmid or 100
        print("# VM creation (qm-based):")
        print(f"qm create {vmid} \\")
        print(f"  --name sndr-{cfg.key} \\")
        print("  --memory 65536 --cores 8 \\")
        print("  --net0 model=virtio,bridge=vmbr0 \\")
        print("  --scsi0 local-lvm:64 \\")
        print("  --ostype l26")
        if p.gpu_passthrough:
            print("# GPU passthrough — operator runs (replace 0000:01:00.0 with actual):")
            print(f"qm set {vmid} --hostpci0 0000:01:00.0,pcie=1")
        print(f"qm start {vmid}")
    elif p.mode == "host":
        print("# Bare-metal on PVE host (NO ISOLATION — expert only):")
        print("# Just install python venv + sndr directly on the host:")
        print("apt install -y python3.12 python3.12-venv")
        print("python3.12 -m venv /opt/sndr-venv")
        print("/opt/sndr-venv/bin/pip install vllm-sndr-core")
        print(f"# Then: sndr launch {cfg.key}")
    # `else` removed — every mode reaching this point is in
    # _VALID_RENDER_MODES (guarded above). Adding a new mode means
    # updating both the tuple and adding its own branch.
    return 0


# ─── status

def run_status(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    p = cfg.proxmox
    if not _has_pve():
        _io.error("not a PVE host")
        return 1
    if p.container_id_or_vmid is None:
        _io.warn(f"preset {args.config!r} has no container_id_or_vmid declared")
        return 1
    if p.mode == "lxc":
        cmd = ["pct", "status", str(p.container_id_or_vmid)]
    elif p.mode == "vm":
        cmd = ["qm", "status", str(p.container_id_or_vmid)]
    else:
        _io.info(f"mode={p.mode} — no status command")
        return 0
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    if r.stdout:
        print(r.stdout.rstrip())
    if r.stderr:
        print(r.stderr.rstrip())
    return r.returncode
