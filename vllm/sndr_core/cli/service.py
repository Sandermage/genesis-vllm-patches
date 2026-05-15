# SPDX-License-Identifier: Apache-2.0
"""C13 (UNIFIED_CONFIG plan 2026-05-09) — `sndr service` lifecycle CLI.

Reads a preset's Y10 `service` block and emits / applies the
appropriate service-management commands per declared backend
(systemd / docker_compose / podman_quadlet / kubernetes / proxmox /
bare_metal).

Subcommands:
  sndr service install <key>   — emit/install the unit file
  sndr service start <key>     — start the service
  sndr service stop <key>      — stop the service
  sndr service status <key>    — query status
  sndr service logs <key>      — tail logs
  sndr service uninstall <key> — remove the unit

Default `--dry-run` mode (operator must add `--yes` to actually
mutate the host). For systemd backend, the unit file is written to
`~/.config/systemd/user/` for user-mode or `/etc/systemd/system/`
for system-mode (latter requires --system + sudo).
"""
from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from . import _io


__all__ = ["add_argparser", "run_install", "run_start", "run_stop",
           "run_status", "run_logs", "run_uninstall"]


def add_argparser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "service",
        help="Service-lifecycle wrapper around Y10 ServiceConfig (UNIFIED_CONFIG C13).",
        description=(
            "Manage the running vllm service via the backend declared in "
            "the preset's Y10 service block. Default --dry-run; --yes to "
            "actually mutate the host."
        ),
    )
    sub = p.add_subparsers(dest="service_cmd", required=True)

    for cmd, helper, fn in (
        ("install", "Render + install the unit file", run_install),
        ("start", "Start the service", run_start),
        ("stop", "Stop the service", run_stop),
        ("status", "Query service status", run_status),
        ("logs", "Tail service logs", run_logs),
        ("uninstall", "Remove the unit file", run_uninstall),
    ):
        sp = sub.add_parser(cmd, help=helper)
        sp.add_argument("config", help="model_config preset key")
        sp.add_argument("--yes", action="store_true",
                          help="Actually mutate the host (default: dry-run).")
        sp.add_argument("--system", action="store_true",
                          help="systemd: write to /etc/systemd/system/ "
                               "(default: ~/.config/systemd/user/).")
        sp.add_argument("--lines", type=int, default=50,
                          help="logs: number of recent lines (default 50).")
        sp.set_defaults(func=fn)


def _resolve(key: str):
    from vllm.sndr_core.model_configs.registry import get
    cfg = get(key)
    if cfg is None:
        _io.warn(f"unknown preset key {key!r}")
        return None
    if cfg.service is None:
        _io.warn(f"preset {key!r} has no Y10 service block; "
                  f"add `service:` to the YAML to use this CLI.")
        return None
    return cfg


def _systemd_unit_path(key: str, *, system: bool) -> Path:
    if system:
        return Path(f"/etc/systemd/system/sndr-{key}.service")
    return Path.home() / ".config/systemd/user" / f"sndr-{key}.service"


_SYSTEMD_RESTART = {
    "no": "no",
    "always": "always",
    "on-failure": "on-failure",
    # Docker-style "unless-stopped" maps cleanly enough to systemd "always":
    # systemctl stop is an explicit manager operation and will not loop-restart.
    "unless-stopped": "always",
}


def _render_systemd_unit(cfg, *, system: bool = False) -> str:
    """Render the [Service] body from cfg.service + cfg.docker."""
    s = cfg.service
    user_line = f"User={s.user}\n" if s.user else ""
    wd_line = f"WorkingDirectory={s.working_dir}\n" if s.working_dir else ""
    env_line = f"EnvironmentFile={s.env_file}\n" if s.env_file else ""
    restart = _SYSTEMD_RESTART.get(s.restart, "on-failure")
    wanted_by = "multi-user.target" if system else "default.target"
    key_arg = shlex.quote(cfg.key)
    return (
        f"[Unit]\n"
        f"Description=SNDR service for {cfg.key}\n"
        f"Wants=network-online.target\n"
        f"After=network-online.target docker.service\n"
        f"\n"
        f"[Service]\n"
        f"Type=simple\n"
        f"{user_line}{wd_line}{env_line}"
        f"ExecStart=/usr/bin/env sndr launch --non-interactive {key_arg}\n"
        f"Restart={restart}\n"
        f"RestartSec=5\n"
        f"\n"
        f"[Install]\n"
        f"WantedBy={wanted_by}\n"
    )


def _systemctl(*args, system: bool, dry_run: bool) -> int:
    cmd = ["systemctl"]
    if not system:
        cmd.append("--user")
    cmd.extend(args)
    if dry_run:
        _io.info(f"[dry-run] would: {' '.join(cmd)}")
        return 0
    if shutil.which("systemctl") is None:
        _io.error("systemctl not on PATH (this host is not systemd-based)")
        return 1
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.stdout:
        print(r.stdout.rstrip())
    if r.stderr:
        print(r.stderr.rstrip())
    return r.returncode


def _docker_cmd(*args, dry_run: bool) -> int:
    cmd = ["docker"] + list(args)
    if dry_run:
        _io.info(f"[dry-run] would: {' '.join(cmd)}")
        return 0
    if shutil.which("docker") is None:
        _io.error("docker not on PATH")
        return 1
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.stdout:
        print(r.stdout.rstrip())
    if r.stderr:
        print(r.stderr.rstrip())
    return r.returncode


# ─── install

def run_install(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    backend = cfg.service.backend
    dry_run = not args.yes

    if backend == "systemd":
        unit_path = _systemd_unit_path(cfg.key, system=args.system)
        body = _render_systemd_unit(cfg, system=args.system)
        if dry_run:
            _io.info(f"[dry-run] would write systemd unit to: {unit_path}")
            print()
            print(body)
            print()
            _io.info(f"[dry-run] would: systemctl"
                       f"{' --user' if not args.system else ''} "
                       f"daemon-reload && enable sndr-{cfg.key}.service")
            return 0
        unit_path.parent.mkdir(parents=True, exist_ok=True)
        unit_path.write_text(body)
        _io.success(f"wrote systemd unit: {unit_path}")
        rc = _systemctl("daemon-reload", system=args.system, dry_run=False)
        if rc != 0:
            return rc
        return _systemctl("enable", f"sndr-{cfg.key}.service",
                           system=args.system, dry_run=False)

    if backend in ("docker_compose", "podman_quadlet"):
        _io.info(f"backend={backend} — install means: ensure compose/quadlet "
                  f"file in place. Genesis does NOT generate compose files; "
                  f"use `sndr launch --dry-run {cfg.key}` to render the docker run "
                  f"line, then wrap it in your compose stack manually.")
        return 0

    if backend == "kubernetes":
        _io.info(f"backend=kubernetes — use `sndr k8s render {cfg.key}` "
                  f"(when implemented) to generate manifests.")
        return 0

    if backend == "bare_metal":
        _io.info(f"backend=bare_metal — no service unit; just run "
                  f"`sndr launch {cfg.key}` from a screen/tmux session.")
        return 0

    _io.warn(f"backend={backend!r} — install not implemented yet")
    return 1


# ─── start / stop / status / logs / uninstall

def run_start(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    dry_run = not args.yes
    if cfg.service.backend == "systemd":
        return _systemctl("start", f"sndr-{cfg.key}.service",
                           system=args.system, dry_run=dry_run)
    if cfg.service.backend in ("docker_compose", "podman_quadlet"):
        container = cfg.docker.container_name if cfg.docker else f"sndr-{cfg.key}"
        return _docker_cmd("start", container, dry_run=dry_run)
    _io.info(f"backend={cfg.service.backend} — start not implemented; "
              f"run `sndr launch {cfg.key}` directly.")
    return 0


def run_stop(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    dry_run = not args.yes
    if cfg.service.backend == "systemd":
        return _systemctl("stop", f"sndr-{cfg.key}.service",
                           system=args.system, dry_run=dry_run)
    if cfg.service.backend in ("docker_compose", "podman_quadlet"):
        container = cfg.docker.container_name if cfg.docker else f"sndr-{cfg.key}"
        return _docker_cmd("stop", container, dry_run=dry_run)
    _io.info(f"backend={cfg.service.backend} — stop not implemented")
    return 0


def run_status(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    if cfg.service.backend == "systemd":
        return _systemctl("status", f"sndr-{cfg.key}.service",
                           system=args.system, dry_run=False)
    if cfg.service.backend in ("docker_compose", "podman_quadlet"):
        container = cfg.docker.container_name if cfg.docker else f"sndr-{cfg.key}"
        return _docker_cmd("ps", "-a", "--filter", f"name={container}",
                            dry_run=False)
    _io.info(f"backend={cfg.service.backend} — status not implemented")
    return 0


def run_logs(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    if cfg.service.backend == "systemd":
        cmd = ["journalctl", "--user-unit" if not args.system else "--unit",
               f"sndr-{cfg.key}.service", "-n", str(args.lines), "--no-pager"]
        if shutil.which("journalctl") is None:
            _io.error("journalctl not on PATH")
            return 1
        r = subprocess.run(cmd, timeout=10)
        return r.returncode
    if cfg.service.backend in ("docker_compose", "podman_quadlet"):
        container = cfg.docker.container_name if cfg.docker else f"sndr-{cfg.key}"
        return _docker_cmd("logs", "--tail", str(args.lines), container,
                            dry_run=False)
    _io.info(f"backend={cfg.service.backend} — logs not implemented")
    return 0


def run_uninstall(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    dry_run = not args.yes
    if cfg.service.backend == "systemd":
        unit_path = _systemd_unit_path(cfg.key, system=args.system)
        if dry_run:
            _io.info(f"[dry-run] would: systemctl disable + remove {unit_path}")
            return 0
        rc = _systemctl("disable", f"sndr-{cfg.key}.service",
                         system=args.system, dry_run=False)
        if unit_path.exists():
            unit_path.unlink()
            _io.success(f"removed unit: {unit_path}")
        return rc
    _io.info(f"backend={cfg.service.backend} — uninstall noop")
    return 0
