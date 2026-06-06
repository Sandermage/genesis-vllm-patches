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


def _kubectl(*args, dry_run: bool, timeout: int = 30) -> int:
    """Run `kubectl <args>` against the cluster the operator's kubeconfig
    points at. Honours dry-run + missing-binary fallback identically to
    the docker / systemctl helpers above."""
    cmd = ["kubectl"] + list(args)
    if dry_run:
        _io.info(f"[dry-run] would: {' '.join(cmd)}")
        return 0
    if shutil.which("kubectl") is None:
        _io.error(
            "kubectl not on PATH — install kubectl OR run `sndr k8s render "
            "<preset> | kubectl apply -f -` from a node that has it"
        )
        return 1
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.stdout:
        print(r.stdout.rstrip())
    if r.stderr:
        print(r.stderr.rstrip())
    return r.returncode


def _k8s_object_name(cfg) -> str:
    """Return the Deployment / Service name used by the k8s renderer.
    Matches the kebab-case derivation in
    `compat/model_config_cli.py::_render_kubernetes` so kubectl can
    target the resources without re-parsing the manifest."""
    if cfg.docker is not None and cfg.docker.container_name:
        return cfg.docker.container_name.replace("_", "-").lower()
    return f"sndr-{cfg.key}".replace("_", "-").lower()


def _k8s_manifest_path(cfg) -> Path:
    """Where the operator's previous `sndr service install` wrote the
    rendered Kubernetes manifest for this preset. Install puts it next
    to the docker_compose YAML so both backends share `~/.sndr/`."""
    return Path.home() / ".sndr" / "k8s" / f"{cfg.key}.yaml"


def _k8s_namespace(cfg) -> str:
    """Namespace declared by the operator on the Y10 service block, or
    the cluster default ('default'). Service objects don't carry the
    namespace today, so we look at service.options for an override and
    fall back to 'default'."""
    options = getattr(cfg.service, "options", None) or {}
    ns = options.get("namespace") if isinstance(options, dict) else None
    return ns if isinstance(ns, str) and ns else "default"


def _pct(*args, dry_run: bool, timeout: int = 30) -> int:
    """Proxmox VE container CLI. `pct` is only available on PVE hosts;
    the helper checks for it and emits a clean error otherwise."""
    cmd = ["pct"] + list(args)
    if dry_run:
        _io.info(f"[dry-run] would: {' '.join(cmd)}")
        return 0
    if shutil.which("pct") is None:
        _io.error(
            "pct not on PATH — run `sndr service <cmd>` on the Proxmox VE "
            "host where the LXC container lives"
        )
        return 1
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.stdout:
        print(r.stdout.rstrip())
    if r.stderr:
        print(r.stderr.rstrip())
    return r.returncode


def _proxmox_ctid(cfg) -> str:
    """Resolve the operator-overridable Proxmox CTID for this preset.

    Looks in three places, first hit wins:
      1. `proxmox.container_id_or_vmid` on the Y6 ProxmoxConfig block.
      2. `service.options.ctid` on the Y10 ServiceConfig block.
      3. SNDR_CTID env var (mirrors the lxc_proxmox renderer's default).

    Falls back to "200" — the same default used by the lxc_proxmox
    renderer when nothing else is specified."""
    import os
    proxmox = getattr(cfg, "proxmox", None)
    if proxmox is not None:
        ctid = getattr(proxmox, "container_id_or_vmid", None)
        if ctid is not None:
            return str(ctid)
    options = getattr(cfg.service, "options", None) or {}
    if isinstance(options, dict) and "ctid" in options:
        return str(options["ctid"])
    env_ctid = os.environ.get("SNDR_CTID")
    if env_ctid:
        return env_ctid
    return "200"


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

    if backend == "docker_compose":
        # Render the compose YAML to the canonical path. The compose
        # CLI already has render logic; service install delegates to
        # it so the two surfaces stay aligned.
        from vllm.sndr_core.cli.compose import render_compose_yaml
        try:
            yaml_body = render_compose_yaml(cfg)
        except Exception as e:
            _io.error(f"compose render failed: {type(e).__name__}: {e}")
            return 1
        target_dir = Path.home() / ".sndr" / "compose"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / f"{cfg.key}.yml"
        target_file.write_text(yaml_body)
        _io.info(f"backend=docker_compose — wrote {target_file} "
                 f"({len(yaml_body)} bytes)")
        _io.info(f"  start with: docker compose -f {target_file} up -d")
        _io.info(f"  or:         sndr compose up {cfg.key}")
        return 0
    if backend == "podman_quadlet":
        # The `sndr quadlet` subcommand renders systemd / podman quadlet
        # units when present; install path just points at it. Quadlet
        # generation is a parallel CLI surface that hasn't been wired
        # into this entry point yet — operator-driven for now.
        _io.info(f"backend=podman_quadlet — render with "
                 f"`sndr quadlet render {cfg.key}` and place under "
                 f"~/.config/containers/systemd/.")
        return 0

    if backend == "kubernetes":
        # Render the Deployment+Service+ConfigMap manifest and either
        # apply it directly (operator passed --yes) or write it next to
        # the compose / quadlet artefacts under ~/.sndr/k8s/.
        from vllm.sndr_core.compat.model_config_cli import _render_kubernetes
        manifest = _render_kubernetes(cfg)
        target_dir = Path.home() / ".sndr" / "k8s"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / f"{cfg.key}.yaml"
        target_file.write_text(manifest)
        _io.info(f"backend=kubernetes — wrote {target_file} "
                 f"({len(manifest)} bytes)")
        if not args.yes:
            _io.info(f"  apply with: kubectl apply -f {target_file}")
            _io.info(f"  or re-run:  sndr service install {cfg.key} --yes")
            return 0
        ns = _k8s_namespace(cfg)
        return _kubectl(
            "apply", "-n", ns, "-f", str(target_file), dry_run=False,
        )

    if backend == "proxmox":
        # Render the runnable LXC deployment script and either execute it
        # (operator passed --yes, we run it on the local PVE host) or
        # leave it on disk for the operator to scp + execute on their PVE
        # box. The renderer is idempotent so re-runs are safe.
        from vllm.sndr_core.compat.model_config_cli import _render_lxc_proxmox
        script_body = _render_lxc_proxmox(cfg)
        target_dir = Path.home() / ".sndr" / "proxmox"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / f"{cfg.key}.sh"
        target_file.write_text(script_body)
        target_file.chmod(0o755)
        _io.info(f"backend=proxmox — wrote {target_file} "
                 f"({len(script_body)} bytes)")
        if not args.yes:
            _io.info(f"  run with:  bash {target_file}")
            _io.info(f"  remote:    scp {target_file} pve-host: && "
                     f"ssh pve-host bash $(basename {target_file})")
            return 0
        if shutil.which("pct") is None:
            _io.error("pct not on PATH — this host is not Proxmox VE. "
                      "Copy the script to your PVE host and run it there.")
            return 1
        r = subprocess.run(["bash", str(target_file)], timeout=600)
        return r.returncode

    if backend == "bare_metal":
        _io.info(f"backend=bare_metal — no service unit; just run "
                  f"`sndr launch {cfg.key}` from a screen/tmux session.")
        return 0

    _io.warn(f"backend={backend!r} — install not implemented yet")
    return 1


# ─── start / stop / status / logs / uninstall

def _compose_file_path(cfg) -> Path:
    """Where service install wrote the compose YAML for this preset."""
    return Path.home() / ".sndr" / "compose" / f"{cfg.key}.yml"


def run_start(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    dry_run = not args.yes
    if cfg.service.backend == "systemd":
        return _systemctl("start", f"sndr-{cfg.key}.service",
                           system=args.system, dry_run=dry_run)
    if cfg.service.backend == "docker_compose":
        # Prefer compose-level lifecycle. Falls back to raw `docker start
        # <container>` when the compose file isn't where install placed
        # it (operator may have copied it elsewhere).
        compose_file = _compose_file_path(cfg)
        if compose_file.is_file():
            return _docker_cmd("compose", "-f", str(compose_file), "up", "-d",
                               dry_run=dry_run)
        container = cfg.docker.container_name if cfg.docker else f"sndr-{cfg.key}"
        return _docker_cmd("start", container, dry_run=dry_run)
    if cfg.service.backend == "podman_quadlet":
        return _systemctl("start", f"sndr-{cfg.key}.service",
                           system=args.system, dry_run=dry_run)
    if cfg.service.backend == "kubernetes":
        # Scale the Deployment from 0 → 1 replica. The first install puts
        # the manifest under ~/.sndr/k8s/; if the operator skipped install
        # they get a clear error from kubectl rather than from us.
        ns = _k8s_namespace(cfg)
        name = _k8s_object_name(cfg)
        return _kubectl(
            "scale", "-n", ns, f"deployment/{name}", "--replicas=1",
            dry_run=dry_run,
        )
    if cfg.service.backend == "proxmox":
        return _pct("start", _proxmox_ctid(cfg), dry_run=dry_run)
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
    if cfg.service.backend == "docker_compose":
        compose_file = _compose_file_path(cfg)
        if compose_file.is_file():
            return _docker_cmd("compose", "-f", str(compose_file), "down",
                               dry_run=dry_run)
        container = cfg.docker.container_name if cfg.docker else f"sndr-{cfg.key}"
        return _docker_cmd("stop", container, dry_run=dry_run)
    if cfg.service.backend == "podman_quadlet":
        return _systemctl("stop", f"sndr-{cfg.key}.service",
                           system=args.system, dry_run=dry_run)
    if cfg.service.backend == "kubernetes":
        # Scale Deployment down to 0; pods terminate, manifest stays.
        ns = _k8s_namespace(cfg)
        name = _k8s_object_name(cfg)
        return _kubectl(
            "scale", "-n", ns, f"deployment/{name}", "--replicas=0",
            dry_run=dry_run,
        )
    if cfg.service.backend == "proxmox":
        return _pct("stop", _proxmox_ctid(cfg), dry_run=dry_run)
    _io.info(f"backend={cfg.service.backend} — stop not implemented")
    return 0


def run_status(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    backend = cfg.service.backend
    if backend == "systemd":
        return _systemctl("status", f"sndr-{cfg.key}.service",
                           system=args.system, dry_run=False)
    if backend == "docker_compose":
        # Compose-level ps so the operator sees service-defined state
        # (replicas, healthchecks). Fall back to docker ps when the
        # compose file is absent.
        compose_file = _compose_file_path(cfg)
        if compose_file.is_file():
            return _docker_cmd("compose", "-f", str(compose_file), "ps",
                               dry_run=False)
        container = cfg.docker.container_name if cfg.docker else f"sndr-{cfg.key}"
        return _docker_cmd("ps", "-a", "--filter", f"name={container}",
                            dry_run=False)
    if backend == "podman_quadlet":
        # Quadlet generates systemd units, so status goes through
        # systemctl, not docker — docker isn't even installed in many
        # quadlet deployments.
        return _systemctl("status", f"sndr-{cfg.key}.service",
                           system=args.system, dry_run=False)
    if backend == "kubernetes":
        # Show Deployment summary (replicas, age, image) + the bound
        # Service so the operator can see the cluster-side state in one
        # call. `kubectl get` is read-only.
        ns = _k8s_namespace(cfg)
        name = _k8s_object_name(cfg)
        return _kubectl(
            "get", "-n", ns, f"deployment/{name}", f"service/{name}",
            "-o", "wide", dry_run=False,
        )
    if backend == "proxmox":
        # pct status returns "status: running|stopped|...". Same one-shot
        # contract the docker-compose branch above provides.
        return _pct("status", _proxmox_ctid(cfg), dry_run=False)
    _io.info(f"backend={backend} — status not implemented")
    return 0


def run_logs(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    backend = cfg.service.backend
    if backend == "systemd":
        cmd = ["journalctl", "--user-unit" if not args.system else "--unit",
               f"sndr-{cfg.key}.service", "-n", str(args.lines), "--no-pager"]
        if shutil.which("journalctl") is None:
            _io.error("journalctl not on PATH")
            return 1
        r = subprocess.run(cmd, timeout=10)
        return r.returncode
    if backend == "docker_compose":
        compose_file = _compose_file_path(cfg)
        if compose_file.is_file():
            return _docker_cmd(
                "compose", "-f", str(compose_file), "logs",
                "--tail", str(args.lines),
                dry_run=False,
            )
        container = cfg.docker.container_name if cfg.docker else f"sndr-{cfg.key}"
        return _docker_cmd("logs", "--tail", str(args.lines), container,
                            dry_run=False)
    if backend == "podman_quadlet":
        # journalctl streams the quadlet-managed unit output; docker
        # would not see it because podman runs the workload.
        cmd = ["journalctl", "--user-unit" if not args.system else "--unit",
               f"sndr-{cfg.key}.service", "-n", str(args.lines), "--no-pager"]
        if shutil.which("journalctl") is None:
            _io.error("journalctl not on PATH")
            return 1
        r = subprocess.run(cmd, timeout=10)
        return r.returncode
    if backend == "kubernetes":
        # `kubectl logs deployment/<name>` picks the most-recent pod by
        # default, which matches the operator's mental model when they
        # ran `sndr service status` and saw a single replica.
        ns = _k8s_namespace(cfg)
        name = _k8s_object_name(cfg)
        return _kubectl(
            "logs", "-n", ns, f"deployment/{name}",
            "--tail", str(args.lines), dry_run=False, timeout=60,
        )
    if backend == "proxmox":
        # The LXC renderer writes the inner launch.sh to /opt/sndr-venv/
        # inside the CT; journalctl picks up any systemd-wrapped variant.
        # We try journalctl first (operator-friendly default) and fall
        # back to a foreground `pct exec` log dump when systemd has no
        # service unit registered.
        ctid = _proxmox_ctid(cfg)
        # Prefer journalctl inside the CT (one round-trip to PVE host).
        return _pct(
            "exec", ctid, "--", "bash", "-lc",
            f"journalctl --no-pager -n {args.lines} -u 'vllm*' 2>/dev/null "
            "|| journalctl --no-pager -n " + str(args.lines),
            dry_run=False, timeout=30,
        )
    _io.info(f"backend={backend} — logs not implemented")
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
    if cfg.service.backend == "docker_compose":
        # Compose `down --volumes` removes containers + named volumes but
        # leaves the manifest on disk. Drop the rendered YAML too so the
        # operator can re-install from scratch.
        compose_file = _compose_file_path(cfg)
        rc = 0
        if compose_file.is_file():
            rc = _docker_cmd("compose", "-f", str(compose_file), "down",
                             "--volumes", dry_run=dry_run)
            if not dry_run and rc == 0:
                compose_file.unlink()
                _io.success(f"removed compose manifest: {compose_file}")
        return rc
    if cfg.service.backend == "podman_quadlet":
        # Quadlet units live under ~/.config/containers/systemd/ — disable
        # them and remove the file. `daemon-reload` is the operator's
        # responsibility (we don't run systemctl when --yes is absent).
        if dry_run:
            _io.info(f"[dry-run] would: systemctl disable sndr-{cfg.key}.service "
                     "+ remove quadlet file under ~/.config/containers/systemd/")
            return 0
        rc = _systemctl("disable", f"sndr-{cfg.key}.service",
                         system=args.system, dry_run=False)
        quadlet_path = (
            Path.home() / ".config" / "containers" / "systemd"
            / f"sndr-{cfg.key}.container"
        )
        if quadlet_path.exists():
            quadlet_path.unlink()
            _io.success(f"removed quadlet: {quadlet_path}")
        return rc
    if cfg.service.backend == "kubernetes":
        # Delete the Deployment + Service via `kubectl delete -f` to make
        # the operation match the install-time apply contract exactly.
        manifest = _k8s_manifest_path(cfg)
        if not manifest.is_file():
            _io.warn(f"no manifest at {manifest} — nothing to delete")
            return 0
        ns = _k8s_namespace(cfg)
        rc = _kubectl(
            "delete", "-n", ns, "-f", str(manifest),
            "--ignore-not-found=true", dry_run=dry_run,
        )
        if not dry_run and rc == 0:
            manifest.unlink()
            _io.success(f"removed manifest: {manifest}")
        return rc
    if cfg.service.backend == "proxmox":
        # `pct destroy` is destructive (drops the rootfs); we always
        # stop first, then destroy. Skip when --yes is absent so dry-run
        # cannot accidentally wipe a live container.
        ctid = _proxmox_ctid(cfg)
        if dry_run:
            _io.info(f"[dry-run] would: pct stop {ctid} && pct destroy {ctid}")
            return 0
        # Best-effort stop (ignore failure when already stopped).
        _pct("stop", ctid, dry_run=False)
        rc = _pct("destroy", ctid, "--purge", dry_run=False)
        # Drop the rendered launch script too so re-install starts clean.
        script = Path.home() / ".sndr" / "proxmox" / f"{cfg.key}.sh"
        if script.exists():
            script.unlink()
            _io.success(f"removed launch script: {script}")
        return rc
    _io.info(f"backend={cfg.service.backend} — uninstall noop")
    return 0
