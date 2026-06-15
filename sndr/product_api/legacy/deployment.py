# SPDX-License-Identifier: Apache-2.0
"""Enterprise deployment planner for the admin dashboard.

Given a preset and a deployment *target*, this module produces — read-only,
without any subprocess or host mutation — everything an operator needs to roll
the preset onto a host:

* the rendered deployment artifact (Docker Compose, Podman Quadlet, Kubernetes
  manifests, a systemd unit, a bare-metal launch script, or a Proxmox LXC
  provisioning script);
* the resolved *fine launch parameters* (image, container, ports, tensor
  parallelism, KV-cache dtype, context window, the full ``vllm serve`` argv,
  the Genesis patch flag count and pin);
* the live host inventory (OS / Python / Docker / NVIDIA / vLLM); and
* the dependency install plan for that preset on this host (blockers,
  warnings and the suggested install commands).

All heavy imports (model_configs registry, runtime command builder, deps
inventory) are done lazily inside functions so the product API stays
import-cheap and torch-free.
"""
from __future__ import annotations

import re
from typing import Any, Optional

# ``${var}`` placeholders used in preset docker mounts.
_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Conservative, operator-overridable default host paths for the symbolic
# mounts a preset references. These are *suggestions*; the GUI exposes them as
# editable fields so the artifact reflects the real host layout.
_PATH_DEFAULTS: dict[str, str] = {
    "models_dir": "/mnt/models",
    "hf_cache": "/mnt/models/hf-cache",
    "cache_root": "/var/lib/sndr/cache",
    "sndr_src": "/opt/genesis/sndr",
    "plugin_src": "/opt/genesis/plugin",
}


def _default_path_for(var: str) -> str:
    if var in _PATH_DEFAULTS:
        return _PATH_DEFAULTS[var]
    if "model" in var:
        return "/mnt/models"
    if "cache" in var:
        return "/var/lib/sndr/cache"
    return f"/var/lib/sndr/{var}"


_TARGETS: tuple[dict[str, str], ...] = (
    {
        "id": "compose",
        "label": "Docker Compose",
        "filename": "docker-compose.yml",
        "kind": "yaml",
        "needs": "docker",
        "summary": "Single-host Docker stack. Best for one GPU box you control end to end.",
    },
    {
        "id": "quadlet",
        "label": "Podman Quadlet",
        "filename": "sndr.container",
        "kind": "ini",
        "needs": "podman",
        "summary": "Rootless Podman managed by systemd via a .container unit. Daemonless, auto-restart.",
    },
    {
        "id": "kubernetes",
        "label": "Kubernetes",
        "filename": "sndr-deployment.yaml",
        "kind": "yaml",
        "needs": "kubectl",
        "summary": "ConfigMap + Service + PVC + Deployment with GPU limits and health probes.",
    },
    {
        "id": "systemd",
        "label": "systemd + Docker",
        "filename": "sndr-vllm.service",
        "kind": "ini",
        "needs": "docker",
        "summary": "Bare-host systemd unit that runs the pinned container with restart-on-failure.",
    },
    {
        "id": "bare_metal",
        "label": "Bare metal (native)",
        "filename": "launch.sh",
        "kind": "bash",
        "needs": "",
        "summary": "Native vllm serve launcher with all Genesis flags exported. No container runtime.",
    },
    {
        "id": "proxmox",
        "label": "Proxmox LXC",
        "filename": "provision-lxc.sh",
        "kind": "bash",
        "needs": "",
        "summary": "Provision a GPU-passthrough LXC container, install Docker, then run the stack.",
    },
    {
        "id": "proxmox_vm",
        "label": "Proxmox VM (KVM)",
        "filename": "provision-vm.sh",
        "kind": "bash",
        "needs": "",
        "summary": "Create a KVM VM with NVIDIA PCIe passthrough + cloud-init that installs Docker and runs the pinned stack.",
    },
    {
        "id": "sndr_daemon",
        "label": "SNDR daemon (gui-api)",
        "filename": "run-sndr-daemon.sh",
        "kind": "bash",
        "needs": "",
        "summary": "Run the SNDR management daemon (Product API + GUI) on this server, bound to localhost. Reach it via SSH tunnel and switch the GUI's top connection to it for a native per-server view (its own patches / configs / patcher version).",
    },
)

_TARGET_BY_ID = {t["id"]: t for t in _TARGETS}


def list_targets() -> list[dict[str, str]]:
    """Static catalogue of supported deployment targets (display metadata)."""
    return [dict(t) for t in _TARGETS]


def list_preset_keys() -> list[str]:
    """Return every resolvable preset key (V1 monolithic + V2 aliases).

    Phase 10.5 (2026-06-01): V1 monolithic preset tier fully retired
    so the V1 registry contributes nothing here. V2 alias files under
    `builtin/presets/<alias>.yaml` are the operator-facing canonical
    keys post-sunset — `_resolve_cfg` already accepts both namespaces,
    so this function now mirrors that union.
    """
    from sndr.model_configs import registry as reg

    keys: set[str] = set(reg.list_keys())
    try:
        from sndr.model_configs.registry_v2 import list_presets
        keys.update(list_presets())
    except Exception:
        pass
    return sorted(keys)


def _resolve_cfg(preset_id: str):
    """Resolve a preset id to a V1 ``ModelConfig``.

    Accepts both the V2 alias ids the GUI presents (``prod-qwen3.6-35b-multiconc``,
    composed via ``registry_v2.load_alias``) and the legacy V1 monolithic keys
    (``a5000-2x-35b-prod``). Raises ``KeyError`` if neither resolves.
    """
    from . import presets

    # V2 alias first — this is the namespace the rest of the GUI uses.
    try:
        cfg = presets.compose_for(preset_id)
        if cfg is not None:
            return cfg
    except Exception:
        pass

    from sndr.model_configs import registry as reg

    cfg = reg.get(preset_id)
    if cfg is not None:
        return cfg

    # Final fallback: a raw V2 profile id — what a rendered launcher's
    # `sndr.preset` label carries (e.g. `qwen3.6-35b-balanced`). Compose it
    # exactly as the launcher did (same parent model + the same auto-picked
    # hardware) so a profile-launched engine links AND its live runtime can be
    # diffed against the YAML it came from.
    try:
        from sndr.model_configs.registry_v2 import (
            compose_by_ids, load_model, load_profile,
        )
        from sndr.cli.legacy.profile import _pick_default_hardware
        profile = load_profile(preset_id)
        hw_id = profile.target_hardware or _pick_default_hardware(load_model(profile.parent_model)).id
        return compose_by_ids(profile.parent_model, hw_id, preset_id)
    except Exception:
        pass
    raise KeyError(preset_id)


def preset_has_docker(preset_id: str) -> bool:
    try:
        return getattr(_resolve_cfg(preset_id), "docker", None) is not None
    except Exception:
        return False


import threading as _threading
import time as _time

_INV_CACHE: dict[str, Any] = {"data": None, "ts": 0.0}
_INV_LOCK = _threading.Lock()
_INV_TTL = 300.0


def host_inventory(*, max_age: float = _INV_TTL) -> dict[str, Any]:
    """Live host snapshot — OS / Python / Docker / NVIDIA / vLLM — cached.

    ``inspect_host`` shells out to ``nvidia-smi`` / ``docker``, which can take
    several seconds (e.g. inside a GPU-less management-daemon sidecar where
    nvidia-smi has no device to talk to). The GUI hits this on the Hosts page, so
    we collect at most once per TTL and serve the cached copy — and the daemon
    warms it in the background at startup (see ``create_app``)."""
    with _INV_LOCK:
        cached = _INV_CACHE["data"]
        if cached is not None and (_time.time() - _INV_CACHE["ts"]) < max_age:
            return cached
    from sndr.deps import inspect_host

    data = inspect_host().to_dict()
    with _INV_LOCK:
        _INV_CACHE["data"] = data
        _INV_CACHE["ts"] = _time.time()
    return data


def warm_host_inventory() -> None:
    """Populate the inventory cache off the request path (daemon startup)."""
    try:
        host_inventory()
    except Exception:  # noqa: BLE001 - best-effort warm-up, never fatal
        pass


def _mount_specs(cfg) -> list[str]:
    docker = getattr(cfg, "docker", None)
    return list(getattr(docker, "mounts", None) or []) if docker else []


def _placeholders(cfg) -> list[str]:
    found: list[str] = []
    for spec in _mount_specs(cfg):
        for var in _PLACEHOLDER_RE.findall(str(spec)):
            if var not in found:
                found.append(var)
    return found


def _host_paths(cfg, overrides: Optional[dict[str, str]]) -> dict[str, str]:
    table = {var: _default_path_for(var) for var in _placeholders(cfg)}
    for key, value in (overrides or {}).items():
        if key in table and value:
            table[key] = str(value)
    return table


def _mount_vars(cfg, host_paths: dict[str, str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for spec in _mount_specs(cfg):
        text = str(spec)
        names = _PLACEHOLDER_RE.findall(text)
        if not names:
            continue
        var = names[0]
        if var in seen:
            continue
        seen.add(var)
        # spec looks like "${var}/sub:/container/path:mode" — pull container path
        container = ""
        parts = text.split(":")
        if len(parts) >= 2:
            container = parts[1]
        out.append({"name": var, "value": host_paths.get(var, _default_path_for(var)), "container": container})
    return out


def _runtime_argv(cfg) -> list[str]:
    from sndr.model_configs.runtime_command import build_runtime_command

    return list(build_runtime_command(cfg).argv)


def _genesis_env(cfg) -> dict[str, str]:
    raw = getattr(cfg, "genesis_env", None) or {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(value, bool):
            out[str(key)] = "1" if value else "0"
        else:
            out[str(key)] = str(value)
    return out


def _parameters(cfg) -> dict[str, Any]:
    docker = getattr(cfg, "docker", None)
    hw = getattr(cfg, "hardware", None)
    genv = _genesis_env(cfg)
    return {
        "image": docker.effective_image_ref() if docker else "",
        "container_name": (docker.container_name if docker else "") or f"sndr-{cfg.key}",
        "host_port": docker.effective_host_port() if docker else None,
        "tensor_parallel": int(getattr(hw, "n_gpus", 1) or 1),
        "min_vram_per_gpu_mib": getattr(hw, "min_vram_per_gpu_mib", None) if hw else None,
        "model_path": getattr(cfg, "model_path", None),
        "max_model_len": getattr(cfg, "max_model_len", None),
        "max_num_seqs": getattr(cfg, "max_num_seqs", None),
        "gpu_memory_utilization": getattr(cfg, "gpu_memory_utilization", None),
        "kv_cache_dtype": getattr(cfg, "kv_cache_dtype", None),
        "genesis_pin": getattr(cfg, "genesis_pin", None),
        "genesis_env_count": len(genv),
        "lifecycle": getattr(cfg, "lifecycle", None),
        "maintainer": getattr(cfg, "maintainer", None),
        "argv": _runtime_argv(cfg),
    }


def _deps_plan(cfg) -> dict[str, Any]:
    from sndr.deps import inspect_host, plan_changes

    return plan_changes(cfg, inspect_host()).to_dict()


# --------------------------------------------------------------------------
# Per-target artifact + command renderers
# --------------------------------------------------------------------------

def _artifact_compose(cfg, host_paths):
    from sndr.cli.legacy.compose import render_compose_yaml

    return render_compose_yaml(cfg, host_paths=host_paths)


def _commands_compose(cfg):
    return [
        "docker compose -f docker-compose.yml up -d",
        "docker compose -f docker-compose.yml logs -f",
        "docker compose -f docker-compose.yml down",
    ]


def _artifact_quadlet(cfg, host_paths):
    from sndr.cli.legacy.quadlet import render_quadlet

    return render_quadlet(cfg, host_paths=host_paths)


def _commands_quadlet(cfg):
    docker = getattr(cfg, "docker", None)
    unit = (docker.container_name if docker else None) or f"sndr-{cfg.key}"
    return [
        "cp sndr.container ~/.config/containers/systemd/",
        "systemctl --user daemon-reload",
        f"systemctl --user start {unit}.service",
        f"systemctl --user status {unit}.service",
    ]


def _dns_1123(value: str, *, max_len: int = 40) -> str:
    """Coerce a string into a DNS-1123 label (lowercase alphanumerics +
    hyphens, must start/end alphanumeric). V2 composed keys are long and
    contain dots/double-hyphens that k8s rejects."""
    out = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:max_len].strip("-")
    return out or "sndr"


def _artifact_kubernetes(cfg, host_paths, name_hint=None):
    import dataclasses

    from sndr.cli.legacy import k8s

    # k8s resource names derive from cfg.key (`sndr-<key>`). V2 composed keys
    # blow past the DNS-1123 63-char limit, so render with a short safe key
    # taken from the requested preset id.
    safe_key = _dns_1123(name_hint or cfg.key)
    if safe_key != cfg.key:
        cfg = dataclasses.replace(cfg, key=safe_key)

    # Presets do not ship a kubernetes block by default. Synthesize a sane one
    # from the docker + hardware blocks so the manifest renders with the right
    # image and GPU count instead of failing on a missing config.
    if getattr(cfg, "kubernetes", None) is None:
        from sndr.model_configs.schema import KubernetesConfig

        docker = getattr(cfg, "docker", None)
        hw = getattr(cfg, "hardware", None)
        k8s_cfg = KubernetesConfig(
            namespace="genesis",
            image=docker.effective_image_ref() if docker else "vllm/vllm-openai:nightly",
            gpu_count=int(getattr(hw, "n_gpus", 1) or 1),
            pvc={"models": "/models"},
            pvc_size_gib={"models": 1000},
        )
        cfg = dataclasses.replace(cfg, kubernetes=k8s_cfg)
    return k8s._all_yaml(cfg)


def _commands_kubernetes(cfg):
    return [
        "kubectl apply -f sndr-deployment.yaml",
        f"kubectl rollout status deploy/sndr-{cfg.key}",
        f"kubectl logs -f deploy/sndr-{cfg.key}",
    ]


def _systemd_unit(cfg, params) -> str:
    image = params["image"]
    name = params["container_name"]
    port = params["host_port"] or 8000
    genv = _genesis_env(cfg)
    env_args = " ".join(f"-e {k}={v}" for k, v in genv.items())
    serve = " ".join(params["argv"])
    run = (
        f"/usr/bin/docker run --rm --name {name} --gpus all "
        f"-p {port}:8000 {env_args} {image} {serve}"
    ).strip()
    return "\n".join(
        [
            "[Unit]",
            f"Description=Genesis vLLM ({cfg.key}) — pin {params['genesis_pin']}",
            "After=network-online.target docker.service",
            "Requires=docker.service",
            "",
            "[Service]",
            "Type=simple",
            f"ExecStartPre=-/usr/bin/docker rm -f {name}",
            f"ExecStart={run}",
            f"ExecStop=/usr/bin/docker stop {name}",
            "Restart=on-failure",
            "RestartSec=10s",
            "TimeoutStartSec=900",
            "",
            "[Install]",
            "WantedBy=multi-user.target",
        ]
    )


def _commands_systemd(cfg):
    return [
        "sudo cp sndr-vllm.service /etc/systemd/system/",
        "sudo systemctl daemon-reload",
        "sudo systemctl enable --now sndr-vllm.service",
        "journalctl -u sndr-vllm.service -f",
    ]


def _bare_metal_script(cfg, params) -> str:
    genv = _genesis_env(cfg)
    lines = [
        "#!/usr/bin/env bash",
        f"# Genesis vLLM bare-metal launcher — preset {cfg.key}",
        f"# Pin: {params['genesis_pin']}  |  reference image: {params['image']}",
        "# Requires: NVIDIA driver + CUDA, Python venv with the pinned vLLM build.",
        "set -euo pipefail",
        "",
        "# --- Genesis patch flags ---",
    ]
    for key, value in genv.items():
        lines.append(f"export {key}={value}")
    if not genv:
        lines.append("# (this preset ships no Genesis patch flags)")
    lines += [
        "",
        "# --- Launch (native vLLM, no container) ---",
        "exec " + " ".join(params["argv"]),
    ]
    return "\n".join(lines)


def _commands_bare_metal(cfg):
    return [
        "python -m venv .venv && source .venv/bin/activate",
        "pip install 'vllm==<pin>'  # match the preset's genesis_pin build",
        "chmod +x launch.sh && ./launch.sh",
    ]


def _proxmox_script(cfg, params, *, with_daemon: bool = False) -> str:
    name = params["container_name"]
    port = params["host_port"] or 8000
    lines = [
            "#!/usr/bin/env bash",
            f"# Genesis vLLM Proxmox LXC provisioner — preset {cfg.key}",
            "# Run on the Proxmox host. Provisions a GPU-passthrough LXC, installs",
            "# Docker, and starts the pinned vLLM stack. Review CTID/STORAGE first.",
            "set -euo pipefail",
            "",
            "# Discover sane defaults from THIS node so the script is portable across",
            "# clusters (storage layout, template version and free CTID all vary).",
            "CTID=${CTID:-$(pvesh get /cluster/nextid 2>/dev/null || echo 9100)}",
            "# First active storage that can hold a container rootfs (rootdir).",
            "STORAGE=${STORAGE:-$(pvesm status -content rootdir 2>/dev/null | awk 'NR>1 && $3==\"active\"{print $1; exit}')}",
            "STORAGE=${STORAGE:-local-lvm}",
            "BRIDGE=${BRIDGE:-vmbr0}",
            "# Newest Ubuntu/Debian LXC template already on local storage; download if none.",
            "TEMPLATE=${TEMPLATE:-$(pveam list local 2>/dev/null | awk '/(ubuntu|debian).*tar\\.(zst|gz)/{print $1}' | sort -V | tail -1)}",
            "if [ -z \"$TEMPLATE\" ]; then",
            "  echo 'No LXC template on local storage — downloading ubuntu-24.04…'",
            "  pveam update >/dev/null 2>&1 || true",
            "  _T=$(pveam available --section system 2>/dev/null | awk '/ubuntu-24.*standard/{print $2}' | sort -V | tail -1)",
            "  pveam download local \"$_T\" && TEMPLATE=\"local:vztmpl/$_T\"",
            "fi",
            "echo \"Using CTID=$CTID STORAGE=$STORAGE BRIDGE=$BRIDGE TEMPLATE=$TEMPLATE\"",
            "",
            "# 1. Create a privileged LXC (privileged needed for GPU passthrough).",
            f'pct create "$CTID" "$TEMPLATE" \\',
            f'  --hostname {name} \\',
            "  --cores 8 --memory 32768 --swap 8192 \\",
            '  --rootfs "$STORAGE:64" \\',
            '  --net0 name=eth0,bridge="$BRIDGE",ip=dhcp \\',
            "  --features nesting=1 --unprivileged 0 --onboot 1",
            "",
            "# 2. GPU passthrough — expose the NVIDIA devices to the container.",
            "#    Requires the NVIDIA driver on the Proxmox HOST (so /dev/nvidia* exist).",
            "if [ ! -e /dev/nvidia0 ]; then",
            "  echo 'WARNING: /dev/nvidia0 not found on the host — install the NVIDIA driver' >&2",
            "  echo '         on the Proxmox node first, or the container will have no GPU.' >&2",
            "fi",
            'cat >> "/etc/pve/lxc/$CTID.conf" <<EOF',
            "lxc.cgroup2.devices.allow: c 195:* rwm",
            "lxc.cgroup2.devices.allow: c 243:* rwm",
            "lxc.mount.entry: /dev/nvidia0 dev/nvidia0 none bind,optional,create=file",
            "lxc.mount.entry: /dev/nvidiactl dev/nvidiactl none bind,optional,create=file",
            "lxc.mount.entry: /dev/nvidia-uvm dev/nvidia-uvm none bind,optional,create=file",
            "EOF",
            "",
            'pct start "$CTID"',
            "",
            "# 3. Inside the container: install Docker + NVIDIA container toolkit.",
            'pct exec "$CTID" -- bash -c "apt-get update && apt-get install -y docker.io"',
            "",
            "# 4. Copy docker-compose.yml into the container and bring the stack up.",
            f'pct push "$CTID" docker-compose.yml /root/docker-compose.yml',
            'pct exec "$CTID" -- docker compose -f /root/docker-compose.yml up -d',
    ]
    if with_daemon:
        lines += _proxmox_lxc_daemon_block()
    lines += [
            "",
            f"# vLLM OpenAI API will listen on the container IP, port {port}.",
    ]
    return "\n".join(lines)


def _commands_proxmox(cfg):
    return [
        "chmod +x provision-lxc.sh",
        # CTID/STORAGE/TEMPLATE auto-discover from the node; override only if needed.
        "./provision-lxc.sh",
        'pct exec "$CTID" -- docker compose logs -f',
    ]


def _proxmox_vm_script(cfg, params, *, with_daemon: bool = False) -> str:
    name = params["container_name"]
    port = params["host_port"] or 8000
    write_files = ["write_files:", *_cloudinit_daemon_writefile()] if with_daemon else []
    daemon_runcmd = ["  - [bash, /root/run-sndr-daemon.sh]"] if with_daemon else []
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            f"# Genesis vLLM Proxmox VM provisioner — preset {cfg.key}",
            "# Run on the Proxmox host. Creates a KVM VM with NVIDIA PCIe",
            "# passthrough and a cloud-init that installs Docker + the pinned",
            "# stack on first boot. Set PCI from: lspci -nn | grep -i nvidia.",
            "# VFIO passthrough must already be configured on the host.",
            "set -euo pipefail",
            "",
            "VMID=${VMID:-9200}",
            "STORAGE=${STORAGE:-local-lvm}",
            "BRIDGE=${BRIDGE:-vmbr0}",
            "PCI=${PCI:-01:00}    # NVIDIA GPU PCI address",
            "IMG=${IMG:-noble-server-cloudimg-amd64.img}  # Ubuntu 24.04 cloud image",
            "",
            "# 1. Create the VM shell (q35 + UEFI + host CPU for passthrough).",
            f'qm create "$VMID" --name {name} --memory 49152 --cores 12 \\',
            '  --cpu host --machine q35 --bios ovmf --ostype l26 \\',
            '  --net0 virtio,bridge="$BRIDGE" --scsihw virtio-scsi-single',
            "",
            "# 2. Import the cloud image as the boot disk + a cloud-init drive.",
            'qm importdisk "$VMID" "$IMG" "$STORAGE"',
            'qm set "$VMID" --scsi0 "$STORAGE:vm-$VMID-disk-0" --boot order=scsi0',
            'qm set "$VMID" --ide2 "$STORAGE:cloudinit" --serial0 socket --vga serial0',
            'qm disk resize "$VMID" scsi0 200G',
            "",
            "# 3. NVIDIA PCIe passthrough.",
            'qm set "$VMID" --hostpci0 "$PCI,pcie=1"',
            "",
            "# 4. cloud-init: user + Docker + the pinned vLLM stack on first boot.",
            "#    Drop the rendered docker-compose.yml at /root/docker-compose.yml",
            "#    (scp it in, or inline it into the snippet below).",
            'qm set "$VMID" --ciuser sndr --sshkeys ~/.ssh/authorized_keys --ipconfig0 ip=dhcp',
            "cat > /var/lib/vz/snippets/sndr-vendor.yaml <<'CLOUDINIT'",
            "#cloud-config",
            "package_update: true",
            "packages: [docker.io, docker-compose-plugin]",
            *write_files,
            "runcmd:",
            "  - [systemctl, enable, --now, docker]",
            "  - [docker, compose, -f, /root/docker-compose.yml, up, -d]",
            *daemon_runcmd,
            "CLOUDINIT",
            'qm set "$VMID" --cicustom "vendor=local:snippets/sndr-vendor.yaml"',
            "",
            'qm start "$VMID"',
            f"# After boot, the vLLM OpenAI API listens on the VM IP, port {port}.",
        ]
    )


def _commands_proxmox_vm(cfg):
    return [
        "chmod +x provision-vm.sh",
        "VMID=9200 STORAGE=local-lvm PCI=01:00 ./provision-vm.sh",
        'qm terminal "$VMID"   # watch cloud-init, or: ssh sndr@<vm-ip>',
    ]


def _sndr_daemon_script(cfg, params) -> str:
    """Launcher for the SNDR management daemon as a *sidecar container* built from
    the running vLLM engine's image — that image already ships sndr_core + uvicorn,
    whereas the bare host does not. A fresh container has no ``GENESIS_ENABLE_*``
    env, so no patches apply: it is a clean, GPU-less management daemon serving the
    server's own catalog/patches/configs (Path B). ``--network host`` + bind
    127.0.0.1 keeps it on the loopback — reach it over an SSH tunnel."""
    return (
        "#!/usr/bin/env bash\n"
        "# Run the SNDR management daemon as a sidecar container from the vLLM\n"
        "# engine's image, RE-MOUNTING the same host sndr directory the engine\n"
        "# uses (sndr is mounted in at runtime, not baked into the image). A\n"
        "# fresh container has no GENESIS_ENABLE_* env, so no patches apply — a\n"
        "# clean, GPU-less management daemon. --network host + bind LAN so the\n"
        "# central GUI can switch straight to http://<this-host>:8765 (no tunnel).\n"
        "set -euo pipefail\n"
        'PORT="${SNDR_GUI_PORT:-8765}"\n'
        'NAME="${SNDR_DAEMON_NAME:-sndr-daemon}"\n'
        'BIND="${SNDR_BIND:-0.0.0.0}"\n'
        "ENGINE=$(docker ps --filter name=vllm --format '{{.Names}}' | head -1)\n"
        'if [ -z "$ENGINE" ]; then echo "no running vLLM container found"; exit 1; fi\n'
        'IMAGE=$(docker inspect "$ENGINE" --format \'{{.Config.Image}}\')\n'
        "# v12: the canonical package is the top-level `sndr/`, mounted into the\n"
        "# engine's site-packages at runtime. Replicate that mount so the daemon\n"
        "# resolves `import sndr`, else it dies with `No module named 'sndr'`.\n"
        "MNT=$(docker inspect \"$ENGINE\" --format '{{range .Mounts}}{{.Source}}:{{.Destination}}{{println}}{{end}}' | grep '/dist-packages/sndr$' | head -1)\n"
        'if [ -z "$MNT" ]; then echo "could not find the sndr mount on $ENGINE"; exit 1; fi\n'
        'SNDR_SRC="${MNT%%:*}"; SNDR_DEST="${MNT##*:}"\n'
        'if [ ! -d "$SNDR_SRC" ]; then echo "canonical sndr/ package not found at $SNDR_SRC"; exit 1; fi\n'
        'echo "[sndr] image=$IMAGE  sndr=$SNDR_SRC:$SNDR_DEST  bind=$BIND:$PORT"\n'
        'docker rm -f "$NAME" >/dev/null 2>&1 || true\n'
        "# Launch the Product API directly (not via the full `sndr` CLI) so the daemon\n"
        "# needs only product_api/ + its deps, immune to cli/ divergence between nodes.\n"
        "# Import the canonical `sndr.product_api.legacy.http_app` (no vllm namespace).\n"
        'docker run -d --name "$NAME" --restart unless-stopped --network host \\\n'
        "  --entrypoint python3 \\\n"
        '  -e SNDR_BIND="$BIND" -e SNDR_GUI_PORT="$PORT" -e SNDR_ENABLE_APPLY="${SNDR_ENABLE_APPLY:-}" \\\n'
        '  -v "${SNDR_SRC}:${SNDR_DEST}:ro" \\\n'
        '  "$IMAGE" \\\n'
        "  -c \"import os; from sndr.product_api.legacy.http_app import run_server; "
        "run_server(host=os.environ.get('SNDR_BIND','0.0.0.0'), port=int(os.environ.get('SNDR_GUI_PORT') or 8765), "
        "enable_apply=bool(os.environ.get('SNDR_ENABLE_APPLY')))\"\n"
        'echo "[sndr] container \'$NAME\' started — connect the GUI to http://<this-host>:$PORT"\n'
    )


def _commands_sndr_daemon(cfg):
    # The script runs `docker run -d` (detached) so it returns immediately; then
    # a health probe, dumping container logs if it isn't up yet.
    return [
        "chmod +x run-sndr-daemon.sh",
        "./run-sndr-daemon.sh",
        "sleep 4; curl -sf http://127.0.0.1:8765/api/v1/health >/dev/null 2>&1 "
        "&& echo 'daemon healthy on 127.0.0.1:8765' "
        "|| (echo 'not healthy yet — recent logs:'; docker logs --tail 25 sndr-daemon 2>&1 | tail -25)",
    ]


def _daemon_native_unit() -> str:
    """A systemd unit that runs the management daemon natively (no Docker) —
    for the bare-metal engine path, where sndr_core is on the host via pip."""
    return (
        "[Unit]\n"
        "Description=SNDR management daemon (GUI Product API)\n"
        "After=network-online.target\nWants=network-online.target\n\n"
        "[Service]\nType=simple\n"
        "Environment=SNDR_BIND=0.0.0.0\nEnvironment=SNDR_GUI_PORT=8765\n"
        "ExecStart=/usr/bin/env python3 -m sndr.cli gui-api\n"
        "Restart=on-failure\nRestartSec=3\n\n"
        "[Install]\nWantedBy=multi-user.target\n"
    )


def _daemon_install_commands(target: str) -> list[str]:
    """Steps to ALSO install the SNDR management daemon after the engine, so a
    single apply lands GUI(daemon) + sndr_core + engine. Docker-based engine
    targets get a sidecar container built from the engine image; the bare-metal
    target gets a native systemd unit (no Docker on the host)."""
    if target == "bare_metal":
        unit = _daemon_native_unit()
        return [
            "# --- SNDR management daemon (native systemd, no Docker) ---",
            f"sudo tee /etc/systemd/system/sndr-daemon.service >/dev/null <<'SNDR_DAEMON_EOF'\n{unit}SNDR_DAEMON_EOF",
            "sudo systemctl daemon-reload && sudo systemctl enable --now sndr-daemon.service",
            "sleep 3; curl -sf http://127.0.0.1:8765/api/v1/health >/dev/null "
            "&& echo 'daemon healthy on 127.0.0.1:8765' "
            "|| sudo journalctl -u sndr-daemon --no-pager -n 25",
        ]
    # Docker-based engine (compose / quadlet / kubernetes / systemd): sidecar.
    script = _sndr_daemon_script(None, None)
    return [
        "# --- SNDR management daemon (sidecar container from the engine image) ---",
        f"cat > run-sndr-daemon.sh <<'SNDR_DAEMON_EOF'\n{script}SNDR_DAEMON_EOF",
        *_commands_sndr_daemon(None),
    ]


def _proxmox_lxc_daemon_block() -> list[str]:
    """Script lines that install the daemon sidecar INSIDE a provisioned LXC.

    The script is materialised on the Proxmox host with a *quoted* heredoc (no
    host-side expansion), pushed into the container, and run there — so the new
    container hosts its own management daemon, reachable by the central GUI
    without a manual post-provision SSH."""
    script = _sndr_daemon_script(None, None)
    return [
        "",
        "# 5. Install the SNDR management daemon (sidecar) INSIDE the container.",
        f"cat > /tmp/run-sndr-daemon.sh <<'SNDR_DAEMON_EOF'\n{script}SNDR_DAEMON_EOF",
        'pct push "$CTID" /tmp/run-sndr-daemon.sh /root/run-sndr-daemon.sh',
        'pct exec "$CTID" -- bash /root/run-sndr-daemon.sh',
    ]


def _cloudinit_daemon_writefile() -> list[str]:
    """cloud-init ``write_files`` lines for the daemon script (indented under
    ``content: |``), to run the daemon sidecar inside a provisioned VM."""
    script = _sndr_daemon_script(None, None)
    body = ["  - path: /root/run-sndr-daemon.sh", "    permissions: '0755'", "    content: |"]
    body += [f"      {line}" for line in script.splitlines()]
    return body


_RENDERERS = {
    "compose": (_artifact_compose, _commands_compose, True),
    "quadlet": (_artifact_quadlet, _commands_quadlet, True),
    "kubernetes": (_artifact_kubernetes, _commands_kubernetes, True),
    "systemd": (None, _commands_systemd, False),
    "bare_metal": (None, _commands_bare_metal, False),
    "proxmox": (None, _commands_proxmox, False),
    "proxmox_vm": (None, _commands_proxmox_vm, False),
    "sndr_daemon": (None, _commands_sndr_daemon, False),
}


def _apply_image_override(cfg, image_override: Optional[str]):
    """Return a copy of ``cfg`` whose docker image is pinned to ``image_override``.

    Lets the operator install a preset at an explicit vLLM pin without editing
    the preset. A ``@sha256:`` reference is treated as a digest (most
    reproducible, wins over the tag); anything else replaces the image tag and
    clears any preset digest so the chosen tag is what actually runs. The
    override flows to BOTH the parameters and every artifact renderer because
    they all read ``cfg.docker``.
    """
    import dataclasses

    docker = getattr(cfg, "docker", None)
    ov = (image_override or "").strip()
    if not ov or docker is None:
        return cfg
    if "@sha256:" in ov:
        new_docker = dataclasses.replace(docker, image_digest=ov)
    else:
        new_docker = dataclasses.replace(docker, image=ov, image_digest=None)
    return dataclasses.replace(cfg, docker=new_docker)


def build_deployment(
    preset_id: str,
    target: str,
    *,
    host_paths: Optional[dict[str, str]] = None,
    image_override: Optional[str] = None,
    with_daemon: bool = False,
) -> dict[str, Any]:
    """Build the full read-only deployment plan for ``preset_id`` on ``target``.

    ``image_override`` pins the engine image/tag at install time (e.g. a specific
    vLLM nightly), overriding the preset's image. ``with_daemon`` also installs
    the SNDR management daemon (GUI Product API) so one apply lands
    GUI(daemon) + sndr_core + engine — a sidecar container for docker-based
    targets, a native systemd unit for bare-metal, and inside the guest for
    Proxmox. Raises ``KeyError`` for an unknown preset and ``ValueError`` for an
    unknown target — the HTTP layer maps these to 404 / 400.
    """
    if target not in _TARGET_BY_ID:
        raise ValueError(f"unknown deployment target: {target!r}")
    cfg = _apply_image_override(_resolve_cfg(preset_id), image_override)

    meta = _TARGET_BY_ID[target]
    resolved_paths = _host_paths(cfg, host_paths)
    params = _parameters(cfg)

    art_fn, cmd_fn, needs_paths = _RENDERERS[target]
    if target == "systemd":
        content = _systemd_unit(cfg, params)
    elif target == "bare_metal":
        content = _bare_metal_script(cfg, params)
    elif target == "proxmox":
        content = _proxmox_script(cfg, params, with_daemon=with_daemon)
    elif target == "proxmox_vm":
        content = _proxmox_vm_script(cfg, params, with_daemon=with_daemon)
    elif target == "sndr_daemon":
        content = _sndr_daemon_script(cfg, params)
    elif target == "kubernetes":
        content = _artifact_kubernetes(cfg, resolved_paths, name_hint=preset_id)
    else:
        content = art_fn(cfg, resolved_paths if needs_paths else None)

    commands = cmd_fn(cfg)
    # Docker-based + bare-metal engine targets append the daemon-install steps
    # (Proxmox embeds the daemon inside its provision script above instead).
    if with_daemon and target in ("compose", "quadlet", "kubernetes", "systemd", "bare_metal"):
        commands = list(commands) + _daemon_install_commands(target)

    return {
        "preset_id": preset_id,
        "preset_label": getattr(cfg, "description", None) or cfg.key,
        "target": target,
        "target_label": meta["label"],
        "artifact": {
            "kind": meta["kind"],
            "filename": meta["filename"],
            "content": content,
        },
        "parameters": params,
        "image_override": (image_override or "").strip() or None,
        "with_daemon": bool(with_daemon),
        "mount_vars": _mount_vars(cfg, resolved_paths),
        "dependencies": _deps_plan(cfg),
        "commands": commands,
    }
