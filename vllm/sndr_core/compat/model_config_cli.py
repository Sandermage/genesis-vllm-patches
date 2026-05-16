# SPDX-License-Identifier: Apache-2.0
"""Genesis model-config CLI — comprehensive launch+verify orchestrator.

Subcommands:
    list                          enumerate all available configs
    show <key>                    print full YAML
    render <key>                  emit launch script to stdout
    save <key> <path>             write launch script to disk
    audit <key>                   run audit_rules (17 rules: P98/P67/PN59/cudagraph_mode/...)
    validate <key>                schema + audit + cross-ref PATCH_REGISTRY
    preflight <key>               pre-launch env checks (mounts/GPU/pin)
    diagnose <key>                runtime diagnose (running container)
    verify <key>                  bench + diff vs reference (CI gate)
    where <key>                   show source tier
    new <key> --template <other>  clone existing
    new <key> --from-running <c>  capture from running docker
    launch <key> [--dry-run]      execute the rendered script
    bench-and-update <key>        boot + bench + write metrics back
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vllm.sndr_core.model_configs import (
    load_all, get, list_keys, dump_yaml,
)
from vllm.sndr_core.model_configs.registry import source_of, path_for
from vllm.sndr_core.model_configs.audit_rules import audit
from vllm.sndr_core.model_configs.preflight import (
    preflight_all, has_blockers as preflight_blockers,
)
from vllm.sndr_core.model_configs.diagnose import (
    diagnose_all,
)
from vllm.sndr_core.model_configs.verify import (
    verify, bench_metrics,
)


def _cfg_or_die(key: str):
    """Return cfg or print error + raise SystemExit(1) for CLI use."""
    cfg = get(key)
    if cfg is None:
        print(f"ERROR: config '{key}' not found", file=sys.stderr)
        print(f"Available: {', '.join(list_keys())}", file=sys.stderr)
        raise SystemExit(1)
    return cfg


def cmd_list(args) -> int:
    configs = load_all()
    include_tested = getattr(args, "include_tested", False)

    # JSON output: every preset emitted as a single object. Tested/QA
    # entries are filtered out unless the operator explicitly opted in.
    if getattr(args, "json", False):
        import json
        rows: list[dict] = []
        for key, c in sorted(configs.items()):
            is_tested = c.lifecycle == "tested"
            if is_tested and not include_tested:
                continue
            rm = c.reference_metrics
            rows.append({
                "key": key,
                "source": source_of(key) or "?",
                "title": c.title,
                "lifecycle": c.lifecycle,
                "tested": is_tested,
                "reference_metrics": None if rm is None else {
                    "long_gen_sustained_tps": rm.long_gen_sustained_tps,
                    "tool_call_score": rm.tool_call_score,
                    "stability_cv_pct": rm.stability_cv_pct,
                },
            })
        print(json.dumps(
            {"configs": rows, "count": len(rows)},
            ensure_ascii=False, indent=2,
        ))
        return 0

    if not configs:
        print("(no configs found under vllm/sndr_core/model_configs/builtin/)")
        return 0

    # Split working configs from tested/QA configs. By default, tested
    # configs render in a separate section so they do NOT pollute the
    # "what should I actually launch" view. --include-tested merges
    # everything into one block (operator opt-in).
    working: dict = {}
    tested: dict = {}
    for k, c in configs.items():
        (tested if c.lifecycle == "tested" else working)[k] = c

    label_total = len(configs)
    label_working = len(working)
    label_tested = len(tested)
    print(
        f"Genesis model configs ({label_total} total · {label_working} working"
        + (f" · {label_tested} tested" if label_tested else "")
        + ")\n"
    )

    def _row(k: str, c) -> None:
        rm = c.reference_metrics
        tier = source_of(k) or "?"
        tps = f"{rm.long_gen_sustained_tps:.1f}" if rm else "—"
        tool = rm.tool_call_score if rm else "—"
        cv = f"{rm.stability_cv_pct:.2f}" if rm else "—"
        print(f"  {k:<40}  {tier:<10}  {tps:>7}  {tool:<7}  {cv:>6}  {c.title}")

    _hdr = (
        f"  {'KEY':<40}  {'TIER':<10}  {'TPS':>7}  {'TOOL':<7}  "
        f"{'CV%':>6}  TITLE"
    )
    _sep = (
        f"  {'-'*40}  {'-'*10}  {'-'*7}  {'-'*7}  {'-'*6}  -----"
    )

    print(_hdr)
    print(_sep)
    for k in sorted(working):
        _row(k, working[k])

    if tested and include_tested:
        print("\n  ── tested / QA-only (do NOT compare with working) ──")
        print(_hdr)
        print(_sep)
        for k in sorted(tested):
            _row(k, tested[k])
    elif tested:
        print(f"\n  ({label_tested} tested/QA configs hidden — pass --include-tested to show)")

    print(
        "\n  Use:  sndr model-config validate <key>     # schema + audit"
        "\n        sndr model-config preflight <key>    # env check"
        "\n        sndr model-config launch <key>       # boot"
        "\n        sndr model-config diagnose <key>     # runtime check"
        "\n        sndr model-config verify <key>       # bench vs reference"
    )
    return 0


def cmd_show(args) -> int:
    cfg = _cfg_or_die(args.key)
    print(f"# Source tier: {source_of(args.key)}")
    print(dump_yaml(cfg))
    return 0


def cmd_render(args) -> int:
    cfg = _cfg_or_die(args.key)
    runtime = getattr(args, "runtime", None)
    # W-runtime 2026-05-06: --runtime override + symbolic mount resolution
    if runtime is not None:
        if runtime not in cfg.deploy.KNOWN_RUNTIMES:
            print(f"ERROR: --runtime '{runtime}' not in known runtimes "
                  f"{cfg.deploy.KNOWN_RUNTIMES}", file=sys.stderr)
            return 1
        if not getattr(cfg.deploy, runtime):
            supported = cfg.deploy.supported_runtimes()
            if not getattr(args, "force", False):
                print(f"ERROR: config '{args.key}' does not support runtime "
                      f"'{runtime}' (deploy.{runtime}=False). "
                      f"Supported: {supported}. "
                      f"Use --force to render anyway (untested artifact, "
                      f"do not deploy without validation).", file=sys.stderr)
                return 1
            else:
                print(f"# WARN: rendering '{runtime}' artifact for config "
                      f"that has not declared support (deploy.{runtime}=False). "
                      f"Output is UNTESTED. Validate before deploying.",
                      file=sys.stderr)
    else:
        runtime = cfg.deploy.default
    # Today only `docker` has a render template (cfg.to_launch_script());
    # other runtimes will land in follow-up sessions. Print informative
    # placeholder so operator knows what's missing.
    if runtime == "docker":
        # Resolve symbolic mounts in cfg.docker.mounts via host.yaml
        if cfg.docker is not None and cfg.docker.mounts:
            try:
                from vllm.sndr_core.model_configs.host import load_host_config
                from vllm.sndr_core.model_configs.schema import (
                    resolve_symbolic_mounts,
                )
                hc = load_host_config()
                if hc.paths:
                    cfg.docker.mounts = resolve_symbolic_mounts(
                        cfg.docker.mounts, hc.paths,
                    )
            except Exception as e:
                # If host.yaml absent / paths missing, fall back gracefully —
                # mounts may already be absolute (legacy configs)
                print(f"# WARN: symbolic-mount resolution skipped ({e})",
                      file=sys.stderr)
        print(cfg.to_launch_script())
        return 0
    elif runtime == "bare_metal":
        # Render bare-metal venv launch script (Proxmox-friendly per noonghunna)
        # Audit closure 2026-05-08 (S0.3): mode selects wheel/dev/dev_legacy
        # so production paths don't silently `pip install -e plugin || true`.
        mode = getattr(args, "mode", None) or "wheel"
        print(_render_bare_metal(cfg, mode=mode))
        return 0
    elif runtime == "lxc_proxmox":
        # Audit C4 closure (2026-05-16): emit runnable artifact, not the
        # old skeleton. The script handles the full lifecycle — create
        # CT, wire GPU passthrough, bootstrap venv, install Genesis,
        # write launch.sh inside the container — while still surfacing
        # the Proxmox-specific values (CTID, storage pool, bridge) as
        # operator-overridable env vars at the top of the file so the
        # operator does not have to edit the body to deploy.
        print(_render_lxc_proxmox(cfg))
        return 0
    elif runtime == "kubernetes":
        # Render k8s manifest (Deployment + Service + ConfigMap).
        # Currently NO symbolic-mount resolution: in k8s the operator
        # provides a PersistentVolumeClaim or HostPath, configured at
        # cluster level — we don't resolve `${var}` to absolute paths.
        # Instead, hostPath references are rendered as literal mount
        # specs that operator translates to PV/PVC manually.
        print(_render_kubernetes(cfg))
        return 0
    elif runtime == "podman":
        # Podman = docker render with substitutions per noonghunna
        # CONTAINER_RUNTIMES.md: --gpus → --device nvidia.com/gpu=all,
        # docker → podman binary swap. Most compose semantics identical.
        # Same symbolic-mount resolution as docker path.
        if cfg.docker is not None and cfg.docker.mounts:
            try:
                from vllm.sndr_core.model_configs.host import load_host_config
                from vllm.sndr_core.model_configs.schema import (
                    resolve_symbolic_mounts,
                )
                hc = load_host_config()
                if hc.paths:
                    cfg.docker.mounts = resolve_symbolic_mounts(
                        cfg.docker.mounts, hc.paths,
                    )
            except Exception as e:
                print(f"# WARN: symbolic-mount resolution skipped ({e})",
                      file=sys.stderr)
        print(_render_podman(cfg))
        return 0
    else:
        print(f"ERROR: render template for runtime '{runtime}' not yet "
              f"implemented. Currently supported: docker, bare_metal. "
              f"k8s/podman/lxc_proxmox templates land in follow-up sessions. "
              f"For now, render docker variant + manually translate.",
              file=sys.stderr)
        return 2


def _render_lxc_proxmox(cfg) -> str:
    """Render a runnable Proxmox LXC deployment script (audit C4 closure).

    Emits a single bash script that an operator runs on a Proxmox VE host
    to:
      1. Create an unprivileged LXC container from an Ubuntu template
         (CTID, storage pool, bridge, CPU/RAM all overridable via env vars
         at the top of the script).
      2. Inject the NVIDIA cgroup + bind-mount entries that GPU
         passthrough requires (lxc.cgroup2.devices.allow + lxc.mount.entry).
      3. Start the container and bootstrap a Python 3.12 venv inside.
      4. Install the captured ``vllm`` pin + the Genesis plugin into the
         venv (matches the bare_metal renderer's behaviour).
      5. Write the per-container launch.sh that runs ``vllm serve …`` with
         the exact CLI flags + Genesis env vars from this YAML.

    The artifact is intentionally idempotent: a re-run reuses the
    existing CTID, re-creates the launch script, and skips already-
    installed apt/pip packages. The operator only edits the env-var
    block at the top if their cluster topology differs from the
    defaults (CTID=200, local-lvm:64, vmbr0, nesting=1).

    Per noonghunna/club-3090 docs/CONTAINER_RUNTIMES.md the recommended
    path on Proxmox kernel 6.17.x is bare_metal venv — but for operators
    who must use LXC for tenant isolation, this renderer produces a
    runnable artifact instead of the old guide-only skeleton.
    """
    if cfg.docker is None:
        return (
            "# ERROR: lxc_proxmox render needs a `docker:` block in the\n"
            "# ModelConfig (used for image / port / mounts). Add a docker\n"
            "# section to the YAML OR run --runtime bare_metal instead.\n"
        )

    # Reference metrics header (matches kubernetes renderer style).
    ref_str = ""
    if cfg.reference_metrics:
        rm = cfg.reference_metrics
        ref_str = (f"# Reference: {rm.long_gen_sustained_tps:.1f} TPS sustained / "
                   f"{rm.tool_call_score} tool / CV {rm.stability_cv_pct:.2f}% / "
                   f"VRAM {rm.vram_total_mib} MiB\n")

    # Build the inner vllm serve command. The bare_metal renderer in
    # schema.py already knows how to assemble these flags; we delegate
    # to it so the LXC path stays consistent with the venv path.
    vllm_parts = cfg._build_vllm_cmd() if hasattr(cfg, "_build_vllm_cmd") else []
    # Drop the leading `vllm serve` token — we re-emit it as the first
    # line of the inner script so the operator can read the flags
    # without scanning past a shebang.
    if vllm_parts and vllm_parts[0] == "vllm serve":
        vllm_parts = vllm_parts[1:]
    inner_cmd_block = " \\\n  ".join(["vllm serve", *vllm_parts])

    # Render env-var exports for genesis_env + system_env. Keys are
    # stored verbatim with their full canonical prefix (matches the
    # bare_metal renderer in schema.py:2349-2358), so we emit them
    # without further prefix injection.
    env_lines: list[str] = []
    for key, val in sorted(cfg.system_env.items()):
        env_lines.append(f'export {key}={shlex_quote(str(val))}')
    for key, val in sorted(cfg.genesis_env.items()):
        env_lines.append(f'export {key}={shlex_quote(str(val))}')
    env_block = "\n".join(env_lines) if env_lines else "# (no env overrides)"

    # Mounts: translate "host:guest[:ro]" specs into pct mount-point flags
    # (mp0/mp1/...). LXC mount points are different from docker bind
    # mounts — pct accepts `--mpN <host>,mp=<guest>[,ro=1]` syntax.
    mp_lines: list[str] = []
    for i, m in enumerate(cfg.docker.mounts or []):
        parts = m.split(":")
        if len(parts) < 2:
            continue
        host_path = parts[0]
        guest_path = parts[1]
        ro = len(parts) > 2 and parts[2] == "ro"
        spec = f"{host_path},mp={guest_path}"
        if ro:
            spec += ",ro=1"
        mp_lines.append(f"  --mp{i} {spec} \\")
    mp_block = "\n".join(mp_lines) if mp_lines else ""

    vllm_pin = cfg.vllm_pin_required or "<set vllm_pin_required in YAML>"
    n_gpus = cfg.hardware.n_gpus or 1

    return f"""#!/usr/bin/env bash
# Generated by Genesis model_config render --runtime lxc_proxmox
#   key:           {cfg.key}
#   title:         {cfg.title}
#   maintainer:    {cfg.maintainer}
#   schema_v:      {cfg.schema_version}
#   genesis_pin:   {cfg.genesis_pin or '<unspecified>'}
#   vllm_pin:      {cfg.vllm_pin_required or '<unspecified>'}
{ref_str}#
# Runnable Proxmox VE LXC deployment script. Execute on the PVE host
# (as root or via sudo). Idempotent — safe to re-run.
#
# Reviewer checklist:
#   [ ] Adjust the env-var block below to your cluster topology
#       (CTID, storage pool, bridge, CPU/RAM).
#   [ ] Confirm /var/lib/vz/template/cache/ubuntu-24.04-standard_*.tar.zst exists.
#       If not: pveam update && pveam download local ubuntu-24.04-standard_24.04-2_amd64.tar.zst
#   [ ] Confirm NVIDIA driver is installed on the PVE host
#       (`nvidia-smi` returns successfully).
#   [ ] {n_gpus} GPU(s) will be passed through (host devices /dev/nvidia0..{n_gpus - 1}).

set -euo pipefail

# ─── Operator-overridable parameters ────────────────────────────────────
SNDR_CTID="${{SNDR_CTID:-200}}"
SNDR_HOSTNAME="${{SNDR_HOSTNAME:-genesis-{cfg.key}}}"
SNDR_TEMPLATE="${{SNDR_TEMPLATE:-local:vztmpl/ubuntu-24.04-standard_24.04-2_amd64.tar.zst}}"
SNDR_STORAGE="${{SNDR_STORAGE:-local-lvm:64}}"
SNDR_BRIDGE="${{SNDR_BRIDGE:-vmbr0}}"
SNDR_CORES="${{SNDR_CORES:-8}}"
SNDR_MEMORY_MIB="${{SNDR_MEMORY_MIB:-65536}}"
SNDR_VLLM_PIN="${{SNDR_VLLM_PIN:-{vllm_pin}}}"
SNDR_N_GPUS="${{SNDR_N_GPUS:-{n_gpus}}}"

step() {{ echo; echo "==> $*"; }}
die()  {{ echo "ERROR: $*" >&2; exit 1; }}

command -v pct >/dev/null || die "pct not found — run this on a Proxmox VE host"

# ─── Step 1: create LXC container if it does not exist ──────────────────
step "Step 1/5 — provision LXC container CTID=${{SNDR_CTID}}"
if pct status "${{SNDR_CTID}}" >/dev/null 2>&1; then
  echo "  container ${{SNDR_CTID}} already exists — reusing"
else
  pct create "${{SNDR_CTID}}" "${{SNDR_TEMPLATE}}" \\
    --hostname "${{SNDR_HOSTNAME}}" \\
    --cores "${{SNDR_CORES}}" \\
    --memory "${{SNDR_MEMORY_MIB}}" \\
    --rootfs "${{SNDR_STORAGE}}" \\
    --net0 "name=eth0,bridge=${{SNDR_BRIDGE}},ip=dhcp" \\
    --features "nesting=1" \\
    --unprivileged 0 \\
{mp_block}
    --onboot 1
fi

# ─── Step 2: inject NVIDIA GPU passthrough into /etc/pve/lxc/<CTID>.conf ──
step "Step 2/5 — wire NVIDIA GPU passthrough"
CFG_FILE="/etc/pve/lxc/${{SNDR_CTID}}.conf"
GENESIS_MARKER="# >>> genesis-{cfg.key} nvidia passthrough <<<"
if grep -q "${{GENESIS_MARKER}}" "${{CFG_FILE}}" 2>/dev/null; then
  echo "  GPU passthrough already wired — skipping"
else
  {{
    echo "${{GENESIS_MARKER}}"
    echo "lxc.cgroup2.devices.allow: c 195:* rwm   # nvidia"
    echo "lxc.cgroup2.devices.allow: c 234:* rwm   # nvidia-uvm"
    echo "lxc.cgroup2.devices.allow: c 235:* rwm   # nvidia-uvm-tools"
    echo "lxc.cgroup2.devices.allow: c 509:* rwm   # nvidia-caps"
    echo "lxc.mount.entry: /dev/nvidiactl       dev/nvidiactl       none bind,optional,create=file"
    echo "lxc.mount.entry: /dev/nvidia-uvm      dev/nvidia-uvm      none bind,optional,create=file"
    echo "lxc.mount.entry: /dev/nvidia-uvm-tools dev/nvidia-uvm-tools none bind,optional,create=file"
    for i in $(seq 0 $((SNDR_N_GPUS - 1))); do
      echo "lxc.mount.entry: /dev/nvidia${{i}} dev/nvidia${{i}} none bind,optional,create=file"
    done
    echo "# <<< genesis-{cfg.key} nvidia passthrough >>>"
  }} >> "${{CFG_FILE}}"
fi

# ─── Step 3: start the container ────────────────────────────────────────
step "Step 3/5 — start container"
if [ "$(pct status "${{SNDR_CTID}}" | awk '{{print $2}}')" != "running" ]; then
  pct start "${{SNDR_CTID}}"
  # Give the container a moment to come up before we exec into it.
  sleep 3
fi

# ─── Step 4: bootstrap venv + install vllm + Genesis plugin inside CT ───
step "Step 4/5 — bootstrap Python venv + vllm pin + Genesis plugin"
pct exec "${{SNDR_CTID}}" -- bash -lc '
  set -euo pipefail
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq python3.12 python3.12-venv python3-pip git curl ca-certificates >/dev/null
  if [ ! -x /opt/sndr-venv/bin/python3 ]; then
    python3.12 -m venv /opt/sndr-venv
  fi
  /opt/sndr-venv/bin/pip install -q --upgrade pip
  /opt/sndr-venv/bin/pip install -q "vllm=='"${{SNDR_VLLM_PIN}}"'"
'
# Genesis plugin is mounted via the pct --mp* flags above; if the
# operator has not mounted the genesis repo, the launch script will
# fall back to PyPI on the next pip install line. Operators can also
# pre-bake the plugin into a template tarball.

# ─── Step 5: write the per-container launch.sh ──────────────────────────
step "Step 5/5 — write launch.sh inside the container"
LAUNCH_SCRIPT="/opt/sndr-venv/launch.sh"
pct exec "${{SNDR_CTID}}" -- bash -lc "mkdir -p /opt/sndr-venv && cat > ${{LAUNCH_SCRIPT}} <<'GENESIS_LAUNCH_EOF'
#!/usr/bin/env bash
# Generated launch script for Genesis preset {cfg.key!r}
set -euo pipefail
source /opt/sndr-venv/bin/activate

# ─── Genesis + system environment ────────────────────────────────────────
{env_block}

# ─── Run vLLM with the captured flags ────────────────────────────────────
exec {inner_cmd_block}
GENESIS_LAUNCH_EOF
chmod +x ${{LAUNCH_SCRIPT}}"

cat <<EOM

✓ Genesis preset {cfg.key!r} is provisioned in CT ${{SNDR_CTID}}.

  Start vLLM inside the container:
    pct exec ${{SNDR_CTID}} -- /opt/sndr-venv/launch.sh

  Access the OpenAI-compatible API from the PVE host:
    curl http://<container-ip>:{cfg.docker.port}/v1/models

  Inspect logs:
    pct exec ${{SNDR_CTID}} -- journalctl -u vllm.service -f   # if systemd-wrapped
    pct exec ${{SNDR_CTID}} -- /opt/sndr-venv/launch.sh        # foreground for debug

  Re-run this script after editing the YAML to refresh launch.sh in-place
  (the script is idempotent — CT creation and apt steps are skipped).
EOM
"""


def _render_lxc_proxmox_skeleton(cfg) -> str:
    """Deprecated alias — kept for one release while operators migrate.

    Old call sites (custom scripts, ad-hoc tooling) that imported this
    helper directly continue to work; the body now delegates to the
    runnable renderer above. Removed in a follow-up release after the
    docstring tombstone has aged out.
    """
    return _render_lxc_proxmox(cfg)


# Local alias so the renderer body stays readable — shlex.quote is the
# canonical way to escape a single shell argument, but importing it at
# module level (rather than inline) keeps the renderer free of repeated
# `import shlex` calls in hot paths. shlex is in stdlib, no extra cost.
def shlex_quote(s: str) -> str:
    import shlex
    return shlex.quote(s)


def _render_kubernetes(cfg) -> str:
    """Render Genesis config as Kubernetes manifest (Deployment + Service + ConfigMap).

    Per noonghunna/club-3090 docs/CONTAINER_RUNTIMES.md and disc#48 from
    @apnar (microk8s reporter): a working k8s flow translates the docker
    compose `command:` block into pod spec `args:`, model weights mount
    via HostPath/PVC, GPUs via NVIDIA k8s device plugin (`nvidia.com/gpu`
    resource), and Genesis env vars via ConfigMap.

    This template emits a minimal viable manifest as one YAML stream
    (Deployment + Service + ConfigMap, separated by `---`). Operator
    deploys via `kubectl apply -f -` from stdin OR pipes to a file.

    Caveats (vs full helm chart):
    - HostPath used for mounts (single-node clusters). For multi-node,
      operator MUST swap to PVC pointing at NFS / Ceph / etc.
    - No HorizontalPodAutoscaler (vLLM stateful, doesn't scale that way)
    - No resource limits on CPU/RAM (let scheduler choose); GPU req is hard
    - Service is ClusterIP (operator wraps in Ingress / LoadBalancer if
      external access needed)
    - genesis_src + plugin_src mounted via initContainer that copies into
      shared emptyDir (avoids RO bind-mount complications in some k8s setups)

    Returns YAML stream with embedded comments. NOT compose-equivalent —
    operator should review before `kubectl apply`.

    Audit Phase A W-runtime 2026-05-06: implemented as YAML manifest
    generator. Helm chart variant is a follow-up.
    """
    if cfg.docker is None:
        # k8s render needs container info from cfg.docker (image, port, mounts).
        # Bare-metal-only configs can't translate to k8s without manual work.
        return (
            "# ERROR: cannot render k8s manifest — config has no `docker:` block.\n"
            "# k8s rendering reuses docker.image / port / mounts. Add a docker\n"
            "# block to the config OR translate manually.\n"
        )

    # Pod name derives from container_name (kebab-case it)
    name = cfg.docker.container_name.replace("_", "-").lower()

    # Build env list from genesis_env + system_env (k8s expects list of {name,value})
    env_entries = []
    for env_dict in (cfg.system_env, cfg.genesis_env):
        for k, v in sorted(env_dict.items()):
            env_entries.append(f"        - name: {k}\n          value: {str(v)!r}")
    env_block = "\n".join(env_entries) if env_entries else "        []"

    # Build vllm serve args (split by space — k8s uses YAML list).
    # Use yaml.safe_dump per-element to handle embedded JSON, single
    # quotes, special chars correctly. Strip trailing newline yaml adds.
    import yaml as _yaml_for_render
    args = cfg._build_vllm_cmd() if hasattr(cfg, "_build_vllm_cmd") else []

    def _yaml_arg(a):
        # safe_dump returns "<scalar>\n" — strip newline for inline use
        return _yaml_for_render.safe_dump(
            a, default_style='"', allow_unicode=True,
        ).rstrip("\n")
    args_yaml = "\n".join(f"        - {_yaml_arg(a)}" for a in args) if args else "        []"

    # Build hostPath mounts. Operator may need to swap to PVC for multi-node.
    # Symbolic ${var} mounts pass through literal — k8s admin resolves at
    # cluster level OR operator runs `genesis model-config render --runtime
    # docker` first to get resolved paths.
    volume_mounts = []
    volumes = []
    for i, mount in enumerate(cfg.docker.mounts or []):
        # mount format: "<host>:<container>[:ro|rw]"
        parts = mount.split(":")
        if len(parts) < 2:
            continue
        host_path, container_path = parts[0], parts[1]
        ro = "ro" in parts[2:] if len(parts) > 2 else False
        vol_name = f"vol-{i}"
        volume_mounts.append(
            f"        - name: {vol_name}\n          mountPath: {container_path}\n"
            + ("          readOnly: true" if ro else "          readOnly: false")
        )
        volumes.append(
            f"      - name: {vol_name}\n"
            f"        hostPath:\n"
            f"          path: {host_path}\n"
            f"          # NOTE: HostPath is single-node-only. For multi-node\n"
            f"          # k8s clusters, swap to PersistentVolumeClaim pointing\n"
            f"          # at NFS / Ceph / S3 / cloud volumes."
        )
    vm_block = "\n".join(volume_mounts) if volume_mounts else "        []"
    vol_block = "\n".join(volumes) if volumes else "      []"

    # Reference metrics for documentation header
    ref_str = ""
    if cfg.reference_metrics:
        rm = cfg.reference_metrics
        ref_str = (f"# Reference: {rm.long_gen_sustained_tps:.1f} TPS sustained / "
                   f"{rm.tool_call_score} tool / CV {rm.stability_cv_pct:.2f}% / "
                   f"VRAM {rm.vram_total_mib} MiB\n")

    return f"""# Generated by Genesis model_config render --runtime kubernetes
#   key:           {cfg.key}
#   title:         {cfg.title}
#   maintainer:    {cfg.maintainer}
#   schema_v:      {cfg.schema_version}
#   genesis_pin:   {cfg.genesis_pin or '<unspecified>'}
#   vllm_pin:      {cfg.vllm_pin_required or '<unspecified>'}
{ref_str}#
# 3 resources: ConfigMap (env), Deployment (pod), Service (port).
# Deploy via:  kubectl apply -f - <<< "$(genesis model-config render {cfg.key} --runtime kubernetes)"
#
# Reviewer checklist before `kubectl apply`:
#   [ ] HostPath mounts compatible with single-node OR replaced with PVC
#   [ ] NVIDIA device plugin installed (`kubectl get nodes -o yaml | grep nvidia.com/gpu`)
#   [ ] Image `{cfg.docker.image}` accessible from cluster (push to private registry if needed)
#   [ ] genesis_src + plugin_src paths exist on each node OR ConfigMap+initContainer used
#   [ ] Resource limits (CPU/RAM) match cluster capacity
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: {name}-env
  labels:
    app: {name}
    genesis-key: {cfg.key}
data:
  # ConfigMap is a placeholder for cluster-level overrides. Actual env vars
  # are inlined in the Deployment.spec.containers.env (below) since they
  # need to be Genesis-config-coupled. Operator can extend this map for
  # cluster-specific tweaks (e.g. proxy URLs, log shippers) and reference
  # via envFrom.
  GENESIS_KEY: {cfg.key!r}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
  labels:
    app: {name}
    genesis-key: {cfg.key}
spec:
  replicas: 1   # vLLM is stateful — keep at 1 unless using true distributed setup
  selector:
    matchLabels:
      app: {name}
  template:
    metadata:
      labels:
        app: {name}
    spec:
      restartPolicy: Always
      # NCCL P2P disabled for consumer Ampere (no NVLink) — REQUIRED on most rigs
      containers:
      - name: {name}
        image: {cfg.docker.image}
        # Args from cfg._build_vllm_cmd() — see ConfigMap.VLLM_SERVE_ARGS for raw form
        args:
{args_yaml}
        ports:
        - containerPort: {cfg.docker.port}
          name: http
        env:
{env_block}
        resources:
          limits:
            nvidia.com/gpu: {cfg.hardware.n_gpus}
          requests:
            nvidia.com/gpu: {cfg.hardware.n_gpus}
        volumeMounts:
{vm_block}
        # Health check via /v1/models endpoint
        readinessProbe:
          httpGet:
            path: /v1/models
            port: http
            httpHeaders:
            - name: Authorization
              value: "Bearer {cfg.api_key}"
          initialDelaySeconds: 120   # vLLM boot ~2-5min
          periodSeconds: 15
          timeoutSeconds: 5
          failureThreshold: 3
      volumes:
{vol_block}
---
apiVersion: v1
kind: Service
metadata:
  name: {name}
  labels:
    app: {name}
spec:
  type: ClusterIP    # operator may switch to LoadBalancer / NodePort for external access
  selector:
    app: {name}
  ports:
  - name: http
    port: {cfg.docker.port}
    targetPort: http
    protocol: TCP
"""


def _render_podman(cfg) -> str:
    """Render docker launch script with podman substitutions.

    Per noonghunna club-3090 CONTAINER_RUNTIMES.md the differences from
    docker are minimal:
      - swap `docker` binary → `podman`
      - swap `--gpus all` (Docker syntax) → `--device nvidia.com/gpu=all`
        (Podman + NVIDIA Container Device Interface syntax)

    Everything else (mounts, env, network, labels) is compatible. We
    delegate to cfg.to_launch_script() (the existing docker render) and
    post-process the output. Caveats per noonghunna:
      - `docker compose project labels` may differ — not affected here
        (we render `docker run`, not compose)
      - if operator uses `COMPOSE_BIN=podman compose`, that's a separate
        path and works without this render variant

    Audit P2 W-runtime 2026-05-06: implemented as text post-process to
    avoid duplicating ~150 LOC of docker render. If podman semantics
    diverge further, refactor to dedicated render template.
    """
    docker_script = cfg.to_launch_script()
    # Header comment — annotate runtime in generated artefact
    header = (
        "#!/usr/bin/env bash\n"
        "# Generated by Genesis model_config render --runtime podman\n"
        "# (post-processed from docker render: docker→podman + --gpus→--device)\n"
        "#\n"
    )
    # Strip the original `#!/usr/bin/env bash` if present (we replace it)
    body = docker_script
    if body.startswith("#!/usr/bin/env bash\n"):
        body = body[len("#!/usr/bin/env bash\n"):]
    # Substitutions (in order — gpu flag substitution must precede docker→podman
    # so we don't double-substitute "podman --gpus")
    body = body.replace("--gpus all", "--device nvidia.com/gpu=all")
    body = body.replace("docker stop", "podman stop")
    body = body.replace("docker rm", "podman rm")
    body = body.replace("docker run", "podman run")
    body = body.replace("docker logs", "podman logs")
    body = body.replace("docker exec", "podman exec")
    return header + body


def _render_bare_metal(cfg, *, mode: str = "wheel") -> str:
    """Render a bash launch script that runs `vllm serve` in a native venv.

    No Docker, no container — just `python -m vllm serve` after sourcing
    a venv. Use case: Proxmox VE LXC where Docker image hits the kernel
    6.17.x asyncio footgun (per noonghunna club-3090 CONTAINER_RUNTIMES.md).

    Operator must have already created the venv + pip-installed vLLM at
    the version matching cfg.vllm_pin_required. Script asserts venv path
    exists and aborts cleanly if not.

    Audit closure 2026-05-08 (P2-2 / S0.3): three modes now selectable
    so production paths don't silently `pip install -e plugin || true`.

      • ``mode="wheel"`` (default, production): script verifies
        ``vllm.sndr_core`` imports cleanly. NO ``pip install``. NO
        ``|| true`` masking. Hard-fails if the wheel isn't installed.
      • ``mode="dev"``: editable install of plugin source path, but
        WITHOUT ``|| true`` — error visible. PYTHONPATH set so live
        edits to ``vllm/sndr_core/`` are picked up.
      • ``mode="dev_legacy"``: legacy behavior with ``|| true`` for
        backward compatibility with operators relying on silent retry.
        Marked deprecated; emits a runtime WARN line.
    """
    from vllm.sndr_core.model_configs.host import load_host_config

    if mode not in ("wheel", "dev", "dev_legacy"):
        raise ValueError(
            f"unknown bare-metal render mode {mode!r}; "
            "expected one of: wheel, dev, dev_legacy"
        )

    hc = load_host_config()
    venv_path = hc.paths.get("vllm_venv", "/opt/vllm-env")
    genesis_src = hc.paths.get("genesis_src", "${genesis_src}")
    plugin_src = hc.paths.get("plugin_src", "${plugin_src}")

    lines = [
        "#!/usr/bin/env bash",
        f"# Generated by Genesis model_config render --runtime bare_metal --mode {mode}",
        f"#   key:           {cfg.key}",
        f"#   title:         {cfg.title}",
        f"#   maintainer:    {cfg.maintainer}",
        f"#   genesis_pin:   {cfg.genesis_pin or '<unspecified>'}",
        f"#   vllm_pin:      {cfg.vllm_pin_required or '<unspecified>'}",
        "#",
        "# Bare-metal venv launch — for Proxmox LXC kernel 6.17.x footgun",
        "# workaround OR minimal-deps environments. Operator must have a",
        "# working venv with vllm pip-installed at the matching pin.",
        f"#   mode={mode}: " + {
            "wheel": "production — assumes vllm-sndr-core wheel installed; no editable install",
            "dev": "dev — editable install of plugin source, errors visible",
            "dev_legacy": "DEPRECATED legacy — silent `|| true` on plugin install",
        }[mode],
        "",
        "set -euo pipefail",
        "",
        f"VENV={venv_path}",
        "if [ ! -d \"$VENV\" ]; then",
        "  echo \"ERROR: venv not found at $VENV\" >&2",
        "  echo \"Create with: python3 -m venv $VENV && source $VENV/bin/activate "
        f"&& pip install vllm=={cfg.vllm_pin_required or '<pin>'}\" >&2",
        "  exit 1",
        "fi",
        "",
        "source \"$VENV/bin/activate\"",
        "",
    ]
    if mode == "wheel":
        lines.extend([
            "# Wheel mode (production): verify vllm-sndr-core importable WITHOUT",
            "# attempting any editable install. Fail-fast posture.",
            "python3 -c 'import vllm.sndr_core' || {",
            "  echo \"ERROR: vllm.sndr_core not importable in this venv.\" >&2",
            "  echo \"Install the wheel:  pip install vllm-sndr-core\" >&2",
            "  exit 1",
            "}",
            "",
        ])
    elif mode == "dev":
        lines.extend([
            "# Dev mode: editable install of plugin source — errors visible (no `|| true`)",
            f"export PYTHONPATH=\"{genesis_src}/..:${{PYTHONPATH:-}}\"",
            f"pip install --quiet -e {plugin_src}",
            "",
        ])
    else:  # dev_legacy
        lines.extend([
            "# DEPRECATED legacy mode — silent `|| true` masks install failures.",
            "echo \"[Genesis WARN] bare-metal mode=dev_legacy is deprecated; "
            "use --mode wheel for production or --mode dev for dev.\" >&2",
            f"export PYTHONPATH=\"{genesis_src}/..:${{PYTHONPATH:-}}\"",
            f"pip install --quiet -e {plugin_src} 2>/dev/null || true",
            "",
        ])
    lines.append("# ── System env ──")
    for k, v in sorted(cfg.system_env.items()):
        lines.append(f"export {k}={v!r}" if any(c.isspace() for c in str(v))
                     else f"export {k}={v}")
    lines.append("")
    lines.append("# ── Genesis env (P*/PN*/GENESIS_*) ──")
    for k, v in sorted(cfg.genesis_env.items()):
        lines.append(f"export {k}={v}")
    lines.append("")
    lines.append("# ── vllm serve ──")
    extra_args = " ".join(cfg.vllm_extra_args) if cfg.vllm_extra_args else ""
    lines.append(
        f"exec vllm serve --model {cfg.model_path} "
        f"--tensor-parallel-size {cfg.hardware.n_gpus} "
        f"--gpu-memory-utilization {cfg.gpu_memory_utilization} "
        f"--max-model-len {cfg.max_model_len} "
        f"--dtype {cfg.dtype} "
        f"--api-key {cfg.api_key} --host {cfg.host} --port 8000 "
        f"{extra_args}"
    )
    return "\n".join(lines) + "\n"


def cmd_save(args) -> int:
    cfg = _cfg_or_die(args.key)
    out = Path(args.path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(cfg.to_launch_script())
    out.chmod(0o755)
    print(f"Wrote launch script to {out}")
    return 0


def _print_section(title: str, items: list, name_fn, msg_fn, sev_fn,
                   passed_fn) -> tuple[int, int, int]:
    """Helper — print a section, return (errors, warnings, infos)."""
    e = w = i = 0
    if not items:
        print(f"  ✓ no {title.lower()} issues")
        return 0, 0, 0
    for item in items:
        sev = sev_fn(item)
        passed = passed_fn(item)
        if sev == "error" and not passed:
            mark = "✗ ERROR  "
            e += 1
        elif sev == "warning" and not passed:
            mark = "⚠ WARN   "
            w += 1
        else:
            mark = "✓ ok     "
            i += 1
        print(f"  {mark}{name_fn(item):<35}  {msg_fn(item)}")
    return e, w, i


def cmd_audit(args) -> int:
    cfg = _cfg_or_die(args.key)
    print(f"=== audit {args.key} ===\n")
    issues = audit(cfg)
    e = w = 0
    if not issues:
        from vllm.sndr_core.model_configs.audit_rules import RULES
        print(f"  ✓ all {len(RULES)} rules pass")
        return 0
    for rid, sev, title, msg in issues:
        if sev == "error":
            mark = "✗ ERROR  "
            e += 1
        elif sev == "warning":
            mark = "⚠ WARN   "
            w += 1
        else:
            mark = "ℹ INFO   "
        print(f"  {mark}[{rid}] {title}")
        for line in msg.splitlines():
            print(f"             {line}")
    print()
    print(f"  Summary: {e} errors, {w} warnings, "
          f"{len([i for i in issues if i[1] == 'info'])} info")
    return 1 if e > 0 else 0


def cmd_validate(args) -> int:
    """Combined: schema check + audit_rules. Exit 1 on any error severity."""
    cfg = _cfg_or_die(args.key)
    print(f"=== validate {args.key} ===\n")

    # Schema validation already happened on load. Re-run for explicit confirm.
    print("[1/2] schema check")
    try:
        cfg.validate()
        print("  ✓ schema OK")
    except Exception as ex:
        print(f"  ✗ schema FAIL: {ex}")
        return 1

    print("\n[2/2] audit_rules (cross-patch consistency, env checks)")
    rc = cmd_audit(args)
    return rc


def cmd_preflight(args) -> int:
    cfg = _cfg_or_die(args.key)
    print(f"=== preflight {args.key} ===\n")
    checks = preflight_all(cfg)
    if not checks:
        print("  (no preflight checks applicable)")
        return 0
    e = w = 0
    for c in checks:
        if c.severity == "error" and not c.passed:
            print(f"  ✗ ERROR  {c.name:<35}  {c.message}")
            e += 1
        elif c.severity == "warning" and not c.passed:
            print(f"  ⚠ WARN   {c.name:<35}  {c.message}")
            w += 1
        else:
            print(f"  ✓ ok     {c.name:<35}  {c.message}")
    print(f"\n  Summary: {e} blockers, {w} warnings")
    return 1 if e > 0 else 0


def cmd_diagnose(args) -> int:
    cfg = _cfg_or_die(args.key)
    policy = getattr(args, "policy", None)
    print(f"=== diagnose {args.key} (runtime"
          + (f", policy={policy}" if policy else "")
          + ") ===\n")
    findings = diagnose_all(cfg, port=args.port, policy=policy)
    e = w = 0
    for f in findings:
        if f.severity == "error" and not f.passed:
            print(f"  ✗ ERROR  {f.name:<35}  {f.message}")
            e += 1
        elif f.severity == "warning" and not f.passed:
            print(f"  ⚠ WARN   {f.name:<35}  {f.message}")
            w += 1
        else:
            print(f"  ✓ ok     {f.name:<35}  {f.message}")
    print(f"\n  Summary: {e} blockers, {w} warnings")
    return 1 if e > 0 else 0


def cmd_verify(args) -> int:
    cfg = _cfg_or_die(args.key)
    print(f"=== verify {args.key} (bench vs reference) ===\n")
    if cfg.reference_metrics is None:
        print("  ✗ ERROR  no reference_metrics — run `bench-and-update` first")
        return 1

    rm = cfg.reference_metrics
    print(f"Reference: {rm.long_gen_sustained_tps:.1f} TPS / "
          f"{rm.tool_call_score} tool / "
          f"CV {rm.stability_cv_pct:.2f}% / "
          f"VRAM {rm.vram_total_mib} MiB")
    print(f"Bench'd:   {rm.measured_at} on {rm.vllm_pin}")
    print()
    results = verify(cfg, port=args.port)
    e = w = 0
    for r in results:
        if r.severity == "error" and not r.passed:
            mark = "✗ ERROR  "
            e += 1
        elif r.severity == "warning" and not r.passed:
            mark = "⚠ WARN   "
            w += 1
        else:
            mark = "✓ ok     "
        print(f"  {mark}{r.metric:<20}  expected={r.expected:<15} "
              f"actual={r.actual:<15} {r.delta}")
    print(f"\n  Summary: {e} blockers, {w} warnings")
    return 1 if e > 0 else 0


def cmd_where(args) -> int:
    src = source_of(args.key)
    if src is None:
        print(f"ERROR: config '{args.key}' not found", file=sys.stderr)
        return 1
    cfg = get(args.key)
    print(f"{args.key}:")
    print(f"  tier:           {src}")
    print(f"  title:          {cfg.title}")
    print(f"  schema_version: {cfg.schema_version}")
    print(f"  maintainer:     {cfg.maintainer}")
    if cfg.last_validated:
        print(f"  last_validated: {cfg.last_validated}")
    if cfg.genesis_pin:
        print(f"  genesis_pin:    {cfg.genesis_pin}")
    if cfg.vllm_pin_required:
        print(f"  vllm_pin:       {cfg.vllm_pin_required}")
    return 0


def cmd_launch(args) -> int:
    cfg = _cfg_or_die(args.key)
    script = cfg.to_launch_script()
    if args.dry_run:
        print("# DRY RUN — would execute:")
        print(script)
        return 0
    if not args.skip_preflight:
        print(f"=== preflight {args.key} ===")
        checks = preflight_all(cfg)
        for c in checks:
            if c.severity == "error" and not c.passed:
                print(f"  ✗ ERROR: {c.name} — {c.message}")
        if preflight_blockers(checks):
            print("\nERROR: preflight has blockers. Use --skip-preflight to "
                  "override (not recommended).", file=sys.stderr)
            return 1
        print("  ✓ preflight clean\n")
    import subprocess
    print(f"=== launching {args.key} ===")
    proc = subprocess.run(["bash", "-c", script], check=False)
    return proc.returncode


def cmd_new(args) -> int:
    if args.template:
        src = get(args.template)
        if src is None:
            print(f"ERROR: template '{args.template}' not found",
                  file=sys.stderr)
            return 1
        from copy import deepcopy
        new_cfg = deepcopy(src)
        new_cfg.key = args.key
        new_cfg.title = f"{src.title} (copy: {args.key})"
        new_cfg.maintainer = "<your-username>"
        new_cfg.last_validated = None
        new_cfg.reference_metrics = None
        new_cfg.verified_on = []
        new_cfg.lifecycle = "experimental"
        from vllm.sndr_core.model_configs.registry import _user_dir
        out_dir = _user_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.key}.yaml"
        if out_path.exists() and not args.force:
            print(f"ERROR: {out_path} exists. Use --force to overwrite.",
                  file=sys.stderr)
            return 1
        out_path.write_text(dump_yaml(new_cfg))
        print(f"✓ Created {out_path}")
        print(f"  Edit it, then `genesis model-config launch {args.key}`.")
        print(f"  After bench, `genesis model-config bench-and-update "
              f"{args.key}` to capture metrics.")
        return 0
    elif args.from_running:
        # Docker-inspect-based captor (audit C2 closure 2026-05-16):
        # parses Entrypoint + Cmd + Config.Env + Mounts + HostConfig and
        # reverse-engineers a ModelConfig YAML. Read-only; no engine-side
        # introspection — works against any vllm/vllm-openai derivative.
        from vllm.sndr_core.compat.from_running import (
            CaptureError, capture_from_running,
        )
        from vllm.sndr_core.model_configs.registry import _user_dir
        try:
            new_cfg = capture_from_running(
                args.from_running,
                key=args.key,
                maintainer=getattr(args, "maintainer", None)
                    or "<your-username>",
            )
        except CaptureError as exc:
            print(f"ERROR: --from-running capture failed: {exc}",
                  file=sys.stderr)
            return 1
        out_dir = _user_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.key}.yaml"
        if out_path.exists() and not args.force:
            print(f"ERROR: {out_path} exists. Use --force to overwrite.",
                  file=sys.stderr)
            return 1
        out_path.write_text(dump_yaml(new_cfg))
        print(f"✓ Captured running container -> {out_path}")
        print()
        print("Review checklist (auto-capture cannot infer these):")
        print("  - hardware.gpu_match_keys: replace "
              "'__REPLACE_WITH_HOST_GPU_KEY__' with your GPU id "
              "(a5000, a100-40gb, h100, rtx-3090, …)")
        print("  - hardware.min_vram_per_gpu_mib: replace placeholder "
              "value '1' with the actual minimum VRAM in MiB")
        print("  - docker.image_digest: pin via `docker inspect "
              "-f '{{index .RepoDigests 0}}' <image>`")
        print("  - mounts: replace absolute host paths with ${var} symbolic "
              "mount references where portability matters")
        print()
        print(f"Then run: sndr model-config validate {args.key}")
        return 0
    else:
        print("ERROR: --template OR --from-running required", file=sys.stderr)
        return 1


def _git_short_sha(repo_root: Path) -> str | None:
    """Return short HEAD SHA of repo at `repo_root`, or None."""
    import subprocess
    try:
        r = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short=7", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip() or None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _format_reference_metrics_block(metrics: dict, genesis_pin: str | None,
                                     vllm_pin: str | None,
                                     bench_method: str) -> str:
    """Render `reference_metrics:` YAML block. Caller surgically swaps it in."""
    from datetime import datetime, timezone
    measured = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "reference_metrics:",
        f"  measured_at: '{measured}'",
        f"  bench_method: {bench_method}",
    ]
    # Order matches schema.ReferenceMetrics for human readability
    field_order = [
        "long_gen_sustained_tps", "long_gen_mean_lat_s",
        "short_gen_tps",
        "tool_call_score",
        "stability_mean_s", "stability_cv_pct",
        "concurrent_4_total_s",
        "vram_used_mib_per_gpu", "vram_total_mib",
    ]
    for k in field_order:
        if k not in metrics:
            continue
        v = metrics[k]
        if isinstance(v, list):
            lines.append(f"  {k}: {v}")
        elif isinstance(v, str):
            lines.append(f"  {k}: '{v}'")
        else:
            lines.append(f"  {k}: {v}")
    if genesis_pin:
        lines.append(f"  genesis_pin: {genesis_pin}")
    if vllm_pin:
        lines.append(f"  vllm_pin: {vllm_pin}")
    return "\n".join(lines)


def _swap_yaml_top_key(text: str, key: str, new_block: str) -> str:
    """Replace a top-level YAML key's value (scalar or nested block).

    Preserves all other text — comments, ordering, blank lines, anchors.
    Matches `^<key>:` followed by either an inline value (everything up to
    EOL) or a nested block (subsequent indented lines).
    """
    import re
    pat = re.compile(
        rf"^{re.escape(key)}:[^\n]*(?:\n[ \t]+[^\n]*)*",
        re.MULTILINE,
    )
    m = pat.search(text)
    if not m:
        # Append at end (separated by blank line)
        return text.rstrip() + "\n\n" + new_block + "\n"
    return text[:m.start()] + new_block + text[m.end():]


def _bump_top_scalar(text: str, key: str, new_value: str) -> str:
    """Surgical replace of `<key>: <scalar>` (single line). No-op if absent."""
    import re
    pat = re.compile(rf"^({re.escape(key)}:)[ \t]+[^\n]*$", re.MULTILINE)
    repl = rf"\1 {new_value}"
    return pat.sub(repl, text, count=1)


def cmd_promote(args) -> int:
    """W-A 2026-05-06: promote a community config along its lifecycle.

    Lifecycle ladder:
      community-test → community-dev → community-prod

    Promotion gates:
      community-test  → community-dev: requires `genesis model-config verify <key>`
                        to pass. Adds an entry to verified_by automatically.
      community-dev   → community-prod: requires ≥2 verified_by entries
                        (cross-rig validation), reference_metrics non-null,
                        AND ≥7 days since test_started_at (cooling-off window).

    Usage:
      genesis model-config promote <key> --to community-dev [--rig-tag rtx-a5000]
                                                            [--handle sandermage]
      genesis model-config promote <key> --to community-prod [--force]

    --force bypasses the cooling-off window only (NOT the cross-rig requirement).
    Schema validation always runs after promotion; on failure the change is
    rolled back.
    """
    from datetime import datetime
    cfg = _cfg_or_die(args.key)
    yaml_path = path_for(args.key)
    if yaml_path is None:
        print(f"ERROR: cannot resolve YAML path for '{args.key}' "
              f"(only writable for user-tier configs)", file=sys.stderr)
        return 1

    target = args.to
    valid_targets = ("community-dev", "community-prod")
    if target not in valid_targets:
        print(f"ERROR: --to must be one of {valid_targets} "
              f"(got '{target}')", file=sys.stderr)
        return 1

    current = cfg.lifecycle
    valid_transitions = {
        "community-test": "community-dev",
        "community-dev": "community-prod",
    }
    if target != valid_transitions.get(current):
        print(f"ERROR: cannot promote from '{current}' to '{target}'. "
              f"Valid path: community-test → community-dev → community-prod. "
              f"Current state: '{current}'.", file=sys.stderr)
        return 1

    # Pre-promotion gates per target
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if target == "community-dev":
        # community-test → community-dev: needs successful verify
        # We don't run verify here (operator should have just done it);
        # we just append the verifier entry.
        rig_tag = getattr(args, "rig_tag", None) or "unknown-rig"
        handle = getattr(args, "handle", None) or cfg.maintainer
        new_verifier = f"{rig_tag}@{handle}-{today}"
        if new_verifier in cfg.verified_by:
            print(f"WARNING: verifier '{new_verifier}' already in "
                  f"verified_by — adding anyway")
        new_verified_by = list(cfg.verified_by) + [new_verifier]
        cfg.verified_by = new_verified_by
        cfg.lifecycle = "community-dev"
    elif target == "community-prod":
        # community-dev → community-prod: cross-rig + age + reference
        if len(cfg.verified_by) < 2:
            print(f"ERROR: community-prod requires ≥2 verified_by entries. "
                  f"Got {len(cfg.verified_by)}: {cfg.verified_by}",
                  file=sys.stderr)
            return 1
        if cfg.reference_metrics is None:
            print(f"ERROR: community-prod requires reference_metrics. "
                  f"Run `genesis model-config bench-and-update {args.key}` "
                  f"first.", file=sys.stderr)
            return 1
        if cfg.test_started_at and not getattr(args, "force", False):
            try:
                started = datetime.strptime(cfg.test_started_at, "%Y-%m-%d")
                age_days = (datetime.utcnow() - started).days
                if age_days < 7:
                    print(f"ERROR: community-prod requires ≥7 days in "
                          f"community-test/-dev (got {age_days} days since "
                          f"{cfg.test_started_at}). Use --force to bypass "
                          f"cooling-off window.", file=sys.stderr)
                    return 1
            except ValueError:
                print(f"WARNING: test_started_at='{cfg.test_started_at}' "
                      f"not parseable as YYYY-MM-DD — proceeding")
        cfg.lifecycle = "community-prod"

    # Post-edit validation — if anything off, abort
    try:
        cfg.validate()
    except Exception as e:
        print(f"ERROR: promoted config fails schema validation: {e}",
              file=sys.stderr)
        return 1

    # Write back to YAML — surgical edit on lifecycle line + verified_by list
    text = yaml_path.read_text()
    text = _bump_top_scalar(text, "lifecycle", target)
    # verified_by is a list — re-emit if changed
    if target == "community-dev":
        verified_yaml = "verified_by:\n" + "\n".join(
            f"  - '{v}'" for v in cfg.verified_by) + "\n"
        # Replace existing block or append
        import re
        if re.search(r"^verified_by:.*\n(?:  - .*\n)*", text, re.MULTILINE):
            text = re.sub(
                r"^verified_by:.*\n(?:  - .*\n)*",
                verified_yaml, text, count=1, flags=re.MULTILINE,
            )
        else:
            text += "\n" + verified_yaml
    yaml_path.write_text(text)
    print(f"OK: '{args.key}' promoted {current} → {target}")
    print(f"   YAML: {yaml_path}")
    if target == "community-dev":
        print(f"   verified_by now has {len(cfg.verified_by)} entries")
    return 0


def cmd_bench_and_update(args) -> int:
    """Run bench against running config, write metrics back into YAML.

    Surgical edit — preserves comments, ordering, anchors. Looks up the
    config's source file via registry.path_for(). With --promote, also
    flips lifecycle: experimental → stable and refreshes last_validated +
    genesis_pin.
    """
    cfg = _cfg_or_die(args.key)
    yaml_path = path_for(args.key)
    if yaml_path is None:
        print(f"ERROR: cannot resolve YAML path for '{args.key}'",
              file=sys.stderr)
        return 1
    if not yaml_path.is_file():
        print(f"ERROR: {yaml_path} does not exist", file=sys.stderr)
        return 1

    print(f"=== bench-and-update {args.key} ===")
    print(f"  source: {yaml_path}")

    p = args.port or (cfg.docker.port if cfg.docker else 8000)
    print(f"  benching against http://localhost:{p} ...")
    metrics = bench_metrics(cfg, port=p)
    if not metrics:
        print("ERROR: all benches failed — is the container running?",
              file=sys.stderr)
        return 1

    # Provenance
    repo_root = Path(__file__).resolve().parents[3]  # project root
    genesis_pin = _git_short_sha(repo_root) or cfg.genesis_pin
    vllm_pin = cfg.vllm_pin_required
    bench_method = (
        "genesis model-config bench-and-update "
        "(verify._bench_long_gen×3 / tool×10 / stability×5)"
    )

    print("  captured:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")
    if genesis_pin:
        print(f"    genesis_pin: {genesis_pin}")
    if vllm_pin:
        print(f"    vllm_pin: {vllm_pin}")

    # Build new reference_metrics: block
    new_block = _format_reference_metrics_block(
        metrics, genesis_pin=genesis_pin, vllm_pin=vllm_pin,
        bench_method=bench_method,
    )

    text = yaml_path.read_text()
    new_text = _swap_yaml_top_key(text, "reference_metrics", new_block)

    # Optional bumps
    if args.promote:
        new_text = _bump_top_scalar(new_text, "lifecycle", "stable")
        from datetime import date
        new_text = _bump_top_scalar(
            new_text, "last_validated", f"'{date.today().isoformat()}'"
        )
        if genesis_pin:
            new_text = _bump_top_scalar(new_text, "genesis_pin", genesis_pin)

    if args.dry_run:
        print("\n--- DIFF (dry-run) ---")
        print(new_block)
        if args.promote:
            print("  + lifecycle: stable")
            print("  + last_validated bumped")
        return 0

    yaml_path.write_text(new_text)
    print(f"\n✓ wrote {yaml_path}")
    if args.promote:
        print("  promoted lifecycle: stable, refreshed last_validated + "
              "genesis_pin")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="genesis model-config",
        description="Manage vetted model launch configurations",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="enumerate all configs")
    p_list.add_argument(
        "--include-tested", action="store_true",
        help="Show tested/QA configs alongside working ones "
             "(default: tested entries are hidden so they don't "
             "pollute the 'what should I actually launch' view).",
    )
    p_list.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON instead of the human table.",
    )
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="print full YAML")
    p_show.add_argument("key")
    p_show.set_defaults(func=cmd_show)

    p_render = sub.add_parser("render", help="emit launch script")
    p_render.add_argument("key")
    p_render.add_argument(
        "--runtime",
        choices=["docker", "podman", "kubernetes", "lxc_proxmox", "bare_metal"],
        default=None,
        help=(
            "override deploy.default runtime (must satisfy "
            "deploy.<runtime>=True for the config OR pass --force). "
            "Currently implemented: docker (default), bare_metal "
            "(Proxmox LXC venv), podman (docker-compatible w/ "
            "--device nvidia.com/gpu=all), kubernetes "
            "(Deployment+Service+ConfigMap manifest, ready for "
            "kubectl apply), lxc_proxmox (runnable Proxmox LXC "
            "deployment script — review host-specific env vars "
            "before executing)."
        ),
    )
    p_render.add_argument(
        "--force", action="store_true",
        help=(
            "render even if config doesn't declare support for --runtime "
            "(deploy.<runtime>=False). Output is untested — validate before "
            "deploying."
        ),
    )
    p_render.add_argument(
        "--mode",
        choices=["wheel", "dev", "dev_legacy"],
        default="wheel",
        help=(
            "Bare-metal launch script mode (only consulted when "
            "--runtime bare_metal). 'wheel' (default, production) verifies "
            "vllm-sndr-core wheel is installed and fails fast otherwise; no "
            "editable install. 'dev' editable-installs the plugin source — "
            "errors visible. 'dev_legacy' restores the silent `|| true` "
            "behaviour — DEPRECATED, kept for backward compat. "
            "Audit closure 2026-05-08 (S0.3)."
        ),
    )
    p_render.set_defaults(func=cmd_render)

    p_save = sub.add_parser("save", help="write launch script to file")
    p_save.add_argument("key")
    p_save.add_argument("path")
    p_save.set_defaults(func=cmd_save)

    p_audit = sub.add_parser("audit",
                             help="run 16 audit_rules (cross-patch checks)")
    p_audit.add_argument("key")
    p_audit.set_defaults(func=cmd_audit)

    p_validate = sub.add_parser("validate",
                                help="schema + audit (recommended pre-launch)")
    p_validate.add_argument("key")
    p_validate.set_defaults(func=cmd_validate)

    p_pre = sub.add_parser("preflight",
                           help="pre-launch environment checks")
    p_pre.add_argument("key")
    p_pre.set_defaults(func=cmd_preflight)

    p_diag = sub.add_parser("diagnose",
                            help="runtime diagnose — query running container")
    p_diag.add_argument("key")
    p_diag.add_argument("--port", type=int, default=None)
    p_diag.add_argument(
        "--policy",
        choices=("compat", "safe", "minimal"),
        default=None,
        help=(
            "Phase D (2026-05-16): compare against the policy-filtered "
            "plan.env instead of cfg.genesis_env raw. Use when the "
            "container was launched with the same --policy flag, "
            "otherwise the legacy diff flags expected drop-outs as "
            "errors."
        ),
    )
    p_diag.set_defaults(func=cmd_diagnose)

    p_ver = sub.add_parser("verify",
                           help="bench vs reference_metrics (CI gate)")
    p_ver.add_argument("key")
    p_ver.add_argument("--port", type=int, default=None)
    p_ver.set_defaults(func=cmd_verify)

    p_where = sub.add_parser("where", help="show source tier")
    p_where.add_argument("key")
    p_where.set_defaults(func=cmd_where)

    p_new = sub.add_parser("new", help="create a user config")
    p_new.add_argument("key")
    p_new.add_argument("--template", help="seed from existing builtin/user config")
    p_new.add_argument(
        "--from-running", metavar="CONTAINER",
        help=(
            "capture YAML from a running docker/podman container — runs "
            "`docker inspect <container>` and reverse-engineers a "
            "ModelConfig from Entrypoint+Cmd+Env+Mounts. Read-only."
        ),
    )
    p_new.add_argument(
        "--maintainer",
        help="github-style username for the captured YAML header "
             "(default: <your-username>)",
    )
    p_new.add_argument("--force", action="store_true")
    p_new.set_defaults(func=cmd_new)

    # W-A 2026-05-06 — community lifecycle promotion CLI
    p_promote = sub.add_parser(
        "promote",
        help="promote community config along community-test → -dev → -prod",
        description=(
            "Promote a community-submitted config along the lifecycle "
            "ladder. Schema gates (cross-rig validation, reference_metrics, "
            "cooling-off window) enforce safe progression."
        ),
    )
    p_promote.add_argument("key")
    p_promote.add_argument(
        "--to", required=True,
        choices=["community-dev", "community-prod"],
        help="target lifecycle state",
    )
    p_promote.add_argument(
        "--rig-tag", default=None,
        help="hardware tag for verifier entry (e.g. 'rtx-a5000', '2x-3090')",
    )
    p_promote.add_argument(
        "--handle", default=None,
        help="GitHub handle for verifier entry (defaults to maintainer)",
    )
    p_promote.add_argument(
        "--force", action="store_true",
        help="bypass cooling-off window (NOT cross-rig requirement)",
    )
    p_promote.set_defaults(func=cmd_promote)

    p_lau = sub.add_parser("launch", help="execute the rendered script")
    p_lau.add_argument("key")
    p_lau.add_argument("--dry-run", action="store_true")
    p_lau.add_argument("--skip-preflight", action="store_true")
    p_lau.set_defaults(func=cmd_launch)

    p_bau = sub.add_parser("bench-and-update",
                           help="bench + write metrics back into YAML")
    p_bau.add_argument("key")
    p_bau.add_argument("--port", type=int, default=None)
    p_bau.add_argument("--promote", action="store_true",
                       help="also flip lifecycle: experimental → stable + "
                            "refresh last_validated + genesis_pin")
    p_bau.add_argument("--dry-run", action="store_true",
                       help="print the would-be reference_metrics block, "
                            "do NOT write the file")
    p_bau.set_defaults(func=cmd_bench_and_update)

    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
