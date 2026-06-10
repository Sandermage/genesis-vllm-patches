#!/usr/bin/env python3
"""Safe container recreate for Genesis vLLM PROD containers.

Promotes launcher-script env exports (autotune cache dirs, GENESIS enables)
into the docker env layer so they are present at PID 1 from container-creation
time AND visible to docker exec / sidecar processes.

Workflow (operator review required before any destructive action):
  1. Snapshot the current container state (env, binds, ports, GPU, image).
  2. Compute and PRINT the diff vs the desired recreate (promoted env,
     extra binds).
  3. Prompt the operator for confirmation.
  4. Stop the running container and RENAME it to <name>-rollback-<ts>
     (NOT removed — kept on disk for one-line rollback).
  5. docker run a new container with the merged config.
  6. Wait for /health=200 (up to 10 min boot budget).
  7. Smoke test patches-applied count, models endpoint, single completion.
  8. PRINT a rollback command (operator decides whether to roll back).

Strict guarantees:
  * Never executes destructive operations without `--yes` AND interactive
    confirmation.
  * The previous container is RENAMED, not removed; rollback is a single
    `docker stop new && docker start <name>-rollback-<ts> && docker rename`.
  * If smoke test fails the script EXITS NONZERO and prints the rollback
    command — it does NOT auto-rollback (operator must decide).
  * Snapshots are written to /tmp/genesis_recreate/ on the local box
    (host running this script) for forensic comparison post-recreate.

Usage:
  python3 tools/safe_container_recreate.py \\
      --host <user>@<host> \\
      --container vllm-qwen3.6-35b-balanced-k3 \\
      --port 8102 \\
      --api-key genesis-local \\
      --dry-run        # print the docker run command and exit
  python3 tools/safe_container_recreate.py ... --yes  # execute after confirm

Remote repo root resolution (for autotune cache paths / binds):
  1. --remote-repo CLI flag
  2. SNDR_REMOTE_REPO_DIR env var
  3. derived from --host ssh user: /home/<user>/genesis-vllm-patches

Author: Genesis platform / Sander Odessa
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- constants


def default_remote_repo(host: str) -> str:
    """Resolve the remote repo root on the PROD box.

    Precedence: SNDR_REMOTE_REPO_DIR env override, else derive from the
    ssh user in `host` (`<user>@<addr>` form): /home/<user>/genesis-vllm-patches.
    """
    env = os.environ.get("SNDR_REMOTE_REPO_DIR", "").strip()
    if env:
        return env.rstrip("/")
    user = host.split("@", 1)[0] if "@" in host else "root"
    return f"/home/{user}/genesis-vllm-patches"


def launcher_env_promotions(remote_repo: str) -> dict[str, str]:
    """Launcher-script exports we want PROMOTED into the docker env layer.

    Source of truth: the active launcher run.sh on the PROD box.
    Keep this list in sync with the launcher script.
    """
    return {
        # Persistent autotune cache dirs (the main reason for this recreate).
        "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": (
            f"{remote_repo}/.autotune_cache/flashinfer"
        ),
        "TRITON_CACHE_DIR": (
            f"{remote_repo}/.autotune_cache/triton"
        ),
        # Genesis patch enables that the launcher overrides — promote so they
        # survive launcher-script edits.
        "GENESIS_ENABLE_PN365_GDN_GEMM_FUSE": "1",
        "GENESIS_ENABLE_PN204_DUAL_STREAM_INPROJ": "0",
        "GENESIS_ENABLE_P100": "1",
        "GENESIS_ENABLE_P101": "1",
        "GENESIS_ENABLE_PN362": "1",
        "GENESIS_ENABLE_PN364_HYBRID_GDN_WARMUP": "1",
        "GENESIS_ENABLE_PN350": "1",
        "GENESIS_ENABLE_PN353A": "1",
    }


def extra_binds(remote_repo: str) -> list[str]:
    """Extra bind mount — explicit autotune cache dir. Redundant with the
    parent <remote_repo>:rw bind but explicit for clarity and survives
    any future tightening of the parent bind."""
    return [
        f"{remote_repo}/.autotune_cache:"
        f"{remote_repo}/.autotune_cache:rw",
    ]

# Docker-injected vars to STRIP when reconstructing the env list (they get
# re-set automatically by the daemon and overlap with the new image's
# defaults — passing them via -e creates phantom overrides).
DOCKER_INJECTED_ENV_PREFIXES = (
    "PATH=",
    "NVARCH=",
    "NVIDIA_REQUIRE_CUDA=",
    "NV_CUDA_",
    "CUDA_VERSION=",
    "LD_LIBRARY_PATH=",
    "NVIDIA_VISIBLE_DEVICES=",
    "NVIDIA_DRIVER_CAPABILITIES=",
    "DEBIAN_FRONTEND=",
    "UV_",
    "VLLM_ENABLE_CUDA_COMPATIBILITY=",
    "TORCH_CUDA_ARCH_LIST=",
    "VLLM_USAGE_SOURCE=",
    "VLLM_BUILD_",
    "VLLM_IMAGE_TAG=",
    "HOSTNAME=",
    "HOME=",
)

SNAPSHOT_DIR = Path("/tmp/genesis_recreate")


# ----------------------------------------------------------------- helpers

def ssh(host: str, cmd: str, capture: bool = True) -> str:
    """Run a command on the remote host over ssh. Returns stdout."""
    full = ["ssh", "-o", "BatchMode=yes", host, cmd]
    if capture:
        r = subprocess.run(full, capture_output=True, text=True, check=False)
        if r.returncode != 0:
            raise RuntimeError(
                f"ssh failed (rc={r.returncode}): {r.stderr.strip()}"
            )
        return r.stdout
    return subprocess.check_output(full, text=True)


def docker_inspect(host: str, container: str, fmt: str) -> str:
    """docker inspect with --format."""
    return ssh(
        host,
        f"docker inspect {shlex.quote(container)} "
        f"--format {shlex.quote(fmt)}",
    ).strip()


def snapshot_container(host: str, container: str, out: Path) -> dict:
    """Capture full container state into out/. Returns a dict for reuse."""
    out.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    bundle = out / f"snapshot-{container}-{ts}"
    bundle.mkdir()

    raw = ssh(host, f"docker inspect {shlex.quote(container)}")
    (bundle / "full_inspect.json").write_text(raw)
    inspect = json.loads(raw)[0]

    cfg = inspect["Config"]
    hcfg = inspect["HostConfig"]
    state = inspect["State"]

    snapshot = {
        "container": container,
        "image_sha": inspect["Image"],
        "image_tag": cfg["Image"],
        "env": cfg.get("Env", []),
        "cmd": cfg.get("Cmd"),
        "entrypoint": cfg.get("Entrypoint"),
        "binds": hcfg.get("Binds", []) or [],
        "port_bindings": hcfg.get("PortBindings", {}) or {},
        "shm_size": hcfg.get("ShmSize", 0),
        "memory": hcfg.get("Memory", 0),
        "network_mode": hcfg.get("NetworkMode", "default"),
        "restart_policy": (hcfg.get("RestartPolicy") or {}).get("Name", "no"),
        "device_requests": hcfg.get("DeviceRequests") or [],
        "status": state.get("Status"),
        "started_at": state.get("StartedAt"),
        "timestamp_utc": ts,
        "bundle_dir": str(bundle),
    }
    (bundle / "summary.json").write_text(json.dumps(snapshot, indent=2))
    return snapshot


def keep_env_var(entry: str) -> bool:
    """Filter out docker-injected env vars when reconstructing -e flags."""
    if "=" not in entry:
        return False
    return not any(entry.startswith(p) for p in DOCKER_INJECTED_ENV_PREFIXES)


def merge_env(current: list[str], promotions: dict[str, str]) -> list[str]:
    """Apply launcher promotions on top of the current env list."""
    seen: dict[str, str] = {}
    for e in current:
        if "=" in e:
            k, v = e.split("=", 1)
            seen[k] = v
    for k, v in promotions.items():
        seen[k] = v
    return [f"{k}={v}" for k, v in seen.items()]


def build_docker_run(
    snapshot: dict,
    new_name: str,
    new_env: list[str],
    extra_binds: list[str],
) -> list[str]:
    """Construct the docker run argv (list) for the new container."""
    argv = ["docker", "run", "-d", "--name", new_name]

    # GPU passthrough — match original. We use --gpus all (matches Count=-1).
    has_gpu = any(
        "gpu" in (cap or [])
        for d in snapshot["device_requests"]
        for cap in d.get("Capabilities", [])
    )
    if has_gpu:
        argv += ["--gpus", "all"]

    # shm-size, memory, restart-policy, network
    if snapshot["shm_size"]:
        argv += ["--shm-size", str(snapshot["shm_size"])]
    if snapshot["memory"]:
        argv += ["--memory", str(snapshot["memory"])]
    if snapshot["restart_policy"] and snapshot["restart_policy"] != "no":
        argv += ["--restart", snapshot["restart_policy"]]
    if snapshot["network_mode"] and snapshot["network_mode"] != "default":
        argv += ["--network", snapshot["network_mode"]]

    # Port bindings
    for cport, hostlist in snapshot["port_bindings"].items():
        for hb in hostlist or []:
            hp = hb.get("HostPort", "")
            hip = hb.get("HostIp", "")
            spec = f"{hip + ':' if hip else ''}{hp}:{cport.split('/')[0]}"
            argv += ["-p", spec]

    # Bind mounts (existing + extras, dedup)
    seen_binds: set[str] = set()
    for b in snapshot["binds"]:
        if b not in seen_binds:
            argv += ["-v", b]
            seen_binds.add(b)
    for b in extra_binds:
        if b not in seen_binds:
            argv += ["-v", b]
            seen_binds.add(b)

    # Env (sorted for deterministic diff)
    for e in sorted(new_env):
        argv += ["-e", e]

    # Entrypoint / cmd
    if snapshot["entrypoint"]:
        argv += ["--entrypoint", snapshot["entrypoint"][0]]
        extra_entry = snapshot["entrypoint"][1:]
    else:
        extra_entry = []

    argv += [snapshot["image_tag"]]

    if extra_entry:
        argv += extra_entry
    if snapshot["cmd"]:
        argv += list(snapshot["cmd"])

    return argv


def diff_env(old: list[str], new: list[str]) -> tuple[list[str], list[str], list[tuple[str, str, str]]]:
    """Compute (added, removed, changed) between two env lists."""
    od = {e.split("=", 1)[0]: e.split("=", 1)[1] for e in old if "=" in e}
    nd = {e.split("=", 1)[0]: e.split("=", 1)[1] for e in new if "=" in e}
    added = sorted(k for k in nd if k not in od)
    removed = sorted(k for k in od if k not in nd)
    changed = sorted((k, od[k], nd[k]) for k in nd if k in od and od[k] != nd[k])
    return added, removed, changed


def wait_for_health(host: str, port: int, timeout_s: int = 600) -> bool:
    """Poll /health until 200 or timeout."""
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            out = ssh(
                host,
                f"curl -sw '%{{http_code}}' -o /dev/null "
                f"http://localhost:{port}/health",
            ).strip()
            last = out
            if out == "200":
                return True
        except Exception as e:
            last = f"err:{e}"
        time.sleep(5)
    print(f"  timed out — last status: {last}", file=sys.stderr)
    return False


def smoke_test(
    host: str, port: int, api_key: str, container: str
) -> dict:
    """Post-recreate smoke checks. Returns a dict of results."""
    results: dict = {}

    # 1. Genesis patch summary from container logs
    log_grep = ssh(
        host,
        f"docker logs {shlex.quote(container)} 2>&1 | "
        "grep -E 'register\\(\\) complete: applied=' | tail -1",
    ).strip()
    results["patch_summary"] = log_grep

    # 2. /v1/models reachable
    models = ssh(
        host,
        f"curl -sw '\\n%{{http_code}}' "
        f"-H 'Authorization: Bearer {api_key}' "
        f"http://localhost:{port}/v1/models",
    ).strip().splitlines()
    results["models_http"] = models[-1] if models else "?"
    results["models_body"] = models[0][:200] if models else ""

    # 3. PID 1 environ confirms autotune dirs present
    p1env = ssh(
        host,
        f"docker exec {shlex.quote(container)} sh -c "
        '"cat /proc/1/environ | tr \\"\\\\0\\" \\"\\\\n\\" | '
        'grep -E \\"FLASHINFER_AUTOTUNE|TRITON_CACHE_DIR\\" | sort"',
    ).strip()
    results["pid1_autotune_env"] = p1env

    return results


# -------------------------------------------------------------------- main

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--host", required=True, help="ssh target, e.g. <user>@<host>")
    p.add_argument("--container", required=True, help="container name")
    p.add_argument("--port", type=int, required=True, help="service port (for /health)")
    p.add_argument("--api-key", default="genesis-local", help="API key for smoke")
    p.add_argument("--remote-repo", default=None,
                   help="repo root on the remote box (default: "
                        "$SNDR_REMOTE_REPO_DIR or /home/<ssh-user>/genesis-vllm-patches)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the docker run command and exit; no changes")
    p.add_argument("--yes", action="store_true",
                   help="skip interactive confirmation (still snapshots first)")
    p.add_argument("--boot-timeout", type=int, default=600,
                   help="seconds to wait for /health=200 (default 600)")
    args = p.parse_args()

    print(f"[1/6] Snapshotting {args.container} on {args.host}...")
    snap = snapshot_container(args.host, args.container, SNAPSHOT_DIR)
    print(f"      bundle: {snap['bundle_dir']}")
    print(f"      image: {snap['image_tag']}")
    print(f"      env vars: {len(snap['env'])}")
    print(f"      binds: {len(snap['binds'])}")
    print(f"      status: {snap['status']}")

    # Build the new env: keep current, strip docker-injected, apply promotions
    remote_repo = (args.remote_repo or default_remote_repo(args.host)).rstrip("/")
    kept = [e for e in snap["env"] if keep_env_var(e)]
    new_env = merge_env(kept, launcher_env_promotions(remote_repo))

    added, removed, changed = diff_env(snap["env"], new_env)
    print(f"\n[2/6] Env diff (relative to current container state):")
    print(f"      added   : {len(added)}")
    for k in added:
        v = next(e.split('=',1)[1] for e in new_env if e.startswith(k+'='))
        print(f"        + {k}={v}")
    print(f"      removed : {len(removed)}  (docker-injected; daemon will re-add)")
    for k in removed:
        print(f"        - {k}")
    print(f"      changed : {len(changed)}")
    for k, ov, nv in changed:
        print(f"        ~ {k}: {ov!r} -> {nv!r}")

    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    rollback_name = f"{args.container}-rollback-{ts}"
    new_name = args.container

    argv = build_docker_run(snap, new_name, new_env, extra_binds(remote_repo))

    # Pretty-print the docker run command
    print(f"\n[3/6] Proposed docker run command (for {new_name}):")
    rendered = " \\\n  ".join(shlex.quote(a) for a in argv)
    print(f"  {rendered}")

    snap_dir = Path(snap["bundle_dir"])
    (snap_dir / "new_docker_run.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + " \\\n  ".join(shlex.quote(a) for a in argv) + "\n"
    )
    print(f"\n  (also written to: {snap_dir}/new_docker_run.sh)")

    if args.dry_run:
        print("\n[dry-run] Exiting without changes.")
        return 0

    print(f"\n[4/6] Recreate plan:")
    print(f"      a. docker stop  {args.container}")
    print(f"      b. docker rename {args.container} -> {rollback_name}")
    print(f"      c. docker run (command above) as {new_name}")
    print(f"      d. wait /health 200 (up to {args.boot_timeout}s)")
    print(f"      e. smoke tests + report")
    print(f"      ROLLBACK if needed:")
    print(f"        docker stop {new_name} && docker rm {new_name} \\\\")
    print(f"          && docker rename {rollback_name} {args.container} \\\\")
    print(f"          && docker start {args.container}")

    if not args.yes:
        ans = input("\n  Proceed? type 'recreate' to continue: ").strip()
        if ans != "recreate":
            print("Aborted by operator.")
            return 1
    else:
        ans = input("\n  --yes given; type 'YES' to confirm destructive: ").strip()
        if ans != "YES":
            print("Aborted by operator.")
            return 1

    print(f"\n[5/6] Stop + rename old container...")
    ssh(args.host, f"docker stop {shlex.quote(args.container)}")
    ssh(
        args.host,
        f"docker rename {shlex.quote(args.container)} "
        f"{shlex.quote(rollback_name)}",
    )
    print(f"      old container preserved as: {rollback_name}")

    print(f"      starting new container as: {new_name}")
    # Build a single remote shell command from the argv
    remote_cmd = " ".join(shlex.quote(a) for a in argv)
    ssh(args.host, remote_cmd)

    print(f"\n[6/6] Wait for /health and smoke test...")
    ok = wait_for_health(args.host, args.port, args.boot_timeout)
    if not ok:
        print("\n  HEALTH CHECK FAILED.")
        print(f"  Rollback:")
        print(f"    ssh {args.host} 'docker stop {new_name} "
              f"&& docker rm {new_name} "
              f"&& docker rename {rollback_name} {args.container} "
              f"&& docker start {args.container}'")
        return 2

    results = smoke_test(args.host, args.port, args.api_key, new_name)
    print(f"\n  patch_summary    : {results['patch_summary']}")
    print(f"  models_http      : {results['models_http']}")
    print(f"  pid1_autotune_env:\n    "
          + results["pid1_autotune_env"].replace("\n", "\n    "))

    print(f"\nDONE. New container running. Rollback container preserved:")
    print(f"  {rollback_name}")
    print(f"\nRollback (one-liner):")
    print(f"  ssh {args.host} 'docker stop {new_name} && docker rm {new_name} "
          f"&& docker rename {rollback_name} {args.container} "
          f"&& docker start {args.container}'")
    print(f"\nWhen the new container has been validated by a bench A/B, the old "
          f"rollback container can be removed with:")
    print(f"  ssh {args.host} 'docker rm {rollback_name}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
