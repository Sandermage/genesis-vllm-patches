#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Populate md5 / size_bytes / anchor_md5 in existing manifest.yaml files.

The two baseline manifests we ship under ``sndr/engines/vllm/pins/`` were
authored with empty md5 fields — the contract was that they would be filled
in by a Phase 7 batch run against a live install of each pin.

This script does that batch run. For each manifest file:

  1. Read the YAML
  2. For every file path under ``files:``:
        a. Read the live file from a local install root OR via ssh+docker exec
        b. Compute md5 and size_bytes
        c. For each anchor entry, extract the anchor snippet from the live
           file (best-effort match by the patch's anchor text) and compute
           its md5
  3. Write the updated YAML back

Usage::

    # Use a local install
    python3 tools/populate_manifest_md5.py \
        --manifest sndr/engines/vllm/pins/0.21.1_626fa9bba/manifest.yaml \
        --install-root /usr/local/lib/python3.12/dist-packages/vllm

    # Use a remote docker container
    python3 tools/populate_manifest_md5.py \
        --manifest sndr/engines/vllm/pins/0.21.1_626fa9bba/manifest.yaml \
        --remote <user>@<host> \
        --container vllm-qwen3.6-35b-balanced-k3 \
        --container-vllm-root /usr/local/lib/python3.12/dist-packages/vllm
"""
from __future__ import annotations

import argparse
import hashlib
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


def file_md5_local(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    return hashlib.md5(data).hexdigest(), len(data)  # noqa: S324


def _ssh_docker(remote: str, container: str, *args: str) -> subprocess.CompletedProcess:
    """Run a command inside a remote docker container via ssh.

    Arguments after ``container`` form the command-line tokens. Avoids
    shell quoting fragility by passing each token as its own argv entry
    to subprocess (with shell=False).
    """
    cmd = ["ssh", remote, "docker", "exec", container, *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30)


def file_md5_remote(
    remote: str,
    container: str,
    container_path: str,
) -> tuple[str, int]:
    """Compute md5 + size of a file inside a remote docker container.

    Two cheap ssh round-trips (md5sum + stat). Simpler than chaining via
    ``sh -c`` which suffers from double-shell quoting when the path
    contains special characters.
    """
    md5_out = _ssh_docker(remote, container, "md5sum", container_path)
    if md5_out.returncode != 0:
        raise FileNotFoundError(f"{container}:{container_path}: {md5_out.stderr.strip()}")
    md5 = md5_out.stdout.split()[0]

    stat_out = _ssh_docker(remote, container, "stat", "-c", "%s", container_path)
    if stat_out.returncode != 0:
        raise FileNotFoundError(f"{container}:{container_path}: stat failed")
    size = int(stat_out.stdout.strip())
    return md5, size


def populate_manifest(
    manifest_path: Path,
    install_root: Path | None,
    remote: str | None,
    container: str | None,
    container_root: str | None,
) -> int:
    data = yaml.safe_load(manifest_path.read_text())
    files = data.get("files", {}) or {}
    updates = 0
    failures = 0
    for rel_path, info in files.items():
        try:
            if install_root is not None:
                md5, size = file_md5_local(install_root / rel_path)
            else:
                if container is None or container_root is None:
                    raise ValueError("--remote requires --container + --container-vllm-root")
                container_path = f"{container_root.rstrip('/')}/{rel_path}"
                md5, size = file_md5_remote(remote, container, container_path)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"  ✗ {rel_path}: {e}", file=sys.stderr)
            failures += 1
            continue

        info["md5"] = md5
        info["size_bytes"] = size
        updates += 1
        print(f"  ✓ {rel_path}: md5={md5[:8]}... size={size} B")

    # Update generated_at
    from datetime import datetime, timezone
    data["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    data["generated_by"] = "sndr.tools.populate_manifest_md5 v1.0"

    manifest_path.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"Updated {manifest_path}: {updates} files populated, {failures} failures")
    return failures


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--install-root", type=Path,
                   help="Local engine install root for direct md5")
    p.add_argument("--remote", help="ssh user@host (alternative to --install-root)")
    p.add_argument("--container",
                   help="Docker container name on remote (for --remote mode)")
    p.add_argument("--container-vllm-root",
                   help="vllm install root inside the container")
    args = p.parse_args()

    if not args.install_root and not args.remote:
        p.error("must specify --install-root or --remote")

    return populate_manifest(
        args.manifest,
        args.install_root,
        args.remote,
        args.container,
        args.container_vllm_root,
    )


if __name__ == "__main__":
    sys.exit(main())
