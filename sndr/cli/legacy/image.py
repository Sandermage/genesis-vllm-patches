# SPDX-License-Identifier: Apache-2.0
"""C3 (UNIFIED_CONFIG plan 2026-05-09) — `sndr image` Docker image management.

Reads a preset's `docker.image` + `docker.image_digest` and:

  sndr image inspect <key>     — `docker inspect` + verify digest matches
  sndr image pull <key>        — `docker pull` the declared image
  sndr image resolve <key>     — print resolved image:tag@sha256:...
  sndr image verify <key>      — fail loudly when digest mismatches
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from typing import Any, Optional

from . import _io


__all__ = ["add_argparser", "run_inspect", "run_pull", "run_resolve",
           "run_verify"]


def add_argparser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "image",
        help="Docker image management — inspect/pull/verify (UNIFIED_CONFIG C3).",
        description=(
            "Wraps docker inspect/pull around the preset's docker.image + "
            "docker.image_digest fields. Verifies the local image's "
            "RepoDigest matches the declared sha256."
        ),
    )
    sub = p.add_subparsers(dest="image_cmd", required=True)

    for cmd, helper, fn in (
        ("inspect", "Print docker inspect for the preset's image", run_inspect),
        ("pull",    "docker pull the preset's image:tag",          run_pull),
        ("resolve", "Print resolved image reference + digest",     run_resolve),
        ("verify",  "Verify local image digest matches declared",  run_verify),
    ):
        sp = sub.add_parser(cmd, help=helper)
        sp.add_argument("config", help="model_config preset key")
        sp.add_argument("--json", action="store_true",
                          help="JSON output where applicable")
        sp.set_defaults(func=fn)


def _resolve(key: str):
    from sndr.model_configs.registry import get
    cfg = get(key)
    if cfg is None:
        _io.warn(f"unknown preset key {key!r}")
        return None
    if cfg.docker is None:
        _io.warn(f"preset {key!r} has no docker block")
        return None
    return cfg


def _docker_inspect(image: str) -> Optional[dict]:
    if shutil.which("docker") is None:
        return None
    try:
        r = subprocess.run(
            ["docker", "inspect", image],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return {"_error": r.stderr.strip()}
        data = json.loads(r.stdout)
        if isinstance(data, list) and data:
            return data[0]
    except Exception:
        return None
    return None


# ─── inspect

def run_inspect(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    image = cfg.docker.image
    info = _docker_inspect(image)
    if info is None:
        _io.error(f"docker not available or inspect failed for {image!r}")
        return 1
    if "_error" in info:
        _io.error(f"inspect failed: {info['_error']}")
        return 1
    if args.json:
        print(json.dumps(info, indent=2))
        return 0
    digests = info.get("RepoDigests", [])
    print(f"sndr image inspect '{args.config}'")
    print("─" * 60)
    print(f"  image:        {image}")
    print(f"  declared dgst: {cfg.docker.image_digest or '_unset_'}")
    print(f"  Id:           {info.get('Id', '?')}")
    print(f"  Created:      {info.get('Created', '?')}")
    print(f"  Size:         {info.get('Size', 0) / (1<<30):.2f} GiB")
    print("  RepoDigests:")
    for d in digests:
        print(f"    {d}")
    return 0


# ─── pull

def run_pull(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    if shutil.which("docker") is None:
        _io.error("docker not on PATH")
        return 1
    image = cfg.docker.image
    print(f"sndr image pull {image}")
    r = subprocess.run(["docker", "pull", image], timeout=1200)
    return r.returncode


# ─── resolve

def run_resolve(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    image = cfg.docker.image
    info = _docker_inspect(image)
    digests = (info or {}).get("RepoDigests", [])
    out = {
        "preset": args.config,
        "image_tag": image,
        "image_digest_declared": cfg.docker.image_digest,
        "local_digests": digests,
    }
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"sndr image resolve '{args.config}'")
        print("─" * 60)
        print(f"  tag:                {image}")
        print(f"  declared digest:    {cfg.docker.image_digest or '_unset_'}")
        print("  local repo digests:")
        for d in digests:
            print(f"    {d}")
    return 0


# ─── verify

def run_verify(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    declared = cfg.docker.image_digest
    if not declared:
        _io.warn(f"preset {args.config!r} has no image_digest declared — "
                  f"add B2 digest pin to enable strict verify")
        return 0  # not declared = not enforced = not a failure
    image = cfg.docker.image
    info = _docker_inspect(image)
    if info is None:
        _io.error(f"docker inspect failed for {image!r}")
        return 1
    if "_error" in info:
        _io.error(f"inspect: {info['_error']}")
        return 1
    digests = info.get("RepoDigests", [])
    # Match: declared 'org/name@sha256:xxx' must equal at least one
    # entry in RepoDigests, OR the sha256 part must match.
    declared_sha = (
        declared.split("@", 1)[1] if "@" in declared else declared
    )
    matched = False
    for d in digests:
        if d == declared:
            matched = True
            break
        if "@" in d and d.split("@", 1)[1] == declared_sha:
            matched = True
            break
    if matched:
        _io.success("image digest matches declared")
        return 0
    _io.error("image digest MISMATCH:")
    print(f"  declared: {declared}")
    print("  local:")
    for d in digests:
        print(f"    {d}")
    return 1
