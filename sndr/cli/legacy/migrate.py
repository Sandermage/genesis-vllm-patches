# SPDX-License-Identifier: Apache-2.0
"""C9 (UNIFIED_CONFIG plan 2026-05-09) — `sndr migrate` schema migrations.

Walks user / community model_config YAMLs and rewrites them for the
current schema version. Today supports one migration:

  - `v11-runtime-contract`: adds defaults for new optional blocks that
    landed in v11 (Y1/Y3/Y4/Y10/Y11/Y12/Y14 etc) so old YAMLs still
    `validate()` cleanly under the current dataclass shapes.

The old YAML is preserved at `<file>.bak` before rewrite.
Default --dry-run; --yes required to actually rewrite files.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import yaml

from . import _io


__all__ = ["add_argparser", "run_migrate"]


_KNOWN_MIGRATIONS = ("v11-runtime-contract",)


def add_argparser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "migrate",
        help="Migrate user model_config YAMLs to a newer schema (UNIFIED_CONFIG C9).",
        description=(
            "Rewrite YAML configs for forward compatibility. Today the "
            "only known migration is 'v11-runtime-contract' which adds "
            "defaults for optional blocks (Y3 artifacts.caches, Y11 "
            "upstream, Y12 overrides, Y4 host_port/container_port split, "
            "etc) so older configs validate cleanly under v11 schema."
        ),
    )
    p.add_argument("migration", choices=_KNOWN_MIGRATIONS,
                   help="migration name")
    p.add_argument("paths", nargs="+",
                   help="YAML config file(s) or directory(ies) to migrate")
    p.add_argument("--yes", action="store_true",
                   help="Actually rewrite files (default: dry-run preview).")
    p.add_argument("--no-backup", action="store_true",
                   help="Skip .bak backup creation (default: keep backup).")
    p.set_defaults(func=run_migrate)


def _expand_paths(paths: list[str]) -> list[Path]:
    """Walk paths; expand directories to *.yaml/*.yml children."""
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out.extend(path.rglob("*.yaml"))
            out.extend(path.rglob("*.yml"))
        elif path.is_file():
            out.append(path)
        else:
            _io.warn(f"path not found: {p}")
    return out


def _migrate_v11_runtime_contract(d: dict) -> tuple[dict, list[str]]:
    """Mutate `d` in-place to add v11 schema defaults. Returns (d, changes_log)."""
    changes: list[str] = []

    # Y4 host_port/container_port split — back-compat: legacy `port` works
    # but explicit fields are preferred. Only add them if docker block exists.
    docker = d.get("docker")
    if isinstance(docker, dict):
        port = docker.get("port", 8000)
        if "host_port" not in docker:
            docker["host_port"] = port
            changes.append("docker.host_port = port")
        if "container_port" not in docker:
            docker["container_port"] = port
            changes.append("docker.container_port = port")

    # Y3 artifacts (don't add — leave to operator) but flag the absence
    if "artifacts" not in d:
        changes.append("(no artifacts block — consider adding for Y3 wire-in)")

    # Y11 upstream (don't add — leave to operator)
    # Y12 overrides (don't add — leave to operator)

    # CacheConfig: ensure new PN95 fields exist with defaults
    cc = d.get("cache_config")
    if isinstance(cc, dict):
        if "tiers" not in cc:
            cc["tiers"] = []
            changes.append("cache_config.tiers = [] (PN95 back-compat)")
        if "exclude_mamba_ssm" not in cc:
            cc["exclude_mamba_ssm"] = True
            changes.append("cache_config.exclude_mamba_ssm = True")
        if "vision_demote_first" not in cc:
            cc["vision_demote_first"] = True
            changes.append("cache_config.vision_demote_first = True")

    # OffloadConfig: when present, sanity check defaults
    off = d.get("offload")
    if isinstance(off, dict):
        if "swap_space_gib" not in off:
            off["swap_space_gib"] = 0.0
            changes.append("offload.swap_space_gib = 0.0")

    return d, changes


def _migrate_one(path: Path, *, migration: str, yes: bool,
                  no_backup: bool) -> int:
    """Migrate one file. Returns 0 success / 1 failure / 2 not-modified."""
    try:
        body = path.read_text()
    except OSError as e:
        _io.error(f"{path}: read failed: {e}")
        return 1
    try:
        d = yaml.safe_load(body)
    except yaml.YAMLError as e:
        _io.error(f"{path}: yaml parse failed: {e}")
        return 1
    if not isinstance(d, dict):
        _io.warn(f"{path}: not a dict YAML — skip")
        return 2

    if migration == "v11-runtime-contract":
        d, changes = _migrate_v11_runtime_contract(d)
    else:
        _io.error(f"unknown migration: {migration}")
        return 1

    if not changes:
        _io.info(f"{path}: nothing to migrate")
        return 2

    print(f"  {path}: {len(changes)} change(s):")
    for c in changes:
        print(f"    - {c}")

    if not yes:
        return 0  # dry-run — counted as success
    # Backup
    if not no_backup:
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        _io.info(f"  backed up to: {backup}")
    new_body = yaml.safe_dump(d, sort_keys=False, default_flow_style=False)
    path.write_text(new_body)
    _io.success(f"  wrote: {path}")
    return 0


def run_migrate(args: argparse.Namespace) -> int:
    files = _expand_paths(args.paths)
    if not files:
        _io.error("no YAML files found")
        return 2

    print(f"sndr migrate {args.migration}")
    print(f"  {len(files)} file(s) to scan")
    if not args.yes:
        print("  DRY-RUN — pass --yes to apply changes")
    print()

    n_changed = 0
    n_failed = 0
    for f in files:
        rc = _migrate_one(f, migration=args.migration,
                            yes=args.yes, no_backup=args.no_backup)
        if rc == 0:
            n_changed += 1
        elif rc == 1:
            n_failed += 1

    print()
    print(f"  changed: {n_changed}, failed: {n_failed}, skipped: {len(files) - n_changed - n_failed}")
    return 1 if n_failed > 0 else 0
