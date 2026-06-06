#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generic module migration tool — moves any subtree from legacy to sndr/.

This is the workhorse for Phase 4-11 migration. Unlike migrate_patches.py
(which knows about family layout), this tool takes arbitrary source and
destination paths.

Operations per file:
  1. Move legacy file to new location (creating dirs as needed).
  2. At the legacy location, write a backward-compat shim re-exporting
     from the new canonical location.
  3. For ``__init__.py`` files that use ``__getattr__`` (lazy-load),
     replicate the real content at both locations instead of shimming.

Usage::

    # Dry run
    python3 tools/migrate_module.py \\
        --src vllm/sndr_core/dispatcher \\
        --dst sndr/dispatcher \\
        --legacy-prefix vllm.sndr_core.dispatcher \\
        --new-prefix sndr.dispatcher

    # Apply
    python3 tools/migrate_module.py \\
        --src vllm/sndr_core/dispatcher \\
        --dst sndr/dispatcher \\
        --legacy-prefix vllm.sndr_core.dispatcher \\
        --new-prefix sndr.dispatcher \\
        --apply
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def has_lazy_getattr(file_path: Path) -> bool:
    """Return True if a file declares ``def __getattr__(...)`` at module level."""
    try:
        content = file_path.read_text()
    except (OSError, UnicodeDecodeError):
        return False
    return "def __getattr__(" in content


def shim_content(new_module_path: str) -> str:
    return f'''# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim.

Canonical location: ``{new_module_path}``.

This file re-exports the entire public surface from the new location so
existing imports continue to work during v12.x migration window. Will be
removed in v13.0.
"""
from {new_module_path} import *  # noqa: F401,F403
try:
    from {new_module_path} import __all__  # noqa: F401
except ImportError:
    pass
'''


def migrate_file(
    src: Path,
    dst: Path,
    new_module: str,
    *,
    dry_run: bool,
) -> str:
    """Migrate one file. Returns a log line describing the action."""
    if src.name == "__init__.py" and has_lazy_getattr(src):
        # Replicate (don't shim) — the shim would break lazy __getattr__ chains.
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(src), str(dst))
            # Leave the original at src — it works as the legacy entry point.
        return f"  REPLICATE {src.name} (lazy __getattr__)"

    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text(shim_content(new_module))
    return f"  MOVE {src.relative_to(REPO_ROOT)}"


def migrate_tree(
    src_root: Path,
    dst_root: Path,
    legacy_prefix: str,
    new_prefix: str,
    *,
    dry_run: bool,
) -> tuple[int, list[str]]:
    """Migrate every .py file in src_root to dst_root."""
    log: list[str] = []
    count = 0

    if not src_root.is_dir():
        log.append(f"[SKIP] source dir does not exist: {src_root}")
        return 0, log

    for src_file in sorted(src_root.rglob("*.py")):
        if "__pycache__" in src_file.parts:
            continue

        rel = src_file.relative_to(src_root)
        dst_file = dst_root / rel

        # Build canonical module path for the destination.
        rel_no_ext = rel.with_suffix("")
        parts = list(rel_no_ext.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        new_module = new_prefix
        if parts:
            new_module = f"{new_prefix}." + ".".join(parts)

        log.append(migrate_file(src_file, dst_file, new_module, dry_run=dry_run))
        count += 1

    return count, log


def migrate_single_file(
    src: Path,
    dst: Path,
    new_module: str,
    *,
    dry_run: bool,
) -> str:
    """Migrate a single .py file (not a tree)."""
    if not src.is_file():
        return f"[SKIP] {src} does not exist"
    return migrate_file(src, dst, new_module, dry_run=dry_run)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src", type=Path, required=True, help="Source path (file or directory)")
    p.add_argument("--dst", type=Path, required=True, help="Destination path")
    p.add_argument("--legacy-prefix", required=True, help="Legacy import prefix")
    p.add_argument("--new-prefix", required=True, help="New canonical import prefix")
    p.add_argument("--apply", action="store_true", help="Apply changes (default: dry run)")
    args = p.parse_args()

    dry_run = not args.apply
    if dry_run:
        print("=== DRY RUN ===\n")

    src = (REPO_ROOT / args.src).resolve()
    # Treat `--dst` literally — operator must give a path with separators
    # (slashes), not dotted module-style names. Earlier versions silently
    # accepted "sndr.X.Y" which produced literal directories with dots.
    dst_str = str(args.dst)
    if "." in dst_str.replace(".py", "") and "/" not in dst_str:
        print(
            f"ERROR: --dst looks dotted ({args.dst!r}); use slashes "
            "(e.g. sndr/engines/vllm/X). The tool refuses dotted paths to "
            "prevent literal-dot directory names.",
            file=sys.stderr,
        )
        return 2
    dst = (REPO_ROOT / args.dst).resolve()

    if src.is_dir():
        n, log = migrate_tree(src, dst, args.legacy_prefix, args.new_prefix, dry_run=dry_run)
    elif src.is_file():
        line = migrate_single_file(src, dst, args.new_prefix, dry_run=dry_run)
        log = [line]
        n = 1 if "MOVE" in line or "REPLICATE" in line else 0
    else:
        print(f"ERROR: source does not exist: {src}", file=sys.stderr)
        return 2

    for line in log:
        print(line)

    print(f"\nTotal: {n} files {'would be' if dry_run else 'were'} processed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
