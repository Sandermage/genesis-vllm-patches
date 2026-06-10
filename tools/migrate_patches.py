#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 4 migration tool: move patches from integrations/ to engines/vllm/patches/.

Operations (per family, atomic):

  1. Move ``vllm/sndr_core/integrations/<family>/*.py`` to
     ``sndr/engines/vllm/patches/<family>/*.py`` (preserving subdir structure).
  2. Update ``sndr/dispatcher/registry.py`` apply_module paths from
     ``vllm.sndr_core.integrations.X`` → ``sndr.engines.vllm.patches.X``.
  3. Create per-file backward compat shims at the old location.
  4. Run apply matrix to verify no regressions.

Usage::

    # Dry-run: show what would be moved
    python3 tools/migrate_patches.py --family attention --dry-run

    # Apply migration for one family
    python3 tools/migrate_patches.py --family attention --apply

    # Migrate all families
    python3 tools/migrate_patches.py --all --apply

Recommendation: migrate families one at a time, with apply matrix
verification between each. The order recommended:

  1. observability/ (3 files, low risk, no inter-patch deps)
  2. compile_safety/ (small)
  3. memory/ (small)
  4. lora/, multimodal/ (model-class-scoped)
  5. tool_parsing/, reasoning/ (model-class-scoped)
  6. quantization/, kv_cache/ (moderate)
  7. moe/, kernels/ (more deps)
  8. attention/ (largest, most interdependent — LAST)
  9. spec_decode/ (largest, after attention)

This script is INTENTIONALLY non-destructive in dry-run mode. Always
run dry-run first; review the output; then apply.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SRC_BASE = REPO_ROOT / "sndr" / "engines" / "vllm" / "patches"
DST_BASE = REPO_ROOT / "sndr" / "engines" / "vllm" / "patches"

FAMILIES_RECOMMENDED_ORDER = [
    "observability",
    "compile_safety",
    "memory",
    "loader",
    "lora",
    "multimodal",
    "tool_parsing",
    "reasoning",
    "scheduler",
    "serving",
    "worker",
    "streaming",
    "offload",
    "model_compat",
    "gemma4",
    "quantization",
    "kv_cache",
    "moe",
    "kernels",
    "detection",
    "attention",
    "spec_decode",
]


def list_family_files(family: str) -> list[Path]:
    """List all .py files under integrations/<family>/."""
    src = SRC_BASE / family
    if not src.is_dir():
        return []
    return sorted(p for p in src.rglob("*.py") if "__pycache__" not in p.parts)


def make_shim(original_module_path: str) -> str:
    """Generate the body of a backward-compat shim file."""
    new_path = original_module_path.replace(
        "vllm.sndr_core.integrations.",
        "sndr.engines.vllm.patches.",
        1,
    )
    return f'''# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim.

Canonical location: ``{new_path}``.

This file re-exports the entire public surface from the new location so
existing imports continue to work during v12.x migration window. Will be
removed in v13.0.
"""
from {new_path} import *  # noqa: F401,F403
try:
    from {new_path} import __all__  # noqa: F401
except ImportError:
    pass
'''


def migrate_family(family: str, *, dry_run: bool = True) -> tuple[int, list[str]]:
    """Migrate one family. Returns (file_count, log_lines)."""
    src_root = SRC_BASE / family
    dst_root = DST_BASE / family
    log: list[str] = []

    if not src_root.is_dir():
        log.append(f"[SKIP] {family}: source directory does not exist")
        return 0, log

    files = list_family_files(family)
    log.append(f"[{family}] found {len(files)} .py files to migrate")

    for src_file in files:
        rel = src_file.relative_to(src_root)
        dst_file = dst_root / rel
        log.append(f"  MOVE {rel}")

        if not dry_run:
            # Create destination parent dir
            dst_file.parent.mkdir(parents=True, exist_ok=True)

            # Move the actual file (shutil.move preserves content)
            shutil.move(str(src_file), str(dst_file))

            # Compute the module path for the shim
            rel_no_ext = rel.with_suffix("")
            module_parts = list(rel_no_ext.parts)
            original_module = (
                f"vllm.sndr_core.integrations.{family}." + ".".join(module_parts)
            )

            # Write the shim at the OLD location
            shim_content = make_shim(original_module)
            src_file.parent.mkdir(parents=True, exist_ok=True)
            src_file.write_text(shim_content)

    return len(files), log


def update_registry(dry_run: bool = True) -> list[str]:
    """Update apply_module paths in registry.py.

    NOTE: For v12, registry lives at sndr/dispatcher/registry.py (Phase 5).
    For now, we patch vllm/sndr_core/dispatcher/registry.py directly.
    """
    registry = REPO_ROOT / "sndr" / "dispatcher" / "registry.py"
    if not registry.is_file():
        return ["[WARN] registry.py not found; skip update"]

    content = registry.read_text()
    new_content = content.replace(
        '"vllm.sndr_core.integrations.',
        '"sndr.engines.vllm.patches.',
    )

    log = []
    delta = content.count('"vllm.sndr_core.integrations.') - new_content.count('"vllm.sndr_core.integrations.')
    log.append(f"[registry] would update {delta} apply_module references")
    if not dry_run:
        registry.write_text(new_content)
        log.append(f"[registry] updated.")
    return log


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--family", help="Migrate a single family (e.g. attention)")
    p.add_argument("--all", action="store_true", help="Migrate all families in recommended order")
    p.add_argument("--apply", action="store_true", help="Actually perform the migration (default: dry run)")
    p.add_argument("--update-registry", action="store_true", help="Also update apply_module paths in registry.py")
    args = p.parse_args()

    if not args.family and not args.all:
        p.error("Must specify --family X or --all")

    dry_run = not args.apply
    if dry_run:
        print("=== DRY RUN — no files will be modified ===\n")

    families = (
        FAMILIES_RECOMMENDED_ORDER if args.all
        else [args.family] if args.family else []
    )

    total = 0
    for fam in families:
        n, lines = migrate_family(fam, dry_run=dry_run)
        total += n
        for line in lines:
            print(line)
        print()

    if args.update_registry:
        print("=== Registry update ===")
        for line in update_registry(dry_run=dry_run):
            print(line)
        print()

    print(f"=== TOTAL: {total} files {'would be' if dry_run else 'were'} moved ===")
    if dry_run:
        print("Re-run with --apply to perform the migration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
