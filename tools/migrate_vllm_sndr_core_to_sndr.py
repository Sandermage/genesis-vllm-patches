#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bulk-migrate ``from vllm.sndr_core.X import Y`` → ``from sndr.X' import Y``.

Some submodules were RENAMED during the v12 refactor. This tool encodes the
mapping precisely so a blanket s/vllm\\.sndr_core/sndr/ doesn't silently
re-target paths to dead modules.

Run::

    python3 tools/migrate_vllm_sndr_core_to_sndr.py --dry-run sndr/ tools/
    python3 tools/migrate_vllm_sndr_core_to_sndr.py sndr/ tools/

Idempotent — files already migrated are skipped.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Mapping of ``vllm.sndr_core.<key>`` prefixes to the canonical sndr.* path.
# Order matters: longest-prefix match first. Keys are the segment AFTER
# ``vllm.sndr_core.`` (so ``apply`` matches ``vllm.sndr_core.apply.X`` too).
RENAMED_PREFIXES: list[tuple[str, str]] = [
    # Engine-specific subtrees were relocated under sndr.engines.vllm.
    ("integrations", "sndr.engines.vllm.patches"),
    ("kernels", "sndr.engines.vllm.kernels_legacy"),
    ("locations", "sndr.engines.vllm.locations"),
    ("middleware", "sndr.engines.vllm.middleware"),
    ("wiring", "sndr.engines.vllm.wiring"),
    # Pre-v11 ``paths`` was renamed to ``locations``.
    ("paths", "sndr.engines.vllm.locations"),
    # Engine-agnostic primitives moved to sndr.kernel.
    ("core", "sndr.kernel"),
    # Detection split: hardware-only stays in sndr.detection; engine-specific
    # bits moved to sndr.engines.vllm.detection. The mapping below is
    # CONSERVATIVE (engine path) since most legacy callers want
    # vllm-specific probes. Hand-fix the few hardware-only callers
    # afterwards if any layer-rule violations appear.
    ("detection", "sndr.engines.vllm.detection"),
    # The v11 CLI and product_api trees live under a ``legacy`` subpackage in
    # v12 (the bare ``sndr.cli`` / ``sndr.product_api`` roots host the NEW
    # command/route framework) — a blanket tail-map to ``sndr.<X>`` retargets
    # these to the wrong package.
    ("cli", "sndr.cli.legacy"),
    ("product_api", "sndr.product_api.legacy"),
]

# Tail (no prefix match): blanket ``vllm.sndr_core.<X>`` → ``sndr.<X>``.
# Most legacy submodules (apply, env, version, cli, license, dispatcher,
# model_configs, runtime, brand, caveats, observability, plugins, etc.)
# kept their name in the new tree.


def _migrate_text(text: str) -> tuple[str, int]:
    """Run the mapping over a file's text. Returns (new_text, num_changes)."""
    n_changes = 0

    # Patterns we touch:
    #   from vllm.sndr_core.<SEG>...
    #   import vllm.sndr_core.<SEG>...
    pattern = re.compile(
        r"(\b(?:from|import)\s+)vllm\.sndr_core(?:\.([a-zA-Z_][\w]*))?"
    )

    def _replace(m: re.Match) -> str:
        nonlocal n_changes
        n_changes += 1
        verb = m.group(1)
        seg = m.group(2)
        if seg is None:
            # Bare ``import vllm.sndr_core`` (no dot suffix) → ``import sndr``.
            return f"{verb}sndr"
        for prefix, target in RENAMED_PREFIXES:
            if seg == prefix:
                return f"{verb}{target}"
        return f"{verb}sndr.{seg}"

    new_text = pattern.sub(_replace, text)
    return new_text, n_changes


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("paths", nargs="+", type=Path,
                   help="Directories or files to scan")
    p.add_argument("--dry-run", action="store_true",
                   help="Report changes without writing")
    args = p.parse_args()

    targets: list[Path] = []
    for root in args.paths:
        if root.is_file() and root.suffix == ".py":
            targets.append(root)
        elif root.is_dir():
            for f in root.rglob("*.py"):
                if "__pycache__" in f.parts:
                    continue
                targets.append(f)

    total_changes = 0
    files_touched = 0
    for f in targets:
        try:
            text = f.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        if "vllm.sndr_core" not in text:
            continue
        new_text, n = _migrate_text(text)
        if n == 0 or new_text == text:
            continue
        total_changes += n
        files_touched += 1
        if args.dry_run:
            print(f"  would touch {f}: {n} edits")
        else:
            f.write_text(new_text)
            print(f"  ✓ {f}: {n} edits")

    print()
    print(f"Files {'would be ' if args.dry_run else ''}touched: {files_touched}")
    print(f"Total {'would-be ' if args.dry_run else ''}edits: {total_changes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
