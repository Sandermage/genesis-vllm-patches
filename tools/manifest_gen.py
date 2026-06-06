#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate per-pin manifest for an engine.

A manifest captures the state of every upstream file that our patches care
about, at a specific pin. The manifest is then committed to the repo
under ``sndr/engines/<engine>/pins/<pin>/manifest.yaml`` and used by the
drift detection tool to identify when upstream has changed.

Usage::

    # Generate a manifest for the currently-installed vllm
    python3 tools/manifest_gen.py --engine vllm --auto-pin

    # Generate for a specific pin (must match a docker tag)
    python3 tools/manifest_gen.py --engine vllm --pin 0.22.1_da1daf40b

The tool extracts:
  - For each file referenced by a registered patch:
      - File's full content md5
      - The anchor snippet (text the patch finds and modifies)
      - The anchor snippet's md5
      - Line range
  - Engine version, upstream commit SHA, timestamp

This is Phase 7 of the sndr-platform refactor. See Master Spec Part 5.4.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def compute_file_md5(path: Path) -> str:
    """Compute md5 of a file's content."""
    h = hashlib.md5()  # noqa: S324 — md5 used as content fingerprint, not security
    h.update(path.read_bytes())
    return h.hexdigest()


def generate_manifest(
    engine: str,
    pin: str,
    install_root: Path,
    file_paths: list[str],
) -> dict:
    """Build a manifest dict for the given pin.

    Args:
        engine: Engine identifier (e.g. "vllm").
        pin: Normalized pin identifier (e.g. "0.22.1_da1daf40b").
        install_root: Filesystem root of the engine package.
        file_paths: Relative paths (under install_root) to track.

    Returns:
        Manifest dictionary suitable for yaml.safe_dump.
    """
    files = {}
    for rel in file_paths:
        abs_path = install_root / rel
        if not abs_path.is_file():
            files[rel] = {"missing": True}
            continue
        files[rel] = {
            "md5": compute_file_md5(abs_path),
            "size_bytes": abs_path.stat().st_size,
            # TODO Phase 7+: extract anchor snippets per patch and compute their md5s.
            "anchors": {},
        }

    return {
        "engine": engine,
        "pin": pin,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "sndr.tools.manifest_gen v1.0",
        "files": files,
    }


def collect_patch_targets(engine: str) -> list[str]:
    """Collect file paths referenced by registered patches.

    Phase 7 stub: returns an empty list. Future iterations will:
      1. Walk sndr/engines/<engine>/patches/ for all patch modules
      2. Import each patch's _make_patcher() (or similar) to discover targets
      3. Deduplicate and return sorted

    For now we accept --file paths on the CLI.
    """
    return []


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--engine", default="vllm", help="Engine name")
    p.add_argument("--pin", required=True, help="Pin identifier (e.g. 0.22.1_da1daf40b)")
    p.add_argument(
        "--install-root",
        help="Path to engine install (default: import the package)",
    )
    p.add_argument(
        "--file",
        action="append",
        default=[],
        help="Track this file (relative to install root). Repeat for multiple files.",
    )
    p.add_argument(
        "--output",
        type=Path,
        help="Output path (default: sndr/engines/<engine>/pins/<pin>/manifest.yaml)",
    )
    args = p.parse_args()

    if args.install_root:
        install_root = Path(args.install_root)
    else:
        if args.engine == "vllm":
            try:
                import vllm  # type: ignore
                install_root = Path(vllm.__file__).parent
            except ImportError:
                print("ERROR: vllm not installed; use --install-root", file=sys.stderr)
                return 1
        else:
            print(f"ERROR: --install-root required for engine '{args.engine}'", file=sys.stderr)
            return 1

    file_paths = list(args.file)
    if not file_paths:
        # Try to collect from registered patches.
        file_paths = collect_patch_targets(args.engine)
    if not file_paths:
        print("WARN: no file paths to track. Use --file PATH (repeatable).", file=sys.stderr)

    manifest = generate_manifest(args.engine, args.pin, install_root, file_paths)

    # Default output path
    repo_root = Path(__file__).parent.parent
    output = args.output or (
        repo_root / "sndr" / "engines" / args.engine / "pins" / args.pin / "manifest.yaml"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(manifest, sort_keys=False))

    print(f"Wrote manifest to {output}")
    print(f"  files: {len(manifest['files'])}")
    print(f"  engine: {args.engine}")
    print(f"  pin: {args.pin}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
