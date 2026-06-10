#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render Genesis canonical paths as a sourcable shell snippet.

Operator workflow (audit F-013 closure 2026-05-11 — "no hardcoded paths,
all via single settings file"):

  1. python3 scripts/emit_paths_env.py > ~/.genesis_paths.env
  2. In server start-scripts, BEFORE `docker run`:
       source ~/.genesis_paths.env
  3. Mount references in Docker use the same canonical values:
       -v "${GENESIS_MODELS_DIR}":/models:ro
       -v "${GENESIS_COMPILE_CACHE_DIR}":/root/.cache/vllm/torch_compile_cache
       -v "${GENESIS_TRITON_CACHE_DIR}":/root/.triton/cache
       -v "${GENESIS_HF_CACHE_DIR}":/root/.cache/huggingface

Single source of truth: edit env vars before running this script OR edit
defaults in `vllm/sndr_core/locations/project_paths.py`. Both Python
code (via the module's helpers) and bash scripts (via the rendered
env file) pick up identical values.

Modes:
  --emit-env       Render shell `export` snippet (default)
  --print          Pretty-print all paths (key = value)
  --prefix PFX     Use a different env prefix (default: GENESIS)

Examples:
  python3 scripts/emit_paths_env.py > ~/.genesis_paths.env
  python3 scripts/emit_paths_env.py --print           # human-readable
  python3 scripts/emit_paths_env.py --prefix SNDR     # SNDR_* prefix
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo's `vllm/` package importable regardless of where this
# script is invoked from (no pip install required to render paths).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--emit-env", action="store_true", default=True,
        help="Render shell `export` snippet (default)",
    )
    g.add_argument(
        "--print", dest="pretty", action="store_true",
        help="Pretty-print all paths (key = value)",
    )
    p.add_argument(
        "--prefix", default="GENESIS",
        help="Env var prefix (default: GENESIS; canonical alt: SNDR)",
    )
    args = p.parse_args()

    from sndr.engines.vllm.locations.project_paths import (
        all_paths, emit_env_shell,
    )

    if args.pretty:
        for k, v in all_paths().items():
            print(f"{k:30} = {v}")
        return

    sys.stdout.write(emit_env_shell(prefix=args.prefix))


if __name__ == "__main__":
    main()
