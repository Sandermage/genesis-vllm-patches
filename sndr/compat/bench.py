# SPDX-License-Identifier: Apache-2.0
"""Genesis bench — unified-CLI shim for `genesis_bench_suite.py`.

The real benchmark suite lives at `sndr/extras/tools/genesis_bench_suite.py`
(canonical post-v12) — it needs to ship as a single self-contained
script people can curl, run, and share without installing the full
Genesis package. This module is a thin pass-through that lets it be
reached via the unified CLI:

    python3 -m sndr.compat.cli bench --quick
    python3 -m sndr.compat.cli bench --mode standard --ctx 8k
    python3 -m sndr.compat.cli bench --compare a.json b.json

All argv after the `bench` subcommand is forwarded verbatim to
`genesis_bench_suite.main()`.

If the bench script can't be located (e.g. running from a slim
deployed package), the shim fails with a clear error pointing at
`docs/BENCHMARKS.md` rather than a confusing import traceback.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

log = logging.getLogger("genesis.compat.bench")


def _locate_bench_module() -> Path | None:
    """Find genesis_bench_suite.py on disk.

    Wave 10 (2026-05-15) made the package self-contained; the v12
    relocation moved the canonical home to
    `sndr/extras/tools/genesis_bench_suite.py` — no runtime imports
    from outside the package. Operator-side fallback locations stay in
    the search order for slim deployments / dev checkouts.

    Search order (first-hit wins):
      1. <package-internal>/tools/genesis_bench_suite.py — Wave 10 canonical
      2. $GENESIS_REPO_ROOT/tools/genesis_bench_suite.py — operator override
      3. <repo-root>/tools/genesis_bench_suite.py — operator-facing back-compat shim
      4. cwd-relative tools/genesis_bench_suite.py — last-resort
    """
    import os

    candidates: list[Path] = []
    # 1. Canonical sndr location: compat/bench.py → parents[0]=compat, [1]=sndr,
    #    [2]=repo-root after the relocation. The committed suite is at
    #    sndr/extras/tools/genesis_bench_suite.py.
    candidates.append(
        Path(__file__).resolve().parents[1] / "extras" / "tools" / "genesis_bench_suite.py"
    )
    # 2. Operator-side override (env var)
    env_root = os.environ.get("GENESIS_REPO_ROOT")
    if env_root:
        candidates.append(Path(env_root) / "tools" / "genesis_bench_suite.py")
    # 3. Operator-side back-compat shim path at repo root
    candidates.append(
        Path(__file__).resolve().parents[2] / "tools" / "genesis_bench_suite.py"
    )
    # 4. CWD fallback
    candidates.append(Path.cwd() / "tools" / "genesis_bench_suite.py")
    return next((p for p in candidates if p.is_file()), None)


def _load_bench_module():
    """Import genesis_bench_suite.py as a module.

    The script lives in sndr/extras/tools/ (canonical post-v12).
    It is intentionally not a regular package submodule because operators
    can also run it standalone via `python3 …/genesis_bench_suite.py …`,
    so we use an explicit `spec_from_file_location` loader.
    """
    bench_path = _locate_bench_module()
    if bench_path is None:
        raise FileNotFoundError(
            "Could not locate genesis_bench_suite.py. Canonical location "
            "is sndr/extras/tools/genesis_bench_suite.py. Set "
            "GENESIS_REPO_ROOT to point at a Genesis checkout, or run "
            "the bench directly per docs/BENCHMARKS.md."
        )
    spec = importlib.util.spec_from_file_location(
        "genesis_bench_suite", bench_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build module spec for {bench_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    """Forward argv to genesis_bench_suite.main()."""
    if argv is None:
        argv = sys.argv[1:]

    # `--help` / `-h` should work even if the bench script can't be
    # located, so handle it before attempting the import.
    if argv and argv[0] in ("-h", "--help"):
        # Try to surface the real help; fall back to a stub.
        try:
            mod = _load_bench_module()
            return mod.main(["--help"])
        except (FileNotFoundError, ImportError) as e:
            print("genesis bench — Genesis Benchmark Suite (shim)")
            print()
            print("The full benchmark suite lives at "
                  "sndr/extras/tools/genesis_bench_suite.py.")
            print()
            print(f"Could not load it from this deployment: {e}")
            print()
            print("See docs/BENCHMARKS.md for a manual invocation "
                  "recipe.")
            return 0

    try:
        mod = _load_bench_module()
    except (FileNotFoundError, ImportError) as e:
        print(f"[genesis bench] {e}", file=sys.stderr)
        return 3

    return mod.main(argv)


if __name__ == "__main__":
    sys.exit(main())
