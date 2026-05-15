#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wave 10 back-compat shim — genesis_bench_suite.py moved into sndr_core.

The canonical file is now at:
    vllm/sndr_core/tools/genesis_bench_suite.py

This shim preserves the legacy operator UX so existing commands still work:
    python3 tools/genesis_bench_suite.py --quick
    bash some_old_script.sh                              # which calls the above

It loads the real script from the new canonical location and forwards argv.
If the canonical file is missing (slim deployment), it falls back to a
helpful error pointing at docs/BENCHMARK_GUIDE.md.

Sander 2026-05-15.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _find_real_script() -> Path | None:
    here = Path(__file__).resolve().parent  # repo/tools/
    # repo/vllm/sndr_core/tools/genesis_bench_suite.py
    canonical = here.parent / "vllm" / "sndr_core" / "tools" / "genesis_bench_suite.py"
    if canonical.is_file():
        return canonical
    return None


def main() -> int:
    target = _find_real_script()
    if target is None:
        sys.stderr.write(
            "ERROR: canonical genesis_bench_suite.py not found at "
            "vllm/sndr_core/tools/genesis_bench_suite.py.\n"
            "Genesis Wave 10 (2026-05-15) moved this script INSIDE the\n"
            "sndr_core package. See docs/BENCHMARK_GUIDE.md for the new\n"
            "invocation, or use: python3 -m vllm.sndr_core.compat.cli bench …\n"
        )
        return 2

    spec = importlib.util.spec_from_file_location("genesis_bench_suite", target)
    if spec is None or spec.loader is None:
        sys.stderr.write(f"ERROR: could not build module spec for {target}\n")
        return 3
    module = importlib.util.module_from_spec(spec)
    sys.modules["genesis_bench_suite"] = module
    spec.loader.exec_module(module)

    main_fn = getattr(module, "main", None)
    if main_fn is None:
        sys.stderr.write(
            f"ERROR: {target} has no main() function — cannot forward argv.\n"
        )
        return 4
    return int(main_fn() or 0)


if __name__ == "__main__":
    sys.exit(main())
