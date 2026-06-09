#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Back-compat shim — genesis_bench_suite.py moved into sndr/.

History:
* Wave 10 (2026-05-15): moved from ``tools/`` into ``vllm/sndr_core/tools/``
  inside the legacy ``vllm.sndr_core`` namespace.
* Wave 12 (2026-06-01+): ``vllm/sndr_core/`` was removed from the wheel;
  the canonical file lives at ``sndr/extras/tools/genesis_bench_suite.py``.

This shim preserves the legacy operator UX so existing commands still work:
    python3 tools/genesis_bench_suite.py --quick
    bash some_old_script.sh                              # which calls the above

Search order (tried in sequence; first hit wins):

  1. ``sndr/extras/tools/genesis_bench_suite.py`` — Wave 12+ canonical
  2. ``vllm/sndr_core/tools/genesis_bench_suite.py`` — legacy bind-mount
  3. ``sndr_private/archive/v11_vllm_sndr_core_shims/tools/genesis_bench_suite.py``
     — archived v11 copy (only present on dev box)

Fix history: updated 2026-06-09 to add the Wave 12+ canonical path.

For new bench harness see ``tools/genesis_full_bench.py`` (richer
multi-metric report — runs decode-TPOT, sustained aggregate, stability
CV, quality regression and tool-call regression in one pass).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _find_real_script() -> Path | None:
    here = Path(__file__).resolve().parent  # repo/tools/
    repo = here.parent
    candidates = [
        # Wave 12+ canonical (2026-06-01+)
        repo / "sndr" / "extras" / "tools" / "genesis_bench_suite.py",
        # Wave 10 legacy (2026-05-15 ← removed in Wave 12)
        repo / "vllm" / "sndr_core" / "tools" / "genesis_bench_suite.py",
        # v11 archive (only on dev box, not committed to wheel)
        repo / "sndr_private" / "archive" / "v11_vllm_sndr_core_shims"
             / "tools" / "genesis_bench_suite.py",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def main() -> int:
    target = _find_real_script()
    if target is None:
        sys.stderr.write(
            "ERROR: canonical genesis_bench_suite.py not found at any of:\n"
            "  - sndr/extras/tools/genesis_bench_suite.py  (Wave 12+ canonical)\n"
            "  - vllm/sndr_core/tools/genesis_bench_suite.py  (Wave 10 legacy)\n"
            "Genesis Wave 12 (2026-06-01+) removed vllm/sndr_core/ from the\n"
            "wheel. See docs/BENCHMARK_GUIDE.md or use:\n"
            "  python3 tools/genesis_full_bench.py  (multi-metric harness)\n"
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
