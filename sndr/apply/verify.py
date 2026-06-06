# SPDX-License-Identifier: Apache-2.0
"""SNDR Core apply — verify_live_rebinds() post-register check.

Verifies that all monkey-patched runtime attributes are actually live
in the current process. Used by `apply/orchestrator.run()` and CLI
`sndr verify` for post-apply sanity check.

Migration history:
  - Original location: vllm/_genesis/patches/apply_all.py (Stage 0).
  - Stage 3 (CURRENT): extracted into apply/verify.py.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("genesis.apply_all.verify")


def verify_live_rebinds() -> dict[str, Any]:
    """Post-register verification: confirm runtime rebinds are actually live
    in the current process (TDD discipline from master plan Part 3).

    Returns a dict:
      {
        "P22": {"expected": True, "actual": True, "ok": True},
        "P31": {"expected": True, "actual": True, "ok": True},
        "P14": {"expected": True, "actual": True, "ok": True},
        ...
      }

    Only patches with Python-attribute rebinds are checked. Text-patches
    (P3, P4, P5, P6, P8, P15) modify source files and are verified by the
    diagnostic probes in validate_integration.sh (grep file for markers).

    Usage (end-of-register hook or test):
      from vllm.sndr_core.apply import verify_live_rebinds
      results = verify_live_rebinds()
      for name, r in results.items():
          if not r["ok"]:
              log.warning("[Genesis] rebind %s not live: expected=%s actual=%s",
                          name, r["expected"], r["actual"])
    """
    results: dict[str, dict] = {}

    def _check(patch_id: str):
        """Resolve patch_id via `compat.categories.module_for`, import the
        wiring module, and invoke `is_applied()`."""
        from vllm.sndr_core.compat.categories import module_for
        dotted = module_for(patch_id)
        if dotted is None:
            results[patch_id] = {
                "expected": True, "actual": False, "ok": False,
                "error": f"no wiring module resolved for {patch_id}",
            }
            return
        try:
            import importlib
            mod = importlib.import_module(dotted)
        except Exception as e:
            results[patch_id] = {
                "expected": True, "actual": False, "ok": False,
                "error": f"import {dotted} failed: {e}",
            }
            return
        is_applied_fn = getattr(mod, "is_applied", None)
        if is_applied_fn is None or not callable(is_applied_fn):
            results[patch_id] = {
                "expected": True, "actual": None, "ok": True,
                "note": "module has no is_applied() — skipped",
            }
            return
        try:
            actual = bool(is_applied_fn())
        except Exception as e:
            results[patch_id] = {
                "expected": True, "actual": False, "ok": False,
                "error": f"is_applied() raised: {e}",
            }
            return
        results[patch_id] = {
            "expected": True, "actual": actual, "ok": actual,
        }

    # Runtime rebinds (set attrs on live vLLM classes/modules).
    # Each ID is resolved via `compat.categories.module_for` to its
    # canonical `vllm.sndr_core.integrations.*` location.
    _check("P22")
    _check("P31")
    _check("P14")
    _check("P28")
    # v7.2 / v7.3 additions — both have symmetric `apply/is_applied/revert`
    # trios per their wiring surface contracts.
    _check("P38")
    _check("P39a")

    return results


