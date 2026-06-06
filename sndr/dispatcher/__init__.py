# SPDX-License-Identifier: Apache-2.0
"""SNDR Core dispatcher — public API.

Composed of 5 modules (split from vllm/_genesis/dispatcher.py 2828-LOC
monolith at Stage 3):

  - registry.py  — PATCH_REGISTRY data (2000+ LOC, data-only exempt cap)
  - decision.py  — should_apply, _check_applies_to, log_decision
  - audit.py     — ValidationIssue, validate_registry, validate_apply_plan
  - reporting.py — apply_matrix, structured_boot_summary
  - pins.py      — KNOWN_GOOD_VLLM_PINS (re-exports from guards.py)

Public API used by patch wirings + CLI.
"""
from __future__ import annotations

import logging

from .audit import (  # noqa: F401
    ValidationIssue,
    log_validation_issues,
    validate_apply_plan,
    validate_registry,
)
from .decision import (  # noqa: F401
    log_decision,
    should_apply,
    # Module-private list reached into by some test fixtures (boot
    # summary, decision-replay tests). Re-exported at package level so
    # tests can do `dispatcher._DECISIONS` without knowing the split
    # module layout.
    _DECISIONS,
)
from .pins import (  # noqa: F401
    KNOWN_GOOD_VLLM_PINS,
    assert_vllm_pin_allowed,
    is_genesis_pin_validated,
)
from .registry import PATCH_REGISTRY  # noqa: F401
from .spec import (  # noqa: F401
    PatchSpec,
    iter_patch_specs,
    patch_spec_for,
    validate_apply_module_coverage,
)
from .reporting import (  # noqa: F401
    dump_apply_matrix,
    dump_structured_boot_summary,
    get_apply_matrix,
    log_apply_matrix,
    log_structured_boot_summary,
)

log = logging.getLogger("genesis.dispatcher")

__all__ = [
    # registry
    "PATCH_REGISTRY",
    # decision
    "should_apply",
    "log_decision",
    # audit
    "ValidationIssue",
    "validate_registry",
    "validate_apply_plan",
    "log_validation_issues",
    # reporting
    "get_apply_matrix",
    "dump_apply_matrix",
    "log_apply_matrix",
    "dump_structured_boot_summary",
    "log_structured_boot_summary",
    # pins
    "KNOWN_GOOD_VLLM_PINS",
    "is_genesis_pin_validated",
    "assert_vllm_pin_allowed",
]


def main() -> int:
    """CLI entrypoint — dump apply matrix as ASCII table."""
    print(dump_apply_matrix())
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
