# SPDX-License-Identifier: Apache-2.0
"""Pure-data query layer for ``sndr patches doctor`` (M.6.1)."""
from __future__ import annotations

from .types import DoctorReport


def run_doctor() -> DoctorReport:
    """Run the registry validator + apply_module coverage probe.

    Returns a frozen snapshot containing the size of ``PATCH_REGISTRY``,
    the tuple of validation issues (severity / patch_id / message), and
    the coverage report (``mapped``, ``unmapped``, ``intentionally_unmapped``).
    """
    from vllm.sndr_core.dispatcher import PATCH_REGISTRY, validate_registry
    from vllm.sndr_core.dispatcher.spec import validate_apply_module_coverage

    issues = tuple(validate_registry())
    coverage = validate_apply_module_coverage()
    return DoctorReport(
        registry_size=len(PATCH_REGISTRY),
        issues=issues,
        coverage=coverage,
    )
