# SPDX-License-Identifier: Apache-2.0
"""SNDR Core dispatcher — pin validation (re-export from guards).

KNOWN_GOOD_VLLM_PINS allowlist + pin-validation helpers. Currently
re-exports from `vllm._genesis.guards` (Stage 4 will move impl here).

Used by:
  - apply orchestrator boot-time check (warn if running pin not in allowlist)
  - model_configs/audit_rules.py (validate `vllm_pin_required` field)
  - manifest_cache (verify manifest pin matches running vllm)
"""
from __future__ import annotations

from vllm.sndr_core.detection.guards import (  # noqa: F401
    KNOWN_GOOD_VLLM_PINS,
    is_genesis_pin_validated,
    assert_vllm_pin_allowed,
)

__all__ = [
    "KNOWN_GOOD_VLLM_PINS",
    "is_genesis_pin_validated",
    "assert_vllm_pin_allowed",
]
