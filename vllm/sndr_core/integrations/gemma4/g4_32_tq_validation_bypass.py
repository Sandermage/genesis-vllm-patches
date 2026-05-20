# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim — G4_32 relocated.

Real implementation: vllm.sndr_core.integrations.attention.turboquant.g4_32_tq_validation_bypass
Shim window: one release. Remove this file after external imports migrate.
"""
from vllm.sndr_core.integrations.attention.turboquant.g4_32_tq_validation_bypass import *  # noqa: F401,F403
