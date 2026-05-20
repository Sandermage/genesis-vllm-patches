# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim — G4_31 relocated.

Real implementation: vllm.sndr_core.integrations.attention.turboquant.g4_31_preserve_tq_dtype
Shim window: one release. Remove this file after external imports migrate.
"""
from vllm.sndr_core.integrations.attention.turboquant.g4_31_preserve_tq_dtype import *  # noqa: F401,F403
