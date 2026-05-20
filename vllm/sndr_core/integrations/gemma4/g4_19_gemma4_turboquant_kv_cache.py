# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim — G4_19 relocated.

Real implementation: vllm.sndr_core.integrations.attention.turboquant.g4_19_turboquant_kv_cache
Shim window: one release. Remove this file after external imports migrate.
"""
from vllm.sndr_core.integrations.attention.turboquant.g4_19_turboquant_kv_cache import *  # noqa: F401,F403
