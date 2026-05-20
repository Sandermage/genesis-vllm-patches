# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim — G4_60E relocated.

Real implementation: vllm.sndr_core.integrations.attention.turboquant.g4_60e_kv_cache_utils
Shim window: one release. Remove this file after external imports migrate.
"""
from vllm.sndr_core.integrations.attention.turboquant.g4_60e_kv_cache_utils import *  # noqa: F401,F403
