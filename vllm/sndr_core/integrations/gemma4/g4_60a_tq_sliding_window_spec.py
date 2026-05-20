# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim — G4_60A relocated.

Real implementation: vllm.sndr_core.integrations.attention.turboquant.g4_60a_tq_sliding_window_spec
Shim window: one release. Remove this file after external imports migrate.
"""
from vllm.sndr_core.integrations.attention.turboquant.g4_60a_tq_sliding_window_spec import *  # noqa: F401,F403
