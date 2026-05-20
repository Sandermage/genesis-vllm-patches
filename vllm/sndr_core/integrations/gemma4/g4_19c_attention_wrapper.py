# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim — G4_19C relocated.

Real implementation: vllm.sndr_core.integrations.attention.turboquant.g4_19c_attention_wrapper
Shim window: one release. Remove this file after external imports migrate.
"""
from vllm.sndr_core.integrations.attention.turboquant.g4_19c_attention_wrapper import *  # noqa: F401,F403
