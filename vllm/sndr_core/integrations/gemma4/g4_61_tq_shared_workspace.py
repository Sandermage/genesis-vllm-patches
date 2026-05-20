# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim — G4_61 relocated.

Real implementation: vllm.sndr_core.integrations.attention.turboquant.g4_61_tq_shared_workspace
Shim window: one release. Remove this file after external imports migrate.
"""
from vllm.sndr_core.integrations.attention.turboquant.g4_61_tq_shared_workspace import *  # noqa: F401,F403
