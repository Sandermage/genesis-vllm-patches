# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim — G4_62 relocated.

Real implementation: vllm.sndr_core.integrations.attention.turboquant.g4_62_tq_kernel_warmup
Shim window: one release. Remove this file after external imports migrate.
"""
from vllm.sndr_core.integrations.attention.turboquant.g4_62_tq_kernel_warmup import *  # noqa: F401,F403
