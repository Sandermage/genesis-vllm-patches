# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim — g4_tq_write_triton relocated.

Real implementation: vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_write_triton
Shim window: one release. Remove this file after external imports migrate.
"""
from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_write_triton import *  # noqa: F401,F403
