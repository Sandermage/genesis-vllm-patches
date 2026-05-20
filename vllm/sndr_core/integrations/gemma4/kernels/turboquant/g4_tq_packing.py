# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim — g4_tq_packing relocated.

Real implementation: vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_packing
Shim window: one release. Remove this file after external imports migrate.
"""
from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_packing import *  # noqa: F401,F403
