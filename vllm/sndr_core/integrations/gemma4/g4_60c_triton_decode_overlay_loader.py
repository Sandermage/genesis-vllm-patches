# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim — G4_60C relocated.

Real implementation: vllm.sndr_core.integrations.attention.turboquant.g4_60c_triton_decode_overlay_loader
Shim window: one release. Remove this file after external imports migrate.
"""
from vllm.sndr_core.integrations.attention.turboquant.g4_60c_triton_decode_overlay_loader import *  # noqa: F401,F403
