# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim — G4_69 relocated.

Real implementation: vllm.sndr_core.integrations.attention.turboquant.g4_69_skip_layers_native_backend
Shim window: one release. Remove this file after external imports migrate.
"""
from vllm.sndr_core.integrations.attention.turboquant.g4_69_skip_layers_native_backend import *  # noqa: F401,F403
