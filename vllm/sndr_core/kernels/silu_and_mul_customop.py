# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim.

Canonical location: ``sndr.engines.vllm.kernels_legacy.silu_and_mul_customop``.

This file re-exports the entire public surface from the new location so
existing imports continue to work during v12.x migration window. Will be
removed in v13.0.
"""
from sndr.engines.vllm.kernels_legacy.silu_and_mul_customop import *  # noqa: F401,F403
try:
    from sndr.engines.vllm.kernels_legacy.silu_and_mul_customop import __all__  # noqa: F401
except ImportError:
    pass
