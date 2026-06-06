# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim: vllm.sndr_core.detection.driver_check.

Canonical location: :mod:`sndr.engines.vllm.detection.driver_check` (Layer 1 —
vllm-specific detection).

Will be removed in v13.0.
"""
from sndr.engines.vllm.detection.driver_check import *  # noqa: F401,F403
try:
    from sndr.engines.vllm.detection.driver_check import __all__  # noqa: F401
except ImportError:
    pass
