# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim: vllm.sndr_core.detection.guards.

Canonical location: :mod:`sndr.engines.vllm.detection.guards` (Layer 1 —
vllm-specific detection).

Will be removed in v13.0.
"""
from sndr.engines.vllm.detection.guards import *  # noqa: F401,F403
try:
    from sndr.engines.vllm.detection.guards import __all__  # noqa: F401
except ImportError:
    pass
