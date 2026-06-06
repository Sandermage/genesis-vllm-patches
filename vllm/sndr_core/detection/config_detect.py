# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim: vllm.sndr_core.detection.config_detect.

Canonical location: :mod:`sndr.engines.vllm.detection.config_detect` (Layer 1 —
vllm-specific detection).

Will be removed in v13.0.
"""
from sndr.engines.vllm.detection.config_detect import *  # noqa: F401,F403
try:
    from sndr.engines.vllm.detection.config_detect import __all__  # noqa: F401
except ImportError:
    pass
