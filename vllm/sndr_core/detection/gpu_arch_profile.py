# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim: vllm.sndr_core.detection.gpu_arch_profile.

Canonical location: :mod:`sndr.detection.gpu_arch_profile` (Layer 0 —
engine-agnostic hardware detection).

This shim re-exports the public API so existing imports continue to work
during the v12.x migration window. Will be removed in v13.0.
"""
from sndr.detection.gpu_arch_profile import *  # noqa: F401,F403
from sndr.detection.gpu_arch_profile import __all__  # noqa: F401
