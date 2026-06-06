# SPDX-License-Identifier: Apache-2.0
"""vLLM engine adapter package.

In Phase 1, this is a thin stub that delegates to the existing
vllm.sndr_core code via backward-compatibility shims. As we progress
through Phases 2-4, more functionality moves into this package directly.
"""
from sndr.engines.vllm.adapter import VllmEngine

__all__ = ["VllmEngine"]
