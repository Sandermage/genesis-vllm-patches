# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim.

Canonical location: ``sndr.engines.vllm.patches.attention.gdn.pn293_mamba_attn_prefill_fastpath``.

This file re-exports the entire public surface from the new location so
existing imports continue to work during v12.x migration window. Will be
removed in v13.0.
"""
from sndr.engines.vllm.patches.attention.gdn.pn293_mamba_attn_prefill_fastpath import *  # noqa: F401,F403
try:
    from sndr.engines.vllm.patches.attention.gdn.pn293_mamba_attn_prefill_fastpath import __all__  # noqa: F401
except ImportError:
    pass
