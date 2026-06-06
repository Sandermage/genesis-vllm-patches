# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim.

Canonical location: ``sndr.engines.vllm.patches.serving.pn127_chat_template_qwen36``.

This file re-exports the entire public surface from the new location so
existing imports continue to work during v12.x migration window. Will be
removed in v13.0.
"""
from sndr.engines.vllm.patches.serving.pn127_chat_template_qwen36 import *  # noqa: F401,F403
try:
    from sndr.engines.vllm.patches.serving.pn127_chat_template_qwen36 import __all__  # noqa: F401
except ImportError:
    pass
