# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim: vllm.sndr_core.core.manifest_cache.

The canonical location is :mod:`sndr.kernel.manifest` (note the rename:
``manifest_cache`` → ``manifest`` in the new structure).
"""
from sndr.kernel.manifest import *  # noqa: F401,F403
from sndr.kernel.manifest import __all__  # noqa: F401
