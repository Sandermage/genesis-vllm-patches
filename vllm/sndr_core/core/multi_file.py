# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim: vllm.sndr_core.core.multi_file.

The canonical location is :mod:`sndr.kernel.multi_file`.
"""
from sndr.kernel.multi_file import *  # noqa: F401,F403
from sndr.kernel.multi_file import __all__  # noqa: F401
