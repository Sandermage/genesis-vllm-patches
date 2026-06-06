# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim: vllm.sndr_core.observability.

Canonical location: :mod:`sndr.observability`.

Will be removed in v13.0.
"""
from sndr.observability import *  # noqa: F401,F403
from sndr.observability import __all__  # noqa: F401
