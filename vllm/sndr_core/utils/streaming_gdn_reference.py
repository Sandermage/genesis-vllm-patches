# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim.

Canonical location: ``sndr.utils.streaming_gdn_reference``.

This file re-exports the entire public surface from the new location so
existing imports continue to work during v12.x migration window. Will be
removed in v13.0.
"""
from sndr.utils.streaming_gdn_reference import *  # noqa: F401,F403
try:
    from sndr.utils.streaming_gdn_reference import __all__  # noqa: F401
except ImportError:
    pass
