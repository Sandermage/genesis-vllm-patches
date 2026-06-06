# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim.

Canonical location: ``sndr.apply._state``.

This file re-exports the entire public surface from the new location so
existing imports continue to work during v12.x migration window. Will be
removed in v13.0.
"""
from sndr.apply._state import *  # noqa: F401,F403
try:
    from sndr.apply._state import __all__  # noqa: F401
except ImportError:
    pass
