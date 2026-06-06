# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim.

Canonical location: ``sndr.utils.gdn_composability``.

This file re-exports the entire public surface from the new location so
existing imports continue to work during v12.x migration window. Will be
removed in v13.0.
"""
from sndr.utils.gdn_composability import *  # noqa: F401,F403
try:
    from sndr.utils.gdn_composability import __all__  # noqa: F401
except ImportError:
    pass
