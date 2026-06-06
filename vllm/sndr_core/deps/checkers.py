# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim.

Canonical location: ``sndr.deps.checkers``.

This file re-exports the entire public surface from the new location so
existing imports continue to work during v12.x migration window. Will be
removed in v13.0.
"""
from sndr.deps.checkers import *  # noqa: F401,F403
try:
    from sndr.deps.checkers import __all__  # noqa: F401
except ImportError:
    pass

# `import *` skips underscore-prefixed names, but some consumers (the
# product_api container watcher + its tests) reference these private
# helpers through this shim path. Re-export them explicitly so the shim
# is a faithful stand-in until it is removed in v13.0.
from sndr.deps.checkers import _docker_socket_present  # noqa: F401
