# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim.

Canonical location: ``sndr.engines.vllm.patches.serving.pn91_developer_role_normalizer``.

This file re-exports the entire public surface from the new location so
existing imports continue to work during v12.x migration window. Will be
removed in v13.0.
"""
from sndr.engines.vllm.patches.serving.pn91_developer_role_normalizer import *  # noqa: F401,F403
# Full back-compat: mirror the canonical module's entire namespace (names
# outside __all__ + private helpers) so legacy imports keep resolving.
import sndr.engines.vllm.patches.serving.pn91_developer_role_normalizer as _sndr_src  # noqa: E402
globals().update({_k: _v for _k, _v in vars(_sndr_src).items() if not _k.startswith("__")})
del _sndr_src

try:
    from sndr.engines.vllm.patches.serving.pn91_developer_role_normalizer import __all__  # noqa: F401
except ImportError:
    pass
