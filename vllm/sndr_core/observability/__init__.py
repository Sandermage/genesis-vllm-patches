# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim: vllm.sndr_core.observability.

Canonical location: :mod:`sndr.observability`.

Will be removed in v13.0.
"""
from sndr.observability import *  # noqa: F401,F403
# Full back-compat: mirror the canonical module's entire namespace (names
# outside __all__ + private helpers) so legacy imports keep resolving.
import sndr.observability as _sndr_src  # noqa: E402
globals().update({_k: _v for _k, _v in vars(_sndr_src).items() if not _k.startswith("__")})
del _sndr_src

from sndr.observability import __all__  # noqa: F401
