# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim: vllm.sndr_core.core.multi_file.

The canonical location is :mod:`sndr.kernel.multi_file`.
"""
from sndr.kernel.multi_file import *  # noqa: F401,F403
# Full back-compat: mirror the canonical module's entire namespace (names
# outside __all__ + private helpers) so legacy imports keep resolving.
import sndr.kernel.multi_file as _sndr_src  # noqa: E402
globals().update({_k: _v for _k, _v in vars(_sndr_src).items() if not _k.startswith("__")})
del _sndr_src

from sndr.kernel.multi_file import __all__  # noqa: F401
