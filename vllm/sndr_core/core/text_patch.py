# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim: vllm.sndr_core.core.text_patch.

The canonical location is :mod:`sndr.kernel.text_patch`.

This shim re-exports the entire public surface so existing imports continue
to work::

    # OLD path (deprecated, removed in v13):
    from vllm.sndr_core.core.text_patch import TextPatch

    # NEW path (canonical):
    from sndr.kernel.text_patch import TextPatch
    # or
    from sndr.kernel import TextPatch

Module-level state (caches, etc.) lives in the canonical module — there is
only ONE copy, regardless of which import path is used.
"""
# Re-export EVERYTHING. We use ``from X import *`` followed by ``__all__``
# from the upstream module so that any new exports added in sndr.kernel.text_patch
# automatically flow through.
from sndr.kernel.text_patch import *  # noqa: F401,F403
# Full back-compat: mirror the canonical module's entire namespace (names
# outside __all__ + private helpers) so legacy imports keep resolving.
import sndr.kernel.text_patch as _sndr_src  # noqa: E402
globals().update({_k: _v for _k, _v in vars(_sndr_src).items() if not _k.startswith("__")})
del _sndr_src

from sndr.kernel.text_patch import __all__  # noqa: F401
