# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim: vllm.sndr_core.core.

The canonical location of these primitives is :mod:`sndr.kernel`.

This shim re-exports the public API so that all existing imports continue
to work during the v12.x migration window::

    # OLD (still works, but deprecated):
    from vllm.sndr_core.core import TextPatcher, TextPatch

    # NEW (preferred):
    from sndr.kernel import TextPatcher, TextPatch

Will be removed in v13.0.
"""
from sndr.kernel import (  # noqa: F401
    MultiFilePatchTransaction,
    TextPatch,
    TextPatchFailure,
    TextPatchResult,
    TextPatcher,
    marker_present_in_target,
    result_to_wiring_status,
)

__all__ = [
    "MultiFilePatchTransaction",
    "TextPatch",
    "TextPatchFailure",
    "TextPatchResult",
    "TextPatcher",
    "marker_present_in_target",
    "result_to_wiring_status",
]
