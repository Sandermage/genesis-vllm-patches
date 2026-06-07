# SPDX-License-Identifier: Apache-2.0
"""sndr.kernel — engine-agnostic primitives.

This is Layer 0 of the layered architecture (see Master Spec Part 4).
Contents here:
  - **MUST NOT** import from vllm, sndr.engines, or any higher layer.
  - **MUST** be usable by every engine adapter without modification.
  - **MUST** have type hints and pass mypy --strict.

Public API:
    TextPatch, TextPatcher, TextPatchResult, TextPatchFailure
        Per-file text patching with marker-based idempotency, anchor matching,
        and upstream-drift detection.

    MultiFilePatchTransaction
        Atomic cross-file commits for patches that touch multiple files.

    result_to_wiring_status
        Maps internal TextPatchResult enum to the legacy ("applied", "skipped",
        "failed") string tuple expected by old wiring modules.

    Manifest helpers (from .manifest module):
        cached_load_manifest, derive_rel_path_from_target, md5_bytes

Migration note: this module was relocated from vllm/sndr_core/core/ in
Phase 2 of the sndr-platform refactor (2026-06-05). Old import paths
continue to work via the shim in vllm/sndr_core/core/__init__.py.
"""
from sndr.kernel.multi_file import MultiFilePatchTransaction
from sndr.kernel.text_patch import (
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
