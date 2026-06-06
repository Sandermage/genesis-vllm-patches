# SPDX-License-Identifier: Apache-2.0
"""SNDR Core — wiring infrastructure (file cache, rebind, anchor manifest, patcher registry).

v10 (2026-05-07): canonical home for wiring-layer infrastructure that
all per-patch wirings depend on.

PR38 cleanup (2026-05-08): legacy `vllm._genesis.wiring` is being
removed; this package now re-exports the public surface (`AttributeRebinder`,
`WiringRegistry`, `TextPatch`, `TextPatcher`, `TextPatchResult`) at
package level so `from vllm.sndr_core.wiring import X` works the same
as the legacy form.
"""
from __future__ import annotations

from vllm.sndr_core.wiring.rebind import AttributeRebinder, WiringRegistry
from vllm.sndr_core.core import text_patch  # noqa: F401  (legacy submodule path)
from vllm.sndr_core.core.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

__all__ = [
    "AttributeRebinder",
    "WiringRegistry",
    "TextPatch",
    "TextPatcher",
    "TextPatchResult",
    "text_patch",
]
