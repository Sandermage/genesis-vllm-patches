# SPDX-License-Identifier: Apache-2.0
"""Patcher registry — opt-in API for wiring modules to register their
TextPatcher objects for inclusion in the Site Map anchor manifest.

Part of P2.1 (Node 2 of design doc, 2026-05-07). The flat
`PATCH_REGISTRY: list[tuple[str, Callable]]` in apply_all.py records
the apply ENTRY POINTS but does NOT expose the underlying TextPatcher
objects (each entry point creates patchers dynamically inside its
function body via `_make_*_patcher()` factories).

For the anchor manifest builder, we need to walk the actual TextPatcher
instances. Rather than refactor 100+ wiring modules, we use a simple
opt-in registry: patches that want to be cached by the Site Map call
`register_text_patcher()` after constructing their patcher.

Co-existence semantics:

  - Patches that DON'T register stay on the legacy O(N×M) full-scan path
    forever. No regression for them.
  - Patches that DO register get O(1) anchor lookup once Phase 3 runtime
    path lands. MVP only collects entries; runtime usage is later.
  - The apply_all `register_patch` decorator and this registry are
    independent — registration here doesn't change apply order or
    registry membership.

Thread safety: process-global dict + mutex. Genesis itself is not
multi-threaded (vllm worker spawn = process fork), but Python's
import system can race in edge cases (concurrent test collection,
plugin loaders). Lock is paranoid but cheap.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:
    from sndr.kernel.text_patch import TextPatcher

log = logging.getLogger("genesis.wiring.patcher_registry")


# Process-global registry. patch_id -> TextPatcher.
# patch_id is the canonical Genesis patch identifier (e.g. "PN79.Sub-1",
# "PN79.Sub-2"). Use sub-ids when one logical patch maps to multiple
# files via MultiFilePatchTransaction — each TextPatcher gets its own
# entry so manifest builder can pair patcher with file.
_REGISTERED: dict[str, "TextPatcher"] = {}
_LOCK = threading.Lock()


def register_text_patcher(patch_id: str, patcher: "TextPatcher") -> None:
    """Register a TextPatcher for inclusion in the Site Map manifest.

    Args:
      patch_id: canonical identifier — convention `<PATCH_ID>.<SubID>`.
        Examples: "PN79.Sub-1", "PN79.Sub-2", "PN79.Sub-3", "PN79.Sub-4".
      patcher: the TextPatcher instance.

    Raises ValueError on:
      - duplicate patch_id (with different patcher object)
      - non-string patch_id
      - patcher not TextPatcher-shaped (duck-typed: needs target_file,
        marker, sub_patches attributes)

    Re-registration with the SAME patcher object is a no-op (allows
    module reimport during testing).
    """
    if not isinstance(patch_id, str) or not patch_id:
        raise ValueError(f"patch_id must be non-empty str, got {patch_id!r}")

    # Duck-typed shape check — accept anything TextPatcher-like
    for attr in ("target_file", "marker", "sub_patches"):
        if not hasattr(patcher, attr):
            raise ValueError(
                f"patcher missing required attribute {attr!r} — "
                f"got {type(patcher).__name__}"
            )

    with _LOCK:
        existing = _REGISTERED.get(patch_id)
        if existing is None:
            _REGISTERED[patch_id] = patcher
            log.debug("registered patcher %s -> %s", patch_id, patcher.target_file)
            return
        # Allow re-registration if the SAME object (idempotent under reimport)
        if existing is patcher:
            return
        # Different patcher object with same id — programming error
        raise ValueError(
            f"patch_id {patch_id!r} already registered with a different "
            f"patcher (existing target_file={existing.target_file!r}, "
            f"new={patcher.target_file!r})"
        )


def get_registered_patcher(patch_id: str) -> Optional["TextPatcher"]:
    """Lookup. None if patch_id not registered."""
    with _LOCK:
        return _REGISTERED.get(patch_id)


def iter_registered_patchers() -> Iterator[tuple[str, "TextPatcher"]]:
    """Iterate (patch_id, patcher) pairs in registration order.

    Snapshot semantics — caller iterates a list copy, so concurrent
    register/unregister doesn't trip iteration.
    """
    with _LOCK:
        snapshot = list(_REGISTERED.items())
    return iter(snapshot)


def registered_count() -> int:
    """Total patchers currently registered."""
    with _LOCK:
        return len(_REGISTERED)


def clear_registry() -> None:
    """Wipe the registry. ONLY for tests — production code MUST NOT
    call this (would silently disable Site Map lookups for other patches).
    """
    with _LOCK:
        _REGISTERED.clear()
