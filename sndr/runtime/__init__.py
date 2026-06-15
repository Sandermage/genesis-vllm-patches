# SPDX-License-Identifier: Apache-2.0
"""SNDR Core runtime utilities — VRAM / memory / interface helpers.

v10 (2026-05-07): all submodules are loaded LAZILY via `__getattr__`.
Eager imports broke torch-less environments (CLI, schema validator,
doctor, pre-commit hooks, registry audit) because `prealloc.py`
imports torch at module top-level. The legacy `vllm/_genesis/__init__.py`
documents the same trap (v7 G-002 fix). Apply the same pattern here so
`import sndr` and `python -m sndr.cli --help` work
without torch installed.

Canonical imports (preferred for new code):

    from sndr.runtime import buffer_mode, prealloc, pool_budget
    from sndr.runtime.gpu_profile import detect_gpu

Migration history:
  - Original location: vllm/_genesis/<module>.py (Stage 0).
  - Stage 4 (2026-05-07): runtime/ re-export shims created.
  - Stage 6 (2026-05-07): patches may switch imports to sndr_core paths.
  - v10  (2026-05-07): canonical impl moved here from `_genesis/`;
                       eager submodule imports replaced with lazy
                       `__getattr__` to keep torch-less imports clean.
"""
from __future__ import annotations

import importlib

_LAZY_SUBMODULES = (
    "buffer_mode",
    "prealloc",
    "prealloc_budget",
    "pool_budget",
    "memory_metrics",
    "gpu_profile",
    "spec_meta",
    "interface_guard",
)


def __getattr__(name: str):
    """Lazy submodule loader. `prealloc` (and any future torch-using
    submodule) is loaded only on first attribute access, so simple
    `import sndr.runtime` does not pull torch."""
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"sndr.runtime.{name}")
    raise AttributeError(f"module 'sndr.runtime' has no attribute {name!r}")


def __dir__():
    return sorted(set(_LAZY_SUBMODULES) | set(globals().keys()))


__all__ = list(_LAZY_SUBMODULES)
