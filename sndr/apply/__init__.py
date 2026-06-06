# SPDX-License-Identifier: Apache-2.0
"""SNDR Core apply — public API.

Composed of 4 modules (split from vllm/_genesis/patches/apply_all.py
5273-LOC monolith at Stage 3):

  - _state.py              — PatchResult, PatchStats, _APPLY_MODE, helpers
  - _per_patch_dispatch.py — 95 apply_patch_X functions (parking lot
                             until Stage 6 subsystem reorg)
  - orchestrator.py        — run() + main()
  - verify.py              — verify_live_rebinds()

Public API used by CLI + integrators.
"""
from __future__ import annotations

import sys as _sys

from ._state import (  # noqa: F401
    PatchResult,
    PatchStats,
    PATCH_REGISTRY,
    register_patch,
)
from .orchestrator import main, run  # noqa: F401
from .verify import verify_live_rebinds  # noqa: F401

# PR38 cleanup (2026-05-08): legacy `vllm/_genesis/patches/apply_all.py`
# was the all-in-one entry. Tests historically did
# `from vllm.sndr_core.apply import register_patch, run, ...`
# After Stage-3 split + PR38 _genesis removal, the same symbols live on
# `vllm.sndr_core.apply` (this package). Tests now import from here, but
# some still reach for an `apply_all` *attribute* (the legacy module
# object). Expose this package itself as `apply_all` so that pattern
# keeps working — `getattr(vllm.sndr_core.apply, "apply_all")` returns
# this module, and `from vllm.sndr_core.apply import apply_all` rebinds
# it as a local name.
apply_all = _sys.modules[__name__]


def __getattr__(name):
    """Forward attribute access to `_per_patch_dispatch` for `apply_patch_X`
    function lookups.

    PR38 cleanup (2026-05-08): tests historically did
    `from vllm._genesis.patches import apply_all` then
    `assert hasattr(apply_all, "apply_patch_NXX")`. After migration,
    they do `from vllm.sndr_core.apply import apply_all` (which is now
    THIS package) then the same hasattr check. This `__getattr__`
    forwards lookups for any `apply_patch_*` name into the parking-lot
    dispatch module so the contract holds.
    """
    if name.startswith("apply_patch_"):
        from vllm.sndr_core.apply import _per_patch_dispatch
        if hasattr(_per_patch_dispatch, name):
            return getattr(_per_patch_dispatch, name)
    raise AttributeError(
        f"module 'vllm.sndr_core.apply' has no attribute {name!r}"
    )


def __dir__():
    """Include `apply_patch_*` names in dir() for tab-completion + introspection."""
    base = list(globals().keys())
    try:
        from vllm.sndr_core.apply import _per_patch_dispatch
        base += [n for n in dir(_per_patch_dispatch) if n.startswith("apply_patch_")]
    except ImportError:
        pass
    return sorted(set(base))


__all__ = [
    "PatchResult",
    "PatchStats",
    "PATCH_REGISTRY",
    "register_patch",
    "run",
    "main",
    "verify_live_rebinds",
    "apply_all",
]


if __name__ == "__main__":
    import sys
    sys.exit(main())
