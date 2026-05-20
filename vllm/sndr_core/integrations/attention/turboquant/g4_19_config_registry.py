# SPDX-License-Identifier: Apache-2.0
"""Module-level singleton for the active G4-TurboQuant config.

================================================================
WHY THIS EXISTS
================================================================

vllm v1 (dev371+) made ``VllmConfig`` a **strict dataclass** that
enforces field membership during IPC pickle/unpickle. Attempting
``vllm_config._g4_19_turboquant_config = tq_config`` (the original
G4_19 approach) breaks worker startup::

    ValueError: Field '_g4_19_turboquant_config' not found in VllmConfig.

The strict-field check fires when the EngineCore process spawns
worker subprocesses and unpickles ``VllmConfig`` on the other side.

Instead, we publish the config to a **module-level singleton** in this
file. Each worker subprocess imports ``vllm.sndr_core`` at startup
(via the entrypoint or the import-time hook in ``sndr_core/__init__.py``)
and re-builds its own G4-TurboQuant config from environment variables.
This avoids the cross-process pickling path entirely — the registry is
populated independently in each process from the env vars set by docker
``-e`` flags.

================================================================
USAGE
================================================================

Producer (G4_19 apply() hook on Gemma4Config.verify_and_update_config)::

    from .g4_19_config_registry import set_active_config
    tq_config = G4TurboQuantConfig(...)
    set_active_config(tq_config)

Consumer (KV cache wrapper, attention layer hook)::

    from .g4_19_config_registry import get_active_config
    tq_config = get_active_config()
    if tq_config is None:
        # G4-TQ not active — fall through to standard vllm path
        return
    # use tq_config to dispatch the right kernel

================================================================
THREAD-SAFETY
================================================================

Set/get/clear are guarded by a module-level threading.Lock. Reads are
lock-free for hot-path performance (Python's GIL makes attribute load
atomic; we don't need a lock to read a single reference).

================================================================
MULTI-PROCESS NOTES
================================================================

Each worker (TP rank, DP shard, etc.) is a SEPARATE Python process. The
registry value is NOT shared across processes — each process must call
``set_active_config()`` itself.

Genesis arranges this via ``vllm.sndr_core.__init__._g4_19_import_time_hook``
which runs in every process that imports ``sndr_core`` (parent + workers).
That hook invokes ``g4_19.apply()`` which in turn calls ``set_active_config``.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .kernels.g4_tq_cache import G4TurboQuantConfig  # noqa: F401

log = logging.getLogger("genesis.gemma4.g4_19_registry")

GENESIS_G4_19_REGISTRY_MARKER = (
    "Genesis G4_19 module-level config registry v1 "
    "(replaces vllm_config attach blocked by dev371 strict dataclass)"
)

_LOCK = threading.Lock()
_ACTIVE_CONFIG: Optional["G4TurboQuantConfig"] = None


def set_active_config(config) -> None:
    """Register the active G4-TurboQuant config for THIS process.

    Idempotent: re-setting the same config object is a no-op. Setting a
    different config overwrites — caller is responsible for not racing
    different sub-systems.
    """
    global _ACTIVE_CONFIG
    with _LOCK:
        if _ACTIVE_CONFIG is config:
            return  # idempotent
        prev = _ACTIVE_CONFIG
        _ACTIVE_CONFIG = config
        if prev is None:
            log.info(
                "[G4_19_registry] config registered: pack=%s wht=%s "
                "bits_sliding=%d bits_global=%d head_dim=%d",
                getattr(config, "pack_mode", "?"),
                getattr(config, "wht_mode", "?"),
                getattr(config, "bits_sliding", -1),
                getattr(config, "bits_global", -1),
                getattr(config, "head_dim", -1),
            )
        else:
            log.warning(
                "[G4_19_registry] config replaced (previous: pack=%s wht=%s)",
                getattr(prev, "pack_mode", "?"),
                getattr(prev, "wht_mode", "?"),
            )


def get_active_config():
    """Return the active config, or None if G4-TurboQuant not activated.

    Lock-free read — the GIL makes a single attribute load atomic. A
    caller observing ``None`` simply falls back to the standard vllm
    KV-cache path.
    """
    return _ACTIVE_CONFIG


def clear_active_config() -> None:
    """Test helper — drop the active config."""
    global _ACTIVE_CONFIG
    with _LOCK:
        _ACTIVE_CONFIG = None


def is_active() -> bool:
    """Convenience predicate for hot-path checks."""
    return _ACTIVE_CONFIG is not None


__all__ = [
    "GENESIS_G4_19_REGISTRY_MARKER",
    "set_active_config",
    "get_active_config",
    "clear_active_config",
    "is_active",
]
