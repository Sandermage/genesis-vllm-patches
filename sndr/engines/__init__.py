# SPDX-License-Identifier: Apache-2.0
"""Engine adapter registry.

The registry is the only public way to obtain an EngineAdapter class. This
ensures all engine instantiations go through a single decision point where
we can audit/log/gate.

Usage:
    >>> from sndr.engines import get_engine
    >>> EngineCls = get_engine("vllm")
    >>> adapter = EngineCls(config=SndrConfig.from_env())
    >>> adapter.bootstrap()
"""
from __future__ import annotations

from sndr.engines.base import EngineAdapter, EngineInfo, ModelProfile
from sndr.exceptions import EngineUnsupportedError

# Internal registry. Populated by imports below.
_REGISTRY: dict[str, type[EngineAdapter]] = {}


def register_engine(name: str, adapter_cls: type[EngineAdapter]) -> None:
    """Register an engine adapter class.

    Engine adapters call this from their module-level imports. The vllm
    adapter is registered eagerly (always available); sglang is registered
    only if its adapter module can be imported.
    """
    _REGISTRY[name] = adapter_cls


def get_engine(name: str) -> type[EngineAdapter]:
    """Return the adapter class for the given engine name.

    Raises:
        EngineUnsupportedError: If name is not registered.
    """
    if name not in _REGISTRY:
        raise EngineUnsupportedError(
            f"Engine {name!r} is not registered. Available: {list(_REGISTRY)}",
            requested=name,
            available=list(_REGISTRY),
        )
    return _REGISTRY[name]


def list_engines() -> list[str]:
    """Return all registered engine names."""
    return sorted(_REGISTRY.keys())


# --- Eager imports of built-in engines -------------------------------------
# vllm is always available (community-tier).
# sglang is optional; if the module exists but cannot be imported (e.g.
# missing dependency), we silently skip — the engine simply will not appear
# in list_engines(). This is intentional: we do not want a broken sglang
# adapter to prevent vllm-only operators from using sndr.

try:
    from sndr.engines.vllm import VllmEngine
    register_engine("vllm", VllmEngine)
except ImportError as e:
    import logging
    logging.getLogger("sndr.engines").warning(
        "Failed to register vllm engine adapter: %s", e,
    )

try:
    from sndr.engines.sglang import SglangEngine  # type: ignore[attr-defined]
    register_engine("sglang", SglangEngine)
except ImportError:
    # sglang adapter not yet implemented (skeleton only).
    pass

__all__ = [
    "EngineAdapter",
    "EngineInfo",
    "ModelProfile",
    "get_engine",
    "list_engines",
    "register_engine",
]
