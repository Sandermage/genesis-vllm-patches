# SPDX-License-Identifier: Apache-2.0
"""sndr-platform — multi-engine inference patch orchestrator.

Quick start::

    import sndr
    sndr.init()                          # uses env vars
    engine = sndr.active_engine()
    print(engine.info.version)

Configuration is via environment variables; see ``sndr.config.SndrConfig``
for the full list, or ``docs/reference/ENV_VARS.md`` for prose docs.

Public API:
    __version__       Single source of truth for the package version.
    SndrConfig        Typed runtime configuration dataclass.
    init()            Bootstrap the platform (idempotent).
    active_engine()   Return the currently-bootstrapped engine adapter.
    is_initialized()  Probe whether init() has run.

This package follows a layered architecture (kernel → engines → dispatcher
→ apply → product_api → cli). See ``docs/concepts/ARCHITECTURE.md`` and
``docs/_adr/0001-multi-engine-refactor.md``.
"""
from __future__ import annotations

import logging

from sndr.config import SndrConfig
from sndr.engines import EngineAdapter, get_engine, list_engines
from sndr.exceptions import SndrError
from sndr.observability import configure_logging
from sndr.version import __version__, __version_major__

log = logging.getLogger("sndr")

# Module-level state. Treat as private.
_initialized: bool = False
_active_engine: EngineAdapter | None = None
_active_config: SndrConfig | None = None


def init(config: SndrConfig | None = None) -> EngineAdapter:
    """Bootstrap the sndr platform.

    Performs:
      1. Load configuration (from env if config is None).
      2. Configure observability (structured logging).
      3. Ensure SNDR_HOME directories exist.
      4. Select and bootstrap the engine adapter.

    Idempotent: subsequent calls return the existing engine adapter without
    re-running bootstrap.

    Args:
        config: Optional explicit configuration. If None, loads from env.

    Returns:
        The bootstrapped EngineAdapter instance.

    Raises:
        ConfigError: Invalid configuration.
        EngineUnsupportedError: SNDR_ENGINE refers to an unknown engine.
        EngineNotInstalledError: Engine package not importable.
        LicenseError: License token present but invalid (only for engine tier).
    """
    global _initialized, _active_engine, _active_config

    if _initialized and _active_engine is not None:
        return _active_engine

    _active_config = config or SndrConfig.from_env()
    _active_config.ensure_home()

    configure_logging(level=_active_config.log_level)

    log.info(
        "sndr.lifecycle.imported",
        extra={
            "version": __version__,
            "engine": _active_config.engine,
            "engine_pin": _active_config.engine_pin,
            "sndr_home": str(_active_config.sndr_home),
        },
    )

    EngineCls = get_engine(_active_config.engine)
    _active_engine = EngineCls(config=_active_config)
    _active_engine.bootstrap()

    log.info(
        "sndr.lifecycle.ready",
        extra={
            "engine": _active_engine.name,
            "engine_version": _active_engine.info.version,
            "engine_pin": _active_engine.info.pin,
            "supported": _active_engine.info.supported,
        },
    )

    _initialized = True
    return _active_engine


def active_engine() -> EngineAdapter | None:
    """Return the currently-bootstrapped engine adapter, or None if not init'd."""
    return _active_engine


def active_config() -> SndrConfig | None:
    """Return the currently-active configuration, or None if not init'd."""
    return _active_config


def is_initialized() -> bool:
    """Return True if init() has been called successfully."""
    return _initialized


def _reset_for_tests() -> None:
    """Test-only: reset module state. Do not call from production code."""
    global _initialized, _active_engine, _active_config
    _initialized = False
    _active_engine = None
    _active_config = None


__all__ = [
    "__version__",
    "__version_major__",
    "SndrConfig",
    "SndrError",
    "EngineAdapter",
    "init",
    "active_engine",
    "active_config",
    "is_initialized",
    "get_engine",
    "list_engines",
]
