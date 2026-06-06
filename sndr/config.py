# SPDX-License-Identifier: Apache-2.0
"""Typed configuration for sndr-platform.

Configuration is loaded from environment variables by default, with optional
override from a YAML config file pointed to by SNDR_CONFIG.

Engineering principle: configuration is **explicit**. Every value has a
documented env var and a documented default. No magic, no surprises.

Environment variables (full reference in docs/reference/ENV_VARS.md):

    SNDR_ENGINE              Required. "vllm" | "sglang".
    SNDR_ENGINE_PIN          Optional. If unset, engine adapter auto-detects.
    SNDR_CONFIG              Optional. Path to YAML config file.
    SNDR_HOME                Optional. State directory (default: ~/.sndr/).
    SNDR_LOG_LEVEL           Optional. "DEBUG" | "INFO" | "WARNING" | "ERROR".
                             Default: "INFO".
    SNDR_STRICT_DRIFT        Optional. "1" aborts boot on drift. Default: "0".
    SNDR_STRICT_APPLY        Optional. "1" aborts boot on patch failure.
                             Default: "0".
    SNDR_STRICT_DEPS         Optional. "1" aborts on missing patch deps.
                             Default: "0".
    SNDR_AUDIT_ON_APPLY      Optional. Emit audit log per patch. Default: "1".
    SNDR_OTEL_ENDPOINT       Optional. OpenTelemetry collector URL.
                             If unset, traces are not exported.
    SNDR_ENGINE_LICENSE_KEY  Optional. License token for engine tier.
                             If unset, falls back to file at ~/.sndr/license.json.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from sndr.exceptions import ConfigError

EngineName = Literal["vllm", "sglang"]


def _bool_env(name: str, default: bool = False) -> bool:
    """Parse a boolean from environment with explicit truthy values."""
    raw = os.environ.get(name, "")
    return raw.lower() in ("1", "true", "yes", "on") if raw else default


@dataclass(frozen=True)
class SndrConfig:
    """Frozen runtime configuration.

    Created once at sndr.init() via from_env(). Treat as immutable.
    """

    # Engine selection -------------------------------------------------
    engine: EngineName
    """Which engine adapter to load. Required."""

    engine_pin: str | None
    """Specific pin to target. If None, adapter auto-detects from
    the installed engine package."""

    # File system ------------------------------------------------------
    sndr_home: Path
    """State directory for sessions, audit logs, license cache, etc."""

    config_path: Path | None
    """Optional YAML config override path."""

    # Operational strictness ------------------------------------------
    strict_drift: bool
    """If True, abort boot when manifest drift is detected.
    If False, log warning and continue (recommended for production)."""

    strict_apply: bool
    """If True, abort boot when any patch fails to apply.
    If False, log error per failed patch and continue."""

    strict_deps: bool
    """If True, abort boot when an enabled patch's required dependency
    is not enabled."""

    audit_on_apply: bool
    """If True, write an audit log entry per patch application."""

    # Observability ----------------------------------------------------
    log_level: str
    """Python logging level: DEBUG, INFO, WARNING, ERROR."""

    otel_endpoint: str | None
    """OpenTelemetry collector URL. If None, traces not exported."""

    # Internal flags ---------------------------------------------------
    _legacy_compat: bool = field(default=True)
    """If True (default), backward compatibility shims for vllm.sndr_core
    are active. Disable in v13+ when shims are removed."""

    @classmethod
    def from_env(cls) -> "SndrConfig":
        """Construct config from environment variables.

        Raises:
            ConfigError: If a required env var is missing or invalid.
        """
        engine_raw = os.environ.get("SNDR_ENGINE", "vllm").lower()
        if engine_raw not in ("vllm", "sglang"):
            raise ConfigError(
                f"Invalid SNDR_ENGINE={engine_raw!r}. Must be 'vllm' or 'sglang'.",
                value=engine_raw,
                allowed=["vllm", "sglang"],
            )

        sndr_home = Path(
            os.environ.get("SNDR_HOME") or Path.home() / ".sndr"
        ).expanduser().resolve()

        config_path: Path | None = None
        if (cp := os.environ.get("SNDR_CONFIG")):
            config_path = Path(cp).expanduser().resolve()
            if not config_path.is_file():
                raise ConfigError(
                    f"SNDR_CONFIG points to non-existent file: {config_path}",
                    value=str(config_path),
                )

        log_level = os.environ.get("SNDR_LOG_LEVEL", "INFO").upper()
        if log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ConfigError(
                f"Invalid SNDR_LOG_LEVEL={log_level!r}",
                value=log_level,
            )

        return cls(
            engine=engine_raw,                                       # type: ignore[arg-type]
            engine_pin=os.environ.get("SNDR_ENGINE_PIN") or None,
            sndr_home=sndr_home,
            config_path=config_path,
            strict_drift=_bool_env("SNDR_STRICT_DRIFT", False),
            strict_apply=_bool_env("SNDR_STRICT_APPLY", False),
            strict_deps=_bool_env("SNDR_STRICT_DEPS", False),
            audit_on_apply=_bool_env("SNDR_AUDIT_ON_APPLY", True),
            log_level=log_level,
            otel_endpoint=os.environ.get("SNDR_OTEL_ENDPOINT") or None,
        )

    def ensure_home(self) -> None:
        """Create sndr_home directory and standard subdirs if missing.

        Idempotent. Called from sndr.init().
        """
        self.sndr_home.mkdir(parents=True, exist_ok=True)
        (self.sndr_home / "sessions").mkdir(exist_ok=True)
        (self.sndr_home / "audit").mkdir(exist_ok=True)
        (self.sndr_home / "cache").mkdir(exist_ok=True)


__all__ = ["SndrConfig", "EngineName"]
