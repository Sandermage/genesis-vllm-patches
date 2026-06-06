# SPDX-License-Identifier: Apache-2.0
"""Typed exception hierarchy for sndr-platform.

Every error raised by sndr code derives from SndrError. The hierarchy supports:

  - Structured logging (every exception carries a stable `code` attribute)
  - HTTP error mapping (each exception declares `http_status`)
  - RFC 7807 Problem Details rendering (extensions dict)

Engineering principle: NEVER raise bare Exception. Always raise a typed
SndrError subclass so consumers can pattern-match and surface meaningful
errors to operators.

Example:
    >>> try:
    ...     verify_license(token)
    ... except LicenseExpiredError as e:
    ...     log.warning("license.expired", extra=e.context)
"""
from __future__ import annotations

from typing import Any


class SndrError(Exception):
    """Base for every sndr-platform exception.

    Attributes:
        code: Stable string identifier (e.g. "sndr.license.expired").
            Used for telemetry aggregation and error catalog lookup.
        http_status: HTTP status code to return if this exception escapes
            to the API layer.
        message: Human-readable error message.
        context: Structured fields for logging and API extensions.
    """

    code: str = "sndr.error"
    http_status: int = 500

    def __init__(self, message: str = "", **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation suitable for logging."""
        return {
            "code": self.code,
            "message": self.message,
            "http_status": self.http_status,
            **self.context,
        }


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------


class ConfigError(SndrError):
    """Invalid or missing configuration."""
    code = "sndr.config.invalid"
    http_status = 400


class ConfigMissingError(ConfigError):
    """Required configuration value is not set."""
    code = "sndr.config.missing"


# ---------------------------------------------------------------------------
# License errors
# ---------------------------------------------------------------------------


class LicenseError(SndrError):
    """Base for license-related errors. Maps to HTTP 402 Payment Required."""
    code = "sndr.license.error"
    http_status = 402


class LicenseExpiredError(LicenseError):
    """The license token has expired."""
    code = "sndr.license.expired"


class LicenseVersionMismatchError(LicenseError):
    """The license token's engine_major does not match the installed version."""
    code = "sndr.license.version_mismatch"


class LicenseBadSignatureError(LicenseError):
    """The license token's Ed25519 signature failed verification."""
    code = "sndr.license.bad_signature"


class LicenseBadFormatError(LicenseError):
    """The license token has malformed structure."""
    code = "sndr.license.bad_format"


class LicenseNoPackageError(LicenseError):
    """sndr_engine wheel is required by the license but is not installed."""
    code = "sndr.license.no_package"
    http_status = 503


# ---------------------------------------------------------------------------
# Engine errors
# ---------------------------------------------------------------------------


class EngineError(SndrError):
    """Base for engine adapter errors."""
    code = "sndr.engine.error"


class EngineNotInstalledError(EngineError):
    """The requested engine package is not installed in this environment."""
    code = "sndr.engine.not_installed"
    http_status = 503


class EngineUnsupportedError(EngineError):
    """The requested engine is not supported by this version of sndr."""
    code = "sndr.engine.unsupported"
    http_status = 400


class EngineVersionMismatchError(EngineError):
    """The installed engine version is incompatible with sndr."""
    code = "sndr.engine.version_mismatch"


# ---------------------------------------------------------------------------
# Patch errors
# ---------------------------------------------------------------------------


class PatchError(SndrError):
    """Base for patch-related errors."""
    code = "sndr.patch.error"


class PatchAnchorDriftError(PatchError):
    """The patch's anchor was not found in the upstream file."""
    code = "sndr.patch.anchor_drift"
    http_status = 409


class PatchTargetMissingError(PatchError):
    """The patch's target file does not exist in the engine install."""
    code = "sndr.patch.target_missing"
    http_status = 404


class PatchApplyFailedError(PatchError):
    """The patch failed to apply (in strict mode)."""
    code = "sndr.patch.apply_failed"


# ---------------------------------------------------------------------------
# Pin errors
# ---------------------------------------------------------------------------


class PinError(SndrError):
    """Base for pin-related errors."""
    code = "sndr.pin.error"


class PinNotSupportedError(PinError):
    """The requested pin is not in the supported list for this engine."""
    code = "sndr.pin.not_supported"
    http_status = 400


class PinManifestMissingError(PinError):
    """The manifest for the requested pin is not present."""
    code = "sndr.pin.manifest_missing"
    http_status = 404


# ---------------------------------------------------------------------------
# Drift errors
# ---------------------------------------------------------------------------


class DriftDetectedError(SndrError):
    """An anchor or file md5 has changed in upstream relative to the manifest."""
    code = "sndr.drift.detected"
    http_status = 409


# ---------------------------------------------------------------------------
# Authentication / authorization errors
# ---------------------------------------------------------------------------


class AuthError(SndrError):
    """Base for authentication errors."""
    code = "sndr.auth.error"
    http_status = 401


class AuthInvalidCredentialsError(AuthError):
    """Username/password combination rejected."""
    code = "sndr.auth.invalid_credentials"


class AuthForbiddenError(AuthError):
    """User is authenticated but lacks permission for this action."""
    code = "sndr.auth.forbidden"
    http_status = 403


class AuthSessionExpiredError(AuthError):
    """The session token has expired or is invalid."""
    code = "sndr.auth.session_expired"


__all__ = [
    "SndrError",
    "ConfigError",
    "ConfigMissingError",
    "LicenseError",
    "LicenseExpiredError",
    "LicenseVersionMismatchError",
    "LicenseBadSignatureError",
    "LicenseBadFormatError",
    "LicenseNoPackageError",
    "EngineError",
    "EngineNotInstalledError",
    "EngineUnsupportedError",
    "EngineVersionMismatchError",
    "PatchError",
    "PatchAnchorDriftError",
    "PatchTargetMissingError",
    "PatchApplyFailedError",
    "PinError",
    "PinNotSupportedError",
    "PinManifestMissingError",
    "DriftDetectedError",
    "AuthError",
    "AuthInvalidCredentialsError",
    "AuthForbiddenError",
    "AuthSessionExpiredError",
]
