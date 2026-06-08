# SPDX-License-Identifier: Apache-2.0
"""SNDR Core — license / tier eligibility checks.

Phase 4 (F-010-012 audit fix, 2026-05-08): the previous tier gate in
`dispatcher.decision._check_engine_tier_eligible` only checked whether
`vllm.sndr_engine` could be imported. That's not a real paid/free
boundary — anyone who clones the repo gets the engine package.

This module adds a structured license check the gate consults BEFORE
allowing a `tier="engine"` patch to apply:

  1. `SNDR_ENGINE_LICENSE_KEY` env — Ed25519-signed token (required).
  2. `~/.sndr/license.json` file — JSON with `{"key": "..."}`.
  3. `vllm.sndr_engine` package present AND its version compatible
     with the community-tier `SNDR_CORE_VERSION`.

**Legacy unsigned key support is opt-in only.** By default any
non-signed key (no `.` separator) returns `BAD_SIGNATURE`. Setting
`SNDR_ALLOW_LEGACY_LICENSE_KEYS=1` re-enables dev/CI/transition-window
acceptance with a `LICENSED_LEGACY` status and a one-time warning;
production deployments must use signed tokens.

E-05 fix (2026-05-08): cryptographic signature verification using
Ed25519. Tokens have the form `<base64url-payload>.<base64url-sig>`
where the payload is JSON `{"customer_id":..., "issued_at":...,
"expires_at":..., "engine_major":...}` and the signature is over the
raw base64url-payload bytes by Sander's offline private key.

Etap 0.1 (audit 2026-05-12): payload contract is now strictly enforced.
Missing or wrong-type fields return `BAD_PAYLOAD` (distinct from
`BAD_SIGNATURE`). Previously a signature-valid token without
`expires_at` was treated as unlimited — fixed via `_validate_payload_contract`.

Etap 0.5 (audit 2026-05-12): `verify_token`, `is_placeholder_anchor`,
`TokenVerification` are part of the public `__all__`. Use these instead
of the private `_`-prefixed equivalents in operator scripts and ceremony
documentation.

The check returns a structured `LicenseStatus` enum so callers can
emit operator-friendly messages ("install the package" vs "set the
key" vs "version mismatch" vs "expired" vs "bad payload") rather
than a single failure mode.

Author: Sandermage (Sander) Barzov Aleksandr.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("genesis.license")


# ─── Trust anchor ────────────────────────────────────────────────────────
#
# The current public-repo build ships a development-only Ed25519
# anchor (32 raw bytes, base64url-encoded, no padding). Production
# release MUST replace this constant with a production-only key
# generated under the offline ceremony described in
# docs/security/TRUST_ANCHOR_CEREMONY.md, AND the
# `audit-license-anchor` gate refuses any release whose fingerprint
# still matches `_DEV_ANCHOR_FINGERPRINT_FORBIDDEN` below.
#
# Rotation flow (maintainer-only — end users never run this):
#
#   1. Run the offline trust-anchor ceremony on an air-gapped machine.
#      The private key never reaches stdout under the security-first
#      default; it is written to a secure path under the maintainer's
#      control.
#   2. Store the private key offline (YubiKey / paper / safe).
#   3. Update this constant with the new public key, re-issue all
#      active customer tokens, ship the wheel.
#   4. Old tokens stop verifying as soon as the wheel updates — that
#      is the intended trust-rotation pattern.
_TRUST_ANCHOR_PUBKEY_B64URL = (
    "iSk29MUb9HldKokPRyOG7bAjwYaQdgqYsS17yfskE8s"
)

# Fingerprint of the development-only anchor. The release-gate audit
# blocks any production build that ships this value verbatim; rotate
# via the ceremony before flipping to public release.
_DEV_ANCHOR_FINGERPRINT_FORBIDDEN: frozenset[str] = frozenset({
    "iSk29MUb9HldKokPRyOG7bAjwYaQdgqYsS17yfskE8s",
})

# Boot-time warning latch — when set we emit once per process so
# operators notice the unsigned-mode posture in their startup logs.
_TRUST_ANCHOR_PLACEHOLDER_DETECTED = False


def _is_placeholder_anchor() -> bool:
    """Detect the all-zero placeholder pubkey."""
    try:
        import base64 as _b64
        s = _TRUST_ANCHOR_PUBKEY_B64URL
        pad = "=" * (-len(s) % 4)
        raw = _b64.urlsafe_b64decode(s + pad)
        return raw == b"\x00" * 32
    except Exception:
        return False


def _maybe_log_placeholder_warning() -> None:
    """Idempotent boot-time warning surface. Called once per process."""
    global _TRUST_ANCHOR_PLACEHOLDER_DETECTED
    if _TRUST_ANCHOR_PLACEHOLDER_DETECTED:
        return
    _TRUST_ANCHOR_PLACEHOLDER_DETECTED = True
    if _is_placeholder_anchor():
        log.warning(
            "[Genesis license] Trust anchor is the PLACEHOLDER zero-key. "
            "Signed Ed25519 license tokens will be rejected with status "
            "BAD_SIGNATURE — only plain `SNDR_ALLOW_LEGACY_LICENSE_KEYS=1` "
            "tokens will work. Production deployments must replace the "
            "placeholder via the offline trust-anchor ceremony and ship "
            "a fresh wheel."
        )


_maybe_log_placeholder_warning()


# ─── Enums + result types ────────────────────────────────────────────────


class LicenseStatus(str, Enum):
    """Outcome of the engine-tier license check."""
    LICENSED = "licensed"
    LICENSED_LEGACY = "licensed_legacy"        # plain key, signature not checked
    NO_PACKAGE = "no_package"
    NO_KEY = "no_key"
    BAD_SIGNATURE = "bad_signature"
    BAD_PAYLOAD = "bad_payload"                # E-0.1: signature OK, payload contract violated
    EXPIRED = "expired"
    VERSION_MISMATCH = "version_mismatch"
    OVERRIDE = "override"


@dataclass(frozen=True)
class EligibilityResult:
    """Structured outcome of `check_engine_tier_eligible()`.

    `eligible=True` means the engine-tier patch is allowed to apply;
    callers should respect `reason` for operator-facing log lines.
    """
    eligible: bool
    status: LicenseStatus
    reason: str


# ─── Sources of license signal ───────────────────────────────────────────


_LICENSE_FILE_DEFAULT = Path("~/.sndr/license.json").expanduser()


def _read_license_key_from_env() -> Optional[str]:
    """Read `SNDR_ENGINE_LICENSE_KEY` env. Empty → None."""
    val = os.environ.get("SNDR_ENGINE_LICENSE_KEY", "").strip()
    return val or None


def _read_license_key_from_file(
    path: Optional[Path] = None,
) -> Optional[str]:
    """Read `~/.sndr/license.json` `{"key": "..."}`. Missing file or
    bad JSON → None."""
    p = path if path is not None else _LICENSE_FILE_DEFAULT
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        log.debug("[license] %s unreadable: %s", p, e)
        return None
    if not isinstance(data, dict):
        return None
    key = data.get("key")
    if not isinstance(key, str) or not key.strip():
        return None
    return key.strip()


def _resolve_license_key() -> Optional[str]:
    """Try env first, then `~/.sndr/license.json`. Returns first hit."""
    return _read_license_key_from_env() or _read_license_key_from_file()


# ─── Engine package presence + version ───────────────────────────────────


def _engine_overlay_available() -> bool:
    """DA-010 (audit 2026-05-08): probe via `engine_available()` —
    NOT raw `import vllm.sndr_engine`.

    Why: a skeleton `vllm.sndr_engine` package may exist in the public
    repo (or in operator's site-packages) without being a real
    commercial overlay. Using `import` as the availability signal
    falsely activates the engine gate. The canonical signal is
    `engine_available()` returning True, which only happens when a
    REAL private overlay is registered (via `sndr.engine.overlay`
    entry point — see roadmap §15.1).
    """
    try:
        from vllm.sndr_engine import engine_available
    except ImportError:
        return False
    try:
        return bool(engine_available())
    except Exception:
        return False


def _engine_package_version() -> Optional[str]:
    """Return the engine package's declared version, or None if no
    real engine overlay is present.

    DA-010: prefers the overlay's own version when available; falls
    back to the skeleton package's `__version__` only when
    `engine_available()` returned True. A skeleton-only install
    returns None here so the license gate stays closed.
    """
    if not _engine_overlay_available():
        return None
    try:
        import vllm.sndr_engine as _engine
    except ImportError:
        return None
    return getattr(_engine, "__version__", None)


def _versions_compatible(core_v: str, engine_v: str) -> bool:
    """Compatibility rule: major component must match.

    Both versions are simple `MAJOR.MINOR.PATCH` strings (no PEP 440
    pre-release suffixes today). Compare the leading integer.
    """
    try:
        core_major = int(core_v.lstrip("v").split(".", 1)[0])
        engine_major = int(engine_v.lstrip("v").split(".", 1)[0])
    except (ValueError, AttributeError):
        # Can't parse → conservatively treat as incompatible.
        return False
    return core_major == engine_major


# ─── Cryptographic token verification ────────────────────────────────────


@dataclass(frozen=True)
class TokenVerification:
    """Result of `verify_token()` / `_verify_signed_token()`.

    Public API: use `verify_token(token)` to verify signed
    license tokens against the trust anchor. Returns a structured outcome
    with `status` (LicenseStatus), `payload` (dict with customer_id etc.),
    and `detail` (operator-readable explanation).
    """
    status: LicenseStatus
    payload: Optional[dict[str, Any]]
    detail: str


# Back-compat alias — internal code previously imported the private name.
_TokenVerification = TokenVerification


def _b64url_decode(s: str) -> bytes:
    """URL-safe base64 decode, tolerant of missing padding."""
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _looks_signed(key: str) -> bool:
    """A signed token has exactly one '.' splitting payload from sig."""
    return key.count(".") == 1 and len(key) > 16


# strict payload contract.
#
# Without this, silent-pass: a signed token without `expires_at` validated
# as LICENSED and became indefinite. Every field is required, types are
# strict, bool is rejected for numeric fields (Python `True/False` is a
# subclass of int and can slip past isinstance checks).
_PAYLOAD_CONTRACT: dict[str, tuple[type, ...]] = {
    "customer_id":  (str,),
    "issued_at":    (int, float),
    "expires_at":   (int, float),
    "engine_major": (int,),
}
_CLOCK_SKEW_SECONDS = 60.0


def _validate_payload_contract(
    payload: dict[str, Any], *, now_epoch: float,
) -> Optional[str]:
    """Strict contract gate. Returns error message or None when valid.

    Checks (in order):
      1. All required fields are present.
      2. Each field has the correct type.
      3. bool is not accepted as int/float (Python quirk).
      4. customer_id is a non-empty string after strip.
      5. issued_at / expires_at are positive epoch seconds.
      6. expires_at > issued_at (sane interval).
      7. issued_at <= now + skew (token issued in the future = clock attack
         or misconfigured signer).
    """
    for field, types in _PAYLOAD_CONTRACT.items():
        if field not in payload:
            return f"missing required field {field!r}"
        value = payload[field]
        # bool is a subclass of int; reject it explicitly for numeric fields.
        if isinstance(value, bool) and (int in types or float in types):
            return f"field {field!r} must not be bool"
        if not isinstance(value, types):
            wanted = "/".join(t.__name__ for t in types)
            return (
                f"field {field!r} has wrong type "
                f"{type(value).__name__} (expected {wanted})"
            )
    if not payload["customer_id"].strip():
        return "field 'customer_id' must be non-empty"
    if payload["issued_at"] <= 0 or payload["expires_at"] <= 0:
        return "issued_at/expires_at must be positive epoch seconds"
    if payload["expires_at"] <= payload["issued_at"]:
        return "expires_at must be greater than issued_at"
    if payload["issued_at"] > now_epoch + _CLOCK_SKEW_SECONDS:
        return (
            f"issued_at={payload['issued_at']:.0f} in the future "
            f"(now {now_epoch:.0f}, max skew {_CLOCK_SKEW_SECONDS:.0f}s)"
        )
    return None


def _verify_signed_token(
    key: str,
    *,
    pubkey_b64url: str = _TRUST_ANCHOR_PUBKEY_B64URL,
    now_epoch: Optional[float] = None,
) -> _TokenVerification:
    """Decode + verify an Ed25519-signed license token.

    Token format: `<base64url-payload>.<base64url-signature>`.
    Payload is JSON `{customer_id, issued_at, expires_at, engine_major}`.
    Signature is Ed25519 over the raw base64url-payload bytes.

    Requires the optional `cryptography` package. Without it, returns
    BAD_SIGNATURE so a misconfigured environment fails closed instead
    of silently downgrading to legacy mode.
    """
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except ImportError:
        return _TokenVerification(
            status=LicenseStatus.BAD_SIGNATURE,
            payload=None,
            detail=(
                "cryptography package not installed — cannot verify "
                "signed license token. `pip install cryptography` to "
                "enable engine-tier with signed keys."
            ),
        )

    payload_b64, sig_b64 = key.split(".", 1)
    try:
        sig_bytes = _b64url_decode(sig_b64)
        pubkey_bytes = _b64url_decode(pubkey_b64url)
    except Exception as e:
        return _TokenVerification(
            status=LicenseStatus.BAD_SIGNATURE,
            payload=None,
            detail=f"token components not base64url-decodable: {e}",
        )

    if pubkey_bytes == b"\x00" * 32:
        return _TokenVerification(
            status=LicenseStatus.BAD_SIGNATURE,
            payload=None,
            detail=(
                "trust anchor public key is the placeholder zero key — "
                "production sndr_core not yet shipped with a real signing "
                "key. Use a legacy key or wait for the next release."
            ),
        )

    try:
        pubkey = Ed25519PublicKey.from_public_bytes(pubkey_bytes)
        pubkey.verify(sig_bytes, payload_b64.encode("ascii"))
    except InvalidSignature:
        return _TokenVerification(
            status=LicenseStatus.BAD_SIGNATURE,
            payload=None,
            detail="signature did not verify against trust anchor",
        )
    except Exception as e:
        return _TokenVerification(
            status=LicenseStatus.BAD_SIGNATURE,
            payload=None,
            detail=f"verify raised {type(e).__name__}: {e}",
        )

    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception as e:
        return _TokenVerification(
            status=LicenseStatus.BAD_SIGNATURE,
            payload=None,
            detail=f"payload not JSON after signature OK: {e}",
        )
    if not isinstance(payload, dict):
        return _TokenVerification(
            status=LicenseStatus.BAD_SIGNATURE,
            payload=None,
            detail="payload is not a JSON object",
        )

    now = now_epoch if now_epoch is not None else time.time()

    # Etap 0.1: strict payload contract — reject missing/wrong-type
    # fields BEFORE the expires check, otherwise bug "missing expires_at = unlimited".
    contract_error = _validate_payload_contract(payload, now_epoch=now)
    if contract_error:
        return _TokenVerification(
            status=LicenseStatus.BAD_PAYLOAD,
            payload=payload,
            detail=f"payload contract: {contract_error}",
        )

    exp = payload["expires_at"]  # type guaranteed by the validator above
    if exp < now:
        return _TokenVerification(
            status=LicenseStatus.EXPIRED,
            payload=payload,
            detail=(
                f"license expired at epoch {exp:.0f} "
                f"(now {now:.0f}); contact licensing for renewal"
            ),
        )

    return _TokenVerification(
        status=LicenseStatus.LICENSED,
        payload=payload,
        detail=f"signature OK, customer_id={payload['customer_id']}",
    )


# ─── Public API ──────────────────────────────────────────────────────────


def is_placeholder_anchor() -> bool:
    """Return True if the trust anchor is the 32-zero placeholder.

    Operator-facing alias for `_is_placeholder_anchor`. Used in
    CI gates and the ceremony doc to avoid depending on private names.
    """
    return _is_placeholder_anchor()


def verify_token(
    token: str,
    *,
    pubkey_b64url: Optional[str] = None,
    now_epoch: Optional[float] = None,
) -> TokenVerification:
    """Verify an Ed25519-signed Genesis license token.

    Etap 0.5 (audit 2026-05-12): public API for the ceremony doc and custom
    licensing integrations. Previously the documentation referenced this
    function, but in reality only the private `_verify_signed_token` existed.

    Args:
        token: signed token string `<base64url-payload>.<base64url-signature>`.
        pubkey_b64url: override the bundled trust anchor (testing / alternative
            anchor). None → uses the built-in constant
            `_TRUST_ANCHOR_PUBKEY_B64URL`.
        now_epoch: override the wall clock for expiry checks (testing).

    Returns:
        TokenVerification with status, payload (dict if decoded), detail.
        Status may be: LICENSED / BAD_SIGNATURE / BAD_PAYLOAD / EXPIRED.
    """
    if pubkey_b64url is None:
        pubkey_b64url = _TRUST_ANCHOR_PUBKEY_B64URL
    if not _looks_signed(token):
        return TokenVerification(
            status=LicenseStatus.BAD_SIGNATURE,
            payload=None,
            detail=(
                "token format invalid: expected "
                "`<base64url-payload>.<base64url-signature>`"
            ),
        )
    return _verify_signed_token(
        token, pubkey_b64url=pubkey_b64url, now_epoch=now_epoch,
    )


def check_engine_tier_eligible(
    *,
    license_key: Optional[str] = None,
    license_file: Optional[Path] = None,
    skip_override_check: bool = False,
    now_epoch: Optional[float] = None,
) -> EligibilityResult:
    """Returns an `EligibilityResult` describing whether the caller is
    allowed to apply a `tier="engine"` patch.

    Args:
        license_key: explicit override for testing. If None, falls back
            to env + license file.
        license_file: explicit override for testing. If None, uses
            `~/.sndr/license.json`.
        skip_override_check: if True, ignore SNDR_ENABLE_TIER_OVERRIDE.
            Default False — the env override is honored.
        now_epoch: explicit time for testing token expiration. None →
            wall clock.

    Resolution order:
      1. `SNDR_ENABLE_TIER_OVERRIDE=1` — short-circuit to OVERRIDE
         (community-only mode forced, even if licensed)
      2. `vllm.sndr_engine` package importable
      3. License key present (env OR file)
      4. If key looks signed (one '.', ed25519): verify signature +
         expiration. Failure → BAD_SIGNATURE / EXPIRED.
      5. Otherwise: legacy plain key. Accepted with a deprecation
         warning logged once per process; status=LICENSED_LEGACY.
      6. Engine version major matches core version major
    """
    if not skip_override_check:
        from sndr.env import Flags, is_enabled
        if is_enabled(Flags.TIER_OVERRIDE):
            return EligibilityResult(
                eligible=False,
                status=LicenseStatus.OVERRIDE,
                reason=(
                    "SNDR_ENABLE_TIER_OVERRIDE=1 forces community-only mode "
                    "— engine-tier patches skipped"
                ),
            )

    engine_version = _engine_package_version()
    if engine_version is None:
        return EligibilityResult(
            eligible=False,
            status=LicenseStatus.NO_PACKAGE,
            reason=(
                "vllm.sndr_engine not installed — engine-tier patches "
                "require the commercial SNDR Engine package. Contact "
                "Sandermage for licensing or use community-tier alternatives."
            ),
        )

    if license_key is None:
        license_key = (
            _read_license_key_from_env()
            or _read_license_key_from_file(license_file)
        )
    if not license_key:
        return EligibilityResult(
            eligible=False,
            status=LicenseStatus.NO_KEY,
            reason=(
                "vllm.sndr_engine installed but no license key found. "
                "Set SNDR_ENGINE_LICENSE_KEY env OR write the key to "
                "~/.sndr/license.json `{\"key\": \"...\"}`."
            ),
        )

    # Signature verification path
    license_status: LicenseStatus
    license_detail: str
    if _looks_signed(license_key):
        verification = _verify_signed_token(license_key, now_epoch=now_epoch)
        if verification.status != LicenseStatus.LICENSED:
            return EligibilityResult(
                eligible=False,
                status=verification.status,
                reason=f"license token rejected: {verification.detail}",
            )
        license_status = LicenseStatus.LICENSED
        license_detail = verification.detail

        # Optional: token can pin engine major version. If present and
        # mismatched, reject. If absent, fall through to package check.
        if verification.payload:
            tok_major = verification.payload.get("engine_major")
            if isinstance(tok_major, int):
                try:
                    engine_major = int(engine_version.lstrip("v").split(".", 1)[0])
                except (ValueError, AttributeError):
                    engine_major = -1
                if tok_major != engine_major:
                    return EligibilityResult(
                        eligible=False,
                        status=LicenseStatus.VERSION_MISMATCH,
                        reason=(
                            f"license token bound to engine major {tok_major}, "
                            f"installed engine is {engine_version!r}"
                        ),
                    )
    else:
        # Legacy unsigned key — only accepted when explicitly allowed
        # by the operator. P1-3 fix (audit 2026-05-08): production
        # `pip install vllm-sndr-engine` must reject unsigned keys
        # so the license boundary actually means something. The
        # `SNDR_ALLOW_LEGACY_LICENSE_KEYS=1` env gate exists for the
        # transition window (testing, CI, dev) where signed-token
        # infrastructure isn't fully wired yet.
        legacy_allowed = os.environ.get(
            "SNDR_ALLOW_LEGACY_LICENSE_KEYS", ""
        ).strip().lower() in ("1", "true", "yes", "on")
        if not legacy_allowed:
            return EligibilityResult(
                eligible=False,
                status=LicenseStatus.BAD_SIGNATURE,
                reason=(
                    "license key is unsigned (no `payload.signature` "
                    "format detected). Production deployments require "
                    "an Ed25519-signed token. Set "
                    "SNDR_ALLOW_LEGACY_LICENSE_KEYS=1 ONLY for dev / "
                    "CI / transition-window environments where signed-"
                    "token infrastructure isn't yet wired up."
                ),
            )
        if not getattr(check_engine_tier_eligible, "_legacy_warned", False):
            log.warning(
                "[license] SNDR_ALLOW_LEGACY_LICENSE_KEYS=1 — accepting "
                "unsigned key (dev mode). Production must use signed tokens."
            )
            check_engine_tier_eligible._legacy_warned = True  # type: ignore[attr-defined]
        license_status = LicenseStatus.LICENSED_LEGACY
        license_detail = "legacy unsigned key (dev mode)"

    # Engine package vs core major-version compat (independent of token).
    from sndr.version import SNDR_CORE_VERSION
    if not _versions_compatible(SNDR_CORE_VERSION, engine_version):
        return EligibilityResult(
            eligible=False,
            status=LicenseStatus.VERSION_MISMATCH,
            reason=(
                f"sndr_engine version {engine_version!r} incompatible with "
                f"sndr_core {SNDR_CORE_VERSION!r}. Major version must match. "
                "Upgrade or downgrade one of the two packages."
            ),
        )

    return EligibilityResult(
        eligible=True,
        status=license_status,
        reason=(
            f"engine-tier eligible (sndr_engine v{engine_version}, "
            f"core v{SNDR_CORE_VERSION}, {license_detail})"
        ),
    )


__all__ = [
    "LicenseStatus",
    "EligibilityResult",
    "TokenVerification",
    "check_engine_tier_eligible",
    "verify_token",
    "is_placeholder_anchor",
    # public/private boundary CLI surface (not engine-tier gate)
    "EngineDetection",
    "CoreLicenseStatus",
    "LicenseVerifyResult",
    "is_engine_installed",
    "core_license_status",
    "verify_license_file",
]


# ─── Phase 4.6 — public/private boundary CLI surface ───────────────────
#
# These are operator-facing helpers for `sndr license status` / `verify`
# and the `audit-security` release gate. They are intentionally SEPARATE
# from the per-patch engine-tier eligibility above (`check_engine_tier_eligible`)
# because they answer a different question:
#
#   - check_engine_tier_eligible(patch_id) → "is THIS patch allowed to run?"
#   - core_license_status()                → "is the engine installed at all?
#                                             what does the boundary look like?"
#
# Public core NEVER blocks on `core_license_status()` — it always reports
# the core as "public (unlicensed)" and surfaces engine detection if
# present. This preserves the invariant that public core is fully
# functional without any license file.


@dataclass(frozen=True)
class EngineDetection:
    """Result of optional private engine discovery (Phase 4.6 boundary)."""
    installed: bool
    module_name: Optional[str] = None
    version: Optional[str] = None


@dataclass(frozen=True)
class CoreLicenseStatus:
    """Returned by `core_license_status()`. Stable shape for tooling.

    Distinct from `EligibilityResult` (per-patch gate) and from the
    legacy `LicenseStatus` enum (token verification states). This is the
    public boundary view consumed by `sndr license status`.
    """
    core: str                            # always "public (unlicensed)"
    engine: Optional[EngineDetection]
    license_path: Optional[str] = None
    license_subject: Optional[str] = None
    license_tier: Optional[str] = None
    license_expires: Optional[str] = None
    license_signature_valid: Optional[bool] = None
    premium_patches_enabled: int = 0


@dataclass(frozen=True)
class LicenseVerifyResult:
    """Returned by `verify_license_file()` (Phase 4.6 boundary)."""
    valid: bool
    reason: str
    subject: Optional[str] = None
    expires: Optional[str] = None


def is_engine_installed() -> EngineDetection:
    """Detect whether `vllm.sndr_engine` is importable (Phase 4.6).

    Uses `importlib.util.find_spec` so the engine module is NOT actually
    imported here (importing could trigger engine-side initialization
    that public core does not own).
    """
    try:
        import importlib.util
        spec = importlib.util.find_spec("vllm.sndr_engine")
    except (ModuleNotFoundError, ImportError, ValueError):
        return EngineDetection(installed=False)

    if spec is None:
        return EngineDetection(installed=False)

    # Read version metadata without importing the engine module.
    version: Optional[str] = None
    try:
        from importlib.metadata import PackageNotFoundError, version as _ver
        try:
            version = _ver("vllm-sndr-engine")
        except PackageNotFoundError:
            version = None
    except Exception:
        version = None

    return EngineDetection(
        installed=True,
        module_name="vllm.sndr_engine",
        version=version,
    )


def core_license_status() -> CoreLicenseStatus:
    """Return the public-core view of license status.

    INVARIANT: always reports the public core as `"public (unlicensed)"`.
    License-related fields are populated only when the engine declares
    itself active AND has loaded a license. Public core never blocks
    on this function.
    """
    engine = is_engine_installed()
    if not engine.installed:
        return CoreLicenseStatus(core="public (unlicensed)", engine=None)
    # Engine present — public core only reports its presence. Reading the
    # license file is the engine's responsibility, not ours.
    return CoreLicenseStatus(
        core="public (unlicensed)",
        engine=engine,
    )


def verify_license_file(path: str) -> LicenseVerifyResult:
    """Offline license-file signature verification (Phase 4.6 boundary).

    Public-core stub: returns `deferred` because real verification needs
    the private engine's pub-key + canonical message format. When the
    engine is absent, this function refuses to invent a verification
    result — that's the correct behavior.
    """
    engine = is_engine_installed()
    if not engine.installed:
        return LicenseVerifyResult(
            valid=False,
            reason=(
                "verification deferred — public core cannot verify "
                "license signatures. Install vllm-sndr-engine to enable "
                "offline signature verification."
            ),
        )
    # Engine present — delegate. Public core does NOT directly import the
    # engine's internals; the engine registers its hook via the standard
    # entry-point mechanism at install time.
    try:
        from vllm.sndr_engine.license import verify_license_file as _engine_verify
    except ImportError:
        return LicenseVerifyResult(
            valid=False,
            reason=(
                "engine module detected but missing license.verify_license_file "
                "hook — likely engine version mismatch"
            ),
        )
    return _engine_verify(path)
