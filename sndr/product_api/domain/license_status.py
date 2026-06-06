# SPDX-License-Identifier: Apache-2.0
"""License status service — surfaces license info to the GUI."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from sndr.product_api.schemas.licensing import LicenseStatus


def get_license_status() -> LicenseStatus:
    """Return the current license status, suitable for display.

    The license token itself is NEVER returned. Only metadata (status,
    customer hash, expiry, engine_major) is exposed. This matches the
    security model documented in COMMERCIAL_TIER.md.
    """
    # Try to import the legacy license module. In Phase 4.5 the canonical
    # location is sndr.license (with a shim at vllm.sndr_core.license).
    try:
        from sndr import license as license_mod  # type: ignore[attr-defined]
    except ImportError:
        return LicenseStatus(
            status="unknown",
            message="License module not importable.",
        )

    # The legacy module exposes a status enum (LicenseStatus) plus a
    # ``core_license_status()`` helper. We translate its result into our
    # schema. If the helper isn't available we fall back to a probe via
    # ``check_engine_tier_eligible``.
    try:
        check = license_mod.check_engine_tier_eligible  # type: ignore[attr-defined]
        result = check()
    except (AttributeError, Exception):  # noqa: BLE001
        return LicenseStatus(
            status="unknown",
            message="License probe not available.",
        )

    # Extract status string from whatever check returned.
    status_value = "unknown"
    if isinstance(result, tuple) and result:
        # check_engine_tier_eligible historically returned (bool, status_enum)
        try:
            status_value = str(result[1]).split(".")[-1].lower()
        except (IndexError, AttributeError):
            status_value = "unknown"
    elif hasattr(result, "name"):
        status_value = result.name.lower()
    elif isinstance(result, str):
        status_value = result.lower()

    # Probe for plugin patches (entry points)
    try:
        from sndr.plugins.loader import get_plugin_info
        plugin_info = get_plugin_info()
        patches = sum(len(v) for v in plugin_info.values())
    except Exception:  # noqa: BLE001
        patches = 0

    # Probe for sndr_engine package
    try:
        import importlib
        importlib.import_module("sndr_engine")
        package_installed = True
    except ImportError:
        package_installed = False

    return LicenseStatus(
        status=status_value if status_value in {
            "licensed", "licensed_legacy", "expired", "bad_signature",
            "version_mismatch", "no_key", "no_package",
        } else "unknown",
        engine_package_installed=package_installed,
        engine_patches_available=patches,
    )


def _hash_customer_id(customer_id: str) -> str:
    """Return a short, non-reversible identifier for log correlation."""
    return hashlib.sha256(customer_id.encode("utf-8")).hexdigest()[:8]


__all__ = ["get_license_status"]
