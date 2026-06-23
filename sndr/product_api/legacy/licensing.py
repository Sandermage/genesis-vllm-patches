# SPDX-License-Identifier: Apache-2.0
"""Daemon bridge to the SNDR license / sndr_engine tier status.

Surfaces, read-only, what the licensing layer in ``sndr.license``
already computes: whether the commercial ``vllm.sndr_engine`` overlay is
installed, whether a valid Ed25519-signed license entitles the engine tier, the
customer/expiry/tier from the token, and how many engine-tier patches that
unlocks. The GUI uses this to show the active tier and to gate engine-only
modules instead of silently failing.

Degrades gracefully: if the license module can't be imported (it always ships
with sndr_core, but be defensive), every entry point returns an
``{"available": False}`` envelope rather than raising.
"""
from __future__ import annotations

from typing import Any


def _engine_tier_patch_count() -> int:
    try:
        from . import patches
        return sum(1 for p in patches.list_patches() if getattr(p, "tier", None) == "engine")
    except Exception:  # noqa: BLE001
        return 0


def status() -> dict[str, Any]:
    """Full license + sndr_engine status for the GUI."""
    try:
        from sndr import license as lic
    except Exception as exc:  # noqa: BLE001 — license layer should always be present
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    core = lic.core_license_status()
    elig = lic.check_engine_tier_eligible()
    det = lic.is_engine_installed()

    tier = getattr(core, "license_tier", None) or ("engine" if elig.eligible else "community")
    return {
        "available": True,
        "core": getattr(core, "core", "public (unlicensed)"),
        "tier": tier,
        "engine": {
            "installed": bool(getattr(det, "installed", False)),
            "module": getattr(det, "module_name", None),
            "version": getattr(det, "version", None),
        },
        "license": {
            "subject": getattr(core, "license_subject", None),
            "expires": getattr(core, "license_expires", None),
            "signature_valid": getattr(core, "license_signature_valid", None),
            "path": getattr(core, "license_path", None),
        },
        "eligible": bool(elig.eligible),
        "status": elig.status.value if hasattr(elig.status, "value") else str(elig.status),
        "reason": elig.reason,
        "premium_patches_enabled": int(getattr(core, "premium_patches_enabled", 0) or 0),
        "engine_tier_patches": _engine_tier_patch_count(),
    }
