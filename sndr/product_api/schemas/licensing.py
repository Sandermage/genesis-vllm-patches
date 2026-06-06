# SPDX-License-Identifier: Apache-2.0
"""Pydantic schemas for licensing resources."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

LicenseTierStatus = Literal[
    "licensed",
    "licensed_legacy",
    "expired",
    "bad_signature",
    "version_mismatch",
    "no_key",
    "no_package",
    "unknown",
]


class LicenseStatus(BaseModel):
    """Current license status, suitable for display in the GUI."""

    status: LicenseTierStatus
    customer_id_hash: str | None = Field(
        default=None,
        description="First 8 hex chars of sha256(customer_id) — for correlation, not identity",
    )
    expires_at: datetime | None = None
    days_until_expiry: int | None = None
    engine_major: int | None = None
    engine_package_installed: bool = False
    engine_patches_available: int = Field(
        default=0,
        description="Number of patches discoverable via entry points",
    )
    message: str | None = None


__all__ = ["LicenseStatus", "LicenseTierStatus"]
