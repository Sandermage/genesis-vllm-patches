# SPDX-License-Identifier: Apache-2.0
"""Tests for the daemon license / sndr_engine status bridge."""
from sndr.product_api.legacy import licensing as L


def test_status_shape():
    s = L.status()
    assert s["available"] is True
    # Always-present keys the GUI relies on.
    for key in ("core", "tier", "engine", "license", "eligible", "status", "reason",
                "premium_patches_enabled", "engine_tier_patches"):
        assert key in s
    assert isinstance(s["engine"], dict) and "installed" in s["engine"]
    assert isinstance(s["license"], dict) and "subject" in s["license"]
    assert isinstance(s["eligible"], bool)
    assert isinstance(s["premium_patches_enabled"], int)


def test_community_default_without_engine():
    # In the public checkout sndr_engine isn't installed → community, not eligible.
    s = L.status()
    if not s["engine"]["installed"]:
        assert s["eligible"] is False
        assert s["tier"] == "community"
        assert s["status"] in ("no_package", "no_key", "expired", "version_mismatch",
                               "bad_signature", "bad_payload")
