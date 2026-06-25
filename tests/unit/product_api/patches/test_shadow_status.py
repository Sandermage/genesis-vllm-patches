# SPDX-License-Identifier: Apache-2.0
"""Tests for ``product_api.patches.shadow_status`` — apply-order shadow diff."""
from __future__ import annotations

from sndr.product_api.legacy.patches import shadow_status


class TestShadowStatus:
    def test_shape(self):
        d = shadow_status.shadow_status()
        assert isinstance(d, dict)
        for key in ("legacy_count", "spec_count", "spec_boot_unsafe",
                    "spec_only_unexpected", "legacy_unparseable"):
            assert key in d
        assert isinstance(d["legacy_count"], int)
        assert isinstance(d["spec_boot_unsafe"], list)

    def test_fail_safe(self, monkeypatch):
        import sndr.apply.shadow as sh

        monkeypatch.setattr(
            sh, "compare_apply_orders",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        d = shadow_status.shadow_status()
        assert d["error"] == "RuntimeError"
        assert d["spec_boot_unsafe"] == [] and d["legacy_count"] == 0
