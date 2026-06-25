# SPDX-License-Identifier: Apache-2.0
"""Tests for ``product_api.patches.retire_impact_status`` — surfaces the
anchor-SoT retire-impact detector for the GUI."""
from __future__ import annotations

from sndr.product_api.legacy.patches import retire_impact_status


class TestRetireImpactStatus:
    def test_shape_on_live_registry(self):
        d = retire_impact_status.retire_impact_status()
        assert isinstance(d, dict)
        assert {"high_count", "medium_count", "edges"} <= set(d)
        assert isinstance(d["edges"], list)
        assert isinstance(d["high_count"], int)
        assert isinstance(d["medium_count"], int)

    def test_edge_fields_when_present(self):
        d = retire_impact_status.retire_impact_status()
        for e in d["edges"]:
            assert {"retired", "dependent", "severity", "via"} <= set(e)
            assert e["severity"] in ("HIGH", "MEDIUM")
            # ``via`` is a sequence (tuple in-process; JSON array on the wire).
            assert isinstance(e["via"], (list, tuple))

    def test_fail_safe_on_detector_error(self, monkeypatch):
        # A detector explosion must degrade to an empty report, never raise.
        import sndr.engines.vllm.retire_impact as ri

        monkeypatch.setattr(
            ri, "detect_on_live_registry",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        d = retire_impact_status.retire_impact_status()
        assert d["edges"] == [] and d["high_count"] == 0
        assert d.get("error") == "RuntimeError"
