# SPDX-License-Identifier: Apache-2.0
"""Tests for ``product_api.patches.pn95`` — M.6.3."""
from __future__ import annotations

import json

from sndr.product_api.legacy.patches import pn95
from sndr.product_api.legacy.patches.pn95 import Pn95Report


def _write_stats(path, **overrides):
    base = {
        "ticks_total": 100,
        "ticks_pressure_check": 5,
        "ticks_demote_triggered": 1,
        "blocks_demoted_total": 8,
        "blocks_promoted_total": 4,
        "last_free_mib": 1000,
        "prefix_store_entries": 3,
        "prefix_store_promote_hits": 0,
    }
    base.update(overrides)
    path.write_text(json.dumps(base), encoding="utf-8")
    return path


class TestReadPn95Status:
    def test_missing_file_returns_unavailable(self, tmp_path):
        report = pn95.read_pn95_status(str(tmp_path / "absent.json"))
        assert isinstance(report, Pn95Report)
        assert report.available is False
        assert report.parse_error is False
        assert "stats file not found" in report.reason

    def test_parse_error_returns_unavailable_with_flag(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{ not json }", encoding="utf-8")
        report = pn95.read_pn95_status(str(bad))
        assert report.available is False
        assert report.parse_error is True
        assert report.reason.startswith("parse error:")

    def test_success_returns_stats_and_hints(self, tmp_path):
        path = _write_stats(tmp_path / "stats.json")
        report = pn95.read_pn95_status(str(path))
        assert report.available is True
        assert report.parse_error is False
        assert report.stats["ticks_total"] == 100
        # ``hints`` is a tuple of dicts (severity/message).
        for h in report.hints:
            assert "severity" in h
            assert "message" in h


class TestHintPredicates:
    def test_zero_ticks_warn_fires(self, tmp_path):
        path = _write_stats(tmp_path / "stats.json", ticks_total=0)
        report = pn95.read_pn95_status(str(path))
        severities = [h["severity"] for h in report.hints]
        messages = [h["message"] for h in report.hints]
        assert "warn" in severities
        assert any("Zero scheduler ticks" in m for m in messages)

    def test_promote_hits_ok_fires(self, tmp_path):
        path = _write_stats(
            tmp_path / "stats.json", prefix_store_promote_hits=42,
        )
        report = pn95.read_pn95_status(str(path))
        severities = [h["severity"] for h in report.hints]
        assert "ok" in severities

    def test_low_free_mib_warn_fires(self, tmp_path):
        path = _write_stats(tmp_path / "stats.json", last_free_mib=50)
        report = pn95.read_pn95_status(str(path))
        messages = [h["message"] for h in report.hints]
        assert any("GPU free memory below 200 MiB" in m for m in messages)

    def test_missing_optional_keys_tolerated(self, tmp_path):
        """``KeyError`` / ``TypeError`` from incomplete stats must be
        swallowed — the predicate evaluator should never raise."""
        path = tmp_path / "partial.json"
        # Minimal subset; many predicate keys missing.
        path.write_text(json.dumps({"ticks_total": 0}), encoding="utf-8")
        report = pn95.read_pn95_status(str(path))
        # The "Zero scheduler ticks" predicate still fires; others
        # silently skip because of KeyError on missing keys.
        assert report.available is True
        assert any("Zero scheduler ticks" in h["message"] for h in report.hints)


class TestDiskTierBestEffort:
    def test_disk_tier_field_always_dict(self, tmp_path):
        path = _write_stats(tmp_path / "stats.json")
        report = pn95.read_pn95_status(str(path))
        assert isinstance(report.disk_tier, dict)
