# SPDX-License-Identifier: Apache-2.0
"""Tests for ``product_api.patches.doctor`` — M.6.1."""
from __future__ import annotations

from sndr.product_api.legacy.patches import doctor
from sndr.product_api.legacy.patches.types import DoctorReport


class TestRunDoctor:
    def test_returns_report(self):
        report = doctor.run_doctor()
        assert isinstance(report, DoctorReport)
        assert report.registry_size >= 100

    def test_coverage_totals_match_registry_size(self):
        report = doctor.run_doctor()
        # Coverage.total counts every registry entry.
        assert report.coverage.total == report.registry_size

    def test_issues_is_tuple(self):
        """Frozen-dataclass invariant: issues is an immutable tuple."""
        report = doctor.run_doctor()
        assert isinstance(report.issues, tuple)
        for issue in report.issues:
            assert hasattr(issue, "severity")
            assert hasattr(issue, "patch_id")
            assert hasattr(issue, "message")

    def test_unmapped_lists_are_sequences(self):
        report = doctor.run_doctor()
        assert hasattr(report.coverage, "unmapped")
        assert hasattr(report.coverage, "intentionally_unmapped")
