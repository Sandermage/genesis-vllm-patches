# SPDX-License-Identifier: Apache-2.0
"""Tests for the aggregated environment/readiness doctor."""
from __future__ import annotations

from dataclasses import asdict

from sndr.product_api.legacy.doctor import DoctorFinding, collect_doctor_report


def test_collect_doctor_report_aggregates_categories():
    report = collect_doctor_report()

    assert report.findings, "doctor should produce findings"
    assert all(isinstance(f, DoctorFinding) for f in report.findings)
    severities = {f.severity for f in report.findings}
    assert severities <= {"ok", "info", "warning", "blocked"}

    categories = {f.category for f in report.findings}
    # Real, always-present aggregation sources.
    assert "environment" in categories
    assert "runtime" in categories
    assert "catalog" in categories
    assert "patches" in categories

    # Summary counts cover every finding.
    assert sum(report.summary.values()) == len(report.findings)
    assert report.summary["ok"] >= 0

    # JSON-safe.
    payload = asdict(report)
    assert isinstance(payload["findings"], (list, tuple))
    assert payload["findings"][0]["title"]


def test_collect_doctor_report_is_torch_free_shape():
    report = collect_doctor_report()
    # Each finding has the operator-facing contract fields.
    for finding in report.findings:
        assert finding.category and finding.id and finding.title
        assert finding.severity in {"ok", "info", "warning", "blocked"}
