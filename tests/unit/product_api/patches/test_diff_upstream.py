# SPDX-License-Identifier: Apache-2.0
"""Tests for ``product_api.patches.diff_upstream`` — M.6.1."""
from __future__ import annotations

from sndr.product_api.legacy.patches import diff_upstream
from sndr.product_api.legacy.patches.types import DiffReport


class TestDiffUpstream:
    def test_returns_diff_report(self):
        report = diff_upstream.diff_upstream()
        assert isinstance(report, DiffReport)

    def test_two_buckets_immutable(self):
        report = diff_upstream.diff_upstream()
        assert isinstance(report.merged_upstream, tuple)
        assert isinstance(report.has_upstream_pr, tuple)

    def test_merged_upstream_entry_shape(self):
        report = diff_upstream.diff_upstream()
        for entry in report.merged_upstream:
            assert "patch_id" in entry
            assert "title" in entry
            assert "upstream_pr" in entry
            assert "credit" in entry

    def test_has_upstream_pr_entry_shape(self):
        report = diff_upstream.diff_upstream()
        for entry in report.has_upstream_pr:
            assert "patch_id" in entry
            assert "title" in entry
            assert "upstream_pr" in entry
            assert "lifecycle" in entry
            assert "default_on" in entry
            assert entry["upstream_pr"] is not None

    def test_no_duplicate_entries_across_buckets(self):
        """A patch in ``merged_upstream`` must not also be in
        ``has_upstream_pr``; the buckets are disjoint by construction."""
        report = diff_upstream.diff_upstream()
        merged_ids = {e["patch_id"] for e in report.merged_upstream}
        active_ids = {e["patch_id"] for e in report.has_upstream_pr}
        assert merged_ids & active_ids == set()
