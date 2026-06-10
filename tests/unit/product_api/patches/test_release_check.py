# SPDX-License-Identifier: Apache-2.0
"""Tests for ``product_api.patches.release_check`` — M.6.2."""
from __future__ import annotations

import pytest

from sndr.product_api.legacy.patches import release_check
from sndr.product_api.legacy.patches.release_check import (
    ReleaseCheckResult,
)


class TestReleaseCheck:
    def test_report_mode_never_blocks(self, tmp_path):
        result = release_check.release_check(mode="report", out_dir=tmp_path)
        assert isinstance(result, ReleaseCheckResult)
        assert result.release_blocked is False

    def test_raw_dict_has_canonical_keys(self, tmp_path):
        result = release_check.release_check(mode="report", out_dir=tmp_path)
        for key in ("policy", "verdicts", "considered", "total",
                    "passed_count", "failed_count", "release_blocked"):
            assert key in result.raw

    def test_policy_passes_through(self, tmp_path):
        result = release_check.release_check(
            mode="require-static",
            out_dir=tmp_path,
            max_regression_pct=5.0,
        )
        pol = result.policy
        assert pol["mode"] == "require-static"
        assert pol["max_regression_pct"] == 5.0

    def test_invalid_mode_raises(self, tmp_path):
        from sndr.proof.release_check import ReleaseCheckError

        with pytest.raises(ReleaseCheckError):
            release_check.release_check(
                mode="not-a-mode", out_dir=tmp_path,
            )

    def test_patch_filter_subset(self, tmp_path):
        # Passing an explicit patch filter restricts the considered set.
        result = release_check.release_check(
            mode="report", out_dir=tmp_path, patch_filter=["P67"],
        )
        assert result.considered <= 1

    def test_production_subset_scope_expands_filter(self, tmp_path):
        # ``scope=production-subset`` widens the filter to the canonical
        # production patch set when no explicit ``patch_filter`` is given.
        result = release_check.release_check(
            mode="report", out_dir=tmp_path, scope="production-subset",
        )
        # ``patch_filter`` is recorded in the policy; non-zero entries.
        assert result.policy.get("patch_filter") not in (None, [])
