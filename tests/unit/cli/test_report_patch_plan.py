# SPDX-License-Identifier: Apache-2.0
"""`sndr report bundle` — patch_plan.json artifact.

Phase D extension: when ``--preset`` is provided to ``sndr report
bundle``, the bundle includes a new ``patch_plan.json`` artifact
that captures the resolver output for compat / safe / minimal
policies side-by-side. Reviewers seeing a bundle three weeks later
know exactly what would have been launched under any policy
without re-running the resolver.

Tests target the collector function directly — bundling itself is
covered by other report tests; we only verify the new artifact
shape + presence.
"""
from __future__ import annotations

import json

import pytest

from sndr.cli.legacy.report import _collect_patch_plan


class TestCollectorShape:
    def test_returns_none_when_no_preset(self):
        """Bundle without --preset has no patch_plan to collect."""
        assert _collect_patch_plan(None) is None

    def test_returns_dict_keyed_by_policy_when_preset_given(self):
        out = _collect_patch_plan("prod-qwen3.6-35b-balanced")
        assert out is not None
        assert isinstance(out, dict)
        assert out["preset"] == "prod-qwen3.6-35b-balanced"
        plans = out["plans"]
        for policy in ("compat", "safe", "minimal"):
            assert policy in plans
            p = plans[policy]
            for key in ("included_count", "excluded_count",
                        "passthrough_count", "warnings"):
                assert key in p

    def test_compat_minimal_excluded_count_grows(self):
        """Under minimal more toggles are excluded than under compat."""
        out = _collect_patch_plan("prod-qwen3.6-35b-balanced")
        assert out is not None
        plans = out["plans"]
        assert plans["minimal"]["excluded_count"] > plans["compat"]["excluded_count"]

    def test_unknown_preset_returns_error_marker(self):
        out = _collect_patch_plan("this-preset-does-not-exist")
        assert out is not None
        assert "error" in out


class TestArtifactRegistered:
    """The collector is registered under 'patch_plan.json' in
    _SCOPE_ARTIFACTS['all'] so a bundle without --scope picks it up."""

    def test_patch_plan_in_all_scope(self):
        from sndr.cli.legacy.report import _SCOPE_ARTIFACTS
        assert "patch_plan.json" in _SCOPE_ARTIFACTS["all"]

    def test_patch_plan_in_patches_scope(self):
        """The 'patches' scope (used when triaging patch issues) must
        carry patch_plan.json — it's the most useful piece for that
        triage flow."""
        from sndr.cli.legacy.report import _SCOPE_ARTIFACTS
        assert "patch_plan.json" in _SCOPE_ARTIFACTS["patches"]
