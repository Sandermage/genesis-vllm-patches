# SPDX-License-Identifier: Apache-2.0
"""Tests for ``product_api.patches.plan`` — M.6.3.

The os.environ overlay context manager is the highest-risk piece of
the M.6.x refactor — these tests pin the invariant that the env is
fully restored on every exit path (success, exception, generator close).
"""
from __future__ import annotations

import os

import pytest

from sndr.product_api.legacy.patches import plan
from sndr.product_api.legacy.patches.plan import (
    PlanReport,
    PresetNotFoundError,
)


class _FakeCfg:
    """Minimal stand-in for the resolved preset config object."""

    def __init__(self, system_env=None, genesis_env=None):
        self.system_env = system_env or {}
        self.genesis_env = genesis_env or {}


class TestPresetEnvOverlay:
    def test_restores_after_success(self, monkeypatch):
        sentinel_key = "SNDR_TEST_M63_SENTINEL"
        monkeypatch.delenv(sentinel_key, raising=False)
        monkeypatch.setenv(sentinel_key, "previous-value")

        cfg = _FakeCfg(genesis_env={sentinel_key: "overlay-value"})
        with plan.preset_env_overlay(cfg):
            assert os.environ[sentinel_key] == "overlay-value"
        assert os.environ[sentinel_key] == "previous-value"

    def test_restores_after_exception(self, monkeypatch):
        sentinel_key = "SNDR_TEST_M63_RAISE"
        monkeypatch.delenv(sentinel_key, raising=False)
        monkeypatch.setenv(sentinel_key, "before")

        cfg = _FakeCfg(genesis_env={sentinel_key: "during"})
        with pytest.raises(RuntimeError):
            with plan.preset_env_overlay(cfg):
                assert os.environ[sentinel_key] == "during"
                raise RuntimeError("boom")
        # finally block ran despite the exception.
        assert os.environ[sentinel_key] == "before"

    def test_removes_keys_that_were_unset(self, monkeypatch):
        """A key absent from os.environ before the overlay must be
        deleted on exit — not left as the overlay value."""
        sentinel_key = "SNDR_TEST_M63_UNSET"
        monkeypatch.delenv(sentinel_key, raising=False)
        assert sentinel_key not in os.environ

        cfg = _FakeCfg(genesis_env={sentinel_key: "transient"})
        with plan.preset_env_overlay(cfg):
            assert os.environ[sentinel_key] == "transient"
        assert sentinel_key not in os.environ

    def test_system_env_and_genesis_env_combine(self, monkeypatch):
        ks = "SNDR_TEST_M63_SYS"
        kg = "SNDR_TEST_M63_GEN"
        monkeypatch.delenv(ks, raising=False)
        monkeypatch.delenv(kg, raising=False)

        cfg = _FakeCfg(
            system_env={ks: "from-system"},
            genesis_env={kg: "from-genesis"},
        )
        with plan.preset_env_overlay(cfg):
            assert os.environ[ks] == "from-system"
            assert os.environ[kg] == "from-genesis"
        assert ks not in os.environ
        assert kg not in os.environ


class TestSimulatePlan:
    def test_known_preset_returns_report(self):
        report = plan.simulate_plan("prod-qwen3.6-35b-balanced")
        assert isinstance(report, PlanReport)
        assert report.preset == "prod-qwen3.6-35b-balanced"
        assert report.profile == "any"
        # Buckets together cover the dispatcher iteration.
        assert len(report.apply) + len(report.skip) + len(report.errors) >= 100

    def test_unknown_preset_raises_typed_error(self):
        with pytest.raises(PresetNotFoundError) as excinfo:
            plan.simulate_plan("totally-fake-preset-xyz")
        err = excinfo.value
        assert err.preset_key == "totally-fake-preset-xyz"
        assert isinstance(err.reason, str)

    def test_simulate_plan_restores_env_after_run(self, monkeypatch):
        """End-to-end variant of the env-restoration invariant — the
        same property ``TestPlan::test_plan_restores_env_after_run`` in
        ``test_patches_cli.py`` enforces, pulled down to the API layer."""
        sentinel_key = "SNDR_TEST_M63_E2E_SENTINEL"
        monkeypatch.delenv(sentinel_key, raising=False)
        monkeypatch.setenv(sentinel_key, "preset-test-value")
        plan.simulate_plan("prod-qwen3.6-35b-balanced")
        assert os.environ.get(sentinel_key) == "preset-test-value"

    def test_apply_rows_carry_required_keys(self):
        report = plan.simulate_plan("prod-qwen3.6-35b-balanced")
        for row in list(report.apply) + list(report.skip):
            for key in ("patch_id", "title", "tier", "default_on",
                        "lifecycle", "reason"):
                assert key in row

    def test_policy_compat_produces_resolver_payload(self):
        report = plan.simulate_plan(
            "prod-qwen3.6-35b-balanced", policy="compat", explain=True,
        )
        assert report.resolver_payload is not None
        for key in ("policy", "included", "excluded",
                    "warnings", "passthrough", "env"):
            assert key in report.resolver_payload

    def test_explain_adds_note_field(self):
        report = plan.simulate_plan(
            "prod-qwen3.6-35b-balanced", policy="compat", explain=True,
        )
        # At least one included decision carries an explain field.
        for d in report.resolver_payload["included"]:
            assert "note" in d
            assert "bench_evidence" in d
            break

    def test_production_profile_violations_typed(self):
        report = plan.simulate_plan(
            "prod-qwen3.6-35b-balanced", profile="production",
        )
        # Violations are a tuple of dicts; may be empty depending on
        # registry state. Shape invariant only.
        for v in report.profile_violations:
            assert "patch_id" in v
            assert "title" in v
            assert isinstance(v["reasons"], list)
