# SPDX-License-Identifier: Apache-2.0
"""CONFIG-UX.4.1 — tests for `_rollout.py` stage + severity helper.

Covers the locked matrix from CONFIG_UX_4_R §2.2:

  bucket                    | s0   | s1   | s2 default | s2 strict | s3+
  ------------------------- + ---- + ---- + ---------- + --------- + ----
  transparent               | warn | warn | warn       | warn      | warn
  needs_operator_choice     | warn | warn | warn       | error     | error
  deprecated                | warn | warn | warn       | error     | error
  tombstone                 | error|error | error      | error     | error
  card_less_prod            | warn | warn | warn       | error     | error
  card_less_non_prod        | info | info | info       | info      | info
  missing_override_policy   | warn | warn | warn       | error     | error

Plus env-parsing edge cases (invalid string, out-of-range, empty).
"""
from __future__ import annotations

import pytest

from sndr.model_configs._rollout import (
    BUCKETS,
    DEFAULT_STAGE,
    SEVERITIES,
    effective_severity,
    is_disabled,
    rollout_stage,
)


# ─── Env parsing ────────────────────────────────────────────────────────────


class TestRolloutStageEnv:
    def test_empty_env_returns_default(self):
        assert rollout_stage(env_value="") == DEFAULT_STAGE
        assert rollout_stage(env_value=None) == DEFAULT_STAGE  # reads real env

    def test_valid_stages(self):
        assert rollout_stage(env_value="0") == 0
        assert rollout_stage(env_value="1") == 1
        assert rollout_stage(env_value="2") == 2
        assert rollout_stage(env_value="3") == 3

    @pytest.mark.parametrize("bad", ["4", "99", "-1", "abc", "1.5", "0x1"])
    def test_invalid_stage_falls_back_to_default(self, bad):
        # Squelch the one-time warning that the helper emits.
        with pytest.warns(UserWarning, match="SNDR_V1_ROLLOUT_STAGE"):
            stage = rollout_stage(env_value=bad)
        assert stage == DEFAULT_STAGE

    def test_default_stage_current_contract(self):
        """Operator guard: CONFIG-UX.4.2 (2026-05-24) flipped default
        stage 0 → 1. Stage 0 and Stage 1 produce functionally identical
        observable severity for non-tombstone buckets (verified by the
        TestEffectiveSeverityMatrix class), so the flip is a no-op for
        operators staying at the default while preparing the source
        tree for Stage 2/3 escalation in CONFIG-UX.4.3."""
        assert DEFAULT_STAGE == 1


# ─── effective_severity matrix ──────────────────────────────────────────────


class TestEffectiveSeverityMatrix:
    @pytest.mark.parametrize("stage", [0, 1, 2, 3])
    def test_tombstone_always_error(self, stage):
        assert effective_severity(bucket="tombstone", stage=stage) == "error"
        assert effective_severity(bucket="tombstone", stage=stage, strict_mode=True) == "error"

    @pytest.mark.parametrize("stage", [0, 1, 2, 3])
    def test_transparent_always_warn(self, stage):
        assert effective_severity(bucket="transparent", stage=stage) == "warn"
        assert effective_severity(bucket="transparent", stage=stage, strict_mode=True) == "warn"

    @pytest.mark.parametrize("stage", [0, 1, 2, 3])
    def test_card_less_non_prod_always_info(self, stage):
        """Operator decision §10.3: non-prod card-less stays INFO forever."""
        assert effective_severity(bucket="card_less_non_prod", stage=stage) == "info"
        assert effective_severity(bucket="card_less_non_prod", stage=stage, strict_mode=True) == "info"

    @pytest.mark.parametrize("bucket", [
        "needs_operator_choice", "deprecated",
        "card_less_prod", "missing_override_policy",
    ])
    @pytest.mark.parametrize("stage", [0, 1])
    def test_escalating_bucket_warn_at_stage_0_1(self, bucket, stage):
        assert effective_severity(bucket=bucket, stage=stage) == "warn"
        assert effective_severity(bucket=bucket, stage=stage, strict_mode=True) == "warn"

    @pytest.mark.parametrize("bucket", [
        "needs_operator_choice", "deprecated",
        "card_less_prod", "missing_override_policy",
    ])
    def test_escalating_bucket_stage_2_default_vs_strict(self, bucket):
        assert effective_severity(bucket=bucket, stage=2) == "warn"
        assert effective_severity(bucket=bucket, stage=2, strict_mode=True) == "error"

    @pytest.mark.parametrize("bucket", [
        "needs_operator_choice", "deprecated",
        "card_less_prod", "missing_override_policy",
    ])
    def test_escalating_bucket_stage_3_always_error(self, bucket):
        assert effective_severity(bucket=bucket, stage=3) == "error"
        assert effective_severity(bucket=bucket, stage=3, strict_mode=True) == "error"


# ─── Stage = None reads env ────────────────────────────────────────────────


class TestStageFromEnv:
    def test_stage_none_reads_env(self, monkeypatch):
        monkeypatch.setenv("SNDR_V1_ROLLOUT_STAGE", "2")
        # Stage 2 + strict → error for deprecated bucket
        assert effective_severity(
            bucket="deprecated", strict_mode=True,
        ) == "error"
        # Stage 2 default (no strict) → warn
        assert effective_severity(bucket="deprecated") == "warn"

    def test_stage_none_no_env_defaults_to_zero(self, monkeypatch):
        monkeypatch.delenv("SNDR_V1_ROLLOUT_STAGE", raising=False)
        assert effective_severity(bucket="deprecated") == "warn"


# ─── is_disabled escape hatch ──────────────────────────────────────────────


class TestIsDisabled:
    def test_not_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("GENESIS_DISABLE_V1_DEPRECATION_WARNING", raising=False)
        assert is_disabled() is False

    def test_disabled_when_env_set(self, monkeypatch):
        monkeypatch.setenv("GENESIS_DISABLE_V1_DEPRECATION_WARNING", "1")
        assert is_disabled() is True

    def test_disabled_with_truthy_value(self, monkeypatch):
        monkeypatch.setenv("GENESIS_DISABLE_V1_DEPRECATION_WARNING", "yes")
        assert is_disabled() is True


# ─── Sanity: matrix exhaustive ──────────────────────────────────────────────


class TestMatrixExhaustive:
    @pytest.mark.parametrize("bucket", BUCKETS)
    @pytest.mark.parametrize("stage", [0, 1, 2, 3])
    @pytest.mark.parametrize("strict", [False, True])
    def test_severity_in_enum(self, bucket, stage, strict):
        """Every (bucket × stage × strict) tuple must produce a value
        from the SEVERITIES enum."""
        sev = effective_severity(bucket=bucket, stage=stage, strict_mode=strict)
        assert sev in SEVERITIES
