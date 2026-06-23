# SPDX-License-Identifier: Apache-2.0
"""TDD for W-A — community lifecycle extension (community-test/-dev/-prod).

Goal: enable community-submitted model configs to flow through a
verified safety gate before promoting to community-prod tier. Operator
remains gate-keeper but the schema + CLI enforce the verification
contract automatically.

Lifecycle states (extended set):
  - experimental   → under active dev, not bench-validated
  - tested         → QA/regression-only, NOT recommended for prod
  - stable         → bench-validated, production-ready (built-in tier)
  - deprecated     → outgoing, kept for migration only
  - community-test → JUST submitted, awaiting initial verification
  - community-dev  → verified once on submitter's rig, awaiting cross-rig
  - community-prod → cross-verified ≥2 rigs, ≥7 days stable in community-test/-dev

Promotion requires:
  community-test → community-dev: 1 successful `genesis model-config verify`
  community-dev → community-prod: ≥2 verified_by entries + reference_metrics
                                  set + ≥7 days since test_started_at
"""
from __future__ import annotations

import pytest

from sndr.model_configs.schema import (
    ModelConfig,
    HardwareSpec,
    SchemaError,
)


def _minimal(**kwargs) -> ModelConfig:
    """Build a minimal valid ModelConfig with overridable fields."""
    base = dict(
        key="test-config",
        title="Test config",
        description="Minimal test",
        schema_version=1,
        maintainer="testuser",
        model_path="/models/Test",
        hardware=HardwareSpec(
            gpu_match_keys=["rtx a5000"], n_gpus=2, min_vram_per_gpu_mib=20000,
        ),
    )
    base.update(kwargs)
    return ModelConfig(**base)


class TestNewLifecycleStates:
    """Extended enum must accept community-* states."""

    def test_community_test_accepted(self):
        cfg = _minimal(lifecycle="community-test")
        cfg.validate()  # should not raise

    def test_community_dev_accepted(self):
        cfg = _minimal(lifecycle="community-dev")
        cfg.validate()

    def test_community_prod_accepted(self):
        cfg = _minimal(lifecycle="community-prod")
        # community-prod requires verified_by ≥2 + reference_metrics
        cfg.community_submitted = True
        cfg.verified_by = [
            "rtx-a5000@sandermage-2026-05-06",
            "rtx-3090@noonghunna-2026-05-08",
        ]
        from sndr.model_configs.schema import ReferenceMetrics
        cfg.reference_metrics = ReferenceMetrics(
            measured_at="2026-05-06", bench_method="genesis_bench_suite",
            long_gen_sustained_tps=170.0, long_gen_mean_lat_s=5.5,
            tool_call_score="10/10", stability_mean_s=1.5, stability_cv_pct=2.0,
            vram_used_mib_per_gpu=[22000, 22000], vram_total_mib=44000,
            genesis_pin="abc1234", vllm_pin="0.20.2rc1.dev9+g01d4d1ad3",
        )
        cfg.test_started_at = "2026-04-20"  # >7 days before today
        cfg.validate()

    def test_unknown_lifecycle_rejected(self):
        cfg = _minimal(lifecycle="bogus-state")
        with pytest.raises(SchemaError, match="lifecycle"):
            cfg.validate()


class TestNewFields:
    """New schema fields: community_submitted, verified_by, test_started_at."""

    def test_default_community_submitted_false(self):
        cfg = _minimal()
        assert cfg.community_submitted is False

    def test_default_verified_by_empty_list(self):
        cfg = _minimal()
        assert cfg.verified_by == []

    def test_default_test_started_at_none(self):
        cfg = _minimal()
        assert cfg.test_started_at is None

    def test_verified_by_accepts_list_of_strings(self):
        cfg = _minimal(
            lifecycle="community-dev",
            community_submitted=True,
            verified_by=["rtx-a5000@user1-2026-05-06"],
        )
        cfg.validate()


class TestPromotionGates:
    """community-prod requires verified_by ≥2 + reference_metrics + age."""

    def test_community_prod_without_reference_metrics_rejected(self):
        cfg = _minimal(
            lifecycle="community-prod",
            community_submitted=True,
            verified_by=["rig1", "rig2"],
            test_started_at="2026-04-20",
            reference_metrics=None,  # missing
        )
        with pytest.raises(SchemaError, match="reference_metrics"):
            cfg.validate()

    def test_community_prod_with_one_verifier_rejected(self):
        from sndr.model_configs.schema import ReferenceMetrics
        cfg = _minimal(
            lifecycle="community-prod",
            community_submitted=True,
            verified_by=["rig1"],  # only one
            test_started_at="2026-04-20",
            reference_metrics=ReferenceMetrics(
                measured_at="2026-05-06", bench_method="genesis_bench_suite",
                long_gen_sustained_tps=170.0, long_gen_mean_lat_s=5.5,
                tool_call_score="10/10", stability_mean_s=1.5, stability_cv_pct=2.0,
                vram_used_mib_per_gpu=[22000, 22000], vram_total_mib=44000,
                genesis_pin="abc1234", vllm_pin="0.20.2rc1.dev9+g01d4d1ad3",
            ),
        )
        with pytest.raises(SchemaError, match="verified_by"):
            cfg.validate()

    def test_community_submitted_without_community_lifecycle_rejected(self):
        cfg = _minimal(
            lifecycle="stable",
            community_submitted=True,  # mismatched
        )
        with pytest.raises(SchemaError, match="community"):
            cfg.validate()


class TestExistingConfigsStillValid:
    """All 8 existing builtin configs must still validate after schema extension."""

    def test_a5000_2x_35b_prod(self):
        # Phase 10 (2026-06-01): migrated from V1 registry.get() to V2
        # load_alias() — V1 file scheduled for sunset, V2 alias
        # composes byte-identical ModelConfig (TRANSPARENT bucket per
        # migration table). cfg.validate() asserts schema parity.
        from sndr.model_configs.registry_v2 import load_alias
        cfg = load_alias("prod-qwen3.6-35b-balanced")
        assert cfg is not None
        cfg.validate()  # must still pass

    def test_a5000_2x_27b_int4_tq_k8v4(self):
        # Phase 10 migration: V2 alias replaces V1 key (TRANSPARENT bucket).
        from sndr.model_configs.registry_v2 import load_alias
        cfg = load_alias("prod-qwen3.6-27b-tq-k8v4")
        assert cfg is not None
        cfg.validate()
