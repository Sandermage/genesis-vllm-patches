# SPDX-License-Identifier: Apache-2.0
"""Tests for G4_84 — generic MoE-geometry advisor + wna16 config-provider."""
from __future__ import annotations

import importlib

import pytest

MOD = "sndr.engines.vllm.patches.moe.g4_84_moe_geometry_advisor"


@pytest.fixture
def g4_84():
    return importlib.import_module(MOD)


class TestMarlinMarginalDetector:
    """marlin_moe_marginal mirrors vLLM check_moe_marlin_supports_layer:
    Marlin needs intermediate_per_partition % max(64, group_size) == 0."""

    def test_gemma4_26b_tp2_g32_is_ineligible(self, g4_84):
        # 26B: moe_intermediate=704, TP=2 -> 352; 352 % max(64,32)=64 = 32 != 0.
        assert g4_84.marlin_moe_marginal(352, 32) is True

    def test_g128_also_ineligible_at_352(self, g4_84):
        assert g4_84.marlin_moe_marginal(352, 128) is True

    def test_padded_384_is_eligible(self, g4_84):
        # G4_08 pads K 352->384; 384 % 128 == 0 -> Marlin-eligible.
        assert g4_84.marlin_moe_marginal(384, 128) is False

    def test_tp1_704_g64_is_eligible(self, g4_84):
        # TP=1: per-shard intermediate 704; 704 % max(64,64)=64 == 0 -> eligible.
        assert g4_84.marlin_moe_marginal(704, 64) is False

    def test_tp1_704_g128_still_ineligible(self, g4_84):
        # 704 % 128 = 64 != 0 -> ineligible even at TP=1 with g128.
        assert g4_84.marlin_moe_marginal(704, 128) is True

    def test_group_size_zero_treated_as_64(self, g4_84):
        # group_size <= 0 (per-channel) -> divisor max(64, .) = 64.
        assert g4_84.marlin_moe_marginal(512, 0) is False  # 512 % 64 == 0
        assert g4_84.marlin_moe_marginal(352, 0) is True   # 352 % 64 != 0


class TestProviderTableShape:
    def test_config_table_is_dict_fail_open(self, g4_84):
        # Empty/extensible table; unknown shapes fall through (fail-open).
        assert isinstance(g4_84._GENESIS_MOE_WNA16_CONFIGS, dict)

    def test_has_apply_contract(self, g4_84):
        for fn in ("apply", "is_applied", "revert"):
            assert callable(getattr(g4_84, fn))

    def test_marker_present(self, g4_84):
        assert "G4_84" in g4_84.GENESIS_G4_84_MARKER
