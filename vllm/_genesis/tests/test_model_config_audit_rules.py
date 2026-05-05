# SPDX-License-Identifier: Apache-2.0
"""TDD for audit_rules.py — exhaustive 16-rule database."""
from __future__ import annotations

import pytest

from vllm._genesis.model_configs import (
    ModelConfig, HardwareSpec, SpecDecodeConfig,
)
from vllm._genesis.model_configs.audit_rules import audit, RULES


def _base_cfg(**overrides):
    """Minimal ModelConfig for testing audit rules."""
    defaults = dict(
        key="test-cfg", title="Test", description="Test",
        schema_version=1, maintainer="test",
        model_path="/models/Qwen3.6-35B-A3B-FP8",
        hardware=HardwareSpec(gpu_match_keys=["rtx a5000"], n_gpus=2,
                              min_vram_per_gpu_mib=22000),
        kv_cache_dtype=None,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
        max_num_seqs=2,
        spec_decode=None,
        genesis_env={},
        vllm_pin_required=None,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


class TestRulesDB:
    def test_at_least_16_rules(self):
        assert len(RULES) >= 16

    def test_all_rules_have_id_and_title(self):
        for r in RULES:
            assert r.rule_id and r.rule_id.startswith("R-")
            assert r.title


# ─── R-001: P98 for TQ k8v4 + hybrid ──────────────────────────────────


class TestR001:
    def test_hybrid_tq_without_p98_flagged(self):
        cfg = _base_cfg(
            model_path="/models/Qwen3.6-27B-int4-AutoRound",
            kv_cache_dtype="turboquant_k8v4",
            genesis_env={},
        )
        issues = audit(cfg)
        assert any("R-001" in i[0] for i in issues)

    def test_hybrid_tq_with_p98_clean(self):
        cfg = _base_cfg(
            model_path="/models/Qwen3.6-27B-int4-AutoRound",
            kv_cache_dtype="turboquant_k8v4",
            genesis_env={"GENESIS_ENABLE_P98": "1"},
        )
        issues = audit(cfg)
        assert not any("R-001" in i[0] for i in issues)

    def test_dense_moe_fp8_not_flagged(self):
        # 35B-A3B-FP8 is dense MoE, not hybrid GDN — R-001 should not fire
        cfg = _base_cfg(
            model_path="/models/Qwen3.6-35B-A3B-FP8",
            kv_cache_dtype="turboquant_k8v4",
            genesis_env={},
        )
        issues = audit(cfg)
        assert not any("R-001" in i[0] for i in issues)


# ─── R-005: PN59 for long-ctx hybrid ──────────────────────────────────


class TestR005:
    def test_long_ctx_hybrid_without_pn59_flagged(self):
        cfg = _base_cfg(
            model_path="/models/Qwen3.6-27B-int4-AutoRound",
            max_model_len=131072,
        )
        issues = audit(cfg)
        assert any("R-005" in i[0] for i in issues)

    def test_short_ctx_hybrid_clean(self):
        cfg = _base_cfg(
            model_path="/models/Qwen3.6-27B-int4-AutoRound",
            max_model_len=32768,
        )
        issues = audit(cfg)
        assert not any("R-005" in i[0] for i in issues)


# ─── R-009: prefix-caching DANGER ─────────────────────────────────────


class TestR009:
    def test_prefix_caching_on_hybrid_tq_blocks(self):
        cfg = _base_cfg(
            model_path="/models/Qwen3.6-27B-int4-AutoRound",
            kv_cache_dtype="turboquant_k8v4",
            genesis_env={"GENESIS_ENABLE_P98": "1"},
        )
        cfg.vllm_extra_args = ["--enable-prefix-caching"]
        issues = audit(cfg)
        flagged = [i for i in issues if i[0] == "R-009"]
        assert len(flagged) == 1
        assert flagged[0][1] == "error"  # severity error


# ─── R-010: 27B + TQ + cudagraph FULL ─────────────────────────────────


class TestR010:
    def test_27b_tq_cudagraph_full_blocked(self):
        cfg = _base_cfg(
            model_path="/models/Qwen3.6-27B-int4-AutoRound",
            kv_cache_dtype="turboquant_k8v4",
            genesis_env={"GENESIS_ENABLE_P98": "1"},
        )
        # No --enforce-eager AND no PIECEWISE → blocks
        cfg.enforce_eager = False
        cfg.vllm_extra_args = []
        issues = audit(cfg)
        assert any(i[0] == "R-010" for i in issues)

    def test_27b_tq_with_piecewise_clean(self):
        cfg = _base_cfg(
            model_path="/models/Qwen3.6-27B-int4-AutoRound",
            kv_cache_dtype="turboquant_k8v4",
            genesis_env={"GENESIS_ENABLE_P98": "1"},
        )
        cfg.enforce_eager = False
        cfg.vllm_extra_args = ['--compilation-config {"cudagraph_mode":"PIECEWISE"}']
        issues = audit(cfg)
        assert not any(i[0] == "R-010" for i in issues)


# ─── R-011: typo in env name ──────────────────────────────────────────


class TestR011:
    def test_typo_env_flagged(self):
        cfg = _base_cfg(
            genesis_env={"GENESIS_ENABLE_PXX9": "1"},  # fake patch
        )
        issues = audit(cfg)
        flagged = [i for i in issues if i[0] == "R-011"]
        assert len(flagged) == 1
        assert "PXX9" in flagged[0][3]

    def test_known_env_clean(self):
        cfg = _base_cfg(
            genesis_env={
                "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL": "1",
                "GENESIS_P67_NUM_KV_SPLITS": "32",  # tunable, allow
            },
        )
        issues = audit(cfg)
        assert not any(i[0] == "R-011" for i in issues)


# ─── R-013: vllm_pin must be in allowlist ─────────────────────────────


class TestR013:
    def test_unknown_pin_flagged(self):
        cfg = _base_cfg(vllm_pin_required="0.99.99-fake")
        issues = audit(cfg)
        assert any(i[0] == "R-013" for i in issues)

    def test_known_pin_clean(self):
        cfg = _base_cfg(vllm_pin_required="0.20.2rc1.dev9+g01d4d1ad3")
        issues = audit(cfg)
        assert not any(i[0] == "R-013" for i in issues)


# ─── Smoke: builtin configs are clean ─────────────────────────────────


class TestBuiltinConfigsClean:
    def test_a5000_2x_35b_prod_audit_clean_or_info_only(self):
        from vllm._genesis.model_configs import get
        cfg = get("a5000-2x-35b-prod")
        assert cfg is not None
        issues = audit(cfg)
        # No errors; warnings allowed
        errors = [i for i in issues if i[1] == "error"]
        assert errors == [], f"Builtin 35B config has errors: {errors}"

    def test_a5000_2x_27b_int4_balanced_no_errors(self):
        from vllm._genesis.model_configs import get
        cfg = get("a5000-2x-27b-int4-balanced")
        issues = audit(cfg)
        errors = [i for i in issues if i[1] == "error"]
        assert errors == [], f"Builtin 27B config has errors: {errors}"
