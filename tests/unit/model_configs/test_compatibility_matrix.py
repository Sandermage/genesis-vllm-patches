# SPDX-License-Identifier: Apache-2.0
"""Tests for `CompatibilityMatrix` in model_configs.schema.

Coverage:
  - CompatibilityRule structure (validate, severity enum).
  - Regression test for every registered rule (predicate fires on
    a canonical bad config and stays silent on a good one).
  - Integration with ModelConfig.validate — forbidden rules raise
    SchemaError, blocking the cfg from being constructed.
  - Integration with ModelConfig.audit — discouraged rules surface
    in the warnings list without aborting validation.
  - Duplicate id registration is rejected.
"""
from __future__ import annotations

import pytest

from vllm.sndr_core.model_configs.schema import (
    COMPATIBILITY_MATRIX,
    CompatibilityMatrix,
    CompatibilityRule,
    HardwareSpec,
    ModelConfig,
    SchemaError,
    SpecDecodeConfig,
)


# ─── CompatibilityRule structure ───────────────────────────────────────


class TestCompatibilityRule:
    def test_valid_rule_validates(self):
        r = CompatibilityRule(
            id="TEST-001", severity="forbidden",
            title="x", message="y", mitigation="z",
        )
        r.validate()  # no raise

    def test_missing_id_raises(self):
        r = CompatibilityRule(
            id="", severity="forbidden",
            title="x", message="y", mitigation="z",
        )
        with pytest.raises(SchemaError):
            r.validate()

    def test_bad_severity_raises(self):
        r = CompatibilityRule(
            id="X", severity="MAYBE",
            title="t", message="m", mitigation="g",
        )
        with pytest.raises(SchemaError):
            r.validate()

    @pytest.mark.parametrize("missing", ["title", "message", "mitigation"])
    def test_missing_text_field_raises(self, missing):
        kwargs = dict(id="X", severity="forbidden",
                      title="t", message="m", mitigation="g")
        kwargs[missing] = ""
        with pytest.raises(SchemaError):
            CompatibilityRule(**kwargs).validate()


# ─── CompatibilityMatrix bookkeeping ───────────────────────────────────


class TestMatrixBookkeeping:
    def test_register_duplicate_id_raises(self):
        m = CompatibilityMatrix()
        r1 = CompatibilityRule(
            id="DUP", severity="forbidden",
            title="t", message="m", mitigation="g",
        )
        m.register(r1, lambda c: False)
        with pytest.raises(SchemaError):
            m.register(r1, lambda c: True)

    def test_predicate_exception_is_swallowed(self):
        m = CompatibilityMatrix()
        m.register(
            CompatibilityRule(
                id="ERR", severity="forbidden",
                title="t", message="m", mitigation="g",
            ),
            lambda c: (_ for _ in ()).throw(RuntimeError("oops")),
        )
        # evaluate() must return empty lists, not raise
        forb, disc = m.evaluate(_minimal_cfg())
        assert forb == []
        assert disc == []

    def test_canonical_matrix_has_rules(self):
        rules = COMPATIBILITY_MATRIX.rules()
        ids = {r.id for r in rules}
        # Sanity floor: at least 4 rules registered by the schema layer
        for expected in ("COMPAT-001", "COMPAT-002",
                          "COMPAT-003", "COMPAT-004"):
            assert expected in ids


# ─── Predicate regression tests ────────────────────────────────────────


def _minimal_cfg(**overrides) -> ModelConfig:
    """Builder for a minimally-valid ModelConfig with defaults
    individual tests can override."""
    base = dict(
        key="test-cfg", title="t", description="d",
        schema_version=1, maintainer="x",
        model_path="/models/dense-fp16",
        hardware=HardwareSpec(
            gpu_match_keys=["test"], n_gpus=1,
            min_vram_per_gpu_mib=24576,
        ),
    )
    base.update(overrides)
    return ModelConfig(**base)


class TestRuleDFlashOnQwenNext:
    """COMPAT-001 — DFlash blocked for Qwen-next architecture only.

    Other hybrid-GDN models (Qwen3.6 Lorbus + PN59) work with DFlash
    when a separate drafter checkpoint is provided.
    """

    def test_triggers_for_qwen_next_path(self):
        cfg = _minimal_cfg(
            model_path="/models/qwen-next-30b",
            spec_decode=SpecDecodeConfig(
                method="dflash", num_speculative_tokens=4,
                model="/drafter/dflash",
            ),
        )
        forb, _ = COMPATIBILITY_MATRIX.evaluate(cfg)
        assert any(r.id == "COMPAT-001" for r, _m in forb)

    def test_triggers_for_qwen3_next_variant(self):
        cfg = _minimal_cfg(
            model_path="/models/Qwen3-next-30b-fp8",
            spec_decode=SpecDecodeConfig(
                method="dflash", num_speculative_tokens=4,
                model="/drafter/dflash",
            ),
        )
        forb, _ = COMPATIBILITY_MATRIX.evaluate(cfg)
        assert any(r.id == "COMPAT-001" for r, _m in forb)

    def test_not_triggered_for_qwen36_lorbus_dflash(self):
        """Existing PROD preset (a5000-2x-27b-dflash-true) — Qwen3.6 hybrid
        Mamba + DFlash drafter is a valid combination; rule should not fire."""
        cfg = _minimal_cfg(
            model_path="/models/Qwen3.6-27B-int4-AutoRound",
            spec_decode=SpecDecodeConfig(
                method="dflash", num_speculative_tokens=5,
                model="/models/Qwen3.6-27B-DFlash",
            ),
            genesis_env={"GENESIS_ENABLE_PN59_STREAMING_GDN": "1"},
        )
        forb, _ = COMPATIBILITY_MATRIX.evaluate(cfg)
        assert not any(r.id == "COMPAT-001" for r, _m in forb)

    def test_not_triggered_when_mtp_on_qwen_next(self):
        cfg = _minimal_cfg(
            model_path="/models/qwen-next-30b",
            spec_decode=SpecDecodeConfig(
                method="mtp", num_speculative_tokens=3,
            ),
        )
        forb, _ = COMPATIBILITY_MATRIX.evaluate(cfg)
        assert not any(r.id == "COMPAT-001" for r, _m in forb)


class TestRuleTqK8v4HybridNoP98:
    """COMPAT-002."""

    def test_triggers_when_tq_hybrid_no_p98(self):
        cfg = _minimal_cfg(
            kv_cache_dtype="turboquant_k8v4",
            genesis_env={"GENESIS_ENABLE_PN59_STREAMING_GDN": "1"},
        )
        _, disc = COMPATIBILITY_MATRIX.evaluate(cfg)
        assert any(r.id == "COMPAT-002" for r, _m in disc)

    def test_not_triggered_when_p98_enabled(self):
        cfg = _minimal_cfg(
            kv_cache_dtype="turboquant_k8v4",
            genesis_env={
                "GENESIS_ENABLE_PN59_STREAMING_GDN": "1",
                "GENESIS_ENABLE_P98": "1",
            },
        )
        _, disc = COMPATIBILITY_MATRIX.evaluate(cfg)
        assert not any(r.id == "COMPAT-002" for r, _m in disc)

    def test_not_triggered_on_dense_model(self):
        cfg = _minimal_cfg(kv_cache_dtype="turboquant_k8v4")
        _, disc = COMPATIBILITY_MATRIX.evaluate(cfg)
        assert not any(r.id == "COMPAT-002" for r, _m in disc)


class TestRuleNgramOnTqK8v4Long:
    """COMPAT-003."""

    def test_triggers_for_long_ctx(self):
        cfg = _minimal_cfg(
            spec_decode=SpecDecodeConfig(
                method="ngram", num_speculative_tokens=4,
            ),
            kv_cache_dtype="turboquant_k8v4",
            max_model_len=200_000,
        )
        _, disc = COMPATIBILITY_MATRIX.evaluate(cfg)
        assert any(r.id == "COMPAT-003" for r, _m in disc)

    def test_not_triggered_short_ctx(self):
        cfg = _minimal_cfg(
            spec_decode=SpecDecodeConfig(
                method="ngram", num_speculative_tokens=4,
            ),
            kv_cache_dtype="turboquant_k8v4",
            max_model_len=8192,
        )
        _, disc = COMPATIBILITY_MATRIX.evaluate(cfg)
        assert not any(r.id == "COMPAT-003" for r, _m in disc)

    def test_not_triggered_mtp(self):
        cfg = _minimal_cfg(
            spec_decode=SpecDecodeConfig(
                method="mtp", num_speculative_tokens=3,
            ),
            kv_cache_dtype="turboquant_k8v4",
            max_model_len=200_000,
        )
        _, disc = COMPATIBILITY_MATRIX.evaluate(cfg)
        assert not any(r.id == "COMPAT-003" for r, _m in disc)


# Note: COMPAT-004 (DFlash without drafter) is implicitly tested via
# SpecDecodeConfig.validate, which already raises SchemaError when method=dflash
# and model is None — meaning ModelConfig never reaches
# CompatibilityMatrix evaluation in that shape. The matrix rule
# exists for declarative visibility (CLI rendering), not as an
# additional enforcement gate.


# ─── ModelConfig integration ───────────────────────────────────────────


class TestModelConfigIntegration:
    def test_validate_raises_on_forbidden(self):
        cfg = _minimal_cfg(
            model_path="/models/qwen-next-30b",
            spec_decode=SpecDecodeConfig(
                method="dflash", num_speculative_tokens=4,
                model="/drafter/dflash",
            ),
        )
        with pytest.raises(SchemaError) as exc:
            cfg.validate()
        assert "COMPAT-001" in str(exc.value)

    def test_audit_surfaces_discouraged(self):
        cfg = _minimal_cfg(
            kv_cache_dtype="turboquant_k8v4",
            genesis_env={"GENESIS_ENABLE_PN59_STREAMING_GDN": "1"},
        )
        # validate() itself must not raise (discouraged != forbidden)
        cfg.validate()
        warnings = cfg.audit()
        assert any("COMPAT-002" in w for w in warnings)

    def test_clean_config_no_violations(self):
        cfg = _minimal_cfg(
            spec_decode=SpecDecodeConfig(
                method="mtp", num_speculative_tokens=3,
            ),
        )
        cfg.validate()  # no raise
        # audit() may surface other warnings (e.g. missing
        # reference_metrics) but must not flag COMPAT-* on a clean cfg.
        warnings = cfg.audit()
        compat_warnings = [w for w in warnings if w.startswith("[COMPAT-")]
        assert compat_warnings == []
