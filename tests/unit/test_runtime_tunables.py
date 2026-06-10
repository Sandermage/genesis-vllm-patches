# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr.runtime_tunables` — the registry of runtime
tunable env knobs split out from `audit_rules._check_env_keys_exist`.

These tests are the TDD contract for P1-D (audit closure 2026-05-12).
"""
from __future__ import annotations

import pytest

from sndr.runtime_tunables import (
    TUNABLE_KNOBS,
    TunableKnob,
    is_known_tunable,
    tunable_prefixes,
)


class TestRegistryShape:
    def test_registry_nonempty(self):
        assert len(TUNABLE_KNOBS) >= 10, (
            "registry should declare at least the historical 10+ knobs"
        )

    def test_each_entry_is_tunable_knob(self):
        for key, knob in TUNABLE_KNOBS.items():
            assert isinstance(knob, TunableKnob), key
            assert knob.name == key, (
                f"key {key!r} must match knob.name {knob.name!r}"
            )
            assert knob.kind in ("scalar", "family"), knob.kind
            assert knob.type in (
                "bool", "int", "float", "string", "enum",
            ), knob.type
            if knob.kind == "family":
                assert key.endswith("_"), (
                    f"family knob {key!r} must end with `_`"
                )

    def test_every_knob_has_doc(self):
        for key, knob in TUNABLE_KNOBS.items():
            assert knob.doc.strip(), f"{key} has empty doc"


class TestKnownTunableSemantics:
    @pytest.mark.parametrize("name", [
        "GENESIS_OBSERVABILITY",
        "GENESIS_PROFILE_RUN_CAP_M",
        "GENESIS_TQ_MAX_MODEL_LEN",
        "GENESIS_BUFFER_MODE",
    ])
    def test_scalar_known(self, name):
        assert is_known_tunable(name) is True, name

    @pytest.mark.parametrize("name", [
        "GENESIS_PN16_TOOL_THINK_BUDGET",
        "GENESIS_PN95_CONFIG_KEY",
        "GENESIS_PN95_DEMOTE_FREE_MIB_THRESHOLD",
        "GENESIS_P67_NUM_KV_SPLITS",
        "GENESIS_P82_THRESHOLD_SINGLE",
    ])
    def test_family_known(self, name):
        assert is_known_tunable(name) is True, name

    @pytest.mark.parametrize("name", [
        "GENESIS_NOT_REAL",
        "GENESIS_ENABLE_P67",  # patch enable flag, NOT tunable
        "RANDOM_ENV_VAR",
        "",
    ])
    def test_unknown(self, name):
        assert is_known_tunable(name) is False, name


class TestBackCompatTunablePrefixes:
    def test_returns_tuple(self):
        assert isinstance(tunable_prefixes(), tuple)

    def test_contains_known_prefixes(self):
        prefixes = tunable_prefixes()
        for required in (
            "GENESIS_PN16_",
            "GENESIS_PN95_",
            "GENESIS_P67_",
            "GENESIS_OBSERVABILITY",
            "GENESIS_TQ_MAX_MODEL_LEN",
        ):
            assert required in prefixes, (
                f"{required} missing from tunable_prefixes()"
            )


class TestAuditRulesIntegration:
    """Audit gate must accept tunables and reject unknown env vars."""

    def test_audit_accepts_pn95_knobs(self):
        from sndr.model_configs.schema import ModelConfig

        cfg = ModelConfig.__new__(ModelConfig)
        cfg.key = "test"
        cfg.genesis_env = {
            # 2026-06-01: PN95 architectural unblock changed CONFIG_KEY
            # canonical value from V1 `a5000-2x-tier-aware-example` to
            # PN95-internal `a5000-2x-tier-aware`. Test fixture mirrors
            # production env (qwen3.6-27b/35b model files updated in
            # the same commit). V1 fallback still works for backward
            # compat.
            "GENESIS_PN95_CONFIG_KEY": "a5000-2x-tier-aware",
            "GENESIS_PN95_TICK_EVERY": "1",
            "GENESIS_PN95_DEMOTE_FREE_MIB_THRESHOLD": "1024",
        }
        from sndr.model_configs.audit_rules import _check_env_keys_exist
        result = _check_env_keys_exist(cfg)
        assert result is None or "GENESIS_PN95_" not in (result or ""), (
            f"audit rejected PN95 tunable: {result}"
        )

    def test_audit_rejects_truly_unknown_env(self):
        from sndr.model_configs.schema import ModelConfig

        cfg = ModelConfig.__new__(ModelConfig)
        cfg.key = "test"
        cfg.genesis_env = {"GENESIS_NOT_A_REAL_KNOB_OR_FLAG": "1"}
        from sndr.model_configs.audit_rules import _check_env_keys_exist
        result = _check_env_keys_exist(cfg)
        assert result is not None, (
            "audit should flag completely unknown env vars"
        )
        assert "GENESIS_NOT_A_REAL_KNOB_OR_FLAG" in result, result
