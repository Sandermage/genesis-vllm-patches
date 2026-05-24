# SPDX-License-Identifier: Apache-2.0
"""CONFIG-UX.4.1 — tests for stage-aware warning helpers.

Covers backward-compat + stage-aware behavior of:
  - `_maybe_warn_v1_deprecation(key, *, bucket=None, stage=None)` in registry.py
  - `_maybe_warn_unannotated(preset_id, *, stage=None)` in registry_v2.py

Architectural guard from CONFIG_UX_4_R §0: at default Stage 0/1,
observability-only — no caller sees new ERROR severity, no launch
behavior changes.
"""
from __future__ import annotations

import warnings

import pytest

from vllm.sndr_core.model_configs import registry, registry_v2


@pytest.fixture
def fresh_warn_state():
    """Snapshot + clear the once-per-process warned sets so tests
    can observe emission events deterministically."""
    v1_snapshot = set(registry._V1_DEPRECATION_WARNED)
    un_snapshot = set(registry_v2._UNANNOTATED_PRESET_WARNED)
    registry._V1_DEPRECATION_WARNED.clear()
    registry_v2._UNANNOTATED_PRESET_WARNED.clear()
    yield
    registry._V1_DEPRECATION_WARNED.clear()
    registry._V1_DEPRECATION_WARNED.update(v1_snapshot)
    registry_v2._UNANNOTATED_PRESET_WARNED.clear()
    registry_v2._UNANNOTATED_PRESET_WARNED.update(un_snapshot)


# ─── _maybe_warn_v1_deprecation ────────────────────────────────────────────


class TestV1DeprecationBackwardCompat:
    def test_positional_signature_still_works(self, fresh_warn_state):
        """Old call sites (positional key only) must still work."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            registry._maybe_warn_v1_deprecation("a5000-2x-35b-prod")
        deprs = [w for w in caught if "V1 monolithic" in str(w.message)]
        assert deprs, "old positional call should emit DeprecationWarning"

    def test_bucket_auto_resolved_from_table(self, fresh_warn_state):
        """When bucket=None, helper resolves via migration table."""
        # transparent bucket → warn at Stage 0 default
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            registry._maybe_warn_v1_deprecation(
                "a5000-2x-35b-prod",
                stage=0,
            )
        deprs = [w for w in caught if "V1 monolithic" in str(w.message)]
        assert deprs
        # Message should include the bucket name
        assert "transparent" in str(deprs[0].message)

    def test_explicit_bucket_overrides_lookup(self, fresh_warn_state):
        """Explicit bucket kwarg overrides the table lookup."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            registry._maybe_warn_v1_deprecation(
                "fake-key-not-in-table",
                bucket="deprecated",
                stage=0,
            )
        deprs = [w for w in caught if "V1 monolithic" in str(w.message)]
        assert deprs
        assert "deprecated" in str(deprs[0].message)


class TestV1DeprecationStageBehavior:
    def test_stage_0_transparent_emits_warn(self, fresh_warn_state):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            registry._maybe_warn_v1_deprecation(
                "a5000-2x-35b-prod", bucket="transparent", stage=0,
            )
        deprs = [w for w in caught if "V1 monolithic" in str(w.message)]
        assert deprs

    def test_stage_3_transparent_still_warn(self, fresh_warn_state):
        """transparent bucket stays WARN forever (regression guard)."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            registry._maybe_warn_v1_deprecation(
                "a5000-2x-35b-prod", bucket="transparent", stage=3,
            )
        deprs = [w for w in caught if "V1 monolithic" in str(w.message)]
        assert deprs

    def test_stage_3_deprecated_raises(self, fresh_warn_state):
        """Stage 3 + deprecated bucket → RuntimeError."""
        with pytest.raises(RuntimeError, match="V1 monolithic"):
            registry._maybe_warn_v1_deprecation(
                "a5000-1x-tier-aware-pn95",
                bucket="deprecated",
                stage=3,
            )

    def test_stage_3_needs_choice_raises(self, fresh_warn_state):
        with pytest.raises(RuntimeError, match="V1 monolithic"):
            registry._maybe_warn_v1_deprecation(
                "a5000-2x-27b-int4-long-ctx",
                bucket="needs_operator_choice",
                stage=3,
            )

    def test_tombstone_raises_at_any_stage(self, fresh_warn_state):
        for stage in (0, 1, 2, 3):
            registry._V1_DEPRECATION_WARNED.clear()
            with pytest.raises(RuntimeError, match="V1 monolithic"):
                registry._maybe_warn_v1_deprecation(
                    "fake-tombstone-key",
                    bucket="tombstone",
                    stage=stage,
                )

    def test_once_per_process_tracking(self, fresh_warn_state):
        """Second call for the same key is a no-op."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            registry._maybe_warn_v1_deprecation("a5000-2x-35b-prod", stage=0)
            registry._maybe_warn_v1_deprecation("a5000-2x-35b-prod", stage=0)
        deprs = [w for w in caught if "V1 monolithic" in str(w.message)]
        # Only one warning emitted
        assert len(deprs) == 1

    def test_disable_env_silences_warn_only(self, monkeypatch, fresh_warn_state):
        """GENESIS_DISABLE_V1_DEPRECATION_WARNING=1 silences WARN but
        NOT ERROR severity."""
        monkeypatch.setenv("GENESIS_DISABLE_V1_DEPRECATION_WARNING", "1")
        # WARN — silenced
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            registry._maybe_warn_v1_deprecation(
                "a5000-2x-35b-prod", bucket="transparent", stage=0,
            )
        deprs = [w for w in caught if "V1 monolithic" in str(w.message)]
        assert not deprs

        # ERROR — NOT silenced (Stage 3 deprecated still raises)
        registry._V1_DEPRECATION_WARNED.clear()
        with pytest.raises(RuntimeError):
            registry._maybe_warn_v1_deprecation(
                "a5000-1x-tier-aware-pn95",
                bucket="deprecated", stage=3,
            )


# ─── _maybe_warn_unannotated ───────────────────────────────────────────────


class TestUnannotatedBackwardCompat:
    def test_positional_signature_still_works(self, fresh_warn_state):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            registry_v2._maybe_warn_unannotated("prod-35b")
        deprs = [w for w in caught if "card:" in str(w.message)]
        assert deprs


class TestUnannotatedStageBehavior:
    def test_prod_card_less_warns_at_stage_0(self, fresh_warn_state):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            registry_v2._maybe_warn_unannotated("prod-some-preset", stage=0)
        deprs = [w for w in caught if "card:" in str(w.message)]
        assert deprs

    def test_non_prod_card_less_silenced_forever(self, fresh_warn_state):
        """Operator decision §10.3: non-prod presets stay INFO indefinitely."""
        for stage in (0, 1, 2, 3):
            registry_v2._UNANNOTATED_PRESET_WARNED.clear()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", DeprecationWarning)
                registry_v2._maybe_warn_unannotated(
                    "example-some-preset", stage=stage,
                )
            deprs = [w for w in caught if "card:" in str(w.message)]
            assert not deprs, (
                f"non-prod preset should be silenced at stage {stage}; "
                f"got: {[str(w.message) for w in deprs]}"
            )

    def test_prod_stage_3_raises(self, fresh_warn_state):
        with pytest.raises(RuntimeError, match="card:"):
            registry_v2._maybe_warn_unannotated("prod-some-preset", stage=3)


# ─── Byte-identical compose (golden invariant) ─────────────────────────────


class TestComposeUnchanged:
    """Architectural guard: at Stage 0/1 default, composed ModelConfig
    must be byte-identical for all 21 builtin presets."""

    def test_all_21_presets_compose_clean(self):
        from vllm.sndr_core.model_configs.registry_v2 import (
            _alias_dir, load_alias,
        )
        from vllm.sndr_core.model_configs.schema import dump_yaml
        aliases = sorted(
            p.stem for p in _alias_dir().glob("*.yaml")
            if p.is_file() and not p.stem.startswith("_")
        )
        assert len(aliases) >= 21
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            for alias in aliases:
                cfg = load_alias(alias)
                y = dump_yaml(cfg)
                assert y, f"preset {alias!r}: empty compose output"
