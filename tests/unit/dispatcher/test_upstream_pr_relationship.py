# SPDX-License-Identifier: Apache-2.0
"""Phase 5.1.A (2026-05-22) — `upstream_pr_relationship` enum schema.

Schema-additive change that records the semantic relationship between
a Genesis patch and the upstream PR it cites via `upstream_pr`.

Covered here (5.1.A — no registry migration yet):

  1. `VALID_UPSTREAM_PR_RELATIONSHIPS` exposes the 6 canonical values.
  2. `PatchSpec.upstream_pr_relationship` defaults to ``"backport"``.
  3. `patch_spec_for()` propagates an explicit registry value through.
  4. `patch_spec_for()` honors the legacy `enables_upstream_feature:
     True` boolean as a synonym for ``"enables_upstream"`` (back-compat
     during the 5.1.A → 5.1.C migration window).
  5. `validate_registry()` rejects unknown enum values (ERROR).
  6. `validate_registry()` warns when the relationship is set without
     an `upstream_pr` target (likely copy-paste mistake).
  7. The shipped PATCH_REGISTRY validates clean with the new check
     (regression guard for the migration window).

After Phase 5.1.C, items (4) and the absence-without-upstream-pr
case escalate; this file will be updated in that commit.
"""
from __future__ import annotations

import pytest

from vllm.sndr_core import dispatcher
from vllm.sndr_core.dispatcher import spec as spec_mod
from vllm.sndr_core.dispatcher.spec import (
    VALID_UPSTREAM_PR_RELATIONSHIPS,
    PatchSpec,
    patch_spec_for,
)


# ─── Enum surface ─────────────────────────────────────────────────────────


class TestValidEnumSurface:
    def test_enum_has_exactly_six_values(self):
        """The enum is intentionally closed at 6 values to keep the
        relationship taxonomy small. Adding a value requires a design
        note in `PHASE_5_1_RELATIONSHIP_SCHEMA_DESIGN_*` + audit-bucket
        update + test."""
        assert len(VALID_UPSTREAM_PR_RELATIONSHIPS) == 6

    @pytest.mark.parametrize("value", [
        "backport",
        "counter_regression",
        "intentional_inverse",
        "enables_upstream",
        "related_not_superseding",
        "defensive_overlay",
    ])
    def test_canonical_value_in_enum(self, value):
        assert value in VALID_UPSTREAM_PR_RELATIONSHIPS

    def test_default_is_backport(self):
        """First entry is the default and the back-compat fallback."""
        assert VALID_UPSTREAM_PR_RELATIONSHIPS[0] == "backport"


# ─── PatchSpec field default + derivation ─────────────────────────────────


def _bare_spec(**overrides) -> PatchSpec:
    """Build a PatchSpec with sensible defaults for the required
    positional fields. Phase 5.1.A tests only care about the
    relationship-related fields; everything else is filler."""
    base = dict(
        patch_id="P_TEST",
        title="t",
        tier="community",
        family="attention",
        env_flag="GENESIS_ENABLE_TEST",
        default_on=False,
        lifecycle="stable",
        upstream_pr=12345,
        apply_module=None,
    )
    base.update(overrides)
    return PatchSpec(**base)


class TestPatchSpecRelationshipField:
    def test_default_value_is_backport(self):
        s = _bare_spec()
        assert s.upstream_pr_relationship == "backport"

    def test_explicit_value_propagated(self):
        s = _bare_spec(upstream_pr_relationship="counter_regression")
        assert s.upstream_pr_relationship == "counter_regression"

    def test_builder_reads_explicit_field(self):
        meta = {
            "title": "test",
            "env_flag": "GENESIS_ENABLE_TEST",
            "default_on": False,
            "upstream_pr": 12345,
            "upstream_pr_relationship": "defensive_overlay",
        }
        spec = patch_spec_for("P_TEST", meta)
        assert spec.upstream_pr_relationship == "defensive_overlay"

    def test_builder_missing_field_defaults_to_backport(self):
        """Phase 5.1.A migration window: missing field is implicit
        ``"backport"`` for back-compat. Will be tightened in 5.1.C when
        the field becomes REQUIRED for any patch with ``upstream_pr``."""
        meta = {
            "title": "test",
            "env_flag": "GENESIS_ENABLE_TEST",
            "default_on": False,
            "upstream_pr": 12345,
        }
        spec = patch_spec_for("P_TEST", meta)
        assert spec.upstream_pr_relationship == "backport"

    def test_builder_legacy_boolean_maps_to_enables_upstream(self):
        """`enables_upstream_feature: True` is the LEGACY relationship
        hint (predates the enum). The builder honors it as a synonym
        for ``"enables_upstream"`` until Phase 5.1.C cleanup migrates
        the 2 existing entries (P75, P99) to explicit field syntax."""
        meta = {
            "title": "test",
            "env_flag": "GENESIS_ENABLE_TEST",
            "default_on": False,
            "upstream_pr": 12345,
            "enables_upstream_feature": True,
        }
        spec = patch_spec_for("P_TEST", meta)
        assert spec.upstream_pr_relationship == "enables_upstream"

    def test_explicit_field_overrides_legacy_boolean(self):
        """When both are set, the explicit field wins (so 5.1.C
        cleanup can migrate P75/P99 without removing the boolean
        in the same commit)."""
        meta = {
            "title": "test",
            "env_flag": "GENESIS_ENABLE_TEST",
            "default_on": False,
            "upstream_pr": 12345,
            "enables_upstream_feature": True,
            "upstream_pr_relationship": "related_not_superseding",
        }
        spec = patch_spec_for("P_TEST", meta)
        assert spec.upstream_pr_relationship == "related_not_superseding"


# ─── Validator behavior ───────────────────────────────────────────────────


def _fake_registry_with_relationship(rel_value, upstream_pr=12345):
    return {
        "P_FOO": {
            "title": "test",
            "env_flag": "GENESIS_ENABLE_FOO",
            "default_on": False,
            "tier": "community",
            "lifecycle": "stable",
            "upstream_pr": upstream_pr,
            "upstream_pr_relationship": rel_value,
        },
    }


class TestValidatorEnum:
    @pytest.mark.parametrize("value", VALID_UPSTREAM_PR_RELATIONSHIPS)
    def test_each_valid_value_is_accepted(self, value, monkeypatch):
        monkeypatch.setattr(
            dispatcher, "PATCH_REGISTRY",
            _fake_registry_with_relationship(value),
        )
        issues = dispatcher.validate_registry()
        bad = [
            i for i in issues
            if "upstream_pr_relationship" in i.message
            and i.severity == "ERROR"
        ]
        assert bad == [], (
            f"valid value {value!r} rejected by validator:\n"
            + "\n".join(f"  {i.severity}: {i.message}" for i in bad)
        )

    def test_invalid_enum_value_is_error(self, monkeypatch):
        monkeypatch.setattr(
            dispatcher, "PATCH_REGISTRY",
            _fake_registry_with_relationship("totally_bogus_value"),
        )
        issues = dispatcher.validate_registry()
        errors = [
            i for i in issues
            if "upstream_pr_relationship" in i.message
            and i.severity == "ERROR"
        ]
        assert errors, "invalid relationship value did not produce ERROR"
        assert "totally_bogus_value" in errors[0].message

    def test_relationship_set_without_upstream_pr_is_warning(self, monkeypatch):
        """Phase 5.1.A: setting the relationship without an upstream_pr
        target is a likely-mistake. WARNING during the migration window;
        will be ERROR after Phase 5.1.C."""
        monkeypatch.setattr(
            dispatcher, "PATCH_REGISTRY",
            _fake_registry_with_relationship(
                "intentional_inverse", upstream_pr=None,
            ),
        )
        # Remove the None upstream_pr so it's truly absent (matches
        # the "field forgotten" copy-paste mistake we want to catch).
        fake = dispatcher.PATCH_REGISTRY
        del fake["P_FOO"]["upstream_pr"]
        issues = dispatcher.validate_registry()
        warnings = [
            i for i in issues
            if "upstream_pr_relationship" in i.message
            and i.severity == "WARNING"
        ]
        assert warnings, (
            "relationship without upstream_pr did not produce WARNING"
        )

    def test_missing_relationship_field_is_silent(self, monkeypatch):
        """Phase 5.1.A migration window: a registry entry with
        `upstream_pr` but no `upstream_pr_relationship` must NOT
        surface an error or warning — it's treated as implicit
        ``"backport"``. After Phase 5.1.C, this becomes ERROR."""
        fake = {
            "P_FOO": {
                "title": "test",
                "env_flag": "GENESIS_ENABLE_FOO",
                "default_on": False,
                "tier": "community",
                "lifecycle": "stable",
                "upstream_pr": 12345,
                # upstream_pr_relationship intentionally absent
            },
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake)
        issues = dispatcher.validate_registry()
        related = [
            i for i in issues
            if "upstream_pr_relationship" in i.message
        ]
        assert related == [], (
            "missing relationship field should be silent during "
            "the 5.1.A migration window; got:\n"
            + "\n".join(f"  {i.severity}: {i.message}" for i in related)
        )


# ─── Regression guard: live registry validates clean ──────────────────────


class TestLiveRegistryClean:
    def test_shipped_registry_has_no_relationship_errors(self):
        """After 5.1.A lands, the shipped PATCH_REGISTRY must NOT
        produce any ERROR-severity relationship issues. WARNING is
        also a regression at this point: 5.1.A doesn't migrate any
        entry, so no entry should have a relationship set without
        an upstream_pr target."""
        issues = dispatcher.validate_registry()
        relationship_issues = [
            i for i in issues
            if "upstream_pr_relationship" in i.message
        ]
        blocking = [
            i for i in relationship_issues
            if i.severity in ("ERROR", "WARNING")
        ]
        assert blocking == [], (
            "shipped registry has relationship issues:\n"
            + "\n".join(
                f"  {i.severity}: {i.patch_id}: {i.message}"
                for i in blocking
            )
        )


# ─── Module surface ───────────────────────────────────────────────────────


class TestModuleSurface:
    def test_enum_is_importable_from_spec(self):
        """Audit script + tests both depend on this import path."""
        from vllm.sndr_core.dispatcher.spec import (
            VALID_UPSTREAM_PR_RELATIONSHIPS,
        )
        assert isinstance(VALID_UPSTREAM_PR_RELATIONSHIPS, tuple)
        assert len(VALID_UPSTREAM_PR_RELATIONSHIPS) >= 1

    def test_enum_is_tuple_not_list(self):
        """Enums in this module are tuples (immutable, set-membership
        friendly). Mirrors `VALID_SOURCES`, `_VALID_TIERS`, etc."""
        assert isinstance(spec_mod.VALID_UPSTREAM_PR_RELATIONSHIPS, tuple)
