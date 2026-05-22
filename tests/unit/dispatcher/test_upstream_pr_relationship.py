# SPDX-License-Identifier: Apache-2.0
"""Phase 5.1.A + 5.1.C (2026-05-22) — `upstream_pr_relationship` enum.

Schema-additive change (5.1.A) plus migration (5.1.B) plus
legacy-fallback removal (5.1.C). Records the semantic relationship
between a Genesis patch and the upstream PR it cites via `upstream_pr`.

Covered here:

  1. `VALID_UPSTREAM_PR_RELATIONSHIPS` exposes the 6 canonical values.
  2. `PatchSpec.upstream_pr_relationship` defaults to ``"backport"``.
  3. `patch_spec_for()` propagates an explicit registry value through.
  4. `patch_spec_for()` no longer honors the legacy
     `enables_upstream_feature: True` boolean — removed in 5.1.C; the
     two former consumers (P75, P99) now carry explicit field syntax.
  5. `validate_registry()` rejects unknown enum values (ERROR).
  6. `validate_registry()` errors when `upstream_pr` is set but the
     relationship field is missing (5.1.C escalation; was silent in
     5.1.A's migration window).
  7. `validate_registry()` warns when the relationship is set without
     an `upstream_pr` target (likely copy-paste mistake).
  8. The shipped PATCH_REGISTRY validates clean (regression guard).
  9. The 8 special 5.1.B-migrated patches resolve to their assigned
     relationship value via the PatchSpec builder.
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
        """When the registry entry omits the field entirely, the builder
        defaults to ``"backport"``. Note: this case is only legitimate
        for entries WITHOUT an integer `upstream_pr` after Phase 5.1.C
        (the registry validator now errors on missing-when-set). The
        PatchSpec builder still defaults so the field is never None on
        the dataclass, but the validator catches the registry mistake."""
        meta = {
            "title": "test",
            "env_flag": "GENESIS_ENABLE_TEST",
            "default_on": False,
            "upstream_pr": 12345,
        }
        spec = patch_spec_for("P_TEST", meta)
        assert spec.upstream_pr_relationship == "backport"

    def test_builder_legacy_boolean_no_longer_recognized(self):
        """Phase 5.1.C cleanup: `enables_upstream_feature: True` boolean
        is no longer a relationship-hint synonym. With it set but the
        explicit field absent, the builder falls back to default
        ``"backport"`` — exactly as if the boolean weren't there. P75
        and P99 carry explicit `upstream_pr_relationship` after 5.1.B."""
        meta = {
            "title": "test",
            "env_flag": "GENESIS_ENABLE_TEST",
            "default_on": False,
            "upstream_pr": 12345,
            "enables_upstream_feature": True,
        }
        spec = patch_spec_for("P_TEST", meta)
        assert spec.upstream_pr_relationship == "backport"


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

    def test_missing_relationship_field_with_upstream_pr_is_error(
        self, monkeypatch,
    ):
        """Phase 5.1.C escalation: a registry entry with integer
        ``upstream_pr`` but no ``upstream_pr_relationship`` is now an
        ERROR (was silent during the 5.1.A migration window). Forces
        operators adding new backports to record relationship intent
        explicitly rather than letting an implicit default silently
        misclassify the patch in the audit script."""
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
            and i.severity == "ERROR"
        ]
        assert related, (
            "missing relationship field with upstream_pr set should "
            "surface ERROR after 5.1.C; got:\n"
            + "\n".join(f"  {i.severity}: {i.message}"
                        for i in issues
                        if "upstream_pr" in i.message)
        )

    def test_missing_relationship_no_upstream_pr_is_silent(self, monkeypatch):
        """The "missing-when-set" ERROR fires ONLY when upstream_pr is
        an integer. Entries without an upstream_pr (154 of 226 in the
        live registry) must not be touched by the new check."""
        fake = {
            "P_FOO": {
                "title": "test",
                "env_flag": "GENESIS_ENABLE_FOO",
                "default_on": False,
                "tier": "community",
                "lifecycle": "stable",
                # neither upstream_pr nor upstream_pr_relationship set
            },
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake)
        issues = dispatcher.validate_registry()
        related = [
            i for i in issues
            if "upstream_pr_relationship" in i.message
            or "upstream_pr is set" in i.message
        ]
        assert related == [], (
            "no-upstream-pr entries must not trigger relationship "
            f"checks; got: {[(i.severity, i.message) for i in related]}"
        )


# ─── Regression guard: live registry validates clean ──────────────────────


class TestLiveRegistryClean:
    def test_shipped_registry_has_no_relationship_errors(self):
        """Live PATCH_REGISTRY must validate clean — no ERROR and no
        WARNING. After 5.1.B all 72 upstream_pr-bearing entries carry
        an explicit relationship, and after 5.1.C the validator errors
        on any future operator who forgets to add one."""
        issues = dispatcher.validate_registry()
        relationship_issues = [
            i for i in issues
            if "upstream_pr_relationship" in i.message
            or "upstream_pr is set" in i.message
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


# ─── Phase 5.1.B special-patch regression ─────────────────────────────────


class TestSpecialPatchesResolveCorrectly:
    """The 8 patches given non-default relationships in Phase 5.1.B
    must remain so — guards against future operators editing the
    registry without checking the audit-routing impact."""

    EXPECTED_SPECIAL = {
        "PN116": "counter_regression",
        "P98":   "intentional_inverse",
        "P75":   "enables_upstream",
        "P99":   "enables_upstream",
        "PN51":  "defensive_overlay",
        "PN90":  "related_not_superseding",
        "PN24":  "related_not_superseding",
        "P61":   "related_not_superseding",
    }

    def test_each_special_patch_resolves_to_expected_value(self):
        for spec in spec_mod.iter_patch_specs():
            expected = self.EXPECTED_SPECIAL.get(spec.patch_id)
            if expected is None:
                continue
            assert spec.upstream_pr_relationship == expected, (
                f"{spec.patch_id}: expected "
                f"upstream_pr_relationship={expected!r}, got "
                f"{spec.upstream_pr_relationship!r}"
            )

    def test_all_8_special_patches_present_in_live_registry(self):
        """Defensive: if a 5.1.B special patch is ever removed or
        renamed in the registry, fail loudly here rather than in
        an audit-script bucket count down the line."""
        seen = {
            spec.patch_id
            for spec in spec_mod.iter_patch_specs()
            if spec.patch_id in self.EXPECTED_SPECIAL
        }
        missing = set(self.EXPECTED_SPECIAL) - seen
        assert not missing, (
            f"special-patch set drifted; missing in live registry: "
            f"{sorted(missing)}"
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
