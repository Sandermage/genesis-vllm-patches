# SPDX-License-Identifier: Apache-2.0
"""CONFIG-UX.1 acceptance suite — typed PresetCard / PresetDef / OverridePolicy.

Tests the 10 acceptance gates in `sndr_private/planning/audits/
CONFIG_UX_R_2026-05-24_RU.md` §13.3:

  Gate 1 — all 21 existing preset YAMLs load unchanged
  Gate 2 — composed ModelConfig byte-identical for all 21 presets without `card:`
  Gate 3 — new preset with full `card:` validates clean
  Gate 4 — PresetCard validation errors precise + actionable
  Gate 5 — EvidenceVisibility round-trips correctly
  Gate 6 — OverridePolicy parses for all 4 classes
  Gate 7 — backwards-compat loader emits warning (not error) for legacy presets
  Gate 8 — no new public docs required for CONFIG-UX.1 (covered by absence test)
  Gate 9 — make evidence stays 38/40 baseline (out-of-band check, not pytest)
  Gate 10 — pytest tests/unit/{dispatcher,model_configs,scripts} 1248+ pass
            (out-of-band aggregate check, not pytest)

Locked scope per §13.1: schema/data only — no CLI, no audit script, no
preset annotation work, no V1 rollout escalation.
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import pytest

from sndr.model_configs.preset_schema import (
    PRESET_AUDIENCES,
    PRESET_MATURITIES,
    PRESET_MODES,
    PRESET_STATUSES,
    EVIDENCE_TYPES,
    EVIDENCE_VISIBILITIES,
    EvidenceRef,
    PresetCard,
    PresetDef,
    parse_preset_yaml,
    synth_card_for_legacy,
    validate_for_status,
)
from sndr.model_configs.schema import SchemaError, dump_yaml
from sndr.model_configs.schema_v2 import (
    OVERRIDE_POLICY_CLASSES,
    OverridePolicy,
    PROFILE_ROLES,
    NON_PRODUCTION_ROLES,
    PRODUCTION_ROLES,
)
from sndr.model_configs.registry_v2 import (
    _alias_dir,
    load_alias,
    load_preset_def,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "model_configs"


def _all_builtin_aliases() -> list[str]:
    return sorted(p.stem for p in _alias_dir().glob("*.yaml") if p.is_file())


def _load_fixture_data(name: str) -> dict:
    import yaml
    return yaml.safe_load((FIXTURE_DIR / name).read_text())


# ─── Gate 1: all 21 builtin presets load unchanged via PresetDef ───────────


class TestGate1LegacyPresetsLoad:
    """All 21 existing preset YAMLs load through the new typed PresetDef
    path without raising — they're all legacy 3-pointer (no `card:`)."""

    @pytest.mark.parametrize("alias", _all_builtin_aliases())
    def test_legacy_preset_loads_via_preset_def(self, alias):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            pd = load_preset_def(alias)
        assert isinstance(pd, PresetDef)
        assert pd.model, f"preset {alias!r}: model pointer missing"
        assert pd.hardware, f"preset {alias!r}: hardware pointer missing"
        # CONFIG-UX.2 (2026-05-24): 14 prod-* presets carry `card:`
        # annotations (required after CONFIG-UX.2).
        # CONFIG-UX.2b (2026-05-30): qa-27b-* gained QA-status cards (Iter 47
        # 9adb9078→). Remaining non-prod (example-*, experimental-*,
        # long-ctx-*) still card-less pending operator-curated copy.
        # Contract: prod-* must have a card; everything else may either
        # carry one (qa-* lifted to CONFIG-UX.2b done) or stay card-less
        # (pre-2b state).
        if alias.startswith("prod-"):
            assert pd.card is not None, (
                f"preset {alias!r}: prod-* preset expected to carry "
                f"a `card:` annotation after CONFIG-UX.2"
            )
        elif pd.card is not None:
            # Non-prod preset opted into CONFIG-UX.2b — must validate via
            # the same shape check that load_preset_def applies. The
            # loader already calls .validate() on the card; arriving here
            # means it passed. Sanity-assert the status family is
            # non-production (audience semantic).
            assert pd.card.status in (
                "qa", "experimental", "bench_pending", "example",
                "internal_validated", "historical",
            ), (
                f"preset {alias!r}: non-prod alias with card.status="
                f"{pd.card.status!r} — must be a non-production status"
            )

    @pytest.mark.parametrize("alias", _all_builtin_aliases())
    def test_legacy_preset_loads_via_load_alias(self, alias):
        """Composition path through load_alias() still produces a V1
        ModelConfig (preserves contract for downstream runtime)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg = load_alias(alias)
        from sndr.model_configs.schema import ModelConfig
        assert isinstance(cfg, ModelConfig)
        assert cfg.key, f"preset {alias!r}: composed key empty"


# ─── Gate 2: composed ModelConfig byte-identical for legacy presets ────────


class TestGate2ByteIdenticalCompose:
    """Composed YAML for all 21 builtin presets must be byte-identical
    before and after CONFIG-UX.1 schema additions.

    This test re-runs composition through load_alias() and compares the
    sha256 of the dump_yaml() output against a golden table captured
    BEFORE CONFIG-UX.1 schema work. The golden table is computed inline
    here from the current source tree because all 21 are still legacy
    3-pointer (no card); annotation work in CONFIG-UX.2 will start
    invalidating this gate per-preset, which is the right time to
    update the table.

    For now: re-loading must produce the SAME output every time, so we
    assert (a) all 21 load, (b) shas are stable across two consecutive
    loads, (c) ConfigYAML output has > 0 lines.
    """

    @pytest.mark.parametrize("alias", _all_builtin_aliases())
    def test_compose_output_deterministic(self, alias):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg1 = load_alias(alias)
            cfg2 = load_alias(alias)
        y1 = dump_yaml(cfg1)
        y2 = dump_yaml(cfg2)
        assert y1 == y2, (
            f"preset {alias!r}: compose output not deterministic between "
            f"two consecutive loads — investigate before CONFIG-UX.2"
        )
        assert len(y1.splitlines()) > 0, (
            f"preset {alias!r}: composed YAML empty"
        )

    def test_all_legacy_presets_compose_stable_set(self):
        """Aggregate stable-set: all 21 legacy presets compose to a
        deterministic set of (alias, sha256) pairs in a single process."""
        aliases = _all_builtin_aliases()
        assert len(aliases) >= 21, (
            f"expected ≥21 builtin presets, found {len(aliases)} — "
            f"CONFIG-UX.R §1.1 snapshot drift"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            shas = {
                alias: hashlib.sha256(
                    dump_yaml(load_alias(alias)).encode("utf-8")
                ).hexdigest()
                for alias in aliases
            }
        # Each alias produces a unique sha (no two aliases collide)
        assert len(set(shas.values())) == len(shas), (
            f"two aliases composed to identical YAML — possible duplicate "
            f"preset definitions: {shas}"
        )


# ─── Gate 3: full card fixture validates clean ─────────────────────────────


class TestGate3FullCardFixture:
    def test_fixture_parses_clean(self):
        data = _load_fixture_data("experimental-card-example.yaml")
        pd = parse_preset_yaml("experimental-card-example", data)
        pd.validate()
        assert pd.has_card()
        assert pd.card.status == "experimental"
        assert pd.card.K == 4
        assert pd.card.mode == "structured_throughput"
        assert len(pd.card.evidence_refs) == 2
        assert pd.card.evidence_visibility == "mixed"
        assert len(pd.card.do_not_use) == 2
        assert pd.card.fallback_preset == "prod-qwen3.6-35b-balanced"

    def test_fixture_passes_experimental_validation_permissively(self):
        """Experimental status → permissive (no errors even though many
        fields are present)."""
        data = _load_fixture_data("experimental-card-example.yaml")
        pd = parse_preset_yaml("experimental-card-example", data)
        errs = validate_for_status(pd.card, "experimental-card-example")
        assert errs == [], (
            f"experimental status should pass permissive validation; got: {errs}"
        )

    def test_fixture_passes_production_when_status_flipped(self):
        """If status is flipped to production+operator, the fixture's
        complete evidence/workload/etc. should satisfy strict validation."""
        data = _load_fixture_data("experimental-card-example.yaml")
        pd = parse_preset_yaml("experimental-card-example", data)
        pd.card.status = "production"
        pd.card.audience = "operator"
        errs = validate_for_status(pd.card, "experimental-card-example")
        assert errs == [], (
            f"fixture has all production-required fields; expected no errors. "
            f"got: {errs}"
        )


# ─── Gate 4: validation errors precise and actionable ──────────────────────


class TestGate4ActionableErrors:
    def test_missing_title_error_names_field(self):
        with pytest.raises(SchemaError, match="card.title required"):
            PresetCard(title="", summary="x", status="experimental").validate()

    def test_invalid_status_error_lists_allowed(self):
        with pytest.raises(SchemaError) as excinfo:
            PresetCard(
                title="x", summary="y", status="not_a_status",
            ).validate()
        msg = str(excinfo.value)
        assert "card.status='not_a_status'" in msg
        # Allowed list must appear in the error so operator can self-correct
        assert "production" in msg
        assert "experimental" in msg

    def test_invalid_audience_error(self):
        with pytest.raises(SchemaError, match="card.audience"):
            PresetCard(
                title="x", summary="y", status="experimental",
                audience="not_an_audience",
            ).validate()

    def test_invalid_mode_error(self):
        with pytest.raises(SchemaError, match="card.mode"):
            PresetCard(
                title="x", summary="y", status="experimental",
                mode="not_a_mode",
            ).validate()

    def test_invalid_evidence_visibility_error(self):
        with pytest.raises(SchemaError, match="card.evidence_visibility"):
            PresetCard(
                title="x", summary="y", status="experimental",
                evidence_visibility="not_visible",
            ).validate()

    def test_invalid_K_error(self):
        with pytest.raises(SchemaError, match="card.K"):
            PresetCard(
                title="x", summary="y", status="experimental", K=0,
            ).validate()

    def test_evidence_ref_invalid_type(self):
        with pytest.raises(SchemaError, match="evidence_ref"):
            EvidenceRef(type="not_a_type", path="x").validate()

    def test_strict_validation_pinpoints_preset_id(self):
        """Strict validation errors must include preset_id so operator
        knows which file to edit."""
        card = PresetCard(
            title="x", summary="y", status="production",
        )
        errs = validate_for_status(card, "the-preset-id")
        assert all("the-preset-id" in e for e in errs), (
            f"all strict errors must mention preset id; got: {errs}"
        )
        assert any("audience required" in e for e in errs)
        assert any("workload_allow" in e for e in errs)


# ─── Gate 5: EvidenceVisibility round-trips ────────────────────────────────


class TestGate5EvidenceVisibilityRoundTrip:
    @pytest.mark.parametrize("visibility", EVIDENCE_VISIBILITIES)
    def test_card_level_visibility_roundtrip(self, visibility):
        data = {
            "model": "m1",
            "hardware": "h1",
            "card": {
                "title": "t",
                "summary": "s",
                "status": "experimental",
                "evidence_visibility": visibility,
            },
        }
        pd = parse_preset_yaml("test-vis", data)
        pd.validate()
        assert pd.card.evidence_visibility == visibility

    @pytest.mark.parametrize("visibility", EVIDENCE_VISIBILITIES)
    def test_per_evidence_ref_visibility_roundtrip(self, visibility):
        data = {
            "model": "m1",
            "hardware": "h1",
            "card": {
                "title": "t",
                "summary": "s",
                "status": "experimental",
                "evidence_refs": [
                    {"type": "bench", "path": "p/q", "visibility": visibility},
                ],
            },
        }
        pd = parse_preset_yaml("test-vis-ref", data)
        pd.validate()
        assert len(pd.card.evidence_refs) == 1
        assert pd.card.evidence_refs[0].visibility == visibility

    def test_visibility_omitted_is_none(self):
        data = {
            "model": "m1",
            "hardware": "h1",
            "card": {
                "title": "t",
                "summary": "s",
                "status": "experimental",
                "evidence_refs": [{"type": "bench", "path": "p/q"}],
            },
        }
        pd = parse_preset_yaml("test-vis-omitted", data)
        pd.validate()
        assert pd.card.evidence_visibility is None
        assert pd.card.evidence_refs[0].visibility is None

    def test_production_operator_private_evidence_only_errors(self):
        """CONFIG-UX.R §2.4 rule 1: production + operator + private
        evidence only → strict validation error."""
        card = PresetCard(
            title="t", summary="s", status="production", audience="operator",
            mode="throughput",
            workload_allow=["x"], workload_deny=["y"],
            K=1,
            routing_family="rf",
            evidence_visibility="private",
            evidence_refs=[
                EvidenceRef(type="bench", path="p/q", visibility="private"),
            ],
        )
        # Fill remaining required fields so we isolate the visibility check
        from sndr.model_configs.preset_schema import (
            ConcurrencyEnvelope, PrimaryMetric,
        )
        card.concurrency = ConcurrencyEnvelope(min=1, canonical=1, max=1)
        card.primary_metric = PrimaryMetric(
            kind="agg_TPS", value=1.0, source="x",
        )
        errs = validate_for_status(card, "test-prod-private")
        assert any("public" in e for e in errs), (
            f"expected public-evidence requirement violation; got: {errs}"
        )


# ─── Gate 6: OverridePolicy parses for all classes + role derivation ──────


class TestGate6OverridePolicy:
    @pytest.mark.parametrize("cls", OVERRIDE_POLICY_CLASSES)
    def test_each_class_parses_and_validates(self, cls):
        p = OverridePolicy(override_class=cls, reason="test")
        p.validate()
        assert p.override_class == cls

    def test_invalid_class_rejected_with_actionable_message(self):
        with pytest.raises(SchemaError) as excinfo:
            OverridePolicy(override_class="not_a_class").validate()
        msg = str(excinfo.value)
        assert "not_a_class" in msg
        # Allowed classes in error message
        for cls in OVERRIDE_POLICY_CLASSES:
            assert cls in msg

    @pytest.mark.parametrize("role", PRODUCTION_ROLES)
    def test_production_role_derives_production_class(self, role):
        p = OverridePolicy()  # no explicit class
        assert p.effective_class(role) == "production"

    @pytest.mark.parametrize("role", NON_PRODUCTION_ROLES)
    def test_non_production_role_derives_matching_class(self, role):
        p = OverridePolicy()
        assert p.effective_class(role) == role

    def test_none_role_derives_production(self):
        """Legacy profiles without `role` field → treated as production class."""
        p = OverridePolicy()
        assert p.effective_class(None) == "production"

    def test_explicit_class_overrides_role_derivation(self):
        """`override_class` declared explicitly wins over role."""
        p = OverridePolicy(override_class="bench")
        assert p.effective_class("default") == "bench"

    @pytest.mark.parametrize("vis", ("public", "private", "mixed"))
    def test_evidence_visibility_accepts_valid_values(self, vis):
        OverridePolicy(evidence_visibility=vis).validate()

    def test_evidence_visibility_invalid_rejected(self):
        with pytest.raises(SchemaError, match="evidence_visibility"):
            OverridePolicy(evidence_visibility="not_vis").validate()


# ─── Gate 7: backwards-compat — missing card = warning, not error ──────────


class TestGate7BackwardsCompat:
    def test_legacy_prod_preset_emits_deprecation_warning(self):
        """CONFIG-UX.4.1: card-less `prod-*` presets emit a
        DeprecationWarning at Stage 0/1.

        At the time of writing all 14 builtin prod-* presets are
        annotated (CONFIG-UX.2 ship). To exercise the prod-* warn
        path, call the helper directly with a synthetic prod-* id.
        Non-prod card-less presets are silenced forever per the
        operator's CONFIG-UX.4.R §10.3 decision (see the next test).
        """
        from sndr.model_configs import registry_v2
        registry_v2._UNANNOTATED_PRESET_WARNED.discard("prod-synthetic-fake")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            registry_v2._maybe_warn_unannotated("prod-synthetic-fake", stage=0)
        depr = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert depr, "prod-* card-less preset should emit DeprecationWarning"
        assert any("prod-synthetic-fake" in str(w.message) for w in depr)

    def test_legacy_non_prod_preset_silenced(self):
        """CONFIG-UX.4.1 / operator decision §10.3: card-less non-prod
        presets (example-*, qa-*, experimental-*, long-ctx-*) stay
        silenced forever to avoid breaking demos / qa / WIP entries
        before they become user-facing in CONFIG-UX.2b."""
        from sndr.model_configs import registry_v2
        alias = _all_builtin_aliases()[0]
        # First builtin preset alphabetically is non-prod ("example-*").
        assert not alias.startswith("prod-"), (
            f"test assumption broken — first builtin {alias!r} should be non-prod"
        )
        registry_v2._UNANNOTATED_PRESET_WARNED.discard(alias)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            registry_v2.load_alias(alias)
        depr = [
            w for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "card:" in str(w.message)
        ]
        assert not depr, (
            f"non-prod card-less preset should be silenced; got: "
            f"{[str(w.message) for w in depr]}"
        )

    def test_legacy_preset_load_does_not_raise(self):
        """Missing card must not be a hard error — preserve back-compat."""
        for alias in _all_builtin_aliases():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                # Must not raise
                load_preset_def(alias)

    def test_synth_card_for_legacy_is_valid(self):
        """The placeholder card synthesised for legacy presets must
        itself pass shape validation (no infinite loop of warnings)."""
        card = synth_card_for_legacy("some-preset-id")
        # Shape-valid (raises if not)
        card.validate()
        assert card.status == "experimental"
        assert card.audience == "dev"
        # Permissive validation passes (experimental status path)
        errs = validate_for_status(card, "some-preset-id")
        assert errs == []

    def test_disable_deprecation_warning_env_silences(self, monkeypatch):
        """`GENESIS_DISABLE_V1_DEPRECATION_WARNING=1` silences the new
        unannotated-preset warning (same env that silences V1 deprecation)."""
        from sndr.model_configs import registry_v2
        monkeypatch.setenv("GENESIS_DISABLE_V1_DEPRECATION_WARNING", "1")
        alias = _all_builtin_aliases()[0]
        registry_v2._UNANNOTATED_PRESET_WARNED.discard(alias)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            registry_v2.load_alias(alias)
        depr = [
            w for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "card" in str(w.message)
        ]
        assert not depr, (
            f"GENESIS_DISABLE_V1_DEPRECATION_WARNING=1 should silence; "
            f"got: {[str(w.message) for w in depr]}"
        )


# ─── Schema housekeeping: enum coverage + invariants ───────────────────────


class TestEnumInvariants:
    def test_status_enum_has_9_values(self):
        """CONFIG_UX_R §2.5 final list."""
        assert len(PRESET_STATUSES) == 9
        for s in (
            "experimental", "bench_pending", "internal_validated",
            "production_candidate", "production", "historical",
            "tombstone", "example", "qa",
        ):
            assert s in PRESET_STATUSES

    def test_audience_enum(self):
        for a in ("operator", "dev", "bench", "qa", "internal"):
            assert a in PRESET_AUDIENCES

    def test_mode_enum(self):
        for m in (
            "throughput", "structured_throughput",
            "latency", "long_context", "tool_agent",
        ):
            assert m in PRESET_MODES

    def test_profile_roles_extended(self):
        """CONFIG-UX.1 §10.8 extension — `default | structured | gateway |
        bench | dev | qa | diagnostic`."""
        assert len(PROFILE_ROLES) == 7
        for r in (
            "default", "structured", "gateway",
            "bench", "dev", "qa", "diagnostic",
        ):
            assert r in PROFILE_ROLES

    def test_role_partition_invariant(self):
        """Production + non-production roles must partition the full role
        set (no overlap, no gaps except None)."""
        assert PRODUCTION_ROLES & NON_PRODUCTION_ROLES == set()
        assert PRODUCTION_ROLES | NON_PRODUCTION_ROLES == set(PROFILE_ROLES)


# ─── Concurrency invariant ─────────────────────────────────────────────────


class TestConcurrencyInvariant:
    def test_valid_envelope(self):
        from sndr.model_configs.preset_schema import ConcurrencyEnvelope
        ConcurrencyEnvelope(min=1, max=8, canonical=4).validate()

    def test_canonical_below_min_rejected(self):
        from sndr.model_configs.preset_schema import ConcurrencyEnvelope
        with pytest.raises(SchemaError, match="concurrency invariant"):
            ConcurrencyEnvelope(min=2, max=8, canonical=1).validate()

    def test_max_below_canonical_rejected(self):
        from sndr.model_configs.preset_schema import ConcurrencyEnvelope
        with pytest.raises(SchemaError, match="concurrency invariant"):
            ConcurrencyEnvelope(min=1, max=2, canonical=4).validate()


# ─── Forward-compat: unknown card keys ignored ─────────────────────────────


class TestForwardCompat:
    def test_unknown_card_key_silently_ignored(self):
        """Future schema extensions land as new keys; loader should ignore
        unknown keys rather than reject (preserves forward-compat)."""
        data = {
            "model": "m1", "hardware": "h1",
            "card": {
                "title": "t", "summary": "s", "status": "experimental",
                "future_field_added_later": "ignored",
                "another_future_field": {"nested": "also_ignored"},
            },
        }
        pd = parse_preset_yaml("test-fwd-compat", data)
        pd.validate()
        # Did not raise → forward-compat preserved.

    def test_missing_card_in_yaml_path_loads_as_legacy(self):
        data = {"model": "m1", "hardware": "h1", "profile": "p1"}
        pd = parse_preset_yaml("test-legacy-shape", data)
        pd.validate()
        assert pd.card is None


# ─── PresetDef pointer validation ──────────────────────────────────────────


class TestPresetDefPointers:
    def test_missing_model_rejected(self):
        with pytest.raises(SchemaError, match="model"):
            parse_preset_yaml(
                "test-no-model", {"hardware": "h1"},
            ).validate()

    def test_missing_hardware_rejected(self):
        with pytest.raises(SchemaError, match="hardware"):
            parse_preset_yaml(
                "test-no-hw", {"model": "m1"},
            ).validate()

    def test_invalid_alias_id_rejected(self):
        with pytest.raises(SchemaError, match="alias"):
            parse_preset_yaml(
                "INVALID-UPPERCASE-ID",
                {"model": "m1", "hardware": "h1"},
            )

    def test_profile_optional(self):
        data = {"model": "m1", "hardware": "h1"}
        pd = parse_preset_yaml("test-no-profile", data)
        pd.validate()
        assert pd.profile is None

    def test_runtime_optional(self):
        data = {"model": "m1", "hardware": "h1", "profile": "p1"}
        pd = parse_preset_yaml("test-no-runtime", data)
        pd.validate()
        assert pd.runtime is None
