# SPDX-License-Identifier: Apache-2.0
"""Phase A — schema tests for ModelDef.patches_attribution.

The new optional `patches_attribution: dict[patch_id, PatchAttribution]`
block lives next to the existing `patches: dict[env_flag, value]` map.
Stored metadata explains WHY each patch is in the model's canonical set:

  role             one of {load_bearing, defensive, optional_perf,
                   suspected_regression, no_op, unknown}
  note             human rationale; required for load_bearing /
                   suspected_regression so reviewers see the why
  bench_evidence   short empirical anchor; required for optional_perf
                   so claims like "+5% TPS" don't sit unverified
  candidate_when   free-form predicate dict (model_class, cuda_capability_in,
                   max_num_seqs_gte, ...); parsed later by the resolver

Phase A keeps compose() / runtime untouched — this commit only stores
+ validates the metadata. Phase B will read it from `sndr patches plan`.

See `docs/_internal/PATCH_ATTRIBUTION_COMPOSE_GENERATOR_INTEGRATION_PLAN_2026-05-16_RU.md`
for the full multi-phase design and the inline-vs-separate-file
decision (we chose inline; this file enforces that choice).
"""
from __future__ import annotations

import pytest

from sndr.model_configs.schema import SchemaError
from sndr.model_configs.schema_v2 import (
    HardwareSpec,
    ModelDef,
    PatchAttribution,
)


# ─── Builders ────────────────────────────────────────────────────────────


def _make_model(**kw) -> ModelDef:
    base = dict(
        schema_version=2, kind="model", id="qwen3.6-attr-test",
        title="Test", maintainer="x", last_validated="2026-05-16",
        license="apache-2.0", model_path="/m",
    )
    base.update(kw)
    return ModelDef(**base)


# ─── Backwards compatibility ─────────────────────────────────────────────


class TestBackwardsCompat:
    def test_model_without_attribution_loads(self):
        """Old format (no patches_attribution field) stays valid."""
        m = _make_model()
        m.validate()
        assert m.patches_attribution == {}

    def test_empty_attribution_dict_loads(self):
        """An explicit empty dict is also fine."""
        m = _make_model(patches_attribution={})
        m.validate()
        assert m.patches_attribution == {}

    def test_attribution_does_not_break_patches_dict(self):
        """Old `patches: {flag: value}` still parses alongside new block."""
        m = _make_model(
            patches={"GENESIS_ENABLE_PN17": "1"},
            patches_attribution={
                "PN17": PatchAttribution(role="defensive", note="long-ctx safety"),
            },
        )
        m.validate()
        assert m.patches["GENESIS_ENABLE_PN17"] == "1"
        assert m.patches_attribution["PN17"].role == "defensive"


# ─── Role enum ───────────────────────────────────────────────────────────


class TestRoleEnum:
    @pytest.mark.parametrize("role", [
        "load_bearing",
        "defensive",
        "optional_perf",
        "suspected_regression",
        "no_op",
        "unknown",
    ])
    def test_valid_role_accepted(self, role):
        """Each enum value parses cleanly when paired with the
        required-by-role auxiliary fields."""
        if role in ("load_bearing", "suspected_regression"):
            entry = PatchAttribution(role=role, note=f"why {role}")
        elif role == "optional_perf":
            entry = PatchAttribution(role=role, bench_evidence=f"bench for {role}")
        else:
            entry = PatchAttribution(role=role)
        m = _make_model(patches_attribution={"PN17": entry})
        m.validate()
        assert m.patches_attribution["PN17"].role == role

    def test_invalid_role_raises(self):
        m = _make_model(patches_attribution={
            "PN17": PatchAttribution(role="bogus"),
        })
        with pytest.raises(SchemaError, match="role"):
            m.validate()


# ─── Role-conditional required fields ────────────────────────────────────


class TestRoleConditionalFields:
    def test_load_bearing_requires_note(self):
        m = _make_model(patches_attribution={
            "PN17": PatchAttribution(role="load_bearing"),
        })
        with pytest.raises(SchemaError, match="load_bearing.*note"):
            m.validate()

    def test_suspected_regression_requires_note(self):
        m = _make_model(patches_attribution={
            "PN134": PatchAttribution(role="suspected_regression"),
        })
        with pytest.raises(SchemaError, match="suspected_regression.*note"):
            m.validate()

    def test_optional_perf_requires_bench_evidence(self):
        m = _make_model(patches_attribution={
            "PN204": PatchAttribution(role="optional_perf"),
        })
        with pytest.raises(SchemaError, match="optional_perf.*bench_evidence"):
            m.validate()

    def test_defensive_no_required_aux(self):
        """defensive doesn't force note/evidence — it's the cheap default."""
        m = _make_model(patches_attribution={
            "PN17": PatchAttribution(role="defensive"),
        })
        m.validate()

    def test_no_op_no_required_aux(self):
        m = _make_model(patches_attribution={
            "PN32": PatchAttribution(role="no_op"),
        })
        m.validate()


# ─── Key must be a valid patch ID ────────────────────────────────────────


class TestKeyValidation:
    def test_uppercase_patch_id_accepted(self):
        m = _make_model(patches_attribution={
            "PN204": PatchAttribution(role="defensive"),
        })
        m.validate()

    def test_lowercase_key_rejected(self):
        m = _make_model(patches_attribution={
            "pn204": PatchAttribution(role="defensive"),
        })
        with pytest.raises(SchemaError, match=r"patches_attribution.*P\[N\]\?"):
            m.validate()

    def test_random_string_key_rejected(self):
        m = _make_model(patches_attribution={
            "RANDOM_THING": PatchAttribution(role="defensive"),
        })
        with pytest.raises(SchemaError, match=r"patches_attribution.*P\[N\]\?"):
            m.validate()


# ─── Free-form candidate_when stays free-form (Phase A) ──────────────────


class TestCandidateWhen:
    def test_candidate_when_dict_preserved(self):
        m = _make_model(patches_attribution={
            "PN204": PatchAttribution(
                role="optional_perf",
                bench_evidence="dev371 35B conc=8: neutral within CV (675 vs 689)",
                candidate_when={"max_num_seqs_gte": 4, "cuda_capability_in": [[9, 0]]},
            ),
        })
        m.validate()
        cw = m.patches_attribution["PN204"].candidate_when
        assert cw == {"max_num_seqs_gte": 4, "cuda_capability_in": [[9, 0]]}

    def test_candidate_when_none_by_default(self):
        m = _make_model(patches_attribution={
            "PN17": PatchAttribution(role="defensive"),
        })
        assert m.patches_attribution["PN17"].candidate_when is None


# ─── YAML round-trip via registry_v2 loader ──────────────────────────────


class TestYamlRoundTrip:
    def test_yaml_dict_form_loads(self, tmp_path):
        """YAML payload with patches_attribution parses through the same
        _dataclass_from_dict loader registry_v2.load_model() uses."""
        from sndr.model_configs.registry_v2 import _dataclass_from_dict

        data = {
            "schema_version": 2, "kind": "model",
            "id": "qwen3.6-yaml-roundtrip",
            "title": "T", "maintainer": "x",
            "last_validated": "2026-05-16", "license": "apache-2.0",
            "model_path": "/m",
            "patches": {"GENESIS_ENABLE_PN204": "0"},
            "patches_attribution": {
                "PN204": {
                    "role": "optional_perf",
                    "bench_evidence": "dev371 35B conc=8: 675 vs 689 TPS",
                    "note": "Enabled in qwen3.6-35b-multiconc profile.",
                    "candidate_when": {"max_num_seqs_gte": 4},
                },
            },
        }
        m = _dataclass_from_dict(ModelDef, data)
        m.validate()
        attr = m.patches_attribution["PN204"]
        assert attr.role == "optional_perf"
        assert "675 vs 689" in attr.bench_evidence
        assert attr.candidate_when == {"max_num_seqs_gte": 4}


# ─── Tracked builtin YAMLs still load (regression anchor) ────────────────


class TestCommittedYamlsLoad:
    @pytest.mark.parametrize("model_id", [
        "qwen3.6-27b-dflash",
        "qwen3.6-27b-int4-autoround-fp8kv",
        "qwen3.6-27b-int4-autoround-tq-k8v4",
        "qwen3.6-35b-a3b-fp8-dflash",
        "qwen3.6-35b-a3b-fp8",
        "qwen3.6-7b-dense",
    ])
    def test_every_committed_model_yaml_loads(self, model_id):
        """All 6 committed builtin model YAMLs must keep loading after
        the schema extension — Phase A is additive and non-breaking."""
        from sndr.model_configs.registry_v2 import load_model
        m = load_model(model_id)
        # patches_attribution is optional — committed YAMLs may have {} or
        # entries; either is fine. The point is .validate() did not raise.
        assert isinstance(m.patches_attribution, dict)
