# SPDX-License-Identifier: Apache-2.0
"""Phase D extension — profile-level attribution override.

ModelDef.patches_attribution declares the canonical role/note/evidence
for each patch in the model's matrix. But operator scenarios vary:

  * The "long-ctx" profile may need PN204 (currently optional_perf on
    the latency profile) marked load_bearing because OOM risk grows
    with context length.
  * The "experimental-AB" profile may downgrade PN125 from defensive
    to no_op while it's being A/B-tested against the latest pin.

Forcing operators to fork the model YAML for these scenarios defeats
the V2 layering — that's exactly what ProfileDef.patches_delta exists
for. Phase D extends PatchesDelta with an ``attribution`` field that
overrides (or adds to) the model's attribution at compose time.

Semantics:

  - compose() merges model.patches_attribution first, then applies
    profile.patches_delta.attribution as a per-key OVERRIDE (full
    replacement of the entry, not a field-level merge).
  - The override key MUST still be a valid patch ID
    (_check_patch_id() runs at .validate() time).
  - Attribution-only profiles (no enable/disable/override) compose
    cleanly — the merged ModelConfig carries the augmented map.
"""
from __future__ import annotations

import pytest

from sndr.model_configs.compose import compose
from sndr.model_configs.schema import (
    HardwareSpec,
    PatchAttribution,
    SchemaError,
)
from sndr.model_configs.schema_v2 import (
    HardwareDef,
    HardwareSizing,
    ModelCapabilities,
    ModelDef,
    PatchesDelta,
    ProfileDef,
    RuntimeBlock,
    RuntimeDockerBlock,
)


def _stub_hw() -> HardwareDef:
    return HardwareDef(
        schema_version=2, kind="hardware", id="stub-hw",
        title="t", maintainer="x",
        hardware=HardwareSpec(
            gpu_match_keys=["g"], n_gpus=1, min_vram_per_gpu_mib=8000,
        ),
        runtime=RuntimeBlock(
            default="docker", supported=["docker"],
            docker=RuntimeDockerBlock(image="x:stub"),
        ),
        sizing=HardwareSizing(
            max_model_len=4096, max_num_seqs=1,
            max_num_batched_tokens=2048, gpu_memory_utilization=0.9,
        ),
    )


def _stub_model(patches_attribution=None) -> ModelDef:
    return ModelDef(
        schema_version=2, kind="model", id="stub-m",
        title="t", maintainer="x", last_validated="2026-05-16",
        license="apache-2.0", model_path="/m",
        capabilities=ModelCapabilities(attention_arch="dense"),
        patches={"GENESIS_ENABLE_PN204": "1"},
        patches_attribution=patches_attribution or {},
    )


def _profile(attribution=None, **kw) -> ProfileDef:
    delta_kw = {"attribution": attribution} if attribution is not None else {}
    return ProfileDef(
        schema_version=2, kind="profile", id="stub-prof",
        parent_model="stub-m", maintainer="x",
        patches_delta=PatchesDelta(**delta_kw),
        **kw,
    )


# ─── PatchesDelta schema ─────────────────────────────────────────────────


class TestSchemaAcceptsAttributionField:
    def test_empty_attribution_default(self):
        d = PatchesDelta()
        d.validate()
        assert d.attribution == {}

    def test_attribution_accepts_valid_entries(self):
        d = PatchesDelta(attribution={
            "PN204": PatchAttribution(
                role="load_bearing", note="long-ctx OOM hedge"),
        })
        d.validate()
        assert "PN204" in d.attribution
        assert d.attribution["PN204"].role == "load_bearing"

    def test_attribution_validates_role_enum(self):
        d = PatchesDelta(attribution={
            "PN204": PatchAttribution(role="bogus_role"),
        })
        with pytest.raises(SchemaError, match="role"):
            d.validate()

    def test_attribution_key_must_be_valid_patch_id(self):
        d = PatchesDelta(attribution={
            "lowercase_id": PatchAttribution(role="defensive"),
        })
        with pytest.raises(SchemaError, match=r"P\[N\]\?"):
            d.validate()


# ─── compose() merges profile attribution onto model attribution ─────────


class TestComposeAttributionMerge:
    def test_model_only_attribution_passes_through(self):
        m = _stub_model(patches_attribution={
            "PN204": PatchAttribution(role="optional_perf",
                                       bench_evidence="model-level"),
        })
        cfg = compose(m, _stub_hw(), profile=_profile())
        assert cfg.patches_attribution["PN204"].bench_evidence == "model-level"

    def test_profile_attribution_overrides_model(self):
        m = _stub_model(patches_attribution={
            "PN204": PatchAttribution(role="optional_perf",
                                       bench_evidence="model bench"),
        })
        prof = _profile(attribution={
            "PN204": PatchAttribution(role="load_bearing",
                                       note="overridden by long-ctx profile"),
        })
        cfg = compose(m, _stub_hw(), profile=prof)
        # Full replacement, not field merge — note from profile, no
        # bench_evidence inherited from model.
        attr = cfg.patches_attribution["PN204"]
        assert attr.role == "load_bearing"
        assert attr.note == "overridden by long-ctx profile"
        assert attr.bench_evidence is None

    def test_profile_attribution_adds_new_entry(self):
        m = _stub_model()  # no attribution at model level
        prof = _profile(attribution={
            "PN17": PatchAttribution(
                role="defensive", note="added by profile"),
        })
        cfg = compose(m, _stub_hw(), profile=prof)
        assert "PN17" in cfg.patches_attribution
        assert cfg.patches_attribution["PN17"].note == "added by profile"

    def test_model_attribution_not_in_profile_survives(self):
        m = _stub_model(patches_attribution={
            "PN17": PatchAttribution(role="defensive", note="model"),
            "PN204": PatchAttribution(role="optional_perf",
                                       bench_evidence="m-bench"),
        })
        prof = _profile(attribution={
            "PN204": PatchAttribution(
                role="suspected_regression",
                note="profile flagged this for now"),
        })
        cfg = compose(m, _stub_hw(), profile=prof)
        # PN17 unchanged (not in profile delta).
        assert cfg.patches_attribution["PN17"].role == "defensive"
        # PN204 fully replaced by profile entry.
        assert cfg.patches_attribution["PN204"].role == "suspected_regression"


# ─── Resolver sees the merged attribution ────────────────────────────────


class TestResolverSeesProfileOverride:
    def test_minimal_policy_drops_suspected_regression_via_profile(self):
        """End-to-end: model says PN204 is optional_perf with bench;
        profile-A flags it suspected_regression for an A/B period. Under
        --policy minimal, the profile-augmented attribution wins → PN204
        gets dropped."""
        from sndr.model_configs.patch_plan import resolve_patch_plan
        m = _stub_model(patches_attribution={
            "PN204": PatchAttribution(role="optional_perf",
                                       bench_evidence="m-bench"),
        })
        prof = _profile(attribution={
            "PN204": PatchAttribution(
                role="suspected_regression",
                note="A/B period — disable in minimal"),
        })
        cfg = compose(m, _stub_hw(), profile=prof)
        plan = resolve_patch_plan(cfg, policy="minimal")
        included_ids = {d.patch_id for d in plan.included}
        excluded_ids = {d.patch_id for d in plan.excluded}
        assert "PN204" in excluded_ids
        assert "PN204" not in included_ids
