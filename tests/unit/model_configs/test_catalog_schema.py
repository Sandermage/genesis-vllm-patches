# SPDX-License-Identifier: Apache-2.0
"""CONFIG-UX.5.1 — tests for `vllm/sndr_core/model_configs/catalog_schema.py`."""
from __future__ import annotations

import pytest

from sndr.model_configs.catalog_schema import (
    MATCH_QUALITIES,
    ROW_TYPES,
    SCHEMA_VERSION,
    BaselineRow,
    CatalogRowBase,
    HardwareRow,
    ModelRow,
    PresetRow,
    ProfileRow,
    RedactedEvidenceRef,
    is_private_visibility,
    is_redactable_path,
)


class TestSchemaVersioning:
    def test_schema_version_is_one(self):
        assert SCHEMA_VERSION == 1

    def test_row_types_complete(self):
        assert set(ROW_TYPES) == {
            "preset", "profile", "model", "hardware", "baseline",
        }

    def test_match_qualities_complete(self):
        assert set(MATCH_QUALITIES) == {
            "exact_preset", "model_only", "family_only", "none",
        }


class TestRedactionHelper:
    @pytest.mark.parametrize("path", [
        "sndr_private/runs/foo.md",
        "sndr_private/planning/audits/bar.md",
    ])
    def test_sndr_private_paths_redactable(self, path):
        assert is_redactable_path(path) is True

    @pytest.mark.parametrize("path", [
        "/Users/sander/Documents/foo",
        "/home/user/data",
        "/tmp/scratch",
        "/var/log/test",
    ])
    def test_absolute_local_paths_redactable(self, path):
        assert is_redactable_path(path) is True

    @pytest.mark.parametrize("path", [
        "tests/integration/baselines/27b_v11_wave9.json",
        "vllm/sndr_core/model_configs/builtin/hardware/single-3090-24gbvram.yaml",
        "docs/PATCHES.md",
        "external://docs.example/something",
        "external://club-3090/issues/58",
        "",
    ])
    def test_public_paths_not_redactable(self, path):
        assert is_redactable_path(path) is False

    def test_is_private_visibility(self):
        assert is_private_visibility("private") is True
        assert is_private_visibility("public") is False
        assert is_private_visibility("mixed") is False
        assert is_private_visibility(None) is False


class TestRedactedEvidenceRef:
    def test_default_shape(self):
        ref = RedactedEvidenceRef(type="bench")
        assert ref.redacted is True
        assert ref.visibility == "private"
        assert ref.type == "bench"
        assert "not exposed" in ref.note.lower()

    def test_immutable(self):
        ref = RedactedEvidenceRef(type="bench")
        with pytest.raises((TypeError, dataclasses_FrozenInstanceError := Exception)):
            ref.redacted = False  # type: ignore[misc]


class TestRowConstruction:
    """Smoke-test that each row dataclass can be instantiated with the
    fields the generator emits."""

    def test_preset_row(self):
        row = PresetRow(
            schema_version=1, row_type="preset", id="x",
            source_path="vllm/.../x.yaml", source_sha256="abc",
            status="production_candidate", family="rf",
            tags=[], updated_from_git_commit=None,
            generated_at="2026-05-24T00:00:00Z",
            model_id="m", hardware_id="h", profile_id="p",
            composed_key="m--h--p", composed_sha256="def",
            has_card=True,
            card_title="t", card_status="production_candidate",
            card_audience="operator", card_mode="throughput",
            card_workload_allow=["x"], card_workload_deny=["y"],
            card_K=4, card_routing_family="rf",
            card_default_for_family=False,
            card_fallback_preset="x-fallback",
            card_primary_metric_kind="agg_TPS",
            card_primary_metric_value=100.0,
            card_evidence_visibility="public",
            card_evidence_ref_count=0,
            card_evidence_refs=[],
        )
        assert row.id == "x"
        assert row.has_card is True

    def test_profile_row(self):
        row = ProfileRow(
            schema_version=1, row_type="profile", id="p",
            source_path="vllm/.../p.yaml", source_sha256="abc",
            status="experimental", family=None,
            tags=[], updated_from_git_commit=None,
            generated_at="2026-05-24T00:00:00Z",
            parent_model="m", role=None,
            sizing_max_model_len=131072,
            sizing_max_num_seqs=4,
            sizing_max_num_batched_tokens=4096,
            sizing_gpu_memory_utilization=0.92,
            sizing_enable_chunked_prefill=True,
            sizing_enforce_eager=False,
            has_override_policy=True,
            override_class="bench",
            override_reason="bench_pending: ...",
            override_evidence_ref_count=0,
            override_evidence_visibility="public",
            override_expires_at="2026-08-22",
            override_allowed_to_exceed_hardware_default=False,
            class4_clean=True,
            patches_enable_count=0,
            patches_disable_count=0,
            patches_override_count=0,
        )
        assert row.has_override_policy is True
        assert row.class4_clean is True

    def test_baseline_row(self):
        row = BaselineRow(
            schema_version=1, row_type="baseline", id="b",
            source_path="tests/.../b.json", source_sha256="abc",
            status=None, family=None,
            tags=[], updated_from_git_commit=None,
            generated_at="2026-05-24T00:00:00Z",
            bench_model="qwen3.6-27b",
            bench_vllm_pin="dev371",
            bench_ctx="8k",
            bench_max_tokens=1024,
            bench_prompts_set="short",
            bench_runs=5,
            match_quality="exact_preset",
            matched_model_ids=["qwen3.6-27b-int4-autoround-tq-k8v4"],
            matched_preset_ids=["prod-qwen3.6-27b-tq-k8v4"],
        )
        assert row.match_quality == "exact_preset"
