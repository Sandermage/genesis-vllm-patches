# SPDX-License-Identifier: Apache-2.0
"""Tests for preset catalog Product API used by GUI/web callers."""
from __future__ import annotations

import sys
from dataclasses import asdict, is_dataclass

import pytest

from vllm.sndr_core.product_api import presets
from vllm.sndr_core.product_api.presets import (
    PresetListResult,
    PresetRecommendResult,
    PresetRecord,
    UnknownWorkloadError,
)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    # Count only the builtin catalog — no operator-local presets from a shared
    # $SNDR_HOME — so counts are deterministic regardless of test order.
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))


def test_list_presets_returns_catalog_records():
    result = presets.list_presets()

    # chat-K3 promotion session (2026-06-01): +2 preset aliases
    # (prod-gemma4-31b-tq-mtp-chat-k3 + prod-gemma4-26b-mtp-chat-k3
    # promoted from profile-only to operator-facing presets) → 21 → 23.
    assert isinstance(result, PresetListResult)
    assert result.total == 23
    assert result.matched == 23
    assert result.load_errors == ()
    assert all(isinstance(row, PresetRecord) for row in result.presets)

    # Every builtin preset now carries a card (bench-and-update annotated the
    # last unannotated ones); the catalog has no unannotated presets.
    assert all(row.has_card for row in result.presets)


def test_list_presets_filters_status_and_family():
    by_status = presets.list_presets(status="production_candidate")
    by_family = presets.list_presets(family="qwen3_6_35b_a3b_fp8")

    # chat-K3 promotion (2026-06-01): the two new presets ship as
    # production_candidate, lifting the production_candidate-filter
    # count by 2 (14 → 16).
    assert by_status.matched == 16
    assert by_family.matched == 2
    assert {row.id for row in by_family.presets} == {
        "prod-qwen3.6-35b-balanced",
        "prod-qwen3.6-35b-multiconc",
    }


def test_get_preset_and_drill_field_are_json_safe():
    record = presets.get_preset("prod-qwen3.6-35b-balanced")

    assert isinstance(record, PresetRecord)
    assert record.has_card is True
    assert record.card["status"] == "production_candidate"
    assert presets.drill_field(asdict(record), "card.routing_family") == (
        "qwen3_6_35b_a3b_fp8"
    )
    assert "35b_v11_wave9.json" in presets.drill_field(
        asdict(record),
        "card.evidence_refs.0.path",
    )


def test_drill_field_reports_invalid_paths():
    record = asdict(presets.get_preset("prod-qwen3.6-35b-balanced"))

    with pytest.raises(KeyError, match="nonexistent"):
        presets.drill_field(record, "card.nonexistent")
    with pytest.raises(KeyError, match="out of range"):
        presets.drill_field(record, "card.evidence_refs.999.path")


def test_explain_preset_returns_composed_runtime_summary():
    result = presets.explain_preset("prod-qwen3.6-35b-multiconc")

    assert is_dataclass(result)
    payload = asdict(result)
    assert payload["id"] == "prod-qwen3.6-35b-multiconc"
    assert payload["card"]["status"] == "production_candidate"
    assert payload["composed"]["max_num_seqs"] == 8
    assert "enabled_patches_count" in payload["composed"]


def test_recommend_presets_honors_allow_deny_and_ranking():
    result = presets.recommend_presets(
        workload="free_chat",
        hardware="a5000-2x-24gbvram-16cpu-128gbram",
        concurrency=8,
        top=5,
    )

    assert isinstance(result, PresetRecommendResult)
    ids = [row.id for row in result.results]
    assert "prod-qwen3.6-35b-multiconc" in ids
    assert "prod-gemma4-26b-mtp-k4" not in ids
    assert "prod-gemma4-26b-multiconc" not in ids
    assert [row.rank for row in result.results] == list(
        range(1, len(result.results) + 1)
    )
    assert result.total_candidates >= result.total_matches


def test_recommend_presets_rejects_unknown_workload():
    with pytest.raises(UnknownWorkloadError):
        presets.recommend_presets(workload="freechat")


def test_recommend_presets_accepts_custom_workload_with_no_matches():
    result = presets.recommend_presets(workload="custom:my-special-task")

    assert result.results == ()
    assert result.total_matches == 0


def test_presets_product_api_does_not_import_torch():
    if "torch" in sys.modules:
        pytest.skip("torch already imported by another test/runtime")

    presets.list_presets(status="production_candidate")
    presets.recommend_presets(workload="structured_json.short", top=2)

    assert "torch" not in sys.modules
