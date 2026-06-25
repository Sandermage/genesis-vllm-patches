# SPDX-License-Identifier: Apache-2.0
"""Tests for V2 config editor Product API."""
from __future__ import annotations

from pathlib import Path

import pytest

from sndr.product_api.legacy.config_editor import (
    apply_v2_config_plan,
    apply_v2_layer,
    collect_v2_config_catalog,
    get_v2_layer,
    list_user_presets,
    plan_v2_config_edit,
    preview_v2_config,
)


def test_apply_v2_layer_writes_operator_local(monkeypatch, tmp_path):
    monkeypatch.setenv("SNDR_MODEL_CONFIG_DIR", str(tmp_path))
    yaml_text = "schema_version: 2\nkind: model\nid: gui-edit-model\ntitle: Edited\n"
    result = apply_v2_layer(kind="model", layer_id="gui-edit-model", yaml_text=yaml_text)
    assert result.status == "applied"
    assert result.written is True
    target = Path(result.target_path)
    assert target.is_file()
    assert target.parent == tmp_path / "model"
    assert target.read_text(encoding="utf-8").startswith("schema_version: 2")


def test_apply_v2_layer_backs_up_and_rejects_bad_kind(monkeypatch, tmp_path):
    monkeypatch.setenv("SNDR_MODEL_CONFIG_DIR", str(tmp_path))
    blocked = apply_v2_layer(kind="widget", layer_id="x", yaml_text="a: 1")
    assert blocked.status == "blocked"
    assert blocked.written is False

    (tmp_path / "hardware").mkdir(parents=True)
    target = tmp_path / "hardware" / "gui-edit-hw.yaml"
    target.write_text("# old\n", encoding="utf-8")
    result = apply_v2_layer(kind="hardware", layer_id="gui-edit-hw", yaml_text="schema_version: 2\n")
    assert result.action == "update"
    assert result.backup_path is not None
    assert Path(result.backup_path).read_text(encoding="utf-8") == "# old\n"


def test_get_v2_layer_returns_full_model_definition():
    layer = get_v2_layer("model", "qwen3.6-35b-a3b-fp8")
    assert layer["kind"] == "model"
    assert layer["id"] == "qwen3.6-35b-a3b-fp8"
    definition = layer["definition"]
    assert definition["model_path"]
    assert definition["capabilities"]["attention_arch"]
    assert isinstance(definition["patches"], dict) and definition["patches"]
    assert definition["requires"]["min_gpu_count"] >= 1
    assert layer["source"].endswith("qwen3.6-35b-a3b-fp8.yaml")


def test_get_v2_layer_unknown_kind_raises():
    with pytest.raises(ValueError):
        get_v2_layer("widget", "whatever")


def test_get_v2_layer_supports_all_element_kinds():
    hardware = get_v2_layer("hardware", "a5000-2x-24gbvram-16cpu-128gbram")
    assert hardware["definition"]["sizing"]["max_model_len"] >= 1
    profile = get_v2_layer("profile", "qwen3.6-35b-multiconc")
    assert profile["definition"]["parent_model"]
    preset = get_v2_layer("preset", "prod-qwen3.6-35b-multiconc")
    assert preset["definition"]["model"] == "qwen3.6-35b-a3b-fp8"


def _use_temp_config_dir(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("SNDR_MODEL_CONFIG_DIR", str(tmp_path))
    return tmp_path


def test_collect_v2_config_catalog_returns_all_layers():
    catalog = collect_v2_config_catalog()

    assert len(catalog.models) >= 1
    assert len(catalog.hardware) >= 1
    assert len(catalog.profiles) >= 1
    assert len(catalog.presets) >= 1
    assert any(item.id == "qwen3.6-35b-a3b-fp8" for item in catalog.models)
    assert any(item.id == "qwen3.6-35b-multiconc" for item in catalog.profiles)
    assert any(item.id == "prod-qwen3.6-35b-multiconc" for item in catalog.presets)

    model = next(item for item in catalog.models if item.id == "qwen3.6-35b-a3b-fp8")
    assert model.kind == "model"
    assert model.fields["patch_count"] >= 1
    assert model.source.endswith("qwen3.6-35b-a3b-fp8.yaml")
    # Widened projection: the required pin + spec-decode method/drafter must reach
    # the catalog (so config views show pin alignment without the v2Layer call).
    assert model.fields["vllm_pin_required"] == "0.23.1rc1.dev424+g3f5a1e173"
    assert model.fields["spec_decode_method"] == "mtp"
    assert "spec_decode_drafter" in model.fields
    assert "reference_metrics_ref" in model.fields


def test_preview_v2_config_composes_selection_and_draft_yaml():
    preview = preview_v2_config(
        model_id="qwen3.6-35b-a3b-fp8",
        hardware_id="a5000-2x-24gbvram-16cpu-128gbram",
        profile_id="qwen3.6-35b-multiconc",
        runtime="docker",
    )

    assert preview.compatible is True
    assert preview.status == "ready"
    assert preview.selection["profile"] == "qwen3.6-35b-multiconc"
    assert preview.composed["max_num_seqs"] == 8
    assert "profile: qwen3.6-35b-multiconc" in preview.draft_yaml


def test_preview_v2_config_blocks_profile_parent_mismatch():
    preview = preview_v2_config(
        model_id="gemma-4-26b-a4b-it-awq",
        hardware_id="a5000-2x-24gbvram-16cpu-128gbram",
        profile_id="qwen3.6-35b-multiconc",
    )

    assert preview.compatible is False
    assert preview.status == "blocked"
    assert preview.composed == {}
    assert "parent_model does not match" in preview.messages[0]


def test_plan_v2_config_edit_returns_read_only_diff_plan():
    plan = plan_v2_config_edit(
        preset_id="gui-draft-qwen3.6-35b-multiconc",
        model_id="qwen3.6-35b-a3b-fp8",
        hardware_id="a5000-2x-24gbvram-16cpu-128gbram",
        profile_id="qwen3.6-35b-multiconc",
        runtime="docker",
    )

    assert plan.valid is True
    assert plan.read_only is True
    assert plan.apply_enabled is False
    assert plan.action in {"create", "update"}
    assert plan.target_path.endswith("gui-draft-qwen3.6-35b-multiconc.yaml")
    assert any(line.startswith("+model: qwen3.6-35b-a3b-fp8") for line in plan.diff_lines)
    assert "profile: qwen3.6-35b-multiconc" in plan.draft_yaml


def test_apply_v2_config_plan_creates_operator_local_file(monkeypatch, tmp_path):
    _use_temp_config_dir(monkeypatch, tmp_path)

    result = apply_v2_config_plan(
        preset_id="gui-draft-qwen3.6-35b-multiconc",
        model_id="qwen3.6-35b-a3b-fp8",
        hardware_id="a5000-2x-24gbvram-16cpu-128gbram",
        profile_id="qwen3.6-35b-multiconc",
        runtime="docker",
    )

    assert result.status == "applied"
    assert result.written is True
    assert result.action == "create"
    assert result.backup_path is None
    target = Path(result.target_path)
    assert target.is_file()
    assert target.parent == tmp_path / "presets"
    assert "model: qwen3.6-35b-a3b-fp8" in target.read_text(encoding="utf-8")
    # Apply must stay inside the operator-local config dir.
    assert str(target).startswith(str(tmp_path))


def test_apply_v2_config_plan_backs_up_existing_file(monkeypatch, tmp_path):
    _use_temp_config_dir(monkeypatch, tmp_path)
    presets_dir = tmp_path / "presets"
    presets_dir.mkdir(parents=True)
    target = presets_dir / "gui-draft-qwen3.6-35b-multiconc.yaml"
    target.write_text("# previous content\n", encoding="utf-8")

    result = apply_v2_config_plan(
        preset_id="gui-draft-qwen3.6-35b-multiconc",
        model_id="qwen3.6-35b-a3b-fp8",
        hardware_id="a5000-2x-24gbvram-16cpu-128gbram",
        profile_id="qwen3.6-35b-multiconc",
        runtime="docker",
    )

    assert result.status == "applied"
    assert result.action == "update"
    assert result.backup_path is not None
    backup = Path(result.backup_path)
    assert backup.is_file()
    assert backup.read_text(encoding="utf-8") == "# previous content\n"
    assert "model: qwen3.6-35b-a3b-fp8" in target.read_text(encoding="utf-8")


def test_apply_v2_config_plan_refuses_invalid_selection(monkeypatch, tmp_path):
    _use_temp_config_dir(monkeypatch, tmp_path)

    result = apply_v2_config_plan(
        preset_id="gui-draft-bad",
        model_id="gemma-4-26b-a4b-it-awq",
        hardware_id="a5000-2x-24gbvram-16cpu-128gbram",
        profile_id="qwen3.6-35b-multiconc",
    )

    assert result.status == "blocked"
    assert result.written is False
    assert result.blocked_reasons
    assert not (tmp_path / "presets" / "gui-draft-bad.yaml").exists()


def test_apply_v2_config_plan_refuses_plan_id_mismatch(monkeypatch, tmp_path):
    _use_temp_config_dir(monkeypatch, tmp_path)

    result = apply_v2_config_plan(
        preset_id="gui-draft-qwen3.6-35b-multiconc",
        model_id="qwen3.6-35b-a3b-fp8",
        hardware_id="a5000-2x-24gbvram-16cpu-128gbram",
        profile_id="qwen3.6-35b-multiconc",
        runtime="docker",
        expected_plan_id="cfgplan_stale000000",
    )

    assert result.status == "conflict"
    assert result.written is False
    assert not (tmp_path / "presets" / "gui-draft-qwen3.6-35b-multiconc.yaml").exists()


def test_list_user_presets_returns_applied_drafts(monkeypatch, tmp_path):
    _use_temp_config_dir(monkeypatch, tmp_path)
    apply_v2_config_plan(
        preset_id="gui-draft-qwen3.6-35b-multiconc",
        model_id="qwen3.6-35b-a3b-fp8",
        hardware_id="a5000-2x-24gbvram-16cpu-128gbram",
        profile_id="qwen3.6-35b-multiconc",
        runtime="docker",
    )

    presets = list_user_presets()

    assert len(presets) == 1
    record = presets[0]
    assert record.id == "gui-draft-qwen3.6-35b-multiconc"
    assert record.model == "qwen3.6-35b-a3b-fp8"
    assert record.profile == "qwen3.6-35b-multiconc"
    assert record.path.endswith("gui-draft-qwen3.6-35b-multiconc.yaml")


def test_list_user_presets_empty_when_no_user_dir(monkeypatch, tmp_path):
    _use_temp_config_dir(monkeypatch, tmp_path)
    assert list_user_presets() == ()
