# SPDX-License-Identifier: Apache-2.0
"""Operator-local preset awareness in the V2 registry.

Presets written by the GUI (configs/v2/apply) land in
``model_configs_user_dir()/presets``. They must then participate in catalog
listing and compose resolution, closing the GUI edit loop. Model/hardware/
profile layer resolution is intentionally NOT user-overridden here (it would
risk silently shadowing builtin runtime layers).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sndr.model_configs import registry_v2


def _write_user_preset(root: Path, alias: str, model: str, hardware: str) -> None:
    presets = root / "presets"
    presets.mkdir(parents=True, exist_ok=True)
    (presets / f"{alias}.yaml").write_text(
        f"model: {model}\nhardware: {hardware}\n", encoding="utf-8"
    )


def _compatible_model_hardware() -> tuple[str, str]:
    """Reuse an existing builtin preset's pair so compose VRAM checks pass."""
    for alias in registry_v2.list_presets():
        preset = registry_v2.load_preset_def(alias)
        if preset.model and preset.hardware:
            return preset.model, preset.hardware
    raise AssertionError("no builtin preset with model+hardware found")


def test_user_preset_listed_and_resolvable(tmp_path, monkeypatch):
    monkeypatch.setenv("SNDR_MODEL_CONFIG_DIR", str(tmp_path))
    # Reference a real builtin pair so compose succeeds.
    model, hardware = _compatible_model_hardware()
    _write_user_preset(tmp_path, "gui-user-test", model, hardware)

    ids = registry_v2.list_presets()
    assert "gui-user-test" in ids
    # Builtin presets are still listed (additive union).
    assert any(i != "gui-user-test" for i in ids)

    preset = registry_v2.load_preset_def("gui-user-test")
    assert preset.model == model
    assert preset.hardware == hardware
    # Full compose resolves through builtin layers.
    composed = registry_v2.load_alias("gui-user-test")
    assert composed is not None


def test_user_preset_overrides_builtin_same_id(tmp_path, monkeypatch):
    monkeypatch.setenv("SNDR_MODEL_CONFIG_DIR", str(tmp_path))
    builtin_ids = registry_v2.list_presets()
    assert builtin_ids, "expected builtin presets"
    target = builtin_ids[0]
    model = registry_v2.list_models()[0]
    hardware = registry_v2.list_hardware()[0]
    _write_user_preset(tmp_path, target, model, hardware)
    # User dir takes precedence for an operator-edited preset of the same id.
    preset = registry_v2.load_preset_def(target)
    assert preset.model == model
    assert preset.hardware == hardware


def test_no_user_dir_is_builtin_only(tmp_path, monkeypatch):
    # Pointing at an empty user dir must not drop builtin presets.
    monkeypatch.setenv("SNDR_MODEL_CONFIG_DIR", str(tmp_path))
    assert len(registry_v2.list_presets()) >= 1
    with pytest.raises(Exception):
        registry_v2.load_preset_def("definitely-not-a-preset")
