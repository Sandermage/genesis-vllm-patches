# SPDX-License-Identifier: Apache-2.0
"""Tests for the read-only service lifecycle plan."""
from __future__ import annotations

import pytest

from sndr.product_api.legacy.presets import PresetNotFoundError
from sndr.product_api.legacy.service_plan import build_service_plan


def test_build_service_plan_status_is_read_only():
    plan = build_service_plan(preset_id="prod-qwen3.6-35b-multiconc", action="status")
    assert plan.action == "status"
    assert plan.mutating is False
    assert plan.steps
    assert plan.cli_mirror
    assert plan.container_name == "vllm-prod-qwen3.6-35b-multiconc"


def test_build_service_plan_start_is_gated():
    plan = build_service_plan(
        preset_id="prod-qwen3.6-35b-multiconc",
        action="start",
        runtime_target="docker_compose",
    )
    assert plan.mutating is True
    assert plan.actionable is False  # write API not enabled
    assert plan.side_effects
    assert any("up" in step.command or "start" in step.command for step in plan.steps)
    assert plan.plan_id.startswith("svcplan_")


def test_build_service_plan_unknown_action_raises():
    with pytest.raises(ValueError):
        build_service_plan(preset_id="prod-qwen3.6-35b-multiconc", action="frobnicate")


def test_build_service_plan_unknown_preset_raises():
    with pytest.raises(PresetNotFoundError):
        build_service_plan(preset_id="not-a-real-preset", action="status")
