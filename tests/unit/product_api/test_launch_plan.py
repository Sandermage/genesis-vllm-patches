# SPDX-License-Identifier: Apache-2.0
"""Tests for the read-only GUI launch plan Product API."""
from __future__ import annotations

from sndr.product_api.legacy.launch_plan import build_launch_plan


def test_build_launch_plan_returns_backend_owned_artifacts_and_gates():
    plan = build_launch_plan(
        preset_id="prod-qwen3.6-35b-multiconc",
        runtime_target="docker_compose",
        patch_policy="safe",
        host="gpu-build-01",
        mode="remote",
    )

    # plan_id slug derives from the preset id; the qwen3.6 → qwen3_6
    # mass-rename (commit 91daa11d) carried into the slug, so the
    # `prod-qwen3.6-35b-multiconc` preset id now slugs as
    # `plan_prod_qwen3_6_35b_multiconc_<hash>`.
    assert plan.plan_id.startswith("plan_prod_qwen3_6_35b_multiconc_")
    assert plan.preset_id == "prod-qwen3.6-35b-multiconc"
    assert plan.runtime_target == "docker_compose"
    assert plan.patch_policy == "safe"
    assert plan.host == "gpu-build-01"
    assert plan.summary["model"] == "qwen3.6-35b-a3b-fp8"
    assert plan.summary["max_num_seqs"] == 8
    assert plan.summary["enabled_patches_count"] >= 1

    artifact_kinds = {artifact.kind for artifact in plan.artifacts}
    assert artifact_kinds == {"compose", "systemd", "commands", "env"}
    compose = next(
        artifact.content for artifact in plan.artifacts
        if artifact.kind == "compose"
    )
    assert "SNDR_PLAN_ID" in compose
    assert "prod-qwen3.6-35b-multiconc" in compose

    gate_status = {gate.id: gate.status for gate in plan.gates}
    assert gate_status["catalog"] == "pass"
    assert gate_status["patch_doctor"] == "pass"
    # Lifecycle plan/apply API is implemented now → gate passes; proof is a
    # recommendation (warning), not a hard blocker.
    assert gate_status["service_lifecycle"] == "pass"
    assert gate_status["release_proof"] == "warning"
    # No blocked gates remain → the plan is actionable (execution still gated by
    # --enable-apply + confirm at the apply endpoint).
    assert plan.actionable is True
    assert any("launch plan" in line for line in plan.cli_mirror)


def test_launch_plan_id_changes_with_target_inputs():
    compose_plan = build_launch_plan(
        preset_id="prod-qwen3.6-35b-multiconc",
        runtime_target="docker_compose",
        patch_policy="safe",
    )
    local_plan = build_launch_plan(
        preset_id="prod-qwen3.6-35b-multiconc",
        runtime_target="local_bare_metal",
        patch_policy="minimal",
    )

    assert compose_plan.plan_id != local_plan.plan_id
    assert compose_plan.summary["routing_family"] == local_plan.summary["routing_family"]
