# SPDX-License-Identifier: Apache-2.0
"""Tests for the persistent job store + service apply."""
from __future__ import annotations

import pytest

from sndr.product_api.legacy import jobs as jobstore
from sndr.product_api.legacy.jobs import apply_service_action, get_job, list_jobs
from sndr.product_api.legacy.presets import PresetNotFoundError


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    # Isolate the persistent store per test so counts/ordering are deterministic.
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    jobstore._reset_state()
    yield
    jobstore._reset_state()


def test_apply_service_action_creates_dry_run_job():
    job = apply_service_action(preset_id="prod-qwen3.6-35b-multiconc", action="start")
    assert job.kind == "service.start"
    assert job.status == "dry_run"
    assert job.dry_run is True
    assert job.steps
    assert all(step.command for step in job.steps)
    assert job.job_id.startswith("job_")
    # the store returns an equal (persisted) job by id
    assert get_job(job.job_id) == job
    assert any(j.job_id == job.job_id for j in list_jobs())


def test_apply_service_action_unknown_action_raises():
    with pytest.raises(ValueError):
        apply_service_action(preset_id="prod-qwen3.6-35b-multiconc", action="frobnicate")


def test_apply_service_action_unknown_preset_raises():
    with pytest.raises(PresetNotFoundError):
        apply_service_action(preset_id="not-a-real-preset", action="status")


def test_list_jobs_is_newest_first():
    apply_service_action(preset_id="prod-qwen3.6-35b-multiconc", action="status")
    apply_service_action(preset_id="prod-qwen3.6-35b-multiconc", action="restart")
    listed = list_jobs()
    assert len(listed) == 2
    assert listed[0].kind == "service.restart"
