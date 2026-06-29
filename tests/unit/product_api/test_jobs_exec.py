# SPDX-License-Identifier: Apache-2.0
"""Tests for persistent jobs/events + the live background executor."""
from __future__ import annotations

import pytest

from sndr.product_api.legacy import background_exec, jobs


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    jobs._reset_state()
    yield tmp_path
    jobs._reset_state()


# ---- persistence ----

def test_jobs_persist_across_restart(home):
    job = jobs.create_dry_run_job(kind="x", title="t", summary={}, steps=[("s", "echo hi")], cli_mirror=["echo hi"])
    assert (home / "state" / "jobs.json").exists()
    # simulate a restart: drop the in-memory cache and reload from disk
    jobs._reset_state()
    reloaded = jobs.get_job(job.job_id)
    assert reloaded is not None and reloaded.title == "t"


def test_events_persist_and_seq_is_monotonic(home):
    jobs.record_event("auth", "first")
    jobs.record_event("auth", "second")
    jobs._reset_state()
    events = jobs.list_events(since_seq=0)
    assert [e["message"] for e in events][-2:] == ["first", "second"]
    # seq keeps climbing after reload (no reuse)
    e3 = jobs.record_event("auth", "third")
    assert e3["seq"] == 3


def test_update_job_changes_status_and_log(home):
    job = jobs.create_running_job(kind="x", title="t", summary={}, command="echo hi")
    updated = jobs.update_job(job.job_id, status="succeeded", append_log="line2", progress=100.0)
    assert updated.status == "succeeded"
    assert "line2" in updated.log and updated.progress == 100.0
    jobs._reset_state()
    assert jobs.get_job(job.job_id).status == "succeeded"


# ---- background executor ----

def test_background_command_runs_and_succeeds(home):
    job = background_exec.run_background_command(
        kind="model.download", title="dl", summary={},
        command="printf 'Downloading 10%%\\nDownloading 100%%\\n'", _spawn=False,
    )
    final = jobs.get_job(job.job_id)
    assert final.status == "succeeded"
    assert any("Downloading" in line for line in final.log)
    assert final.progress == 100.0


def test_background_command_captures_failure(home):
    job = background_exec.run_background_command(
        kind="x", title="boom", summary={}, command="echo before; exit 3", _spawn=False,
    )
    final = jobs.get_job(job.job_id)
    assert final.status == "failed"
    assert any("[exit 3]" in line for line in final.log)


def test_progress_extracted_from_output(home):
    background_exec.run_background_command(
        kind="x", title="p", summary={}, command="printf 'step 42%%\\n'; exit 1", _spawn=False,
    )
    job = jobs.list_jobs()[0]
    # 42% was parsed during the run even though it ultimately failed
    assert any("42%" in line for line in job.log)


# ---- model download route ----

def test_download_dry_run_when_apply_off(home, monkeypatch):
    pytest.importorskip("fastapi")
    monkeypatch.delenv("SNDR_ENABLE_APPLY", raising=False)
    from fastapi.testclient import TestClient
    from sndr.product_api.legacy.http_app import create_app

    client = TestClient(create_app(enable_apply=False, allowed_origins=()))
    # invalid id -> 400; unknown id -> 404
    assert client.post("/api/v1/models/download", json={"model_id": "evil; rm -rf /"}).status_code == 400
    assert client.post("/api/v1/models/download", json={"model_id": "no-such-model-xyz"}).status_code == 404


def test_download_executes_when_apply_on(home, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from sndr.product_api.legacy import http_app
    from sndr.product_api.legacy.http_app import create_app

    # Pretend one model id is known + stub the background runner so no real pull.
    captured = {}
    monkeypatch.setattr(
        http_app, "collect_model_cache_report",
        lambda: type("R", (), {"models": [type("E", (), {"model_id": "known-model"})()]})(),
    )
    monkeypatch.setattr(
        "sndr.product_api.legacy.background_exec.run_background_command",
        lambda **kw: (captured.update(kw) or jobs.create_running_job(kind=kw["kind"], title=kw["title"], summary=kw["summary"], command=kw["command"])),
    )
    client = TestClient(create_app(enable_apply=True, allowed_origins=()))
    resp = client.post("/api/v1/models/download", json={"model_id": "known-model"})
    assert resp.status_code == 200 and resp.json()["status"] == "running"
    assert "model pull known-model" in captured["command"]


def test_background_command_reaps_child_when_update_job_raises(home, monkeypatch):
    """If update_job raises mid-stream (e.g. disk full), the child must STILL be
    reaped (proc.wait) and its stdout closed — no zombie process, no leaked fd,
    and the exception must not escape the worker."""
    import sndr.product_api.legacy.background_exec as bx

    reaped = {"wait": 0, "closed": False}

    class _FakeStdout:
        closed = False

        def __iter__(self):
            return iter(["line1\n", "line2\n"])

        def close(self):
            reaped["closed"] = True
            self.closed = True

    class _FakeProc:
        def __init__(self):
            self.stdout = _FakeStdout()

        def wait(self, timeout=None):
            reaped["wait"] += 1
            return 0

        def poll(self):
            return 0 if reaped["wait"] else None

        def kill(self):
            pass

    monkeypatch.setattr(bx.subprocess, "Popen", lambda *a, **k: _FakeProc())

    real_update = bx.update_job
    state = {"n": 0}

    def _boom(*a, **k):
        state["n"] += 1
        if state["n"] == 1:  # first flush (mid-loop, flush_interval=0) → blow up
            raise OSError("disk full")
        return real_update(*a, **k)

    monkeypatch.setattr(bx, "update_job", _boom)

    # Must not raise out of the worker, and must reap the child.
    bx.run_background_command(kind="x", title="t", summary={}, command="echo hi",
                             _spawn=False, flush_interval=0.0)
    assert reaped["wait"] >= 1, "child must be reaped (proc.wait) even when update_job raises"
    assert reaped["closed"], "child stdout must be closed"
