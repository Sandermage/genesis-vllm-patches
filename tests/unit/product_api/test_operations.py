# SPDX-License-Identifier: Apache-2.0
"""Tests for the project Operations console — curated, server-defined commands
that surface sndr_core's canonical maintenance/diagnostic workflows as GUI jobs.

Safety contract: the client only ever sends an ``operation`` id; the command is
looked up from a server-side allowlist (never user input). Execution respects
the same apply gate as the rest of the Product API (dry-run when apply is off).
"""
from __future__ import annotations

import pytest

# Canonical module (not the sndr.* shim): run_operation looks up
# _run_background as its own module global, so monkeypatching must target the
# module the code actually executes in.
from sndr.product_api.legacy import operations as ops


def test_catalog_is_nonempty_and_well_formed():
    catalog = ops.list_operations()
    assert len(catalog) >= 8
    ids = {op["id"] for op in catalog}
    # core diagnostics must be present
    assert {"doctor", "self-test", "validate-schema", "patches-doctor"} <= ids
    groups = {op["group"] for op in catalog}
    assert len(groups) >= 3
    for op in catalog:
        assert op["id"] and op["label"] and op["group"]
        # every command targets the installed sndr_core package (no injection,
        # no arbitrary scripts) — it must invoke the CLI module.
        assert "sndr.cli" in op["command"]
        assert op["mutating"] is False


def test_unknown_operation_raises():
    with pytest.raises(KeyError):
        ops.run_operation("no-such-op", apply_on=True)


def test_dry_run_when_apply_off(monkeypatch, tmp_path):
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    from sndr.product_api.legacy import jobs
    jobs._reset_state()
    job = ops.run_operation("doctor", apply_on=False)
    assert job.dry_run is True
    # the exact command is mirrored so the operator can copy/run it
    assert any("sndr.cli doctor" in line for line in job.cli_mirror)


def test_executes_when_apply_on(monkeypatch, tmp_path):
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    from sndr.product_api.legacy import jobs
    jobs._reset_state()
    captured = {}

    def fake_spawn(**kwargs):
        captured.update(kwargs)
        # mimic background_exec returning a running job without spawning
        return jobs.create_running_job(
            kind=kwargs["kind"], title=kwargs["title"], summary=kwargs.get("summary", {}),
            command=kwargs["command"],
        )

    monkeypatch.setattr(ops, "_run_background", fake_spawn)
    job = ops.run_operation("validate-schema", apply_on=True)
    assert captured["kind"] == "op.validate-schema"
    assert "validate-schema" in captured["command"]
    assert job.dry_run is False


# ---- HTTP routes ----

def test_operations_routes(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    from fastapi.testclient import TestClient

    from sndr.product_api.legacy import jobs
    jobs._reset_state()
    from sndr.product_api.legacy.http_app import create_app

    client = TestClient(create_app(enable_apply=False, allowed_origins=()))
    listing = client.get("/api/v1/operations")
    assert listing.status_code == 200
    body = listing.json()
    assert any(op["id"] == "doctor" for op in body["operations"])
    assert body["apply_enabled"] is False

    # dry-run (apply off)
    run = client.post("/api/v1/operations/run", json={"operation": "doctor"})
    assert run.status_code == 200 and run.json()["dry_run"] is True
    # unknown op -> 404
    assert client.post("/api/v1/operations/run", json={"operation": "nope"}).status_code == 404
