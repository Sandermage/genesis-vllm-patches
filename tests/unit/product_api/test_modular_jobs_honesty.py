# SPDX-License-Identifier: Apache-2.0
"""Honesty contract for the modular jobs router.

The modular ``/api/v1/jobs`` routes have NO producer: the in-memory
``_JOBS`` store in ``observability_service`` is only ever written by
``register_job()``, which has zero callers — the live daemon's real job
system is the separate legacy ``sndr.product_api.legacy.jobs`` module.

Before this fix the modular routes returned a lying ``200 OK`` with an
always-empty list. They now return an explicit ``501 Not Implemented``
so a consumer is never misled into thinking 'no jobs' when the truth is
'this surface is not wired'. These tests lock that honesty.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from sndr.product_api.routes.observability import jobs_router  # noqa: E402


def _client() -> TestClient:
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(jobs_router)
    return TestClient(app, raise_server_exceptions=True)


def test_list_jobs_returns_501_not_lying_200():
    resp = _client().get("/api/v1/jobs")
    assert resp.status_code == 501
    detail = resp.json()["detail"].lower()
    assert "legacy" in detail
    assert "job" in detail


def test_get_job_returns_501_not_404_or_200():
    resp = _client().get("/api/v1/jobs/some-id")
    assert resp.status_code == 501
    assert "legacy" in resp.json()["detail"].lower()


def test_list_jobs_with_state_filter_still_501():
    resp = _client().get("/api/v1/jobs?state=running")
    assert resp.status_code == 501


def test_jobs_router_docstring_documents_migration_status():
    """The module must carry an explicit note about the unwired modular
    jobs surface so the next maintainer is not surprised."""
    import sndr.product_api.routes.observability as obs
    doc = (obs.__doc__ or "").lower()
    assert "legacy" in doc
    assert "job" in doc
