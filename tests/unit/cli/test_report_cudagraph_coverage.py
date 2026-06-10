# SPDX-License-Identifier: Apache-2.0
"""Sprint 2.6: tests for `sndr report cudagraph-coverage` CLI."""
from __future__ import annotations

import argparse
import json

import pytest

from sndr.cli.legacy.report import run_cudagraph_coverage
from sndr.observability import (
    record_cudagraph_dispatch,
)
from sndr.observability import cudagraph_dispatch as cgd


@pytest.fixture(autouse=True)
def _reset():
    cgd._reset_module_state()
    yield
    cgd._reset_module_state()


@pytest.fixture
def trace_on(monkeypatch):
    monkeypatch.setenv("GENESIS_CUDAGRAPH_DISPATCH_TRACE", "1")
    yield


def _make_opts(json_out: bool = False) -> argparse.Namespace:
    return argparse.Namespace(json=json_out)


class TestNoEvents:
    def test_human_output_explains_no_events(self, capsys):
        rc = run_cudagraph_coverage(_make_opts())
        out = capsys.readouterr().out
        assert rc == 0
        assert "no dispatch events" in out
        assert "GENESIS_CUDAGRAPH_DISPATCH_TRACE" in out

    def test_json_output_when_no_events(self, capsys):
        rc = run_cudagraph_coverage(_make_opts(json_out=True))
        out = capsys.readouterr().out
        assert rc == 0
        data = json.loads(out)
        assert data["total"] == 0
        assert data["hit_rate_pct"] is None


class TestWithEvents:
    def test_human_output_shows_hit_rate(self, trace_on, capsys):
        for _ in range(8):
            record_cudagraph_dispatch(matched=True)
        for _ in range(2):
            record_cudagraph_dispatch(matched=False)
        rc = run_cudagraph_coverage(_make_opts())
        out = capsys.readouterr().out
        assert rc == 0
        assert "80.0%" in out  # hit rate
        assert "8 captured-graph hits" in out
        assert "10 total" in out

    def test_high_miss_rate_emits_warning(self, trace_on, capsys):
        # 60% miss rate (>10% threshold)
        for _ in range(4):
            record_cudagraph_dispatch(matched=True)
        for _ in range(6):
            record_cudagraph_dispatch(matched=False)
        rc = run_cudagraph_coverage(_make_opts())
        out = capsys.readouterr().out
        assert rc == 0
        assert "High eager-fallback rate" in out
        assert "PN16 V1 regression pattern" in out

    def test_low_miss_rate_no_warning(self, trace_on, capsys):
        # Only 5% miss rate — under threshold
        for _ in range(95):
            record_cudagraph_dispatch(matched=True)
        for _ in range(5):
            record_cudagraph_dispatch(matched=False)
        rc = run_cudagraph_coverage(_make_opts())
        out = capsys.readouterr().out
        assert "High eager-fallback rate" not in out

    def test_json_output_with_events(self, trace_on, capsys):
        for _ in range(7):
            record_cudagraph_dispatch(matched=True)
        for _ in range(3):
            record_cudagraph_dispatch(matched=False)
        rc = run_cudagraph_coverage(_make_opts(json_out=True))
        out = capsys.readouterr().out
        assert rc == 0
        data = json.loads(out)
        assert data["hits"] == 7
        assert data["misses"] == 3
        assert data["total"] == 10
        assert data["hit_rate_pct"] == 70.0
