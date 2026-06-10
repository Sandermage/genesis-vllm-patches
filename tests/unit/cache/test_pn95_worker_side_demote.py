# SPDX-License-Identifier: Apache-2.0
"""Tests for `worker_side_proactive_demote` and the `sndr patches
pn95-status` self-diagnosis CLI.

Why these exist
---------------

The original PN95 design routes pressure-driven proactive demote
through `scheduler_tick`, which runs in the EngineCore process. In a
multiproc vLLM deploy the BlockPool refs live in Worker processes —
so EngineCore's `_PN95_BLOCK_POOL_REFS` stays empty and the proactive
branch silently no-ops. `worker_side_proactive_demote` closes that
gap; this suite locks the behavioural contract so the regression
can't slip back in.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch as mpatch

import pytest


# ─── worker_side_proactive_demote ────────────────────────────────────


class TestWorkerSideProactiveDemote:
    def setup_method(self):
        # Always start from a clean enable + reset state.
        from sndr.cache import _pn95_runtime as rt
        rt.reset_for_tests()
        os.environ["GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"] = "1"

    def teardown_method(self):
        from sndr.cache import _pn95_runtime as rt
        os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)
        rt.reset_for_tests()

    def test_returns_zero_when_disabled(self):
        from sndr.cache._pn95_runtime import (
            worker_side_proactive_demote,
        )
        os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)
        assert worker_side_proactive_demote(object()) == 0

    def test_returns_zero_when_tm_not_installed(self):
        from sndr.cache._pn95_runtime import (
            worker_side_proactive_demote,
        )
        # _TM is None after reset_for_tests; helper must short-circuit.
        assert worker_side_proactive_demote(object()) == 0

    def test_returns_zero_on_null_block_pool(self):
        from sndr.cache._pn95_runtime import (
            worker_side_proactive_demote, init_from_config,
        )
        # Install a TM via a synthetic config.
        class Tier:
            device = "cpu"
            capacity_gib = 1.0
            eviction_policy = "lru"
            low_water_pct = 0.9
            promote_on_hit = True

        class CC:
            tiers = [Tier(), Tier()]
            vision_demote_first = True
        class Cfg:
            cache_config = CC()
        init_from_config(Cfg())
        assert worker_side_proactive_demote(None) == 0

    def test_returns_zero_when_pool_has_no_queue(self):
        from sndr.cache._pn95_runtime import (
            worker_side_proactive_demote, init_from_config,
        )

        class Tier:
            device = "cpu"
            capacity_gib = 1.0
            eviction_policy = "lru"
            low_water_pct = 0.9
            promote_on_hit = True
        class CC:
            tiers = [Tier(), Tier()]
            vision_demote_first = True
        class Cfg:
            cache_config = CC()
        init_from_config(Cfg())

        class FakePool:
            free_block_queue = None
        assert worker_side_proactive_demote(FakePool()) == 0

    def test_registers_pool_idempotently(self):
        from sndr.cache import _pn95_runtime as rt
        from sndr.cache._pn95_runtime import (
            worker_side_proactive_demote, init_from_config,
        )

        class Tier:
            device = "cpu"
            capacity_gib = 1.0
            eviction_policy = "lru"
            low_water_pct = 0.9
            promote_on_hit = True
        class CC:
            tiers = [Tier(), Tier()]
            vision_demote_first = True
        class Cfg:
            cache_config = CC()
        init_from_config(Cfg())

        class FakePool:
            free_block_queue = None
        p = FakePool()
        worker_side_proactive_demote(p)
        worker_side_proactive_demote(p)
        # Pool registered exactly once even after two calls.
        assert rt._PN95_BLOCK_POOL_REFS.count(p) == 1


# ─── pn95-status CLI ─────────────────────────────────────────────────


class TestPN95StatusCLI:
    """Drives the CLI via its `_run_pn95_status` function with a
    synthetic stats file, then asserts on captured stdout."""

    def _run(self, stats: dict, capsys):
        import argparse
        from sndr.cli.legacy.patches import _run_pn95_status

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as fh:
            json.dump(stats, fh)
            path = fh.name
        try:
            opts = argparse.Namespace(stats_file=path, json=False)
            rc = _run_pn95_status(opts)
            out = capsys.readouterr().out
            return rc, out
        finally:
            os.unlink(path)

    def _baseline_stats(self) -> dict:
        return {
            "ticks_total": 1000,
            "ticks_pressure_check": 100,
            "ticks_demote_triggered": 0,
            "blocks_demoted_total": 0,
            "blocks_promoted_total": 0,
            "last_free_mib": 4096,
            "last_demote_count": 0,
            "prefix_store_entries": 0,
            "prefix_store_promote_hits": 0,
            "prefix_store_bytes_used": 0,
            "prefix_lookups_total": 0,
            "prefix_lookups_cold_miss": 0,
            "prefix_hit_rate": 0.0,
            "async_stream_enabled": True,
            "async_demote_count": 0,
            "async_promote_count": 0,
            "timestamp": 0,
        }

    def test_renders_banner_and_counters(self, capsys):
        rc, out = self._run(self._baseline_stats(), capsys)
        assert rc == 0
        assert "ticks_total" in out
        assert "1000" in out
        assert "prefix_store_entries" in out

    def test_diagnoses_multiproc_gap(self, capsys):
        s = self._baseline_stats()
        s["ticks_pressure_check"] = 4084
        rc, out = self._run(s, capsys)
        assert rc == 0
        assert "multiproc gap" in out
        assert "worker_side_proactive_demote" in out

    def test_diagnoses_critical_low_free_gpu(self, capsys):
        s = self._baseline_stats()
        s["last_free_mib"] = 50
        s["ticks_pressure_check"] = 0  # skip multiproc hint
        rc, out = self._run(s, capsys)
        assert rc == 0
        assert "below 200 MiB" in out
        assert "activation-buffer" in out

    def test_reports_prefix_hits_positively(self, capsys):
        s = self._baseline_stats()
        s["blocks_demoted_total"] = 200
        s["prefix_store_entries"] = 50
        s["prefix_store_promote_hits"] = 12
        s["ticks_pressure_check"] = 0
        rc, out = self._run(s, capsys)
        assert rc == 0
        assert "actively serving cache hits" in out

    def test_missing_stats_file_returns_nonzero(self, capsys):
        import argparse
        from sndr.cli.legacy.patches import _run_pn95_status

        opts = argparse.Namespace(
            stats_file="/this/file/does/not/exist.json",
            json=False,
        )
        rc = _run_pn95_status(opts)
        assert rc == 1

    def test_json_mode_emits_parseable_payload(self, capsys):
        import argparse
        from sndr.cli.legacy.patches import _run_pn95_status

        stats = self._baseline_stats()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as fh:
            json.dump(stats, fh)
            path = fh.name
        try:
            opts = argparse.Namespace(stats_file=path, json=True)
            rc = _run_pn95_status(opts)
            assert rc == 0
            payload = json.loads(capsys.readouterr().out)
            assert payload["available"] is True
            assert payload["stats"]["ticks_total"] == 1000
            assert "hints" in payload
        finally:
            os.unlink(path)
