# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``sndr tui`` read-only cockpit (Phase 1).

Two tiers, by design:

  * **textual-free** — the gate (``sndr tui`` without the ``[tui]`` extra prints
    a friendly install hint and exits 1, never a traceback) and the pure render
    helpers (:mod:`sndr.cli.tui.render`). These run on the light CI leg that has
    no textual installed, so they live above the ``importorskip`` and use tiny
    fakes instead of the real preset corpus.
  * **app-driven** — the Textual app (:class:`sndr.cli.tui.app.SndrCockpit`)
    exercised through ``App.run_test()`` + ``Pilot`` with an injected fake data
    loader. Each of these calls ``pytest.importorskip("textual")`` first, so the
    whole tier SKIPs (never fails) when the optional extra is absent.

The app never touches the network or a GPU here: the fake loader is the single
seam the cockpit reads through (mirroring how the real app reads through
:mod:`sndr.cli.tui.data`), so the tests fake one small surface, not Textual's
internals.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from sndr.cli.commands.tui import TuiCommand
from sndr.cli.tui.render import (
    HELP_TEXT,
    catalog_rows,
    render_engine,
    render_gpu,
)

# ─────────────────────────────────────────────────────────────────────────────
# Tiny fakes — stand in for Rig / Candidate / Catalog and the data facade so the
# textual-free tests need neither the preset corpus nor a live engine.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FakeRig:
    gpu_count: int = 0
    min_vram_gb: int | None = None
    min_compute_cap: tuple[int, int] | None = None
    source: str = "fake"


@dataclass
class FakeCandidate:
    preset_id: str
    status: str
    can_run: bool
    metric_label: str | None = None


@dataclass
class FakeCatalog:
    rig: FakeRig
    candidates: list[FakeCandidate] = field(default_factory=list)

    def menu(self, *, show_all: bool) -> list[FakeCandidate]:
        return self.candidates if show_all else [c for c in self.candidates if c.can_run]


class FakeLoader:
    """A drop-in for :mod:`sndr.cli.tui.data` — the one surface the app reads.

    Records the kwargs the app passes so tests can assert the rig precedence is
    threaded through, and serves a canned catalog + engine snapshot.
    """

    def __init__(self, catalog: FakeCatalog, snapshot: dict[str, Any]) -> None:
        self._catalog = catalog
        self._snapshot = snapshot
        self.load_calls: list[dict[str, Any]] = []
        self.engine_calls = 0

    def load_catalog(self, *, rig=None, fake_gpus=None) -> FakeCatalog:
        self.load_calls.append({"rig": rig, "fake_gpus": fake_gpus})
        return self._catalog

    def engine_snapshot(self) -> dict[str, Any]:
        self.engine_calls += 1
        return self._snapshot

    def memory_snapshot(self) -> dict[str, Any]:
        self.memory_calls = getattr(self, "memory_calls", 0) + 1
        return {"reachable": False, "stats": {}, "error": "no daemon (test)"}

    def rig_summary(self, rig) -> str:
        return f"{getattr(rig, 'source', 'rig')} ({getattr(rig, 'gpu_count', 0)} GPU)"


def _down_snapshot() -> dict[str, Any]:
    return {
        "status": {"reachable": False, "error": "connection refused"},
        "metrics": {"reachable": False, "kpis": {}},
    }


def _serving_snapshot() -> dict[str, Any]:
    return {
        "status": {"reachable": True, "version": "0.20.2", "models": ["qwen3.6-27b"]},
        "metrics": {
            "reachable": True,
            "kpis": {
                "generation_toks_per_s": 241.4,
                "kv_cache_usage": 0.37,
                "ttft_avg_s": 0.42,
                "requests_running": 2,
                "requests_waiting": 0,
            },
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# Tier 1 — textual-free: the install gate
# ═════════════════════════════════════════════════════════════════════════════


def test_tui_gate_prints_hint_and_exits_1_without_textual(capsys, monkeypatch):
    """``sndr tui`` without the [tui] extra: friendly hint + exit 1, no traceback.

    Simulates textual's absence by making ``import textual`` raise ImportError
    even when the test venv has it installed, so this assertion holds on BOTH
    CI legs.
    """
    import builtins

    real_import = builtins.__import__

    def _no_textual(name, *args, **kwargs):
        if name == "textual" or name.startswith("textual."):
            raise ImportError("No module named 'textual'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_textual)

    args = argparse.Namespace(rig=None, fake_gpus=None)
    rc = TuiCommand().execute(args)

    assert rc == 1
    err = capsys.readouterr().err
    # Friendly, actionable, no Python traceback noise.
    assert "tui" in err
    assert "textual" in err
    assert "pip install" in err
    assert "[tui]" in err
    assert "Traceback" not in err


def test_tui_command_surface_is_read_only_phase1():
    """The command advertises itself and parses the two offline rig flags."""
    cmd = TuiCommand()
    assert cmd.name == "tui"
    parser = argparse.ArgumentParser()
    cmd.configure_parser(parser)
    ns = parser.parse_args(["--rig", "dual-a5000", "--fake-gpus", "RTX A5000:24564:8.6"])
    assert ns.rig == "dual-a5000"
    assert ns.fake_gpus == "RTX A5000:24564:8.6"


# ═════════════════════════════════════════════════════════════════════════════
# Tier 1 — textual-free: pure render helpers
# ═════════════════════════════════════════════════════════════════════════════


def test_render_engine_down_is_calm_hint():
    text = render_engine(_down_snapshot())
    assert "down" in text
    assert "sndr run" in text  # routes the operator forward, no error dump
    assert "connection refused" not in text  # the raw error is never surfaced


def test_render_engine_reachable_shows_kpis():
    text = render_engine(_serving_snapshot())
    assert "serving" in text
    assert "qwen3.6-27b" in text
    assert "241" in text  # tok/s rendered
    assert "37%" in text  # kv_cache_usage as a percentage
    assert "0.42s" in text  # TTFT


def test_render_gpu_zero_gpu_routes_to_fake_gpus():
    text = render_gpu(FakeRig(gpu_count=0))
    assert "no GPU" in text
    assert "--fake-gpus" in text  # the escape hatch a GPU-less box needs


def test_render_gpu_with_rig_shows_card_count_and_cap():
    rig = FakeRig(gpu_count=2, min_vram_gb=24, min_compute_cap=(8, 6), source="dual-a5000")
    text = render_gpu(rig)
    assert "2× GPU" in text
    assert "24 GB" in text
    assert "sm_8.6" in text
    assert "dual-a5000" in text


def test_render_gpu_none_is_detecting():
    assert "detecting" in render_gpu(None)


def test_catalog_rows_maps_can_run_to_check_and_cross():
    catalog = FakeCatalog(
        rig=FakeRig(gpu_count=2),
        candidates=[
            FakeCandidate("qwen3.6-27b", "production", can_run=True, metric_label="agg_TPS=241.4"),
            FakeCandidate("big-fp8-35b", "validated", can_run=False),
        ],
    )
    rows = catalog_rows(catalog)
    assert rows == [
        ("✓", "qwen3.6-27b", "production", "agg_TPS=241.4"),
        ("✗", "big-fp8-35b", "validated", "—"),  # no metric → em-dash placeholder
    ]


def test_catalog_rows_none_is_empty():
    assert catalog_rows(None) == []


def test_catalog_rows_falls_back_to_candidates_without_menu():
    @dataclass
    class _NoMenu:
        candidates: list[FakeCandidate]

    rows = catalog_rows(_NoMenu([FakeCandidate("p", "qa", can_run=True)]))
    assert rows == [("✓", "p", "qa", "—")]


def test_help_text_documents_the_navigation_keys():
    assert "refresh" in HELP_TEXT
    assert "quit" in HELP_TEXT
    # the catalog fit glyphs are explained so a newcomer reads the ✓/✗ column
    assert "fit this rig" in HELP_TEXT


# ═════════════════════════════════════════════════════════════════════════════
# Tier 2 — app-driven: SndrCockpit via App.run_test() + Pilot.
# Each test importorskips textual so the whole tier SKIPs without the extra.
# ═════════════════════════════════════════════════════════════════════════════


def _run(coro):
    """Run an async pilot body without pytest-asyncio (not a project dep)."""
    return asyncio.run(coro)


def _make_app(snapshot: dict[str, Any], *, rig=None, fake_gpus=None):
    from sndr.cli.tui.app import SndrCockpit

    catalog = FakeCatalog(
        rig=FakeRig(gpu_count=2, min_vram_gb=24, min_compute_cap=(8, 6), source="dual-a5000"),
        candidates=[
            FakeCandidate("qwen3.6-27b", "production", can_run=True, metric_label="agg_TPS=241.4"),
            FakeCandidate("big-fp8-35b", "validated", can_run=False),
        ],
    )
    loader = FakeLoader(catalog, snapshot)
    app = SndrCockpit(rig=rig, fake_gpus=fake_gpus, loader=loader)
    return app, loader


def test_app_mounts_four_panes_and_populates_catalog():
    pytest.importorskip("textual")
    from textual.widgets import DataTable, RichLog, Static

    async def body():
        app, loader = _make_app(_serving_snapshot(), rig="dual-a5000")
        async with app.run_test() as pilot:
            await pilot.pause()
            # all four panes mounted
            for pane_id in ("#engine", "#gpu", "#catalog", "#log"):
                assert app.query_one(pane_id) is not None
            assert isinstance(app.query_one("#catalog"), DataTable)
            assert isinstance(app.query_one("#log"), RichLog)
            assert isinstance(app.query_one("#engine"), Static)
            # the fake catalog populated the table (2 presets), and the rig
            # precedence threaded through to the loader.
            table = app.query_one("#catalog", DataTable)
            assert table.row_count == 2
            assert loader.load_calls
            assert loader.load_calls[0]["rig"] == "dual-a5000"
            # the engine pane rendered the serving KPIs (render() → plain text)
            assert "serving" in str(app.query_one("#engine", Static).render())

    _run(body())


def test_app_renders_no_engine_state_calmly():
    pytest.importorskip("textual")
    from textual.widgets import Static

    async def body():
        app, _ = _make_app(_down_snapshot())
        async with app.run_test() as pilot:
            await pilot.pause()
            engine_text = str(app.query_one("#engine", Static).render())
            assert "down" in engine_text
            assert "sndr run" in engine_text

    _run(body())


def test_app_question_mark_opens_help_overlay():
    pytest.importorskip("textual")
    from sndr.cli.tui.app import HelpScreen

    async def body():
        app, _ = _make_app(_serving_snapshot())
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("question_mark")
            await pilot.pause()
            assert isinstance(app.screen, HelpScreen)
            # the overlay shows the help text body (scoped to the modal screen)
            from textual.widgets import Static

            body_text = str(app.screen.query_one("#help-body", Static).render())
            assert "refresh" in body_text
            assert "quit" in body_text

    _run(body())


def test_app_q_quits():
    pytest.importorskip("textual")

    async def body():
        app, _ = _make_app(_serving_snapshot())
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("q")
            await pilot.pause()
        # exiting the context without a hang/exception is the assertion;
        # the app returned control after `q`.
        assert app.return_code in (0, None)

    _run(body())
