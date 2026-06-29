# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``sndr tui`` cockpit ACTIONS (Phase 2).

Phase 1 was read-only (refresh / filter / help / quit). Phase 2 adds the
operate verbs onto the same cockpit — serve / stop / doctor / chat / settings —
each routed through the :mod:`sndr.cli.tui.data` facade (the single seam the app
reads/writes through) so the app stays a thin view and the tests fake one small
surface, exactly as Phase 1 does.

Two tiers, mirroring ``test_tui.py``:

  * **textual-free** — the new data-facade write seams (``serve`` / ``stop`` /
    ``run_doctor``). These call the SAME pipeline functions the CLI uses
    (``run._pull_if_missing`` / ``run._launch_detached`` / ``up._stop_engine``),
    monkeypatched here so nothing launches a real container.
  * **app-driven** — the key bindings (Enter / k / d / c / S) exercised through
    ``App.run_test()`` + ``Pilot`` with an injected fake loader; each
    ``importorskip("textual")`` so the tier SKIPs without the optional extra.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest


# ═════════════════════════════════════════════════════════════════════════════
# Tier 1 — textual-free: the data-facade write seams
# ═════════════════════════════════════════════════════════════════════════════


def test_help_text_documents_phase2_action_keys():
    """The help overlay now lists the live operate verbs + their keys, and no
    longer calls the cockpit read-only / Phase 1."""
    from sndr.cli.tui.render import HELP_TEXT

    assert "serve" in HELP_TEXT and "stop" in HELP_TEXT
    assert "doctor" in HELP_TEXT and "chat" in HELP_TEXT
    assert "Enter" in HELP_TEXT  # the serve key
    assert "Phase 1" not in HELP_TEXT  # the read-only era is over


def test_serve_runs_pull_then_launch_and_reports_ok(monkeypatch):
    """``data.serve`` ensures weights then launches detached — the same two
    pipeline steps ``sndr run`` performs (minus the blocking wait/chat; the
    cockpit's live refresh shows the engine come up). Reports a structured ok."""
    from sndr.cli.tui import data

    calls: list[tuple] = []

    def _fake_pull(preset_id, *, dry_run=False):
        calls.append(("pull", preset_id, dry_run))
        return 0

    def _fake_launch(preset_id, *, port=None, dry_run=False, quiet=False):
        calls.append(("launch", preset_id, port, dry_run))
        return 0

    monkeypatch.setattr("sndr.cli.commands.run._pull_if_missing", _fake_pull)
    monkeypatch.setattr("sndr.cli.commands.run._launch_detached", _fake_launch)

    result = data.serve("prod-qwen3.6-27b")

    assert result["ok"] is True
    assert result["preset_id"] == "prod-qwen3.6-27b"
    assert result["rc"] == 0
    assert result.get("error") is None
    # pull first (no-op when present), then a detached launch — no real wait.
    assert calls == [
        ("pull", "prod-qwen3.6-27b", False),
        ("launch", "prod-qwen3.6-27b", None, False),
    ]


def test_serve_reports_failure_when_launch_fails(monkeypatch):
    """A non-zero launch rc surfaces as ``ok=False`` with the rc + a hint, never
    an exception (the worker turns this into a calm log line)."""
    from sndr.cli.tui import data

    monkeypatch.setattr("sndr.cli.commands.run._pull_if_missing",
                        lambda preset_id, *, dry_run=False: 0)
    monkeypatch.setattr("sndr.cli.commands.run._launch_detached",
                        lambda preset_id, *, port=None, dry_run=False, quiet=False: 125)

    result = data.serve("big-fp8-35b")

    assert result["ok"] is False
    assert result["rc"] == 125
    assert "launch failed" in result["error"]


def test_serve_launches_quietly_from_tui(monkeypatch):
    """``data.serve`` runs inside a live cockpit worker (Textual owns the
    alt-screen), so the detached child launch must NOT inherit the terminal —
    serve passes ``quiet=True`` to suppress the child's stdout/stderr."""
    from sndr.cli.tui import data

    seen: dict[str, object] = {}
    monkeypatch.setattr("sndr.cli.commands.run._pull_if_missing",
                        lambda preset_id, *, dry_run=False: 0)

    def _fake_launch(preset_id, *, port=None, dry_run=False, quiet=False):
        seen["quiet"] = quiet
        return 0

    monkeypatch.setattr("sndr.cli.commands.run._launch_detached", _fake_launch)
    data.serve("p")
    assert seen.get("quiet") is True, "TUI serve must launch quietly (no tty inherit)"


def test_stop_calls_stop_engine_and_reports(monkeypatch):
    """``data.stop`` reuses ``up._stop_engine`` (the same ``docker stop`` verb
    ``sndr down`` uses) and reports whether a container was actually stopped."""
    from sndr.cli.tui import data

    seen: list[Any] = []

    def _fake_stop(preset_id, *, dry_run=False):
        seen.append((preset_id, dry_run))
        return True

    monkeypatch.setattr("sndr.cli.commands.up._stop_engine", _fake_stop)

    result = data.stop("prod-qwen3.6-27b")

    assert result["ok"] is True
    assert result["stopped"] is True
    assert result["preset_id"] == "prod-qwen3.6-27b"
    assert seen == [("prod-qwen3.6-27b", False)]


def test_stop_reports_nothing_running_when_no_container(monkeypatch):
    """When no container matched, ``stopped`` is False but the call still
    succeeds (idempotent teardown — pressing k twice is harmless)."""
    from sndr.cli.tui import data

    monkeypatch.setattr("sndr.cli.commands.up._stop_engine",
                        lambda preset_id, *, dry_run=False: False)

    result = data.stop("prod-qwen3.6-27b")

    assert result["ok"] is True
    assert result["stopped"] is False


def test_run_doctor_invokes_compat_doctor(monkeypatch):
    """``data.run_doctor`` runs the SAME ``doctor`` the CLI promotes — the
    ``sndr.compat.cli`` bridge target the ``DoctorCommand`` pass-through uses —
    so the cockpit's ``d`` and ``sndr doctor`` cannot drift. Returns its rc."""
    from sndr.cli.tui import data

    seen: list[Any] = []
    monkeypatch.setattr("sndr.compat.cli.main",
                        lambda argv: seen.append(list(argv)) or 0)

    rc = data.run_doctor()

    assert rc == 0
    assert seen == [["doctor"]]


def test_run_chat_invokes_chat_command_with_preset(monkeypatch):
    """``data.run_chat`` runs the SAME native ``ChatCommand`` ``sndr chat`` uses
    (the thin REPL over a running engine that the GUI also shares), threading the
    preset + host/port through so the REPL targets the right engine."""
    from sndr.cli.tui import data

    seen: list[Any] = []

    def _fake_execute(self, args):
        seen.append((args.preset, args.host, args.port))
        return 0

    monkeypatch.setattr("sndr.cli.commands.chat.ChatCommand.execute", _fake_execute)

    rc = data.run_chat("prod-qwen3.6-27b")

    assert rc == 0
    assert seen == [("prod-qwen3.6-27b", "127.0.0.1", None)]


def test_run_chat_without_preset_lets_chat_autodiscover(monkeypatch):
    """No preset → ``ChatCommand`` with ``preset=None`` (it resolves the default
    port and discovers the running engine itself)."""
    from sndr.cli.tui import data

    seen: list[Any] = []
    monkeypatch.setattr("sndr.cli.commands.chat.ChatCommand.execute",
                        lambda self, args: seen.append(args.preset) or 0)

    data.run_chat(None)

    assert seen == [None]


def test_save_settings_persists_and_applies_to_env(monkeypatch, tmp_path):
    """``data.save_settings`` writes the values under SNDR_HOME (so the next TUI
    launch loads them) AND applies them to the live process env, so THIS
    session's serve/pull (which read SNDR_MODELS_DIR / HF_TOKEN) pick them up —
    the child ``sndr launch`` inherits the env."""
    import os

    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    monkeypatch.delenv("SNDR_MODELS_DIR", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    from sndr.cli.tui import data

    result = data.save_settings(model_dir="/data/models", hf_token="hf_secret")

    assert result["ok"] is True
    assert os.environ["SNDR_MODELS_DIR"] == "/data/models"
    assert os.environ["HF_TOKEN"] == "hf_secret"
    # persisted: a fresh load (new process would do the same) sees the values
    loaded = data.load_settings()
    assert loaded["model_dir"] == "/data/models"
    assert loaded["hf_token"] == "hf_secret"


def test_save_settings_blank_token_does_not_clobber_existing(monkeypatch, tmp_path):
    """Leaving the token field blank keeps whatever was already set (so editing
    only the model dir never wipes a configured token)."""
    import os

    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN", "hf_existing")
    monkeypatch.delenv("SNDR_MODELS_DIR", raising=False)
    from sndr.cli.tui import data

    data.save_settings(model_dir="/data/models", hf_token="")

    assert os.environ["HF_TOKEN"] == "hf_existing"  # untouched
    assert os.environ["SNDR_MODELS_DIR"] == "/data/models"


def test_save_settings_does_not_persist_env_sourced_token_to_disk(monkeypatch, tmp_path):
    """A token that lives ONLY in the env must NOT be written to the settings
    file when the operator leaves the token field blank (e.g. editing just the
    model dir). Persisting an env secret to a plaintext file is a leak."""
    import json
    import os

    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN", "hf_env_only_secret")
    monkeypatch.delenv("SNDR_MODELS_DIR", raising=False)
    from sndr.cli.tui import data

    data.save_settings(model_dir="/data/models", hf_token="")  # blank token field

    saved = json.loads((tmp_path / "state" / "tui_settings.json").read_text())
    assert saved.get("hf_token", "") == "", (
        "env-sourced token must NOT be persisted to disk on a blank field"
    )
    # The live env is untouched (the operator's token still works this session).
    assert os.environ["HF_TOKEN"] == "hf_env_only_secret"


def test_settings_file_is_chmod_0600(monkeypatch, tmp_path):
    """The settings file may hold a token the operator typed — it must be
    owner-only (0600), never world/group-readable."""
    import os
    import stat

    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    from sndr.cli.tui import data

    data.save_settings(model_dir="/data/models", hf_token="hf_typed_secret")

    mode = stat.S_IMODE(os.stat(tmp_path / "state" / "tui_settings.json").st_mode)
    assert mode == 0o600, f"settings file must be 0600, got {oct(mode)}"


def test_load_settings_falls_back_to_env_when_no_state(monkeypatch, tmp_path):
    """With no saved state, ``load_settings`` reflects the current env so the
    Settings modal opens pre-filled with what the engine actually uses."""
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    monkeypatch.setenv("SNDR_MODELS_DIR", "/env/models")
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    from sndr.cli.tui import data

    loaded = data.load_settings()

    assert loaded["model_dir"] == "/env/models"
    assert loaded["hf_token"] == "hf_env"


# ═════════════════════════════════════════════════════════════════════════════
# Tier 2 — app-driven: the Phase 2 key bindings via App.run_test() + Pilot.
# Each importorskips textual so the whole tier SKIPs without the optional extra.
# ═════════════════════════════════════════════════════════════════════════════
import contextlib  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import Optional  # noqa: E402


@dataclass
class FakeRig:
    gpu_count: int = 2
    min_vram_gb: Optional[int] = 24
    min_compute_cap: Optional[tuple] = (8, 6)
    source: str = "dual-a5000"


@dataclass
class FakeCandidate:
    preset_id: str
    status: str
    can_run: bool
    metric_label: Optional[str] = None


@dataclass
class FakeCatalog:
    rig: FakeRig
    candidates: list = field(default_factory=list)

    def menu(self, *, show_all: bool):
        return self.candidates if show_all else [c for c in self.candidates if c.can_run]


class FakeActionLoader:
    """The single facade surface the cockpit reads AND writes through — Phase 1
    reads (load_catalog / engine_snapshot / rig_summary) plus the Phase 2 write
    seams (serve / stop / run_doctor / run_chat), each recording its calls."""

    def __init__(self, catalog: FakeCatalog, snapshot: dict[str, Any]) -> None:
        self._catalog = catalog
        self._snapshot = snapshot
        self.serve_calls: list[str] = []
        self.stop_calls: list[str] = []
        self.doctor_calls = 0
        self.chat_calls: list[Optional[str]] = []
        self.saved_settings: list[dict] = []

    # Phase 1 reads
    def load_catalog(self, *, rig=None, fake_gpus=None):
        return self._catalog

    def engine_snapshot(self):
        return self._snapshot

    def rig_summary(self, rig):
        return "dual-a5000 (2 GPU)"

    # Phase 2 writes
    def serve(self, preset_id, *, port=None):
        self.serve_calls.append(preset_id)
        return {"ok": True, "preset_id": preset_id, "rc": 0, "error": None}

    def stop(self, preset_id, *, dry_run=False):
        self.stop_calls.append(preset_id)
        return {"ok": True, "preset_id": preset_id, "stopped": True, "error": None}

    def run_doctor(self):
        self.doctor_calls += 1
        return 0

    def run_chat(self, preset_id=None, *, host="127.0.0.1", port=None):
        self.chat_calls.append(preset_id)
        return 0

    def load_settings(self):
        return {"model_dir": "/cur/models", "hf_token": ""}

    def save_settings(self, *, model_dir="", hf_token=""):
        self.saved_settings.append({"model_dir": model_dir, "hf_token": hf_token})
        return {"ok": True, "model_dir": model_dir, "error": None}


def _serving_snapshot() -> dict[str, Any]:
    return {
        "status": {"reachable": True, "version": "0.23.1", "models": ["qwen3.6-27b"]},
        "metrics": {"reachable": True, "kpis": {"generation_toks_per_s": 241.0}},
    }


def _make_action_app():
    from sndr.cli.tui.app import SndrCockpit

    catalog = FakeCatalog(
        rig=FakeRig(),
        candidates=[
            FakeCandidate("prod-qwen3.6-27b", "production", can_run=True, metric_label="agg_TPS=241"),
            FakeCandidate("big-fp8-35b", "validated", can_run=False),
        ],
    )
    loader = FakeActionLoader(catalog, _serving_snapshot())
    app = SndrCockpit(rig="dual-a5000", loader=loader)
    return app, loader


def _run(coro):
    return asyncio.run(coro)


def test_enter_on_fitting_row_confirms_then_serves():
    pytest.importorskip("textual")
    from sndr.cli.tui.app import ConfirmScreen

    async def body():
        app, loader = _make_action_app()
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.press("enter")  # row 0 = fitting preset
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)  # confirm gate first
            await pilot.press("y")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert loader.serve_calls == ["prod-qwen3.6-27b"]

    _run(body())


def test_enter_then_cancel_does_not_serve():
    pytest.importorskip("textual")

    async def body():
        app, loader = _make_action_app()
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("n")  # decline
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert loader.serve_calls == []

    _run(body())


def test_enter_on_nonfitting_row_refuses_without_confirm():
    pytest.importorskip("textual")
    from sndr.cli.tui.app import ConfirmScreen

    async def body():
        app, loader = _make_action_app()
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.press("down")  # move to row 1 = non-fitting (✗)
            await pilot.press("enter")
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmScreen)  # no serve gate
            assert loader.serve_calls == []

    _run(body())


def test_k_confirms_then_stops():
    pytest.importorskip("textual")
    from sndr.cli.tui.app import ConfirmScreen

    async def body():
        app, loader = _make_action_app()
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.press("k")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert len(loader.stop_calls) == 1

    _run(body())


def test_d_runs_doctor_under_suspend():
    pytest.importorskip("textual")

    async def body():
        app, loader = _make_action_app()
        # suspend() drops to the real terminal — not supported by the headless
        # test driver; swap in a no-op context so the action runs end-to-end.
        app.suspend = lambda: contextlib.nullcontext()
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.press("d")
            await pilot.pause()
            assert loader.doctor_calls == 1

    _run(body())


def test_c_runs_chat_under_suspend_for_served_model():
    pytest.importorskip("textual")

    async def body():
        app, loader = _make_action_app()
        app.suspend = lambda: contextlib.nullcontext()
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.press("c")
            await pilot.pause()
            assert len(loader.chat_calls) == 1

    _run(body())


def test_s_opens_settings_prefilled_and_save_persists():
    pytest.importorskip("textual")
    from textual.widgets import Input

    from sndr.cli.tui.app import SettingsScreen

    async def body():
        app, loader = _make_action_app()
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            assert isinstance(app.screen, SettingsScreen)
            # pre-filled from the loader's current settings
            md_input = app.screen.query_one("#set-model-dir", Input)
            assert md_input.value == "/cur/models"
            # edit + submit
            md_input.value = "/data/models"
            md_input.focus()
            await pilot.pause()
            await pilot.press("enter")  # on_input_submitted → save
            await pilot.pause()
            assert loader.saved_settings == [{"model_dir": "/data/models", "hf_token": ""}]

    _run(body())


def test_tui_command_parses_lean_flag():
    """``sndr tui --lean`` opts into the beginner layout (textual-free parse)."""
    import argparse

    from sndr.cli.commands.tui import TuiCommand

    parser = argparse.ArgumentParser()
    TuiCommand().configure_parser(parser)
    assert parser.parse_args([]).lean is False
    assert parser.parse_args(["--lean"]).lean is True


def test_lean_mode_hides_operator_panes():
    pytest.importorskip("textual")

    async def body():
        from sndr.cli.tui.app import SndrCockpit

        catalog = FakeCatalog(
            rig=FakeRig(),
            candidates=[FakeCandidate("prod-qwen3.6-27b", "production", can_run=True)],
        )
        loader = FakeActionLoader(catalog, _serving_snapshot())
        app = SndrCockpit(rig="dual-a5000", loader=loader, lean=True)
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            # operator detail hidden; the beginner essentials stay
            assert app.query_one("#gpu").display is False
            assert app.query_one("#log").display is False
            assert app.query_one("#engine").display is True
            assert app.query_one("#catalog").display is True

    _run(body())


def test_default_mode_shows_all_panes():
    pytest.importorskip("textual")

    async def body():
        app, _ = _make_action_app()  # lean defaults off
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            for pane in ("#engine", "#gpu", "#catalog", "#log"):
                assert app.query_one(pane).display is True

    _run(body())


def test_settings_escape_cancels_without_saving():
    pytest.importorskip("textual")

    async def body():
        app, loader = _make_action_app()
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert loader.saved_settings == []

    _run(body())


def test_slash_filter_narrows_catalog_to_matching_presets():
    """`/` opens a filter input; typing narrows the catalog table to presets
    whose id matches (case-insensitive); clearing restores every row. The
    binding + help advertise a working filter — this proves it actually filters
    instead of just focusing the table."""
    pytest.importorskip("textual")
    from textual.widgets import DataTable

    async def body():
        app, _ = _make_action_app()  # prod-qwen3.6-27b (✓) + big-fp8-35b (✗)
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            table = app.query_one("#catalog", DataTable)
            assert table.row_count == 2  # both presets shown initially
            await pilot.press("slash")  # open the filter input
            await pilot.pause()
            await pilot.press("2")
            await pilot.press("7")
            await pilot.press("b")
            await pilot.pause()
            assert table.row_count == 1, "filter should narrow to the matching preset"
            await pilot.press("backspace")
            await pilot.press("backspace")
            await pilot.press("backspace")
            await pilot.pause()
            assert table.row_count == 2, "clearing the filter restores all rows"

    _run(body())
