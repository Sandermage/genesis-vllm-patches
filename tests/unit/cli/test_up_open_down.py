# SPDX-License-Identifier: Apache-2.0
"""v12 UX R3 — Harbor-style one-command full-stack bring-up.

R1 gave the no-args wizard + ``sndr run`` (cold start → chat). R2 unified the
CLI surface. R3 closes the last gap: ONE command that brings up the WHOLE
product — the engine AND the product-API + GUI daemon — and prints the local
URL, the way ``harbor up`` does.

These tests drive the thin orchestration with every side-effectful step mocked,
asserting only the wiring (the call sequence), never spinning a GPU engine or a
real uvicorn daemon:

  * ``sndr up --dry-run`` plans engine + daemon WITHOUT starting either;
  * ``sndr up`` (mocked) launches the engine, starts the daemon, waits for both
    ready, and prints the "open http://127.0.0.1:8765" pointer;
  * ``sndr up --no-engine`` skips the engine and brings up daemon + GUI only;
  * ``sndr open`` resolves the local daemon URL and handles a headless host
    (no browser) gracefully — always printing the URL;
  * ``sndr down`` stops what ``up`` started (engine container + daemon).

The engine path reuses the R1 ``run``/``launch`` child-process launcher; the
daemon path reuses the existing ``gui-api`` Product API server. Nothing here
reimplements an engine or a server — it is orchestration over existing seams.
"""
from __future__ import annotations

import argparse
import io
from contextlib import redirect_stderr, redirect_stdout

import pytest

pytest.importorskip("pydantic")

import sndr.cli.commands.up as up_mod  # noqa: E402
from sndr.cli.commands.up import DownCommand, OpenCommand, UpCommand  # noqa: E402

TWO_A5000 = "RTX A5000:24564:8.6;RTX A5000:24564:8.6"


def _up_ns(**kw):
    base = dict(
        preset=None,
        rig=None,
        fake_gpus=None,
        port=None,
        gui_port=8765,
        dry_run=False,
        no_input=False,
        no_engine=False,
        timeout=300,
        output="text",
    )
    base.update(kw)
    return argparse.Namespace(**base)


def _down_ns(**kw):
    base = dict(preset=None, fake_gpus=None, rig=None, gui_port=8765, dry_run=False, output="text")
    base.update(kw)
    return argparse.Namespace(**base)


def _open_ns(**kw):
    base = dict(gui_port=8765, output="text")
    base.update(kw)
    return argparse.Namespace(**base)


def _patch_pipeline(monkeypatch, *, events, engine_ok=True, daemon_ok=True):
    """Stub every side-effectful step so the orchestration is observable.

    ``events`` is a list the stubs append to, in call order.
    """

    def fake_launch_engine(preset_id, *, port=None, dry_run=False):
        events.append(("launch_engine", preset_id, port, dry_run))
        return 0

    def fake_wait_engine(host, port, *, timeout, on_progress=None):
        events.append(("wait_engine", host, port, timeout))
        return {"reachable": engine_ok, "models": ["served-model"], "error": None if engine_ok else "timed out"}

    def fake_start_daemon(host, port):
        events.append(("start_daemon", host, port))
        return object()  # an opaque handle

    def fake_wait_daemon(host, port, *, timeout):
        events.append(("wait_daemon", host, port, timeout))
        return daemon_ok

    monkeypatch.setattr(up_mod, "_launch_engine_detached", fake_launch_engine)
    monkeypatch.setattr(up_mod, "_wait_engine_ready", fake_wait_engine)
    monkeypatch.setattr(up_mod, "_start_daemon", fake_start_daemon)
    monkeypatch.setattr(up_mod, "_wait_daemon_ready", fake_wait_daemon)


# ── 1. registration ─────────────────────────────────────────────────────────


class TestRegistered:
    def test_up_open_down_in_registry(self):
        from sndr.cli.commands import COMMAND_REGISTRY
        from sndr.cli.main import build_parser

        build_parser()
        for name in ("up", "open", "down"):
            assert name in COMMAND_REGISTRY, f"`sndr {name}` must resolve on the canonical surface"


# ── 2. sndr up --dry-run plans engine + daemon WITHOUT starting ──────────────


class TestUpDryRun:
    def test_dry_run_plans_both_without_starting(self, monkeypatch):
        events: list = []
        _patch_pipeline(monkeypatch, events=events)
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            rc = UpCommand().execute(_up_ns(preset="example-2x-tier-aware", fake_gpus=TWO_A5000, dry_run=True))
        assert rc == 0
        # Nothing is started in dry-run.
        kinds = [e[0] for e in events]
        assert "launch_engine" not in kinds
        assert "start_daemon" not in kinds
        assert "wait_engine" not in kinds
        assert "wait_daemon" not in kinds
        text = out.getvalue()
        # The plan names BOTH the engine and the daemon + the local URL.
        assert "example-2x-tier-aware" in text
        assert "8765" in text


# ── 3. sndr up (mocked) brings up the full stack in order ────────────────────


class TestUpCallSequence:
    def test_full_stack_order(self, monkeypatch):
        events: list = []
        _patch_pipeline(monkeypatch, events=events)
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            rc = UpCommand().execute(_up_ns(preset="example-2x-tier-aware", fake_gpus=TWO_A5000))
        assert rc == 0
        # Engine first (launch → wait), then daemon (start → wait).
        assert [e[0] for e in events] == ["launch_engine", "wait_engine", "start_daemon", "wait_daemon"]

    def test_up_prints_local_url_pointer(self, monkeypatch):
        events: list = []
        _patch_pipeline(monkeypatch, events=events)
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            rc = UpCommand().execute(_up_ns(preset="example-2x-tier-aware", fake_gpus=TWO_A5000))
        assert rc == 0
        text = err.getvalue()
        assert "sndr is up" in text
        assert "http://127.0.0.1:8765" in text
        assert "sndr open" in text

    def test_engine_failure_aborts_before_daemon(self, monkeypatch):
        events: list = []
        _patch_pipeline(monkeypatch, events=events)

        def failing_launch(preset_id, *, port=None, dry_run=False):
            events.append(("launch_engine", preset_id, port, dry_run))
            return 3

        monkeypatch.setattr(up_mod, "_launch_engine_detached", failing_launch)
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            rc = UpCommand().execute(_up_ns(preset="example-2x-tier-aware", fake_gpus=TWO_A5000))
        assert rc == 3
        # Daemon is never started after an engine launch failure.
        assert "start_daemon" not in [e[0] for e in events]
        assert "engine" in err.getvalue().lower()

    def test_engine_not_ready_aborts_before_daemon(self, monkeypatch):
        events: list = []
        _patch_pipeline(monkeypatch, events=events, engine_ok=False)
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            rc = UpCommand().execute(_up_ns(preset="example-2x-tier-aware", fake_gpus=TWO_A5000))
        assert rc != 0
        assert "start_daemon" not in [e[0] for e in events]
        assert "did not become ready" in err.getvalue().lower() or "not ready" in err.getvalue().lower()

    def test_daemon_not_ready_is_reported(self, monkeypatch):
        events: list = []
        _patch_pipeline(monkeypatch, events=events, daemon_ok=False)
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            rc = UpCommand().execute(_up_ns(preset="example-2x-tier-aware", fake_gpus=TWO_A5000))
        assert rc != 0
        # The engine came up, the daemon was started, but readiness failed.
        assert [e[0] for e in events] == ["launch_engine", "wait_engine", "start_daemon", "wait_daemon"]
        assert "daemon" in err.getvalue().lower()


# ── 4. sndr up --no-engine skips the engine ─────────────────────────────────


class TestUpNoEngine:
    def test_no_engine_skips_engine_steps(self, monkeypatch):
        events: list = []
        _patch_pipeline(monkeypatch, events=events)
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            rc = UpCommand().execute(_up_ns(no_engine=True, gui_port=8765))
        assert rc == 0
        # ONLY daemon steps — no engine launch / wait.
        assert [e[0] for e in events] == ["start_daemon", "wait_daemon"]
        assert "http://127.0.0.1:8765" in err.getvalue()

    def test_no_engine_dry_run_plans_daemon_only(self, monkeypatch):
        events: list = []
        _patch_pipeline(monkeypatch, events=events)
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            rc = UpCommand().execute(_up_ns(no_engine=True, dry_run=True))
        assert rc == 0
        assert events == []
        # No engine preset resolution required for the daemon-only plan.
        text = out.getvalue()
        assert "8765" in text


# ── 5. sndr open resolves the URL + handles no-browser ──────────────────────


class TestOpen:
    def test_open_invokes_browser_with_local_url(self, monkeypatch):
        opened: dict[str, str] = {}

        def fake_open(url):
            opened["url"] = url
            return True

        monkeypatch.setattr(up_mod, "_open_browser", fake_open)
        err = io.StringIO()
        out = io.StringIO()
        with redirect_stderr(err), redirect_stdout(out):
            rc = OpenCommand().execute(_open_ns(gui_port=8765))
        assert rc == 0
        assert opened.get("url") == "http://127.0.0.1:8765"
        # The URL is always printed (so a headless operator can copy it).
        assert "http://127.0.0.1:8765" in (out.getvalue() + err.getvalue())

    def test_open_handles_no_browser_gracefully(self, monkeypatch):
        # A headless host (no browser) must still succeed and print the URL.
        def fake_open(url):
            return False

        monkeypatch.setattr(up_mod, "_open_browser", fake_open)
        err = io.StringIO()
        out = io.StringIO()
        with redirect_stderr(err), redirect_stdout(out):
            rc = OpenCommand().execute(_open_ns(gui_port=8765))
        assert rc == 0
        combined = out.getvalue() + err.getvalue()
        assert "http://127.0.0.1:8765" in combined
        # A friendly note on the headless path.
        assert "no browser" in combined.lower() or "open it manually" in combined.lower()

    def test_open_honours_custom_gui_port(self, monkeypatch):
        opened: dict[str, str] = {}
        monkeypatch.setattr(up_mod, "_open_browser", lambda url: opened.__setitem__("url", url) or True)
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            rc = OpenCommand().execute(_open_ns(gui_port=9999))
        assert rc == 0
        assert opened.get("url") == "http://127.0.0.1:9999"


# ── 6. sndr down stops engine + daemon ──────────────────────────────────────


class TestDown:
    def test_down_calls_engine_and_daemon_teardown(self, monkeypatch):
        events: list = []

        def fake_stop_engine(preset_id, *, dry_run=False):
            events.append(("stop_engine", preset_id, dry_run))
            return True

        def fake_stop_daemon(*, dry_run=False):
            events.append(("stop_daemon", dry_run))
            return True

        monkeypatch.setattr(up_mod, "_stop_engine", fake_stop_engine)
        monkeypatch.setattr(up_mod, "_stop_daemon", fake_stop_daemon)
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            rc = DownCommand().execute(_down_ns(preset="example-2x-tier-aware", fake_gpus=TWO_A5000))
        assert rc == 0
        kinds = [e[0] for e in events]
        assert "stop_daemon" in kinds
        assert "stop_engine" in kinds

    def test_down_dry_run_stops_nothing(self, monkeypatch):
        events: list = []
        monkeypatch.setattr(
            up_mod, "_stop_engine",
            lambda preset_id, *, dry_run=False: events.append(("stop_engine", dry_run)) or True,
        )
        monkeypatch.setattr(
            up_mod, "_stop_daemon",
            lambda *, dry_run=False: events.append(("stop_daemon", dry_run)) or True,
        )
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            rc = DownCommand().execute(_down_ns(preset="example-2x-tier-aware", fake_gpus=TWO_A5000, dry_run=True))
        assert rc == 0
        # Both teardown seams are invoked but with dry_run=True (they plan, not act).
        assert all(e[-1] is True for e in events), "down --dry-run must pass dry_run=True to every teardown"

    def test_down_friendly_summary(self, monkeypatch):
        monkeypatch.setattr(up_mod, "_stop_engine", lambda preset_id, *, dry_run=False: True)
        monkeypatch.setattr(up_mod, "_stop_daemon", lambda *, dry_run=False: True)
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            rc = DownCommand().execute(_down_ns(preset="example-2x-tier-aware", fake_gpus=TWO_A5000))
        assert rc == 0
        assert "down" in err.getvalue().lower() or "stopped" in err.getvalue().lower()


# ── 7. R1 + R2 behaviors unbroken ────────────────────────────────────────────


class TestRPriorUnbroken:
    def test_r1_r2_commands_still_registered(self):
        from sndr.cli.commands import COMMAND_REGISTRY
        from sndr.cli.main import build_parser

        build_parser()
        for name in ("run", "chat", "launch", "report", "doctor", "preset", "verify", "pull"):
            assert name in COMMAND_REGISTRY
