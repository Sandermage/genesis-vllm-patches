# SPDX-License-Identifier: Apache-2.0
"""``sndr run [preset]`` — the Ollama ``run`` verb (resolve → pull → launch →
wait-ready → chat).

The whole flow is wired from existing pieces (the launch wizard's fit catalog,
the ``compat.models.pull`` artifacts path, the legacy launcher, and the
product-API engine client's readiness probe + chat proxy). These tests drive
the orchestration with every external step mocked, asserting:

  * preset resolution — explicit preset is honoured; an omitted preset resolves
    to the wizard's top-ranked fitting preset for the (faked) rig;
  * the call sequence — pull-if-missing → launch → wait-ready → chat/ready;
  * ``--dry-run`` plans the flow (resolve + report) WITHOUT launching;
  * the friendly ready message names the chat URL and ``sndr chat``.
"""
from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout

import pytest

pytest.importorskip("pydantic")

import sndr.cli.commands.run as run_mod  # noqa: E402
from sndr.cli.commands.run import RunCommand  # noqa: E402

TWO_A5000 = "RTX A5000:24564:8.6;RTX A5000:24564:8.6"
SINGLE_3090 = "RTX 3090:24576:8.6"


def _ns(**kw):
    import argparse

    base = dict(
        preset=None,
        rig=None,
        fake_gpus=None,
        port=None,
        dry_run=False,
        no_input=False,
        timeout=300,
        output="text",
    )
    base.update(kw)
    return argparse.Namespace(**base)


def _force_tty(monkeypatch, value: bool = True) -> None:
    """Force the interactive-REPL gate so the full flow reaches ``_chat_repl``.

    ``sndr run`` opens the REPL only on an interactive TTY (so scripted callers
    never block on stdin). The orchestration tests want to assert the chat step
    is reached, so they simulate an interactive terminal.
    """
    import sys

    monkeypatch.setattr(sys.stdin, "isatty", lambda: value, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: value, raising=False)


def _patch_pipeline(monkeypatch, *, events):
    """Stub every side-effectful step so the orchestration is observable.

    ``events`` is a list the stubs append to, in call order.
    """

    def fake_pull(preset_id, *, dry_run=False):
        events.append(("pull", preset_id, dry_run))
        return 0

    def fake_launch(preset_id, *, port=None, dry_run=False):
        events.append(("launch", preset_id, port, dry_run))
        return 0

    def fake_wait(host, port, *, timeout, on_progress=None):
        events.append(("wait_ready", host, port, timeout))
        return {"reachable": True, "models": [preset_or_model(port)], "version": "x"}

    def fake_chat(host, port, *, preset_id):
        events.append(("chat", host, port, preset_id))
        return 0

    def preset_or_model(port):
        return "served-model"

    monkeypatch.setattr(run_mod, "_pull_if_missing", fake_pull)
    monkeypatch.setattr(run_mod, "_launch_detached", fake_launch)
    monkeypatch.setattr(run_mod, "_wait_ready", fake_wait)
    monkeypatch.setattr(run_mod, "_chat_repl", fake_chat)


class TestPresetResolution:
    def test_omitted_preset_resolves_top_fit_for_rig(self, monkeypatch):
        events: list = []
        _patch_pipeline(monkeypatch, events=events)
        _force_tty(monkeypatch)
        rc = RunCommand().execute(_ns(fake_gpus=TWO_A5000))
        assert rc == 0
        # First event is the pull, carrying the resolved top-fit preset id.
        kinds = [e[0] for e in events]
        assert kinds == ["pull", "launch", "wait_ready", "chat"]
        resolved = events[0][1]
        assert resolved.startswith("prod-"), f"expected a production top-fit, got {resolved}"

    def test_explicit_preset_is_honoured(self, monkeypatch):
        events: list = []
        _patch_pipeline(monkeypatch, events=events)
        _force_tty(monkeypatch)
        rc = RunCommand().execute(_ns(preset="example-2x-tier-aware", fake_gpus=TWO_A5000))
        assert rc == 0
        assert events[0][1] == "example-2x-tier-aware"


class TestCallSequence:
    def test_full_flow_order(self, monkeypatch):
        events: list = []
        _patch_pipeline(monkeypatch, events=events)
        _force_tty(monkeypatch)
        rc = RunCommand().execute(_ns(preset="example-2x-tier-aware", fake_gpus=TWO_A5000))
        assert rc == 0
        assert [e[0] for e in events] == ["pull", "launch", "wait_ready", "chat"]

    def test_non_tty_ready_prints_pointer_not_repl(self, monkeypatch):
        # Without an interactive TTY, a ready engine prints the chat pointer and
        # exits 0 — it never blocks on stdin in a REPL.
        events: list = []
        _patch_pipeline(monkeypatch, events=events)
        _force_tty(monkeypatch, value=False)
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            rc = RunCommand().execute(_ns(preset="example-2x-tier-aware", fake_gpus=TWO_A5000))
        assert rc == 0
        assert "chat" not in [e[0] for e in events]
        assert "sndr chat example-2x-tier-aware" in err.getvalue()

    def test_launch_failure_aborts_before_wait(self, monkeypatch):
        events: list = []
        _patch_pipeline(monkeypatch, events=events)

        def failing_launch(preset_id, *, port=None, dry_run=False):
            events.append(("launch", preset_id, port, dry_run))
            return 3

        monkeypatch.setattr(run_mod, "_launch_detached", failing_launch)
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            rc = RunCommand().execute(_ns(preset="example-2x-tier-aware", fake_gpus=TWO_A5000))
        assert rc == 3
        # Never reached wait_ready / chat after a launch failure.
        assert [e[0] for e in events] == ["pull", "launch"]
        assert "launch" in err.getvalue().lower()

    def test_not_ready_prints_friendly_pointer_not_repl(self, monkeypatch):
        events: list = []
        _patch_pipeline(monkeypatch, events=events)

        def never_ready(host, port, *, timeout, on_progress=None):
            events.append(("wait_ready", host, port, timeout))
            return {"reachable": False, "error": "timed out"}

        monkeypatch.setattr(run_mod, "_wait_ready", never_ready)
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            rc = RunCommand().execute(_ns(preset="example-2x-tier-aware", fake_gpus=TWO_A5000))
        # Engine didn't come up — no chat, a non-zero rc, and a pointer to logs.
        assert rc != 0
        assert "chat" not in [e[0] for e in events]
        text = err.getvalue().lower()
        assert "not ready" in text or "did not become ready" in text


class TestReadyMessage:
    def test_ready_message_names_url_and_sndr_chat(self, monkeypatch):
        events: list = []
        _patch_pipeline(monkeypatch, events=events)
        # Suppress the REPL so we can assert the ready banner that precedes it.
        monkeypatch.setattr(run_mod, "_chat_repl", lambda host, port, *, preset_id: 0)
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            rc = RunCommand().execute(_ns(preset="example-2x-tier-aware", fake_gpus=TWO_A5000))
        assert rc == 0
        text = err.getvalue()
        assert "Ready" in text
        assert "sndr chat" in text
        assert "http://" in text  # the chat URL pointer


class TestDryRun:
    def test_dry_run_plans_without_launching(self, monkeypatch):
        events: list = []
        _patch_pipeline(monkeypatch, events=events)
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            rc = RunCommand().execute(_ns(preset="example-2x-tier-aware", fake_gpus=TWO_A5000, dry_run=True))
        assert rc == 0
        # Dry-run must not launch / wait / chat. The pull step is allowed to run
        # in its own dry-run mode (it plans the download), but never launches.
        kinds = [e[0] for e in events]
        assert "launch" not in kinds
        assert "wait_ready" not in kinds
        assert "chat" not in kinds
        # The resolved plan is reported on stdout.
        assert "example-2x-tier-aware" in out.getvalue()


class TestRegistered:
    def test_run_and_chat_in_registry(self):
        from sndr.cli.commands import COMMAND_REGISTRY
        from sndr.cli.main import build_parser

        build_parser()
        assert "run" in COMMAND_REGISTRY
        assert "chat" in COMMAND_REGISTRY


# ── _launch_detached quiet mode (TUI path must not inherit the terminal) ─────


def test_launch_detached_quiet_suppresses_child_output(monkeypatch):
    """``quiet=True`` (the cockpit path) routes the child ``sndr launch``
    stdout/stderr to DEVNULL so it never corrupts Textual's alt-screen."""
    import subprocess

    captured: dict[str, object] = {}

    def _fake_run(argv, **kw):
        captured["kw"] = kw

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    rc = run_mod._launch_detached("p", quiet=True)
    assert rc == 0
    assert captured["kw"].get("stdout") is subprocess.DEVNULL
    assert captured["kw"].get("stderr") is subprocess.DEVNULL


def test_launch_detached_default_inherits_terminal(monkeypatch):
    """Default (``sndr run`` on a real terminal) keeps inheriting stdout/stderr
    so the operator sees the launch output — no DEVNULL."""
    import subprocess

    captured: dict[str, object] = {}

    def _fake_run(argv, **kw):
        captured["kw"] = kw

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    run_mod._launch_detached("p")
    assert "stdout" not in captured["kw"]
    assert "stderr" not in captured["kw"]
