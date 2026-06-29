# SPDX-License-Identifier: Apache-2.0
"""The ``sndr tui`` Textual application — read-only cockpit (Phase 1).

One keyboard-driven screen with four panes that map the discover → operate →
validate journey onto a single view:

  * ENGINE / HEALTH — live status + KPIs (tok/s, KV%, TTFT, running/waiting)
    from :func:`engine_client.engine_metrics`/``engine_status``;
  * CATALOG — the fit-ranked preset rows from ``launch_wizard.build_catalog``
    (a ``✓``/``✗`` fit glyph + preset + status + measured metric);
  * GPU / RIG — the resolved rig (GPUs, VRAM, compute capability);
  * LOG — a rolling status line.

It owns no business logic: every value flows through :mod:`sndr.cli.tui.data`,
the thin facade over the same seams the CLI uses. Blocking probes AND the operate
verbs run in Textual thread workers so the UI never stalls.

Phase 2 adds the operate verbs onto the same cockpit, each routed through the
data facade so the TUI stays a view-over-the-CLI:

  * Enter — serve the selected catalog preset (confirm → ``data.serve`` =
    ``sndr run``'s pull+launch pipeline; the live refresh shows it come up);
  * k — stop the selected preset's engine (confirm → ``data.stop`` =
    ``sndr down``'s ``docker stop``);
  * d / c — doctor / chat, run under ``App.suspend()`` (drop to the terminal for
    the same ``sndr doctor`` / ``sndr chat``, restore the cockpit on exit).

``textual`` is imported here (the optional ``[tui]`` extra); the ``sndr tui``
command gates on it before importing this module.
"""
from __future__ import annotations

from typing import Any, Optional

from textual import work
from textual.app import App, ComposeResult, SuspendNotSupported
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Header, Input, RichLog, Static

from sndr.cli.tui import data as _data

# Pure render helpers live in the textual-free render module so they unit-test
# without the optional [tui] extra; the app imports them rather than redefining
# the same wording. Re-exported below for back-compat with any caller that
# imported them from this module.
from sndr.cli.tui.render import (
    GLYPH_FIT,
    HELP_TEXT as _HELP_TEXT,
    catalog_rows,
    render_engine,
    render_gpu,
)


class HelpScreen(ModalScreen):
    """A dismissible help overlay (Esc / ? / q to close)."""

    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("question_mark", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
    ]

    def compose(self) -> ComposeResult:
        yield Static(_HELP_TEXT, id="help-body")


class ConfirmScreen(ModalScreen[bool]):
    """A yes/no gate for the heavy operate verbs (serve / stop).

    Dismisses ``True`` on y / Enter, ``False`` on n / Esc — the caller passes a
    callback to ``push_screen`` and only acts on a confirmed yes.
    """

    BINDINGS = [
        ("y", "yes", "Yes"),
        ("enter", "yes", "Yes"),
        ("n", "no", "No"),
        ("escape", "no", "No"),
    ]

    def __init__(self, prompt: str) -> None:
        super().__init__()
        self._prompt = prompt

    def compose(self) -> ComposeResult:
        yield Static(
            f"{self._prompt}\n\n  y / Enter — yes      n / Esc — no",
            id="confirm-body",
        )

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)


class SettingsScreen(ModalScreen[Optional[dict]]):
    """First-run-friendly settings: Model Dir + HF token.

    Pre-filled from ``data.load_settings`` (the live env / saved state), it
    dismisses a ``{"model_dir", "hf_token"}`` dict on save (Enter) or ``None`` on
    cancel (Esc). A blank token field means "keep the current token" — the
    facade treats it as a no-op for that key.
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, current: dict) -> None:
        super().__init__()
        self._current = current or {}

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-body"):
            yield Static("settings", id="settings-title")
            yield Input(
                value=self._current.get("model_dir", ""),
                placeholder="model directory (SNDR_MODELS_DIR)",
                id="set-model-dir",
            )
            yield Input(
                value="",
                placeholder="HF token (blank = keep current)",
                password=True,
                id="set-hf-token",
            )
            yield Static("Enter — save      Esc — cancel", id="settings-hint")

    def on_mount(self) -> None:
        self.query_one("#set-model-dir", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.action_save()

    def action_save(self) -> None:
        self.dismiss({
            "model_dir": self.query_one("#set-model-dir", Input).value,
            "hf_token": self.query_one("#set-hf-token", Input).value,
        })

    def action_cancel(self) -> None:
        self.dismiss(None)


class SndrCockpit(App):
    """The sndr cockpit — live dashboard + operate verbs (serve/stop/doctor/chat)."""

    TITLE = "sndr"
    CSS = """
    #top { height: 1fr; }
    #left { width: 38%; }
    #right { width: 62%; }
    #engine, #gpu { border: round $primary; padding: 0 1; height: 1fr; }
    #catalog { height: 60%; border: round $primary; }
    #log { height: 40%; border: round $secondary; padding: 0 1; }
    HelpScreen, ConfirmScreen { align: center middle; }
    #help-body { width: 64; border: round $primary; padding: 1 2; background: $surface; }
    #confirm-body { width: 60; border: round $warning; padding: 1 2; background: $surface; }
    #settings-body { width: 64; height: auto; border: round $primary; padding: 1 2; background: $surface; }
    #settings-title { text-style: bold; padding-bottom: 1; }
    #settings-hint { color: $text-muted; padding-top: 1; }
    """

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("k", "stop", "Stop"),
        ("d", "doctor", "Doctor"),
        ("c", "chat", "Chat"),
        ("s", "settings", "Settings"),
        ("slash", "filter", "Filter"),
        ("question_mark", "help", "Help"),
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        *,
        rig: Optional[str] = None,
        fake_gpus: Optional[str] = None,
        loader: Any = _data,
        lean: bool = False,
    ) -> None:
        super().__init__()
        self._rig = rig
        self._fake_gpus = fake_gpus
        self._data = loader  # the facade — injectable so tests fake one surface
        self._catalog: Any = None
        # Lean (beginner) mode hides the operator-detail panes — GPU/rig and the
        # status log — leaving just "what can I run" (catalog) + "is it serving"
        # (engine). Action feedback then surfaces as a toast instead of the log.
        self._lean = lean

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="top"):
            with Vertical(id="left"):
                yield Static("loading…", id="engine")
                yield Static("", id="gpu")
            with Vertical(id="right"):
                yield DataTable(id="catalog")
                yield RichLog(id="log", max_lines=200, wrap=True, markup=False)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#catalog", DataTable)
        table.add_columns("fit", "preset", "status", "metric")
        table.cursor_type = "row"
        table.focus()  # ↑/↓ navigate, Enter serves the selected preset
        if self._lean:
            # Hide operator detail; let the catalog + engine fill the screen.
            self.query_one("#gpu", Static).display = False
            self.query_one("#log", RichLog).display = False
            # Reclaim the space the hidden panes leave so nothing looks empty.
            self.query_one("#catalog", DataTable).styles.height = "1fr"
        self._log("sndr cockpit.  Enter serve · k stop · d doctor · c chat · ? help · q quit")
        self._load_catalog()
        self._refresh_engine()
        # Live KPIs refresh on a gentle cadence; catalog is static-ish (manual r).
        self.set_interval(3.0, self._refresh_engine)

    # ── workers (blocking probes off the UI thread) ──────────────────────────

    # Each worker family has its OWN group so a refresh (the `r` key / the 3s
    # interval) never cancels an in-flight serve/stop — Textual's default group
    # + exclusive=True would otherwise cross-cancel them.
    @work(thread=True, exclusive=True, group="catalog")
    def _load_catalog(self) -> None:
        try:
            catalog = self._data.load_catalog(rig=self._rig, fake_gpus=self._fake_gpus)
        except Exception as exc:  # pragma: no cover — defensive
            self._post(self._log, f"catalog error: {exc}")
            return
        self._post(self._apply_catalog, catalog)

    @work(thread=True, exclusive=False, group="engine")
    def _refresh_engine(self) -> None:
        try:
            snap = self._data.engine_snapshot()
        except Exception as exc:  # pragma: no cover — defensive
            self._post(self._log, f"engine probe error: {exc}")
            return
        self._post(self._apply_engine, snap)

    # ── apply (UI thread) ────────────────────────────────────────────────────

    def _apply_catalog(self, catalog: Any) -> None:
        self._catalog = catalog
        table = self.query_one("#catalog", DataTable)
        table.clear()
        rows = catalog_rows(catalog)
        fitting = 0
        for fit, preset_id, status, metric in rows:
            fitting += 1 if fit == "✓" else 0
            table.add_row(fit, preset_id, status, metric)
        rig = getattr(catalog, "rig", None)
        self.query_one("#gpu", Static).update(render_gpu(rig))
        try:
            self.sub_title = self._data.rig_summary(rig) if rig is not None else ""
        except Exception:  # pragma: no cover — cosmetic only
            self.sub_title = ""
        self._log(f"catalog: {len(rows)} presets · {fitting} fit this rig")

    def _apply_engine(self, snap: dict[str, Any]) -> None:
        self.query_one("#engine", Static).update(render_engine(snap))

    # ── serve / stop workers (blocking pipeline off the UI thread) ────────────

    # serve + stop share the "lifecycle" group (mutually exclusive — you don't
    # serve and stop the same engine at once) but are isolated from catalog /
    # engine refresh so pressing `r` can't cancel a launch in flight.
    @work(thread=True, exclusive=True, group="lifecycle")
    def _serve_worker(self, preset_id: str) -> None:
        result = self._data.serve(preset_id)
        if result.get("ok"):
            self._post(self._log, f"launch started for {preset_id} — engine coming up")
        else:
            self._post(self._log, f"serve failed: {result.get('error')}")
        self._post(self._refresh_engine)

    @work(thread=True, exclusive=True, group="lifecycle")
    def _stop_worker(self, preset_id: str) -> None:
        result = self._data.stop(preset_id)
        if not result.get("ok"):
            self._post(self._log, f"stop error: {result.get('error')}")
        elif result.get("stopped"):
            self._post(self._log, f"stopped {preset_id}")
        else:
            self._post(self._log, f"{preset_id}: nothing was running")
        self._post(self._refresh_engine)

    # ── actions ──────────────────────────────────────────────────────────────

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Enter on a catalog row → serve that preset (gated by a confirm)."""
        try:
            row = event.data_table.get_row(event.row_key)
        except Exception:  # pragma: no cover — defensive (race on a cleared table)
            return
        self._serve_row(list(row))

    def _serve_row(self, row: list[Any]) -> None:
        if len(row) < 2:  # pragma: no cover — defensive
            return
        fit, preset_id = row[0], row[1]
        if fit != GLYPH_FIT:
            self._log(f"{preset_id} doesn't fit this rig — pick a ✓ row to serve")
            return

        def _go(confirmed: Optional[bool]) -> None:
            if confirmed:
                self._log(f"serving {preset_id} … (watch the engine pane; k to stop)")
                self._serve_worker(preset_id)
            else:
                self._log("serve cancelled")

        self.push_screen(
            ConfirmScreen(f"Serve {preset_id}?  (pull weights if needed + launch)"),
            _go,
        )

    def action_stop(self) -> None:
        preset_id = self._focused_preset()
        if not preset_id:
            self._log("nothing selected to stop")
            return

        def _go(confirmed: Optional[bool]) -> None:
            if confirmed:
                self._log(f"stopping {preset_id} …")
                self._stop_worker(preset_id)
            else:
                self._log("stop cancelled")

        self.push_screen(ConfirmScreen(f"Stop the {preset_id} engine?"), _go)

    def action_doctor(self) -> None:
        self._log("running doctor … (cockpit resumes after)")
        try:
            with self.suspend():
                rc = self._data.run_doctor()
        except SuspendNotSupported:
            self._log("doctor needs a real terminal here (can't suspend)")
            return
        except Exception as exc:  # the doctor command itself errored
            self._log(f"doctor failed: {exc}")
            return
        self._log(
            "doctor done — back in the cockpit" if not rc
            else f"doctor finished with issues (rc={rc}) — back in the cockpit"
        )

    def action_chat(self) -> None:
        preset_id = self._focused_preset()
        self._log(f"chat → {preset_id or 'default engine'} (cockpit resumes after)")
        try:
            with self.suspend():
                rc = self._data.run_chat(preset_id)
        except SuspendNotSupported:
            self._log("chat needs a real terminal here (can't suspend)")
            return
        except Exception as exc:  # the chat command itself errored
            self._log(f"chat failed: {exc}")
            return
        self._log(
            "chat closed — back in the cockpit" if not rc
            else "no engine reachable — start one with Enter (serve) — back in the cockpit"
        )

    def action_settings(self) -> None:
        def _go(result: Optional[dict]) -> None:
            if not result:
                self._log("settings unchanged")
                return
            res = self._data.save_settings(
                model_dir=result.get("model_dir", ""),
                hf_token=result.get("hf_token", ""),
            )
            if res.get("ok"):
                self._log(f"settings saved — model dir: {res.get('model_dir') or '(unset)'}")
            else:
                self._log(f"settings error: {res.get('error')}")

        self.push_screen(SettingsScreen(self._data.load_settings()), _go)

    def action_refresh(self) -> None:
        self._log("refreshing…")
        self._load_catalog()
        self._refresh_engine()

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_filter(self) -> None:
        # Phase 1: a focused-row jump hint; full filter input lands with actions.
        self._log("filter: type a preset name then ↑/↓ (full filter in Phase 2)")
        self.query_one("#catalog", DataTable).focus()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _post(self, fn: Any, *args: Any) -> None:
        """``call_from_thread`` guarded against a torn-down app.

        A worker thread (serve/stop/refresh) may still be running ``docker`` when
        the user quits; its trailing UI callbacks would then hit a dead loop and
        raise ``RuntimeError('App is not running')`` in the thread. Swallow that
        one teardown race — there is no UI left to update anyway.
        """
        try:
            self.call_from_thread(fn, *args)
        except Exception:  # pragma: no cover — app loop gone (quit mid-worker)
            pass

    def _focused_preset(self) -> Optional[str]:
        """The preset id of the currently-highlighted catalog row, if any."""
        table = self.query_one("#catalog", DataTable)
        if table.row_count == 0:
            return None
        try:
            return table.get_row_at(table.cursor_row)[1]
        except Exception:  # pragma: no cover — defensive
            return None

    def _log(self, msg: str) -> None:
        self.query_one("#log", RichLog).write(msg)
        if self._lean:
            # The log pane is hidden in lean mode — surface feedback as a toast.
            try:
                self.notify(msg)
            except Exception:  # pragma: no cover — notify needs a running app
                pass


def run_tui(
    rig: Optional[str] = None,
    fake_gpus: Optional[str] = None,
    lean: bool = False,
) -> int:
    """Launch the cockpit. Returns a process exit code."""
    # Apply any persisted Model Dir / HF token to the env first, so this
    # session's serve/pull (and the child `sndr launch`) use the saved config
    # without re-typing — the "two-keystroke first run" the Settings modal sets.
    try:
        _data.apply_saved_settings()
    except Exception:  # pragma: no cover — never block launch on a settings read
        pass
    SndrCockpit(rig=rig, fake_gpus=fake_gpus, lean=lean).run()
    return 0


__all__ = [
    "SndrCockpit",
    "HelpScreen",
    "ConfirmScreen",
    "SettingsScreen",
    "run_tui",
    "render_engine",
    "render_gpu",
]
