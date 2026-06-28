# SPDX-License-Identifier: Apache-2.0
"""CLI commands: ``sndr up`` / ``sndr open`` / ``sndr down`` — Harbor-style
one-command bring-up of the WHOLE product.

R1 gave the cold-start-to-chat ``sndr run``; R2 unified the CLI surface. R3
closes the last gap with the verbs a newcomer expects from ``docker compose`` /
``harbor``: ONE command brings up the engine AND the product-API + GUI daemon,
waits until both are ready, and prints the local URL.

These are **thin orchestration** over seams that already exist — never a new
engine, never a new server:

  * the engine is launched through the SAME R1 child-process path the ``run``
    verb uses (``sndr launch <preset>`` → ``docker run -d``), so this
    orchestrator survives to wait for readiness;
  * readiness is the product-API engine client's ``/health`` probe
    (:func:`sndr.product_api.legacy.engine_client.engine_status`), reused
    verbatim from ``run``;
  * the daemon is the EXISTING ``gui-api`` Product API server
    (:mod:`sndr.cli.legacy.gui_api` → ``http_app.run_server``), started as a
    detached child process so this command returns once it answers ``/health``.

``sndr up``:
  1. resolve the preset (explicit, else the wizard's top-fit for the rig — the
     same ranking ``sndr`` / ``sndr run`` show);
  2. launch the engine (detached) and wait for ``/health``;
  3. start the product-API + GUI daemon and wait for its ``/api/v1/health``;
  4. print "✓ sndr is up — open http://127.0.0.1:8765 or run ``sndr open``".

  ``--dry-run`` plans both without starting; ``--no-engine`` brings up only the
  daemon + GUI (for when an engine already runs); ``--no-input`` auto-picks the
  top fit (so CI / scripted callers never block on stdin).

``sndr open`` opens the default browser at the local daemon URL (a friendly,
URL-printing message when there is no browser / a headless host).

``sndr down`` stops what ``up`` started — the engine container (by the preset's
declared ``container_name``) and the daemon process — with a friendly summary.

Examples::

    sndr up                                  # top-fit engine + daemon → URL
    sndr up prod-qwen3.6-35b-balanced         # named preset + daemon
    sndr up --dry-run                          # plan engine + daemon (no start)
    sndr up --no-engine                        # daemon + GUI only
    sndr open                                  # open the GUI in a browser
    sndr down                                  # stop engine + daemon
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Callable, Optional

_DEFAULT_GUI_PORT = 8765
_DEFAULT_GUI_HOST = "127.0.0.1"
_DEFAULT_ENGINE_TIMEOUT_S = 300
_DAEMON_READY_TIMEOUT_S = 60


def _local_url(gui_port: int) -> str:
    """The loopback URL the GUI daemon serves the UI + API on."""
    return f"http://{_DEFAULT_GUI_HOST}:{int(gui_port)}"


# ── sndr up ──────────────────────────────────────────────────────────────────


class UpCommand:
    name = "up"
    help = "Bring up the whole stack — engine + product-API/GUI daemon → URL."

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "preset", nargs="?", default=None,
            help="Preset to launch (e.g. prod-qwen3.6-35b-balanced). Omit to use "
                 "the top-ranked fitting preset for the detected rig.",
        )
        parser.add_argument(
            "--port", type=int, default=None,
            help="Override the engine's port (engine + readiness probe).",
        )
        parser.add_argument(
            "--gui-port", type=int, default=_DEFAULT_GUI_PORT, metavar="PORT",
            help=f"Port for the product-API + GUI daemon (default: {_DEFAULT_GUI_PORT}).",
        )
        parser.add_argument(
            "--no-engine", action="store_true",
            help="Skip the engine; bring up only the daemon + GUI (for when an "
                 "engine already runs).",
        )
        parser.add_argument(
            "--dry-run", action="store_true",
            help="Plan the bring-up (engine + daemon) without starting anything.",
        )
        parser.add_argument(
            "--no-input", action="store_true",
            help="Headless: auto-pick the top-fit preset, never prompt on stdin.",
        )
        parser.add_argument(
            "--timeout", type=int, default=_DEFAULT_ENGINE_TIMEOUT_S, metavar="SECONDS",
            help=f"Seconds to wait for the engine to become ready "
                 f"(default: {_DEFAULT_ENGINE_TIMEOUT_S}).",
        )
        parser.add_argument(
            "--rig", default=None, metavar="HARDWARE_ID",
            help="Resolve the top-fit against a builtin hardware def (offline).",
        )
        parser.add_argument(
            "--fake-gpus", default=None, metavar="SPEC",
            help="Resolve the top-fit against a synthetic rig "
                 "'name:vram_mib:cc;...' (offline).",
        )

    def execute(self, args: argparse.Namespace) -> int:
        def out(msg: str = "") -> None:
            print(msg, file=sys.stderr)

        gui_port = int(getattr(args, "gui_port", _DEFAULT_GUI_PORT))
        no_engine = bool(getattr(args, "no_engine", False))
        url = _local_url(gui_port)

        # ── resolve the engine preset (unless --no-engine) ───────────────────
        preset_id: Optional[str] = None
        engine_port: Optional[int] = None
        if not no_engine:
            from sndr.cli.commands.run import _ResolveError, _resolve_preset_and_port

            try:
                preset_id, engine_port = _resolve_preset_and_port(
                    args.preset, rig_id=args.rig, fake_gpus=args.fake_gpus,
                    port_override=args.port,
                )
            except _ResolveError as exc:
                out(f"sndr up: {exc}")
                out("  list presets: sndr preset list   |   pick interactively: sndr")
                return 2

        out("")
        if no_engine:
            out(f"  sndr up — daemon + GUI only  (→ {url})")
        else:
            out(f"  sndr up — preset: {preset_id}  (engine → 127.0.0.1:{engine_port}, "
                f"GUI → {url})")

        # ── --dry-run: report the plan, start nothing ────────────────────────
        if args.dry_run:
            out("  (dry-run) plan:")
            step = 1
            if not no_engine:
                out(f"    {step}. launch engine            (sndr launch {preset_id})")
                step += 1
                out(f"    {step}. wait for engine ready    (127.0.0.1:{engine_port}/health)")
                step += 1
            out(f"    {step}. start product-API + GUI  (sndr gui-api --port {gui_port})")
            step += 1
            out(f"    {step}. wait for the GUI ready   ({url}/api/v1/health)")
            step += 1
            out(f"    {step}. open the GUI             ({url})")
            # Scriptable mirror on stdout.
            target = preset_id if not no_engine else "(no-engine)"
            print(f"sndr up plan: engine={target} gui={url}")
            return 0

        # ── 1) engine: launch + wait (skipped with --no-engine) ──────────────
        if not no_engine:
            assert preset_id is not None and engine_port is not None
            out("  [engine] launching …")
            rc = _launch_engine_detached(preset_id, port=args.port, dry_run=False)
            if rc not in (0, None):
                out(f"  ✗ engine launch failed (rc={rc}).")
                out(f"    re-run for full diagnostics: sndr launch {preset_id}")
                return rc or 1

            out(f"  [engine] waiting on 127.0.0.1:{engine_port} "
                f"(timeout {args.timeout}s) …")
            status = _wait_engine_ready(
                _DEFAULT_GUI_HOST, engine_port, timeout=args.timeout,
                on_progress=_progress(out, "engine"),
            )
            if not (status or {}).get("reachable"):
                detail = (status or {}).get("error") or "no /health response"
                out("")
                out(f"  ✗ engine did not become ready in {args.timeout}s ({detail}).")
                out("    find the container:  docker ps --filter ancestor=vllm")
                out("    then tail its logs:  docker logs -f <container>")
                return 1
            out("  [engine] ✓ ready")

        # ── 2) daemon: start the existing gui-api server + wait ──────────────
        out(f"  [gui] starting the product-API + GUI on {url} …")
        try:
            handle = _start_daemon(_DEFAULT_GUI_HOST, gui_port)
        except _DaemonStartError as exc:
            out(f"  ✗ could not start the GUI daemon: {exc}")
            out("    install the web extra:  pip install 'vllm-sndr-core[gui-api]'")
            return 3
        ready = _wait_daemon_ready(_DEFAULT_GUI_HOST, gui_port, timeout=_DAEMON_READY_TIMEOUT_S)
        if not ready:
            out(f"  ✗ the GUI daemon did not answer on {url} within "
                f"{_DAEMON_READY_TIMEOUT_S}s.")
            out(f"    start it by hand to see the error:  sndr gui-api --port {gui_port}")
            _detach_handle(handle)
            return 1

        # ── 3) up — print the local URL ──────────────────────────────────────
        out("")
        out(f"  ✓ sndr is up — open {url} or run `sndr open`")
        if not no_engine:
            out(f"    chat from the terminal instead:  sndr chat {preset_id}")
        out(f"    stop everything:                 sndr down"
            + (f" {preset_id}" if not no_engine and preset_id and not str(preset_id).startswith("prod-") else ""))
        return 0


# ── sndr open ────────────────────────────────────────────────────────────────


class OpenCommand:
    name = "open"
    help = "Open the local product-API + GUI in your browser."

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--gui-port", type=int, default=_DEFAULT_GUI_PORT, metavar="PORT",
            help=f"Port the GUI daemon serves on (default: {_DEFAULT_GUI_PORT}).",
        )

    def execute(self, args: argparse.Namespace) -> int:
        gui_port = int(getattr(args, "gui_port", _DEFAULT_GUI_PORT))
        url = _local_url(gui_port)
        # Always print the URL first so a headless operator can copy it even if
        # the browser open is a no-op.
        print(f"  opening {url}", file=sys.stderr)
        opened = _open_browser(url)
        if not opened:
            print(
                f"  no browser available on this host — open it manually: {url}",
                file=sys.stderr,
            )
        # Mirror the URL on stdout for scriptability (`sndr open | …`).
        print(url)
        return 0


# ── sndr down ────────────────────────────────────────────────────────────────


class DownCommand:
    name = "down"
    help = "Stop the stack started by `sndr up` — engine + GUI daemon."

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "preset", nargs="?", default=None,
            help="Preset whose engine container to stop (default: the top-fit "
                 "preset for the rig, matching `sndr up`).",
        )
        parser.add_argument(
            "--gui-port", type=int, default=_DEFAULT_GUI_PORT, metavar="PORT",
            help=f"GUI daemon port to stop (default: {_DEFAULT_GUI_PORT}).",
        )
        parser.add_argument(
            "--dry-run", action="store_true",
            help="Report what would be stopped without stopping anything.",
        )
        parser.add_argument(
            "--rig", default=None, metavar="HARDWARE_ID",
            help="Resolve the engine preset against a builtin hardware def (offline).",
        )
        parser.add_argument(
            "--fake-gpus", default=None, metavar="SPEC",
            help="Resolve the engine preset against a synthetic rig (offline).",
        )

    def execute(self, args: argparse.Namespace) -> int:
        def out(msg: str = "") -> None:
            print(msg, file=sys.stderr)

        dry_run = bool(getattr(args, "dry_run", False))
        out("")
        out("  sndr down — stopping the stack" + (" (dry-run)" if dry_run else "") + " …")

        # Resolve the engine preset so we know which container to stop. Failure
        # here is non-fatal — we still stop the daemon.
        preset_id: Optional[str] = None
        try:
            from sndr.cli.commands.run import _ResolveError, _resolve_preset_and_port

            preset_id, _ = _resolve_preset_and_port(
                args.preset, rig_id=args.rig, fake_gpus=args.fake_gpus,
                port_override=None,
            )
        except Exception as exc:  # noqa: BLE001 — keep going to stop the daemon
            out(f"  (could not resolve an engine preset to stop: {exc})")
            preset_id = None

        engine_stopped = False
        if preset_id is not None:
            engine_stopped = _stop_engine(preset_id, dry_run=dry_run)
            verb = "would stop" if dry_run else ("stopped" if engine_stopped else "no running engine for")
            out(f"  [engine] {verb} {preset_id}")

        daemon_stopped = _stop_daemon(dry_run=dry_run)
        verb = "would stop" if dry_run else ("stopped" if daemon_stopped else "no running")
        out(f"  [gui]    {verb} the product-API + GUI daemon")

        out("")
        if dry_run:
            out("  ✓ sndr down (dry-run) — nothing was stopped")
        else:
            out("  ✓ sndr is down")
        return 0


# ── engine seams (each mocked in tests) — reuse the R1 run path ──────────────


def _launch_engine_detached(preset_id: str, *, port: Optional[int] = None, dry_run: bool = False) -> int:
    """Launch the engine detached via the R1 child-process launcher (reused)."""
    from sndr.cli.commands.run import _launch_detached

    return _launch_detached(preset_id, port=port, dry_run=dry_run)


def _wait_engine_ready(
    host: str, port: int, *, timeout: int,
    on_progress: Optional[Callable[[float], None]] = None,
) -> dict[str, Any]:
    """Poll the engine's ``/health`` until ready (reuses the R1 run probe)."""
    from sndr.cli.commands.run import _wait_ready

    return _wait_ready(host, port, timeout=timeout, on_progress=on_progress)


def _stop_engine(preset_id: str, *, dry_run: bool = False) -> bool:
    """Stop the engine container by the preset's declared ``container_name``.

    Reuses the docker CLI directly (``docker stop <name>``) — the same verb the
    legacy ``service stop`` / ``compose down`` paths shell out to — targeting
    the exact container the launch renders. Returns True when a container was
    stopped (or would be, in dry-run); False when none was found.
    """
    name = _engine_container_name(preset_id)
    if not name:
        return False
    if dry_run:
        return True
    return _docker_stop(name)


def _engine_container_name(preset_id: str) -> Optional[str]:
    """The container name the preset's launch renders (``cfg.docker.container_name``)."""
    try:
        from sndr.model_configs.registry_v2 import load_alias

        cfg = load_alias(preset_id)
        docker = getattr(cfg, "docker", None)
        name = getattr(docker, "container_name", None) if docker is not None else None
        return str(name) if name else None
    except Exception:  # noqa: BLE001 — best-effort name resolution
        return None


def _docker_stop(name: str) -> bool:
    """``docker stop <name>`` — True if it stopped a container, False otherwise."""
    import shutil
    import subprocess

    if shutil.which("docker") is None:
        return False
    try:
        completed = subprocess.run(
            ["docker", "stop", name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return completed.returncode == 0
    except Exception:  # noqa: BLE001 — never raise out of a teardown
        return False


# ── daemon seams (each mocked in tests) — reuse the gui-api server ───────────


class _DaemonStartError(Exception):
    """The product-API + GUI daemon could not be started."""


def _start_daemon(host: str, port: int):
    """Start the product-API + GUI daemon as a detached child process.

    Reuses the EXISTING ``gui-api`` entry point (``python -m sndr.cli.legacy
    gui-api``), so the daemon is the same Product API server the Makefile's
    ``gui-api`` target runs — not a parallel server. Returns the
    :class:`subprocess.Popen` handle (so ``up`` can detach / ``down`` can stop
    it by port). Raises :class:`_DaemonStartError` if the process cannot spawn.
    """
    import subprocess

    argv = [
        sys.executable, "-m", "sndr.cli.legacy", "gui-api",
        "--host", host, "--port", str(int(port)),
    ]
    try:
        return subprocess.Popen(  # noqa: S603 — fixed argv, no shell
            argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except (OSError, ValueError) as exc:
        raise _DaemonStartError(str(exc)) from exc


def _wait_daemon_ready(host: str, port: int, *, timeout: int) -> bool:
    """Poll the daemon's ``/api/v1/health`` until it answers or the timeout."""
    import json
    import urllib.error
    import urllib.request

    url = f"http://{host}:{int(port)}/api/v1/health"
    deadline = time.monotonic() + max(1, int(timeout))
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3.0) as response:  # noqa: S310 — fixed loopback URL
                if 200 <= response.status < 300:
                    body = response.read().decode("utf-8", "replace")
                    try:
                        return (json.loads(body) or {}).get("status") == "ok"
                    except Exception:  # noqa: BLE001 — a 2xx with odd body still counts as up
                        return True
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(1.0)
    return False


def _stop_daemon(*, dry_run: bool = False) -> bool:
    """Stop the running product-API + GUI daemon.

    The daemon is started detached (its child Popen handle does not survive a
    new ``down`` invocation), so we find it by its command line — a python
    process running ``-m sndr.cli.legacy gui-api`` — and terminate it. Returns
    True when one was stopped (or would be, in dry-run); False when none ran.
    """
    pids = _find_daemon_pids()
    if not pids:
        return False
    if dry_run:
        return True
    import os
    import signal

    stopped = False
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            stopped = True
        except (ProcessLookupError, PermissionError, OSError):
            continue
    return stopped


def _find_daemon_pids() -> list[int]:
    """PIDs of running ``sndr gui-api`` daemon processes (via ``pgrep``)."""
    import shutil
    import subprocess

    if shutil.which("pgrep") is None:
        return []
    try:
        completed = subprocess.run(
            ["pgrep", "-f", "sndr.cli.legacy gui-api"],
            capture_output=True, text=True,
        )
    except Exception:  # noqa: BLE001 — never raise out of a teardown
        return []
    pids: list[int] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def _detach_handle(handle: Any) -> None:
    """Best-effort: stop a just-started daemon child that never came ready."""
    try:
        terminate = getattr(handle, "terminate", None)
        if callable(terminate):
            terminate()
    except Exception:  # noqa: BLE001 — best-effort cleanup
        pass


# ── browser seam (mocked in tests) ───────────────────────────────────────────


def _open_browser(url: str) -> bool:
    """Open ``url`` in the default browser. Returns True on a real open, False on
    a headless host / no browser (so the caller can print a friendly pointer)."""
    import webbrowser

    try:
        return bool(webbrowser.open(url))
    except Exception:  # noqa: BLE001 — a missing browser must not raise
        return False


# ── shared progress heartbeat ────────────────────────────────────────────────


def _progress(out: Callable[[str], None], label: str) -> Callable[[float], None]:
    """A throttled readiness heartbeat (one line every ~10s)."""
    state = {"dots": 0}

    def _tick(_now: float) -> None:
        state["dots"] += 1
        if state["dots"] % 5 == 0:
            out(f"    [{label}] … still warming up ({state['dots'] * 2}s)")

    return _tick


__all__ = ["UpCommand", "OpenCommand", "DownCommand"]
