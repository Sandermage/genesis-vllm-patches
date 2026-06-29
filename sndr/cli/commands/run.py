# SPDX-License-Identifier: Apache-2.0
"""CLI command: ``sndr run [preset]`` — the Ollama ``run`` verb.

One command, from a cold start to a chat prompt — wiring the pieces that
already exist, not a new engine:

  1. **Resolve** the preset. An explicit ``sndr run <preset>`` is honoured; an
     omitted preset resolves to the launch wizard's top-ranked *fitting* preset
     for the detected (or ``--rig`` / ``--fake-gpus``) rig — the same
     :func:`sndr.cli.wizard.launch_wizard.build_catalog` ranking the
     interactive wizard shows.
  2. **Ensure weights** — call the existing ``compat.models.pull`` artifacts
     path (``pull_via_artifacts``), which is a no-op when the model is already
     complete (it verifies before downloading). Reused, not reinvented.
  3. **Launch** — hand to the existing ``sndr launch <preset>`` flag path. We
     run it as a CHILD process (not the in-process ``run_launch``, which ends
     in ``os.execvp`` and would replace us) so this orchestrator survives to
     poll readiness and open the chat. The launch renders ``docker run -d`` —
     a detached container — so the child returns once the container is up.
  4. **Wait ready** — poll the product-API engine client's ``/health`` probe
     (:func:`sndr.product_api.legacy.engine_client.engine_status`) with a sane
     timeout and a friendly progress line.
  5. **Chat** — drop into the minimal REPL (:mod:`sndr.cli.chat_repl`) against
     the running engine via the OpenAI-compatible endpoint. If the engine never
     comes up, print a clear "not ready — check the logs" pointer instead.

Headless: ``--dry-run`` resolves + reports the plan WITHOUT launching;
``--no-input`` auto-picks the top fit and, after a ready engine, prints the
ready-pointer instead of blocking on an interactive REPL (so CI / scripted
callers never hang on stdin).

Examples::

    sndr run                                  # top-fit for the live rig → chat
    sndr run prod-qwen3.6-35b-balanced        # named preset → chat
    sndr run --dry-run                         # plan only (no launch)
    sndr run --fake-gpus "RTX 3090:24576:8.6" --dry-run   # offline plan
    sndr run prod-... --no-input               # launch + ready-pointer, no REPL
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Callable, Optional

from sndr.cli._messages import Emitter, heartbeat

_DEFAULT_TIMEOUT_S = 300


class RunCommand:
    name = "run"
    help = "Launch a preset and chat — resolve → pull → launch → wait → chat."

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "preset", nargs="?", default=None,
            help="Preset to run (e.g. prod-qwen3.6-35b-balanced). Omit to use "
                 "the top-ranked fitting preset for the detected rig.",
        )
        parser.add_argument(
            "--port", type=int, default=None,
            help="Override the preset's port (engine + readiness probe).",
        )
        parser.add_argument(
            "--dry-run", action="store_true",
            help="Resolve + report the plan without launching, waiting or chatting.",
        )
        parser.add_argument(
            "--no-input", action="store_true",
            help="Headless: auto-pick the top fit and, once ready, print the "
                 "chat pointer instead of opening an interactive REPL.",
        )
        parser.add_argument(
            "--timeout", type=int, default=_DEFAULT_TIMEOUT_S, metavar="SECONDS",
            help=f"Seconds to wait for the engine to become ready "
                 f"(default: {_DEFAULT_TIMEOUT_S}).",
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

    # ── dispatch ─────────────────────────────────────────────────────────────

    def execute(self, args: argparse.Namespace) -> int:
        em = Emitter()  # advisory output → stderr (stdout stays scriptable)

        # 1) Resolve the preset (explicit, else top-fit for the rig).
        try:
            preset_id, port = _resolve_preset_and_port(
                args.preset, rig_id=args.rig, fake_gpus=args.fake_gpus,
                port_override=args.port,
            )
        except _ResolveError as exc:
            em.line(f"sndr run: {exc}")
            em.line("  list presets: sndr preset list   |   pick interactively: sndr")
            return 2

        host = "127.0.0.1"
        url = f"http://{host}:{port}/v1"
        em.blank()
        em.line(f"  sndr run — preset: {preset_id}  (engine → {url})")

        # --dry-run: report the plan, launch nothing.
        if args.dry_run:
            em.line("  (dry-run) plan:")
            em.line(f"    1. ensure weights present  (pull {preset_id})")
            em.line(f"    2. launch                  (sndr launch {preset_id})")
            em.line(f"    3. wait for engine ready   ({host}:{port}/health, "
                    f"timeout {args.timeout}s)")
            em.line(f"    4. chat                    (sndr chat {preset_id})")
            # Mirror the resolved preset on stdout for scriptability.
            print(f"sndr run plan: {preset_id} (port {port})")
            # Let the pull step plan its own download in dry-run too.
            _pull_if_missing(preset_id, dry_run=True)
            return 0

        # 2) Ensure weights are present (no-op when already complete).
        em.line("  [1/3] ensuring model weights are present …")
        rc = _pull_if_missing(preset_id, dry_run=False)
        if rc not in (0, None):
            em.err(f"weights not ready (pull rc={rc}).")
            em.hint(f"fetch them manually: python3 -m sndr.compat.models.pull "
                    f"--config {preset_id}")
            return rc or 1

        # 3) Launch (detached) via the existing launch flag path.
        em.line("  [2/3] launching the engine …")
        rc = _launch_detached(preset_id, port=args.port, dry_run=False)
        if rc not in (0, None):
            em.err(f"launch failed (rc={rc}).")
            em.hint(f"re-run for the full diagnostics: sndr launch {preset_id}")
            return rc or 1

        # 4) Wait for the engine to become ready.
        em.line(f"  [3/3] waiting for the engine on {host}:{port} "
                f"(timeout {args.timeout}s) …")
        status = _wait_ready(
            host, port, timeout=args.timeout, on_progress=heartbeat(em.line),
        )
        if not (status or {}).get("reachable"):
            detail = (status or {}).get("error") or "no /health response"
            em.blank()
            em.err(f"engine did not become ready in {args.timeout}s ({detail}).")
            em.hint("find the container:        docker ps --filter ancestor=vllm")
            em.hint("then tail its logs:        docker logs -f <container>")
            em.hint(f"once it is up, chat with:  sndr chat {preset_id}")
            return 1

        # 5) Ready — chat (or, headless, the ready-pointer).
        models = (status or {}).get("models") or []
        served = models[0] if models else preset_id
        em.blank()
        em.ok(f"Ready — chat at {url}  (model: {served})")
        em.hint(f"or later:  sndr chat {preset_id}")

        if args.no_input:
            em.line("  (--no-input: not opening the interactive REPL)")
            return 0
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            em.line("  (non-interactive stdin/stdout — not opening the REPL; "
                    f"chat with: sndr chat {preset_id})")
            return 0

        em.blank()
        return _chat_repl(host, port, preset_id=preset_id)


# ── resolution ───────────────────────────────────────────────────────────────


class _ResolveError(Exception):
    """The preset (or top-fit) could not be resolved."""


def _resolve_preset_and_port(
    preset: Optional[str],
    *,
    rig_id: Optional[str],
    fake_gpus: Optional[str],
    port_override: Optional[int],
) -> tuple[str, int]:
    """Resolve (preset_id, host_port). Explicit preset wins; otherwise the
    wizard's top-ranked fitting preset for the rig. Raises :class:`_ResolveError`
    on an unknown preset or when nothing fits."""
    from sndr.model_configs.registry_v2 import list_presets, load_alias

    if preset is not None:
        if preset not in set(list_presets()):
            raise _ResolveError(f"unknown preset {preset!r}")
        preset_id = preset
    else:
        preset_id = _top_fit_preset(rig_id=rig_id, fake_gpus=fake_gpus)

    port = port_override
    if port is None:
        try:
            cfg = load_alias(preset_id)
            docker = getattr(cfg, "docker", None)
            if docker is not None and hasattr(docker, "effective_host_port"):
                port = int(docker.effective_host_port())
            else:
                port = int(getattr(cfg, "port", None) or 8000)
        except Exception:
            port = 8000
    return preset_id, int(port)


def _top_fit_preset(*, rig_id: Optional[str], fake_gpus: Optional[str]) -> str:
    """Return the wizard's top-ranked fitting preset for the rig (the same
    ranking ``sndr launch`` / ``sndr`` shows). Raises when nothing fits."""
    from sndr.cli.wizard.launch_wizard import build_catalog
    from sndr.model_configs.preflight_fit import (
        RigProbe,
        rig_from_fake_spec,
        rig_from_hardware_def,
    )
    from sndr.model_configs.registry_v2 import (
        list_presets,
        load_alias,
        load_hardware,
        load_preset_def,
    )

    if fake_gpus is not None:
        rig = rig_from_fake_spec(fake_gpus)
    elif rig_id is not None:
        rig = rig_from_hardware_def(load_hardware(rig_id), source=f"rig:{rig_id}")
    else:
        rig = RigProbe().detect()

    catalog = build_catalog(
        rig, preset_ids=list_presets(),
        card_loader=load_preset_def, cfg_loader=load_alias,
    )
    fitting = catalog.menu(show_all=False)
    if not fitting:
        raise _ResolveError(
            "no preset fits this rig — run `sndr` to pick one interactively, "
            "or see docs/SINGLE_CARD.md for single-card options"
        )
    return fitting[0].preset_id


# ── pipeline steps (each is a seam mocked in tests) ──────────────────────────


def _has_artifacts_block(preset_id: str) -> bool:
    """True when the preset declares an ``artifacts.models`` block to pull.

    Many V2 presets resolve their model via a host-side mount rather than a
    declared HF artifact; for those there is nothing for the puller to do, and
    calling it would emit a misleading "no artifacts.models block" ERROR. This
    pre-check lets us skip the puller cleanly in that case.
    """
    try:
        from sndr.model_configs.registry_v2 import load_alias

        cfg = load_alias(preset_id)
        artifacts = getattr(cfg, "artifacts", None)
        return bool(artifacts and getattr(artifacts, "models", None))
    except Exception:
        return False


def _pull_if_missing(preset_id: str, *, dry_run: bool = False) -> int:
    """Ensure the preset's model weights are present. Reuses the existing
    artifacts puller, which verifies before downloading (so this is a no-op
    when the weights are already complete). Returns the puller's rc.

    A preset without an ``artifacts.models`` block has nothing to pull (its
    model comes from a host mount) — skip the puller so the launch path's own
    mount resolution / preflight catches a truly missing model path with a
    precise message, instead of a misleading puller ERROR here.
    """
    if not _has_artifacts_block(preset_id):
        return 0
    from sndr.compat.models.pull import pull_via_artifacts

    rc = pull_via_artifacts(preset_id, dry_run=dry_run)
    if rc == 2:
        # Defensive: a late artifacts/verify edge — not a download failure.
        return 0
    return rc


def _launch_detached(
    preset_id: str, *, port: Optional[int] = None, dry_run: bool = False,
    quiet: bool = False,
) -> int:
    """Launch the preset as a CHILD process so this orchestrator survives.

    The in-process launcher (``run_launch``) ends in ``os.execvp`` — it would
    replace us and we could never poll readiness or open the chat. Running
    ``sndr launch <preset>`` in a subprocess keeps us alive: the rendered
    script does ``docker run -d`` (detached), so the child returns once the
    container is up. Returns the child's exit code.

    ``quiet=True`` routes the child's stdout/stderr to ``DEVNULL`` — used by the
    TUI cockpit, where the child would otherwise inherit and corrupt Textual's
    alt-screen terminal. The default (``sndr run`` on a real terminal) inherits
    stdout/stderr so the operator sees the launch output.
    """
    import subprocess

    argv = [sys.executable, "-m", "sndr", "launch", preset_id]
    if port is not None:
        argv += ["--port", str(port)]
    if dry_run:
        argv.append("--dry-run")
    redirect = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL} if quiet else {}
    try:
        completed = subprocess.run(argv, **redirect)
        return completed.returncode
    except FileNotFoundError:
        # `python -m sndr` unavailable (no console entry on PATH) — fall back to
        # the in-process flag path. This execs and replaces us; acceptable as a
        # last resort (the operator still gets a launched engine, just no REPL).
        from sndr.cli.commands.launch import LaunchCommand
        delegate = argparse.Namespace(preset=preset_id, port=port, dry_run=False)
        return LaunchCommand()._run_flag_path(delegate)


def _wait_ready(
    host: str,
    port: int,
    *,
    timeout: int,
    on_progress: Optional[Callable[[float], None]] = None,
) -> dict[str, Any]:
    """Poll the engine's ``/health`` until ``reachable`` or the timeout. Returns
    the last :func:`engine_status` payload (``reachable=False`` on timeout)."""
    from sndr.product_api.legacy import engine_client

    deadline = time.monotonic() + max(1, int(timeout))
    last: dict[str, Any] = {"reachable": False, "error": "not polled yet"}
    while time.monotonic() < deadline:
        try:
            last = engine_client.engine_status(host, port=port, timeout=3.0)
        except Exception as exc:  # noqa: BLE001 — keep polling on transient errors
            last = {"reachable": False, "error": f"{type(exc).__name__}: {exc}"}
        if last.get("reachable"):
            return last
        if on_progress is not None:
            on_progress(time.monotonic())
        time.sleep(2.0)
    return last


def _chat_repl(host: str, port: int, *, preset_id: str) -> int:
    """Open the interactive chat REPL against the running engine."""
    from sndr.cli.chat_repl import chat_loop

    return chat_loop(host, port, preset_id=preset_id)


__all__ = ["RunCommand"]
