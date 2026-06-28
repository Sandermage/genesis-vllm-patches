# SPDX-License-Identifier: Apache-2.0
"""CLI command: ``sndr tui`` — interactive terminal cockpit.

A Textual dashboard over the live engine + the fit-ranked preset catalog +
GPU/hosts + a status log, on one keyboard-driven screen. It REUSES the seams the
CLI already exposes — ``launch_wizard.build_catalog`` for the catalog,
``engine_client.engine_status`` / ``engine_metrics`` for live KPIs, and the
``sndr run`` / ``sndr down`` / ``sndr doctor`` / ``sndr chat`` pipelines for the
operate verbs — and owns no business logic of its own.

The cockpit both shows and DRIVES: Enter serves the selected preset, k stops it,
d runs doctor, c chats — each routed through ``sndr.cli.tui.data`` so the TUI is
a view-and-control-over-the-CLI, never a parallel implementation.

Textual is an optional ``[tui]`` extra: the base CLI and daemon run without it.
When textual is absent ``sndr tui`` prints a one-line install hint and exits
non-zero — never a traceback (mirrors how ``sndr up`` gates the ``gui-api``
extra).
"""
from __future__ import annotations

import argparse

from sndr.cli._messages import Emitter


class TuiCommand:
    name = "tui"
    help = "Open the interactive terminal cockpit (live dashboard + serve/stop/chat)."

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--rig",
            default=None,
            help="Plan the catalog against a named builtin rig instead of the "
            "detected hardware (offline — e.g. for a demo on a GPU-less box).",
        )
        parser.add_argument(
            "--fake-gpus",
            default=None,
            help="Synthesize a rig from a GPU spec (e.g. 'RTX A5000:24564:8.6') "
            "when no GPU is present — the same flag the launch wizard accepts.",
        )
        parser.add_argument(
            "--lean",
            action="store_true",
            help="Beginner layout — hide the operator panes (GPU/rig + log), "
            "leaving just the catalog (what can I run) and engine status.",
        )

    def execute(self, args: argparse.Namespace) -> int:
        em = Emitter()
        try:
            import textual  # noqa: F401
        except ImportError:
            em.err("the terminal cockpit needs the 'tui' extra (textual is not installed)")
            em.hint("install it:  pip install 'vllm-sndr-core[tui]'")
            em.hint("or use the guided menu instead:  sndr")
            return 1

        from sndr.cli.tui.app import run_tui

        return run_tui(
            rig=getattr(args, "rig", None),
            fake_gpus=getattr(args, "fake_gpus", None),
            lean=getattr(args, "lean", False),
        )


__all__ = ["TuiCommand"]
