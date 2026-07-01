# SPDX-License-Identifier: Apache-2.0
"""Command registry. Each command is a class implementing :class:`Command`."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import argparse

from sndr.cli.commands.chat import ChatCommand
from sndr.cli.commands.engines import EnginesInfoCommand, EnginesListCommand
from sndr.cli.commands.health import HealthCommand
from sndr.cli.commands.kv_calc import KvCalcCommand
from sndr.cli.commands.launch import LaunchCommand
from sndr.cli.commands.mem import (
    MemRecallCommand,
    MemRememberCommand,
    MemSearchCommand,
    MemStatsCommand,
)
from sndr.cli.commands.pins import PinsListCommand
from sndr.cli.commands.preflight import PreflightCommand
from sndr.cli.commands.promoted import PROMOTED_COMMANDS
from sndr.cli.commands.run import RunCommand
from sndr.cli.commands.tui import TuiCommand
from sndr.cli.commands.up import DownCommand, OpenCommand, UpCommand


class Command(Protocol):
    """Contract every CLI command implements.

    A command MAY set a class attribute ``add_help = False`` to have the
    registrar build its subparser with ``add_help=False`` — used by the
    promoted pass-through commands so ``--help`` forwards to the legacy
    delegate instead of being intercepted by a stub subparser.
    """
    name: str
    help: str

    def configure_parser(self, parser: argparse.ArgumentParser) -> None: ...
    def execute(self, args: argparse.Namespace) -> int: ...


class _FitAlias(KvCalcCommand):
    """``sndr fit`` — alias for ``sndr kv-calc`` (same byte-level projection)."""
    name = "fit"
    help = "Alias for `kv-calc`: per-card VRAM/KV projection with PASS/TIGHT/FAIL."


COMMAND_REGISTRY: dict[str, Command] = {}


def register(command: Command) -> None:
    COMMAND_REGISTRY[command.name] = command


def build_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Register every command with the parent argument parser."""
    register(EnginesListCommand())
    register(EnginesInfoCommand())
    register(LaunchCommand())
    register(RunCommand())
    register(ChatCommand())
    # UX R3: Harbor-style one-command full-stack bring-up (engine + GUI daemon).
    register(UpCommand())
    register(OpenCommand())
    register(DownCommand())
    register(PinsListCommand())
    register(HealthCommand())
    register(PreflightCommand())
    # Persistent neural-graph memory over the running daemon (distinct from the
    # legacy `sndr memory` VRAM estimator). Dotted names + spaced aliases in main.
    register(MemRememberCommand())
    register(MemRecallCommand())
    register(MemSearchCommand())
    register(MemStatsCommand())
    # TUI cockpit (read-only Phase 1) — the command gates on the optional [tui]
    # extra (textual) with a friendly install hint when it's absent.
    register(TuiCommand())
    register(KvCalcCommand())
    register(_FitAlias())

    # v12 CLI split-brain closure: promote the high-value legacy commands
    # (report / doctor / preset / bench / tune / config) onto the canonical
    # surface. Thin pass-throughs that delegate to the legacy impl, so the
    # canonical and legacy entry points cannot drift.
    for cmd in PROMOTED_COMMANDS:
        register(cmd)

    for name, cmd in sorted(COMMAND_REGISTRY.items()):
        # A command may opt out of argparse's auto-help (``add_help=False``)
        # so ``--help`` forwards verbatim to a delegate. Default is True.
        sub = subparsers.add_parser(
            name, help=cmd.help, add_help=getattr(cmd, "add_help", True),
        )
        cmd.configure_parser(sub)


__all__ = ["COMMAND_REGISTRY", "Command", "build_subparsers", "register"]
