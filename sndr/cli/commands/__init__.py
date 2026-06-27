# SPDX-License-Identifier: Apache-2.0
"""Command registry. Each command is a class implementing :class:`Command`."""
from __future__ import annotations

import argparse
from typing import Protocol

from sndr.cli.commands.engines import EnginesListCommand, EnginesInfoCommand
from sndr.cli.commands.health import HealthCommand
from sndr.cli.commands.kv_calc import KvCalcCommand
from sndr.cli.commands.launch import LaunchCommand
from sndr.cli.commands.pins import PinsListCommand
from sndr.cli.commands.preflight import PreflightCommand


class Command(Protocol):
    """Contract every CLI command implements."""
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
    register(PinsListCommand())
    register(HealthCommand())
    register(PreflightCommand())
    register(KvCalcCommand())
    register(_FitAlias())

    for name, cmd in sorted(COMMAND_REGISTRY.items()):
        sub = subparsers.add_parser(name, help=cmd.help)
        cmd.configure_parser(sub)


__all__ = ["COMMAND_REGISTRY", "Command", "build_subparsers", "register"]
