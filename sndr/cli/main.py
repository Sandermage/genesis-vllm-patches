# SPDX-License-Identifier: Apache-2.0
"""sndr CLI dispatcher.

Examples::

    sndr --version
    sndr engines list
    sndr engines info vllm
    sndr pins list --engine vllm
    sndr health

The CLI is intentionally minimal in v12 — operators primarily use the GUI.
The CLI exists for headless automation (CI scripts, cron jobs, scripts).
"""
from __future__ import annotations

import argparse
import json
import sys

from sndr.cli.commands import COMMAND_REGISTRY, build_subparsers
from sndr.version import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="sndr",
        description="sndr-platform — multi-engine inference patch orchestrator.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--output",
        choices=("json", "yaml", "text"),
        default="text",
        help="Output format (default: text)",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        metavar="COMMAND",
    )
    build_subparsers(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    command = COMMAND_REGISTRY.get(args.command)
    if command is None:
        parser.error(f"Unknown command: {args.command}")

    try:
        return command.execute(args)
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        return 130


if __name__ == "__main__":
    sys.exit(main())
