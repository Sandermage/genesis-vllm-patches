# SPDX-License-Identifier: Apache-2.0
"""CLI commands for pin resources."""
from __future__ import annotations

import argparse
import json

from sndr.product_api.domain.pins_service import list_pins


class PinsListCommand:
    name = "pins.list"
    help = "List pins available for an engine."

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--engine",
            default="vllm",
            help="Engine name (default: vllm)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        pins = list_pins(args.engine)

        if args.output == "json":
            print(json.dumps([p.model_dump(mode="json") for p in pins], indent=2, default=str))
            return 0

        if not pins:
            print(f"No pin manifests found for engine '{args.engine}'.")
            print("Generate one with: sndr manifest generate --engine <engine> --pin <version>")
            return 0

        print(f"{'PIN':<35} {'STATUS':<12} {'VERSION':<40}")
        for p in pins:
            print(f"{p.pin:<35} {p.status:<12} {p.full_version:<40}")
        return 0


__all__ = ["PinsListCommand"]
