# SPDX-License-Identifier: Apache-2.0
"""CLI commands for engine resources."""
from __future__ import annotations

import argparse
import json
import sys

from sndr.exceptions import EngineNotInstalledError, EngineUnsupportedError
from sndr.product_api.domain.engines_service import (
    get_engine_detail,
    list_engine_summaries,
)


class EnginesListCommand:
    name = "engines.list"
    help = "List all registered engines."

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        pass

    def execute(self, args: argparse.Namespace) -> int:
        summaries = list_engine_summaries()

        if args.output == "json":
            payload = [s.model_dump(mode="json") for s in summaries]
            print(json.dumps(payload, indent=2))
            return 0

        # Text output (default): aligned table
        if not summaries:
            print("No engines registered.")
            return 0

        print(f"{'NAME':<10} {'STATUS':<10} {'VERSION':<40} {'PIN':<25}")
        for s in summaries:
            status = "active" if s.active else "inactive"
            print(f"{s.name:<10} {status:<10} {s.version or '-':<40} {s.pin or '-':<25}")
        return 0


class EnginesInfoCommand:
    name = "engines.info"
    help = "Get detailed info about one engine."

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("engine", help="Engine name (e.g. vllm)")

    def execute(self, args: argparse.Namespace) -> int:
        try:
            detail = get_engine_detail(args.engine)
        except EngineUnsupportedError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        except EngineNotInstalledError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 3

        if args.output == "json":
            print(json.dumps(detail.model_dump(mode="json"), indent=2))
            return 0

        print(f"Engine: {detail.display_name} ({detail.name})")
        print(f"  Active:           {detail.active}")
        print(f"  Version:          {detail.version or '-'}")
        print(f"  Pin:              {detail.pin or '-'}")
        print(f"  Install root:     {detail.install_root or '-'}")
        print(f"  Patches:")
        print(f"    Community:      {detail.patch_count_community}")
        print(f"    Engine tier:    {detail.patch_count_engine}")
        print(f"  Supported pins:   {len(detail.supported_pins)}")
        for pin in detail.supported_pins:
            print(f"    - {pin}")
        if detail.notes:
            print("  Notes:")
            for n in detail.notes:
                print(f"    - {n}")
        return 0


__all__ = ["EnginesInfoCommand", "EnginesListCommand"]
