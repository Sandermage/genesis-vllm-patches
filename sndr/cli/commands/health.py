# SPDX-License-Identifier: Apache-2.0
"""CLI command: sndr health."""
from __future__ import annotations

import argparse
import json

from sndr.version import __commit__, __version__


class HealthCommand:
    name = "health"
    help = "Show sndr-platform version and basic health info."

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        pass

    def execute(self, args: argparse.Namespace) -> int:
        info = {
            "version": __version__,
            "commit": __commit__,
            "status": "ok",
        }
        if args.output == "json":
            print(json.dumps(info, indent=2))
        else:
            print(f"sndr-platform {__version__} ({__commit__})")
            print(f"status: ok")
        return 0


__all__ = ["HealthCommand"]
