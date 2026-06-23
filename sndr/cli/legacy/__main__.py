# SPDX-License-Identifier: Apache-2.0
"""`python -m sndr.cli.legacy` entry point."""
import sys

from . import cli_main

if __name__ == "__main__":
    sys.exit(cli_main())
