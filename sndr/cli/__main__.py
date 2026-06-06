# SPDX-License-Identifier: Apache-2.0
"""``python -m sndr.cli`` entry point.

The CLI implementation currently lives under ``sndr.cli.legacy`` during the
v12.x migration; this module makes the modern package path runnable so both
``python -m sndr.cli`` and the ``sndr`` console script reach the same
dispatcher.
"""
import sys

from sndr.cli.legacy import cli_main

if __name__ == "__main__":
    sys.exit(cli_main())
