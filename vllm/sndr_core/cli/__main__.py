# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility entry point for ``python -m vllm.sndr_core.cli``.

Canonical location: ``sndr.cli.legacy``. A bare ``from ... import *`` does
NOT re-run the target module's ``if __name__ == "__main__"`` block (that
only fires when the target is itself the main module), so this shim has to
dispatch ``cli_main`` explicitly — otherwise the command imports and exits
0 without running anything. Will be removed in v13.0.
"""
import sys

from sndr.cli.legacy import cli_main

if __name__ == "__main__":
    sys.exit(cli_main())
