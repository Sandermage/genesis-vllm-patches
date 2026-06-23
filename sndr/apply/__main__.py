# SPDX-License-Identifier: Apache-2.0
"""SNDR Core apply — `python -m sndr.apply` entry point.

v10 (2026-05-07): F-019 fix. The model_config launch script renderer
prefers `python -m sndr.apply` over the legacy
`python -m sndr.apply_all` form. Both still work
(legacy is a forward-shim). Adding this module enables direct package
execution since `apply` is a package, not a module.
"""
from __future__ import annotations

import sys

from sndr.apply import main


if __name__ == "__main__":
    sys.exit(main())
