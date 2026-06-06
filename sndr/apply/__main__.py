# SPDX-License-Identifier: Apache-2.0
"""SNDR Core apply — `python -m vllm.sndr_core.apply` entry point.

v10 (2026-05-07): F-019 fix. The model_config launch script renderer
prefers `python -m vllm.sndr_core.apply` over the legacy
`python -m vllm.sndr_core.apply_all` form. Both still work
(legacy is a forward-shim). Adding this module enables direct package
execution since `apply` is a package, not a module.
"""
from __future__ import annotations

import sys

from vllm.sndr_core.apply import main


if __name__ == "__main__":
    sys.exit(main())
