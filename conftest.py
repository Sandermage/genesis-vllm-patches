# SPDX-License-Identifier: Apache-2.0
"""Root-level pytest conftest — ensures repo root is on sys.path FIRST.

Audit P1 fix 2026-05-05 (genesis_local_consistency_audit_2026-05-05.md):
The deeper conftest at `vllm/_genesis/tests/conftest.py` imports from
`vllm._genesis.__version__` at module level, which requires `vllm` to be
importable. When pytest is invoked from the repo root WITHOUT
`PYTHONPATH=.` set, it would fail at conftest-import time because the
deeper conftest runs before pytest's `pythonpath = .` ini option takes
effect for nested namespace packages.

This root-level conftest forces sys.path[0] = repo root at the earliest
possible point, so any nested conftest can `import vllm._genesis.*`
without operator pre-config.

Empty body otherwise — pure side-effect of the import-side path mutation
above. Single-purpose; do not add fixtures here.
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# Belt-and-suspenders: also mutate PYTHONPATH for any subprocess pytest spawns.
_existing_pp = os.environ.get("PYTHONPATH", "")
if _REPO_ROOT not in _existing_pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = (
        _REPO_ROOT + os.pathsep + _existing_pp if _existing_pp else _REPO_ROOT
    )
