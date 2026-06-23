# SPDX-License-Identifier: Apache-2.0
"""Guard: the `sndr install` clone/update sanity gate must point at files that
actually exist in the current tree.

Regression for the v12 bug (2026-06-23): `step_clone_or_update` hard-coded the
retired vllm/sndr_core/{__init__,apply/orchestrator,compat/cli}.py paths, so the
canonical-file sanity check fatally aborted ("missing ... — wrong pin?") on every
clean v12 checkout — `sndr install` was unrunnable. If the overlay package is ever
renamed again, this test fails loudly instead of shipping a broken installer.
"""
from __future__ import annotations

from pathlib import Path

from sndr.cli.legacy.install import _REQUIRED_GENESIS_FILES

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_required_genesis_files_is_nonempty():
    assert _REQUIRED_GENESIS_FILES, "the canonical-file sanity gate must check something"


def test_required_genesis_files_exist_in_tree():
    """Every path the install sanity gate requires must resolve in the repo —
    otherwise `sndr install` fatals on a valid checkout."""
    missing = [rel for rel in _REQUIRED_GENESIS_FILES
               if not (REPO_ROOT / rel).is_file()]
    assert not missing, (
        f"install sanity gate requires files that do not exist: {missing} "
        "— update _REQUIRED_GENESIS_FILES to the current package layout"
    )


def test_required_genesis_files_are_not_retired_namespace():
    """Catch a regression back to the retired v11 vllm/sndr_core/ namespace."""
    stale = [rel for rel in _REQUIRED_GENESIS_FILES if rel.startswith("vllm/sndr_core/")]
    assert not stale, (
        f"these reference the retired v11 namespace: {stale} — the package is `sndr/` in v12"
    )
