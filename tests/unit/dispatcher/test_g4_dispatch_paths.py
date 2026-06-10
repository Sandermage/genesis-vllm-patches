# SPDX-License-Identifier: Apache-2.0
"""Regression test: every `_G4_PATCHES` tuple's dispatch import path
must resolve to a real module.

Background — what failure this gate prevents:

The boot-time Gemma 4 dispatcher (`sndr.apply._per_patch_dispatch`)
iterates a tuple table `_G4_PATCHES` and constructs each patch's wiring
import path as:

    f"sndr.engines.vllm.patches.{family_pkg}.{module_attr}"

`family_pkg` is the 4th element of each tuple, defaulting to `"gemma4"`
when the tuple is 3-element.

When Phase 2.x of the production cleanup workstream relocates a G4
module file from `integrations/gemma4/` to a new package (e.g.,
`integrations/model_compat/gemma4/`), three places need to be kept
in sync:

  1. The actual file path on disk (`git mv` handles).
  2. `dispatcher/registry.py`'s `apply_module` field for that patch.
  3. The `_G4_PATCHES` tuple's `family_pkg` element.

The family-contract tests catch drift between (1) and (2) because
they import via `apply_module`. But until 2026-05-22 there was NO
test that caught drift between (1)/(2) and (3) — the dispatcher
tuple table — because no unit test exercises the boot-time import
loop. The result: a Phase 2.2 relocation (commit `66ab670b`)
silently left the 18 relocated patches with default
`family_pkg="gemma4"` for THREE commits until Phase 2.4 G-STRUCT-K4
smoke produced 72 `Genesis FAILED: G4_NN ... No module named
'sndr.engines.vllm.patches.gemma4.g4_NN_*'` at container boot.

This test closes that gap. It parses the live `_G4_PATCHES` tuple
table, computes the same import path the dispatcher will build at
boot, and imports each — locally, without CUDA / vLLM runtime.
Any future relocation that updates registry but forgets the tuple
will fail this test before it lands.

Invariant: for every tuple in `_per_patch_dispatch._G4_PATCHES`,
`importlib.import_module(<dispatch_path>)` must succeed (modulo
runtime-only deps like torch/triton, which we explicitly skip on).
"""
from __future__ import annotations

import importlib

import pytest


def _g4_patches():
    """Live tuple table from the boot-time dispatcher source."""
    from sndr.apply import _per_patch_dispatch as pd
    return pd._G4_PATCHES


def _dispatch_import_path(entry: tuple) -> str:
    """Mirror the path-building logic in `_g4_dispatch_factory`."""
    if len(entry) == 4:
        _id, _title, module_attr, family_pkg = entry
    else:
        _id, _title, module_attr = entry
        family_pkg = "gemma4"
    family_dotted = family_pkg.replace("/", ".")
    return f"sndr.engines.vllm.patches.{family_dotted}.{module_attr}"


@pytest.mark.parametrize("entry", _g4_patches())
def test_g4_dispatch_path_resolves(entry):
    """Every G4 tuple's family_pkg + module_attr resolves to an
    importable module — the same import the boot-time dispatcher does.

    Test SKIPS if the import fails only because torch / triton is
    not available in this pure-Python test environment (that's a
    runtime test, not a path-resolution test). Test FAILS if the
    module simply cannot be found — that's the registry/dispatcher
    drift this gate guards against.
    """
    full_path = _dispatch_import_path(entry)
    try:
        importlib.import_module(full_path)
    except ImportError as e:
        msg = str(e)
        # Runtime dep gates — not what this test is for.
        if "torch" in msg or "triton" in msg or "vllm.v1" in msg:
            pytest.skip(f"{full_path}: runtime dep missing ({e!r})")
        # Genuine module-not-found is the drift signal.
        raise AssertionError(
            f"dispatch import resolution failed for tuple {entry[0]}: "
            f"`{full_path}` — module not found. Likely registry-vs-"
            f"_G4_PATCHES drift after a relocation. Update the tuple "
            f"in `vllm/sndr_core/apply/_per_patch_dispatch.py` to add "
            f"the correct `family_pkg` (e.g., 'model_compat.gemma4', "
            f"'attention.turboquant', 'spec_decode', 'kv_cache'). "
            f"Original error: {e}"
        ) from e


def test_g4_patch_table_is_non_empty():
    """Sanity check that the table is loaded (catches accidental empty)."""
    assert len(_g4_patches()) >= 18, (
        "_G4_PATCHES table unexpectedly small — did a refactor delete entries?"
    )
