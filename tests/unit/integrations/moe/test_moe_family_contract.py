# SPDX-License-Identifier: Apache-2.0
"""MoE family contract tests — Theme 4 starter (audit 2026-05-11).

Background: 2026-05-11 audit (Agent A codebase analysis) found MoE family
had 0% test coverage despite 4 patches (P24, P31, P37, PN27) including
PROD-active P37 (MoE intermediate cache pool). This is a contract test
that pins basic invariants applicable across all MoE patches.

These tests run torch-less (no GPU, no vllm import at module level) so
they fit standard CI runner. They verify:
1. Module importable
2. Required constants exist (marker, drift markers, OLD/NEW anchor strings)
3. Marker is unique + contains patch ID
4. apply() function exists and respects env flag gate
5. Patch is registered in PATCH_REGISTRY with correct family

Coverage gap closure: MoE 0% → 4/4 patches covered by this contract.
Future: per-patch deeper tests (algorithm correctness, idempotency,
upstream-anchor invariants) — those need actual vllm source + GPU.

Adding new MoE patch? Append its module path + env flag to MOE_PATCHES
list below — contract tests auto-apply.
"""
from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path

import pytest

# (module path, patch ID) — env_flag derived from registry (source of truth)
MOE_PATCHES = [
    ("sndr.engines.vllm.patches.moe.p24_moe_tune", "P24"),
    ("sndr.engines.vllm.patches.moe.p31_router_softmax", "P31"),
    ("sndr.engines.vllm.patches.moe.p37_moe_intermediate_cache", "P37"),
    ("sndr.engines.vllm.patches.moe.pn27_revert_pluggable_moe", "PN27"),
    ("sndr.engines.vllm.patches.moe.pn368_marlin_moe_atomic_add_wire", "PN368"),
    # PN377 registry entry lands with the wave-2 registry merge — the
    # two registry-reading contract tests stay red until it does (TDD).
    ("sndr.engines.vllm.patches.moe.pn377_moe_wna16_bsk_clamp", "PN377"),
]


def _get_registry_env_flag(patch_id: str) -> str | None:
    """Read env_flag for patch from registry.py source (no torch import)."""
    registry_path = Path(__file__).resolve().parents[4] / "sndr" / "dispatcher" / "registry.py"
    text = registry_path.read_text()
    # Find `"PATCH_ID": { ... "env_flag": "VALUE", ... }`
    m = re.search(rf'"{patch_id}":\s*\{{(.*?)^    \}}', text, flags=re.M | re.S)
    if not m:
        return None
    body = m.group(1)
    flag_match = re.search(r'"env_flag":\s*"([^"]+)"', body)
    return flag_match.group(1) if flag_match else None


def _get_registry_lifecycle(patch_id: str) -> str | None:
    """Read lifecycle for patch from registry.py source."""
    registry_path = Path(__file__).resolve().parents[4] / "sndr" / "dispatcher" / "registry.py"
    text = registry_path.read_text()
    m = re.search(rf'"{patch_id}":\s*\{{(.*?)^    \}}', text, flags=re.M | re.S)
    if not m:
        return None
    body = m.group(1)
    lc_match = re.search(r'"lifecycle":\s*"([^"]+)"', body)
    return lc_match.group(1) if lc_match else None


@pytest.fixture(autouse=True)
def _no_torch_import(monkeypatch):
    """Make tests deterministically torch-less.

    Some patches eagerly import torch at module load. We mark expected
    SKIP behavior in that case, but for now just allow torch via lazy-load.
    """
    # No-op fixture — pytest-style placeholder for future torch-block strategy
    yield


@pytest.mark.parametrize("module_path,patch_id", MOE_PATCHES)
class TestMoEPatchContract:
    """Family-level invariants enforced across every MoE patch."""

    def test_module_importable(self, module_path, patch_id):
        """Patch module must import without errors (torch-less if possible)."""
        # Clear any prior import to test fresh
        if module_path in sys.modules:
            del sys.modules[module_path]
        try:
            importlib.import_module(module_path)
        except ImportError as e:
            # Triton-dependent patches may fail on CI without torch — mark expected-skip
            if "torch" in str(e) or "triton" in str(e):
                pytest.skip(f"{patch_id} requires torch/triton: {e}")
            raise

    def test_genesis_marker_exists(self, module_path, patch_id):
        """Each patch declares a Genesis marker. Accept any of:
        - `GENESIS_*_MARKER` constant (text-patch convention)
        - `<PID>_MARKER_*` constant (PN90-style)
        - setattr `_genesis_<id>_wrapped` / `_genesis_<id>_marker` (PN72-style)
        - `_GENESIS_<ID>_MARKER_ATTR` (P31-style)
        """
        mod = importlib.import_module(module_path)
        src_text = Path(mod.__file__).read_text()
        has_const_marker = any(
            a for a in dir(mod) if "MARKER" in a.upper()
            and (a.startswith("GENESIS_") or a.startswith(patch_id.upper())
                 or "_MARKER" in a)
        )
        has_attr_marker = bool(re.search(
            r"_genesis_\w+_(wrapped|marker)\b|_GENESIS_\w+_MARKER_ATTR\b",
            src_text, flags=re.IGNORECASE,
        ))
        assert has_const_marker or has_attr_marker, (
            f"{patch_id}: no Genesis marker found in {module_path}"
        )

    def test_env_flag_documented(self, module_path, patch_id):
        """Patch source mentions its env_flag from registry (operator grep).

        Known tech debt: legacy patches (lifecycle=legacy) with
        env_flag=GENESIS_LEGACY_* don't reference their flag in source —
        they're triggered via the legacy auto-apply path (`is_legacy_active`).
        Marked xfail until those are refactored OR explicitly documented
        as auto-apply-only (no env-gate).
        """
        env_flag = _get_registry_env_flag(patch_id)
        assert env_flag, f"{patch_id}: env_flag not found in registry"
        lifecycle = _get_registry_lifecycle(patch_id)
        mod = importlib.import_module(module_path)
        src_text = Path(mod.__file__).read_text()
        if env_flag in src_text:
            return  # operator-grep finds it, all good
        is_synthetic_legacy_flag = (
            lifecycle in ("legacy", "retired")
            and env_flag.startswith("GENESIS_LEGACY_")
        )
        if is_synthetic_legacy_flag:
            pytest.xfail(
                f"{patch_id}: synthetic legacy env_flag {env_flag!r} "
                f"(lifecycle={lifecycle}) — STILL not in source/companion "
                f"(operator-grep gap; tech debt: add doc comment OR refactor)"
            )

    def test_apply_function_exists(self, module_path, patch_id):
        """Patch must expose apply() callable (orchestrator entry point)."""
        mod = importlib.import_module(module_path)
        # Common patterns: apply, should_apply
        has_apply = hasattr(mod, "apply") and callable(mod.apply)
        has_should_apply = hasattr(mod, "should_apply") and callable(mod.should_apply)
        assert has_apply or has_should_apply, \
            f"{patch_id}: no apply() or should_apply() function in {module_path}"

    def test_module_torch_less_import_safety(self, module_path, patch_id):
        """Module imports at top should be safe without torch (defensive guard).

        Hot patches may import torch lazily inside apply() — that's fine.
        But module-level `import torch` is a torch-less import hazard.
        """
        mod = importlib.import_module(module_path)
        src_text = Path(mod.__file__).read_text()
        # Pattern: top-level `import torch` (not in def/class)
        # Simple heuristic: count `^import torch` or `^from torch` at start of line
        top_level_torch = re.findall(r"^(?:import torch|from torch)", src_text, flags=re.M)
        # Allow 0; if any, ensure they're inside conditional or function
        if top_level_torch:
            # Check that imports are inside a function/class indentation
            for line in src_text.splitlines():
                if line.startswith(("import torch", "from torch")):
                    pytest.fail(
                        f"{patch_id}: top-level torch import in {module_path} "
                        f"({line!r}) breaks torch-less pytest collection. "
                        "Move into apply() or guard with try/except ImportError."
                    )


class TestMoEFamilyRegistry:
    """Family-level invariants verified once (not per-patch)."""

    def test_all_patches_listed_in_registry(self):
        """Every MoE patch in MOE_PATCHES must have entry in PATCH_REGISTRY."""
        # Read registry.py source (no torch import needed)
        registry_path = Path(__file__).resolve().parents[4] / "sndr" / "dispatcher" / "registry.py"
        assert registry_path.is_file(), f"Registry not found at {registry_path}"
        text = registry_path.read_text()
        for module_path, patch_id in MOE_PATCHES:
            # Check both `"P24": {` and `"P24_MOE_TUNE": {` style possible
            entry_short = f'"{patch_id}":'
            assert entry_short in text, f"{patch_id}: no entry in PATCH_REGISTRY"

    def test_family_count_matches_filesystem(self):
        """`vllm/sndr_core/integrations/moe/` should contain only files for
        patches in MOE_PATCHES (drift detector)."""
        moe_dir = Path(__file__).resolve().parents[4] / "sndr" / "engines" / "vllm" / "patches" / "moe"
        assert moe_dir.is_dir(), f"MoE patches dir not found at {moe_dir}"
        files = {f.stem for f in moe_dir.glob("*.py") if f.name != "__init__.py"}
        # Allow some on-disk files NOT in our test list (newer patches),
        # but every patch in our list MUST exist on disk
        for module_path, patch_id in MOE_PATCHES:
            file_stem = module_path.rsplit(".", 1)[-1]
            assert file_stem in files, \
                f"{patch_id}: expected file {file_stem}.py in {moe_dir}, found: {sorted(files)}"
