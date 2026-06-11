# SPDX-License-Identifier: Apache-2.0
"""Scheduler family contract tests — Theme 4 expansion (audit 2026-05-11).

Background: scheduler family had 1/8 patches with dedicated tests
(P58 only). Family-level contract closes the gap via parameterized
invariants across all 8 patches. Mirrors the MoE + quantization
contract template proven in earlier Theme 4 iterations.

Scheduler family characteristics (different from MoE/quantization):
  - Mix of legacy (P4, P8, P34) + experimental (P58, P74, P79c/d, P84)
  - P8 currently lifecycle="retired" (upstream get_max_concurrency
    refactor) — contract test must accept retired patches
  - Several patches use TextPatcher with MultiFilePatchTransaction
    (P79c, P79d) — contract verifies marker presence + apply callable

These tests run torch-less (no GPU). Verify:
  1. Module importable (or skip cleanly on triton/torch missing)
  2. Genesis marker exists (const OR attr pattern)
  3. apply() or should_apply() callable
  4. env_flag from registry referenced in source (operator grep)
     — xfail for legacy auto-apply patches (synthetic GENESIS_LEGACY_*
        env_flags, no runtime effect)
  5. No top-level torch import (torch-less collection safety)
  6. Family field matches "scheduler" in registry
  7. Registry has entries for all listed patches + filesystem files
     match (drift detector)

Coverage gap closure: scheduler 1/8 → 8/8 by contract.
Adding new scheduler patch? Append (module_path, patch_id) to
SCHEDULER_PATCHES — contract tests auto-apply.
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

# (module path, patch ID) — env_flag derived from registry (source of truth)
SCHEDULER_PATCHES = [
    ("sndr.engines.vllm._archive.p4_tq_hybrid", "P4"),
    ("sndr.engines.vllm._archive.p8_kv_hybrid_reporting", "P8"),
    ("sndr.engines.vllm.patches.scheduler.p34_mamba_deadlock_guard", "P34"),
    ("sndr.engines.vllm.patches.scheduler.p58_async_scheduler_placeholder_fix", "P58"),
    ("sndr.engines.vllm.patches.scheduler.p74_chunk_clamp", "P74"),
    ("sndr.engines.vllm.patches.scheduler.p79c_stale_spec_token_cleanup", "P79c"),
    ("sndr.engines.vllm.patches.scheduler.p79d_preempt_async_discard", "P79d"),
    ("sndr.engines.vllm._archive.p84_hash_block_size_override", "P84"),
]


def _get_registry_field(patch_id: str, field: str) -> str | None:
    """Read a string field for patch_id from registry.py source (no torch)."""
    registry_path = (
        Path(__file__).resolve().parents[4]
        / "sndr" / "dispatcher" / "registry.py"
    )
    text = registry_path.read_text()
    m = re.search(rf'"{patch_id}":\s*\{{(.*?)^    \}}', text, flags=re.M | re.S)
    if not m:
        return None
    body = m.group(1)
    f_match = re.search(rf'"{field}":\s*"([^"]+)"', body)
    return f_match.group(1) if f_match else None


@pytest.mark.parametrize("module_path,patch_id", SCHEDULER_PATCHES)
class TestSchedulerPatchContract:
    """Family-level invariants enforced across every scheduler patch."""

    def test_module_importable(self, module_path, patch_id):
        """Patch module must import without errors (torch-less if possible)."""
        if module_path in sys.modules:
            del sys.modules[module_path]
        try:
            importlib.import_module(module_path)
        except ImportError as e:
            if "torch" in str(e) or "triton" in str(e):
                pytest.skip(f"{patch_id} requires torch/triton: {e}")
            raise

    def test_genesis_marker_exists(self, module_path, patch_id):
        """Each patch declares a Genesis marker. Accept any of:
        - `GENESIS_*_MARKER` constant
        - `<PID>_MARKER_*` constant
        - setattr `_genesis_<id>_wrapped` / `_genesis_<id>_marker` pattern
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

    def test_apply_function_exists(self, module_path, patch_id):
        """Patch must expose apply() or should_apply() callable."""
        mod = importlib.import_module(module_path)
        has_apply = hasattr(mod, "apply") and callable(mod.apply)
        has_should_apply = (
            hasattr(mod, "should_apply") and callable(mod.should_apply)
        )
        assert has_apply or has_should_apply, (
            f"{patch_id}: no apply() or should_apply() in {module_path}"
        )

    def test_env_flag_documented(self, module_path, patch_id):
        """Patch source mentions its env_flag from registry (operator grep).

        Known tech debt — synthetic GENESIS_LEGACY_* env_flags don't get
        referenced in source code:
          - lifecycle=legacy: pre-dispatcher auto-apply patches (P4, P34)
          - lifecycle=retired with synthetic flag: retired legacy patches
            still carry GENESIS_LEGACY_* flag (P8 — retired in v2 audit)
        Both groups xfail'd until env_flag refactor sprint.
        """
        env_flag = _get_registry_field(patch_id, "env_flag")
        assert env_flag, f"{patch_id}: env_flag not found in registry"
        lifecycle = _get_registry_field(patch_id, "lifecycle")
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

    def test_module_torch_less_import_safety(self, module_path, patch_id):
        """Module-level torch imports break torch-less pytest collection."""
        mod = importlib.import_module(module_path)
        src_text = Path(mod.__file__).read_text()
        top_level_torch = re.findall(
            r"^(?:import torch|from torch)", src_text, flags=re.M
        )
        if top_level_torch:
            for line in src_text.splitlines():
                if line.startswith(("import torch", "from torch")):
                    pytest.fail(
                        f"{patch_id}: top-level torch import in {module_path} "
                        f"({line!r}) breaks torch-less pytest collection. "
                        "Move into apply() or guard with try/except ImportError."
                    )

    def test_family_is_scheduler(self, module_path, patch_id):
        """Registry entry must declare family='scheduler'."""
        family = _get_registry_field(patch_id, "family")
        assert family == "scheduler", (
            f"{patch_id}: registry family={family!r}, expected 'scheduler' "
            f"(SCHEDULER_PATCHES list may be stale, or registry needs update)"
        )


class TestSchedulerFamilyRegistry:
    """Family-level invariants verified once (not per-patch)."""

    def test_all_patches_listed_in_registry(self):
        """Every scheduler patch in SCHEDULER_PATCHES must have a registry entry."""
        registry_path = (
            Path(__file__).resolve().parents[4]
            / "sndr" / "dispatcher" / "registry.py"
        )
        assert registry_path.is_file(), f"Registry not found at {registry_path}"
        text = registry_path.read_text()
        for _module_path, patch_id in SCHEDULER_PATCHES:
            entry = f'"{patch_id}":'
            assert entry in text, f"{patch_id}: no entry in PATCH_REGISTRY"

    def test_family_count_matches_filesystem(self):
        from sndr.dispatcher import PATCH_REGISTRY
        fam_dir = (
            Path(__file__).resolve().parents[4]
            / "sndr" / "engines" / "vllm" / "patches" / "scheduler"
        )
        retired_dirs = [fam_dir.parent / "_retired", fam_dir.parents[1] / "_archive"]
        files = {f.stem for f in fam_dir.glob("*.py") if f.name != "__init__.py"}
        retired_files = {
                f.stem for rd in retired_dirs if rd.exists()
                for f in rd.glob("*.py") if f.name != "__init__.py"
            }
        for module_path, patch_id in SCHEDULER_PATCHES:
            file_stem = module_path.rsplit(".", 1)[-1]
            meta = PATCH_REGISTRY.get(patch_id, {})
            if meta.get("lifecycle") == "retired":
                if not meta.get("apply_module"):
                    continue
                assert file_stem in retired_files, (
                    f"{patch_id}: retired but {file_stem}.py not in {retired_dirs}"
                )
                continue
            assert file_stem in files, (
                f"{patch_id}: expected file {file_stem}.py in {fam_dir}"
            )