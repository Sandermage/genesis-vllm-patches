# SPDX-License-Identifier: Apache-2.0
"""Memory family contract tests — Theme 4 expansion (audit 2026-05-11).

Background: memory family had 2/5 patches with dedicated tests
(P15b/P38b + PN19). Family-level contract covers all 5.

Family characteristics:
  - Mix of active (P15b/P38b/P5b) + retired (PN19/PN78 — upstream
    superseded, formalized in iron-rule-#11 retire batch)
  - PN19 + PN78 have `superseded_by` + `vllm_version_range` upper
    bounds; their wiring auto-skips on dev9+
  - Contract test accepts retired patches gracefully

Tests run torch-less. Verify per-patch:
  1. Module importable (skip on torch/triton)
  2. Genesis marker (any convention)
  3. apply() or should_apply() callable
  4. env_flag referenced in source (xfail synthetic legacy)
  5. No top-level torch import
  6. family field is "memory"
Family-level: registry + filesystem drift detector.

Coverage gap closure: memory 2/5 → 5/5 via contract.
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

MEMORY_PATCHES = [
    # Note: registry uses uppercase suffix (P15B/P38B) while filesystem
    # uses lowercase (p15b/p38b). Dispatcher's apply_module map handles
    # both via case-variant registration in spec.py. Contract test uses
    # registry-canonical casing.
    ("sndr.engines.vllm.patches.memory.p5b_page_size_pad_smaller", "P5b"),
    ("sndr.engines.vllm.patches.memory.p15b_fa_varlen_clamp", "P15B"),
    ("sndr.engines.vllm.patches.memory.p38b_compile_safe_hook", "P38B"),
    ("sndr.engines.vllm._archive.pn19_scoped_max_split", "PN19"),
    ("sndr.engines.vllm._archive.pn78_post_warmup_cache_release", "PN78"),
]


def _get_registry_field(patch_id: str, field: str) -> str | None:
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


@pytest.mark.parametrize("module_path,patch_id", MEMORY_PATCHES)
class TestMemoryPatchContract:
    """Family-level invariants enforced across every memory patch."""

    def test_module_importable(self, module_path, patch_id):
        if module_path in sys.modules:
            del sys.modules[module_path]
        try:
            importlib.import_module(module_path)
        except ImportError as e:
            if "torch" in str(e) or "triton" in str(e):
                pytest.skip(f"{patch_id} requires torch/triton: {e}")
            raise

    def test_genesis_marker_exists(self, module_path, patch_id):
        """Marker required EXCEPT for coordinator/retired/scaffold patches
        — these are no-op wrappers or pointer-files (real wiring lives
        in companion helpers or upstream took over)."""
        lifecycle = _get_registry_field(patch_id, "lifecycle")
        impl_status = _get_registry_field(patch_id, "implementation_status")
        if lifecycle in ("coordinator", "retired") or impl_status in (
            "retired", "scaffold", "placeholder"
        ):
            pytest.skip(
                f"{patch_id}: lifecycle={lifecycle} impl_status={impl_status}"
                " — marker not required (no-op or pointer-only)"
            )
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
        mod = importlib.import_module(module_path)
        has_apply = hasattr(mod, "apply") and callable(mod.apply)
        has_should_apply = (
            hasattr(mod, "should_apply") and callable(mod.should_apply)
        )
        assert has_apply or has_should_apply, (
            f"{patch_id}: no apply() or should_apply() in {module_path}"
        )

    def test_env_flag_documented(self, module_path, patch_id):
        env_flag = _get_registry_field(patch_id, "env_flag")
        assert env_flag, f"{patch_id}: env_flag not found in registry"
        lifecycle = _get_registry_field(patch_id, "lifecycle")
        impl_status = _get_registry_field(patch_id, "implementation_status")
        if lifecycle == "retired" and impl_status == "retired":
            pytest.skip(
                f"{patch_id}: retired no-op — env_flag documentation "
                f"not required (apply() returns 'skipped' unconditionally)"
            )
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
        if env_flag in src_text:
            return
        # Companion file fallback
        sndr_root = Path(mod.__file__).resolve().parents[2]
        patch_stem = patch_id.lower()
        for companion in sndr_root.rglob(f"*{patch_stem}*.py"):
            if companion == Path(mod.__file__):
                continue
            try:
                if env_flag in companion.read_text():
                    return
            except OSError:
                continue
        pytest.fail(
            f"{patch_id}: env_flag {env_flag!r} not found in wiring or "
            f"companion files matching '*{patch_stem}*.py'"
        )

    def test_module_torch_less_import_safety(self, module_path, patch_id):
        mod = importlib.import_module(module_path)
        src_text = Path(mod.__file__).read_text()
        top_level_torch = re.findall(
            r"^(?:import torch|from torch)", src_text, flags=re.M
        )
        if top_level_torch:
            for line in src_text.splitlines():
                if line.startswith(("import torch", "from torch")):
                    pytest.fail(
                        f"{patch_id}: top-level torch import in {module_path}"
                    )

    def test_family_is_memory(self, module_path, patch_id):
        family = _get_registry_field(patch_id, "family")
        assert family == "memory", (
            f"{patch_id}: registry family={family!r}, expected 'memory'"
        )


class TestMemoryFamilyRegistry:
    """Family-level invariants verified once."""

    def test_all_patches_listed_in_registry(self):
        registry_path = (
            Path(__file__).resolve().parents[4]
            / "sndr" / "dispatcher" / "registry.py"
        )
        text = registry_path.read_text()
        for _module_path, patch_id in MEMORY_PATCHES:
            assert f'"{patch_id}":' in text, (
                f"{patch_id}: no entry in PATCH_REGISTRY"
            )

    def test_family_count_matches_filesystem(self):
        from sndr.dispatcher import PATCH_REGISTRY
        fam_dir = (
            Path(__file__).resolve().parents[4]
            / "sndr" / "engines" / "vllm" / "patches" / "memory"
        )
        retired_dirs = [fam_dir.parent / "_retired", fam_dir.parents[1] / "_archive"]
        files = {f.stem for f in fam_dir.glob("*.py") if f.name != "__init__.py"}
        retired_files = {
                f.stem for rd in retired_dirs if rd.exists()
                for f in rd.glob("*.py") if f.name != "__init__.py"
            }
        for module_path, patch_id in MEMORY_PATCHES:
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