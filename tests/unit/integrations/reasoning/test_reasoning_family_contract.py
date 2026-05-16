# SPDX-License-Identifier: Apache-2.0
"""Reasoning family contract tests — Theme 4 expansion (audit 2026-05-11).

Background: reasoning family had 5/8 patches with dedicated tests
(P12, P59, PN51, PN58, PN66 — PN67 indirectly via PN66 test). Family
contract closes thin coverage for P27, P61, P61b.

Family characteristics:
  - Mix of active + retired (P61 retired internally per P12 v2 supersession;
    PN51 retired per upstream serving-layer fix on dev209)
  - All patches integrate with Qwen3 reasoning_parser / tool-call paths
  - P12 is legacy auto-apply (synthetic GENESIS_LEGACY_P12 env_flag)

Tests run torch-less. Verify per-patch:
  1. Module importable (skip on torch/triton)
  2. Genesis marker (any convention)
  3. apply() or should_apply() callable
  4. env_flag referenced in source (xfail synthetic legacy)
  5. No top-level torch import
  6. family field is "reasoning"
Family-level: registry + filesystem drift detector.

Coverage gap closure: reasoning 5/8 → 8/8 via contract (+thin-coverage
patches now have family-level baseline).
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

REASONING_PATCHES = [
    ("vllm.sndr_core.integrations.reasoning.p12_tool_call_reasoning", "P12"),
    ("vllm.sndr_core.integrations.reasoning.p27_reasoning_before_think", "P27"),
    ("vllm.sndr_core.integrations.reasoning.p59_qwen3_reasoning_tool_call_recovery", "P59"),
    ("vllm.sndr_core.integrations._retired.p61_qwen3_multi_tool_first_occurrence", "P61"),
    ("vllm.sndr_core.integrations.reasoning.p61b_qwen3_streaming_overlap_guard", "P61b"),
    ("vllm.sndr_core.integrations.reasoning.pn51_qwen3_streaming_thinking_disabled", "PN51"),
    ("vllm.sndr_core.integrations.reasoning.pn58_spec_reasoning_boundary", "PN58"),
    ("vllm.sndr_core.integrations.reasoning.pn66_multiturn_think_leak", "PN66"),
]


def _get_registry_field(patch_id: str, field: str) -> str | None:
    registry_path = (
        Path(__file__).resolve().parents[4]
        / "vllm" / "sndr_core" / "dispatcher" / "registry.py"
    )
    text = registry_path.read_text()
    m = re.search(rf'"{patch_id}":\s*\{{(.*?)^    \}}', text, flags=re.M | re.S)
    if not m:
        return None
    body = m.group(1)
    f_match = re.search(rf'"{field}":\s*"([^"]+)"', body)
    return f_match.group(1) if f_match else None


@pytest.mark.parametrize("module_path,patch_id", REASONING_PATCHES)
class TestReasoningPatchContract:
    """Family-level invariants enforced across every reasoning patch."""

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

    def test_family_is_reasoning(self, module_path, patch_id):
        family = _get_registry_field(patch_id, "family")
        assert family == "reasoning", (
            f"{patch_id}: registry family={family!r}, expected 'reasoning'"
        )


class TestReasoningFamilyRegistry:
    """Family-level invariants verified once."""

    def test_all_patches_listed_in_registry(self):
        registry_path = (
            Path(__file__).resolve().parents[4]
            / "vllm" / "sndr_core" / "dispatcher" / "registry.py"
        )
        text = registry_path.read_text()
        for _module_path, patch_id in REASONING_PATCHES:
            assert f'"{patch_id}":' in text, (
                f"{patch_id}: no entry in PATCH_REGISTRY"
            )

    def test_family_count_matches_filesystem(self):
        from vllm.sndr_core.dispatcher import PATCH_REGISTRY
        reasoning_dir = (
            Path(__file__).resolve().parents[4]
            / "vllm" / "sndr_core" / "integrations" / "reasoning"
        )
        retired_dir = reasoning_dir.parent / "_retired"
        files = {f.stem for f in reasoning_dir.glob("*.py") if f.name != "__init__.py"}
        retired_files = (
            {f.stem for f in retired_dir.glob("*.py") if f.name != "__init__.py"}
            if retired_dir.exists() else set()
        )
        for module_path, patch_id in REASONING_PATCHES:
            file_stem = module_path.rsplit(".", 1)[-1]
            meta = PATCH_REGISTRY.get(patch_id, {})
            if meta.get("lifecycle") == "retired":
                if not meta.get("apply_module"):
                    continue
                assert file_stem in retired_files, (
                    f"{patch_id}: retired but {file_stem}.py not in {retired_dir}"
                )
                continue
            assert file_stem in files, (
                f"{patch_id}: expected file {file_stem}.py in {reasoning_dir}"
            )
