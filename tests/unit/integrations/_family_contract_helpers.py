# SPDX-License-Identifier: Apache-2.0
"""Reusable invariants for family-contract test files.

Theme 4 helper module (audit 2026-05-11): extracted from the 6 hand-
written family contracts (MoE, quantization, scheduler, spec_decode,
worker, memory, reasoning) after the pattern stabilized. Each new
family contract file is now ~40 lines instead of ~200 — just declares
the patches list + family name, and calls `make_family_contract_class()`.

Single source of truth for invariant logic — refining the marker regex
or the env-flag check propagates to all 11+ family contracts at once.

Usage (minimal new family contract file):

    from tests.unit.integrations._family_contract_helpers import (
        make_family_contract_class, make_family_registry_class,
    )

    PATCHES = [
        ("sndr.engines.vllm.patches.<fam>.<file>", "PATCH_ID"),
        ...
    ]

    class TestMyFamilyPatchContract(
        make_family_contract_class("my_family", PATCHES)
    ):
        pass

    class TestMyFamilyFamilyRegistry(
        make_family_registry_class("my_family", PATCHES,
                                   filesystem_dir="my_family")
    ):
        pass

Invariants enforced per patch:
  1. Module importable (skip on torch/triton missing)
  2. Genesis marker exists (accepts 4 conventions; skip for
     coordinator/retired/scaffold/placeholder lifecycles)
  3. apply() or should_apply() callable
  4. env_flag from registry referenced in source OR companion files
     (xfail synthetic GENESIS_LEGACY_*; skip retired no-op)
  5. No top-level torch import (torch-less collection safety)
  6. Family field matches expected in registry

Family-level (one-shot):
  - All PATCHES entries have registry entries
  - All filesystem files match PATCHES (drift detector)
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest


# ─── Registry parsing helpers ─────────────────────────────────────────


def _registry_path() -> Path:
    """Resolve repo's PATCH_REGISTRY source path from this helpers module."""
    # parents: [tests/unit/integrations, tests/unit, tests, repo_root]
    # v12.x moved the registry to sndr/dispatcher/registry.py; the old
    # vllm/sndr_core path is now a re-export shim with no literal to parse.
    repo_root = Path(__file__).resolve().parents[3]
    canonical = repo_root / "sndr" / "dispatcher" / "registry.py"
    if canonical.is_file():
        return canonical
    return repo_root / "sndr" / "dispatcher" / "registry.py"


def get_registry_field(patch_id: str, field: str) -> str | None:
    """Read string field for patch_id from registry.py source (no torch)."""
    text = _registry_path().read_text()
    m = re.search(rf'"{patch_id}":\s*\{{(.*?)^    \}}', text, flags=re.M | re.S)
    if not m:
        return None
    body = m.group(1)
    f_match = re.search(rf'"{field}":\s*"([^"]+)"', body)
    return f_match.group(1) if f_match else None


# ─── Class factory for per-patch contract tests ───────────────────────


def make_family_contract_class(family_name: str, patches: list[tuple[str, str]]):
    """Build a parameterized test class enforcing the 6 invariants.

    Caller subclasses the returned class to add family-specific tests
    (or just `pass` for the bare contract).
    """

    @pytest.mark.parametrize("module_path,patch_id", patches)
    class _FamilyPatchContract:
        """Family-level invariants enforced across every patch."""

        def test_module_importable(self, module_path, patch_id):
            """Module imports without errors (torch-less if possible)."""
            if module_path in sys.modules:
                del sys.modules[module_path]
            try:
                importlib.import_module(module_path)
            except ImportError as e:
                if "torch" in str(e) or "triton" in str(e):
                    pytest.skip(f"{patch_id} requires torch/triton: {e}")
                raise

        def test_genesis_marker_exists(self, module_path, patch_id):
            """Marker required for text-patches (TextPatcher idempotency).
            Skip for:
              - coordinator/retired/scaffold/placeholder lifecycle (no-op)
              - pure runtime hooks (no TextPatch/TextPatcher import — they
                don't text-patch source, so anchor-marker convention N/A)
            """
            lifecycle = get_registry_field(patch_id, "lifecycle")
            impl_status = get_registry_field(patch_id, "implementation_status")
            if lifecycle in ("coordinator", "retired") or impl_status in (
                "retired", "scaffold", "placeholder"
            ):
                pytest.skip(
                    f"{patch_id}: lifecycle={lifecycle} impl={impl_status} "
                    f"— marker not required"
                )
            mod = importlib.import_module(module_path)
            src_text = Path(mod.__file__).read_text()
            # Skip marker check for pure runtime-hook patches that don't
            # text-patch source (no TextPatcher → no anchor-marker need).
            uses_textpatcher = bool(re.search(
                r"\bfrom vllm\.sndr_core\.core\b.*TextPatch|"
                r"\bimport TextPatch\b|"
                r"\bTextPatcher\b|\bTextPatch\(",
                src_text,
            ))
            if not uses_textpatcher:
                pytest.skip(
                    f"{patch_id}: pure runtime hook (no TextPatcher) — "
                    f"anchor-marker convention not applicable"
                )
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
            has_should = (
                hasattr(mod, "should_apply") and callable(mod.should_apply)
            )
            assert has_apply or has_should, (
                f"{patch_id}: no apply() or should_apply() in {module_path}"
            )

        def test_env_flag_documented(self, module_path, patch_id):
            """env_flag from registry referenced in wiring file or
            companion (kernels/, etc). Skip retired no-op. For
            synthetic legacy env_flags: check source first — if
            operator-grep-discoverable (audit 2026-05-11 doc fix
            adds these as documenting comments), test passes. Only
            xfail if STILL missing (operator can't grep)."""
            env_flag = get_registry_field(patch_id, "env_flag")
            assert env_flag, f"{patch_id}: env_flag not found in registry"
            lifecycle = get_registry_field(patch_id, "lifecycle")
            impl_status = get_registry_field(patch_id, "implementation_status")
            if lifecycle == "retired" and impl_status == "retired":
                pytest.skip(
                    f"{patch_id}: retired no-op — env_flag doc not required"
                )
            # Check source/companion first — pass if grep finds the flag
            mod = importlib.import_module(module_path)
            src_text = Path(mod.__file__).read_text()
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
            # Source/companion lack the flag — for synthetic legacy
            # GENESIS_LEGACY_*, mark as documented tech debt (xfail).
            # For non-synthetic flags, this is a real failure.
            is_synthetic = (
                lifecycle in ("legacy", "retired")
                and env_flag.startswith("GENESIS_LEGACY_")
            )
            if is_synthetic:
                pytest.xfail(
                    f"{patch_id}: synthetic legacy env_flag {env_flag!r} "
                    f"(lifecycle={lifecycle}) — STILL not in source/companion "
                    f"(operator-grep gap; tech debt: add doc comment OR "
                    f"refactor to explicit env-gate)"
                )
            pytest.fail(
                f"{patch_id}: env_flag {env_flag!r} not found in wiring "
                f"or companion files matching '*{patch_stem}*.py'"
            )

        def test_module_torch_less_import_safety(self, module_path, patch_id):
            """Module-level torch imports break torch-less collection."""
            mod = importlib.import_module(module_path)
            src_text = Path(mod.__file__).read_text()
            top_level = re.findall(
                r"^(?:import torch|from torch)", src_text, flags=re.M
            )
            if top_level:
                for line in src_text.splitlines():
                    if line.startswith(("import torch", "from torch")):
                        pytest.fail(
                            f"{patch_id}: top-level torch import "
                            f"in {module_path}"
                        )

        def test_family_matches_registry(self, module_path, patch_id):
            """Registry family field matches expected for this contract."""
            family = get_registry_field(patch_id, "family")
            assert family == family_name, (
                f"{patch_id}: registry family={family!r}, "
                f"expected {family_name!r}"
            )

    _FamilyPatchContract.__name__ = (
        f"_{family_name.replace('.', '_').title()}PatchContractBase"
    )
    return _FamilyPatchContract


# ─── Class factory for family-level (one-shot) registry tests ─────────


def make_family_registry_class(
    family_name: str,
    patches: list[tuple[str, str]],
    filesystem_dir: str | None = None,
):
    """Build a registry-level test class verifying all patches are in
    registry + filesystem files match (drift detector).

    Args:
        family_name: matches registry `family` field (e.g. "memory")
        patches: list of (module_path, patch_id) tuples
        filesystem_dir: directory under integrations/ where files live
            (default: family_name; override when nested like "attention/gdn")
    """
    fs_dir = filesystem_dir or family_name

    class _FamilyRegistryContract:
        """Family-level invariants verified once (not per-patch)."""

        def test_all_patches_listed_in_registry(self):
            text = _registry_path().read_text()
            for _module_path, patch_id in patches:
                assert f'"{patch_id}":' in text, (
                    f"{patch_id}: no entry in PATCH_REGISTRY"
                )

        def test_family_count_matches_filesystem(self):
            from sndr.dispatcher import PATCH_REGISTRY
            # v12.x: patch modules moved to sndr/engines/vllm/patches/; the
            # registry sits at sndr/dispatcher/registry.py so parent.parent
            # is the sndr/ package root. Retired patches live under
            # patches/_retired/ or the engine's _archive/.
            sndr_root = _registry_path().parent.parent
            patches_root = sndr_root / "engines" / "vllm" / "patches"
            if not patches_root.is_dir():
                patches_root = sndr_root / "integrations"  # pre-v12.x layout
            fam_dir = patches_root / fs_dir
            retired_dirs = [
                patches_root / "_retired",
                patches_root.parent / "_archive",
            ]
            files = {
                f.stem for f in fam_dir.glob("*.py")
                if f.name != "__init__.py"
            }
            retired_files: set[str] = set()
            for rd in retired_dirs:
                if rd.exists():
                    retired_files |= {
                        f.stem for f in rd.glob("*.py")
                        if f.name != "__init__.py"
                    }
            for module_path, patch_id in patches:
                file_stem = module_path.rsplit(".", 1)[-1]
                meta = PATCH_REGISTRY.get(patch_id, {})
                # Retired patches live in `_retired/` by policy — OR may
                # have been fully deleted (apply_module=None means the
                # patch is a registry-only audit-trail entry; e.g. PN34
                # deleted as duplicate of SNDR_WORKSPACE_001).
                if meta.get("lifecycle") == "retired":
                    if not meta.get("apply_module"):
                        # No wiring expected — registry-only entry.
                        continue
                    assert file_stem in retired_files, (
                        f"{patch_id}: retired but file {file_stem}.py "
                        f"not in {retired_dir}"
                    )
                    continue
                assert file_stem in files, (
                    f"{patch_id}: expected file {file_stem}.py in {fam_dir}"
                )

    _FamilyRegistryContract.__name__ = (
        f"_{family_name.replace('.', '_').title()}RegistryContractBase"
    )
    return _FamilyRegistryContract
