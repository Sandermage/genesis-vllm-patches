# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/build_anchor_manifest.py` — anchor manifest builder.

Contract:

  1. _detect_genesis_pin returns the version string or "unknown".
  2. _detect_vllm_pin_from_fixture parses the README format reliably.
  3. _derive_rel_path handles three strategies:
       a) pristine fixture path → known map lookup
       b) site-packages/vllm/... path → strip prefix
       c) basename → known map lookup
  4. _KNOWN_REL_PATHS maps fixture basenames to canonical rel paths.
  5. _REGISTRY_TARGETS includes the documented stable patches.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_anchor_manifest.py"


def _import_script():
    name = "_build_anchor_manifest_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── Genesis pin detection ────────────────────────────────────────────


class TestGenesisPinDetection:
    def test_returns_string(self):
        mod = _import_script()
        pin = mod._detect_genesis_pin()
        assert isinstance(pin, str)
        # Either the live version or "unknown"
        assert len(pin) > 0


# ─── vllm pin detection from README ──────────────────────────────────


class TestVllmPinFromFixture:
    def test_returns_string(self):
        mod = _import_script()
        pin = mod._detect_vllm_pin_from_fixture()
        assert isinstance(pin, str)

    def test_unknown_when_readme_missing(self, monkeypatch, tmp_path):
        mod = _import_script()
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        # tmp_path/tests/legacy/pristine_fixtures/README.md doesn't exist
        assert mod._detect_vllm_pin_from_fixture() == "unknown"

    def test_parses_backtick_quoted_pin(self, monkeypatch, tmp_path):
        mod = _import_script()
        # Create fake pristine README with vllm: `<pin>` line
        readme_dir = tmp_path / "tests" / "legacy" / "pristine_fixtures"
        readme_dir.mkdir(parents=True)
        readme = readme_dir / "README.md"
        readme.write_text(
            "# Pristine fixtures\n\n"
            "vllm: `0.20.2rc1.dev371+gbf610c2f5`\n"
        )
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        assert mod._detect_vllm_pin_from_fixture() == "0.20.2rc1.dev371+gbf610c2f5"

    def test_parses_bare_value(self, monkeypatch, tmp_path):
        mod = _import_script()
        readme_dir = tmp_path / "tests" / "legacy" / "pristine_fixtures"
        readme_dir.mkdir(parents=True)
        readme = readme_dir / "README.md"
        # No backticks — just `vllm: <pin>` form
        readme.write_text("vllm: 1.0.0-alpha\n")
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        assert mod._detect_vllm_pin_from_fixture() == "1.0.0-alpha"


# ─── _derive_rel_path strategies ──────────────────────────────────────


class TestDeriveRelPath:
    def test_strategy_1_pristine_fixture(self, tmp_path):
        """Target inside pristine_root → basename lookup in _KNOWN_REL_PATHS."""
        mod = _import_script()
        pristine = tmp_path / "pristine"
        pristine.mkdir()
        target = pristine / "chunk.py"
        result = mod._derive_rel_path(target, pristine)
        assert result == "model_executor/layers/fla/ops/chunk.py"

    def test_strategy_2_vllm_install_path(self, tmp_path):
        """Target with `vllm/` segment → strip prefix up to last `vllm`."""
        mod = _import_script()
        target = Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/"
            "model_executor/layers/fla/ops/chunk.py"
        )
        pristine = tmp_path / "pristine"
        pristine.mkdir()
        result = mod._derive_rel_path(target, pristine)
        assert result == "model_executor/layers/fla/ops/chunk.py"

    def test_strategy_3_basename_fallback(self, tmp_path):
        """Unknown path with known basename → _KNOWN_REL_PATHS lookup."""
        mod = _import_script()
        target = Path("/some/random/path/chunk.py")
        pristine = tmp_path / "pristine"
        pristine.mkdir()
        result = mod._derive_rel_path(target, pristine)
        # /some/random/path doesn't contain `vllm`, basename "chunk.py"
        # matches _KNOWN_REL_PATHS
        assert result == "model_executor/layers/fla/ops/chunk.py"

    def test_unknown_basename_returns_none(self, tmp_path):
        mod = _import_script()
        target = Path("/no/such/file.py")
        pristine = tmp_path / "pristine"
        pristine.mkdir()
        result = mod._derive_rel_path(target, pristine)
        assert result is None


# ─── _KNOWN_REL_PATHS map ─────────────────────────────────────────────


class TestKnownRelPaths:
    def test_all_known_paths_are_strings(self):
        mod = _import_script()
        for basename, rel_path in mod._KNOWN_REL_PATHS.items():
            assert isinstance(basename, str)
            assert isinstance(rel_path, str)

    def test_rel_paths_end_with_basename(self):
        """Each rel path should end with the same basename used as key."""
        mod = _import_script()
        for basename, rel_path in mod._KNOWN_REL_PATHS.items():
            assert rel_path.endswith("/" + basename), (
                f"{basename} → {rel_path} (mismatch)"
            )

    def test_stable_patches_covered(self):
        mod = _import_script()
        # Documented stable patches need fixture coverage
        for required in ("chunk.py", "gdn_linear_attn.py",
                         "gpu_model_runner.py", "gemma4.py"):
            assert required in mod._KNOWN_REL_PATHS


# ─── Default manifest output path ────────────────────────────────────


class TestDefaultManifestOutput:
    def test_returns_path_object(self):
        mod = _import_script()
        result = mod._default_manifest_output()
        assert isinstance(result, Path)

    def test_canonical_path_components(self):
        """Path ends with /sndr/manifests/anchor_manifest.json (v12
        layout) whether resolved via project_paths or bootstrap
        fallback."""
        mod = _import_script()
        result = mod._default_manifest_output()
        parts = result.parts
        assert "anchor_manifest.json" == parts[-1]
        assert "manifests" == parts[-2]
        assert "sndr" == parts[-3]
