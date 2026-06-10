# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/generate_sbom.py` — SBOM emitter.

Contract:

  1. _file_sha256 returns a 64-char hex string OR '' on read error.
  2. _read_pyproject returns dict (or {} on failure).
  3. _read_constraints returns list of stripped non-comment lines.
  4. _list_genesis_modules emits (path, sha256) for every sndr/*.py.
  5. _registry_snapshot returns total / by_tier / by_lifecycle / by_default_on.
  6. _model_configs_snapshot returns list of dicts.
  7. _vllm_pins returns list of pin strings.
  8. _installed_distributions returns sorted list with name/version/summary.
  9. emit_cyclonedx writes CycloneDX 1.5 JSON.
  10. emit_spdx writes SPDX 2.3 JSON.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_sbom.py"


def _import_script():
    name = "_generate_sbom_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── File SHA hashing ─────────────────────────────────────────────────


class TestFileSha256:
    def test_returns_64_hex_chars(self, tmp_path):
        mod = _import_script()
        f = tmp_path / "f.txt"
        f.write_text("hello")
        sha = mod._file_sha256(f)
        assert len(sha) == 64
        # Hex only
        assert all(c in "0123456789abcdef" for c in sha)

    def test_returns_empty_on_missing(self, tmp_path):
        mod = _import_script()
        assert mod._file_sha256(tmp_path / "missing.txt") == ""

    def test_known_value(self, tmp_path):
        mod = _import_script()
        f = tmp_path / "f.txt"
        f.write_bytes(b"")  # empty
        # SHA-256 of empty file
        assert (mod._file_sha256(f) ==
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")


# ─── pyproject reader ─────────────────────────────────────────────────


class TestReadPyproject:
    def test_returns_dict(self):
        mod = _import_script()
        result = mod._read_pyproject()
        assert isinstance(result, dict)
        # Live repo has a pyproject.toml with [project] table
        assert "project" in result

    def test_missing_pyproject_returns_empty(self, tmp_path, monkeypatch):
        mod = _import_script()
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        # tmp_path has no pyproject.toml
        assert mod._read_pyproject() == {}


# ─── constraints reader ───────────────────────────────────────────────


class TestReadConstraints:
    def test_strips_comments_and_blanks(self, tmp_path, monkeypatch):
        mod = _import_script()
        cf = tmp_path / "constraints.txt"
        cf.write_text(
            "# this is a comment\n"
            "\n"
            "torch==2.5.1\n"
            "  # indented comment\n"
            "numpy>=1.24\n"
        )
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        result = mod._read_constraints()
        assert "torch==2.5.1" in result
        assert "numpy>=1.24" in result
        # Comments + blanks excluded
        for r in result:
            assert not r.startswith("#")
            assert r.strip()

    def test_missing_file_returns_empty(self, tmp_path, monkeypatch):
        mod = _import_script()
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        assert mod._read_constraints() == []


# ─── Genesis module listing ───────────────────────────────────────────


class TestListGenesisModules:
    def test_live_repo_non_empty(self):
        mod = _import_script()
        modules = mod._list_genesis_modules()
        assert len(modules) > 0
        for m in modules:
            assert "path" in m
            assert "sha256" in m
            assert m["path"].startswith("sndr/")

    def test_skips_pycache(self):
        mod = _import_script()
        modules = mod._list_genesis_modules()
        for m in modules:
            assert "__pycache__" not in m["path"]

    def test_missing_dir_returns_empty(self, tmp_path, monkeypatch):
        mod = _import_script()
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        assert mod._list_genesis_modules() == []

    def test_sha_format(self):
        mod = _import_script()
        modules = mod._list_genesis_modules()
        for m in modules[:5]:
            assert len(m["sha256"]) == 64


# ─── Registry snapshot ───────────────────────────────────────────────


class TestRegistrySnapshot:
    def test_live_snapshot_shape(self):
        mod = _import_script()
        snap = mod._registry_snapshot()
        assert "total" in snap
        assert snap["total"] > 0
        assert "by_tier" in snap
        assert "by_lifecycle" in snap
        assert "by_default_on" in snap
        assert set(snap["by_default_on"]) == {"True", "False"}


# ─── Model configs snapshot ───────────────────────────────────────────


class TestModelConfigsSnapshot:
    def test_returns_list(self):
        mod = _import_script()
        snaps = mod._model_configs_snapshot()
        assert isinstance(snaps, list)
        # Skip exact length check (varies by builtin set)


# ─── Pins ─────────────────────────────────────────────────────────────


class TestVllmPins:
    def test_returns_list(self):
        mod = _import_script()
        pins = mod._vllm_pins()
        assert isinstance(pins, list)


# ─── Installed distributions ──────────────────────────────────────────


class TestInstalledDistributions:
    def test_returns_sorted_list(self):
        mod = _import_script()
        dists = mod._installed_distributions()
        assert isinstance(dists, list)
        # Sorted by name (case-insensitive)
        names = [d["name"].lower() for d in dists]
        assert names == sorted(names)

    def test_each_entry_has_required_fields(self):
        mod = _import_script()
        dists = mod._installed_distributions()
        for d in dists[:10]:
            assert "name" in d
            assert "version" in d
            assert "summary" in d


# ─── SBOM emitters ────────────────────────────────────────────────────


class TestEmitCyclonedx:
    def test_writes_valid_json(self, tmp_path):
        mod = _import_script()
        payload = {
            "pyproject": {"project": {"name": "test-pkg",
                                       "version": "1.0.0",
                                       "dependencies": ["foo>=1.0"]}},
            "installed_distributions": [{"name": "bar", "version": "2.0",
                                          "summary": "test"}],
            "generated_at": "2026-05-30T00:00:00Z",
            "patch_registry": {"total": 42},
            "known_good_vllm_pins": ["0.20.0", "0.21.0"],
            "image_allowlist": [],
            "model_configs": [],
        }
        out = tmp_path / "out.cdx.json"
        mod.emit_cyclonedx(payload, out)
        data = json.loads(out.read_text())
        assert data["bomFormat"] == "CycloneDX"
        assert data["specVersion"] == "1.5"
        assert data["metadata"]["component"]["name"] == "test-pkg"
        assert data["metadata"]["component"]["version"] == "1.0.0"
        # Components include both direct + installed deps
        assert len(data["components"]) == 2

    def test_properties_include_patch_registry_total(self, tmp_path):
        mod = _import_script()
        payload = {
            "pyproject": {"project": {"name": "test", "version": "1"}},
            "installed_distributions": [],
            "generated_at": "2026-05-30T00:00:00Z",
            "patch_registry": {"total": 99},
            "known_good_vllm_pins": [],
            "image_allowlist": [],
            "model_configs": [],
        }
        out = tmp_path / "out.cdx.json"
        mod.emit_cyclonedx(payload, out)
        data = json.loads(out.read_text())
        prop_dict = {p["name"]: p["value"] for p in data["properties"]}
        assert prop_dict["genesis:patch_registry_total"] == "99"


class TestEmitSpdx:
    def test_writes_spdx_shape(self, tmp_path):
        mod = _import_script()
        payload = {
            "pyproject": {"project": {"name": "test-pkg", "version": "1.0.0"}},
            "installed_distributions": [],
            "generated_at": "2026-05-30T00:00:00Z",
            "patch_registry": {"total": 0},
            "known_good_vllm_pins": [],
            "image_allowlist": [],
            "model_configs": [],
        }
        out = tmp_path / "out.spdx.json"
        mod.emit_spdx(payload, out)
        data = json.loads(out.read_text())
        # SPDX 2.3 carries SPDXID + packages
        assert any("SPDXRef-Package-Genesis" == p.get("SPDXID")
                   for p in data.get("packages", []))
