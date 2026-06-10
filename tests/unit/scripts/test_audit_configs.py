# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_configs.py` — preset alias compose gate.

Contract:

  1. _alias_ids() finds preset YAML files (sorted, underscore-prefixed
     excluded as private/draft).
  2. _verify_alias() returns (True, summary) for a known-good alias.
  3. _verify_alias() returns (False, ...) on registry import failure.
  4. Live repo composes ALL presets cleanly (regression anchor).
  5. --json shape exposes {total, failures, results[]}.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_configs.py"


def _import_script():
    name = "_audit_configs_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestAliasIds:
    def test_alias_ids_returns_sorted_list(self):
        mod = _import_script()
        ids = mod._alias_ids()
        assert ids == sorted(ids)

    def test_alias_ids_non_empty_in_live_repo(self):
        mod = _import_script()
        ids = mod._alias_ids()
        assert len(ids) > 0, "live repo should have preset aliases"

    def test_underscore_prefixed_excluded(self):
        mod = _import_script()
        ids = mod._alias_ids()
        for alias in ids:
            assert not alias.startswith("_"), (
                f"underscore-prefixed alias should be skipped: {alias}"
            )


class TestVerifyAliasLive:
    def test_live_aliases_all_compose(self):
        """Regression anchor — every preset under presets/ composes
        cleanly. Fails if a profile/hardware id rename or YAML edit
        breaks the resolver."""
        mod = _import_script()
        failures = []
        for alias in mod._alias_ids():
            ok, summary = mod._verify_alias(alias)
            if not ok:
                failures.append((alias, summary))
        assert not failures, (
            f"{len(failures)} preset(s) failed to compose:\n  "
            + "\n  ".join(f"{a}: {s}" for a, s in failures)
        )


class TestVerifyAliasErrors:
    def test_registry_import_error_handled(self, monkeypatch):
        """If registry_v2 import fails, _verify_alias returns
        (False, "registry_v2 not importable: ...")."""
        mod = _import_script()
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "sndr.model_configs.registry_v2":
                raise ImportError("synthetic")
            return real_import(name, *args, **kwargs)

        sys.modules.pop("sndr.model_configs.registry_v2", None)
        monkeypatch.setattr(builtins, "__import__", fake_import)
        ok, summary = mod._verify_alias("any-alias")
        assert ok is False
        assert "registry_v2 not importable" in summary


class TestMainExitCode:
    def test_main_exits_zero_on_clean(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0


class TestJsonOutput:
    def test_json_shape(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert "total" in payload
        assert "failures" in payload
        assert "results" in payload
        assert payload["failures"] == 0
        assert isinstance(payload["results"], list)
        assert len(payload["results"]) == payload["total"]
        for r in payload["results"]:
            assert "alias" in r
            assert "ok" in r
            assert "summary" in r
