# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_license_anchor.py` — release-tier trust anchor gate.

Contract:

  1. Default mode (no --release) is warn-only — returns 0 even with
     dev anchor active.
  2. --release mode returns 1 when current anchor matches a
     development-only fingerprint.
  3. When anchor is NOT in the forbidden set, returns 0 in both modes.
  4. ImportError on the license module surfaces as exit 2.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_license_anchor.py"


def _import_script():
    name = "_audit_license_anchor_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── Live regression anchor ────────────────────────────────────────────


class TestLive:
    def test_default_mode_returns_0(self):
        """Default mode is warn-only — current dev anchor produces a
        warning but exits 0 (dev builds stay unblocked)."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0


# ─── Dev anchor detection ──────────────────────────────────────────────


class TestDevAnchorDetection:
    def test_dev_anchor_warn_only_in_default(self, monkeypatch):
        mod = _import_script()
        # Force the loader to return a dev anchor.
        monkeypatch.setattr(
            mod, "_load",
            lambda: ("DEV-FAKE-ANCHOR", frozenset({"DEV-FAKE-ANCHOR"})),
        )
        # Default mode (no --release) → warn-only, exit 0.
        rc = mod.main(argv=[])
        assert rc == 0

    def test_dev_anchor_fails_release_mode(self, monkeypatch):
        mod = _import_script()
        monkeypatch.setattr(
            mod, "_load",
            lambda: ("DEV-FAKE-ANCHOR", frozenset({"DEV-FAKE-ANCHOR"})),
        )
        rc = mod.main(argv=["--release"])
        assert rc == 1

    def test_production_anchor_passes_release(self, monkeypatch):
        mod = _import_script()
        monkeypatch.setattr(
            mod, "_load",
            lambda: ("PROD-REAL-ANCHOR", frozenset({"DEV-FAKE-ANCHOR"})),
        )
        rc = mod.main(argv=["--release"])
        assert rc == 0

    def test_production_anchor_passes_default(self, monkeypatch):
        mod = _import_script()
        monkeypatch.setattr(
            mod, "_load",
            lambda: ("PROD-REAL-ANCHOR", frozenset({"DEV-FAKE-ANCHOR"})),
        )
        rc = mod.main(argv=[])
        assert rc == 0


# ─── ImportError surfaces ──────────────────────────────────────────────


class TestLoaderFailure:
    def test_load_exits_2_on_import_error(self, monkeypatch, capsys):
        """The _load() helper catches (ImportError, AttributeError) and
        calls sys.exit(2). Test it directly by monkeypatching the
        import to raise."""
        mod = _import_script()
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "sndr.license":
                raise ImportError("synthetic for test")
            return real_import(name, *args, **kwargs)

        # Drop the cached license module if present so the patched
        # import path actually fires.
        sys.modules.pop("sndr.license", None)
        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(SystemExit) as exc:
            mod._load()
        assert exc.value.code == 2
