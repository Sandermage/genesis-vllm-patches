# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_engine_boundary.py` — §10.3 #5.

Catches unguarded `import vllm.sndr_engine` in vllm/sndr_core/ while
correctly allowing the optional-discovery pattern (import wrapped
inside `try / except ImportError`).
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_engine_boundary.py"


def _import():
    name = "_audit_engine_boundary_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """Rebind script's REPO_ROOT to tmp_path so scratch files can be
    `relative_to`-d cleanly. TestLiveCorpus uses subprocess and is not
    affected."""
    mod = _import()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    return tmp_path


class TestImportPatternDetection:
    def test_unguarded_from_caught(self, fake_repo):
        mod = _import()
        p = fake_repo / "x.py"
        p.write_text("from vllm.sndr_engine import overlay\n")
        hits = mod._check_file(p)
        assert hits
        assert "unguarded" in hits[0]

    def test_unguarded_import_caught(self, fake_repo):
        mod = _import()
        p = fake_repo / "x.py"
        p.write_text("import vllm.sndr_engine\n")
        hits = mod._check_file(p)
        assert hits

    def test_submodule_unguarded_caught(self, fake_repo):
        mod = _import()
        p = fake_repo / "x.py"
        p.write_text("import vllm.sndr_engine.overlay\n")
        hits = mod._check_file(p)
        assert hits

    def test_try_except_importerror_clean(self, fake_repo):
        mod = _import()
        p = fake_repo / "x.py"
        p.write_text(textwrap.dedent("""
            def probe():
                try:
                    from vllm.sndr_engine import engine_available
                except ImportError:
                    return False
                return engine_available()
        """))
        hits = mod._check_file(p)
        assert hits == []

    def test_try_except_modulenotfound_clean(self, fake_repo):
        mod = _import()
        p = fake_repo / "x.py"
        p.write_text(textwrap.dedent("""
            def probe():
                try:
                    import vllm.sndr_engine
                except ModuleNotFoundError:
                    return None
                return vllm.sndr_engine
        """))
        hits = mod._check_file(p)
        assert hits == []

    def test_try_except_tuple_clean(self, fake_repo):
        mod = _import()
        p = fake_repo / "x.py"
        p.write_text(textwrap.dedent("""
            def probe():
                try:
                    import vllm.sndr_engine
                except (ImportError, RuntimeError):
                    return None
                return vllm.sndr_engine
        """))
        hits = mod._check_file(p)
        assert hits == []

    def test_unrelated_except_does_not_protect(self, fake_repo):
        mod = _import()
        p = fake_repo / "x.py"
        p.write_text(textwrap.dedent("""
            def probe():
                try:
                    import vllm.sndr_engine
                except ValueError:
                    return None
                return vllm.sndr_engine
        """))
        hits = mod._check_file(p)
        assert hits

    def test_allow_marker_skips(self, fake_repo):
        mod = _import()
        p = fake_repo / "x.py"
        p.write_text(
            "from vllm.sndr_engine import overlay  # audit-engine-boundary: allow\n"
        )
        hits = mod._check_file(p)
        assert hits == []

    def test_unrelated_import_clean(self, fake_repo):
        mod = _import()
        p = fake_repo / "x.py"
        p.write_text("from sndr.cli.legacy import config_keys\n")
        hits = mod._check_file(p)
        assert hits == []


class TestLiveCorpus:
    def test_live_repo_clean(self):
        rc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert rc.returncode == 0, (
            f"audit-engine-boundary failed on live corpus:\n"
            f"{rc.stdout}\n{rc.stderr}"
        )
