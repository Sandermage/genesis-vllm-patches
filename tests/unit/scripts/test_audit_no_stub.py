# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_no_stub.py` — §10.3 #2 / §10.5 no-stub gate.

Catches bare `raise NotImplementedError`, `TODO(...)` markers, and
`pass  # placeholder|scaffold|FIXME` sentinels in vllm/sndr_core code,
while correctly NOT flagging references inside docstrings / string
literals (which is how patches describe what they replace in upstream).
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_no_stub.py"


def _import():
    name = "_audit_no_stub_test"
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
    `relative_to`-d cleanly. The TestLiveCorpus test invokes the script
    via subprocess (separate process) and is not affected."""
    mod = _import()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    return tmp_path


class TestRaiseNotImplementedAst:
    def test_bare_raise_caught(self, fake_repo):
        """Phase 4.A (2026-05-22): updated fixture to use a multi-
        statement function body. The script's
        `_is_abstract_method_raise()` intentionally exempts SINGLE-
        statement function bodies as "canonical abstract-method shape"
        (also used by protocol stand-ins / interface contracts that
        don't import abc) — see scripts/audit_no_stub.py:104-133.
        A multi-statement body around the `raise` makes the fixture
        exercise the stub-detection path the audit actually catches.
        """
        mod = _import()
        p = fake_repo / "x.py"
        p.write_text(textwrap.dedent("""
            def f(x):
                x += 1
                raise NotImplementedError("bare")
        """))
        hits = mod._check_ast_raises(p, p.read_text())
        assert any("raise NotImplementedError" in h for h in hits)

    def test_raise_without_call_caught(self, fake_repo):
        """Phase 4.A (2026-05-22): same exempt-pattern boundary as
        test_bare_raise_caught — uses a multi-statement body so the
        audit doesn't treat the function as a canonical abstract
        shape and skip."""
        mod = _import()
        p = fake_repo / "x.py"
        p.write_text("def f(x):\n    x += 1\n    raise NotImplementedError\n")
        hits = mod._check_ast_raises(p, p.read_text())
        assert hits

    def test_single_statement_raise_is_exempt(self, fake_repo):
        """Positive-case counterpart: confirm the abstract-method
        exemption. A function whose body is a single
        `raise NotImplementedError(...)` (with or without args, with
        or without a leading docstring) is the canonical abstract /
        protocol shape and must NOT be flagged. Phase 4.A added this
        as a regression guard alongside the multi-statement fixtures
        above."""
        mod = _import()
        p = fake_repo / "x.py"
        p.write_text(textwrap.dedent('''
            def abstract_method(self):
                """Subclasses must implement this."""
                raise NotImplementedError("must override")
        '''))
        hits = mod._check_ast_raises(p, p.read_text())
        assert hits == [], (
            f"abstract-method shape should be exempt; got hits: {hits}"
        )

    def test_string_literal_not_flagged(self, fake_repo):
        """Patches often DESCRIBE a `NotImplementedError` raise they
        replace — that lives in a docstring, not as an actual raise."""
        mod = _import()
        p = fake_repo / "x.py"
        p.write_text(textwrap.dedent('''
            """Patch description: replaces NotImplementedError in upstream."""
            ANCHOR_TEXT = "raise NotImplementedError(\\"x\\")"
            def f():
                return 42
        '''))
        hits = mod._check_ast_raises(p, p.read_text())
        assert hits == []

    def test_allow_marker_on_same_line_skips(self, fake_repo):
        mod = _import()
        p = fake_repo / "x.py"
        p.write_text(textwrap.dedent("""
            class Base:
                def f(self):  # audit-no-stub: allow
                    raise NotImplementedError  # audit-no-stub: allow
        """))
        hits = mod._check_ast_raises(p, p.read_text())
        assert hits == []

    def test_allow_marker_on_prev_line_skips(self, fake_repo):
        mod = _import()
        p = fake_repo / "x.py"
        p.write_text(textwrap.dedent("""
            class Base:
                def f(self):
                    # audit-no-stub: allow — subclass must override
                    raise NotImplementedError
        """))
        hits = mod._check_ast_raises(p, p.read_text())
        assert hits == []


class TestTextualMarkers:
    def test_todo_with_name_caught(self, fake_repo):
        mod = _import()
        p = fake_repo / "x.py"
        p.write_text("# TODO(sandermage): finish me\n")
        hits = mod._check_textual_markers(p, p.read_text())
        assert hits

    def test_bare_todo_not_flagged(self, fake_repo):
        """Bare `TODO` comment without the (name) form is advisory only.
        The §10.5 rule requires `TODO(name): ...` form for tracking."""
        mod = _import()
        p = fake_repo / "x.py"
        p.write_text("# TODO finish later\n")
        hits = mod._check_textual_markers(p, p.read_text())
        assert hits == []

    def test_sentinel_pass_placeholder_caught(self, fake_repo):
        mod = _import()
        p = fake_repo / "x.py"
        p.write_text("def f():\n    pass  # placeholder\n")
        hits = mod._check_textual_markers(p, p.read_text())
        assert hits

    def test_sentinel_pass_fixme_caught(self, fake_repo):
        mod = _import()
        p = fake_repo / "x.py"
        p.write_text("def f():\n    pass  # FIXME — broken\n")
        hits = mod._check_textual_markers(p, p.read_text())
        assert hits

    def test_allow_marker_skips_textual(self, fake_repo):
        mod = _import()
        p = fake_repo / "x.py"
        p.write_text(
            "# TODO(sandermage): finish me  # audit-no-stub: allow\n"
        )
        hits = mod._check_textual_markers(p, p.read_text())
        assert hits == []


class TestTestPathsExempt:
    def test_test_path_filter(self):
        mod = _import()
        assert mod._is_test_path(REPO_ROOT / "tests" / "x.py")
        assert mod._is_test_path(REPO_ROOT / "tests" / "unit" / "test_x.py")
        assert not mod._is_test_path(
            REPO_ROOT / "vllm" / "sndr_core" / "license.py"
        )


class TestLiveCorpus:
    """vllm/sndr_core/ must currently pass the no-stub gate cleanly."""

    def test_live_repo_clean(self):
        rc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert rc.returncode == 0, (
            f"audit-no-stub failed on live corpus:\n{rc.stdout}\n{rc.stderr}"
        )
