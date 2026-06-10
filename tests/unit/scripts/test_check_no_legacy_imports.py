# SPDX-License-Identifier: Apache-2.0
"""Etap 5.4/5.5 (audit 2026-05-12): AST-based legacy import scanner
catches the shapes the old shell regex missed.

Covers:
  • `import vllm.sndr_core.patches.foo`
  • `from vllm.sndr_core.patches.foo import x`
  • `from vllm.sndr_core import patches`        (previously missed)
  • `import vllm._genesis.kernels`
  • Non-Python files (.yml/.toml/.json) flagged by grep path
  • Allowlist suppresses historical archive directories

NOTE: the `vllm.sndr_core` strings in fixtures below are deliberate —
they are the LEGACY NEEDLES this detector must catch. Do not remap
them to the v12 `sndr.` namespace.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Import the script as a module — it has no top-level side effects.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import check_no_legacy_imports as M  # noqa: E402


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


class TestPythonImportShapes:
    """`_check_python_imports` AST scanner — every shape we care about."""

    def test_classic_dotted_import_caught(self, tmp_path):
        f = _write(tmp_path, "x.py", "import vllm.sndr_core.patches.foo\n")
        vs = M._check_python_imports(f)
        assert len(vs) == 1
        assert "vllm.sndr_core.patches.foo" in vs[0][1]

    def test_from_dotted_import_caught(self, tmp_path):
        f = _write(
            tmp_path, "x.py",
            "from vllm.sndr_core.patches.foo import bar\n",
        )
        vs = M._check_python_imports(f)
        assert len(vs) == 1

    def test_from_parent_import_name_caught(self, tmp_path):
        """The shape the old shell scanner missed."""
        f = _write(
            tmp_path, "x.py",
            "from vllm.sndr_core import patches\n",
        )
        vs = M._check_python_imports(f)
        assert len(vs) == 1
        assert "from vllm.sndr_core import patches" in vs[0][1]

    def test_from_vllm_import_genesis_caught(self, tmp_path):
        f = _write(
            tmp_path, "x.py",
            "from vllm import _genesis\n",
        )
        vs = M._check_python_imports(f)
        assert len(vs) == 1

    def test_genesis_dotted_import_caught(self, tmp_path):
        f = _write(
            tmp_path, "x.py",
            "import vllm._genesis.kernels.foo\n",
        )
        vs = M._check_python_imports(f)
        assert len(vs) == 1

    def test_clean_file_zero_violations(self, tmp_path):
        f = _write(
            tmp_path, "x.py",
            "from sndr.engines.vllm.patches.attention import gdn\n"
            "import sndr.dispatcher\n",
        )
        assert M._check_python_imports(f) == []

    def test_syntax_error_swallowed_silently(self, tmp_path):
        """Files with syntax errors are not this gate's concern."""
        f = _write(tmp_path, "broken.py", "def x(:\n")
        assert M._check_python_imports(f) == []


class TestTextScanner:
    """Non-Python files — YAML/TOML/JSON/SH/MD — caught by grep path."""

    def test_yaml_with_legacy_ref_caught(self, tmp_path):
        f = _write(tmp_path, "x.yml",
                    "module: vllm._genesis.kernels.foo\n")
        vs = M._check_text(f)
        assert len(vs) == 1

    def test_toml_caught(self, tmp_path):
        f = _write(tmp_path, "x.toml",
                    'legacy = "vllm.sndr_core.patches.attention"\n')
        vs = M._check_text(f)
        assert len(vs) == 1

    def test_shell_script_caught(self, tmp_path):
        f = _write(tmp_path, "x.sh",
                    "python3 -m vllm._genesis.apply_all\n")
        vs = M._check_text(f)
        assert len(vs) == 1

    def test_clean_text_zero_violations(self, tmp_path):
        f = _write(tmp_path, "x.md",
                    "Genesis sndr_core is the v11 namespace.\n")
        assert M._check_text(f) == []


class TestAllowlist:
    """Archive paths must be allowlisted so historical artifacts stay
    legible without polluting the gate."""

    def test_archive_substring_matches(self):
        # Synthetic path with `scripts/_archive/` substring
        p = REPO_ROOT / "scripts" / "_archive" / "old.sh"
        assert M._is_allowlisted(p)

    def test_baselines_dir_matches(self):
        p = REPO_ROOT / "tests" / "integration" / "baselines" / "v8.json"
        assert M._is_allowlisted(p)

    def test_active_path_not_matched(self):
        p = REPO_ROOT / "vllm" / "sndr_core" / "cli" / "compose.py"
        assert not M._is_allowlisted(p)


class TestRealRepo:
    """End-to-end: the gate should be clean against the live repo."""

    def test_main_returns_zero(self, capsys):
        rc = M.main([])
        captured = capsys.readouterr()
        assert rc == 0, (
            "legacy-import gate is RED in the live repo:\n"
            + captured.out + captured.err
        )
