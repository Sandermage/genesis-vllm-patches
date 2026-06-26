# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/security_scan.py` — release-pipeline security gate.

Contract enforced:

  1. ALLOWLIST_PATHS is a list (mutable by intent — operator may extend).
  2. _is_allowlisted: paths under ALLOWLIST_PATHS prefixes are waived.
  3. _grep_pattern: returns (path, lineno, line) tuples; respects
     allowlist by default; honors inline `security_scan: allow` marker.
  4. check_no_operator_paths detects `/home/sander` and `/Users/sander`.
  5. check_no_private_ips detects RFC 1918 ranges (10.*, 172.16-31.*,
     192.168.*) but ONLY in docs/.
  6. check_no_private_keys detects RSA/OpenSSH/EC/DSA private key markers.
  7. check_no_env_files detects committed .env / .env.* files.
  8. check_no_aws_keys detects AKIA-prefixed access keys.
  9. check_release_artifacts_present requires SBOM + constraints under
     release/ in --public-release mode.
  10. Live repo passes all checks (regression anchor).
  11. main exit 0 on clean, 1 on violations.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "security_scan.py"


def _import_script():
    name = "_security_scan_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── Allowlist semantics ───────────────────────────────────────────────


class TestAllowlist:
    def test_sndr_private_allowlisted(self):
        mod = _import_script()
        assert mod._is_allowlisted("sndr_private/planning/notes.md")
        assert mod._is_allowlisted("sndr_private/research/data.json")

    def test_top_level_underscore_archive_allowlisted(self):
        """ALLOWLIST uses startswith — `_archive/` only matches paths
        that START with `_archive/`, not embedded ones."""
        mod = _import_script()
        # Top-level `_archive/` matches
        assert mod._is_allowlisted("_archive/old.md")
        # Embedded — does NOT match (would be a substring test, not prefix)
        assert not mod._is_allowlisted("docs/_archive/old.md")

    def test_tests_dir_allowlisted(self):
        mod = _import_script()
        assert mod._is_allowlisted("tests/unit/scripts/test_something.py")

    def test_self_allowlisted(self):
        """security_scan.py itself contains the forbidden patterns as
        regex literals — must be waived."""
        mod = _import_script()
        assert mod._is_allowlisted("scripts/security_scan.py")

    def test_normal_path_not_allowlisted(self):
        mod = _import_script()
        assert not mod._is_allowlisted("vllm/sndr_core/module.py")
        assert not mod._is_allowlisted("docs/USAGE.md")


# ─── Operator paths detection ──────────────────────────────────────────


class TestOperatorPaths:
    def test_clean_files_pass(self, tmp_path, monkeypatch):
        mod = _import_script()
        f = tmp_path / "ok.py"
        f.write_text("# no operator paths\nfoo = '/tmp/x'\n")
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        result = mod.check_no_operator_paths(["ok.py"])
        assert result == []

    def test_detects_home_sander(self, tmp_path, monkeypatch):
        mod = _import_script()
        f = tmp_path / "leak.py"
        f.write_text("REPO = '/home/sander/genesis'\n")
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        result = mod.check_no_operator_paths(["leak.py"])
        assert len(result) == 1
        assert "leak.py" in result[0]
        assert "/home/sander" in result[0]

    def test_detects_users_sander(self, tmp_path, monkeypatch):
        mod = _import_script()
        f = tmp_path / "leak.py"
        f.write_text("PATH = '/Users/sander/docs'\n")
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        result = mod.check_no_operator_paths(["leak.py"])
        assert len(result) == 1

    def test_allowlisted_path_skipped(self, tmp_path, monkeypatch):
        mod = _import_script()
        sndr = tmp_path / "sndr_private" / "notes.md"
        sndr.parent.mkdir(parents=True)
        sndr.write_text("Operator at /home/sander\n")
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        # sndr_private/ is allowlisted → ignored
        result = mod.check_no_operator_paths(["sndr_private/notes.md"])
        assert result == []

    def test_inline_allow_marker_skipped(self, tmp_path, monkeypatch):
        mod = _import_script()
        f = tmp_path / "doc.py"
        # Line carries inline marker → grep ignores
        f.write_text(
            "EXAMPLE = '/home/sander/x'  # security_scan: allow\n"
        )
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        result = mod.check_no_operator_paths(["doc.py"])
        assert result == []


# ─── Private IP detection ──────────────────────────────────────────────


class TestPrivateIps:
    def test_detects_192_168_in_docs(self, tmp_path, monkeypatch):
        mod = _import_script()
        d = tmp_path / "docs"
        d.mkdir()
        f = d / "RUNBOOK.md"
        f.write_text("ssh rig at 192.168.1.50\n")
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        result = mod.check_no_private_ips(["docs/RUNBOOK.md"])
        assert len(result) == 1

    def test_detects_10_dot_range(self, tmp_path, monkeypatch):
        mod = _import_script()
        d = tmp_path / "docs"
        d.mkdir()
        f = d / "ARCH.md"
        f.write_text("Backend at 10.0.0.5\n")
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        result = mod.check_no_private_ips(["docs/ARCH.md"])
        assert len(result) == 1

    def test_detects_172_16_31_range(self, tmp_path, monkeypatch):
        mod = _import_script()
        d = tmp_path / "docs"
        d.mkdir()
        f = d / "X.md"
        f.write_text("Server: 172.20.5.10\n")
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        result = mod.check_no_private_ips(["docs/X.md"])
        assert len(result) == 1

    def test_172_15_not_private(self, tmp_path, monkeypatch):
        """172.15.x.x is NOT in RFC 1918 (range is 172.16-31)."""
        mod = _import_script()
        d = tmp_path / "docs"
        d.mkdir()
        f = d / "X.md"
        f.write_text("Public host: 172.15.5.10\n")
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        result = mod.check_no_private_ips(["docs/X.md"])
        assert result == []

    def test_non_docs_path_not_scanned(self, tmp_path, monkeypatch):
        """check_no_private_ips only looks at docs/ paths."""
        mod = _import_script()
        f = tmp_path / "scripts" / "stuff.py"
        f.parent.mkdir()
        f.write_text("HOST = '192.168.1.50'\n")
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        result = mod.check_no_private_ips(["scripts/stuff.py"])
        # Non-docs paths skipped by this check
        assert result == []


# ─── Private keys ──────────────────────────────────────────────────────


class TestPrivateKeys:
    def test_detects_rsa_marker(self, tmp_path, monkeypatch):
        mod = _import_script()
        f = tmp_path / "leak.txt"
        # String split prevents this test source line from itself matching
        # the BEGIN-PRIVATE-KEY regex during the live security_scan run.
        f.write_text("-----BEGIN " + "RSA PRIVATE " + "KEY-----\nAAAA\n")
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        result = mod.check_no_private_keys(["leak.txt"])
        assert len(result) == 1

    def test_detects_openssh_marker(self, tmp_path, monkeypatch):
        mod = _import_script()
        f = tmp_path / "leak.txt"
        f.write_text("-----BEGIN " + "OPENSSH PRIVATE " + "KEY-----\n")
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        result = mod.check_no_private_keys(["leak.txt"])
        assert len(result) == 1

    def test_public_key_not_detected(self, tmp_path, monkeypatch):
        """Public keys (no `PRIVATE` keyword) are fine."""
        mod = _import_script()
        f = tmp_path / "pub.txt"
        f.write_text("-----BEGIN PUBLIC KEY-----\n")
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        result = mod.check_no_private_keys(["pub.txt"])
        assert result == []


# ─── Env files ─────────────────────────────────────────────────────────


class TestEnvFiles:
    def test_detects_dot_env(self):
        mod = _import_script()
        result = mod.check_no_env_files(["foo/bar/.env"])
        assert len(result) == 1

    def test_detects_dot_env_local(self):
        mod = _import_script()
        result = mod.check_no_env_files(["root/.env.local"])
        assert len(result) == 1

    def test_other_dotfiles_pass(self):
        mod = _import_script()
        result = mod.check_no_env_files([".gitignore", "dir/.envrc-example"])
        # .envrc-example: ".envrc-example" starts with ".env" → flagged
        # That's the actual behavior — name.startswith(".env.") needs literal dot.
        # ".envrc-example".startswith(".env.") = False. So it should pass.
        assert result == []

    def test_env_example_flagged(self):
        """`.env.example` IS flagged as a `.env.*` file."""
        mod = _import_script()
        result = mod.check_no_env_files([".env.example"])
        assert len(result) == 1


# ─── AWS keys ──────────────────────────────────────────────────────────


class TestAwsKeys:
    def test_detects_akia_pattern(self, tmp_path, monkeypatch):
        mod = _import_script()
        f = tmp_path / "leak.py"
        f.write_text("KEY = 'AKIAIOSFODNN7EXAMPLE'\n")
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        result = mod.check_no_aws_keys(["leak.py"])
        assert len(result) == 1

    def test_non_akia_string_ignored(self, tmp_path, monkeypatch):
        mod = _import_script()
        f = tmp_path / "clean.py"
        f.write_text("KEY = 'AAAAIOSFODNN7EXAMPLE'\n")  # no AKIA prefix
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        result = mod.check_no_aws_keys(["clean.py"])
        assert result == []


# ─── Release artifacts ─────────────────────────────────────────────────


class TestReleaseArtifacts:
    def test_missing_sbom_and_constraints_flagged(self, tmp_path, monkeypatch):
        mod = _import_script()
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        result = mod.check_release_artifacts_present()
        assert len(result) == 2

    def test_present_artifacts_pass(self, tmp_path, monkeypatch):
        mod = _import_script()
        rel = tmp_path / "release"
        rel.mkdir()
        (rel / "SBOM.spdx.json").write_text("{}")
        (rel / "constraints.txt").write_text("")
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        assert mod.check_release_artifacts_present() == []


# ─── Live regression anchor ────────────────────────────────────────────


class TestLive:
    def test_main_exits_zero_default_mode(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, (
            f"security_scan failed on live repo:\n{result.stdout}"
        )

    def test_json_shape(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert "checks" in payload
        assert "total_failures" in payload
        assert "scanned_files" in payload
        assert payload["total_failures"] == 0
        for check_name in ("operator_paths", "private_ips", "private_keys",
                           "env_files", "aws_keys"):
            assert check_name in payload["checks"]
