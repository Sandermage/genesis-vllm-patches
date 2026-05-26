# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/audit_external_findings.py`` — §9.A.7
(A7-EXTERNAL-FINDINGS-AUDIT.1, 2026-05-27).

Coverage:

  * Missing tracker dir → 0 findings, exit 0 (CI no-op)
  * Valid finding YAML → exit 0
  * Schema-error finding → exit 1
  * Stale finding → warning at default (exit 0); error under
    ``--strict-warnings`` (exit 1)
  * ``--findings-dir`` override
  * ``--with-network`` returns exit 2 with operator-approval message
  * JSON shape stable
  * Live corpus smoke (operator tree clean today)
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from datetime import date, timedelta
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_external_findings.py"


def _import_audit_module():
    spec = importlib.util.spec_from_file_location(
        "audit_external_findings", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audit_external_findings"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def audit_mod():
    return _import_audit_module()


# ─── Helpers ──────────────────────────────────────────────────────────────


def _write_valid_finding(
    target_dir: Path, fid: str = "external-vllm-99999",
    *, last_reviewed: str = None,
) -> Path:
    """Write a schema-valid finding YAML into ``target_dir``."""
    if last_reviewed is None:
        last_reviewed = date.today().isoformat()
    yaml = textwrap.dedent(f"""
        schema_version: 1
        id: {fid}
        source: vllm-pr
        url: https://github.com/vllm-project/vllm/pull/99999
        title: "Synthetic test finding"
        discovered_at: '2026-05-27'
        category: misc
        status: watch
        risk: low
        acceptance: |
          Synthetic finding for audit-script test fixtures.
        last_reviewed: '{last_reviewed}'
        review_cadence: on-pin-bump
    """).strip() + "\n"
    target_dir.mkdir(parents=True, exist_ok=True)
    p = target_dir / f"{fid}.yaml"
    p.write_text(yaml, encoding="utf-8")
    return p


def _write_invalid_finding(target_dir: Path) -> Path:
    """Write a finding with schema errors (unknown source enum)."""
    yaml = textwrap.dedent("""
        schema_version: 1
        id: external-bad-1
        source: invented-source
        url: https://example.com/
        title: "Schema-error finding"
        discovered_at: '2026-05-27'
        category: misc
        status: watch
        risk: low
        acceptance: |
          Used to exercise schema-error path.
        last_reviewed: '2026-05-27'
        review_cadence: on-pin-bump
    """).strip() + "\n"
    target_dir.mkdir(parents=True, exist_ok=True)
    p = target_dir / "external-bad-1.yaml"
    p.write_text(yaml, encoding="utf-8")
    return p


def _write_stale_finding(target_dir: Path) -> Path:
    """Write a finding that is past its weekly cadence (F-4 warning)."""
    old = (date.today() - timedelta(days=30)).isoformat()
    yaml = textwrap.dedent(f"""
        schema_version: 1
        id: external-stale-1
        source: vllm-pr
        url: https://github.com/vllm-project/vllm/pull/12345
        title: "Stale finding for F-4 test"
        discovered_at: '2026-04-01'
        category: misc
        status: watch
        risk: low
        acceptance: |
          Triggers F-4 staleness warning.
        last_reviewed: '{old}'
        review_cadence: weekly
    """).strip() + "\n"
    target_dir.mkdir(parents=True, exist_ok=True)
    p = target_dir / "external-stale-1.yaml"
    p.write_text(yaml, encoding="utf-8")
    return p


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        capture_output=True, text=True, cwd=REPO_ROOT, check=False,
    )


# ─── Tests ────────────────────────────────────────────────────────────────


class TestMissingDir:
    """Tracker dir absent (typical CI/public checkout) → clean no-op."""

    def test_nonexistent_dir_exit_zero(self, tmp_path):
        ghost = tmp_path / "does-not-exist"
        result = _run("--findings-dir", str(ghost))
        assert result.returncode == 0, (
            f"missing tracker dir should exit 0, got rc={result.returncode}\n"
            f"stdout:\n{result.stdout}"
        )
        assert "clean no-op" in result.stdout or "0 findings" in result.stdout.lower() \
            or "tracker dir absent" in result.stdout

    def test_nonexistent_dir_json(self, tmp_path):
        ghost = tmp_path / "does-not-exist"
        result = _run("--findings-dir", str(ghost), "--json")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["exists"] is False
        assert data["finding_count"] == 0
        assert data["errors"] == []
        assert data["warnings"] == []


class TestValidFinding:
    """A schema-valid finding → exit 0."""

    def test_valid_finding_exit_zero(self, tmp_path):
        _write_valid_finding(tmp_path)
        result = _run("--findings-dir", str(tmp_path))
        assert result.returncode == 0, (
            f"valid finding should exit 0, got rc={result.returncode}\n"
            f"stdout:\n{result.stdout}"
        )
        assert "passes validation" in result.stdout

    def test_valid_finding_json_shape(self, tmp_path):
        _write_valid_finding(tmp_path)
        result = _run("--findings-dir", str(tmp_path), "--json")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["finding_count"] == 1
        assert data["errors"] == []
        assert data["warnings"] == []
        assert data["passed_schema"] is True


class TestSchemaError:
    """Schema-error finding → exit 1 with errors list populated."""

    def test_schema_error_exit_one(self, tmp_path):
        _write_invalid_finding(tmp_path)
        result = _run("--findings-dir", str(tmp_path))
        assert result.returncode == 1, (
            f"schema error should exit 1, got rc={result.returncode}\n"
            f"stdout:\n{result.stdout}"
        )

    def test_schema_error_in_json(self, tmp_path):
        _write_invalid_finding(tmp_path)
        result = _run("--findings-dir", str(tmp_path), "--json")
        assert result.returncode == 1
        data = json.loads(result.stdout)
        assert len(data["errors"]) >= 1
        assert any("source" in e["message"].lower() for e in data["errors"])


class TestStaleWarning:
    """F-4 staleness — warning at default; error under --strict-warnings."""

    def test_stale_at_default_exit_zero(self, tmp_path):
        _write_stale_finding(tmp_path)
        result = _run("--findings-dir", str(tmp_path))
        assert result.returncode == 0, (
            f"stale finding at default should exit 0 (informational), "
            f"got rc={result.returncode}\nstdout:\n{result.stdout}"
        )
        # JSON output to confirm warning surfaced:
        json_result = _run("--findings-dir", str(tmp_path), "--json")
        data = json.loads(json_result.stdout)
        assert len(data["warnings"]) >= 1
        assert any("F-4" in w["rule"] for w in data["warnings"])

    def test_stale_under_strict_exit_one(self, tmp_path):
        _write_stale_finding(tmp_path)
        result = _run(
            "--findings-dir", str(tmp_path), "--strict-warnings",
        )
        assert result.returncode == 1, (
            f"stale finding under --strict-warnings should exit 1, "
            f"got rc={result.returncode}\nstdout:\n{result.stdout}"
        )

    def test_clean_under_strict_still_zero(self, tmp_path):
        """--strict-warnings on a clean tracker stays at exit 0."""
        _write_valid_finding(tmp_path)
        result = _run(
            "--findings-dir", str(tmp_path), "--strict-warnings",
        )
        assert result.returncode == 0


class TestWithNetworkGate:
    """``--with-network`` is reserved; exit 2 with operator-approval message."""

    def test_with_network_exit_two(self):
        result = _run("--with-network")
        assert result.returncode == 2
        assert "operator-approval" in result.stderr.lower() \
            or "reserved" in result.stderr.lower()

    def test_with_network_does_not_invoke_validator(self, tmp_path):
        """Even if --findings-dir points at a clean dir, --with-network
        short-circuits to exit 2 without running the validator."""
        _write_valid_finding(tmp_path)
        result = _run("--with-network", "--findings-dir", str(tmp_path))
        assert result.returncode == 2


class TestResolveDir:
    """``_resolve_dir`` falls back to canonical default when no override."""

    def test_explicit_override(self, audit_mod, tmp_path):
        p = audit_mod._resolve_dir(str(tmp_path))
        assert p == tmp_path.resolve()

    def test_no_override_falls_back_to_default(self, audit_mod):
        p = audit_mod._resolve_dir(None)
        # Default resolved by registry — must be absolute path under
        # REPO_ROOT (either sndr_private or docs fallback).
        assert p.is_absolute()


class TestLiveCorpus:
    """Live operator tree must exit 0 with the seeded finding."""

    def test_live_default_exit_zero(self):
        result = _run()
        assert result.returncode == 0, (
            f"live tracker should be clean, got rc={result.returncode}\n"
            f"stdout:\n{result.stdout}"
        )

    def test_live_json_shape(self):
        result = _run("--json")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "findings_dir" in data
        assert "finding_count" in data
        assert "errors" in data
        assert "warnings" in data
        assert data["passed_schema"] is True

    def test_help_works(self):
        result = _run("--help")
        assert result.returncode == 0
        assert "audit_external_findings" in result.stdout
        assert "--strict-warnings" in result.stdout
        assert "--with-network" in result.stdout
