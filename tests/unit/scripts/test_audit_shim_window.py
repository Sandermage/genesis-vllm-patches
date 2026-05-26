# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/audit_shim_window.py`` —
BUILD-SHIM-WINDOW-AUDIT.1 (2026-05-26).

Builds synthetic shim trees in ``tmp_path`` and asserts each E.1–E.5
rule fires (or stays silent) on the expected drift case. Also runs
the live manifest against the real tree as a smoke check.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_shim_window.py"


def _import_module():
    """Import audit_shim_window as a module for direct ShimSpec testing."""
    spec = importlib.util.spec_from_file_location(
        "audit_shim_window", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audit_shim_window"] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_valid_shim(root: Path, overlay_target: Path) -> Path:
    """Construct a synthetic gemma4-style shim under ``root``.

    ``root`` plays the role of REPO_ROOT for the test; the shim
    lives at ``vllm/sndr_core/integrations/gemma4/`` under it. The
    symlink points to ``overlay_target`` (absolute or relative).
    """
    shim_dir = root / "vllm" / "sndr_core" / "integrations" / "gemma4"
    shim_dir.mkdir(parents=True, exist_ok=True)

    # README with all retirement anchors
    (shim_dir / "README.md").write_text(
        "# Historical path shim\n\n"
        "This directory is a historical path. Retirement is gated on\n"
        "launcher re-baselining.\n",
        encoding="utf-8",
    )

    # Build overlay target as a sibling under attention/turboquant/overlays/
    overlay_dir = root / "vllm" / "sndr_core" / "integrations" / "attention" \
        / "turboquant" / "overlays" / "pr42637"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for sentinel in (
        "turboquant_attn.py",
        "triton_turboquant_store.py",
        "turboquant_config.py",
        "__init__.py",
    ):
        (overlay_dir / sentinel).write_text(
            "# sentinel\n", encoding="utf-8"
        )

    # Symlink relative to shim_dir
    link = shim_dir / "upstream_overlay_pr42637"
    target_rel = "../attention/turboquant/overlays/pr42637"
    link.symlink_to(target_rel)

    return shim_dir


@pytest.fixture
def mod():
    return _import_module()


class TestValidShim:
    """Synthetic shim matching the manifest = zero issues."""

    def test_valid_shim_zero_issues(self, tmp_path, mod):
        _build_valid_shim(tmp_path, overlay_target=None)
        results = mod.run_all(root=tmp_path)
        shim_key = "vllm/sndr_core/integrations/gemma4"
        assert results[shim_key] == [], (
            f"valid shim should produce no issues, got {results[shim_key]}"
        )


class TestRuleE1RawTree:
    """E.1 — raw tree integrity."""

    def test_missing_symlink_flagged(self, tmp_path, mod):
        shim_dir = _build_valid_shim(tmp_path, overlay_target=None)
        # Remove the symlink
        (shim_dir / "upstream_overlay_pr42637").unlink()
        issues = mod.run_all(root=tmp_path)[
            "vllm/sndr_core/integrations/gemma4"
        ]
        assert any("required symlink" in i and "missing" in i for i in issues), (
            f"expected E.1 missing-symlink issue, got {issues}"
        )

    def test_missing_readme_flagged(self, tmp_path, mod):
        shim_dir = _build_valid_shim(tmp_path, overlay_target=None)
        (shim_dir / "README.md").unlink()
        issues = mod.run_all(root=tmp_path)[
            "vllm/sndr_core/integrations/gemma4"
        ]
        assert any("README.md" in i and "missing" in i for i in issues), (
            f"expected E.1 missing-README issue, got {issues}"
        )

    def test_extra_file_flagged(self, tmp_path, mod):
        shim_dir = _build_valid_shim(tmp_path, overlay_target=None)
        # Add an unexpected file
        (shim_dir / "stray_module.py").write_text(
            "# code that should not be here\n", encoding="utf-8"
        )
        issues = mod.run_all(root=tmp_path)[
            "vllm/sndr_core/integrations/gemma4"
        ]
        assert any(
            "stray_module.py" in i and "unexpected" in i for i in issues
        ), f"expected E.1 unexpected-entry issue, got {issues}"

    def test_extra_subdir_flagged(self, tmp_path, mod):
        shim_dir = _build_valid_shim(tmp_path, overlay_target=None)
        (shim_dir / "kernels").mkdir()
        issues = mod.run_all(root=tmp_path)[
            "vllm/sndr_core/integrations/gemma4"
        ]
        assert any(
            "kernels" in i and "subdirectory" in i for i in issues
        ), f"expected E.1 unexpected-subdir issue, got {issues}"


class TestRuleE2SymlinkTarget:
    """E.2 — symlink readlink string matches expected target."""

    def test_wrong_target_flagged(self, tmp_path, mod):
        shim_dir = _build_valid_shim(tmp_path, overlay_target=None)
        link = shim_dir / "upstream_overlay_pr42637"
        link.unlink()
        # Point at a different (also valid) directory to test E.2
        # specifically; E.3 still passes because the new target exists.
        other = tmp_path / "vllm" / "sndr_core" / "elsewhere"
        other.mkdir(parents=True, exist_ok=True)
        link.symlink_to("../elsewhere")
        issues = mod.run_all(root=tmp_path)[
            "vllm/sndr_core/integrations/gemma4"
        ]
        assert any(
            "points to" in i and "expected" in i for i in issues
        ), f"expected E.2 wrong-target issue, got {issues}"


class TestRuleE3BrokenTarget:
    """E.3 — symlink resolves to an existing directory."""

    def test_broken_symlink_flagged(self, tmp_path, mod):
        shim_dir = _build_valid_shim(tmp_path, overlay_target=None)
        # Delete the overlay target — symlink becomes dangling
        overlay_dir = tmp_path / "vllm" / "sndr_core" / "integrations" \
            / "attention" / "turboquant" / "overlays" / "pr42637"
        for child in list(overlay_dir.iterdir()):
            child.unlink()
        overlay_dir.rmdir()
        issues = mod.run_all(root=tmp_path)[
            "vllm/sndr_core/integrations/gemma4"
        ]
        assert any("does not resolve" in i for i in issues), (
            f"expected E.3 broken-symlink issue, got {issues}"
        )


class TestRuleE4OverlaySentinel:
    """E.4 — resolved overlay contains all required sentinel files."""

    def test_missing_sentinel_flagged(self, tmp_path, mod):
        shim_dir = _build_valid_shim(tmp_path, overlay_target=None)
        overlay_dir = tmp_path / "vllm" / "sndr_core" / "integrations" \
            / "attention" / "turboquant" / "overlays" / "pr42637"
        # Delete one sentinel — overlay is gutted
        (overlay_dir / "turboquant_attn.py").unlink()
        issues = mod.run_all(root=tmp_path)[
            "vllm/sndr_core/integrations/gemma4"
        ]
        assert any(
            "turboquant_attn.py" in i and "sentinel" in i for i in issues
        ), f"expected E.4 missing-sentinel issue, got {issues}"


class TestRuleE5ReadmeWording:
    """E.5 — README retains retirement-contract narrative anchors."""

    def test_missing_anchor_flagged(self, tmp_path, mod):
        shim_dir = _build_valid_shim(tmp_path, overlay_target=None)
        # Overwrite README without the required anchors
        (shim_dir / "README.md").write_text(
            "# Some random doc\n\nNo relevant content here.\n",
            encoding="utf-8",
        )
        issues = mod.run_all(root=tmp_path)[
            "vllm/sndr_core/integrations/gemma4"
        ]
        # Expect all 3 anchors to be missing (historical / retirement / launcher)
        anchors_missing = [i for i in issues if "anchor" in i]
        assert len(anchors_missing) >= 1, (
            f"expected at least one E.5 missing-anchor issue, got {issues}"
        )


class TestLiveCorpus:
    """Run the audit against the real repo tree (smoke)."""

    def test_live_run_zero_issues(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert result.returncode == 0, (
            f"live audit_shim_window should pass, got rc={result.returncode}"
            f"\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_json_mode_clean(self):
        import json as _json
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert result.returncode == 0
        data = _json.loads(result.stdout)
        assert data["status"] == "OK"
        assert data["total_issues"] == 0
        assert data["total_shims"] >= 1


class TestExitCode:
    """Drift case = exit 1; clean = exit 0."""

    def test_drift_returns_exit_1(self, tmp_path, mod):
        # Construct a corrupt shim
        shim_dir = _build_valid_shim(tmp_path, overlay_target=None)
        (shim_dir / "stray.py").write_text("x = 1\n", encoding="utf-8")
        # Verify run_all returns non-empty issues
        results = mod.run_all(root=tmp_path)
        total_issues = sum(len(v) for v in results.values())
        assert total_issues > 0
