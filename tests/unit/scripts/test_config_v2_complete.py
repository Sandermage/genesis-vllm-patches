# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/config_v2_complete.py` — V2 hardware auto-completer
(Entry 23).

Contract:

  • Clean YAML (already canonical) → status=CLEAN, no diff
  • Drifted YAML in --check mode → status=WOULD_WRITE, file unchanged
  • Drifted YAML in --write mode → status=WRITTEN, file actually rewritten,
    second run on same file is idempotent (status=CLEAN)
  • After --write, `audit_launch_coverage` reports zero drift
  • Original entries + comments survive intact; new entries carry
    the `# E23 auto-added` marker
  • Missing `mounts:` or `system_env:` anchor → status=ERROR (no write)
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "config_v2_complete.py"
AUDIT_SCRIPT = REPO_ROOT / "scripts" / "audit_launch_coverage.py"


def _import_completer():
    name = "_config_v2_complete_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _import_audit():
    name = "_audit_launch_coverage_for_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, AUDIT_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── Helpers ───────────────────────────────────────────────────────────


def _drifted_yaml() -> str:
    """V2 hardware YAML missing 3 mounts (only models + hf_cache present)
    and 4 env keys."""
    return textwrap.dedent("""
        schema_version: 2
        kind: hardware
        id: synth-drifted

        runtime:
          docker:
            mounts:
              - "${models_dir}:/models:ro"
              - "${hf_cache}:/root/.cache/huggingface:ro"

        system_env:
          PYTORCH_CUDA_ALLOC_CONF: 'expandable_segments:True'
          OMP_NUM_THREADS: '1'
          VLLM_NO_USAGE_STATS: '1'
    """).lstrip("\n")


def _canonical_yaml() -> str:
    """All 5 mounts + 7 env keys present."""
    return textwrap.dedent("""
        schema_version: 2
        kind: hardware
        id: synth-canonical

        runtime:
          docker:
            mounts:
              - "${models_dir}:/models:ro"
              - "${hf_cache}:/root/.cache/huggingface:ro"
              - "${triton_cache}:/root/.triton/cache"
              - "${compile_cache}:/root/.cache/vllm/torch_compile_cache"
              - "${plugin_src}:/plugin:ro"

        system_env:
          PYTORCH_CUDA_ALLOC_CONF: 'expandable_segments:True'
          OMP_NUM_THREADS: '1'
          CUDA_DEVICE_MAX_CONNECTIONS: '8'
          TRITON_CACHE_DIR: '/root/.triton/cache'
          VLLM_ALLOW_LONG_MAX_MODEL_LEN: '1'
          VLLM_WORKER_MULTIPROC_METHOD: spawn
          VLLM_NO_USAGE_STATS: '1'
    """).lstrip("\n")


def _write(p: Path, text: str) -> Path:
    p.write_text(text, encoding="utf-8")
    return p


# ─── Anchor-block detection ───────────────────────────────────────────


class TestFindAnchorBlock:
    def test_finds_mounts_block(self):
        mod = _import_completer()
        lines = _drifted_yaml().splitlines()
        result = mod._find_anchor_block(lines, "mounts")
        assert result is not None
        anchor, last_item, indent = result
        assert lines[anchor].strip().startswith("mounts:")
        # last_item should be one of the existing mount entries.
        assert lines[last_item].lstrip().startswith("- ")
        # Indent for new entries should match existing items.
        assert indent == 6   # `      - ...` (2 + 2 + 2)

    def test_finds_system_env_block(self):
        mod = _import_completer()
        lines = _drifted_yaml().splitlines()
        result = mod._find_anchor_block(lines, "system_env")
        assert result is not None
        anchor, last_item, indent = result
        assert lines[anchor].strip().startswith("system_env:")
        assert indent == 2

    def test_missing_anchor_returns_none(self):
        mod = _import_completer()
        lines = "foo: 1\nbar: 2\n".splitlines()
        assert mod._find_anchor_block(lines, "mounts") is None


# ─── Single-YAML completion ───────────────────────────────────────────


class TestCompleteOneYaml:
    def test_canonical_returns_clean(self, tmp_path):
        mod = _import_completer()
        p = _write(tmp_path / "h.yaml", _canonical_yaml())
        r = mod.complete_one_yaml(p, write=False)
        assert r.status == mod.CompletionStatus.CLEAN
        assert r.missing_mounts == []
        assert r.missing_envs == []
        # File should be unchanged.
        assert p.read_text() == _canonical_yaml()

    def test_drift_check_mode_does_not_write(self, tmp_path):
        mod = _import_completer()
        original = _drifted_yaml()
        p = _write(tmp_path / "h.yaml", original)
        r = mod.complete_one_yaml(p, write=False)
        assert r.status == mod.CompletionStatus.WOULD_WRITE
        assert len(r.missing_mounts) == 3
        assert len(r.missing_envs) == 4
        # File unchanged.
        assert p.read_text() == original
        # Diff is populated.
        assert r.diff
        assert "E23 auto-added" in r.diff

    def test_drift_write_mode_rewrites_file(self, tmp_path):
        mod = _import_completer()
        p = _write(tmp_path / "h.yaml", _drifted_yaml())
        r = mod.complete_one_yaml(p, write=True)
        assert r.status == mod.CompletionStatus.WRITTEN
        new_text = p.read_text()
        # Original markers survive.
        assert "synth-drifted" in new_text
        # New entries injected with the canonical marker.
        assert "E23 auto-added" in new_text
        assert "${triton_cache}:/root/.triton/cache" in new_text
        assert "${plugin_src}:/plugin:ro" in new_text
        assert "TRITON_CACHE_DIR" in new_text

    def test_idempotent_after_write(self, tmp_path):
        mod = _import_completer()
        p = _write(tmp_path / "h.yaml", _drifted_yaml())
        first = mod.complete_one_yaml(p, write=True)
        assert first.status == mod.CompletionStatus.WRITTEN
        # Second pass on the now-canonical file → CLEAN, no diff.
        second = mod.complete_one_yaml(p, write=True)
        assert second.status == mod.CompletionStatus.CLEAN
        assert second.missing_mounts == []
        assert second.missing_envs == []

    def test_after_write_audit_passes(self, tmp_path):
        completer = _import_completer()
        audit = _import_audit()
        p = _write(tmp_path / "h.yaml", _drifted_yaml())
        completer.complete_one_yaml(p, write=True)
        r = audit.audit_one_hardware_yaml(p)
        assert r.passed is True, f"missing_mounts={r.missing_mounts}, missing_envs={r.missing_envs}"

    def test_preserves_existing_comments(self, tmp_path):
        mod = _import_completer()
        with_comment = textwrap.dedent("""
            schema_version: 2
            kind: hardware
            id: synth-drifted-comment

            runtime:
              docker:
                mounts:
                  # operator comment must survive
                  - "${models_dir}:/models:ro"  # inline operator comment
                  - "${hf_cache}:/root/.cache/huggingface:ro"

            system_env:
              PYTORCH_CUDA_ALLOC_CONF: 'expandable_segments:True'
              OMP_NUM_THREADS: '1'
              VLLM_NO_USAGE_STATS: '1'
        """).lstrip("\n")
        p = _write(tmp_path / "h.yaml", with_comment)
        mod.complete_one_yaml(p, write=True)
        new = p.read_text()
        assert "operator comment must survive" in new
        assert "inline operator comment" in new

    def test_missing_mounts_anchor_yields_error(self, tmp_path):
        mod = _import_completer()
        bad = textwrap.dedent("""
            schema_version: 2
            kind: hardware
            id: synth-no-mounts
            runtime:
              docker:
                image: foo
            system_env:
              PYTORCH_CUDA_ALLOC_CONF: 'x'
        """).lstrip("\n")
        p = _write(tmp_path / "h.yaml", bad)
        r = mod.complete_one_yaml(p, write=False)
        assert r.status == mod.CompletionStatus.ERROR
        assert "mounts:" in r.error

    def test_dataclass_status_values(self):
        mod = _import_completer()
        assert mod.CompletionStatus.CLEAN.value == "clean"
        assert mod.CompletionStatus.WOULD_WRITE.value == "would_write"
        assert mod.CompletionStatus.WRITTEN.value == "written"
        assert mod.CompletionStatus.ERROR.value == "error"


# ─── Live committed repo — must already be canonical ──────────────────


class TestLiveRepo:
    def test_all_committed_hardware_clean(self):
        """Entry 22 brought the committed V2 hardware YAMLs to canonical
        state. Running the completer in check mode on the live repo
        must return CLEAN for every file."""
        mod = _import_completer()
        results = mod.complete_directory()
        for r in results:
            assert r.status == mod.CompletionStatus.CLEAN, (
                f"{r.hardware_id} not canonical: "
                f"mounts={r.missing_mounts}, envs={r.missing_envs}"
            )
        assert len(results) >= 3


# ─── Script CLI ────────────────────────────────────────────────────────


class TestScriptCLI:
    def test_cli_exits_zero_on_committed_repo(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout

    def test_cli_json_shape(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert "total" in payload
        assert "clean" in payload
        assert "would_write" in payload
        assert "written" in payload
        assert payload["would_write"] == 0
        assert payload["errors"] == 0

    def test_cli_synth_drift_check_mode_exit_1(self, tmp_path):
        fp = _write(tmp_path / "synth.yaml", _drifted_yaml())
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--file", str(fp), "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 1
        payload = json.loads(result.stdout)
        assert payload["would_write"] == 1

    def test_cli_synth_drift_write_mode_exit_0(self, tmp_path):
        fp = _write(tmp_path / "synth.yaml", _drifted_yaml())
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--file", str(fp), "--write", "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert payload["written"] == 1
        # File was actually rewritten.
        new = fp.read_text()
        assert "E23 auto-added" in new
        assert "${triton_cache}:/root/.triton/cache" in new
