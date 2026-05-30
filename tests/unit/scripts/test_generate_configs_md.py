# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/generate_configs_md.py` — config inventory generator.

Contract:

  1. parse_yaml_top_level extracts TOP_FIELDS scalar values from YAML.
  2. parse_yaml_top_level extracts reference_metrics block.
  3. parse_yaml_top_level computes mtp_k from spec_decode block.
  4. parse_yaml_top_level counts enabled GENESIS_ENABLE_* patches.
  5. Quoted YAML values (single + double) are stripped.
  6. render_markdown produces a sorted, lifecycle-grouped report.
  7. Live repo: --check passes (CONFIGS_AUTO.md in sync).
  8. main exits 0 on stdout/write; exit 1 on --check divergence.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_configs_md.py"


def _import_script():
    name = "_generate_configs_md_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── parse_yaml_top_level scalar extraction ───────────────────────────


class TestParseTopLevelScalars:
    def test_extracts_top_fields(self, tmp_path):
        mod = _import_script()
        f = tmp_path / "model.yaml"
        f.write_text(
            "key: my-model\n"
            "title: Test Model\n"
            "lifecycle: stable\n"
            "max_model_len: 131072\n"
            "max_num_seqs: 4\n"
        )
        result = mod.parse_yaml_top_level(f)
        assert result["key"] == "my-model"
        assert result["title"] == "Test Model"
        assert result["lifecycle"] == "stable"
        assert result["max_model_len"] == "131072"
        assert result["max_num_seqs"] == "4"

    def test_strips_single_quotes(self, tmp_path):
        mod = _import_script()
        f = tmp_path / "m.yaml"
        f.write_text("key: 'quoted-key'\n")
        result = mod.parse_yaml_top_level(f)
        assert result["key"] == "quoted-key"

    def test_strips_double_quotes(self, tmp_path):
        mod = _import_script()
        f = tmp_path / "m.yaml"
        f.write_text('key: "double-quoted"\n')
        result = mod.parse_yaml_top_level(f)
        assert result["key"] == "double-quoted"

    def test_missing_field_yields_none(self, tmp_path):
        mod = _import_script()
        f = tmp_path / "m.yaml"
        f.write_text("key: just-key\n")
        result = mod.parse_yaml_top_level(f)
        assert result["title"] is None
        assert result["lifecycle"] is None

    def test_stores_file_basename(self, tmp_path):
        mod = _import_script()
        f = tmp_path / "named.yaml"
        f.write_text("key: x\n")
        result = mod.parse_yaml_top_level(f)
        assert result["_file"] == "named.yaml"


# ─── reference_metrics block ──────────────────────────────────────────


class TestReferenceMetrics:
    def test_metrics_extracted(self, tmp_path):
        mod = _import_script()
        f = tmp_path / "m.yaml"
        f.write_text(
            "key: x\n"
            "reference_metrics:\n"
            "  long_gen_sustained_tps: 211\n"
            "  decode_tpot_ms: 4.5\n"
            "  ttft_ms: 90\n"
            "  tool_call_score: 7/7\n"
            "  stability_cv_pct: 5\n"
            "non_metric: y\n"
        )
        result = mod.parse_yaml_top_level(f)
        assert result["metric_long_gen_sustained_tps"] == "211"
        assert result["metric_decode_tpot_ms"] == "4.5"
        assert result["metric_tool_call_score"] == "7/7"

    def test_missing_block_yields_nones(self, tmp_path):
        mod = _import_script()
        f = tmp_path / "m.yaml"
        f.write_text("key: x\n")
        result = mod.parse_yaml_top_level(f)
        for field in mod.METRIC_FIELDS:
            assert result[f"metric_{field}"] is None


# ─── MTP K extraction ─────────────────────────────────────────────────


class TestMtpK:
    def test_mtp_k_extracted(self, tmp_path):
        mod = _import_script()
        f = tmp_path / "m.yaml"
        f.write_text(
            "key: x\n"
            "spec_decode:\n"
            "  method: mtp\n"
            "  num_speculative_tokens: 3\n"
        )
        result = mod.parse_yaml_top_level(f)
        assert result["mtp_k"] == "3"

    def test_ngram_method_returns_string(self, tmp_path):
        mod = _import_script()
        f = tmp_path / "m.yaml"
        f.write_text(
            "key: x\n"
            "spec_decode:\n"
            "  method: ngram\n"
        )
        result = mod.parse_yaml_top_level(f)
        assert result["mtp_k"] == "ngram"

    def test_no_spec_decode_yields_none(self, tmp_path):
        mod = _import_script()
        f = tmp_path / "m.yaml"
        f.write_text("key: x\n")
        result = mod.parse_yaml_top_level(f)
        assert result["mtp_k"] is None


# ─── Patch count ──────────────────────────────────────────────────────


class TestEnabledPatchCount:
    def test_counts_enabled_only(self, tmp_path):
        mod = _import_script()
        f = tmp_path / "m.yaml"
        f.write_text(
            "key: x\n"
            "genesis_env:\n"
            "  GENESIS_ENABLE_PN1: '1'\n"
            "  GENESIS_ENABLE_PN2: '1'\n"
            "  GENESIS_ENABLE_PN3: '0'\n"  # disabled, not counted
        )
        result = mod.parse_yaml_top_level(f)
        assert result["enabled_patches"] == 2

    def test_zero_when_no_patches(self, tmp_path):
        mod = _import_script()
        f = tmp_path / "m.yaml"
        f.write_text("key: x\n")
        result = mod.parse_yaml_top_level(f)
        assert result["enabled_patches"] == 0


# ─── Render markdown structure ────────────────────────────────────────


class TestRenderMarkdown:
    def test_render_includes_total_count(self):
        mod = _import_script()
        configs = [
            {"key": "a", "lifecycle": "stable", "_file": "a.yaml",
             "enabled_patches": 5},
            {"key": "b", "lifecycle": "experimental", "_file": "b.yaml",
             "enabled_patches": 3},
        ]
        # Fill in default Nones to mimic parse output
        for c in configs:
            for f in mod.TOP_FIELDS + ["mtp_k"]:
                c.setdefault(f, None)
            for f in mod.METRIC_FIELDS:
                c.setdefault(f"metric_{f}", None)
        out = mod.render_markdown(configs)
        assert "Total configs: **2**" in out
        assert "stable" in out
        assert "experimental" in out

    def test_render_sorts_by_lifecycle(self):
        """Lifecycle order: stable, tested, experimental, community-test,
        retired, ?"""
        mod = _import_script()
        configs = [
            {"key": "z-retired", "lifecycle": "retired",
             "_file": "z.yaml", "enabled_patches": 0},
            {"key": "a-stable", "lifecycle": "stable",
             "_file": "a.yaml", "enabled_patches": 0},
            {"key": "m-experimental", "lifecycle": "experimental",
             "_file": "m.yaml", "enabled_patches": 0},
        ]
        for c in configs:
            for f in mod.TOP_FIELDS + ["mtp_k"]:
                c.setdefault(f, None)
            for f in mod.METRIC_FIELDS:
                c.setdefault(f"metric_{f}", None)
        out = mod.render_markdown(configs)
        idx_stable = out.index("`a-stable`")
        idx_exp = out.index("`m-experimental`")
        idx_ret = out.index("`z-retired`")
        assert idx_stable < idx_exp < idx_ret


# ─── Live regression anchor ────────────────────────────────────────────


class TestLive:
    def test_check_passes_on_live_repo(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--check"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, (
            f"--check failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_stdout_emits_markdown(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--stdout"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "Genesis vLLM Patches" in result.stdout
        assert "Total configs:" in result.stdout
