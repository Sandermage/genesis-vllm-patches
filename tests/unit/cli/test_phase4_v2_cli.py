# SPDX-License-Identifier: Apache-2.0
"""Tests for Phase 4 (Roadmap V2) CLI surface — V2 layered config discovery.

Covers:
  - `sndr hardware list` / `show` — HardwareDef registry
  - `sndr profile list` / `show` / `diff` — ProfileDef registry
  - `sndr model list-v2` / `show` — ModelDef registry

The V1 surface (`sndr model list/pull`) is unchanged and tested elsewhere.
"""
from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout

import pytest


# ─── Fixtures / helpers ────────────────────────────────────────────────


def _run(module, attr: str, ns: argparse.Namespace) -> tuple[int, str]:
    """Invoke a `run_*` handler, capture stdout, return (rc, captured)."""
    fn = getattr(module, attr)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = fn(ns)
    return rc, buf.getvalue()


# ─── Hardware CLI ──────────────────────────────────────────────────────


class TestHardwareList:
    def test_text_mode_returns_zero(self):
        from sndr.cli.legacy import hardware as hw_cli
        ns = argparse.Namespace(json=False)
        rc, out = _run(hw_cli, "run_list", ns)
        assert rc == 0
        assert "V2 HardwareDef registry" in out
        # Known V2 hardware files surface in the listing.
        assert "a5000-2x-24gbvram-16cpu-128gbram" in out
        assert "a5000-1x-24gbvram-16cpu-128gbram" in out
        assert "single-3090-24gbvram" in out
        # Summary line includes total count.
        assert "Total:" in out

    def test_json_mode_emits_array(self):
        from sndr.cli.legacy import hardware as hw_cli
        ns = argparse.Namespace(json=True)
        rc, out = _run(hw_cli, "run_list", ns)
        assert rc == 0
        payload = json.loads(out)
        assert "hardware" in payload
        assert isinstance(payload["hardware"], list)
        ids = {h["id"] for h in payload["hardware"]}
        assert "a5000-2x-24gbvram-16cpu-128gbram" in ids
        # No errors expected on the committed registry.
        assert payload["errors"] == []


class TestHardwareShow:
    def test_show_valid_id(self):
        from sndr.cli.legacy import hardware as hw_cli
        ns = argparse.Namespace(
            hw_id="a5000-2x-24gbvram-16cpu-128gbram", json=False
        )
        rc, out = _run(hw_cli, "run_show", ns)
        assert rc == 0
        assert "n_gpus:" in out
        assert "Sizing" in out
        assert "Runtime" in out
        # System env section is rendered.
        assert "PYTORCH_CUDA_ALLOC_CONF" in out

    def test_show_unknown_id_returns_two(self):
        from sndr.cli.legacy import hardware as hw_cli
        ns = argparse.Namespace(hw_id="does-not-exist", json=False)
        rc, _ = _run(hw_cli, "run_show", ns)
        assert rc == 2

    def test_show_json_mode(self):
        from sndr.cli.legacy import hardware as hw_cli
        ns = argparse.Namespace(
            hw_id="a5000-2x-24gbvram-16cpu-128gbram", json=True
        )
        rc, out = _run(hw_cli, "run_show", ns)
        assert rc == 0
        payload = json.loads(out)
        assert payload["id"] == "a5000-2x-24gbvram-16cpu-128gbram"
        assert payload["hardware"]["n_gpus"] == 2


# ─── Profile CLI ───────────────────────────────────────────────────────


class TestProfileList:
    def test_text_mode_returns_zero(self):
        from sndr.cli.legacy import profile as prof_cli
        ns = argparse.Namespace(model=None, json=False)
        rc, out = _run(prof_cli, "run_list", ns)
        assert rc == 0
        assert "ProfileDef registry" in out
        # Known V2 profiles surface.
        assert "wave9-balanced" in out
        assert "wave9-qwen3.6-27b-tq-k8v4" in out
        assert "qa-27b-fp8kv-tested" in out

    def test_filter_by_model(self):
        from sndr.cli.legacy import profile as prof_cli
        ns = argparse.Namespace(model="qwen3.6-35b-a3b-fp8", json=False)
        rc, out = _run(prof_cli, "run_list", ns)
        assert rc == 0
        # Only the 35B-fp8 wave9-balanced profile matches this filter.
        assert "wave9-balanced" in out
        # 27B profiles must not appear under this filter.
        assert "wave9-qwen3.6-27b-tq-k8v4" not in out

    def test_filter_no_match(self):
        from sndr.cli.legacy import profile as prof_cli
        ns = argparse.Namespace(model="qwen3.6-nonexistent", json=False)
        rc, out = _run(prof_cli, "run_list", ns)
        assert rc == 0
        assert "no V2 profile files found" in out

    def test_json_mode(self):
        from sndr.cli.legacy import profile as prof_cli
        ns = argparse.Namespace(model=None, json=True)
        rc, out = _run(prof_cli, "run_list", ns)
        assert rc == 0
        payload = json.loads(out)
        ids = {p["id"] for p in payload["profiles"]}
        assert "wave9-balanced" in ids
        # Sizing-override flag is surfaced.
        assert all("has_sizing_override" in p for p in payload["profiles"])


class TestProfileShow:
    def test_show_valid_profile(self):
        from sndr.cli.legacy import profile as prof_cli
        ns = argparse.Namespace(profile_id="wave9-balanced", json=False)
        rc, out = _run(prof_cli, "run_show", ns)
        assert rc == 0
        assert "parent_model:" in out
        assert "qwen3.6-35b-a3b-fp8" in out
        # wave9-balanced has empty delta — message should reflect that.
        assert "empty" in out.lower()

    def test_show_unknown_profile_returns_two(self):
        from sndr.cli.legacy import profile as prof_cli
        ns = argparse.Namespace(profile_id="does-not-exist", json=False)
        rc, _ = _run(prof_cli, "run_show", ns)
        assert rc == 2


class TestProfileDiff:
    def test_diff_empty_profile_no_changes(self):
        """wave9-balanced has empty patches_delta — diff should be 0/0/0."""
        from sndr.cli.legacy import profile as prof_cli
        ns = argparse.Namespace(profile_id="wave9-balanced", json=True)
        rc, out = _run(prof_cli, "run_diff", ns)
        assert rc == 0
        payload = json.loads(out)
        assert payload["canonical_count"] == payload["merged_count"]
        assert payload["added"] == []
        assert payload["removed"] == []
        assert payload["changed"] == []

    def test_diff_qa_tested_disables_patches(self):
        """qa-27b-fp8kv-tested disables 16 patches — diff should show them."""
        from sndr.cli.legacy import profile as prof_cli
        ns = argparse.Namespace(profile_id="qa-27b-fp8kv-tested", json=True)
        rc, out = _run(prof_cli, "run_diff", ns)
        assert rc == 0
        payload = json.loads(out)
        assert len(payload["removed"]) == 16
        # Specific patches we know are disabled:
        removed_keys = {r["key"] for r in payload["removed"]}
        assert "GENESIS_ENABLE_PN90_PROBABILISTIC_DRAFT" in removed_keys
        assert "GENESIS_ENABLE_PN16_LAZY_REASONER" in removed_keys

    def test_diff_dflash_ab_adds_patches(self):
        """experimental-qwen3.6-27b-tq-dflash-ab enables 6 patches (5 DFlash + P100)."""
        from sndr.cli.legacy import profile as prof_cli
        ns = argparse.Namespace(
            profile_id="experimental-qwen3.6-27b-tq-dflash-ab", json=True
        )
        rc, out = _run(prof_cli, "run_diff", ns)
        assert rc == 0
        payload = json.loads(out)
        assert len(payload["added"]) == 6
        added_keys = {a["key"] for a in payload["added"]}
        assert "GENESIS_ENABLE_PN21_DFLASH_SWA" in added_keys
        assert "GENESIS_ENABLE_PN40_DFLASH_OMNIBUS" in added_keys


# ─── Model V2 CLI ──────────────────────────────────────────────────────


class TestModelListV2:
    def test_text_mode_returns_zero(self):
        from sndr.cli.legacy import model as m_cli
        ns = argparse.Namespace(json=False)
        rc, out = _run(m_cli, "run_list_v2", ns)
        assert rc == 0
        assert "V2 ModelDef registry" in out
        assert "qwen3.6-35b-a3b-fp8" in out
        assert "qwen3.6-27b-int4-autoround-tq-k8v4" in out
        # Architecture + dtype + kv summary surface.
        assert "hybrid_gdn_moe" in out
        assert "turboquant_k8v4" in out or "fp16" in out

    def test_json_mode_summary_shape(self):
        from sndr.cli.legacy import model as m_cli
        ns = argparse.Namespace(json=True)
        rc, out = _run(m_cli, "run_list_v2", ns)
        assert rc == 0
        payload = json.loads(out)
        assert "models" in payload
        # Every summary has the shape we promised.
        required = {"id", "title", "attention_arch", "patches_count",
                    "min_gpu_count", "min_total_vram_mib"}
        for m in payload["models"]:
            assert required.issubset(m.keys())


class TestModelShow:
    def test_show_35b_canonical(self):
        from sndr.cli.legacy import model as m_cli
        ns = argparse.Namespace(model_id="qwen3.6-35b-a3b-fp8", json=False)
        rc, out = _run(m_cli, "run_show", ns)
        assert rc == 0
        # Identity block.
        assert "/models/Qwen3.6-35B-A3B-FP8" in out
        # Capabilities block.
        assert "spec_decode.method:    mtp" in out
        assert "kv_cache_dtype:        turboquant_k8v4" in out
        # Patches dump.
        assert "Canonical patches: 44 entries" in out

    def test_show_unknown_model_returns_two(self):
        from sndr.cli.legacy import model as m_cli
        ns = argparse.Namespace(model_id="qwen-nonexistent", json=False)
        rc, _ = _run(m_cli, "run_show", ns)
        assert rc == 2

    def test_show_json_mode_full_dataclass(self):
        from sndr.cli.legacy import model as m_cli
        ns = argparse.Namespace(model_id="qwen3.6-35b-a3b-fp8", json=True)
        rc, out = _run(m_cli, "run_show", ns)
        assert rc == 0
        payload = json.loads(out)
        # Top-level fields.
        assert payload["id"] == "qwen3.6-35b-a3b-fp8"
        # Nested capabilities + patches present.
        assert payload["capabilities"]["attention_arch"] == "hybrid_gdn_moe"
        assert isinstance(payload["patches"], dict)


# ─── Registration smoke ────────────────────────────────────────────────


class TestRegistration:
    """Phase 4 wires hardware + profile into the top-level argparser."""

    def test_hardware_argparser_registers(self):
        import argparse
        from sndr.cli.legacy.hardware import add_argparser
        p = argparse.ArgumentParser()
        sub = p.add_subparsers()
        add_argparser(sub)
        # Parsing `hardware list` resolves cleanly.
        ns = p.parse_args(["hardware", "list"])
        assert ns.hardware_cmd == "list"

    def test_profile_argparser_registers(self):
        import argparse
        from sndr.cli.legacy.profile import add_argparser
        p = argparse.ArgumentParser()
        sub = p.add_subparsers()
        add_argparser(sub)
        ns = p.parse_args(["profile", "diff", "wave9-balanced"])
        assert ns.profile_cmd == "diff"
        assert ns.profile_id == "wave9-balanced"

    def test_model_list_v2_subcommand(self):
        import argparse
        from sndr.cli.legacy.model import add_argparser
        p = argparse.ArgumentParser()
        sub = p.add_subparsers()
        add_argparser(sub)
        ns = p.parse_args(["model", "list-v2", "--json"])
        assert ns.model_cmd == "list-v2"
        assert ns.json is True

    def test_top_level_argparser_includes_new_commands(self):
        # Phase 4 wires hardware + profile registrators into cli/__init__.py.
        # Verify both import handles exist at module-load time.
        from sndr import cli as cli_mod
        assert hasattr(cli_mod, "_hardware_argparser")
        assert hasattr(cli_mod, "_profile_argparser")
        assert callable(cli_mod._hardware_argparser)
        assert callable(cli_mod._profile_argparser)
