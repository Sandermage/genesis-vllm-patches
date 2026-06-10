# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr memory` CLI — T1.3.

Argparse round-trip + handler smoke tests. Heavy lifting (math
correctness, reading config.json) is covered in
`tests/unit/runtime/test_memory_estimator.py`; these tests verify the
CLI surface (filter wiring, JSON output shape, exit codes).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from sndr.cli.legacy import memory as M


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sndr-test")
    sub = parser.add_subparsers()
    M.add_argparser(sub)
    return parser


# ─── Argparse registration ──────────────────────────────────────────────


class TestArgparser:
    def test_parses_explain_with_overrides(self):
        p = _make_parser()
        ns = p.parse_args([
            "memory", "explain", "prod-qwen3.6-35b-balanced",
            "--ctx", "128k", "--seqs", "4",
            "--kv-dtype", "fp8_e5m2", "--gpu-vram", "48",
            "--json",
        ])
        assert ns.preset == "prod-qwen3.6-35b-balanced"
        assert ns.ctx == "128k"
        assert ns.seqs == 4
        assert ns.kv_dtype == "fp8_e5m2"
        assert ns.gpu_vram == "48"
        assert ns.json is True

    def test_parses_simulate_args(self):
        p = _make_parser()
        ns = p.parse_args([
            "memory", "simulate",
            "--model", "/tmp/fake",
            "--ctx", "32k", "--sequences", "2",
            "--tp-size", "2",
        ])
        assert ns.model == "/tmp/fake"
        assert ns.ctx == "32k"
        assert ns.sequences == "2"
        assert ns.tp_size == "2"

    def test_doctor_accepts_json(self):
        p = _make_parser()
        ns = p.parse_args(["memory", "doctor", "--json"])
        assert ns.json is True


# ─── helpers ────────────────────────────────────────────────────────────


class TestHelpers:
    def test_parse_ctx_handles_k_suffix(self):
        assert M._parse_ctx("128k") == 128 * 1024
        assert M._parse_ctx("256K") == 256 * 1024
        assert M._parse_ctx("32768") == 32768
        assert M._parse_ctx("0.5m") == int(0.5 * 1024 * 1024)

    def test_gib_to_bytes(self):
        assert M._gib_to_bytes("24") == 24 * (1 << 30)
        assert M._gib_to_bytes("48 GiB") == 48 * (1 << 30)
        assert M._gib_to_bytes("96gb") == 96 * (1 << 30)


# ─── `sndr memory explain` ──────────────────────────────────────────────


class TestExplain:
    def test_unknown_preset_exits_2(self, capsys):
        ns = argparse.Namespace(
            preset="totally-fake-preset", gpu_vram=None, ctx=None,
            seqs=None, kv_dtype=None, json=True,
        )
        with pytest.raises(SystemExit) as excinfo:
            M._run_explain(ns)
        assert excinfo.value.code == 2

    def test_real_preset_json_output(self, capsys):
        ns = argparse.Namespace(
            preset="prod-qwen3.6-35b-balanced", gpu_vram=None, ctx=None,
            seqs=None, kv_dtype=None, json=True,
        )
        rc = M._run_explain(ns)
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        # V2 alias `prod-qwen3.6-35b-balanced` resolves through load_alias
        # → composed triplet key (model--hardware--profile form), not the
        # alias name itself. Phase 10 (2026-06-01) migrated this test from
        # V1 `a5000-2x-35b-prod` to V2 alias; the JSON `preset` field now
        # carries the composed key. Assert by substring so V2 internals
        # (hardware naming, separator format) can evolve.
        assert "qwen3-6-35b-balanced" in data["preset"]
        assert "components" in data
        assert "utilization" in data
        # At least 4 components (weights, KV, activations, CUDA graph)
        assert len(data["components"]) >= 4

    def test_gpu_vram_override_changes_utilization(self, capsys):
        # First with default vram
        ns = argparse.Namespace(
            preset="prod-qwen3.6-35b-balanced", gpu_vram=None, ctx=None,
            seqs=None, kv_dtype=None, json=True,
        )
        M._run_explain(ns)
        default_util = json.loads(capsys.readouterr().out)["utilization"]

        # Now override to 96 GiB (Blackwell-class)
        ns96 = argparse.Namespace(
            preset="prod-qwen3.6-35b-balanced", gpu_vram="96", ctx=None,
            seqs=None, kv_dtype=None, json=True,
        )
        M._run_explain(ns96)
        big_util = json.loads(capsys.readouterr().out)["utilization"]
        # Bigger GPU → lower utilization for same workload
        assert big_util <= default_util

    def test_ctx_override_changes_kv_estimate(self, capsys):
        ns = argparse.Namespace(
            preset="prod-qwen3.6-35b-balanced", gpu_vram=None,
            ctx="64k", seqs=None, kv_dtype=None, json=True,
        )
        M._run_explain(ns)
        small = json.loads(capsys.readouterr().out)

        ns2 = argparse.Namespace(
            preset="prod-qwen3.6-35b-balanced", gpu_vram=None,
            ctx="256k", seqs=None, kv_dtype=None, json=True,
        )
        M._run_explain(ns2)
        big = json.loads(capsys.readouterr().out)
        # Bigger context → bigger total (or the same if KV estimate is 0)
        assert big["total_bytes"] >= small["total_bytes"]


# ─── `sndr memory simulate` ─────────────────────────────────────────────


class TestSimulate:
    def test_requires_model(self, capsys):
        ns = argparse.Namespace(
            model=None, ctx=None, sequences=None,
            tp_size=None, kv_dtype=None,
            gpu_vram=None, gpu_name=None, json=True,
        )
        with pytest.raises(SystemExit) as excinfo:
            M._run_simulate(ns)
        assert excinfo.value.code == 2

    def test_simulate_with_synthetic_model(self, tmp_path, capsys):
        cfg = {
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 11008,
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        ns = argparse.Namespace(
            model=str(tmp_path), ctx="32k", sequences="2",
            tp_size="2", kv_dtype="fp8_e5m2",
            gpu_vram="24", gpu_name=None, json=True,
        )
        rc = M._run_simulate(ns)
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["preset"] == "(simulate)"
        assert data["gpu_count"] == 2
        # KV component present
        kv_comps = [c for c in data["components"] if "KV cache" in c["name"]]
        assert len(kv_comps) == 1
        assert kv_comps[0]["bytes"] > 0


# ─── `sndr memory doctor` ───────────────────────────────────────────────


class TestDoctor:
    def test_doctor_json_lists_presets(self, capsys):
        # Phase 10 Step 4 (2026-06-01): V1 monolithic preset tier 100%
        # retired. `sndr memory doctor` iterates only the V1 registry
        # (V2 doctor surface is a separate effort — out of Phase 10
        # scope). With V1 empty, doctor returns 0 rows; the rendering
        # contract (every row has preset + utilization|error) still
        # holds vacuously over the empty list.
        ns = argparse.Namespace(json=True)
        rc = M._run_doctor(ns)
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert "presets" in data
        for row in data["presets"]:
            assert "preset" in row
            assert "utilization" in row or "error" in row

    def test_doctor_text_table_renders(self, capsys):
        ns = argparse.Namespace(json=False)
        rc = M._run_doctor(ns)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Preset" in out and ("Util%" in out or "GPUs" in out)


# ─── component / estimate dict serializers ──────────────────────────────


class TestSerializers:
    def test_component_to_dict_keys(self):
        from sndr.runtime.memory_estimator import MemoryComponent
        c = MemoryComponent(
            "test", 1024 * 1024, notes="n", confidence="high",
        )
        d = M._component_to_dict(c)
        assert d == {
            "name": "test", "bytes": 1024 * 1024, "human": "1 MiB",
            "confidence": "high", "notes": "n",
        }

    def test_estimate_to_dict_includes_total_and_util(self):
        from sndr.runtime.memory_estimator import (
            MemoryEstimate, MemoryComponent,
        )
        est = MemoryEstimate(
            preset_key="x", model_path="/fake",
            gpu_count=1, gpu_vram_bytes=24 * (1 << 30),
            components=(MemoryComponent("a", 1 * (1 << 30)),),
        )
        d = M._estimate_to_dict(est)
        assert d["total_bytes"] == 1 * (1 << 30)
        assert d["utilization"] == round(1 / 24, 4)
        assert isinstance(d["components"], list)


# ─── Phase 4.7 MVP: --ctx-sweep mode ──────────────────────────────────


class TestCtxSweep:
    def test_parse_ctx_sweep_basic(self):
        out = M._parse_ctx_sweep("4096,16384,65536")
        assert out == [4096, 16384, 65536]

    def test_parse_ctx_sweep_with_k_suffix(self):
        out = M._parse_ctx_sweep("4k,16k,64k,128k")
        assert out == [4096, 16384, 65536, 131072]

    def test_parse_ctx_sweep_whitespace_tolerated(self):
        out = M._parse_ctx_sweep(" 4k , 16k ,, 64k ")
        assert out == [4096, 16384, 65536]

    def test_parse_ctx_sweep_empty_raises(self):
        with pytest.raises(ValueError, match="at least one context size"):
            M._parse_ctx_sweep("")

    def test_argparser_accepts_ctx_sweep(self):
        p = _make_parser()
        ns = p.parse_args([
            "memory", "explain", "prod-qwen3.6-35b-balanced",
            "--ctx-sweep", "4k,16k,64k", "--json",
        ])
        assert ns.ctx_sweep == "4k,16k,64k"
        assert ns.json is True

    def test_run_explain_sweep_emits_rows(self, capsys):
        """Real preset + sweep → JSON with N rows + verdict per row."""
        import argparse
        opts = argparse.Namespace(
            preset="prod-qwen3.6-35b-balanced",
            gpu_vram=None, ctx=None, seqs=None, kv_dtype=None,
            ctx_sweep="4096,16384,65536",
            json=True,
        )
        rc = M._run_explain(opts)
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["preset"] == "prod-qwen3.6-35b-balanced"
        assert payload["ctx_sweep"] == [4096, 16384, 65536]
        assert len(payload["rows"]) == 3
        for row in payload["rows"]:
            # UNKNOWN is the correct verdict when the dev box has no
            # readable safetensors for prod-qwen3.6-35b-balanced — the estimator's critical
            # components (Model weights / KV cache) come back zero-byte
            # low-confidence and refusing to claim safety is the whole
            # point of P0.4. The other three verdicts are valid when
            # safetensors are present.
            assert row["verdict"] in {"SAFE", "TIGHT", "OOM_RISK", "UNKNOWN"}
            assert row["median_mib"] >= 0
            assert row["p95_mib"] >= row["median_mib"]
            assert row["worst_mib"] >= row["p95_mib"]
            assert row["budget_mib"] > 0
        # rc=0 unless any verdict is OOM_RISK
        assert rc == 0

    def test_run_explain_sweep_text_mode(self, capsys):
        import argparse
        opts = argparse.Namespace(
            preset="prod-qwen3.6-35b-balanced",
            gpu_vram=None, ctx=None, seqs=None, kv_dtype=None,
            ctx_sweep="4096,16384",
            json=False,
        )
        rc = M._run_explain(opts)
        out = capsys.readouterr().out
        assert "ctx-sweep" in out
        assert "verdict" in out
        # Each ctx value appears in output
        assert "4096" in out
        assert "16384" in out
        assert rc == 0
