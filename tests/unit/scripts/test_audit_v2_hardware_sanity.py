# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_v2_hardware_sanity.py` — Entry 30."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_v2_hardware_sanity.py"


def _import():
    name = "_audit_v2_hardware_sanity_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_hw(*, cc=[8, 6], n_gpus=2, vram=22000, gmu=0.9,
             seqs=2, batch_toks=4096) -> str:
    return textwrap.dedent(f"""
        schema_version: 2
        kind: hardware
        id: synth
        hardware:
          cuda_capability_min: {cc}
          n_gpus: {n_gpus}
          min_vram_per_gpu_mib: {vram}
        sizing:
          gpu_memory_utilization: {gmu}
          max_num_seqs: {seqs}
          max_num_batched_tokens: {batch_toks}
    """).lstrip("\n")


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "hw.yaml"
    p.write_text(body, encoding="utf-8")
    return p


class TestCanonical:
    def test_canonical_passes(self, tmp_path):
        mod = _import()
        p = _write(tmp_path, _make_hw())
        r = mod.check_one_hardware(p)
        assert r.passed is True


class TestViolations:
    def test_cuda_not_list(self, tmp_path):
        mod = _import()
        p = _write(tmp_path, _make_hw(cc="not-a-list"))
        r = mod.check_one_hardware(p)
        assert r.passed is False
        assert any("cuda_capability_min" in v for v in r.violations)

    def test_cuda_major_out_of_range(self, tmp_path):
        mod = _import()
        p = _write(tmp_path, _make_hw(cc=[99, 0]))
        r = mod.check_one_hardware(p)
        assert r.passed is False

    def test_n_gpus_zero_fails(self, tmp_path):
        mod = _import()
        p = _write(tmp_path, _make_hw(n_gpus=0))
        r = mod.check_one_hardware(p)
        assert r.passed is False
        assert any("n_gpus" in v for v in r.violations)

    def test_vram_too_low_fails(self, tmp_path):
        mod = _import()
        p = _write(tmp_path, _make_hw(vram=4000))   # 4 GiB < 8 GiB min
        r = mod.check_one_hardware(p)
        assert r.passed is False

    def test_gmu_zero_fails(self, tmp_path):
        mod = _import()
        p = _write(tmp_path, _make_hw(gmu=0.0))
        r = mod.check_one_hardware(p)
        assert r.passed is False

    def test_gmu_over_one_fails(self, tmp_path):
        mod = _import()
        p = _write(tmp_path, _make_hw(gmu=1.5))
        r = mod.check_one_hardware(p)
        assert r.passed is False

    def test_gmu_exactly_one_passes(self, tmp_path):
        """1.0 is inclusive — operator may want to push to limit."""
        mod = _import()
        p = _write(tmp_path, _make_hw(gmu=1.0))
        r = mod.check_one_hardware(p)
        assert r.passed is True

    def test_max_num_seqs_zero_fails(self, tmp_path):
        mod = _import()
        p = _write(tmp_path, _make_hw(seqs=0))
        r = mod.check_one_hardware(p)
        assert r.passed is False

    def test_cross_field_usable_vram_low_fails(self, tmp_path):
        """High vram but very low gmu — usable too small for any model."""
        mod = _import()
        p = _write(tmp_path, _make_hw(vram=10_000, gmu=0.3))   # 3000 < 4000
        r = mod.check_one_hardware(p)
        assert r.passed is False
        assert any("usable VRAM" in v for v in r.violations)


class TestLiveRepo:
    def test_committed_hardware_clean(self):
        mod = _import()
        results = mod.audit_v2_hardware_sanity()
        failed = [r for r in results if not r.passed]
        assert failed == [], "\n".join(
            f"  {r.hardware_id}: {r.violations}"
            for r in failed
        )
        assert len(results) == 3


class TestScriptCLI:
    def test_cli_zero(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_cli_json(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert "bounds" in payload
        assert payload["failed"] == 0
