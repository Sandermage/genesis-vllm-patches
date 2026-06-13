# SPDX-License-Identifier: Apache-2.0
"""Tests for the G4_81 frozen tuned-tile table wiring in the G4_08
K-pad MoE GEMM kernel module (#45126 sweep transfer, chunk-2 Theme C).

Contract (TDD, written before the implementation):

  * ``DEFAULT_TILE_CONFIG`` == the kernel's historical literals
    ``(64, 64, 64, 1, 4, 2)`` — GROUP_SIZE_M=1 reduces the PID swizzle
    to the original row-major order, so default behavior is unchanged.
  * Table lookup is OFF unless ``GENESIS_ENABLE_G4_81_KPAD_TUNED_TILES``
    is set (Genesis convention: every behavior change is env-gated).
  * M-bucketing is identical to ``tools/triton_gemm_sweep.py`` (and to
    upstream vllm#45126): ``min(max(32, next_power_of_2(M)), 1024)``.
  * Missing table entry / unknown arch → DEFAULT_TILE_CONFIG (fail-open).
  * ``g4_kpad_moe_gemm`` accepts an explicit ``tile_config`` override
    (the sweep harness injection point).
  * ``validate_tile_config`` rejects malformed configs loudly.

These tests are CPU-only: the Triton kernel itself is validated on the
rig by the sweep harness's bit-identical gate.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

try:
    import torch  # noqa: F401 — kernel module imports torch at top level
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

pytestmark = pytest.mark.skipif(not _TORCH_OK, reason="torch not installed")

REPO_ROOT = Path(__file__).resolve().parents[4]
ENV_FLAG = "GENESIS_ENABLE_G4_81_KPAD_TUNED_TILES"


def _mod():
    from sndr.engines.vllm.patches.model_compat.gemma4.kernels import (
        g4_kpad_moe_gemm_triton as m,
    )
    return m


def test_default_tile_config_matches_historical_literals():
    m = _mod()
    assert m.DEFAULT_TILE_CONFIG == (64, 64, 64, 1, 4, 2)


def test_table_lookup_disabled_without_env(monkeypatch):
    m = _mod()
    monkeypatch.delenv(ENV_FLAG, raising=False)
    monkeypatch.setattr(
        m, "G4_KPAD_TUNED_TILES", {(8, 6): {(32, True): (64, 128, 64, 8, 4, 3)}}
    )
    monkeypatch.setattr(m, "_device_capability", lambda: (8, 6))
    assert m._get_tile_config(4, 2880) == m.DEFAULT_TILE_CONFIG


def test_table_lookup_enabled_with_env(monkeypatch):
    m = _mod()
    monkeypatch.setenv(ENV_FLAG, "1")
    monkeypatch.setattr(
        m, "G4_KPAD_TUNED_TILES", {(8, 6): {(32, True): (64, 128, 64, 8, 4, 3)}}
    )
    monkeypatch.setattr(m, "_device_capability", lambda: (8, 6))
    # M=4 → bucket 32; N=2880 < 8192 → small_n True.
    assert m._get_tile_config(4, 2880) == (64, 128, 64, 8, 4, 3)
    # Missing bucket → fail-open to default.
    assert m._get_tile_config(4096, 2880) == m.DEFAULT_TILE_CONFIG


def test_unknown_arch_falls_back_to_default(monkeypatch):
    m = _mod()
    monkeypatch.setenv(ENV_FLAG, "1")
    monkeypatch.setattr(
        m, "G4_KPAD_TUNED_TILES", {(8, 6): {(32, True): (64, 128, 64, 8, 4, 3)}}
    )
    monkeypatch.setattr(m, "_device_capability", lambda: None)
    assert m._get_tile_config(4, 2880) == m.DEFAULT_TILE_CONFIG
    monkeypatch.setattr(m, "_device_capability", lambda: (9, 0))
    assert m._get_tile_config(4, 2880) == m.DEFAULT_TILE_CONFIG


def test_m_bucket_parity_with_sweep_harness():
    m = _mod()
    tool_path = REPO_ROOT / "tools" / "triton_gemm_sweep.py"
    spec = importlib.util.spec_from_file_location("tgs_parity", tool_path)
    tool = importlib.util.module_from_spec(spec)
    sys.modules["tgs_parity"] = tool
    assert spec.loader is not None  # nosec
    spec.loader.exec_module(tool)
    for M in (1, 2, 4, 8, 31, 32, 33, 48, 64, 200, 300, 768, 1024, 1500, 4096):
        assert m._m_bucket(M) == tool.m_bucket(M), f"bucket mismatch at M={M}"


def test_validate_tile_config():
    m = _mod()
    m.validate_tile_config(m.DEFAULT_TILE_CONFIG)  # must not raise
    with pytest.raises(ValueError, match="BLOCK_M is locked"):
        m.validate_tile_config((128, 64, 64, 1, 4, 2))
    with pytest.raises(ValueError, match="BLOCK_N"):
        m.validate_tile_config((64, 96, 64, 1, 4, 2))  # non-power-of-2
    with pytest.raises(ValueError, match="GROUP_SIZE_M"):
        m.validate_tile_config((64, 64, 64, 0, 4, 2))
    with pytest.raises(ValueError, match="num_warps"):
        m.validate_tile_config((64, 64, 64, 1, 3, 2))  # not a power of 2
    with pytest.raises(ValueError, match="num_stages"):
        m.validate_tile_config((64, 64, 64, 1, 4, 0))
    with pytest.raises(ValueError, match="6 fields"):
        m.validate_tile_config((64, 64, 64, 1, 4))  # wrong arity


def test_wrapper_accepts_tile_config_override():
    m = _mod()
    sig = inspect.signature(m.g4_kpad_moe_gemm)
    assert "tile_config" in sig.parameters
    assert sig.parameters["tile_config"].default is None
