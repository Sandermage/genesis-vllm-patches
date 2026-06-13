# SPDX-License-Identifier: Apache-2.0
"""Tests for ``tools/triton_gemm_sweep.py`` — offline Triton GEMM tuning sweep.

TDD contract (written BEFORE the implementation, #45126 sweep transfer,
roadmap chunk-2 Theme C):

  * M-bucketing must be byte-compatible with upstream vllm#45126
    (``min(max(32, next_power_of_2(M)), 1024)`` + ``is_small_n = N < 8192``).
  * sm_86 arch profile (RTX A5000) with the PN345-aligned 99 KiB opt-in
    shared-memory budget; candidates exceeding it are pruned offline.
  * Candidate space includes the PID-swizzle axis (``group_size_m``) and
    always contains the target's default config first.
  * ``--dry-run`` must be fully torch-free AND triton-free (the harness
    plans on this laptop, runs on the rig).
  * Frozen-table emission: per-arch dict literal keyed ``(cc_major,
    cc_minor)`` → ``{(m_bucket, is_small_n): 6-tuple}`` that
    ``ast.literal_eval`` can parse; non-bit-identical winners are
    excluded unless ``allow_tolerance`` is passed (the #45126
    bit-identical gate).
"""
from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "triton_gemm_sweep.py"


def _import_tool():
    spec = importlib.util.spec_from_file_location("triton_gemm_sweep", TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["triton_gemm_sweep"] = mod
    assert spec.loader is not None  # nosec
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _import_tool()


# ─── PR #45126 bucketing parity ──────────────────────────────────────


def test_next_power_of_2(tool):
    assert tool.next_power_of_2(1) == 1
    assert tool.next_power_of_2(2) == 2
    assert tool.next_power_of_2(3) == 4
    assert tool.next_power_of_2(32) == 32
    assert tool.next_power_of_2(33) == 64
    assert tool.next_power_of_2(1500) == 2048


def test_m_bucket_matches_pr45126_semantics(tool):
    # min(max(32, next_power_of_2(M)), 1024) — the #45126 table key.
    assert tool.m_bucket(1) == 32
    assert tool.m_bucket(8) == 32
    assert tool.m_bucket(32) == 32
    assert tool.m_bucket(33) == 64
    assert tool.m_bucket(48) == 64
    assert tool.m_bucket(200) == 256
    assert tool.m_bucket(300) == 512
    assert tool.m_bucket(1024) == 1024
    assert tool.m_bucket(4096) == 1024


# ─── Arch profile: sm_86 (RTX A5000) ─────────────────────────────────


def test_sm86_profile_present(tool):
    prof = tool.ARCH_PROFILES["sm_86"]
    assert prof.cc == (8, 6)
    # 99 KiB opt-in max shared memory per block on GA102/sm_86 —
    # must match the PN345 pruning budget.
    assert prof.shared_mem_bytes == 99 * 1024


# ─── Candidate space ─────────────────────────────────────────────────


def test_g4_target_registered(tool):
    assert "g4_kpad_moe_gemm" in tool.TARGETS
    target = tool.TARGETS["g4_kpad_moe_gemm"]
    # Default config must mirror the live kernel literals
    # (BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, GROUP_SIZE_M=1 row-major,
    # num_warps=4, num_stages=2).
    assert target.default_config.as_tuple() == (64, 64, 64, 1, 4, 2)
    # BLOCK_M is locked: expert-segment alignment assumption.
    assert "block_m" in target.locked_axes


def test_candidate_space_default_first_and_swizzle_axis(tool):
    target = tool.TARGETS["g4_kpad_moe_gemm"]
    cands = tool.candidate_space(target)
    assert cands[0] == target.default_config
    assert len(cands) == len(set(cands)), "candidate space must be deduplicated"
    group_sizes = {c.group_size_m for c in cands}
    assert 1 in group_sizes, "row-major (GROUP_SIZE_M=1) baseline must be swept"
    assert any(g > 1 for g in group_sizes), "PID-swizzle axis missing"
    # Locked axis respected.
    assert {c.block_m for c in cands} == {64}


def test_shared_mem_pruning(tool):
    prof = tool.ARCH_PROFILES["sm_86"]
    elt = 2  # fp16/bf16 staging
    small = tool.TileCandidate(64, 64, 64, 1, 4, 2)
    fat = tool.TileCandidate(64, 256, 128, 8, 8, 5)
    assert tool.estimate_shared_mem_bytes(small, elt) <= prof.shared_mem_bytes
    assert tool.estimate_shared_mem_bytes(fat, elt) > prof.shared_mem_bytes
    kept, pruned = tool.prune_candidates([small, fat], prof, elt)
    assert small in kept
    assert fat in pruned


# ─── Shape validation (torch-free) ───────────────────────────────────


def test_g4_shapes_valid(tool):
    target = tool.TARGETS["g4_kpad_moe_gemm"]
    tool.validate_shapes(target)  # must not raise
    # The Gemma-4-26B-A4B TP=2 down_proj K-pad case is the whole point.
    assert any(
        s.k_real == 352 and s.k_padded == 384 for s in target.shapes
    ), "missing the K=352→384 Gemma-4 26B-A4B TP=2 shape"


def test_validate_shapes_rejects_bad_padding(tool):
    bad = tool.SweepShape(name="bad", n=704, k_real=352, k_padded=320)
    target = tool.TARGETS["g4_kpad_moe_gemm"]
    broken = tool.replace_shapes(target, [bad])
    with pytest.raises(ValueError, match="k_padded"):
        tool.validate_shapes(broken)


# ─── Dry-run: torch-free guarantee ───────────────────────────────────


def test_dry_run_is_torch_and_triton_free():
    """--dry-run must complete with torch AND triton import-blocked."""
    code = textwrap.dedent(
        f"""
        import importlib.abc, importlib.util, sys

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("torch", "triton"):
                    raise ImportError("blocked import: " + name)
                return None

        sys.meta_path.insert(0, Blocker())
        spec = importlib.util.spec_from_file_location(
            "tgs", {str(TOOL_PATH)!r})
        mod = importlib.util.module_from_spec(spec)
        sys.modules["tgs"] = mod
        spec.loader.exec_module(mod)
        rc = mod.main(["--target", "g4_kpad_moe_gemm",
                       "--arch", "sm_86", "--dry-run"])
        sys.exit(rc)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "DRY-RUN" in proc.stdout


def test_dry_run_plan_counts(tool):
    target = tool.TARGETS["g4_kpad_moe_gemm"]
    prof = tool.ARCH_PROFILES["sm_86"]
    plan = tool.build_plan(target, prof)
    assert plan["target"] == "g4_kpad_moe_gemm"
    assert plan["arch"] == "sm_86"
    assert plan["candidates_kept"] >= 1
    assert plan["candidates_kept"] + plan["candidates_pruned"] == plan[
        "candidates_total"
    ]
    assert plan["m_values"], "tuning M grid must be non-empty"
    assert plan["off_grid_m_values"], "off-grid validation Ms must be non-empty"
    # Off-grid validation points must not leak into the tuning grid.
    assert not set(plan["m_values"]) & set(plan["off_grid_m_values"])


# ─── Frozen table emission ───────────────────────────────────────────


def _synthetic_results():
    return {
        "tool": "triton_gemm_sweep v1",
        "target": "g4_kpad_moe_gemm",
        "arch": "sm_86",
        "table_name": "G4_KPAD_TUNED_TILES",
        "records": [
            {
                "m_bucket": 32,
                "is_small_n": True,
                "config": [64, 64, 64, 8, 4, 3],
                "bitwise": True,
                "geomean_speedup": 1.12,
                "points": 4,
            },
            {
                "m_bucket": 64,
                "is_small_n": True,
                "config": [64, 128, 32, 1, 4, 2],
                "bitwise": False,
                "geomean_speedup": 1.30,
                "points": 4,
            },
        ],
    }


def _parse_table(emitted: str) -> dict:
    # Emission is "<NAME> = { ... }" preceded by comment lines (which may
    # themselves contain "=") — split at the assignment line.
    lines = emitted.splitlines()
    start = next(
        i for i, ln in enumerate(lines)
        if not ln.lstrip().startswith("#") and "= {" in ln
    )
    rhs = "\n".join(lines[start:]).split("=", 1)[1]
    return ast.literal_eval(rhs.strip())


def test_emit_frozen_table_bitwise_gate(tool):
    out = tool.emit_frozen_table(_synthetic_results())
    assert "G4_KPAD_TUNED_TILES" in out
    table = _parse_table(out)
    assert (8, 6) in table
    inner = table[(8, 6)]
    assert inner[(32, True)] == (64, 64, 64, 8, 4, 3)
    # Non-bit-identical record must be EXCLUDED by default.
    assert (64, True) not in inner


def test_emit_frozen_table_allow_tolerance(tool):
    out = tool.emit_frozen_table(_synthetic_results(), allow_tolerance=True)
    table = _parse_table(out)
    inner = table[(8, 6)]
    assert inner[(64, True)] == (64, 128, 32, 1, 4, 2)
    assert "NOT bit-identical" in out


def test_emit_table_cli(tool, tmp_path, capsys):
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(_synthetic_results()))
    rc = tool.main(["--emit-table", str(results_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "G4_KPAD_TUNED_TILES" in out


# ─── CLI errors ──────────────────────────────────────────────────────


def test_cli_unknown_target_is_invocation_error(tool):
    rc = tool.main(["--target", "no_such_kernel", "--dry-run"])
    assert rc == 2


def test_cli_unknown_arch_is_invocation_error(tool):
    rc = tool.main(["--target", "g4_kpad_moe_gemm", "--arch", "sm_999", "--dry-run"])
    assert rc == 2
