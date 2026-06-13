#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline Triton GEMM tuning sweep — frozen per-arch tile tables (sm_86).

Transfer of the vllm#45126 tuning recipe (roadmap chunk-2 Theme C,
2026-06-11 50-PR sweep) to Genesis-owned Triton GEMMs. The upstream PR
("NVIDIA-tuned tile configs + PID swizzling for triton_scaled_mm")
established the methodology this harness reproduces:

  1. OFFLINE sweep of (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M,
     num_warps, num_stages) over representative shapes — never
     ``@triton.autotune`` at serving time (compile/cudagraph-safe by
     construction, and no run-to-run winner jitter).
  2. BIT-IDENTICAL gate: every candidate's output is asserted equal to
     the baseline config's output BEFORE it may be timed/selected.
  3. FROZEN PER-ARCH DICT: winners are emitted as a static Python dict
     keyed ``(cc_major, cc_minor)`` → ``{(m_bucket, is_small_n):
     (bm, bn, bk, group_m, warps, stages)}`` with the exact #45126
     bucketing (``m_bucket = min(max(32, next_power_of_2(M)), 1024)``,
     ``is_small_n = N < 8192``). Config selection at serving time is a
     cached pure-Python lookup.
  4. PID SWIZZLE axis: L2-friendly grouped program ordering
     (``GROUP_SIZE_M``); ``GROUP_SIZE_M=1`` reduces exactly to the
     original row-major order, so the baseline is always in the space.
  5. OFF-GRID validation Ms (never seen by the tuning sweep) verify the
     bucketed table generalizes — #45126 used 48/200/300/768/1500.

================================================================
DORMANCY PROBE — does the pin's triton_scaled_mm ever fire here?
================================================================

Verified 2026-06-11 against the PRISTINE pin tree
``0.22.1rc1.dev259+g303916e93`` (/private/tmp/candidate_pin_current/vllm):

``triton_scaled_mm`` (the #45126 target, defined in
``model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py``)
has exactly TWO call sites in the pin:

  a. ``vllm/_custom_ops.py:901`` — runtime fallback inside
     ``cutlass_scaled_mm``, taken only when ``current_platform.is_rocm()``
     or the weight is not 16-aligned (``b.shape[0] % 16 != 0 or
     b.shape[1] % 16 != 0``). All callers live in
     ``model_executor/kernels/linear/scaled_mm/cutlass.py`` (the
     CUTLASS INT8/FP8 ScaledMM linear kernels).
  b. ``model_executor/kernels/linear/scaled_mm/triton.py:134`` —
     ``TritonInt8ScaledMMLinearKernel.apply_weights``. Reachable only
     when ``choose_scaled_mm_linear_kernel`` (``kernels/linear/
     __init__.py:450``) is forced to ``linear_backend == "triton"``
     (line 498) — on CUDA the priority list is
     ``[CutlassInt8ScaledMMLinearKernel, TritonInt8ScaledMMLinearKernel]``
     and CutlassInt8 ``is_supported``/``can_implement`` return
     unconditionally True on CUDA.

Both sites require an INT8 W8A8 (``Int8ScaledMMLinearLayerConfig``)
or CUTLASS-FP8 checkpoint. Our fleet has NONE:

  * Qwen3.6-35B-A3B FP8 — block-scaled FP8: MarlinFP8/
    ``w8a8_triton_block_scaled_mm`` class (the P81 kernel, fp8_utils.py
    — a DIFFERENT Triton GEMM), never triton_scaled_mm.
  * Qwen3.6-27B int4 AutoRound — W4A16 Marlin.
  * Gemma-4-26B-A4B AWQ / Gemma-4-31B AWQ — W4A16 Marlin (+ the
    Genesis G4_08 K-pad Triton MoE GEMM for K=352).

The Genesis 287-patch overlay also never references
``triton_scaled_mm`` (0 grep hits in ``sndr/``). VERDICT: **DORMANT on
our stack — do not vendor #45126's tables; transfer its methodology**
(this harness). That is exactly the roadmap call.

================================================================
WIRED TARGET: g4_kpad_moe_gemm (G4_08 / G4_81)
================================================================

``sndr/engines/vllm/patches/model_compat/gemma4/kernels/
g4_kpad_moe_gemm_triton.py`` — the Genesis K-pad MoE GEMM that routes
the Marlin-rejected K%64≠0 down_proj of Gemma-4-26B-A4B at TP=2
(K_real=352 → K_padded=384, N=hidden_size). The kernel hardcoded
(64, 64, 64) × 4 warps × 2 stages; this harness sweeps the remaining
axes and the table is consumed via ``G4_KPAD_TUNED_TILES`` in the
kernel module, env-gated by ``GENESIS_ENABLE_G4_81_KPAD_TUNED_TILES``
(registry ID G4_81).

BLOCK_M is LOCKED at 64: the kernel reads one expert id per M-tile
("all rows in a sorted MoE block share one expert"), so M-tile
boundaries must coincide with the caller's expert-segment alignment.
Sweeping BLOCK_M would silently break that contract.

Equivalence classes (fp32 accumulator — unlike #45126's int32, fp
addition is NOT associative):
  * GROUP_SIZE_M / num_warps / num_stages — pure scheduling, expected
    bit-identical; gate = ``torch.equal`` (hard assert).
  * BLOCK_N — same per-element K-reduction order; expected
    bit-identical, the gate verifies on real hardware.
  * BLOCK_K — CHANGES the K-tile reduction grouping → not bit-identical
    in general. Such candidates are rejected by default; pass
    ``--allow-tolerance`` to gate them against the torch reference
    (``g4_kpad_moe_gemm_reference``) at the target's atol instead.
    Tolerance-gated entries are marked in the emitted table.

================================================================
CONSUMING THE FROZEN TABLES — the PN362 hookup
================================================================

Two consumption modes for the emitted static tables:

  * MODE A (dict-lookup kernels — this harness's primary output):
    paste the emitted ``G4_KPAD_TUNED_TILES`` literal into the kernel
    module. Selection is a cached pure-Python lookup at launch — no
    Autotuner object exists, so determinism is by construction (the
    same class as #45126 upstream). This kills the "199-vs-228 wall_TPS"
    autotune-jitter class for Genesis-owned kernels.

  * MODE B (``@triton.autotune`` kernels we do NOT own — FLA GDN,
    fused_moe, lmcache): reorder the kernel's ``configs=[...]`` so the
    swept winner is FIRST in declaration order, then run with
    ``VLLM_TRITON_FORCE_FIRST_CONFIG=1`` — Genesis **PN362** (vendor of
    vllm#42425) replaces ``Autotuner.run`` to deterministically pick
    the first config that compiles. Sweep winner + PN362 = static table
    semantics WITHOUT forking the autotuned kernel. Composes with
    PN345 (shmem-aware pre-autotune pruner): this harness's sm_86
    shared-memory estimator uses the same 99 KiB opt-in budget, so
    emitted winners never OOR-walk at PN362 pick time. This is the
    "static tables via PN362 for GDN determinism" item from the
    roadmap synthesis.

================================================================
FOLLOW-UP: P81 RE-TUNE ON SM_86
================================================================

P81 (``p81_fp8_block_scaled_m_le_8``, backport of vllm#40925)
specializes ``w8a8_triton_block_scaled_mm`` for M<=8 with numbers tuned
on **GB10 (sm_121), not sm_86**. Follow-up: add a ``w8a8_block_scaled``
target to this harness and re-sweep on the A5000s; until then P81 is a
GB10-derived heuristic (see the note in its wiring docstring).

================================================================
USAGE
================================================================

Plan offline (this laptop — torch-free, validates shapes/candidates)::

    python3 tools/triton_gemm_sweep.py --target g4_kpad_moe_gemm \
        --arch sm_86 --dry-run

Run on the rig (CUDA + triton + torch required)::

    python3 tools/triton_gemm_sweep.py --target g4_kpad_moe_gemm \
        --arch sm_86 --hidden-size <hidden_size from checkpoint
        config.json — VERIFY, do not guess> --out results_sm86.json

Emit the frozen table from results (torch-free)::

    python3 tools/triton_gemm_sweep.py --emit-table results_sm86.json

Exit codes: 0 OK | 1 sweep/gate failure | 2 invocation error.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Methodology credit: vllm#45126 (tuned tiles + PID swizzle for
triton_scaled_mm); determinism hookup: vllm#42425 via Genesis PN362.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

# IMPORTANT: torch / triton / sndr are imported ONLY inside run_sweep()
# helpers. --dry-run and --emit-table must work on torch-free hosts
# (enforced by tests/unit/tools/test_triton_gemm_sweep.py).

TOOL_NAME = "triton_gemm_sweep v1"


# ─── #45126 bucketing (byte-compatible) ──────────────────────────────


def next_power_of_2(n: int) -> int:
    """Smallest power of two >= n (n >= 1). Mirrors triton.next_power_of_2."""
    if n < 1:
        raise ValueError(f"next_power_of_2 requires n >= 1, got {n}")
    return 1 << (n - 1).bit_length()


def m_bucket(M: int) -> int:
    """The #45126 table key: min(max(32, next_power_of_2(M)), 1024)."""
    return min(max(32, next_power_of_2(M)), 1024)


def is_small_n(N: int) -> bool:
    """The #45126 N-bucket: True when N < 8192."""
    return N < 8192


# ─── Arch profiles ───────────────────────────────────────────────────


@dataclass(frozen=True)
class ArchProfile:
    """Per-architecture sweep profile."""

    name: str
    cc: tuple[int, int]
    # Opt-in max dynamic shared memory per block, bytes. Pruning budget —
    # must match PN345's estimate for the same device class.
    shared_mem_bytes: int
    description: str


ARCH_PROFILES: dict[str, ArchProfile] = {
    "sm_86": ArchProfile(
        name="sm_86",
        cc=(8, 6),
        shared_mem_bytes=99 * 1024,  # GA102 opt-in max per block (PN345 budget)
        description=(
            "Ampere GA102 (RTX A5000, our 2x TP=2 rig). 100 KiB shared "
            "memory per SM, 99 KiB opt-in max per block; favors small "
            "tiles with shallow pipelines (cf. #45126 sm_89 findings)."
        ),
    ),
}


# ─── Candidate configs ───────────────────────────────────────────────


@dataclass(frozen=True, order=True)
class TileCandidate:
    """One sweep point: tile shape + PID swizzle + launch knobs.

    ``group_size_m=1`` reduces the grouped (L2-swizzled) program
    ordering to plain row-major — the pre-existing kernel behavior.
    """

    block_m: int
    block_n: int
    block_k: int
    group_size_m: int
    num_warps: int
    num_stages: int

    def as_tuple(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.block_m,
            self.block_n,
            self.block_k,
            self.group_size_m,
            self.num_warps,
            self.num_stages,
        )


@dataclass(frozen=True)
class SweepShape:
    """One GEMM shape. For plain GEMMs k_padded == k_real."""

    name: str
    n: int
    k_real: int
    k_padded: int


@dataclass(frozen=True)
class KernelTarget:
    """A sweepable Triton GEMM target."""

    name: str
    description: str
    # Bytes per element as staged into shared memory (fp16/bf16 = 2).
    elt_size: int
    default_config: TileCandidate
    # Axis grids. Locked axes are kept at the default config's value.
    block_m_values: tuple[int, ...]
    block_n_values: tuple[int, ...]
    block_k_values: tuple[int, ...]
    group_size_m_values: tuple[int, ...]
    num_warps_values: tuple[int, ...]
    num_stages_values: tuple[int, ...]
    locked_axes: dict[str, str]  # axis -> reason
    shapes: tuple[SweepShape, ...]
    m_values: tuple[int, ...]
    off_grid_m_values: tuple[int, ...]
    atol: float
    # Name of the frozen dict the emission produces.
    table_name: str
    # Lazy runner factory: imports torch/triton/sndr only when called.
    make_runner: Callable[[KernelTarget, argparse.Namespace], Any] | None = None


def replace_shapes(target: KernelTarget, shapes: list[SweepShape]) -> KernelTarget:
    """Return a copy of ``target`` with a different shape grid."""
    return replace(target, shapes=tuple(shapes))


def candidate_space(target: KernelTarget) -> list[TileCandidate]:
    """Cartesian product of the target's axis grids, locked axes pinned.

    The default config is always first; the list is deduplicated.
    """
    axes = {
        "block_m": target.block_m_values,
        "block_n": target.block_n_values,
        "block_k": target.block_k_values,
        "group_size_m": target.group_size_m_values,
        "num_warps": target.num_warps_values,
        "num_stages": target.num_stages_values,
    }
    for axis, _reason in target.locked_axes.items():
        axes[axis] = (getattr(target.default_config, axis),)

    seen: set[TileCandidate] = set()
    out: list[TileCandidate] = [target.default_config]
    seen.add(target.default_config)
    for bm, bn, bk, gm, nw, ns in itertools.product(
        axes["block_m"],
        axes["block_n"],
        axes["block_k"],
        axes["group_size_m"],
        axes["num_warps"],
        axes["num_stages"],
    ):
        cand = TileCandidate(bm, bn, bk, gm, nw, ns)
        if cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out


def estimate_shared_mem_bytes(c: TileCandidate, elt_size: int) -> int:
    """Conservative software-pipelining estimate (PN345-class):
    ``num_stages * (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) * elt_size``."""
    return c.num_stages * (c.block_m * c.block_k + c.block_k * c.block_n) * elt_size


def prune_candidates(
    candidates: list[TileCandidate],
    profile: ArchProfile,
    elt_size: int,
) -> tuple[list[TileCandidate], list[TileCandidate]]:
    """Split candidates into (kept, pruned) by the arch shmem budget."""
    kept, pruned = [], []
    for c in candidates:
        if estimate_shared_mem_bytes(c, elt_size) <= profile.shared_mem_bytes:
            kept.append(c)
        else:
            pruned.append(c)
    return kept, pruned


def validate_shapes(target: KernelTarget) -> None:
    """Torch-free structural validation of the target's shape grid."""
    if not target.shapes:
        raise ValueError(f"{target.name}: empty shape grid")
    for s in target.shapes:
        if s.n < 16 or s.k_real < 16:
            raise ValueError(f"{target.name}/{s.name}: N/K must be >= 16 (tl.dot)")
        if s.k_padded < s.k_real:
            raise ValueError(
                f"{target.name}/{s.name}: k_padded={s.k_padded} < k_real={s.k_real}"
            )
        if s.k_padded % 64 != 0:
            raise ValueError(
                f"{target.name}/{s.name}: k_padded={s.k_padded} not 64-aligned"
            )
    for M in (*target.m_values, *target.off_grid_m_values):
        if M < 1:
            raise ValueError(f"{target.name}: invalid M={M}")
    overlap = set(target.m_values) & set(target.off_grid_m_values)
    if overlap:
        raise ValueError(
            f"{target.name}: off-grid validation Ms leak into the tuning "
            f"grid: {sorted(overlap)}"
        )


# ─── Target registry ─────────────────────────────────────────────────


def _g4_default_shapes(hidden_size: int) -> tuple[SweepShape, ...]:
    """Gemma-4-26B-A4B TP=2 K-pad shapes (the G4_08 route).

    Only the down_proj GEMM has K%64 != 0 (K=352 -> padded 384); that is
    the only GEMM G4_08 routes through the kernel. N = hidden_size —
    READ IT FROM THE CHECKPOINT config.json ON THE RIG (--hidden-size);
    the default below is a planning placeholder, not a verified value.
    """
    return (
        SweepShape(
            name="down_proj_tp2",
            n=hidden_size,
            k_real=352,
            k_padded=384,
        ),
    )


G4_PLACEHOLDER_HIDDEN_SIZE = 2880


def _make_g4_kpad_runner(target: KernelTarget, args: argparse.Namespace):
    """Build the rig-side runner for g4_kpad_moe_gemm. Imports torch,
    triton and sndr — call only in run mode on a CUDA host."""
    import torch  # noqa: PLC0415

    from sndr.engines.vllm.patches.model_compat.gemma4.kernels.g4_kpad_moe_gemm_triton import (  # noqa: PLC0415,E501
        g4_kpad_moe_gemm,
        g4_kpad_moe_gemm_reference,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("g4_kpad_moe_gemm sweep requires a CUDA device")

    num_experts = args.num_experts
    block_m = target.default_config.block_m

    def make_inputs(shape: SweepShape, M: int, seed: int):
        g = torch.Generator(device="cuda").manual_seed(seed)
        a = torch.randn(M, shape.k_real, dtype=torch.float16, device="cuda", generator=g)
        w = torch.randn(
            num_experts, shape.n, shape.k_padded,
            dtype=torch.float16, device="cuda", generator=g,
        )
        # Zero the padding zone — the kernel's masking contract.
        w[:, :, shape.k_real:] = 0
        scales = torch.rand(
            num_experts, shape.n, dtype=torch.float16, device="cuda", generator=g
        ) + 0.5
        # Expert ids constant per BLOCK_M segment — the sorted-MoE
        # alignment contract the kernel assumes (BLOCK_M locked at 64).
        seg = torch.randint(
            0, num_experts, ((M + block_m - 1) // block_m,),
            device="cuda", generator=g,
        )
        expert_ids = seg.repeat_interleave(block_m)[:M].contiguous()
        return a, w, scales, expert_ids

    def run(shape: SweepShape, M: int, cand: TileCandidate, seed: int):
        a, w, scales, expert_ids = make_inputs(shape, M, seed)
        return g4_kpad_moe_gemm(
            a, w, scales, expert_ids,
            K_real=shape.k_real, num_bits=8,
            tile_config=cand.as_tuple(),
        )

    def reference(shape: SweepShape, M: int, seed: int):
        a, w, scales, expert_ids = make_inputs(shape, M, seed)
        return g4_kpad_moe_gemm_reference(
            a, w, scales, expert_ids, K_real=shape.k_real, num_bits=8
        )

    return run, reference


def _build_targets(hidden_size: int = G4_PLACEHOLDER_HIDDEN_SIZE) -> dict[str, KernelTarget]:
    return {
        "g4_kpad_moe_gemm": KernelTarget(
            name="g4_kpad_moe_gemm",
            description=(
                "Genesis G4_08 K-pad Triton MoE GEMM (Gemma-4-26B-A4B "
                "TP=2 down_proj, K=352 padded to 384). Frozen table "
                "consumed via G4_KPAD_TUNED_TILES, env-gated by "
                "GENESIS_ENABLE_G4_81_KPAD_TUNED_TILES (G4_81)."
            ),
            elt_size=2,  # fp16/bf16 staging
            default_config=TileCandidate(64, 64, 64, 1, 4, 2),
            block_m_values=(64,),
            block_n_values=(64, 128),
            block_k_values=(32, 64, 128),
            group_size_m_values=(1, 4, 8),
            num_warps_values=(2, 4, 8),
            num_stages_values=(2, 3, 4, 5),
            locked_axes={
                "block_m": (
                    "expert-segment alignment: the kernel reads ONE "
                    "expert id per M-tile; the G4_08 caller sorts and "
                    "aligns rows at 64-row granularity. Sweeping "
                    "BLOCK_M breaks that contract."
                ),
            },
            shapes=_g4_default_shapes(hidden_size),
            # Tuning grid: decode (M=1..8 with MTP K=3 verify at 4),
            # routed-prefill knee. top_k=8 multiplies token counts
            # post-routing, so the grid extends well past batch size.
            m_values=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048),
            # The #45126 off-grid generalization points.
            off_grid_m_values=(48, 200, 300, 768, 1500),
            atol=1e-2,  # the kernel's documented numerical bound
            table_name="G4_KPAD_TUNED_TILES",
            make_runner=_make_g4_kpad_runner,
        ),
        # FOLLOW-UP (PR sweep 2026-06-11): add a "w8a8_block_scaled"
        # target here to re-tune P81 (vllm#40925 numbers were tuned on
        # GB10 sm_121, not sm_86).
    }


TARGETS: dict[str, KernelTarget] = _build_targets()


# ─── Planning (torch-free) ───────────────────────────────────────────


def build_plan(
    target: KernelTarget,
    profile: ArchProfile,
    m_values: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Torch-free sweep plan: validates shapes, prunes candidates."""
    validate_shapes(target)
    cands = candidate_space(target)
    kept, pruned = prune_candidates(cands, profile, target.elt_size)
    ms = tuple(m_values) if m_values else target.m_values
    buckets = sorted({(m_bucket(M), is_small_n(s.n)) for M in ms for s in target.shapes})
    return {
        "tool": TOOL_NAME,
        "target": target.name,
        "arch": profile.name,
        "cc": list(profile.cc),
        "shared_mem_budget_bytes": profile.shared_mem_bytes,
        "candidates_total": len(cands),
        "candidates_kept": len(kept),
        "candidates_pruned": len(pruned),
        "locked_axes": dict(target.locked_axes),
        "shapes": [
            {"name": s.name, "n": s.n, "k_real": s.k_real, "k_padded": s.k_padded}
            for s in target.shapes
        ],
        "m_values": list(ms),
        "off_grid_m_values": list(target.off_grid_m_values),
        "table_buckets": [list(b) for b in buckets],
        "points_per_candidate": len(ms) * len(target.shapes),
        "table_name": target.table_name,
    }


# ─── Sweep (rig-only) ────────────────────────────────────────────────


def _bench_ms(fn: Callable[[], Any], warmup: int, iters: int) -> float:
    """Median wall ms over ``iters`` after ``warmup`` (CUDA-event timed)."""
    import torch  # noqa: PLC0415

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def _geomean(xs: list[float]) -> float:
    if not xs:
        return 0.0
    prod = 1.0
    for x in xs:
        prod *= x
    return prod ** (1.0 / len(xs))


def run_sweep(target: KernelTarget, profile: ArchProfile, args: argparse.Namespace) -> dict:
    """Full sweep on the rig: bit-identical gate, timing, aggregation.

    Returns the results dict (also what --out serializes). Tolerance
    gating (--allow-tolerance) compares vs the torch reference at
    target.atol and records ``bitwise: false`` for those entries.
    """
    import torch  # noqa: PLC0415

    cap = torch.cuda.get_device_capability()
    if tuple(cap) != profile.cc:
        raise RuntimeError(
            f"device capability {cap} != profile {profile.name} {profile.cc}; "
            f"pass the matching --arch"
        )

    assert target.make_runner is not None
    run, reference = target.make_runner(target, args)
    plan = build_plan(target, profile)
    cands = candidate_space(target)
    kept, _ = prune_candidates(cands, profile, target.elt_size)
    ms = tuple(args.m_values) if args.m_values else target.m_values

    # speedups[(bucket_key)][candidate] -> list of per-point speedups
    speedups: dict[tuple[int, bool], dict[TileCandidate, list[float]]] = {}
    bitwise_ok: dict[TileCandidate, bool] = dict.fromkeys(kept, True)

    for shape in target.shapes:
        for M in ms:
            base_out = run(shape, M, target.default_config, args.seed)
            base_ms = _bench_ms(
                lambda s=shape, m=M: run(s, m, target.default_config, args.seed),
                args.warmup, args.iters,
            )
            ref_out = reference(shape, M, args.seed) if args.allow_tolerance else None
            key = (m_bucket(M), is_small_n(shape.n))
            bucket = speedups.setdefault(key, {})
            for cand in kept:
                try:
                    out = run(shape, M, cand, args.seed)
                except Exception as e:  # noqa: BLE001 — OOR/compile fail = reject
                    print(f"  reject {cand.as_tuple()} @ {shape.name} M={M}: {e!r}")
                    bucket.setdefault(cand, []).append(0.0)
                    continue
                if not torch.equal(out, base_out):
                    bitwise_ok[cand] = False
                    if ref_out is None:
                        # Bit-identical gate (the #45126 assert): reject.
                        bucket.setdefault(cand, []).append(0.0)
                        continue
                    max_abs = (out.float() - ref_out.float()).abs().max().item()
                    if max_abs > target.atol:
                        print(
                            f"  reject {cand.as_tuple()} @ {shape.name} M={M}: "
                            f"max_abs={max_abs:.2e} > atol={target.atol}"
                        )
                        bucket.setdefault(cand, []).append(0.0)
                        continue
                cand_ms = _bench_ms(
                    lambda s=shape, m=M, c=cand: run(s, m, c, args.seed),
                    args.warmup, args.iters,
                )
                bucket.setdefault(cand, []).append(base_ms / max(cand_ms, 1e-9))

    records = []
    for key in sorted(speedups):
        per_cand = {
            c: _geomean(v) for c, v in speedups[key].items() if 0.0 not in v
        }
        per_cand[target.default_config] = 1.0
        best, best_gain = max(per_cand.items(), key=lambda kv: kv[1])
        # #45126 convention: keep the default explicitly when nothing
        # beats it beyond noise.
        if best_gain < 1.0 + args.noise_floor:
            best, best_gain = target.default_config, 1.0
        records.append(
            {
                "m_bucket": key[0],
                "is_small_n": key[1],
                "config": list(best.as_tuple()),
                "bitwise": bool(bitwise_ok.get(best, True)),
                "geomean_speedup": round(best_gain, 4),
                "points": len(speedups[key].get(best, [])) or len(ms),
            }
        )

    return {
        "tool": TOOL_NAME,
        "target": target.name,
        "arch": profile.name,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "table_name": target.table_name,
        "plan": plan,
        "records": records,
    }


# ─── Frozen table emission (torch-free) ──────────────────────────────


def emit_frozen_table(results: dict, allow_tolerance: bool = False) -> str:
    """Render results as a frozen per-arch dict literal (#45126 style).

    Non-bit-identical winners are EXCLUDED unless ``allow_tolerance``;
    omitted buckets fall back to DEFAULT_TILE_CONFIG at lookup time
    (the kernel-side resolver is fail-open).
    """
    arch = results["arch"]
    profile = ARCH_PROFILES[arch]
    table_name = results.get("table_name", "TUNED_TILES")
    lines = [
        f"# Frozen tuned tile table emitted by {results.get('tool', TOOL_NAME)}",
        f"# target={results.get('target', '?')} arch={arch} cc={profile.cc}",
        "# Keyed (cc_major, cc_minor) -> {(m_bucket, is_small_n):",
        "#   (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, num_warps, num_stages)}",
        f"{table_name} = {{",
        f"    {profile.cc!r}: {{",
    ]
    emitted = 0
    for rec in sorted(
        results.get("records", []),
        key=lambda r: (r["m_bucket"], not r["is_small_n"]),
    ):
        cfg = tuple(rec["config"])
        key = (rec["m_bucket"], rec["is_small_n"])
        if not rec.get("bitwise", False):
            if not allow_tolerance:
                lines.append(
                    f"        # {key!r}: {cfg!r} EXCLUDED — "
                    "NOT bit-identical (rerun with --allow-tolerance to include)"
                )
                continue
            lines.append(
                f"        {key!r}: {cfg!r},  "
                f"# tolerance-gated (NOT bit-identical), "
                f"x{rec.get('geomean_speedup', '?')}"
            )
        else:
            lines.append(
                f"        {key!r}: {cfg!r},  "
                f"# x{rec.get('geomean_speedup', '?')}"
            )
        emitted += 1
    lines.append("    },")
    lines.append("}")
    if emitted == 0:
        lines.insert(4, "# WARNING: no entries passed the gate — table is empty.")
    return "\n".join(lines)


# ─── CLI ─────────────────────────────────────────────────────────────


def _parse_int_list(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="triton_gemm_sweep.py",
        description=(
            "Offline Triton GEMM tuning sweep (vllm#45126 methodology): "
            "frozen per-arch tile tables + PID swizzle, bit-identical gate."
        ),
    )
    parser.add_argument("--target", default="g4_kpad_moe_gemm",
                        help="kernel target to sweep (see TARGETS)")
    parser.add_argument("--arch", default="sm_86",
                        help="arch profile (default sm_86 — RTX A5000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="torch-free: validate shapes, print the plan, exit")
    parser.add_argument("--emit-table", metavar="RESULTS_JSON",
                        help="torch-free: render a results JSON as a frozen table")
    parser.add_argument("--allow-tolerance", action="store_true",
                        help="gate BLOCK_K-changing candidates vs the torch "
                             "reference at target atol instead of rejecting "
                             "non-bit-identical outputs")
    parser.add_argument("--hidden-size", type=int, default=None,
                        help="g4 target: per-rank N (hidden_size). VERIFY "
                             "against the checkpoint config.json on the rig; "
                             f"dry-run default {G4_PLACEHOLDER_HIDDEN_SIZE} "
                             "(placeholder)")
    parser.add_argument("--m-values", type=_parse_int_list, default=None,
                        help="comma-separated tuning M grid override")
    parser.add_argument("--num-experts", type=int, default=8,
                        help="experts allocated in synthetic inputs (run mode)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--noise-floor", type=float, default=0.03,
                        help="min geomean speedup to displace the default config")
    parser.add_argument("--out", metavar="RESULTS_JSON",
                        help="write sweep results JSON here (run mode)")
    args = parser.parse_args(argv)

    # --emit-table is a standalone torch-free mode.
    if args.emit_table:
        path = Path(args.emit_table)
        if not path.is_file():
            print(f"ERROR: results file not found: {path}", file=sys.stderr)
            return 2
        try:
            results = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"ERROR: cannot read results JSON: {e}", file=sys.stderr)
            return 2
        if results.get("arch") not in ARCH_PROFILES:
            print(f"ERROR: unknown arch in results: {results.get('arch')!r}",
                  file=sys.stderr)
            return 2
        print(emit_frozen_table(results, allow_tolerance=args.allow_tolerance))
        return 0

    if args.target not in TARGETS:
        print(
            f"ERROR: unknown target {args.target!r} "
            f"(known: {', '.join(sorted(TARGETS))})",
            file=sys.stderr,
        )
        return 2
    if args.arch not in ARCH_PROFILES:
        print(
            f"ERROR: unknown arch {args.arch!r} "
            f"(known: {', '.join(sorted(ARCH_PROFILES))})",
            file=sys.stderr,
        )
        return 2

    hidden = args.hidden_size or G4_PLACEHOLDER_HIDDEN_SIZE
    targets = _build_targets(hidden_size=hidden)
    target = targets[args.target]
    profile = ARCH_PROFILES[args.arch]

    try:
        plan = build_plan(target, profile, m_values=args.m_values)
    except ValueError as e:
        print(f"ERROR: invalid sweep plan: {e}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"DRY-RUN plan ({TOOL_NAME}) — nothing executed, torch-free:")
        print(json.dumps(plan, indent=2))
        if args.hidden_size is None and args.target == "g4_kpad_moe_gemm":
            print(
                f"NOTE: N uses the PLACEHOLDER hidden_size="
                f"{G4_PLACEHOLDER_HIDDEN_SIZE}; pass --hidden-size read "
                "from the checkpoint config.json before a real sweep."
            )
        return 0

    # Run mode (rig-only).
    if args.target == "g4_kpad_moe_gemm" and args.hidden_size is None:
        print(
            "ERROR: run mode requires --hidden-size (read it from the "
            "checkpoint config.json — do not guess).",
            file=sys.stderr,
        )
        return 2
    try:
        results = run_sweep(target, profile, args)
    except (RuntimeError, ImportError) as e:
        print(f"ERROR: sweep failed: {e}", file=sys.stderr)
        return 1
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"results written to {args.out}")
    print(emit_frozen_table(results, allow_tolerance=args.allow_tolerance))
    return 0


if __name__ == "__main__":
    sys.exit(main())
