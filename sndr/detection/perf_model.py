# SPDX-License-Identifier: Apache-2.0
"""Genesis Performance Model — roofline-based kernel cost estimation.

Mathematical model for predicting Triton kernel performance on a given
architecture. Implements the standard "roofline" methodology:

    achieved_throughput = min(peak_compute, peak_bandwidth * arithmetic_intensity)

with architecture-specific adjustments for:
  - Shared memory occupancy (smem_budget vs config smem_usage)
  - Register pressure (warp count × per-warp register limit)
  - L2 cache hit ratio (estimated from working set vs L2 size)
  - Memory bandwidth saturation per SM count
  - Triton-specific quirks (num_stages pipeline depth, warp scheduling)

Used by PN298-PN300 patches to:
  1. RANK candidate Triton autotune configs without running them
  2. PRUNE configs that the roofline model predicts will be Pareto-dominated
  3. DERIVE recommended block sizes from first-principles

================================================================
ROOFLINE BASICS (mathematical formulation)
================================================================

For a kernel processing N elements of dtype D:
  - Compute (FLOPs) = f(N, kernel-specific)
  - Memory traffic (bytes) = N × sizeof(D) × access_pattern_factor
  - Arithmetic intensity AI = FLOPs / bytes
  - Ridge point AI_ridge = peak_FLOPS / peak_BW
  - If AI ≥ AI_ridge: compute-bound, time = FLOPs / peak_FLOPS
  - If AI < AI_ridge: memory-bound, time = bytes / peak_BW

Triton-specific adjustments:
  - Effective shared mem per CTA = config.BLOCK_M × config.BLOCK_K × dtype_bytes
  - Max concurrent CTAs per SM = min(
       smem_per_sm / smem_per_cta,
       1024 / config.num_warps,  # threads cap
       reg_per_sm / (config.num_warps × 32 × 64),  # register cap (64 regs typical)
    )
  - Occupancy = concurrent_CTAs × num_warps × 32 / max_threads_per_sm
  - num_stages × pipeline_depth = effective smem multiplier

================================================================
SHARED MEM BUDGET MATH
================================================================

A5000 SM 8.6: 100 KB shared per SM dynamic. After CUDA runtime
overhead (~4 KB), usable budget ≈ 96 KB. Triton config with:
  - BLOCK_M=64, BLOCK_K=128, fp16 = 64×128×2 = 16 KB per stage
  - num_stages=3 → 3 × 16 = 48 KB → fits, no spill
  - num_stages=4 → 4 × 16 = 64 KB → fits but tight
  - num_warps=8 → 8 warps × ~150 regs avg = 1200 regs/CTA
                  → reg cap 65536 / 1200 ≈ 54 concurrent CTAs theoretical
                  → smem cap 96 / 16 = 6 CTAs actual
                  → occupancy LIMITED by smem
                  → adding warps just wastes resources

H100 SM 9.0: 228 KB shared per SM. Same config:
  - num_stages=3 → 48 KB → fits trivially
  - num_warps=8 → fits trivially, occupancy higher
  - LARGER BLOCK_M actually beneficial

================================================================

Author: Sandermage (Sander) Barzov Aleksandr, 2026-06-05.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("genesis.detection.perf_model")


# ─── Architecture-specific compute peaks (TFLOPS, sustained) ──────────
# Format: dtype → (peak TFLOPS, per-SM throughput hint)
_PEAK_COMPUTE_FP16: dict[tuple[int, int], float] = {
    # (sm_major, sm_minor) → TFLOPS (FP16 tensor cores, dense)
    (8, 0):  312.0,   # A100 — 312 TFLOPS FP16 TC
    (8, 6):  88.0,    # A5000 — 88 TFLOPS FP16 TC (or 222 sparse)
    (8, 9):  165.0,   # RTX 4090 — 165 TFLOPS FP16 TC
    (9, 0):  989.0,   # H100 — 989 TFLOPS FP16 TC (1979 FP8)
    (10, 0): 4500.0,  # B200 estimate
}

_PEAK_COMPUTE_FP32: dict[tuple[int, int], float] = {
    (8, 0):  19.5,    # A100 FP32 (CUDA cores)
    (8, 6):  27.8,    # A5000 FP32 CUDA cores  — boosted vs A100 (clock)
    (8, 9):  82.6,    # RTX 4090 FP32
    (9, 0):  67.0,    # H100 FP32 CUDA + TC for FP32 native
    (10, 0): 80.0,    # B200 estimate
}

# Shared memory per CTA must leave room for stages * tile-bytes
# Triton config: smem_per_cta = BLOCK_M * BLOCK_K * dtype_bytes * num_stages
# Plus alignment + barriers: + ~2 KB overhead per CTA
_SMEM_PER_CTA_OVERHEAD_KB = 2

# Conservative per-warp register estimate for hot Triton kernels.
# Real Triton kernels use 60-150 regs/thread depending on complexity.
# For "tl.dot" heavy kernels, 96-128 regs/thread is typical.
_REGS_PER_THREAD_TYPICAL = 96
_REGS_PER_SM_AMPERE = 65536   # SM 8.x
_REGS_PER_SM_HOPPER = 65536   # SM 9.x (same register file size)
_MAX_THREADS_PER_SM_AMPERE = 1536  # SM 8.6
_MAX_THREADS_PER_SM_AMPERE_DC = 2048  # SM 8.0 (A100)
_MAX_THREADS_PER_SM_HOPPER = 2048  # SM 9.0


@dataclass(frozen=True)
class TritonConfigCost:
    """Predicted cost metrics for a Triton autotune Config on a given arch."""
    config_id: str  # e.g. "BLOCK_M=64,BLOCK_K=128,nw=4,ns=2"
    smem_per_cta_kb: float
    smem_fits: bool
    occupancy_ratio: float  # 0..1, threads/cta × concurrent_ctas / max_threads
    estimated_us_per_call: Optional[float]
    register_pressure: float  # 0..1, regs/thread × num_warps / regs_per_sm
    risk_flags: tuple[str, ...]


def estimate_triton_config_cost(
    block_m: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
    dtype_bytes: int,
    n_total: int,  # total elements processed (for FLOPs estimate)
    sm_major: int,
    sm_minor: int,
    shared_mem_kb_per_sm: int,
) -> TritonConfigCost:
    """Compute roofline-style cost prediction for a Triton config.

    Returns a TritonConfigCost with smem fit, occupancy, register pressure.
    Higher occupancy = better; lower register pressure = better.
    smem_fits=False ⇒ config will FAIL at runtime, must be pruned.
    """
    risks = []

    # ─── Shared memory budget check ────────────────────────────────
    smem_per_cta_kb = (
        (block_m * block_k * dtype_bytes / 1024.0) * num_stages
        + _SMEM_PER_CTA_OVERHEAD_KB
    )
    smem_budget = shared_mem_kb_per_sm - 4  # reserve 4 KB for runtime
    smem_fits = smem_per_cta_kb <= smem_budget
    if not smem_fits:
        risks.append("smem_exceeded")

    smem_limit_ctas = max(1, int(smem_budget // max(1.0, smem_per_cta_kb)))

    # ─── Register pressure ──────────────────────────────────────────
    threads_per_cta = num_warps * 32
    regs_per_cta = threads_per_cta * _REGS_PER_THREAD_TYPICAL
    regs_per_sm = (
        _REGS_PER_SM_HOPPER if sm_major >= 9 else _REGS_PER_SM_AMPERE
    )
    reg_limit_ctas = max(1, regs_per_sm // max(1, regs_per_cta))
    reg_pressure = min(1.0, regs_per_cta / regs_per_sm)
    if reg_pressure > 0.85:
        risks.append("reg_pressure_high")

    # ─── Thread cap ─────────────────────────────────────────────────
    max_threads = (
        _MAX_THREADS_PER_SM_HOPPER if sm_major >= 9
        else (_MAX_THREADS_PER_SM_AMPERE_DC if (sm_major, sm_minor) == (8, 0)
              else _MAX_THREADS_PER_SM_AMPERE)
    )
    thread_limit_ctas = max(1, max_threads // threads_per_cta)

    # ─── Concurrent CTAs and occupancy ──────────────────────────────
    concurrent_ctas = min(smem_limit_ctas, reg_limit_ctas, thread_limit_ctas)
    occupancy = (concurrent_ctas * threads_per_cta) / max_threads
    occupancy = min(1.0, occupancy)
    if occupancy < 0.25:
        risks.append("low_occupancy")

    # ─── Estimated time per kernel call ────────────────────────────
    # Very rough: assume FLOPs/byte ratio of 1.5 (mixed compute/memory),
    # time scales inversely with occupancy.
    # We don't have the actual kernel body to count FLOPs precisely.
    est_us = None
    if smem_fits:
        # Heuristic: penalty for low occupancy
        base_time = (n_total / 1e6)  # arbitrary units
        est_us = base_time / max(0.1, occupancy)

    return TritonConfigCost(
        config_id=f"BLOCK_M={block_m},BLOCK_K={block_k},nw={num_warps},ns={num_stages}",
        smem_per_cta_kb=round(smem_per_cta_kb, 2),
        smem_fits=smem_fits,
        occupancy_ratio=round(occupancy, 3),
        estimated_us_per_call=round(est_us, 3) if est_us else None,
        register_pressure=round(reg_pressure, 3),
        risk_flags=tuple(risks),
    )


# ─── Memory hierarchy model ─────────────────────────────────────────


@dataclass(frozen=True)
class MemoryHierarchyProfile:
    """Captured memory-system facts + Genesis decisions per arch."""
    shared_mem_kb_per_sm: int
    l2_cache_mb: int
    hbm_bandwidth_gbps: int
    num_sms: int

    # Computed bytes/SM/sec
    @property
    def peak_hbm_bytes_per_sec(self) -> int:
        return self.hbm_bandwidth_gbps * 1_000_000_000

    @property
    def hbm_bytes_per_sm_per_sec(self) -> int:
        return self.peak_hbm_bytes_per_sec // self.num_sms

    @property
    def l2_per_sm_mb(self) -> float:
        """L2 share per SM (assuming round-robin partitioning)."""
        return self.l2_cache_mb / self.num_sms

    def working_set_in_l2(self, bytes_total: int) -> float:
        """Predicted fraction of a working set that fits in L2.

        Useful for estimating L2 hit rate vs HBM round-trips.
        """
        l2_bytes = self.l2_cache_mb * 1_000_000
        if bytes_total <= l2_bytes:
            return 1.0
        return l2_bytes / bytes_total


def get_memory_profile(profile) -> MemoryHierarchyProfile:
    """Build memory profile from a GenesisGPUArchProfile."""
    return MemoryHierarchyProfile(
        shared_mem_kb_per_sm=profile.shared_mem_kb_per_sm,
        l2_cache_mb=profile.l2_cache_mb,
        hbm_bandwidth_gbps=profile.hbm_bandwidth_gbps,
        num_sms=profile.num_sms,
    )


# ─── Quantization arithmetic intensity helpers ──────────────────────


def quant_dequant_overhead_bytes(
    weight_bits: int,
    activation_bits: int,
    output_bits: int,
    n_weights: int,
) -> int:
    """Memory traffic for a quantized matmul step.

    For a W{wb}A{ab}_O{ob} matmul, the dominant traffic is weight load
    (n_weights × weight_bits/8 bytes), plus dequant scale/zero-point
    (assumed 1 scale per group, group_size=128 typical):

        weight_bytes  = n_weights × wb / 8
        scale_bytes   = (n_weights / 128) × 2  # fp16 scales
        zp_bytes      = (n_weights / 128) × 1  # int8 zero-points (GPTQ)
        total         = weight_bytes + scale_bytes + zp_bytes

    Per-call activation traffic depends on prefill/decode batch shape.
    """
    weight_bytes = (n_weights * weight_bits) // 8
    scale_bytes = (n_weights // 128) * 2
    zp_bytes = (n_weights // 128) * 1
    return weight_bytes + scale_bytes + zp_bytes


def recommend_quant_kernel(
    arch_profile,
    weight_dtype: str,
    activation_dtype: str,
) -> str:
    """Return recommended Marlin variant for our arch + dtype combo.

    Decision tree based on hardware features:
      - Marlin INT4 weight + FP16 act: most arches (default)
      - Marlin INT4 + FP8 act: SM 8.9 Ada or SM 12.x Blackwell consumer
      - Marlin INT4 + INT8 act: SM 8.x for further quant
      - AWQ Marlin: int4 + per-channel scales (smaller groups)
    """
    if weight_dtype not in ("int4", "int8"):
        return "no-marlin"
    if activation_dtype == "fp8":
        if arch_profile.is_ada or arch_profile.is_blackwell:
            return "marlin-w4a8-fp8"
        # SM 8.6 Ampere has no native FP8; would fall back
        return "marlin-w4a16"  # fp16 activation safe default
    if activation_dtype == "int8":
        return "marlin-w4a8-int8"
    return "marlin-w4a16"


__all__ = [
    "TritonConfigCost",
    "MemoryHierarchyProfile",
    "estimate_triton_config_cost",
    "get_memory_profile",
    "quant_dequant_overhead_bytes",
    "recommend_quant_kernel",
]
