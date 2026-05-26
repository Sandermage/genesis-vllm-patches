# SPDX-License-Identifier: Apache-2.0
"""ReferenceMetrics + VerifyTolerances + ConfigConstraints + RiskScore.

All four were inline classes in ``model_configs/schema.py`` before
M.5.1. Bodies unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ._base import SchemaError


@dataclass
class ReferenceMetrics:
    """Empirically-measured baseline for `verify` to compare against.

    Required fields are core: long_gen TPS, tool quality, stability CV,
    VRAM, and pins (the things `verify` always checks). Optional fields
    (short_gen_tps, concurrent_4_total_s) are richer benchmarks captured
    by `genesis_bench_suite.py` but not by the lightweight
    `verify.bench_metrics()` used by `bench-and-update`.
    """
    measured_at: str  # ISO-8601
    bench_method: str
    long_gen_sustained_tps: float
    long_gen_mean_lat_s: float
    tool_call_score: str  # '10/10'
    stability_mean_s: float
    stability_cv_pct: float
    vram_used_mib_per_gpu: list[int]
    vram_total_mib: int
    genesis_pin: str
    vllm_pin: str
    short_gen_tps: Optional[float] = None
    concurrent_4_total_s: Optional[float] = None
    # Empirical-bake (Genesis Phase D 2026-05-07): per-config measured mamba
    # REQUEST_CONSTANT state size, in MiB. When set, R-018 audit rule uses
    # this exact value instead of the 250 MiB heuristic — gives precise
    # capacity-overflow detection per model architecture. Set automatically
    # by `genesis bench-and-update --measure-mamba-state` post-warmup, or
    # manually via empirical observation of boot logs.
    # NULL/unset → R-018 falls back to 250 MiB conservative default.
    mamba_state_mib_per_request: Optional[float] = None

    # Wave 1+2 canonical genesis_bench_suite
    # output adds richer per-component metrics. All optional — old
    # configs without these fields still load. New canonical bench
    # writes them for future regression detection.
    decode_tpot_ms: Optional[float] = None       # pure decode time (no TTFT)
    ttft_ms: Optional[float] = None              # time to first token
    spec_accept_rate: Optional[float] = None     # MTP acceptance ratio (0..1)
    # Historical reference for regression triage. When the current
    # wave's bench drops below `prev_long_gen_tps` by more than the
    # tolerance, it's a real regression vs the prior known-good.
    prev_long_gen_tps: Optional[float] = None
    prev_genesis_pin: Optional[str] = None
    prev_vllm_pin: Optional[str] = None
    # Second-tier historical reference (Wave 8 closure 2026-05-11):
    # the optimization-sprint winner from the prior wave. Lets the
    # current bench compare against BOTH the previous-baseline AND
    # the previous-best-sweep. Pin-bump candidates that match sprint1
    # within tolerance are net-neutral (acceptable); regressions
    # vs sprint1 but improvements vs baseline are still net wins.
    prev_long_gen_tps_sprint1: Optional[float] = None
    # Wave 8 delta annotations (audit trail — 2026-05-11). Strings
    # (e.g. '+5.78%') because YAML carries the formatted value the
    # operator reviewed at the wave-close meeting. Loaded as-is and
    # surfaced in `verify` output without numeric recomputation.
    wave8_delta_pct_vs_wave7: Optional[str] = None
    wave8_delta_pct_vs_sprint1: Optional[str] = None
    wave8_decode_tpot_delta_pct_vs_sprint1: Optional[str] = None
    wave8_ttft_delta_pct_vs_sprint1: Optional[str] = None
    # Wave 9 delta annotations (audit trail — 2026-05-12). Same pattern
    # as wave8_* — pre-formatted strings. Populated on the 35B PROD
    # config after the dev93→dev209 pin-bump A/B re-bench surfaced a
    # -2.82% A3B-FP8 regression that was absent on 27B hybrid GDN.
    wave9_delta_pct_vs_sprint1: Optional[str] = None
    wave9_decode_tpot_delta_pct_vs_sprint1: Optional[str] = None
    wave9_ttft_delta_pct_vs_sprint1: Optional[str] = None


@dataclass
class VerifyTolerances:
    """Acceptable drift before `verify` returns failure."""
    tps_drop_pct_max: float = 5.0       # fail if drop >5%
    tool_call_min: str = "9/10"          # fail if <9/10
    stability_cv_pct_max: float = 6.0    # fail if jitter doubles
    vram_increase_mib_max: int = 2000    # fail if VRAM grew >2 GB

    def validate(self) -> None:
        if self.tps_drop_pct_max < 0:
            raise SchemaError(
                "VerifyTolerances.tps_drop_pct_max must be >= 0"
            )
        if self.stability_cv_pct_max < 0:
            raise SchemaError(
                "VerifyTolerances.stability_cv_pct_max must be >= 0"
            )


@dataclass
class ConfigConstraints:
    """T1.8 (audit closure §7.2): hardware + flag constraints.

    Operators can declare invariants the launcher must check BEFORE
    starting vllm: minimum GPU VRAM/count, PCIe topology requirements,
    forbidden flags. The launch-time check fails loudly with a precise
    error pointing at the violating field, instead of failing
    mysteriously deep in vllm boot.

    All fields are optional; absence means "no constraint declared".
    """
    min_gpu_memory_gib: Optional[int] = None
    min_gpu_count: Optional[int] = None
    pcie_ok: bool = True
    nvlink_recommended: bool = False
    forbidden_flags: list[str] = field(default_factory=list)
    required_kernel_modules: list[str] = field(default_factory=list)
    notes: str = ""

    def validate(self) -> None:
        if (self.min_gpu_memory_gib is not None
                and self.min_gpu_memory_gib <= 0):
            raise SchemaError(
                "ConfigConstraints.min_gpu_memory_gib must be > 0 "
                f"(got {self.min_gpu_memory_gib})"
            )
        if (self.min_gpu_count is not None
                and self.min_gpu_count <= 0):
            raise SchemaError(
                "ConfigConstraints.min_gpu_count must be > 0 "
                f"(got {self.min_gpu_count})"
            )
        for flag in self.forbidden_flags:
            if not isinstance(flag, str):
                raise SchemaError(
                    "ConfigConstraints.forbidden_flags must be list[str]"
                )

    def check(self, *, hw, vllm_extra_args: list[str]) -> list[str]:
        """Evaluate constraints against (hw, vllm_extra_args).

        Returns a list of human-readable violation messages. Empty list
        means "all constraints satisfied". The launcher consults this
        and aborts if any violation surfaces.
        """
        violations: list[str] = []
        if self.min_gpu_count is not None and hw is not None:
            n = int(getattr(hw, "n_gpus", 0) or 0)
            if n < self.min_gpu_count:
                violations.append(
                    f"min_gpu_count={self.min_gpu_count} but hardware.n_gpus={n}"
                )
        if self.min_gpu_memory_gib is not None and hw is not None:
            mib = int(getattr(hw, "min_vram_per_gpu_mib", 0) or 0)
            gib = mib / 1024
            if gib < self.min_gpu_memory_gib:
                violations.append(
                    f"min_gpu_memory_gib={self.min_gpu_memory_gib} but "
                    f"hardware.min_vram_per_gpu_mib={mib} ({gib:.1f} GiB)"
                )
        flat_args = " ".join(vllm_extra_args)
        for forbidden in self.forbidden_flags:
            if forbidden in flat_args or forbidden in vllm_extra_args:
                violations.append(
                    f"forbidden flag {forbidden!r} present in vllm_extra_args"
                )
        return violations


@dataclass
class RiskScore:
    """T1.8 (audit closure §7.2): per-config risk dimensions.

    Operators (or `sndr model-config score`) populate these to give a
    reviewer a glanceable verdict before running. Each field is a
    0-100 score where 0 = no risk and 100 = will-definitely-blow-up.
    `derive_overall()` computes a weighted sum so dashboards can rank
    configs.

    Dimensions:
      - memory_safety:    KV/scratch/CUDA-graph headroom on declared HW.
      - tool_call:        Empirical tool-parser stability (10/10 = 0).
      - spec_decode:      MTP/ngram acceptance variance + GDN risk.
      - upstream_drift:   How many declared patches have upstream PRs.
      - deployment_ready: Cross-rig + soak-time signal.
    """
    memory_safety: int = 0
    tool_call: int = 0
    spec_decode: int = 0
    upstream_drift: int = 0
    deployment_ready: int = 0
    notes: str = ""

    def validate(self) -> None:
        for name in ("memory_safety", "tool_call", "spec_decode",
                     "upstream_drift", "deployment_ready"):
            v = getattr(self, name)
            if not isinstance(v, int):
                raise SchemaError(
                    f"RiskScore.{name} must be int 0-100 (got {type(v).__name__})"
                )
            if not (0 <= v <= 100):
                raise SchemaError(
                    f"RiskScore.{name} must be in [0,100] (got {v})"
                )

    def derive_overall(self) -> int:
        """Weighted aggregate of the five dimensions, 0-100.

        Weights reflect production impact: memory_safety + deployment_ready
        get the largest weight because they predict launch-success;
        tool_call is medium because operators can fix at request time;
        spec_decode + upstream_drift are auxiliary.
        """
        weights = {
            "memory_safety": 30,
            "deployment_ready": 25,
            "tool_call": 20,
            "spec_decode": 15,
            "upstream_drift": 10,
        }
        total = sum(
            getattr(self, k) * w
            for k, w in weights.items()
        )
        return total // sum(weights.values())
