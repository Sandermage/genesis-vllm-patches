# SPDX-License-Identifier: Apache-2.0
"""§6.8 release-gate consumer — turn proof artefacts into a single
release/no-release verdict.

Entry 12 + 17 wrote static checks. Entry 19 added bench_delta ingest.
Entry 20 surfaced bucket counts. This module is the *decider*: given a
release policy (how strict do you need evidence?) and an optional
regression threshold, return per-patch verdicts + an aggregate
release-blocked flag.

Policy modes — each strictly stronger than the previous:

  report           — never blocks; report bucket counts + verdicts only
  require-static   — block if ANY patch is `dead` or `static_failed`
  require-bench    — also block on `static_only` (must have some bench data)
  require-baseline — also block on `bench_attached` (must have baseline)

Regression check (orthogonal, applies to `bench_with_baseline` patches
when `--max-regression-pct N` is set):

  • TPS metrics (median_tps, p95_tps): negative delta beyond -N% blocks.
  • Latency metrics (decode_tpot_ms, ttft_ms): positive delta beyond +N% blocks.

The metric polarities are baked in (we don't trust per-patch overrides
to set them — that's how regression-detection bugs ship).

The policy is operator-decided. Default (`report`) is the
non-disruptive starting point; CI can tighten the mode over time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import (
    DEFAULT_PROOF_DIR,
    PROOF_STATUS_BUCKETS,
    classify_proof,
    find_proof_artefacts,
    load_proof_artefact,
)


__all__ = [
    "POLICY_MODES",
    "ReleasePolicy",
    "ReleaseVerdict",
    "evaluate_release",
    "ReleaseCheckError",
]


class ReleaseCheckError(Exception):
    """Raised on bad policy input (unknown mode, malformed threshold)."""


# Ordered: each mode supersedes the previous. Used to pre-compute the
# allowed-bucket set per mode.
POLICY_MODES: tuple[str, ...] = (
    "report",
    "require-static",
    "require-bench",
    "require-baseline",
)

# Buckets that satisfy each policy mode. Anything outside this set
# triggers a release block.
_MODE_ALLOWED: dict[str, frozenset[str]] = {
    "report": frozenset(PROOF_STATUS_BUCKETS),
    "require-static": frozenset({
        "static_only", "bench_attached", "bench_with_baseline",
    }),
    "require-bench": frozenset({
        "bench_attached", "bench_with_baseline",
    }),
    "require-baseline": frozenset({"bench_with_baseline"}),
}

# Metric polarity for regression check.
#   "tps"     → higher is better; negative delta_pct is a regression
#   "latency" → lower is better; positive delta_pct is a regression
_METRIC_POLARITY: dict[str, str] = {
    "median_tps_delta_pct": "tps",
    "p95_tps_delta_pct": "tps",
    "decode_tpot_delta_pct": "latency",
    "ttft_delta_pct": "latency",
}


@dataclass(frozen=True)
class ReleasePolicy:
    """The operator's release-readiness policy."""
    mode: str = "report"
    max_regression_pct: Optional[float] = None
    patch_filter: Optional[frozenset[str]] = None
    tier_filter: Optional[frozenset[str]] = None

    def __post_init__(self):
        if self.mode not in POLICY_MODES:
            raise ReleaseCheckError(
                f"unknown policy mode {self.mode!r}. Valid: {list(POLICY_MODES)}"
            )
        if self.max_regression_pct is not None and self.max_regression_pct < 0:
            raise ReleaseCheckError(
                f"max_regression_pct must be ≥ 0, got {self.max_regression_pct}"
            )

    @property
    def allowed_buckets(self) -> frozenset[str]:
        return _MODE_ALLOWED[self.mode]


@dataclass
class ReleaseVerdict:
    """Per-patch decision."""
    patch_id: str
    bucket: str
    family: str
    tier: str
    lifecycle: str
    passed: bool
    reasons: list[str] = field(default_factory=list)
    regressions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "patch_id": self.patch_id,
            "bucket": self.bucket,
            "family": self.family,
            "tier": self.tier,
            "lifecycle": self.lifecycle,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "regressions": list(self.regressions),
        }


def _best_bucket_for(
    patch_id: str, out_dir: Path,
) -> tuple[str, Optional[dict]]:
    """Return `(best_bucket, best_artefact_payload)` across all pins,
    or `("dead", None)` when no artefact exists."""
    artefacts = find_proof_artefacts(patch_id, out_dir)
    if not artefacts:
        return "dead", None

    best_bucket = "static_failed"
    best_payload: Optional[dict] = None
    best_rank = len(PROOF_STATUS_BUCKETS)
    for a in artefacts:
        try:
            data = load_proof_artefact(a)
        except (OSError, json.JSONDecodeError):
            continue
        b = classify_proof(data)
        rank = PROOF_STATUS_BUCKETS.index(b)
        if rank < best_rank:
            best_rank = rank
            best_bucket = b
            best_payload = data
    return best_bucket, best_payload


def _regressions_in(
    bench_delta: dict, max_pct: float,
) -> list[dict]:
    """Return a list of `{metric, delta_pct, polarity}` for every metric
    that exceeds the regression threshold."""
    out: list[dict] = []
    for key, polarity in _METRIC_POLARITY.items():
        v = bench_delta.get(key)
        if v is None:
            continue
        try:
            d = float(v)
        except (TypeError, ValueError):
            continue
        if polarity == "tps" and d < -max_pct:
            out.append({"metric": key, "delta_pct": d, "polarity": polarity})
        elif polarity == "latency" and d > max_pct:
            out.append({"metric": key, "delta_pct": d, "polarity": polarity})
    return out


def evaluate_release(
    policy: ReleasePolicy,
    *,
    registry: Optional[dict] = None,
    out_dir: Path = DEFAULT_PROOF_DIR,
) -> dict:
    """Evaluate every PATCH_REGISTRY entry against `policy`.

    Returns a dict shaped:

        {
          "policy":           {mode, max_regression_pct, ...},
          "total":            <int>,
          "considered":       <int>,   # after patch_filter / tier_filter
          "passed_count":     <int>,
          "failed_count":     <int>,
          "release_blocked":  <bool>,
          "verdicts":         [ReleaseVerdict-dict, ...],
        }
    """
    if registry is None:
        from sndr.dispatcher.registry import PATCH_REGISTRY
        registry = PATCH_REGISTRY

    allowed = policy.allowed_buckets
    verdicts: list[ReleaseVerdict] = []
    considered = 0

    for patch_id, meta in registry.items():
        if policy.patch_filter is not None and patch_id not in policy.patch_filter:
            continue
        tier = meta.get("tier", "?")
        if policy.tier_filter is not None and tier not in policy.tier_filter:
            continue
        considered += 1

        bucket, payload = _best_bucket_for(patch_id, out_dir)
        reasons: list[str] = []
        regressions: list[dict] = []

        if bucket not in allowed:
            reasons.append(
                f"bucket={bucket!r} not allowed under policy mode "
                f"{policy.mode!r} (need one of {sorted(allowed)})"
            )

        # Regression check applies only to bench_with_baseline patches.
        if (
            policy.max_regression_pct is not None
            and bucket == "bench_with_baseline"
            and payload is not None
        ):
            bd = payload.get("bench_delta") or {}
            regressions = _regressions_in(bd, policy.max_regression_pct)
            if regressions:
                reasons.append(
                    f"{len(regressions)} metric(s) regressed beyond "
                    f"±{policy.max_regression_pct}%"
                )

        passed = (policy.mode == "report") or not reasons
        verdicts.append(ReleaseVerdict(
            patch_id=patch_id,
            bucket=bucket,
            family=meta.get("family", "?"),
            tier=tier,
            lifecycle=meta.get("lifecycle", "?"),
            passed=passed,
            reasons=reasons,
            regressions=regressions,
        ))

    passed_count = sum(1 for v in verdicts if v.passed)
    failed_count = considered - passed_count

    return {
        "policy": {
            "mode": policy.mode,
            "max_regression_pct": policy.max_regression_pct,
            "patch_filter": sorted(policy.patch_filter)
                if policy.patch_filter else None,
            "tier_filter": sorted(policy.tier_filter)
                if policy.tier_filter else None,
        },
        "total": len(registry),
        "considered": considered,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "release_blocked": failed_count > 0 and policy.mode != "report",
        "verdicts": [v.to_dict() for v in verdicts],
    }
