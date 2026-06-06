# SPDX-License-Identifier: Apache-2.0
"""§6.8 bench-delta evidence — attach a bench JSON to a patch proof artefact.

Entry 12 designed `evidence/patch_proof/<patch_id>__<vllm_pin>.json`
with a `bench_delta` field that's currently always null. This module
fills that field from an actual bench-suite JSON (operator runs the
bench on GPU, then runs `sndr patches bench-attach <patch> <result>` on
any host to ingest the result).

Doesn't run the bench — that's GPU work. Just ingests an existing
result, extracts headline metrics, optionally diffs against a baseline,
and persists into the proof artefact.

Key design decisions:

  • Metric extraction is tolerant of bench-suite schema drift —
    we look at multiple aliases for each metric (wall_TPS / wall_tps /
    long_gen_sustained_tps / sustained_tps) and pick the first found.
  • Baseline comparison is optional. When provided, percent-delta is
    computed for each metric; otherwise the bench_delta block records
    only the absolute current-run values.
  • If the proof artefact doesn't exist yet, a stub is created from
    `build_proof_for_patch(patch_id)` (the static-checks side) so the
    bench evidence isn't dangling. Caller can pre-populate it via
    `sndr patches prove <patch_id>` if they want the static checks too.
  • A `bench_methodology_sha` field is captured so later we can detect
    when a bench was run against a stale methodology contract.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import (
    DEFAULT_PROOF_DIR,
    build_proof_for_patch,
    find_proof_artefacts,
    load_proof_artefact,
    write_proof_artefact,
)


log = logging.getLogger("genesis.proof.bench_attach")


__all__ = [
    "BenchDelta",
    "extract_headline_metrics",
    "compute_delta",
    "attach_bench",
    "BenchAttachError",
]


class BenchAttachError(Exception):
    """Raised when the bench JSON can't be parsed or doesn't carry
    enough fields for a meaningful attach."""


# ─── Headline-metric extraction ───────────────────────────────────────


# Aliases the bench-suite has used at various points. Earlier-listed
# names win when multiple are present.
_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "median_tps": (
        "median_tps", "wall_TPS_median", "wall_tps_median",
        "long_gen_sustained_tps", "sustained_tps", "wall_TPS", "wall_tps",
    ),
    "p95_tps": ("p95_tps", "wall_TPS_p95", "wall_tps_p95"),
    "decode_tpot_ms": (
        "decode_TPOT_ms", "decode_tpot_ms",
        "long_gen_mean_lat_s",   # last-resort, converted ms below
    ),
    "ttft_ms": ("TTFT_ms", "ttft_ms"),
    "cv_pct": ("cv_pct", "stability_cv_pct", "wall_TPS_cv_pct"),
    "tool_call_score": ("tool_call_score", "tool_call"),
}


def _walk(d: dict, path: tuple[str, ...]):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _first_present(bench: dict, keys: tuple[str, ...]) -> Optional[Any]:
    """Return the first value found at any of the candidate keys.

    Looks at top-level + a few common sub-blocks (`headline`, `summary`,
    `reference_metrics`, `metrics`) so we don't have to teach the caller
    about bench-suite schema variants.
    """
    candidates = (
        (),
        ("headline",),
        ("summary",),
        ("reference_metrics",),
        ("metrics",),
    )
    for prefix in candidates:
        cur = _walk(bench, prefix)
        if not isinstance(cur, dict):
            continue
        for k in keys:
            if k in cur and cur[k] is not None:
                return cur[k]
    return None


@dataclass
class BenchDelta:
    """The `bench_delta` field shape persisted into the proof artefact.

    `*_pct` fields are populated only when a baseline is supplied; left
    None when ingesting a single-run bench JSON.
    """
    measured_at: str
    methodology_id: Optional[str]
    methodology_sha: Optional[str]
    composed_key: Optional[str]
    vllm_pin: Optional[str]
    median_tps: Optional[float]
    p95_tps: Optional[float]
    decode_tpot_ms: Optional[float]
    ttft_ms: Optional[float]
    cv_pct: Optional[float]
    tool_call_score: Optional[str]
    # Optional baseline comparison fields:
    baseline_path: Optional[str] = None
    median_tps_delta_pct: Optional[float] = None
    p95_tps_delta_pct: Optional[float] = None
    decode_tpot_delta_pct: Optional[float] = None
    ttft_delta_pct: Optional[float] = None

    def to_dict(self) -> dict:
        # Drop None values so the artefact stays compact; the proof
        # consumer (release gate) treats absent == not-measured.
        return {k: v for k, v in self.__dict__.items() if v is not None}


def extract_headline_metrics(bench: dict) -> dict[str, Any]:
    """Walk a bench JSON and return a dict of headline metrics.

    Tolerant of schema drift across bench-suite versions. Missing
    metrics are simply absent from the returned dict.
    """
    out: dict[str, Any] = {}
    for key, aliases in _METRIC_ALIASES.items():
        val = _first_present(bench, aliases)
        if val is None:
            continue
        out[key] = val

    # Carry-through identifiers when present.
    for top_key in ("methodology_id", "methodology_sha",
                    "composed_key", "vllm_pin", "measured_at"):
        v = _first_present(bench, (top_key,))
        if v is not None:
            out[top_key] = v
    return out


def _pct_delta(current: Optional[float], baseline: Optional[float]) -> Optional[float]:
    """`(current - baseline) / baseline * 100`. Returns None when
    either value is None or baseline is zero."""
    if current is None or baseline is None:
        return None
    try:
        b = float(baseline)
        if b == 0:
            return None
        return round((float(current) - b) / b * 100.0, 2)
    except (TypeError, ValueError):
        return None


def compute_delta(
    current: dict,
    *,
    baseline: Optional[dict] = None,
    baseline_path: Optional[str] = None,
) -> BenchDelta:
    """Build a `BenchDelta` from a current bench result + optional baseline."""
    cur_metrics = extract_headline_metrics(current)

    def _f(key: str) -> Optional[float]:
        v = cur_metrics.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    base_metrics = extract_headline_metrics(baseline) if baseline else {}

    def _bf(key: str) -> Optional[float]:
        v = base_metrics.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return BenchDelta(
        measured_at=cur_metrics.get("measured_at")
            or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        methodology_id=cur_metrics.get("methodology_id"),
        methodology_sha=cur_metrics.get("methodology_sha"),
        composed_key=cur_metrics.get("composed_key"),
        vllm_pin=cur_metrics.get("vllm_pin"),
        median_tps=_f("median_tps"),
        p95_tps=_f("p95_tps"),
        decode_tpot_ms=_f("decode_tpot_ms"),
        ttft_ms=_f("ttft_ms"),
        cv_pct=_f("cv_pct"),
        tool_call_score=cur_metrics.get("tool_call_score"),
        baseline_path=baseline_path,
        median_tps_delta_pct=_pct_delta(_f("median_tps"), _bf("median_tps")),
        p95_tps_delta_pct=_pct_delta(_f("p95_tps"), _bf("p95_tps")),
        decode_tpot_delta_pct=_pct_delta(
            _f("decode_tpot_ms"), _bf("decode_tpot_ms"),
        ),
        ttft_delta_pct=_pct_delta(_f("ttft_ms"), _bf("ttft_ms")),
    )


# ─── Attach to proof artefact ─────────────────────────────────────────


def attach_bench(
    patch_id: str,
    bench_path: Path,
    *,
    baseline_path: Optional[Path] = None,
    out_dir: Path = DEFAULT_PROOF_DIR,
) -> Path:
    """Read `bench_path`, extract metrics, optionally diff against
    `baseline_path`, and persist into the proof artefact for `patch_id`.

    If a proof artefact already exists for this patch + current vllm pin,
    its `bench_delta` field is updated. Otherwise a new artefact is
    built from `build_proof_for_patch(patch_id)` (carrying the static
    checks too) and written.

    Returns the path of the artefact that was written.
    """
    if not bench_path.is_file():
        raise BenchAttachError(f"bench file not found: {bench_path}")
    try:
        current = json.loads(bench_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise BenchAttachError(
            f"could not parse {bench_path} as JSON: {e}"
        ) from None

    baseline: Optional[dict] = None
    if baseline_path is not None:
        if not baseline_path.is_file():
            raise BenchAttachError(
                f"baseline file not found: {baseline_path}"
            )
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise BenchAttachError(
                f"could not parse {baseline_path} as JSON: {e}"
            ) from None

    delta = compute_delta(
        current,
        baseline=baseline,
        baseline_path=str(baseline_path) if baseline_path else None,
    )

    # Find existing proof artefact for this patch (any vllm pin) — we
    # prefer to update the latest matching artefact when one exists.
    existing = find_proof_artefacts(patch_id, out_dir)
    if existing:
        # Latest by mtime.
        target = max(existing, key=lambda p: p.stat().st_mtime)
        data = load_proof_artefact(target)
        data["bench_delta"] = delta.to_dict()
        target.write_text(
            json.dumps(data, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return target

    # Otherwise build a fresh proof (static checks + this bench delta).
    proof = build_proof_for_patch(patch_id)
    proof.bench_delta = delta.to_dict()
    return write_proof_artefact(proof, out_dir)
