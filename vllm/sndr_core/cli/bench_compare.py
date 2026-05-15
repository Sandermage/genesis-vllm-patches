# SPDX-License-Identifier: Apache-2.0
"""``sndr bench compare A.json B.json`` — A/B harness ergonomics.

S2.5 (audit closure 2026-05-08 noonghunna): operator-friendly multi-metric
comparison of two ``genesis_bench_suite.py`` JSON outputs.

Existing ``genesis_bench_suite --compare A B`` reports only decode_TPOT
delta. This thin CLI wrapper extends the comparison to a side-by-side
table covering the metrics operators actually care about during sweeps:

  • decode_TPOT_ms (primary, length-invariant)
  • wall_TPS (throughput proxy, response-length sensitive)
  • TTFT_ms (first-token latency)
  • spec_accept_rate (when bench captured it)
  • tool_call.passed_positive (regression guard)
  • stability_cv

Output: human-readable table by default, ``--json`` emits structured
JSON suitable for CI gates.

Usage:

    sndr bench compare a.json b.json
    sndr bench compare a.json b.json --json   # machine-readable
    sndr bench compare a.json b.json --regression-budget 5.0   # gate

Exit code: 0 if no regression beyond ``--regression-budget`` (default 5%),
2 if any tracked metric regresses by more than the budget.

Author: Sandermage; S2.5 / Audit Sprint 2 closure 2026-05-09.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ─── Metric extraction ─────────────────────────────────────────────────


def _extract_decode(d: dict, key: str) -> float | None:
    """Pull a decode_bench metric (mean) by key."""
    db = d.get("decode_bench") or {}
    block = db.get(key)
    if isinstance(block, dict):
        return block.get("mean")
    return None


def _extract_ttft(d: dict) -> float | None:
    db = d.get("decode_bench") or {}
    blk = db.get("TTFT_ms")
    if isinstance(blk, dict):
        return blk.get("mean")
    return None


def _extract_accept_rate(d: dict) -> float | None:
    """Multi-turn accept rate window — the most reliable accept signal."""
    mt = d.get("multi_turn") or {}
    return mt.get("window_accept_rate")


def _extract_tool_pass(d: dict) -> tuple[int, int] | None:
    tc = d.get("tool_call") or {}
    return (tc.get("passed_positive"), tc.get("total_positive"))


def _extract_cv(d: dict, key: str) -> float | None:
    db = d.get("decode_bench") or {}
    block = db.get(key)
    if isinstance(block, dict):
        return block.get("cv")
    return None


# ─── Comparison rows ───────────────────────────────────────────────────


_METRIC_SPEC = [
    # (label, extractor, "lower_better" or "higher_better", show_unit)
    ("decode_TPOT_ms", lambda d: _extract_decode(d, "decode_TPOT_ms"),
     "lower_better", "ms"),
    ("wall_TPS", lambda d: _extract_decode(d, "wall_TPS"),
     "higher_better", "tok/s"),
    ("TTFT_ms", _extract_ttft, "lower_better", "ms"),
    ("spec_accept_rate", _extract_accept_rate, "higher_better", ""),
    ("tool_pass_pct", lambda d: _tool_pct(d), "higher_better", "%"),
    ("decode_TPOT_cv", lambda d: _extract_cv(d, "decode_TPOT_ms"),
     "lower_better", ""),
]


def _tool_pct(d: dict) -> float | None:
    tp = _extract_tool_pass(d)
    if not tp or tp[1] in (None, 0):
        return None
    return round(100.0 * tp[0] / tp[1], 2)


def _delta_pct(a: float | None, b: float | None) -> float | None:
    """Return percent change from A → B. None if either is missing/zero."""
    if a is None or b is None:
        return None
    if a == 0:
        return None
    return round(100.0 * (b - a) / a, 2)


def _verdict(direction: str, delta_pct: float | None,
             budget: float) -> str:
    """Short verdict string for a single metric.

    Noise band is ``min(1.0, budget * 0.5)`` so that when an operator
    sets a strict budget (e.g. 0.1%) the noise band shrinks and
    sub-budget regressions are still flagged. Default budget=5.0 →
    noise band 1.0% (sensible default for production sweeps)."""
    if delta_pct is None:
        return "—"
    noise_band = min(1.0, max(0.05, budget * 0.5))
    if abs(delta_pct) < noise_band:
        return "≈ noise"
    if direction == "lower_better":
        if delta_pct < -budget:
            return f"WIN {abs(delta_pct):.1f}%"
        if delta_pct > budget:
            return f"REGRESS {delta_pct:.1f}%"
        return f"~{delta_pct:+.1f}%"
    else:
        if delta_pct > budget:
            return f"WIN +{delta_pct:.1f}%"
        if delta_pct < -budget:
            return f"REGRESS {delta_pct:.1f}%"
        return f"~{delta_pct:+.1f}%"


# ─── Render ────────────────────────────────────────────────────────────


def render_human(a: dict, b: dict, a_name: str, b_name: str,
                 regression_budget: float = 5.0) -> tuple[str, bool]:
    """Format a side-by-side table; return (text, has_regression)."""
    rows = []
    has_regression = False

    rows.append(f"## A: {a_name}")
    rows.append(f"## B: {b_name}\n")
    rows.append(
        f"{'Metric':<22} {'A':>14} {'B':>14} {'Δ%':>10}  "
        f"{'verdict':<18}"
    )
    rows.append("-" * 80)

    for label, extractor, direction, unit in _METRIC_SPEC:
        va = extractor(a)
        vb = extractor(b)
        d = _delta_pct(va, vb)
        v = _verdict(direction, d, regression_budget)
        if "REGRESS" in v:
            has_regression = True

        def fmt(x):
            if x is None:
                return "n/a"
            if isinstance(x, float):
                return f"{x:.4f}{unit}" if unit else f"{x:.4f}"
            return str(x)

        d_str = f"{d:+.2f}%" if d is not None else "—"
        rows.append(
            f"{label:<22} {fmt(va):>14} {fmt(vb):>14} "
            f"{d_str:>10}  {v:<18}"
        )

    return "\n".join(rows), has_regression


def render_json(a: dict, b: dict, a_name: str, b_name: str,
                regression_budget: float) -> dict:
    """Structured comparison output for CI/dashboards."""
    out = {
        "a_name": a_name,
        "b_name": b_name,
        "regression_budget_pct": regression_budget,
        "metrics": {},
        "any_regression": False,
    }
    for label, extractor, direction, unit in _METRIC_SPEC:
        va = extractor(a)
        vb = extractor(b)
        d = _delta_pct(va, vb)
        v = _verdict(direction, d, regression_budget)
        if "REGRESS" in v:
            out["any_regression"] = True
        out["metrics"][label] = {
            "a": va,
            "b": vb,
            "delta_pct": d,
            "direction": direction,
            "unit": unit,
            "verdict": v,
        }
    return out


# ─── CLI ──────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sndr bench compare",
        description=(
            "A/B comparison of two genesis_bench_suite.py JSON outputs. "
            "Reports decode_TPOT (primary), wall_TPS, TTFT, accept_rate, "
            "tool-call quality, and stability CV. Exit code 2 on "
            "regression beyond --regression-budget."
        ),
    )
    p.add_argument("a_path", help="Path to baseline bench JSON")
    p.add_argument("b_path", help="Path to candidate bench JSON")
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON instead of human table",
    )
    p.add_argument(
        "--regression-budget",
        type=float,
        default=5.0,
        help=(
            "Percent regression tolerance per metric before exit code 2. "
            "Default 5.0%%."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    a_path = Path(args.a_path)
    b_path = Path(args.b_path)

    if not a_path.exists():
        print(f"ERROR: baseline JSON not found: {a_path}", file=sys.stderr)
        return 1
    if not b_path.exists():
        print(f"ERROR: candidate JSON not found: {b_path}", file=sys.stderr)
        return 1

    a = json.loads(a_path.read_text())
    b = json.loads(b_path.read_text())

    a_name = a.get("name") or a_path.stem
    b_name = b.get("name") or b_path.stem

    if args.json:
        out = render_json(a, b, a_name, b_name, args.regression_budget)
        print(json.dumps(out, indent=2))
        return 2 if out["any_regression"] else 0

    text, has_regression = render_human(
        a, b, a_name, b_name, args.regression_budget,
    )
    print(text)
    print()
    if has_regression:
        print(
            f"⚠ Regression detected beyond --regression-budget "
            f"{args.regression_budget}%. Exit code 2."
        )
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
