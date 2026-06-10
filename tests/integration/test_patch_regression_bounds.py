# SPDX-License-Identifier: Apache-2.0
"""S2.1 audit closure (2026-05-08 noonghunna): regression bench harness.

CI-gate that runs ``genesis_bench_suite.py --quick`` against a live test
server and asserts current results don't regress beyond a budget vs the
committed baseline JSON.

GATED by ``GENESIS_INTEGRATION_ENDPOINT`` env var. Without it the entire
module is skipped (Mac dev / no-server CI hosts pass cleanly). Set it to
the bench server URL to engage:

    GENESIS_INTEGRATION_ENDPOINT=http://test-rig:8000/v1 \\
    GENESIS_INTEGRATION_API_KEY=genesis-local \\
    GENESIS_INTEGRATION_MODEL=qwen3.6-35b-a3b \\
    GENESIS_INTEGRATION_BASELINE=tests/integration/baselines/35b_v8only.json \\
    pytest tests/integration/test_patch_regression_bounds.py

Per-metric tolerance (default 5%) configurable via
``GENESIS_INTEGRATION_REGRESSION_BUDGET=3.0``.

Each test asserts a single metric stays within budget vs baseline so a
regression on (say) tool-call quality fails the specific test rather
than masking which metric regressed.

Author: Sandermage; S2.1 / Sprint 2 closure 2026-05-09.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from sndr.cli.legacy.bench_compare import render_json


# ─── Gating ────────────────────────────────────────────────────────────


def _endpoint() -> str | None:
    return os.environ.get("GENESIS_INTEGRATION_ENDPOINT")


def _api_key() -> str:
    return os.environ.get("GENESIS_INTEGRATION_API_KEY", "genesis-local")


def _model() -> str | None:
    return os.environ.get("GENESIS_INTEGRATION_MODEL")


def _baseline_path() -> Path | None:
    raw = os.environ.get("GENESIS_INTEGRATION_BASELINE")
    if not raw:
        return None
    p = Path(raw)
    if not p.exists():
        return None
    return p


def _regression_budget(metric: str | None = None) -> float:
    """Per-metric regression tolerance (%).

    TTFT is inherently jittery on real GPU workloads (CV ~30-40% typical
    for the 8-sample bench window). A 5% budget there generates routine
    false-positives; bump TTFT specifically to 15%. TPS/TPOT/tool stay
    at the default 5%.

    Override via env:
      - `GENESIS_INTEGRATION_REGRESSION_BUDGET`           (default for all)
      - `GENESIS_INTEGRATION_REGRESSION_BUDGET_TTFT`      (TTFT-specific)
      - `GENESIS_INTEGRATION_REGRESSION_BUDGET_<METRIC>`  (any metric name)
    """
    default = "5.0"
    if metric and metric.upper() in {"TTFT_MS", "TTFT"}:
        default = "15.0"  # TTFT high-CV exemption
    if metric:
        env_name = f"GENESIS_INTEGRATION_REGRESSION_BUDGET_{metric.upper()}"
        raw = os.environ.get(env_name)
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
    try:
        return float(os.environ.get(
            "GENESIS_INTEGRATION_REGRESSION_BUDGET", default,
        ))
    except (TypeError, ValueError):
        return float(default)


pytestmark = pytest.mark.skipif(
    _endpoint() is None,
    reason=(
        "Integration test gated on GENESIS_INTEGRATION_ENDPOINT — set to "
        "your bench server URL to engage. See module docstring."
    ),
)


# ─── Bench runner ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def candidate_bench(tmp_path_factory) -> dict:
    """Run ``genesis_bench_suite.py --quick`` once, share the JSON across
    all per-metric tests in the module."""
    endpoint = _endpoint()
    api_key = _api_key()
    model = _model()
    repo_root = Path(__file__).resolve().parents[2]
    bench_script = repo_root / "tools" / "genesis_bench_suite.py"
    if not bench_script.exists():
        pytest.fail(f"bench script not found: {bench_script}")

    out_dir = tmp_path_factory.mktemp("regression_bench")
    out_json = out_dir / "candidate.json"
    out_md = out_dir / "candidate.md"

    # Parse host/port out of endpoint (e.g. http://server:8000/v1)
    from urllib.parse import urlparse
    parsed = urlparse(endpoint)
    host = parsed.hostname or "127.0.0.1"
    port = str(parsed.port or 8000)

    cmd = [
        sys.executable, str(bench_script),
        "--host", host, "--port", port,
        "--api-key", api_key,
        "--quick",
        "--skip-stress",
        "--skip-ctx-probe",
        "--name", "regression_bench",
        "--out", str(out_json),
        "--md", str(out_md),
    ]
    if model:
        cmd.extend(["--model", model])

    proc = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=600,  # 10 min cap
    )
    if proc.returncode != 0:
        pytest.fail(
            f"bench failed exit={proc.returncode}\n"
            f"stderr:\n{proc.stderr[-2000:]}"
        )

    if not out_json.exists():
        pytest.fail(f"bench did not emit JSON: {out_json}")

    return json.loads(out_json.read_text())


@pytest.fixture(scope="module")
def baseline_bench() -> dict:
    p = _baseline_path()
    if p is None:
        pytest.fail(
            "GENESIS_INTEGRATION_BASELINE must point at a committed "
            "baseline JSON (e.g. tests/integration/baselines/35b_v8only.json)"
        )
    return json.loads(p.read_text())


@pytest.fixture(scope="module")
def comparison(baseline_bench, candidate_bench) -> dict:
    """Structured A/B comparison via bench_compare.render_json. All
    per-metric tests inspect this shared comparison."""
    return render_json(
        baseline_bench, candidate_bench,
        baseline_bench.get("name", "baseline"),
        candidate_bench.get("name", "candidate"),
        regression_budget=_regression_budget(),
    )


# ─── Per-metric regression assertions ──────────────────────────────────


class TestNoDecodeTPOTRegression:
    """decode_TPOT is the primary, response-length-invariant metric.
    Any regression beyond budget is a real per-token decode slowdown."""

    def test_decode_tpot_within_budget(self, comparison):
        m = comparison["metrics"]["decode_TPOT_ms"]
        assert "REGRESS" not in m["verdict"], (
            f"decode_TPOT_ms regressed beyond budget: "
            f"{m['a']:.4f} → {m['b']:.4f} ({m['delta_pct']:+.2f}%) "
            f"verdict={m['verdict']}"
        )


class TestNoToolCallQualityRegression:
    """tool_pass_pct must not drop. Any regression here is a USER-VISIBLE
    quality cliff (tool_call request fails on a previously-passing case)."""

    def test_tool_pass_pct_within_budget(self, comparison):
        m = comparison["metrics"]["tool_pass_pct"]
        if m["a"] is None or m["b"] is None:
            pytest.skip("baseline or candidate has no tool_call results")
        assert "REGRESS" not in m["verdict"], (
            f"tool-call pass rate regressed: "
            f"{m['a']}% → {m['b']}% ({m['delta_pct']:+.2f}%)"
        )


class TestNoTtftRegression:
    """TTFT is operator-visible latency but has inherently high CV
    (~30-40% over 8 samples). Per-metric budget defaults to 15% to keep
    routine jitter from generating false-positive failures; override via
    `GENESIS_INTEGRATION_REGRESSION_BUDGET_TTFT_MS`."""

    def test_ttft_within_budget(self, comparison):
        m = comparison["metrics"]["TTFT_ms"]
        if m["a"] is None or m["b"] is None:
            pytest.skip("TTFT not measured in one of the runs")
        budget = _regression_budget("TTFT_MS")
        delta = m["delta_pct"]
        # TTFT is "higher = worse" so positive delta = regression.
        assert delta <= budget, (
            f"TTFT_ms regressed beyond budget {budget}%: "
            f"{m['a']:.2f} → {m['b']:.2f} ms ({delta:+.2f}%)"
        )


class TestNoAcceptRateRegression:
    """spec_accept_rate is the spec-decode signal. Drop here = wasted work."""

    def test_accept_rate_within_budget(self, comparison):
        m = comparison["metrics"]["spec_accept_rate"]
        if m["a"] is None or m["b"] is None:
            pytest.skip("accept_rate not captured in one of the runs")
        assert "REGRESS" not in m["verdict"], (
            f"spec_accept_rate regressed: "
            f"{m['a']:.4f} → {m['b']:.4f} ({m['delta_pct']:+.2f}%)"
        )


class TestStabilityCvNotExploding:
    """CV jumping 2× usually means a CUDA graph dispatch miss (PN16 V1
    pattern) — flag it loudly."""

    def test_decode_tpot_cv_within_budget(self, comparison):
        m = comparison["metrics"]["decode_TPOT_cv"]
        if m["a"] is None or m["b"] is None:
            pytest.skip("CV not measured")
        # CV regression: candidate CV > baseline CV by more than budget %
        # (stricter — CV doubling is severe)
        assert "REGRESS" not in m["verdict"], (
            f"decode_TPOT CV regressed (variance up): "
            f"{m['a']:.4f} → {m['b']:.4f} ({m['delta_pct']:+.2f}%)"
        )


# ─── Aggregate gate ────────────────────────────────────────────────────


class TestNoRegressionAcrossAnyMetric:
    """Defense in depth: even if a single per-metric test had a bug, this
    test asserts the structured comparison's any_regression flag, but
    excludes high-CV metrics (TTFT) where 5% gates routinely false-fire."""

    # High-variance metrics that have their own per-metric budget tests
    # above and should not also fire this aggregate gate at the default 5%.
    _NOISY_METRICS = {"TTFT_ms"}

    def test_any_regression_flag(self, comparison):
        # Re-evaluate per-metric with the per-metric budget; ignore the
        # render_json default-budget verdict for high-CV metrics.
        failing = []
        for k, v in comparison["metrics"].items():
            if "REGRESS" not in v["verdict"]:
                continue
            if k in self._NOISY_METRICS:
                budget = _regression_budget(k)
                delta = v.get("delta_pct")
                if delta is None or abs(delta) <= budget:
                    continue
                failing.append(f"{k}: {delta:+.2f}% > budget {budget}%")
                continue
            failing.append(f"{k}: {v['verdict']}")
        if failing:
            pytest.fail(
                "regression detected across one or more metrics:\n  "
                + "\n  ".join(failing)
            )
