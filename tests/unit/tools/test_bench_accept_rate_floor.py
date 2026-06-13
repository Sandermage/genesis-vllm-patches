# SPDX-License-Identifier: Apache-2.0
"""TDD for the spec-decode accept-rate floor in genesis_bench_suite.py.

Companion of PN380 (vendor of vllm#44943) — roadmap chunk-4 Theme 2
"loud startup" validation family. A partially-loaded MTP draft (the
#44943 quantized-MTP failure mode) boots fine and serves fine; the only
externally-visible symptom is the spec-decode accept rate collapsing
(upstream A/B: 65.0% -> 41.9%, ~ -15-20% decode TPS at K=3). The bench
suite already computes the windowed accept rate
(``accept_rate_window.window_accept_rate`` — delta of the Prometheus
counters ``vllm:spec_decode_num_accepted_tokens_total`` /
``vllm:spec_decode_num_draft_tokens_total`` across the run); this
feature adds a floor check (default 0.55 per the roadmap) that emits a
loud WARN verdict when the window rate lands below it.

Floor rationale: healthy Qwen3.6 MTP K=3 runs measure ~0.65-0.78
per-drafted-token acceptance (upstream #44943 A/B + the vLLM recipe
numbers); the corrupted-load mode measures ~0.42. 0.55 splits the two
populations with margin on both sides.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def bench_module():
    """Load the bench module via spec_from_file_location since it lives
    outside the package tree (canonical home: sndr/extras/tools/)."""
    repo_root = Path(__file__).resolve().parents[3]
    bench_path = repo_root / "sndr" / "extras" / "tools" / "genesis_bench_suite.py"
    assert bench_path.is_file(), f"bench suite not found at {bench_path}"
    spec = importlib.util.spec_from_file_location("gbs_floor", bench_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestFloorConstant:
    def test_floor_constant_exists_and_matches_roadmap(self, bench_module):
        assert bench_module.SPEC_DECODE_ACCEPT_RATE_FLOOR == 0.55


class TestEvaluateAcceptRateFloor:
    def test_helper_exists(self, bench_module):
        assert callable(bench_module.evaluate_accept_rate_floor)

    def test_none_rate_is_not_checked(self, bench_module):
        """No spec decode active (or metrics unavailable) — N/A, never
        a false WARN."""
        v = bench_module.evaluate_accept_rate_floor(None)
        assert v["checked"] is False
        assert v["verdict"] == "N/A"
        assert v["floor"] == 0.55

    def test_rate_below_floor_warns(self, bench_module):
        """The #44943 corrupted-load population (~0.42)."""
        v = bench_module.evaluate_accept_rate_floor(0.419)
        assert v["checked"] is True
        assert v["verdict"] == "WARN"
        assert v["window_accept_rate"] == 0.419

    def test_rate_at_floor_passes(self, bench_module):
        v = bench_module.evaluate_accept_rate_floor(0.55)
        assert v["verdict"] == "PASS"

    def test_healthy_rate_passes(self, bench_module):
        """Healthy Qwen3.6 MTP K=3 population (~0.65-0.78)."""
        v = bench_module.evaluate_accept_rate_floor(0.78)
        assert v["verdict"] == "PASS"

    def test_custom_floor_honored(self, bench_module):
        v = bench_module.evaluate_accept_rate_floor(0.60, floor=0.65)
        assert v["verdict"] == "WARN"
        assert v["floor"] == 0.65

    def test_result_shape_is_json_serializable(self, bench_module):
        import json

        for rate in (None, 0.42, 0.7):
            json.dumps(bench_module.evaluate_accept_rate_floor(rate))


class TestCliFlag:
    def test_accept_rate_floor_flag_default(self, bench_module):
        args = bench_module.parse_args(["--quick"])
        assert args.accept_rate_floor == 0.55

    def test_accept_rate_floor_flag_override(self, bench_module):
        args = bench_module.parse_args(["--accept-rate-floor", "0.6"])
        assert args.accept_rate_floor == 0.6
