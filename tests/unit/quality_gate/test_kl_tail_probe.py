# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the KL-divergence tail quality probe (tools/quality_gate).

These pin the part that is easy to get subtly wrong and that a live rig run
cannot cheaply re-verify: the per-token KL math, the percentile-tail computation,
the capture-row normalisation, and the threshold verdict. All on synthetic
distributions with analytically known KL — no GPU, no model, no rig.

The MEASUREMENT (capturing two per-token distributions from a live engine at two
KV dtypes over the same prompts) is the rig-follow-up; this suite proves the
offline core so the rig step only has to produce the captures.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from quality_gate import kl_tail_probe as klp  # noqa: E402


# ---------------------------------------------------------------------------
# KL math — checked against analytically known values.
# ---------------------------------------------------------------------------
def test_kl_of_identical_distributions_is_zero() -> None:
    p = [0.1, 0.2, 0.3, 0.4]
    assert klp.kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)


def test_kl_two_point_distribution_matches_closed_form() -> None:
    # Bernoulli KL: D(P||Q) = p*ln(p/q) + (1-p)*ln((1-p)/(1-q)).
    p = [0.9, 0.1]
    q = [0.5, 0.5]
    expected = 0.9 * math.log(0.9 / 0.5) + 0.1 * math.log(0.1 / 0.5)
    assert klp.kl_divergence(p, q) == pytest.approx(expected, rel=1e-12)


def test_kl_uniform_vs_skewed_matches_closed_form() -> None:
    # D(uniform || skewed) over 4 symbols, fully analytic.
    p = [0.25, 0.25, 0.25, 0.25]
    q = [0.4, 0.3, 0.2, 0.1]
    expected = sum(0.25 * math.log(0.25 / qi) for qi in q)
    assert klp.kl_divergence(p, q) == pytest.approx(expected, rel=1e-12)


def test_kl_is_asymmetric() -> None:
    p = [0.9, 0.1]
    q = [0.5, 0.5]
    assert klp.kl_divergence(p, q) != pytest.approx(klp.kl_divergence(q, p))


def test_kl_renormalises_unnormalised_inputs() -> None:
    # A top-k row that doesn't sum to 1 is renormalised, so [9,1] == [0.9,0.1].
    assert klp.kl_divergence([9, 1], [5, 5]) == pytest.approx(
        klp.kl_divergence([0.9, 0.1], [0.5, 0.5]), rel=1e-12
    )


def test_kl_handles_zero_in_reference_by_convention() -> None:
    # 0 * log(0/q) == 0: a token the reference never emits contributes nothing.
    p = [0.0, 1.0]
    q = [0.5, 0.5]
    assert klp.kl_divergence(p, q) == pytest.approx(math.log(1.0 / 0.5), rel=1e-12)


def test_kl_zero_in_candidate_is_floored_not_infinite() -> None:
    # Q=0 where P>0 would be +inf; the EPS floor keeps it large-but-finite — this
    # is exactly the tail token (candidate dropped all mass) the probe must catch.
    val = klp.kl_divergence([1.0, 0.0], [0.0, 1.0])
    assert math.isfinite(val)
    assert val > 20.0  # ~ -log(EPS); a huge, gate-tripping spike, not inf.


def test_kl_rejects_length_mismatch_and_bad_mass() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        klp.kl_divergence([0.5, 0.5], [1.0])
    with pytest.raises(ValueError, match="non-positive total mass"):
        klp.kl_divergence([0.0, 0.0], [0.5, 0.5])
    with pytest.raises(ValueError, match="negative probability"):
        klp.kl_divergence([1.2, -0.2], [0.5, 0.5])


# ---------------------------------------------------------------------------
# Percentile — checked against the numpy 'linear' interpolation contract.
# ---------------------------------------------------------------------------
def test_percentile_endpoints_and_median() -> None:
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert klp.percentile(xs, 0) == 0.0
    assert klp.percentile(xs, 100) == 4.0
    assert klp.percentile(xs, 50) == 2.0


def test_percentile_interpolates_linearly() -> None:
    xs = [10.0, 20.0]
    # rank = 0.999 * (2-1) = 0.999 -> 10 + 0.999*(20-10) = 19.99
    assert klp.percentile(xs, 99.9) == pytest.approx(19.99, rel=1e-9)


def test_percentile_empty_is_zero() -> None:
    assert klp.percentile([], 99.9) == 0.0


# ---------------------------------------------------------------------------
# Capture-row normalisation — the three documented shapes.
# ---------------------------------------------------------------------------
def test_row_to_probs_probs_shape() -> None:
    assert klp.row_to_probs({"probs": [0.2, 0.8]}) == [0.2, 0.8]


def test_row_to_probs_logits_are_softmaxed() -> None:
    probs = klp.row_to_probs({"logits": [0.0, 0.0]})
    assert probs == pytest.approx([0.5, 0.5])
    # softmax is shift-invariant.
    assert klp.row_to_probs({"logits": [5.0, 5.0]}) == pytest.approx([0.5, 0.5])


def test_row_to_probs_logprobs_are_exponentiated() -> None:
    probs = klp.row_to_probs({"logprobs": {"a": math.log(0.7), "b": math.log(0.3)}})
    assert probs == pytest.approx([0.7, 0.3])


def test_row_to_probs_rejects_unknown_shape() -> None:
    with pytest.raises(ValueError, match="none of"):
        klp.row_to_probs({"foo": [1, 2]})


# ---------------------------------------------------------------------------
# compute_kl_tail — the verdict and the tail focus.
# ---------------------------------------------------------------------------
def test_identical_runs_have_zero_tail_no_regression() -> None:
    dists = [[0.7, 0.2, 0.1] for _ in range(1000)]
    rep = klp.compute_kl_tail(dists, dists, threshold=0.10)
    assert rep.n_tokens == 1000
    assert rep.headline_kl == pytest.approx(0.0, abs=1e-12)
    assert rep.tail["99.9"] == pytest.approx(0.0, abs=1e-12)
    assert rep.regression is False


def test_tail_drives_verdict_not_mean() -> None:
    # 9950 perfectly-matched tokens + 50 badly-diverged tail tokens (0.5% of the
    # sample — wider than the top-0.1% slice the 99.9-pctile reports, so p99.9
    # lands SOLIDLY inside the diverged tail rather than on its boundary). The
    # MEAN stays tiny but the 99.9-pctile catches the tail -> regression. This is
    # the whole point: needle / mean-style checks miss this (recall does not
    # depend on the low-prob tail); the tail metric does not.
    n_clean = 9950
    n_diverged = 50
    ref = [[0.5, 0.5] for _ in range(n_clean + n_diverged)]
    cand = [[0.5, 0.5] for _ in range(n_clean)] + [
        [0.999, 0.001] for _ in range(n_diverged)
    ]
    rep = klp.compute_kl_tail(ref, cand, threshold=0.10, headline_pct=99.9)
    # A MEAN-based gate would PASS (mean << threshold) — the rare tail is washed
    # out — while the TAIL-based gate FAILS. That divergence is the whole point.
    assert rep.mean_kl < rep.threshold
    assert rep.mean_kl < rep.tail["99.9"] / 100  # tail dwarfs the mean
    assert rep.tail["99.9"] > 0.10  # the tail surfaces it
    assert rep.tail["95"] == pytest.approx(0.0, abs=1e-9)  # body is clean
    assert rep.regression is True
    assert "TAIL REGRESSION" in rep.detail


def test_single_outlier_in_thousand_sits_above_p999() -> None:
    # Correctness guard on the percentile contract itself: ONE diverged token in
    # 1000 sits ABOVE the 99.9-pctile (rank 0.999*(N-1)=998.001), so p99.9 does
    # NOT trip on a lone spike at N=1000 — that needs the max or a denser tail.
    # Documents the sample-size dependence so the rig sizes captures accordingly.
    ref = [[0.5, 0.5] for _ in range(1000)]
    cand = [[0.5, 0.5] for _ in range(999)] + [[0.999, 0.001]]
    rep = klp.compute_kl_tail(ref, cand, threshold=0.10, headline_pct=99.9)
    assert rep.max_kl > 0.10  # the spike IS in the data (max sees it)
    assert rep.tail["99.9"] < 0.10  # but one-in-1000 is above the 99.9-pctile


def test_below_threshold_passes() -> None:
    # A uniformly small per-token divergence stays under threshold at the tail.
    ref = [[0.5, 0.5] for _ in range(1000)]
    cand = [[0.51, 0.49] for _ in range(1000)]
    rep = klp.compute_kl_tail(ref, cand, threshold=0.10)
    assert rep.regression is False
    assert rep.tail["99.9"] < 0.10


def test_length_mismatch_truncates_with_warning() -> None:
    ref = [[0.5, 0.5] for _ in range(10)]
    cand = [[0.5, 0.5] for _ in range(7)]  # candidate early-stopped
    rep = klp.compute_kl_tail(ref, cand)
    assert rep.n_tokens == 7
    assert any("truncated" in w for w in rep.warnings)


def test_empty_inputs_are_well_defined_no_regression() -> None:
    rep = klp.compute_kl_tail([], [])
    assert rep.n_tokens == 0
    assert rep.regression is False
    assert rep.warnings


def test_report_serialises_to_dict() -> None:
    rep = klp.compute_kl_tail([[0.6, 0.4]], [[0.6, 0.4]])
    d = rep.as_dict()
    assert set(d) >= {"n_tokens", "tail", "threshold", "headline_kl", "regression"}
    assert isinstance(d["tail"], dict)


# ---------------------------------------------------------------------------
# Offline --from-captures path — the rig-follow-up consumes these.
# ---------------------------------------------------------------------------
def test_from_captures_round_trip(tmp_path) -> None:
    ref_path = tmp_path / "ref.jsonl"
    cand_path = tmp_path / "cand.jsonl"
    # Reference: confident bf16 distribution every position.
    ref_rows = [{"probs": [0.8, 0.15, 0.05]} for _ in range(500)]
    # Candidate: matches except one tail position where it drops the top token.
    cand_rows = [{"probs": [0.8, 0.15, 0.05]} for _ in range(499)] + [
        {"probs": [0.05, 0.15, 0.80]}
    ]
    ref_path.write_text("\n".join(json.dumps(r) for r in ref_rows), encoding="utf-8")
    cand_path.write_text("\n".join(json.dumps(r) for r in cand_rows), encoding="utf-8")
    rep = klp.compute_kl_tail_from_captures(
        str(ref_path), str(cand_path), threshold=0.10
    )
    assert rep.n_tokens == 500
    assert rep.regression is True  # the single dropped-top tail token trips it


def test_from_captures_mixed_shapes(tmp_path) -> None:
    # ref as logits, cand as probs — both normalise to the same distribution.
    ref_path = tmp_path / "ref.jsonl"
    cand_path = tmp_path / "cand.jsonl"
    ref_path.write_text(json.dumps({"logits": [0.0, 0.0]}), encoding="utf-8")
    cand_path.write_text(json.dumps({"probs": [0.5, 0.5]}), encoding="utf-8")
    rep = klp.compute_kl_tail_from_captures(str(ref_path), str(cand_path))
    assert rep.headline_kl == pytest.approx(0.0, abs=1e-12)
    assert rep.regression is False


def test_from_captures_rejects_bad_json(tmp_path) -> None:
    ref_path = tmp_path / "ref.jsonl"
    cand_path = tmp_path / "cand.jsonl"
    ref_path.write_text("{not json}", encoding="utf-8")
    cand_path.write_text(json.dumps({"probs": [0.5, 0.5]}), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        klp.compute_kl_tail_from_captures(str(ref_path), str(cand_path))


# ---------------------------------------------------------------------------
# CLI exit-code contract — 1 on regression, 0 clean (so a driver can gate).
# ---------------------------------------------------------------------------
def test_main_exit_code_signals_regression(tmp_path, capsys) -> None:
    ref_path = tmp_path / "ref.jsonl"
    cand_path = tmp_path / "cand.jsonl"
    ref_path.write_text(
        "\n".join(json.dumps({"probs": [0.5, 0.5]}) for _ in range(100)),
        encoding="utf-8",
    )
    # All-clean candidate -> exit 0.
    cand_path.write_text(
        "\n".join(json.dumps({"probs": [0.5, 0.5]}) for _ in range(100)),
        encoding="utf-8",
    )
    rc = klp.main(["--from-captures", str(ref_path), str(cand_path), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["regression"] is False

    # One catastrophic tail token -> exit 1.
    rows = [{"probs": [0.5, 0.5]} for _ in range(99)] + [{"probs": [0.999, 0.001]}]
    cand_path.write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )
    rc = klp.main(
        ["--from-captures", str(ref_path), str(cand_path), "--threshold", "0.10"]
    )
    assert rc == 1
