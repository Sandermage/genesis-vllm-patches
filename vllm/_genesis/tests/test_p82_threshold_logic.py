# SPDX-License-Identifier: Apache-2.0
"""Unit tests for P82 — SGLang threshold_single OR-clause acceptance.

Pure-Python text-patch generator. No GPU / vLLM dependency — exercises
threshold parsing, replacement-builder, and apply() decision tree.

Covers:
  - threshold parsing: empty → default 0.3, valid float, garbage → default,
    out-of-range → clamped to [0.0, 1.0]
  - replacement contains BOTH the vanilla rule AND the threshold OR-clause
  - replacement preserves the upstream NOTE comment (anchor uniqueness)
  - threshold is baked as a Python float literal (not a runtime env read)
  - apply() returns 'skipped' when env not set or threshold == 0.0
  - marker is versioned so re-applies after bump aren't no-ops
"""
from __future__ import annotations

import re

import pytest

from vllm._genesis.wiring.patch_82_sglang_acceptance_threshold import (
    GENESIS_P82_MARKER,
    GENESIS_P82_MARKER_PREFIX,
    P82_OLD,
    _build_replacement,
    _marker_for,
    _read_threshold,
    _DEFAULT_THRESHOLD,
    apply,
)


# ─── threshold parsing ──────────────────────────────────────────────────


def test_threshold_default_when_unset(monkeypatch):
    monkeypatch.delenv("GENESIS_P82_THRESHOLD_SINGLE", raising=False)
    assert _read_threshold() == _DEFAULT_THRESHOLD


def test_threshold_default_when_empty(monkeypatch):
    monkeypatch.setenv("GENESIS_P82_THRESHOLD_SINGLE", "")
    assert _read_threshold() == _DEFAULT_THRESHOLD


def test_threshold_default_when_garbage(monkeypatch):
    monkeypatch.setenv("GENESIS_P82_THRESHOLD_SINGLE", "not_a_float")
    assert _read_threshold() == _DEFAULT_THRESHOLD


def test_threshold_valid_float(monkeypatch):
    monkeypatch.setenv("GENESIS_P82_THRESHOLD_SINGLE", "0.42")
    assert _read_threshold() == pytest.approx(0.42)


def test_threshold_clamped_above_one(monkeypatch):
    monkeypatch.setenv("GENESIS_P82_THRESHOLD_SINGLE", "1.5")
    assert _read_threshold() == 1.0


def test_threshold_clamped_below_zero(monkeypatch):
    monkeypatch.setenv("GENESIS_P82_THRESHOLD_SINGLE", "-0.1")
    assert _read_threshold() == 0.0


def test_threshold_at_bounds(monkeypatch):
    monkeypatch.setenv("GENESIS_P82_THRESHOLD_SINGLE", "0.0")
    assert _read_threshold() == 0.0
    monkeypatch.setenv("GENESIS_P82_THRESHOLD_SINGLE", "1.0")
    assert _read_threshold() == 1.0


# ─── replacement builder ────────────────────────────────────────────────


def test_replacement_preserves_upstream_NOTE_comment():
    """Anchor uniqueness depends on keeping the NOTE(woosuk) comment."""
    rep = _build_replacement(0.3)
    assert "NOTE(woosuk)" in rep, (
        "replacement must preserve upstream NOTE for anchor uniqueness"
    )
    assert "draft probability should never be 0" in rep


def test_replacement_contains_vanilla_rule():
    rep = _build_replacement(0.3)
    assert "draft_prob > 0 and target_prob / draft_prob >= uniform_prob" in rep, (
        "vanilla rejection-sample rule must be preserved in OR-clause"
    )


def test_replacement_contains_threshold_clause():
    rep = _build_replacement(0.42)
    assert "_genesis_p82_threshold = target_prob >=" in rep, (
        "threshold-side of the OR-clause must reference target_prob"
    )
    # Threshold value baked as a Python literal:
    assert "0.42" in rep, "threshold value 0.42 must appear as a literal"


def test_replacement_uses_OR_combinator():
    rep = _build_replacement(0.3)
    # Final acceptance is `vanilla OR threshold`
    assert re.search(
        r"accepted\s*=\s*_genesis_p82_vanilla\s+or\s+_genesis_p82_threshold",
        rep,
    ), "final 'accepted' assignment must combine vanilla OR threshold"


def test_replacement_carries_genesis_breadcrumb():
    rep = _build_replacement(0.3)
    assert "[Genesis P82" in rep, (
        "replacement must include `[Genesis P82` breadcrumb for drift detection"
    )


def test_replacement_threshold_is_baked_not_env_read():
    """Threshold must be a literal, not an env read at runtime (perf)."""
    rep = _build_replacement(0.3)
    assert "os.environ" not in rep, (
        "replacement must NOT contain runtime env reads — threshold is baked at apply()"
    )


# ─── threshold values produce different literals ─────────────────────────


@pytest.mark.parametrize("threshold", [0.1, 0.25, 0.3, 0.5, 0.7])
def test_replacement_distinct_per_threshold(threshold):
    rep = _build_replacement(threshold)
    # The numeric repr should appear; we don't lock to Python's exact repr
    # form, just that the rounded value shows up.
    rounded = f"{threshold:.4f}".rstrip("0").rstrip(".")
    assert (
        repr(threshold) in rep or rounded in rep
    ), f"threshold {threshold!r} should appear as a literal in the replacement"


# ─── anchor invariants ──────────────────────────────────────────────────


def test_anchor_is_three_lines_with_NOTE():
    """The 3-line anchor including NOTE comment + assignment is needed for
    uniqueness in rejection_sampler.py.
    """
    lines = P82_OLD.split("\n")
    # 3 logical lines + trailing newline = 4 entries when split by \n
    assert "NOTE(woosuk)" in lines[0]
    assert "we check it to avoid NaNs" in lines[1]
    assert "draft_prob > 0 and target_prob / draft_prob >= uniform_prob" in lines[2]


def test_anchor_long_enough_for_uniqueness():
    """Heuristic: an anchor under 100 chars risks ambiguity."""
    assert len(P82_OLD) >= 100, (
        f"anchor too short ({len(P82_OLD)}) — risk of multi-match"
    )


# ─── marker invariants ──────────────────────────────────────────────────


def test_marker_prefix_versioned():
    """Prefix should embed v7.62.11 (the version that fixed B3 — marker
    now encodes the threshold value)."""
    assert "v7.62.11" in GENESIS_P82_MARKER_PREFIX, (
        f"P82 marker prefix {GENESIS_P82_MARKER_PREFIX!r} should embed v7.62.11"
    )
    assert "Genesis P82" in GENESIS_P82_MARKER_PREFIX


# ─── B3 fix: marker encodes threshold ─────────────────────────────────


def test_marker_for_encodes_threshold():
    """The marker must include the threshold so a different bake forces
    re-apply (not silent IDEMPOTENT skip)."""
    m1 = _marker_for(0.30)
    m2 = _marker_for(0.50)
    assert m1 != m2, (
        "Markers for different thresholds must differ (B3 fix). "
        f"Got identical {m1!r} for both 0.30 and 0.50"
    )
    assert "thresh=0.3000" in m1
    assert "thresh=0.5000" in m2


def test_marker_for_stable_to_float_repr():
    """0.30000000000000004 should produce the same marker as 0.3 — round to
    4 decimals avoids spurious re-applies from float representation noise.
    """
    a = _marker_for(0.3)
    b = _marker_for(0.3 + 1e-16)
    assert a == b, (
        f"Marker should be float-repr stable (round to 4 decimals). "
        f"Got {a!r} vs {b!r}"
    )


def test_marker_for_starts_with_prefix():
    """Drift detection still works: the prefix is stable across thresholds."""
    for th in (0.1, 0.3, 0.5, 0.7, 0.999):
        m = _marker_for(th)
        assert m.startswith(GENESIS_P82_MARKER_PREFIX), (
            f"_marker_for({th}) = {m!r} must start with {GENESIS_P82_MARKER_PREFIX!r}"
        )


# ─── apply() decision tree ──────────────────────────────────────────────


def test_apply_skipped_when_disabled(monkeypatch):
    """Without GENESIS_ENABLE_P82=1, dispatcher should reject the apply."""
    monkeypatch.delenv("GENESIS_ENABLE_P82", raising=False)
    status, reason = apply()
    assert status == "skipped", (
        f"P82 should skip when GENESIS_ENABLE_P82 unset; got {status!r}: {reason}"
    )


def test_apply_skipped_when_threshold_zero(monkeypatch):
    """threshold=0.0 → OR-clause never fires → skip patch entirely (keep
    source vanilla). Avoids paying patch overhead for a no-op rule.

    Skipped on dev hosts without vllm installed (apply() short-circuits
    earlier on `vllm_install_root() is None`).
    """
    from vllm._genesis.guards import vllm_install_root
    if vllm_install_root() is None:
        pytest.skip("vllm not installed — apply() short-circuits before threshold check")
    monkeypatch.setenv("GENESIS_ENABLE_P82", "1")
    monkeypatch.setenv("GENESIS_P82_THRESHOLD_SINGLE", "0.0")
    status, reason = apply()
    assert status == "skipped", (
        f"P82 should skip when threshold==0.0; got {status!r}: {reason}"
    )
    assert "0.0" in reason or "OR clause would never fire" in reason
