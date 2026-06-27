# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Genesis quality-gate core (tools/quality_gate).

These pin the load-bearing logic that a full live run cannot cheaply re-verify:
probe-payload generation (NIAH ladder, tool-prefill, IDE-agent, multi-turn,
ceiling ladder), recall checking, and the PASS / WARN / FAIL / SKIP verdict
mapping — including the Genesis-specific cliff/patch attribution and the soak
"PASS != load-bearing" attribution-delta. No GPU / live engine is needed.

A full live run still needs the rig; this suite proves the request shapes and
the verdict thresholds are correct so the harness fails for the RIGHT reasons.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

# tools/quality_gate is a package under tools/ (a namespace dir). Put tools/ on
# the path the same way the runner CLI does, then import the package.
REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from quality_gate import probes, soak  # noqa: E402


# ---------------------------------------------------------------------------
# NIAH probe generation.
# ---------------------------------------------------------------------------
def test_niah_secret_is_three_tokens_and_deterministic_with_seed() -> None:
    s1 = probes.make_niah_secret(random.Random(0))
    s2 = probes.make_niah_secret(random.Random(0))
    assert s1 == s2  # seeded determinism
    parts = s1.split()
    assert len(parts) == 3
    assert parts[0] in probes._NIAH_COLORS
    assert parts[1] in probes._NIAH_ANIMALS
    assert 10 <= int(parts[2]) <= 99


def test_niah_request_embeds_secret_and_scales_with_filler() -> None:
    secret = "crimson otter 42"
    small = probes.make_niah_request("m", 100, secret=secret)
    large = probes.make_niah_request("m", 400, secret=secret)
    small_text = small["messages"][0]["content"]
    large_text = large["messages"][0]["content"]
    # Needle present, placed mid-document (before the trailing question).
    assert secret in small_text
    assert "The hidden phrase is" in small_text
    # More filler -> strictly longer prompt.
    assert len(large_text) > len(small_text)
    # Deterministic shape: thinking disabled, temp 0.
    assert small["temperature"] == 0.0
    assert small["chat_template_kwargs"] == {"enable_thinking": False}


def test_niah_request_places_needle_in_the_middle() -> None:
    secret = "emerald narwhal 70"
    req = probes.make_niah_request("m", 200, secret=secret)
    text = req["messages"][0]["content"]
    idx = text.index(secret)
    # The needle should sit roughly mid-document, not at either extreme.
    frac = idx / len(text)
    assert 0.2 < frac < 0.8


# ---------------------------------------------------------------------------
# Ceiling ladder + scale calibration.
# ---------------------------------------------------------------------------
def test_ceiling_ladder_spans_start_to_fraction_of_n_ctx_inclusive() -> None:
    rungs = probes.ceiling_ladder_rungs(
        262_144, start_tokens=95_000, step_tokens=30_000, fraction=0.92
    )
    assert rungs[0] == 95_000
    top = int(262_144 * 0.92)
    assert rungs[-1] == top
    # monotonic increasing
    assert all(b > a for a, b in zip(rungs, rungs[1:], strict=False))


def test_ceiling_ladder_empty_when_ctx_too_small() -> None:
    # n_ctx whose 0.92 ceiling is below the start -> nothing to ladder.
    assert probes.ceiling_ladder_rungs(80_000, start_tokens=95_000) == []
    assert probes.ceiling_ladder_rungs(0) == []


def test_scale_for_target_tokens_uses_calibrated_ratio_and_floors_at_100() -> None:
    # 95000 tokens at 65 tok/scale -> ~1461 scale.
    assert probes.scale_for_target_tokens(95_000, 65.0) == 1461
    # tiny target floors at 100 (never a degenerate prompt).
    assert probes.scale_for_target_tokens(10, 65.0) == 100
    with pytest.raises(ValueError, match="tok_per_scale_unit"):
        probes.scale_for_target_tokens(1000, 0)


# ---------------------------------------------------------------------------
# Tool-prefill / IDE-agent / multi-turn shapes.
# ---------------------------------------------------------------------------
def test_tool_prefill_request_has_large_tool_message_and_auto_choice() -> None:
    req = probes.make_tool_prefill_request("m", target_chars=40_000)
    tool_msg = next(msg for msg in req["messages"] if msg["role"] == "tool")
    assert len(tool_msg["content"]) >= 40_000
    assert req["tool_choice"] == "auto"
    assert req["tools"][0]["function"]["name"] == "fetch_news"


def test_ide_agent_forces_content_path_with_ten_tools() -> None:
    req = probes.make_ide_agent_request("m")
    # tool_choice=none forces the long-reasoning/code path that surfaces the
    # inductor FFN leak (Cliff 1 mech B) deterministically.
    assert req["tool_choice"] == "none"
    assert len(req["tools"]) == 10
    # System preamble is intentionally bulked up (x5) to the multi-K-char shape
    # that triggers the inductor FFN leak; ~2.8K chars on the current text.
    assert len(req["messages"][0]["content"]) > 2500


def test_multiturn_request_is_four_turn_with_tool_reply() -> None:
    req = probes.make_multiturn_request("m")
    roles = [m["role"] for m in req["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "user"]
    assert req["tool_choice"] == "auto"


# ---------------------------------------------------------------------------
# Recall.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("secret", "content", "expected"),
    [
        ("crimson otter 42", "The hidden phrase is Crimson Otter 42.", True),
        ("crimson otter 42", "crimson  OTTER\n42", True),  # spacing/case robust
        ("crimson otter 42", "crimson otter", False),  # missing number token
        ("emerald narwhal 70", "I do not recall the phrase.", False),
    ],
)
def test_recall_ok(secret: str, content: str, expected: bool) -> None:
    assert probes.recall_ok(secret, content) is expected


# ---------------------------------------------------------------------------
# Failure classification -> Genesis cliff + patch.
# ---------------------------------------------------------------------------
def test_classify_failure_maps_probe_and_code_to_cliff() -> None:
    # GDN single-prompt OOM on a large NIAH / ceiling rung -> Cliff 2 / P103.
    ref = probes.classify_failure("ceiling", 500)
    assert ref is not None
    assert ref.cliff.startswith("Cliff 2")
    assert ref.patch == "P103"
    # Tool-prefill / IDE-agent 500 -> FA2 activation peak / PN17.
    ref = probes.classify_failure("tool_prefill", 500)
    assert ref is not None
    assert ref.cliff == "Cliff 1"
    assert ref.patch == "PN17"
    # Multi-turn 500 -> Cliff 2b / PN59.
    ref = probes.classify_failure("multiturn", 500)
    assert ref is not None
    assert ref.cliff == "Cliff 2b"
    assert ref.patch == "PN59"
    # No response at all -> engine-dead.
    ref = probes.classify_failure("reasoning", 0)
    assert ref is not None
    assert ref.cliff == "engine-down"
    # Healthy -> no cliff.
    assert probes.classify_failure("multiturn", 200) is None


# ---------------------------------------------------------------------------
# NIAH rung verdicts.
# ---------------------------------------------------------------------------
def test_verdict_longctx_pass_warn_skip_fail() -> None:
    secret = "amber falcon 51"
    # PASS — recalled.
    v = probes.verdict_longctx_rung("longctx_small", 200, secret, f"answer: {secret}")
    assert v.status == "PASS"
    assert not v.cliff

    # WARN — HTTP 200 but recall miss (attention-quality ceiling, not a fault).
    v = probes.verdict_longctx_rung("longctx_small", 200, secret, "amber falcon")
    assert v.status == "WARN"

    # SKIP — engine rejected above max-model-len.
    v = probes.verdict_longctx_rung("ceiling", 400, secret, "")
    assert v.status == "SKIP"

    # FAIL — 500 carries the GDN cliff/patch (large rung).
    v = probes.verdict_longctx_rung("longctx_large", 500, secret, "")
    assert v.status == "FAIL"
    assert v.cliff.startswith("Cliff 2")
    assert v.patch == "P103"
    assert v.remediation


def test_verdict_oversize_400_distinguishes_sizing_bug_from_rejection() -> None:
    # target below n_ctx but 400 -> our filler overshot (harness FAIL).
    v = probes.verdict_oversize_400("ceiling", 120_000, 262_144)
    assert v.status == "FAIL"
    assert "sizing overshot" in v.detail
    # target at/above n_ctx -> legitimate engine rejection (SKIP).
    v = probes.verdict_oversize_400("ceiling", 300_000, 262_144)
    assert v.status == "SKIP"


# ---------------------------------------------------------------------------
# HTTP probe verdicts (tool-prefill / ide / multiturn / reasoning).
# ---------------------------------------------------------------------------
def test_verdict_http_probe_pass_on_text_or_toolcall() -> None:
    v = probes.verdict_http_probe(
        "ide_agent", 200, content_len=400, completion_tokens=120
    )
    assert v.status == "PASS"
    v = probes.verdict_http_probe("multiturn", 200, content_len=0, tool_calls=1)
    assert v.status == "PASS"


def test_verdict_http_probe_silent_empty_is_fail_with_cliff() -> None:
    v = probes.verdict_http_probe(
        "tool_prefill", 200, content_len=0, tool_calls=0, finish_reason="stop"
    )
    assert v.status == "FAIL"
    assert v.cliff == "silent-empty"
    assert v.remediation


def test_verdict_http_probe_reasoning_short_generation_warns() -> None:
    # min_tokens guards spec-decode AL-collapse: a 200 OK with too-few tokens.
    v = probes.verdict_http_probe(
        "reasoning", 200, content_len=400, completion_tokens=120, min_tokens=500
    )
    assert v.status == "WARN"
    assert "collapse" in v.detail


def test_verdict_http_probe_500_maps_to_gdn_for_lcb() -> None:
    v = probes.verdict_http_probe("lcb_coding", 500)
    assert v.status == "FAIL"
    assert v.cliff.startswith("Cliff 2")
    assert v.patch == "P103"


# ---------------------------------------------------------------------------
# Soak: continuous ramp fixtures.
# ---------------------------------------------------------------------------
def test_continuous_turns_are_five_and_ramp_tool_results() -> None:
    assert [t["turn"] for t in soak.CONTINUOUS_TURNS] == [1, 2, 3, 4, 5]
    # turn 1 has no preceding tool result; later turns synthesize growing ones.
    assert soak.CONTINUOUS_TURNS[0]["tool_synth"] is None
    sizes = [t["tool_synth"][2] for t in soak.CONTINUOUS_TURNS if t["tool_synth"]]
    assert sizes
    assert max(sizes) >= 32_000  # heaviest tool result by turn 5


def test_synth_filler_sizes_and_kinds() -> None:
    for kind in ("python_code", "grep_output", "command_output"):
        body = soak.synth_filler(kind, 5_000)
        assert len(body) == 5_000
    with pytest.raises(ValueError, match="unknown filler kind"):
        soak.synth_filler("nope", 1000)


def test_continuous_initial_state_has_system_and_counters() -> None:
    st = soak.continuous_initial_state(3)
    assert st["session_id"] == 3
    assert st["messages"][0]["role"] == "system"
    assert st["tool_calls_seen"] == 0


def test_turn_spec_lookup_and_bounds() -> None:
    assert soak.turn_spec(1)["turn"] == 1
    with pytest.raises(ValueError, match="no continuous turn spec"):
        soak.turn_spec(6)


# ---------------------------------------------------------------------------
# Soak verdict thresholds.
# ---------------------------------------------------------------------------
def _row(
    session, status=200, vram=21000, tps=110.0, ttft=500, comp=300, err="", t_ms=2000
):
    return {
        "session_id": session,
        "turn_id": 1,
        "t_ms": t_ms,
        "vram_mib": vram,
        "ttft_ms": ttft,
        "decode_tps": tps,
        "status": status,
        "error": err,
        "completion_tokens": comp,
    }


def test_soak_verdict_pass_when_clean() -> None:
    rows = [_row(1), _row(2), _row(3), _row(4), _row(5, vram=21100)]
    v = soak.compute_soak_verdict(rows, boot_vram_mib=21000, growth_limit_mib=200)
    assert v.verdict == "PASS"
    assert v.exit_code == 0
    assert v.growth_mib == 100


def test_soak_verdict_fail_on_vram_growth_cites_cliff2b() -> None:
    rows = [_row(1), _row(5, vram=24000)]  # +3000 MiB
    v = soak.compute_soak_verdict(rows, boot_vram_mib=21000, growth_limit_mib=200)
    assert v.verdict == "FAIL"
    assert v.exit_code == 1
    assert any("Cliff 2b" in f for f in v.failures)


def test_soak_verdict_fail_on_errors() -> None:
    rows = [_row(1), _row(5, status=500, err="OOM", comp=0)]
    v = soak.compute_soak_verdict(rows, boot_vram_mib=21000)
    assert v.verdict == "FAIL"


def test_soak_verdict_silent_empty_threshold() -> None:
    # >=50% silent-empty -> FAIL; <50% -> WARN.
    rows = [_row(1, comp=0), _row(2, comp=0), _row(3, comp=300), _row(4, comp=300)]
    v = soak.compute_soak_verdict(rows, boot_vram_mib=21000)
    assert v.silent_empty == 2
    assert v.verdict == "FAIL"  # exactly 50%

    rows = [_row(i) for i in range(1, 10)] + [_row(10, comp=0)]
    v = soak.compute_soak_verdict(rows, boot_vram_mib=21000)
    assert v.silent_empty == 1
    assert v.verdict == "PASS"
    assert any("silent-empty" in w for w in v.warnings)


def test_soak_verdict_tps_retention_fail() -> None:
    rows = [_row(i, tps=110.0) for i in range(1, 6)] + [
        _row(i, tps=50.0) for i in range(6, 11)
    ]
    v = soak.compute_soak_verdict(rows, boot_vram_mib=21000)
    # last-5 median (50) / first-5 median (110) ~= 45% < 80% -> FAIL.
    assert v.verdict == "FAIL"
    assert any("retention" in f for f in v.failures)


def test_soak_verdict_inconclusive_on_no_rows() -> None:
    v = soak.compute_soak_verdict([], boot_vram_mib=21000)
    assert v.verdict == "INCONCLUSIVE"
    assert v.exit_code == 2


# ---------------------------------------------------------------------------
# Attribution delta — the "PASS != load-bearing" discipline.
# ---------------------------------------------------------------------------
def _pass_verdict(growth=100):
    return soak.SoakVerdict(
        verdict="PASS",
        boot_vram_mib=21000,
        max_vram_mib=21000 + growth,
        growth_mib=growth,
        growth_limit_mib=200,
        sessions_completed=5,
        errors=0,
        silent_empty=0,
        total_turns=25,
        tps_retention_pct=99.0,
        ttft_ratio=1.0,
        p50_decode_tps=110.0,
        exit_code=0,
    )


def _fail_verdict():
    return soak.SoakVerdict(
        verdict="FAIL",
        boot_vram_mib=21000,
        max_vram_mib=23800,
        growth_mib=2800,
        growth_limit_mib=200,
        sessions_completed=2,
        errors=1,
        silent_empty=0,
        total_turns=12,
        tps_retention_pct=0.0,
        ttft_ratio=0.0,
        p50_decode_tps=0.0,
        failures=["VRAM grew 2800 MiB"],
        exit_code=1,
    )


def test_attribution_load_bearing_when_stripped_fails() -> None:
    res = soak.attribution_delta(
        _pass_verdict(), _fail_verdict(), patch="PN59", topology_tp=1
    )
    assert res.verdict == "LOAD_BEARING"
    assert "PN59" in res.detail


def test_attribution_topology_sidestep_when_both_pass_on_tp2() -> None:
    res = soak.attribution_delta(
        _pass_verdict(), _pass_verdict(), patch="PN59", topology_tp=2
    )
    assert res.verdict == "TOPOLOGY_SIDESTEP"
    assert "#140" in res.detail


def test_attribution_not_load_bearing_when_both_pass_on_tp1() -> None:
    res = soak.attribution_delta(
        _pass_verdict(), _pass_verdict(), patch="PN59", topology_tp=1
    )
    assert res.verdict == "NOT_LOAD_BEARING"


def test_attribution_inconclusive_when_on_run_failed() -> None:
    res = soak.attribution_delta(
        _fail_verdict(), _fail_verdict(), patch="PN59", topology_tp=1
    )
    assert res.verdict == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# club-3090 Cliff-2b verbatim gate: silent_empty==0, growth<200, retention>=98%,
# errors==0, on the 5 sessions x 5 turns ramp shape.
# ---------------------------------------------------------------------------
def _cliff2b_rows(
    *,
    sessions: int = 5,
    turns: int = 5,
    tps_first: float = 110.0,
    tps_last: float = 110.0,
    vram_last: int = 21100,
    boot_vram: int = 21000,
    silent_sessions: tuple[int, ...] = (),
    error_sessions: tuple[int, ...] = (),
):
    """Build a 5x5 ramp telemetry fixture. first session keeps tps_first, last
    keeps tps_last; the last session's VRAM is vram_last (growth = vram_last -
    boot). `silent_sessions` / `error_sessions` inject the named failure on turn 1
    of those sessions."""
    rows = []
    for s in range(1, sessions + 1):
        for t in range(1, turns + 1):
            tps = tps_first if s == 1 else (tps_last if s == sessions else 110.0)
            vram = vram_last if s == sessions else boot_vram
            comp = 300
            status = 200
            err = ""
            if t == 1 and s in silent_sessions:
                comp = 0
            if t == 1 and s in error_sessions:
                status = 500
                err = "OOM"
                comp = 0
            rows.append(
                _row(s, status=status, vram=vram, tps=tps, comp=comp, err=err)
            )
    return rows


def test_cliff2b_pass_on_clean_5x5_ramp() -> None:
    rows = _cliff2b_rows(vram_last=21100)  # +100 MiB, retention 100%, 0 silent
    v = soak.compute_cliff2b_verdict(rows, boot_vram_mib=21000)
    assert v.verdict == "PASS"
    assert v.exit_code == 0
    assert v.silent_empty == 0
    assert v.growth_mib == 100


def test_cliff2b_fails_retention_below_98pct_that_general_gate_passes() -> None:
    # 110 -> 105 = ~95.5% retention: PASSES the general 80% gate but FAILS the
    # strict Cliff-2b 98% gate. This is the headline tightening.
    rows = _cliff2b_rows(tps_first=110.0, tps_last=105.0)
    general = soak.compute_soak_verdict(rows, boot_vram_mib=21000)
    assert general.verdict == "PASS"  # 95.5% clears the 80% floor
    strict = soak.compute_cliff2b_verdict(rows, boot_vram_mib=21000)
    assert strict.verdict == "FAIL"
    assert any("98%" in f and "retention" in f for f in strict.failures)


def test_cliff2b_fails_any_single_silent_empty() -> None:
    # ONE silent-empty turn (1/25 = 4%) PASSES the general <50% rule but FAILS
    # the Cliff-2b silent_empty==0 rule.
    rows = _cliff2b_rows(silent_sessions=(3,))
    general = soak.compute_soak_verdict(rows, boot_vram_mib=21000)
    assert general.verdict == "PASS"
    assert general.silent_empty == 1
    strict = soak.compute_cliff2b_verdict(rows, boot_vram_mib=21000)
    assert strict.verdict == "FAIL"
    assert strict.silent_empty == 1
    assert any("silent-empty" in f for f in strict.failures)


def test_cliff2b_fails_vram_growth_over_200() -> None:
    rows = _cliff2b_rows(vram_last=21300)  # +300 MiB > 200
    v = soak.compute_cliff2b_verdict(rows, boot_vram_mib=21000)
    assert v.verdict == "FAIL"
    assert any("Cliff 2b" in f for f in v.failures)


def test_cliff2b_fails_on_any_error() -> None:
    rows = _cliff2b_rows(error_sessions=(4,))
    v = soak.compute_cliff2b_verdict(rows, boot_vram_mib=21000)
    assert v.verdict == "FAIL"
    assert v.errors >= 1


def test_cliff2b_silent_empty_counts_completion_tokens_not_decode_tps() -> None:
    # A fast tool-call turn has decode_tps==0 but completion_tokens>0 — it must
    # NOT be flagged silent-empty. Only genuine completion_tokens==0 counts.
    rows = _cliff2b_rows()
    # Turn with real output but zero decode_tps (fast tool call).
    rows.append(_row(2, tps=0.0, comp=42))
    v = soak.compute_cliff2b_verdict(rows, boot_vram_mib=21000, require_fixture_shape=False)
    assert v.silent_empty == 0
    assert v.verdict == "PASS"


def test_cliff2b_wrong_shape_fails_even_when_metrics_green() -> None:
    # 2 sessions x 5 turns: every metric is green, but the shape does NOT surface
    # Cliff 2b, so a PASS would be a false certification -> downgraded to FAIL.
    rows = _cliff2b_rows(sessions=2)
    v = soak.compute_cliff2b_verdict(rows, boot_vram_mib=21000)
    assert v.verdict == "FAIL"
    assert any("fixture shape" in f for f in v.failures)


def test_cliff2b_shape_check_can_be_disabled() -> None:
    # With require_fixture_shape=False a clean 2-session run PASSes (for unit
    # tests / partial smoke), so the shape gate is opt-out but ON by default.
    rows = _cliff2b_rows(sessions=2)
    v = soak.compute_cliff2b_verdict(
        rows, boot_vram_mib=21000, expected_sessions=2, require_fixture_shape=False
    )
    assert v.verdict == "PASS"


def test_validate_cliff2b_fixture_shape_detects_short_sessions() -> None:
    rows = _cliff2b_rows()  # full 5x5
    assert soak.validate_cliff2b_fixture_shape(rows) == []
    # Drop session 5 down to 2 rows (turns) -> short-session problem. The shape
    # validator counts rows per session_id (each turn is one row).
    s5 = [r for r in rows if r["session_id"] == 5]
    short = [r for r in rows if r["session_id"] != 5] + s5[:2]
    problems = soak.validate_cliff2b_fixture_shape(short)
    assert any("< 5 turns" in p for p in problems)


def test_cliff2b_threshold_constants_match_club3090() -> None:
    # Pin the verbatim thresholds so a future loosening is a visible diff.
    assert soak.CLIFF2B_RETENTION_FLOOR == 0.98
    assert soak.CLIFF2B_GROWTH_LIMIT_MIB == 200
    assert soak.CLIFF2B_SESSIONS == 5
    assert soak.CLIFF2B_TURNS_PER_SESSION == 5
