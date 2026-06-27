# SPDX-License-Identifier: Apache-2.0
"""Genesis quality-gate soak core — Cliff 2b multi-turn ramp + soak verdict.

Engine-agnostic, unit-testable core for `scripts/soak_continuous.sh`. Two jobs:

  1. Build the continuous (ramping-context) turn fixtures: a single multi-turn
     agentic-coding conversation that grows to ~22-25K accumulated tokens by
     turn 5 — the workload shape that triggers Cliff 2b (GDN multi-turn VRAM
     accretion). Fresh-mode (reset-each-turn) fixtures do NOT surface that
     accretion class, which is exactly why continuous mode exists.

  2. Compute the soak verdict from per-turn telemetry (VRAM growth, TPS
     retention, TTFT drift, silent-empty turns, errors) and — the Genesis
     extension — the patch-attribution DELTA between an overlays-ON run and an
     overlays-STRIPPED run, so a PASS can be proven load-bearing rather than
     assumed.

Provenance — adapted and extended from club-3090's `scripts/soak-helper.py` and
`scripts/soak-test.sh` (github.com/noonghunna/club-3090, MIT). Their
"PASS != patches load-bearing" discipline (issue #140) is adopted directly: a
clean soak on a config whose topology already sidesteps the failure mode (e.g.
TP=2) does not prove the overlay patches did the work. Their remedy — re-run
with overlays stripped and compare — is implemented here as
`attribution_delta()` and wired into the `--strip-overlays` driver mode.

Genesis extensions:
  * `attribution_delta()` — structured ON-vs-STRIPPED comparison that yields a
    LOAD_BEARING / NOT_LOAD_BEARING / TOPOLOGY_SIDESTEP verdict, naming the
    patch under test (e.g. PN59).
  * The silent-empty discriminator uses genuine completion_tokens==0 (Genesis
    soak telemetry always records it), and the failure cross-references the
    Genesis cliff map.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

# Mirror of the ramping coding conversation from club-3090's continuous mode.
# turn -> (user message, synthetic tool filler kind+size for the NEXT turn).
CONTINUOUS_SYSTEM = (
    "You are an autonomous coding assistant working inside a small Python "
    "service repository. The user is debugging a production issue. When file "
    "contents, search results, or command output would materially change your "
    "answer, call the appropriate tool — don't speculate. Keep responses "
    "concise; defer to the tools for raw data.\n\n"
    "Repository layout you can assume:\n"
    "  src/handlers.py  — webhook handler entry points\n"
    "  src/payloads.py  — payload validation + parsing\n"
    "  src/db.py        — database access layer\n"
    "  tests/           — pytest suite mirrors src/\n"
    "  logs/app.log     — recent service logs\n"
)


CONTINUOUS_TURNS: list[dict[str, Any]] = [
    {
        "turn": 1,
        "user": (
            "We're seeing a KeyError 'transaction_id' in production every few "
            "minutes when handle_webhook runs. Investigate the handler and find "
            "where this comes from. Start with src/handlers.py."
        ),
        "tool_synth": None,
        "max_tokens": 350,
        "temp": 0.3,
    },
    {
        "turn": 2,
        "user": "Now show me tests/test_handlers.py — I want to see if this case has a regression test.",
        "tool_synth": ("read_file", "python_code", 20_000),
        "max_tokens": 350,
        "temp": 0.25,
    },
    {
        "turn": 3,
        "user": (
            "Grep across the whole codebase for 'transaction_id' so we can see "
            "every place this key is used. Include test files and log lines."
        ),
        "tool_synth": ("read_file", "python_code", 24_000),
        "max_tokens": 400,
        "temp": 0.25,
    },
    {
        "turn": 4,
        "user": "Run the test suite and show the full failing-test output for the handler path.",
        "tool_synth": ("grep", "grep_output", 24_000),
        "max_tokens": 500,
        "temp": 0.3,
    },
    {
        "turn": 5,
        "user": (
            "Based on everything, write a fix for the KeyError and explain in "
            "4-6 bullets what was wrong, what change closes it, and what "
            "regression test to add. Fix as a code block first, then the prose."
        ),
        "tool_synth": ("run_command", "command_output", 32_000),
        "max_tokens": 1500,
        "temp": 0.35,
    },
]


_FILLER_PYTHON = (
    "def handle_webhook(payload, db_conn=None):\n"
    "    validated = validate_payload(payload)\n"
    "    if validated is None:\n"
    "        raise InvalidPayloadError('payload missing required fields')\n"
    "    txn = validated.get('transaction_id')\n"
    "    record = persist_record(db_conn, txn, validated.get('customer_id'),\n"
    "                            float(validated.get('amount', 0)))\n"
    "    notify_downstream(record)\n"
    "    return {'status': 'ok', 'record_id': record.id}\n\n"
)
_FILLER_GREP = (
    "src/handlers.py:14:    txn = validated.get('transaction_id')\n"
    "src/payloads.py:28:    REQUIRED_KEYS = ('transaction_id', 'customer_id', 'amount')\n"
    "src/db.py:88:    SELECT * FROM transactions WHERE transaction_id = %s\n"
    "tests/test_handlers.py:21:    payload = {'transaction_id': 'txn_001', ...}\n"
    "logs/app.log:142:KeyError: 'transaction_id' at handlers.py:14\n"
)
_FILLER_PYTEST = (
    "tests/test_handlers.py::test_happy_path PASSED                        [  7%]\n"
    "tests/test_handlers.py::test_missing_amount FAILED                    [ 14%]\n"
    "=================================== FAILURES ===================================\n"
    "    def test_missing_amount():\n"
    "        payload = {'transaction_id': 'txn_002', 'customer_id': 'c2'}\n"
    ">       result = handle_webhook(payload)\n"
    "E       KeyError: 'transaction_id'\n"
    "src/handlers.py:14: KeyError\n\n"
)


def synth_filler(kind: str, target_chars: int) -> str:
    """Generate a plausible synthetic tool-result body of ~target_chars.

    The growing tool results are what make accumulated context reach the
    ~22-25K-token Cliff-2b territory by turn 5 regardless of the model's actual
    tool-use behaviour.
    """
    block = {
        "python_code": _FILLER_PYTHON,
        "grep_output": _FILLER_GREP,
        "command_output": _FILLER_PYTEST,
    }.get(kind)
    if block is None:
        raise ValueError(f"unknown filler kind: {kind}")
    repeats = (target_chars // len(block)) + 1
    return (block * repeats)[:target_chars]


def continuous_initial_state(session: int) -> dict[str, Any]:
    """Initial conversation state for a continuous session (system + tools, no
    user yet). Tool-call accounting fields support the fallback-synthesis path."""
    return {
        "session_id": int(session),
        "messages": [{"role": "system", "content": CONTINUOUS_SYSTEM}],
        "tool_calls_seen": 0,
        "fallback_tool_calls_synthesized": 0,
    }


def turn_spec(turn: int) -> dict[str, Any]:
    """Return the fixture spec for a 1-based turn index (1..5)."""
    for spec in CONTINUOUS_TURNS:
        if spec["turn"] == turn:
            return spec
    raise ValueError(f"no continuous turn spec for turn={turn} (valid 1..5)")


# ---------------------------------------------------------------------------
# Soak verdict.
# ---------------------------------------------------------------------------
def _median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def _realistic_tps(t: float) -> bool:
    # Filter the streaming divide-by-tiny artifact (ttft ~= wall) that yields
    # spurious thousands-of-tps values.
    return 0 < t <= 500


@dataclass
class SoakVerdict:
    verdict: str  # PASS / WARN-bearing in failures/warnings lists
    boot_vram_mib: int
    max_vram_mib: int
    growth_mib: int
    growth_limit_mib: int
    sessions_completed: int
    errors: int
    silent_empty: int
    total_turns: int
    tps_retention_pct: float
    ttft_ratio: float
    p50_decode_tps: float
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    exit_code: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_soak_verdict(
    rows: list[dict[str, Any]],
    *,
    boot_vram_mib: int,
    growth_limit_mib: int = 200,
    expected_sessions: int = 5,
    timed_out: bool = False,
    retention_floor: float = 0.80,
    silent_empty_fail_any: bool = False,
    retention_basis: str = "session",
) -> SoakVerdict:
    """Compute the soak verdict from per-turn telemetry rows.

    Each row needs: session_id, t_ms, vram_mib, ttft_ms, decode_tps, status,
    error, completion_tokens. PASS == no failure signal: errors==0, VRAM growth
    within limit, TPS retention >= ``retention_floor``, silent-empty under the
    configured tolerance. The Cliff-2b fingerprint is monotone VRAM growth across
    the ramp; the silent-empty discriminator is genuine completion_tokens==0 (NOT
    the decode_tps==0 proxy, which false-flags fast tool-call turns).

    Thresholds (the two club-3090 Cliff-2b knobs are parameterised so the strict
    Cliff-2b gate and the general soak share one verdict core):
      * ``retention_floor`` — minimum first-5-vs-last-5 decode-TPS retention. The
        general soak uses 0.80; club-3090's verbatim Cliff-2b gate uses 0.98
        (``compute_cliff2b_verdict``).
      * ``silent_empty_fail_any`` — when True, ANY silent-empty turn FAILs (the
        Cliff-2b ``silent_empty == 0`` rule); when False, the general soak's
        graded rule applies (>=50% FAIL, else WARN).
      * ``retention_basis`` — ``"session"`` (default) computes retention over the
        first-5 vs last-5 SESSIONS (the general multi-session soak); ``"turn"``
        computes it over the first-5 vs last-5 telemetry rows (TURNS) in arrival
        order — club-3090's Cliff-2b semantics, the only basis that resolves the
        within-run accretion bleed on the single 5x5 ramp (5 sessions make the
        session-basis first/last sets identical, so retention would be a vacuous
        100%).

    PASS semantics deliberately do NOT assert the overlay patches are
    load-bearing — that requires attribution_delta() against a stripped run.
    """
    if not rows:
        return SoakVerdict(
            verdict="INCONCLUSIVE",
            boot_vram_mib=boot_vram_mib,
            max_vram_mib=boot_vram_mib,
            growth_mib=0,
            growth_limit_mib=growth_limit_mib,
            sessions_completed=0,
            errors=0,
            silent_empty=0,
            total_turns=0,
            tps_retention_pct=0.0,
            ttft_ratio=0.0,
            p50_decode_tps=0.0,
            warnings=["no completed turns"],
            exit_code=2,
        )

    sessions = sorted({int(r["session_id"]) for r in rows})
    first, last = sessions[:5], sessions[-5:]

    def tps_of(subset: list[int]) -> list[float]:
        return [
            float(r["decode_tps"])
            for r in rows
            if int(r["session_id"]) in subset and _realistic_tps(float(r["decode_tps"]))
        ]

    def ttft_of(subset: list[int]) -> list[float]:
        return [
            float(r["ttft_ms"])
            for r in rows
            if int(r["session_id"]) in subset and float(r["ttft_ms"]) > 0
        ]

    def realistic_tps_rows() -> list[float]:
        """Per-turn decode-TPS in arrival order (one entry per telemetry row)."""
        return [
            float(r["decode_tps"])
            for r in rows
            if _realistic_tps(float(r["decode_tps"]))
        ]

    if retention_basis == "turn":
        # club-3090 Cliff-2b basis: first-5 vs last-5 TURNS of the whole run, so
        # the within-run accretion bleed is visible on a single 5x5 ramp (where
        # the session-basis first/last-5 sets coincide and retention is vacuous).
        turn_tps = realistic_tps_rows()
        first_med = _median(turn_tps[:5])
        last_med = _median(turn_tps[-5:])
    else:
        first_med = _median(tps_of(first))
        last_med = _median(tps_of(last))
    retention = (last_med / first_med) if first_med > 0 else 0.0
    ttft_ratio = (
        _median(ttft_of(last)) / _median(ttft_of(first))
        if _median(ttft_of(first)) > 0
        else 0.0
    )
    max_vram = max([int(r["vram_mib"]) for r in rows] + [boot_vram_mib])
    growth = max_vram - boot_vram_mib

    errors = [r for r in rows if int(r["status"]) != 200 or r.get("error")]

    def is_silent_empty(r: dict[str, Any]) -> bool:
        if int(r["status"]) != 200 or r.get("error") or int(r["t_ms"]) < 1000:
            return False
        return int(r.get("completion_tokens", 0)) == 0

    silent = [r for r in rows if is_silent_empty(r)]
    silent_pct = 100.0 * len(silent) / len(rows)
    all_tps = [
        float(r["decode_tps"]) for r in rows if _realistic_tps(float(r["decode_tps"]))
    ]

    failures: list[str] = []
    warnings: list[str] = []
    if errors:
        failures.append(f"{len(errors)} request(s) returned non-200 or stream error.")
    if growth > growth_limit_mib:
        failures.append(
            f"VRAM grew {growth} MiB > {growth_limit_mib} MiB threshold "
            "(Cliff 2b accretion fingerprint — see PN59)."
        )
    if first_med > 0 and retention < retention_floor:
        failures.append(
            f"Decode TPS retention {retention * 100:.1f}% < "
            f"{retention_floor * 100:.0f}%."
        )
    elif first_med == 0:
        warnings.append("No positive decode TPS samples; retention not evaluated.")
    if ttft_ratio > 1.5:
        warnings.append(
            f"TTFT grew {ttft_ratio:.2f}x first->last (Cliff 3 prefill scaling)."
        )
    if silent:
        msg = (
            f"{len(silent)}/{len(rows)} turns ({silent_pct:.1f}%) returned HTTP 200 "
            "with empty completion (silent-empty)."
        )
        # Cliff-2b gate: ANY silent-empty turn FAILs. General soak: graded.
        is_fail = silent_empty_fail_any or silent_pct >= 50.0
        (failures if is_fail else warnings).append(msg)
    if sessions and sessions[-1] < expected_sessions:
        warnings.append(
            f"Only {sessions[-1]} of {expected_sessions} sessions completed."
        )

    verdict = "INCONCLUSIVE" if timed_out else ("FAIL" if failures else "PASS")
    exit_code = 2 if timed_out else (1 if failures else 0)

    return SoakVerdict(
        verdict=verdict,
        boot_vram_mib=boot_vram_mib,
        max_vram_mib=max_vram,
        growth_mib=growth,
        growth_limit_mib=growth_limit_mib,
        sessions_completed=len(sessions),
        errors=len(errors),
        silent_empty=len(silent),
        total_turns=len(rows),
        tps_retention_pct=round(retention * 100, 1),
        ttft_ratio=round(ttft_ratio, 2),
        p50_decode_tps=round(_median(sorted(all_tps)), 2),
        failures=failures,
        warnings=warnings,
        exit_code=exit_code,
    )


# ---------------------------------------------------------------------------
# club-3090 Cliff-2b PASS gate — the verbatim thresholds.
#
# club-3090's continuous soak (issue #182 / #22 Cliff-2b recipe) gates a PASS on
# four hard thresholds. Our general compute_soak_verdict deliberately runs looser
# defaults (80% retention, graded silent-empty) so it does not false-FAIL a
# noisy-but-stable run; the Cliff-2b GATE pins club-3090's exact bars:
#
#   silent_empty == 0     counted by completion_tokens (NOT decode_tps): a single
#                         HTTP-200-with-0-completion turn FAILs.
#   VRAM growth  < 200 MiB from the WARM baseline (after session 1).
#   TPS retention >= 98%  first-5 vs last-5 turns (the 80% general floor is far
#                         too loose to catch the slow GDN accretion bleed).
#   errors       == 0     no non-200 / stream error.
#
# The ONLY workload that surfaces Cliff 2b is the 5 sessions x 5 turns
# ramp-to-~25K-accumulated-context fixture (CONTINUOUS_TURNS). A clean run on any
# other shape does not certify Cliff-2b safety, so the gate validates the shape.
# ---------------------------------------------------------------------------
CLIFF2B_RETENTION_FLOOR = 0.98
CLIFF2B_GROWTH_LIMIT_MIB = 200
CLIFF2B_SESSIONS = 5
CLIFF2B_TURNS_PER_SESSION = 5


def validate_cliff2b_fixture_shape(
    rows: list[dict[str, Any]],
    *,
    expected_sessions: int = CLIFF2B_SESSIONS,
    expected_turns_per_session: int = CLIFF2B_TURNS_PER_SESSION,
) -> list[str]:
    """Return shape problems for the Cliff-2b ramp fixture; [] == correct shape.

    Cliff 2b is GDN multi-turn VRAM accretion — it only surfaces under the
    5 sessions x 5 turns ramp-to-~25K-accumulated-context conversation. A soak run
    on a different shape (too few sessions, too few turns/session) can PASS while
    NEVER exercising the accretion path, so the gate must confirm the shape before
    trusting a green verdict. PURE + unit-tested; the live capture is the
    rig-follow-up.
    """
    problems: list[str] = []
    by_session: dict[int, int] = {}
    for r in rows:
        by_session[int(r["session_id"])] = by_session.get(int(r["session_id"]), 0) + 1
    n_sessions = len(by_session)
    if n_sessions < expected_sessions:
        problems.append(
            f"only {n_sessions} sessions (need {expected_sessions}); the "
            f"{expected_sessions}x{expected_turns_per_session} ramp is the ONLY "
            "shape that surfaces Cliff 2b."
        )
    short = {s: c for s, c in by_session.items() if c < expected_turns_per_session}
    if short:
        problems.append(
            f"sessions with < {expected_turns_per_session} turns: "
            f"{sorted(short)} — the per-session ramp to ~25K accumulated context "
            "is incomplete, so accretion is not exercised."
        )
    return problems


def compute_cliff2b_verdict(
    rows: list[dict[str, Any]],
    *,
    boot_vram_mib: int,
    expected_sessions: int = CLIFF2B_SESSIONS,
    timed_out: bool = False,
    require_fixture_shape: bool = True,
) -> SoakVerdict:
    """Compute the soak verdict under club-3090's verbatim Cliff-2b PASS gate.

    Pins the four hard thresholds (silent_empty==0, growth<200 MiB, retention>=98%,
    errors==0) by delegating to compute_soak_verdict with the strict knobs, then —
    when ``require_fixture_shape`` is set — FAILs if the telemetry was not the
    5x5 ramp that is the only shape surfacing Cliff 2b. A clean run on the wrong
    shape is downgraded to FAIL with a shape diagnostic, NOT silently passed.

    This is the gate to use for an actual Cliff-2b certification; the looser
    compute_soak_verdict stays the general end-to-end stability check.
    """
    verdict = compute_soak_verdict(
        rows,
        boot_vram_mib=boot_vram_mib,
        growth_limit_mib=CLIFF2B_GROWTH_LIMIT_MIB,
        expected_sessions=expected_sessions,
        timed_out=timed_out,
        retention_floor=CLIFF2B_RETENTION_FLOOR,
        silent_empty_fail_any=True,
        retention_basis="turn",
    )
    if require_fixture_shape and rows:
        shape_problems = validate_cliff2b_fixture_shape(
            rows, expected_sessions=expected_sessions
        )
        if shape_problems:
            for p in shape_problems:
                verdict.failures.append(f"Cliff-2b fixture shape: {p}")
            # A wrong-shape run cannot CERTIFY Cliff-2b safety even if every
            # metric is green — downgrade PASS to FAIL.
            if verdict.verdict == "PASS":
                verdict.verdict = "FAIL"
                verdict.exit_code = 1
    return verdict


# ---------------------------------------------------------------------------
# Patch-attribution: ON vs STRIPPED comparison (club-3090 #140 discipline).
# ---------------------------------------------------------------------------
@dataclass
class AttributionResult:
    patch: str
    verdict: str  # LOAD_BEARING / NOT_LOAD_BEARING / TOPOLOGY_SIDESTEP / INCONCLUSIVE
    detail: str
    on_growth_mib: int
    stripped_growth_mib: int
    on_verdict: str
    stripped_verdict: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def attribution_delta(
    on: SoakVerdict,
    stripped: SoakVerdict,
    *,
    patch: str = "overlays",
    topology_tp: int = 1,
) -> AttributionResult:
    """Compare an overlays-ON soak to an overlays-STRIPPED soak and decide
    whether the overlay patches were actually load-bearing for this workload +
    topology.

    Logic (the core of the "PASS != load-bearing" discipline):
      * stripped FAILs and ON PASSes      -> LOAD_BEARING (the patch did the work).
      * both PASS and topology is TP>=2    -> TOPOLOGY_SIDESTEP (TP=2 takes the
        failure mode off the table; the PASS does not prove the patch helped —
        this is exactly the club-3090 #140 trap, e.g. PN59 on dual.yml).
      * both PASS and topology is TP=1     -> NOT_LOAD_BEARING for THIS workload
        (the config survives without the overlay at this depth; the patch may
        still matter at deeper context — note it, don't over-claim).
      * ON FAILs                           -> INCONCLUSIVE (the ON config itself
        didn't pass; fix that before attributing).
    """
    if on.verdict != "PASS":
        return AttributionResult(
            patch=patch,
            verdict="INCONCLUSIVE",
            detail=(
                f"overlays-ON run did not PASS (verdict={on.verdict}); cannot "
                "attribute. Get the ON config green first."
            ),
            on_growth_mib=on.growth_mib,
            stripped_growth_mib=stripped.growth_mib,
            on_verdict=on.verdict,
            stripped_verdict=stripped.verdict,
        )
    if stripped.verdict == "FAIL":
        return AttributionResult(
            patch=patch,
            verdict="LOAD_BEARING",
            detail=(
                f"stripped run FAILED ({'; '.join(stripped.failures) or 'failure'}) "
                f"while overlays-ON PASSED — {patch} is load-bearing for this "
                f"workload (ON growth {on.growth_mib} MiB vs stripped "
                f"{stripped.growth_mib} MiB)."
            ),
            on_growth_mib=on.growth_mib,
            stripped_growth_mib=stripped.growth_mib,
            on_verdict=on.verdict,
            stripped_verdict=stripped.verdict,
        )
    # Both passed.
    if topology_tp >= 2:
        return AttributionResult(
            patch=patch,
            verdict="TOPOLOGY_SIDESTEP",
            detail=(
                f"both ON and stripped PASSED on TP={topology_tp}. Topology alone "
                f"sidesteps the failure mode {patch} targets — the PASS does NOT "
                "prove the patch is load-bearing here (club-3090 #140). Re-run on "
                "a single-card / TP=1 preset to attribute."
            ),
            on_growth_mib=on.growth_mib,
            stripped_growth_mib=stripped.growth_mib,
            on_verdict=on.verdict,
            stripped_verdict=stripped.verdict,
        )
    return AttributionResult(
        patch=patch,
        verdict="NOT_LOAD_BEARING",
        detail=(
            f"both ON and stripped PASSED on TP={topology_tp} at this depth — "
            f"{patch} was not load-bearing for THIS workload (it may still matter "
            "at deeper accumulated context; ramp further before clearing it)."
        ),
        on_growth_mib=on.growth_mib,
        stripped_growth_mib=stripped.growth_mib,
        on_verdict=on.verdict,
        stripped_verdict=stripped.verdict,
    )


if __name__ == "__main__":  # pragma: no cover - manual smoke aid
    import argparse

    ap = argparse.ArgumentParser(description="Genesis quality-gate soak core.")
    ap.add_argument(
        "--dump-turns", action="store_true", help="print the continuous turn specs"
    )
    args = ap.parse_args()
    if args.dump_turns:
        print(
            json.dumps(
                [
                    {"turn": t["turn"], "tool_synth": t["tool_synth"]}
                    for t in CONTINUOUS_TURNS
                ],
                indent=2,
            )
        )
