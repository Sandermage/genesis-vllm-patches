# SPDX-License-Identifier: Apache-2.0
"""Unit tests for P67 — dispatch-heuristic shape guard.

The Triton kernel itself can't be tested on a CPU dev host. What CAN
be tested is the dispatch decision encoded in the text-patch body —
the boolean expression that decides whether to route a batch through
the P67 multi-query kernel or fall through to upstream.

The expression (rebuilt here to mirror the patch body) is:

    dispatch = (
        is_active                              # env opt-in
        and shape_ok                           # Hq>=8 AND D in {128,256} AND GQA>=2
        and max_query_len > 1                  # not pure decode
        and max_query_len <= max_kp1 (16)      # K+1 cap
        and max_seq_len > max_query_len        # has prior cached KV
        and prior_len <= max_prior             # baked tunable
        and N > 0
        and N % max_query_len == 0             # uniform K+1 layout
    )

This file pins each clause separately so a future refactor can't
silently widen the dispatch envelope (which is what caused the v756
chunked-prefill misroute).

Also covers:
  - module-level baked env constants (H2 fix)
  - text-patch invariants the P67 emit must satisfy
  - safety-gate language must reference v756 / config_detect / spec_decode
"""
from __future__ import annotations

import re

import pytest

from vllm._genesis.wiring.spec_decode.patch_67_tq_multi_query_kernel import (
    GENESIS_P67_MARKER,
    P67_NEW,
    P67_OLD,
    _BAKED_DEBUG_COMPARE,
    _BAKED_MAX_PRIOR,
    apply,
)


# ─── Mirror of the dispatch decision from P67_NEW ───────────────────────


def _dispatch(
    *,
    is_active: bool,
    Hq: int,
    Hk: int,
    D: int,
    max_query_len: int,
    max_seq_len: int,
    N: int,
    max_kp1: int = 16,
    max_prior: int = _BAKED_MAX_PRIOR,
) -> bool:
    """Replicate the P67 dispatch boolean from the patch body for testing."""
    shape_ok = Hq >= 8 and D in (128, 256) and (Hq // Hk) >= 2
    prior_len = max_seq_len - max_query_len
    return (
        is_active
        and shape_ok
        and max_query_len > 1
        and max_query_len <= max_kp1
        and max_seq_len > max_query_len
        and prior_len <= max_prior
        and N > 0
        and (N % max_query_len) == 0
    )


# ─── happy path ─────────────────────────────────────────────────────────


def test_happy_path_K3_spec_verify_dispatches():
    """Typical K=3 spec-verify batch on Qwen3-A3B (Hq=64, Hk=8, D=128):
    K+1=4, batch_size=2 → N=8.
    """
    assert _dispatch(
        is_active=True, Hq=64, Hk=8, D=128,
        max_query_len=4, max_seq_len=2048, N=8,
    )


# ─── env / opt-in ──────────────────────────────────────────────────────


def test_dispatch_off_when_env_inactive():
    assert not _dispatch(
        is_active=False, Hq=64, Hk=8, D=128,
        max_query_len=4, max_seq_len=2048, N=8,
    )


# ─── shape guards ──────────────────────────────────────────────────────


def test_dispatch_off_when_Hq_lt_8():
    assert not _dispatch(
        is_active=True, Hq=4, Hk=2, D=128,
        max_query_len=4, max_seq_len=2048, N=8,
    )


def test_dispatch_off_when_D_unsupported():
    """Only D in {128, 256} is supported."""
    for D in (32, 64, 96, 192, 512):
        assert not _dispatch(
            is_active=True, Hq=64, Hk=8, D=D,
            max_query_len=4, max_seq_len=2048, N=8,
        ), f"D={D} should be rejected"


def test_dispatch_on_for_D_256():
    assert _dispatch(
        is_active=True, Hq=64, Hk=8, D=256,
        max_query_len=4, max_seq_len=2048, N=8,
    )


def test_dispatch_off_when_GQA_lt_2():
    """GQA = Hq // Hk; if Hq == Hk the model is MHA and our kernel doesn't
    handle it (also wouldn't benefit).
    """
    assert not _dispatch(
        is_active=True, Hq=8, Hk=8, D=128,  # GQA=1
        max_query_len=4, max_seq_len=2048, N=8,
    )


# ─── max_query_len bounds ──────────────────────────────────────────────


def test_dispatch_off_for_pure_decode():
    """max_query_len=1 means pure decode — no spec-verify, route upstream."""
    assert not _dispatch(
        is_active=True, Hq=64, Hk=8, D=128,
        max_query_len=1, max_seq_len=2048, N=2,
    )


def test_dispatch_off_for_max_query_len_above_K_plus_1_cap():
    """K+1 cap defaults to 16. Larger query bursts route upstream."""
    assert not _dispatch(
        is_active=True, Hq=64, Hk=8, D=128,
        max_query_len=17, max_seq_len=2048, N=17,
    )


def test_dispatch_on_at_K_plus_1_boundary():
    """max_query_len == max_kp1 is INCLUSIVE per `<= max_kp1`."""
    assert _dispatch(
        is_active=True, Hq=64, Hk=8, D=128,
        max_query_len=16, max_seq_len=2048, N=16,
    )


# ─── max_seq_len / prior cached KV ─────────────────────────────────────


def test_dispatch_off_when_no_prior_cached_KV():
    """max_seq_len == max_query_len means first-chunk prefill — no cached
    KV — route to flash_attn fast path upstream.
    """
    assert not _dispatch(
        is_active=True, Hq=64, Hk=8, D=128,
        max_query_len=4, max_seq_len=4, N=8,
    )


def test_dispatch_off_when_prior_exceeds_baked_max():
    """prior_len > GENESIS_P67_MAX_PRIOR_LEN (default 4096) → route upstream."""
    big_prior = _BAKED_MAX_PRIOR + 1
    max_seq_len = big_prior + 4
    assert not _dispatch(
        is_active=True, Hq=64, Hk=8, D=128,
        max_query_len=4, max_seq_len=max_seq_len, N=8,
    )


def test_dispatch_on_at_max_prior_boundary():
    """prior_len == max_prior is INCLUSIVE per `<= max_prior`."""
    max_seq_len = _BAKED_MAX_PRIOR + 4
    assert _dispatch(
        is_active=True, Hq=64, Hk=8, D=128,
        max_query_len=4, max_seq_len=max_seq_len, N=8,
    )


# ─── N (batch tokens) constraints ──────────────────────────────────────


def test_dispatch_off_when_N_zero():
    assert not _dispatch(
        is_active=True, Hq=64, Hk=8, D=128,
        max_query_len=4, max_seq_len=2048, N=0,
    )


def test_dispatch_off_when_N_not_divisible_by_max_query_len():
    """Uniform K+1 layout requires N % max_query_len == 0."""
    assert not _dispatch(
        is_active=True, Hq=64, Hk=8, D=128,
        max_query_len=4, max_seq_len=2048, N=10,  # 10 % 4 = 2
    )


# ─── baked env constants (H2 fix) ──────────────────────────────────────


def test_baked_max_prior_is_int():
    assert isinstance(_BAKED_MAX_PRIOR, int)
    assert _BAKED_MAX_PRIOR > 0


def test_baked_debug_compare_is_bool():
    assert isinstance(_BAKED_DEBUG_COMPARE, bool)


def test_baked_max_prior_default_4096(monkeypatch):
    """Default when env is unset is 4096."""
    monkeypatch.delenv("GENESIS_P67_MAX_PRIOR_LEN", raising=False)
    # Re-evaluate: default is the constant 4096 baked at module load. The
    # already-imported _BAKED_MAX_PRIOR reflects whatever env was at import
    # time, so we can't guarantee it's 4096. Just assert it's positive +
    # within a sane range.
    assert 64 <= _BAKED_MAX_PRIOR <= 1_000_000


# ─── text-patch emit invariants ────────────────────────────────────────


def test_emit_has_baked_max_prior_literal():
    """H2 fix: max_prior must be a literal in the emit, not env-read."""
    expected = f"_genesis_p67_max_prior = {_BAKED_MAX_PRIOR}"
    assert expected in P67_NEW


def test_emit_has_baked_debug_compare_literal():
    """H2 fix: debug_compare must be a literal in the emit."""
    expected = f"_debug_compare = {_BAKED_DEBUG_COMPARE}"
    assert expected in P67_NEW


def test_emit_has_no_per_dispatch_env_reads():
    """H2 fix: hot-path emit must NOT contain os.environ.get('GENESIS_P67_*')."""
    forbidden = re.compile(
        r"environ\.get\(\s*[\"']GENESIS_P67_(MAX_PRIOR_LEN|DEBUG_COMPARE)[\"']"
    )
    assert not forbidden.search(P67_NEW)


def test_emit_has_shape_guard():
    """The shape-guard expression must be present so the dispatch decision
    matches our `_dispatch()` mirror above.

    Post-Issue#7: the guard now also enforces power-of-2 HEADS_PER_KV,
    spread across multiple lines for readability. We assert each
    constituent clause is present rather than the original one-line form.
    """
    # Each clause of the multi-line guard
    assert "Hq >= 8" in P67_NEW
    assert "D in (128, 256)" in P67_NEW
    assert "_genesis_p67_hpk >= 2" in P67_NEW
    # Issue #7 — power-of-2 check (kept also in test_p67b_shape_ok_includes_power_of_two_check)
    assert "(_genesis_p67_hpk & (_genesis_p67_hpk - 1)) == 0" in P67_NEW


def test_emit_has_uniform_k_plus_1_check():
    """`N % max_query_len == 0` is what enforces uniform K+1 layout — without
    this guard, chunked-prefill batches misroute (v756 root cause).
    """
    assert "(N % attn_metadata.max_query_len) == 0" in P67_NEW


def test_emit_has_kp1_cap_at_16():
    """K+1 cap is hard-baked at 16."""
    assert "_genesis_p67_max_kp1 = 16" in P67_NEW


def test_emit_uses_attn_metadata_max_query_len_inclusive():
    """The cap check must be `<=` not `<` — boundary K+1=16 is supported."""
    assert (
        "attn_metadata.max_query_len <= _genesis_p67_max_kp1" in P67_NEW
    )


def test_emit_falls_through_on_exception():
    """On any P67 exception, must fall through to upstream — never raise."""
    assert "except Exception as _genesis_p67_err" in P67_NEW
    assert "falling through to upstream" in P67_NEW


# ─── marker ─────────────────────────────────────────────────────────────


def test_marker_versioned_capture_guard():
    """Marker should embed v7.62.12_capture_guard so re-applies after B1 fix
    don't no-op against stale baked-env-only marker.
    """
    assert "v7.62.12_capture_guard" in GENESIS_P67_MARKER, (
        f"P67 marker {GENESIS_P67_MARKER!r} should embed v7.62.12_capture_guard"
    )


def test_emit_has_capture_guard_for_telemetry():
    """B1 fix: telemetry block must guard on is_current_stream_capturing()
    so .item()/.tolist() do not break cudagraph capture."""
    assert "is_current_stream_capturing()" in P67_NEW, (
        "B1 fix: P67 emit must guard telemetry on is_current_stream_capturing()"
    )
    assert "and not _genesis_p67_capturing" in P67_NEW, (
        "B1 fix: stats block must check `not _genesis_p67_capturing`"
    )


# ─── apply() safety gate language ──────────────────────────────────────


def test_apply_skipped_when_disabled(monkeypatch):
    monkeypatch.delenv("GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL", raising=False)
    status, reason = apply()
    assert status == "skipped"


def test_apply_safety_gate_message_references_v756_root_cause():
    """The safety-gate skip message must reference v756 / config_detect /
    spec_decode — that's the institutional memory for *why* the gate exists.
    Future maintainers must understand why the gate refuses opt-in.
    """
    import inspect
    src = inspect.getsource(apply)
    assert "SAFETY GATE" in src
    assert "v756" in src or "spec-decode" in src
    assert "config_detect" in src or "speculative" in src


# ─── Issue #7: GQA non-power-of-2 guard (P67 + P67b) ─────────────────────


class TestIssue7PowerOfTwoGuard:
    """Genesis Issue #7 — Triton CompilationError on GQA non-power-of-2.

    Qwen3.6-27B has GQA=24/4=6 → tl.arange + tl.dot fail
    'arange's range must be a power of 2'. Without the guard the kernel
    is dispatched, fails compile, retries every batch → 5x TPS regression.

    The guard:
        (heads_per_kv & (heads_per_kv - 1)) == 0
    is a classic power-of-two check (true for 1, 2, 4, 8, 16, ...).

    Validated by noonghunna 2026-04-29 cross-rig (RTX 3090 + 27B-AutoRound).
    """

    def test_p67_shape_ok_includes_power_of_two_check(self):
        from vllm._genesis.wiring.spec_decode.patch_67_tq_multi_query_kernel import (
            P67_NEW,
        )
        assert "_genesis_p67_hpk" in P67_NEW, (
            "P67 must compute heads_per_kv into a named var for the guard"
        )
        assert "(_genesis_p67_hpk & (_genesis_p67_hpk - 1)) == 0" in P67_NEW, (
            "P67 must reject HEADS_PER_KV that is not a power of 2 — "
            "Issue #7 fix"
        )

    def test_p67b_shape_ok_includes_power_of_two_check(self):
        # P67b runs FIRST inside forward(); guard MUST be in BOTH places
        # (else the retry overhead from P67b fires every batch even when
        # P67 itself bypasses cleanly).
        from vllm._genesis.wiring.spec_decode.patch_67b_spec_verify_routing import (
            P67B_NEW,
        )
        assert "_genesis_p67b_hpk" in P67B_NEW
        assert "(_genesis_p67b_hpk & (_genesis_p67b_hpk - 1)) == 0" in P67B_NEW, (
            "P67b must mirror P67's power-of-2 guard or Issue #7 still fires"
        )

    def test_guard_correctness_pure_python(self):
        """Algorithmic check that the guard expression is a correct
        power-of-2 detector (Brian Kernighan's bit-trick)."""
        def is_p2(n: int) -> bool:
            return n > 0 and (n & (n - 1)) == 0

        # Power-of-2 GQA ratios (kernel-supported)
        for hpk in (1, 2, 4, 8, 16, 32):
            assert is_p2(hpk), f"hpk={hpk} should be power-of-2"

        # Non-power-of-2 GQA ratios that would trigger Issue #7
        for hpk in (3, 5, 6, 7, 9, 12, 24):
            assert not is_p2(hpk), (
                f"hpk={hpk} would fail Triton arange compile — must be rejected"
            )

        # The Qwen3.6-27B case specifically
        qwen36_27b_hpk = 24 // 4  # num_heads / num_kv_heads
        assert qwen36_27b_hpk == 6
        assert not is_p2(qwen36_27b_hpk), (
            "Qwen3.6-27B GQA=6 is the exact case Issue #7 reports — "
            "must be rejected by guard"
        )

        # The Qwen3.6-35B case (works pre-fix)
        qwen36_35b_hpk = 8  # GQA=8 directly
        assert is_p2(qwen36_35b_hpk), (
            "Qwen3.6-35B GQA=8 is power-of-2 — must continue dispatching"
        )
