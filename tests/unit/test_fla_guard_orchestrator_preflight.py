# SPDX-License-Identifier: Apache-2.0
"""T2.2 FLA TP overflow preflight wiring tests.

Verifies that the FLA TP device-index guard (helper at
``vllm/sndr_core/kernels/fla_tp_device_index_guard.py``) is actually
called from the apply orchestrator at boot when the operator opts in
via ``GENESIS_FLA_GUARD_*`` env vars.

This closes the audit-flagged "stub" status (T2.2 utility module had
tests but no production caller) — the helper now fires opt-in at
boot.
"""
from __future__ import annotations

import logging

import pytest


def _run_orchestrator(monkeypatch, caplog, **env):
    """Invoke ``orchestrator.run`` with given env, capture log lines."""
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    from sndr.apply import orchestrator
    caplog.set_level(logging.DEBUG)
    # Dry-run mode (apply=False) avoids touching files / rebinding attrs.
    orchestrator.run(verbose=False, apply=False)
    return [r.message for r in caplog.records]


# ─── Opt-in detection ──────────────────────────────────────────────────


class TestPreflightOptIn:
    def test_skipped_when_env_unset(self, monkeypatch, caplog):
        """No GENESIS_FLA_GUARD_* vars → preflight is silently skipped."""
        for k in (
            "GENESIS_FLA_GUARD_TP_SIZE",
            "GENESIS_FLA_GUARD_NUM_HEADS",
            "GENESIS_FLA_GUARD_HEAD_DIM",
            "GENESIS_FLA_GUARD_SEQ_LEN",
        ):
            monkeypatch.delenv(k, raising=False)
        msgs = _run_orchestrator(monkeypatch, caplog)
        assert not any("[Genesis FLA-guard]" in m for m in msgs)

    def test_partial_env_skipped(self, monkeypatch, caplog):
        """Only some vars set → still skipped (all-or-nothing)."""
        msgs = _run_orchestrator(
            monkeypatch, caplog,
            GENESIS_FLA_GUARD_TP_SIZE=2,
            GENESIS_FLA_GUARD_NUM_HEADS=12,
            # head_dim + seq_len missing
        )
        assert not any("[Genesis FLA-guard]" in m for m in msgs)


# ─── Live preflight emission ───────────────────────────────────────────


class TestPreflightFires:
    def test_safe_config_logs_ok(self, monkeypatch, caplog):
        """27B-class shape (tp=2, heads=8, head_dim=128, seq=64K)
        comfortably fits int32 with low margin. Preflight should log
        OK (no warn)."""
        msgs = _run_orchestrator(
            monkeypatch, caplog,
            GENESIS_FLA_GUARD_TP_SIZE=2,
            GENESIS_FLA_GUARD_NUM_HEADS=8,
            GENESIS_FLA_GUARD_HEAD_DIM=128,
            GENESIS_FLA_GUARD_SEQ_LEN=65536,
        )
        assert any(
            "[Genesis FLA-guard] OK" in m for m in msgs
        ), f"missing OK line in: {msgs[-30:]}"

    def test_high_margin_warns(self, monkeypatch, caplog):
        """35B PROD shape (tp=2, heads=12, head_dim=128, seq=320K)
        at fp16 lands at ~91% of int32 — preflight should WARN about
        boundary proximity."""
        msgs = _run_orchestrator(
            monkeypatch, caplog,
            GENESIS_FLA_GUARD_TP_SIZE=2,
            GENESIS_FLA_GUARD_NUM_HEADS=12,
            GENESIS_FLA_GUARD_HEAD_DIM=128,
            GENESIS_FLA_GUARD_SEQ_LEN=320000,
        )
        assert any(
            "[Genesis FLA-guard]" in m and "boundary" in m for m in msgs
        ), f"missing boundary warn in: {msgs[-30:]}"

    def test_int32_overflow_warns(self, monkeypatch, caplog):
        """Cross int32 boundary (tp=8, heads=64, head_dim=128, seq=1M
        ≈ 68 B elements) → WARN about silent corruption risk."""
        msgs = _run_orchestrator(
            monkeypatch, caplog,
            GENESIS_FLA_GUARD_TP_SIZE=8,
            GENESIS_FLA_GUARD_NUM_HEADS=64,
            GENESIS_FLA_GUARD_HEAD_DIM=128,
            GENESIS_FLA_GUARD_SEQ_LEN=1_048_576,
        )
        assert any(
            "exceeds int32 range" in m and "[Genesis FLA-guard]" in m
            for m in msgs
        ), f"missing int32 warn in: {msgs[-30:]}"

    def test_int64_overflow_aborts(self, monkeypatch, caplog):
        """A truly absurd config (tp=128, heads=2048, head_dim=512,
        seq=10M) would cross int64. Preflight must SystemExit(2)."""
        # int64 max ≈ 9.2e18; 128 * 2048 * 512 * 10^7 = 1.34e15 — well
        # under int64. To trigger int64 overflow we'd need 10^19 — the
        # check is forward-defensive; we test the code path by mocking.
        from unittest.mock import patch
        from sndr.engines.vllm.kernels_legacy.fla_tp_device_index_guard import (
            IndexOverflowReport,
        )
        fake = IndexOverflowReport(
            magnitude=1 << 64,
            fits_int32=False,
            fits_int64=False,
            margin_int32_pct=99999.0,
            margin_int64_pct=99999.0,
        )
        with patch(
            "sndr.engines.vllm.kernels_legacy.fla_tp_device_index_guard.check_index_overflow",
            return_value=fake,
        ):
            with pytest.raises(SystemExit) as exc:
                _run_orchestrator(
                    monkeypatch, caplog,
                    GENESIS_FLA_GUARD_TP_SIZE=128,
                    GENESIS_FLA_GUARD_NUM_HEADS=2048,
                    GENESIS_FLA_GUARD_HEAD_DIM=512,
                    GENESIS_FLA_GUARD_SEQ_LEN=10_000_000,
                )
            assert exc.value.code == 2
        assert any(
            "REFUSING apply_all" in m for m in caplog.messages
        )
