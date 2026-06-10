# SPDX-License-Identifier: Apache-2.0
"""TDD for SNDR_DISABLE_X / GENESIS_DISABLE_X opt-out wiring in
`dispatcher.decision.should_apply()`.

Audit 2026-05-14: `env.is_disabled` had existed for a while but was
never consulted by `should_apply()`. That meant `default_on=True`
patches could not be opted-out via env — operators had to edit
`registry.py` to A/B-test a patch's contribution. Community
benchmark workflows ("disable patch X, re-bench, compare") were
blocked by that gap.

The fix:
  - `should_apply()` consults `is_disabled(bare_flag)` after the
    ENABLE check.
  - DISABLE wins over ENABLE when both are set (intent-clear
    opt-out semantics).
  - WARN log is emitted on the conflict so the contradiction is
    visible.

These tests freeze that contract.
"""
from __future__ import annotations

import os
from unittest import mock

import pytest


# Use P108 as the canonical default_on=True patch for these tests.
# We pick P108 because it is the patch whose absence of a DISABLE
# knob produced the original audit finding (Wave 8 → dev338 bench).
# If P108 is ever retired, swap to any other default_on=True patch.
TEST_PATCH_ID = "P108"
TEST_BARE_FLAG = "P108"


def _should_apply():
    """Lazy import so test collection doesn't pull torch."""
    from sndr.dispatcher.decision import should_apply
    return should_apply


class TestDisableEnvKnob:
    """`should_apply()` honours SNDR_DISABLE_X / GENESIS_DISABLE_X."""

    def test_clean_default_on_skips_under_strict_opt_in(self):
        """Strict opt-in (Phase 2026-05-17, decision.py:305-360):

        With no env knobs set, a `default_on=True` patch must SKIP and
        emit an operator-actionable reason that names the canonical
        `GENESIS_ENABLE_<X>` flag plus the `GENESIS_LEGACY_DEFAULT_ON=1`
        legacy escape hatch. Confirms that the post-2026-05-17 policy
        is in effect — under the pre-2026-05-17 auto-apply semantics
        this same setup applied the patch.
        """
        env = {k: v for k, v in os.environ.items()
               if TEST_BARE_FLAG not in k.replace("ENABLE_", "").replace("DISABLE_", "")}
        env.pop("GENESIS_LEGACY_DEFAULT_ON", None)
        env.pop("SNDR_LEGACY_DEFAULT_ON", None)
        with mock.patch.dict(os.environ, env, clear=True):
            should_apply = _should_apply()
            ok, reason = should_apply(TEST_PATCH_ID)
            assert not ok, (
                f"strict opt-in: clean env must SKIP default_on patch, "
                f"got apply with reason: {reason}"
            )
            assert "strict opt-in" in reason.lower()
            assert f"GENESIS_ENABLE_{TEST_BARE_FLAG}" in reason
            assert "GENESIS_LEGACY_DEFAULT_ON" in reason

    def test_explicit_enable_applies(self):
        """Modern strict-opt-in apply path: `GENESIS_ENABLE_<X>=1`
        explicitly engages the patch. Counterpart to the
        `test_clean_default_on_skips_under_strict_opt_in` SKIP case —
        confirms the canonical opt-in env flag works.
        """
        env = {k: v for k, v in os.environ.items()
               if TEST_BARE_FLAG not in k.replace("ENABLE_", "").replace("DISABLE_", "")}
        env[f"GENESIS_ENABLE_{TEST_BARE_FLAG}"] = "1"
        with mock.patch.dict(os.environ, env, clear=True):
            should_apply = _should_apply()
            ok, reason = should_apply(TEST_PATCH_ID)
            assert ok, (
                f"explicit GENESIS_ENABLE_{TEST_BARE_FLAG}=1 should "
                f"apply default_on patch, got skip: {reason}"
            )

    def test_genesis_disable_skips(self):
        """`GENESIS_DISABLE_P108=1` skips even though default_on=True."""
        env = {**os.environ, f"GENESIS_DISABLE_{TEST_BARE_FLAG}": "1"}
        # Make sure no ENABLE knob masks the DISABLE one for this test.
        env.pop(f"GENESIS_ENABLE_{TEST_BARE_FLAG}", None)
        env.pop(f"SNDR_ENABLE_{TEST_BARE_FLAG}", None)
        with mock.patch.dict(os.environ, env, clear=True):
            should_apply = _should_apply()
            ok, reason = should_apply(TEST_PATCH_ID)
            assert not ok, "DISABLE should skip"
            assert "explicitly disabled" in reason.lower()
            assert TEST_BARE_FLAG in reason

    def test_sndr_disable_skips(self):
        """SNDR_DISABLE_<X>=1 takes precedence the same way as the legacy alias."""
        env = {**os.environ, f"SNDR_DISABLE_{TEST_BARE_FLAG}": "1"}
        env.pop(f"GENESIS_ENABLE_{TEST_BARE_FLAG}", None)
        env.pop(f"SNDR_ENABLE_{TEST_BARE_FLAG}", None)
        with mock.patch.dict(os.environ, env, clear=True):
            should_apply = _should_apply()
            ok, reason = should_apply(TEST_PATCH_ID)
            assert not ok, "SNDR_DISABLE should skip"
            assert "explicitly disabled" in reason.lower()

    def test_disable_wins_over_enable(self):
        """If both ENABLE=1 AND DISABLE=1 are set, DISABLE wins.

        Intent-clear opt-out — kill-switch beats opt-in when in conflict.
        """
        env = {
            **os.environ,
            f"GENESIS_ENABLE_{TEST_BARE_FLAG}": "1",
            f"GENESIS_DISABLE_{TEST_BARE_FLAG}": "1",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            should_apply = _should_apply()
            ok, reason = should_apply(TEST_PATCH_ID)
            assert not ok, "DISABLE must beat ENABLE when both set"
            assert "explicitly disabled" in reason.lower()

    def test_disable_falsy_is_noop_under_explicit_enable(self):
        """`GENESIS_DISABLE_X=0` is "no opt-out" — it must NOT override
        an explicit `GENESIS_ENABLE_X=1`. Strict opt-in (2026-05-17)
        requires the explicit ENABLE for any apply path, so the test
        sets both flags: ENABLE=1 (engages the patch) and DISABLE=0
        (must not interfere). The intent of `DISABLE=0` is "no
        opt-out" — falsy disable is a no-op, not a skip trigger.
        """
        env = {**os.environ, f"GENESIS_DISABLE_{TEST_BARE_FLAG}": "0"}
        env[f"GENESIS_ENABLE_{TEST_BARE_FLAG}"] = "1"
        env.pop("GENESIS_LEGACY_DEFAULT_ON", None)
        env.pop("SNDR_LEGACY_DEFAULT_ON", None)
        with mock.patch.dict(os.environ, env, clear=True):
            should_apply = _should_apply()
            ok, reason = should_apply(TEST_PATCH_ID)
            assert ok, (
                f"DISABLE=0 must not interfere with explicit "
                f"ENABLE=1; got skip: {reason}"
            )


# ─── Reason text contract ─────────────────────────────────────────────────


class TestDisableReasonText:
    """Operator-facing reason string must point at the env var that
    triggered the skip, so the diagnostic is actionable from one
    `sndr doctor` line."""

    def test_reason_names_both_prefixes(self):
        env = {**os.environ, f"GENESIS_DISABLE_{TEST_BARE_FLAG}": "1"}
        env.pop(f"GENESIS_ENABLE_{TEST_BARE_FLAG}", None)
        env.pop(f"SNDR_ENABLE_{TEST_BARE_FLAG}", None)
        with mock.patch.dict(os.environ, env, clear=True):
            should_apply = _should_apply()
            _ok, reason = should_apply(TEST_PATCH_ID)
            assert f"SNDR_DISABLE_{TEST_BARE_FLAG}" in reason
            assert f"GENESIS_DISABLE_{TEST_BARE_FLAG}" in reason
            assert "re-engage" in reason or "engage" in reason
