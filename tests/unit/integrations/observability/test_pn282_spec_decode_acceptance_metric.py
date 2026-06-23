# SPDX-License-Identifier: Apache-2.0
"""PN282 wrap module — unit tests.

Strategy: inject a fake ``vllm.v1.sample.rejection_sampler`` module into
``sys.modules`` before invoking ``apply()``. The fake module's
``rejection_sample`` returns a controllable "tensor-like" object that
supports ``.detach().cpu().tolist()`` (matching the real vllm tensor
chain used by both PN248 and PN282).

Covers:

  * Default OFF: apply() returns ("skipped", ...) and rejection_sample
    is NOT wrapped
  * Opt-in: apply() returns ("applied", ...) and rejection_sample IS
    wrapped (marker present)
  * Idempotency: calling apply() twice produces a single wrap
  * Revert restores original
  * Wrap calls into metric module on the result
  * Coexists with a stub PN248-style outer wrap (no double-increment
    from PN282; both wraps run)
"""
from __future__ import annotations

import sys
import types

import pytest

prometheus_client = pytest.importorskip("prometheus_client")


# ─── Fake rejection_sampler module ─────────────────────────────────────


class _FakeTensor:
    """Stand-in for a torch tensor that supports the .detach().cpu().tolist()
    chain used by PN282's wrap to read the rejection_sample output."""

    def __init__(self, rows):
        self._rows = rows

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._rows


def _make_fake_rejection_sampler(default_rows):
    """Build a fake module installed at ``vllm.v1.sample.rejection_sampler``.

    The fake's ``rejection_sample`` ignores all arguments and returns a
    ``_FakeTensor`` wrapping ``default_rows``. Callers can override
    ``module.next_rows`` to change the value returned on the next call.
    """
    fake = types.ModuleType("vllm.v1.sample.rejection_sampler")
    fake.PLACEHOLDER_TOKEN_ID = -1
    fake.next_rows = default_rows
    fake.call_count = 0
    fake.last_args = None

    def rejection_sample(*args, **kwargs):
        fake.call_count += 1
        fake.last_args = (args, kwargs)
        return _FakeTensor(fake.next_rows)

    fake.rejection_sample = rejection_sample
    return fake


@pytest.fixture(autouse=True)
def _module_state_reset(monkeypatch):
    """Wipe PN282 + metric module singleton state between tests AND make
    sure neither env flag is leaking from the host shell.

    Inject a fresh fake ``vllm.v1.sample.rejection_sampler`` module before
    each test so the wrap target is stable and isolated.
    """
    # Reset metric module
    from sndr.observability import spec_decode_metrics as sdm
    sdm._reset_module_state()

    # Reset wrap module by re-importing it cleanly each test
    pn282_modname = (
        "sndr.engines.vllm.patches.observability."
        "pn282_spec_decode_acceptance_metric"
    )
    if pn282_modname in sys.modules:
        sys.modules[pn282_modname]._APPLIED = False
        sys.modules[pn282_modname]._ORIGINAL_REJECTION_SAMPLE = None

    # Env hygiene
    monkeypatch.delenv("SNDR_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC", raising=False)
    monkeypatch.delenv("GENESIS_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC", raising=False)
    monkeypatch.delenv("SNDR_SPEC_DECODE_PROFILE_LABEL", raising=False)

    # Install fresh fake rejection_sampler — default rows = empty batch
    fake = _make_fake_rejection_sampler(default_rows=[])
    # Build the parent module chain. CRITICAL: also bind ``rejection_sampler``
    # as an *attribute* of the parent package, not just a sys.modules entry.
    # apply() reaches the target via ``from vllm.v1.sample import
    # rejection_sampler`` — when the real ``vllm.v1.sample`` package was already
    # imported by an earlier test in the same process, it carries a stale
    # ``rejection_sampler`` attribute, and ``from ... import`` resolves that
    # attribute in preference to our sys.modules fake. Without rebinding the
    # attribute the wrap lands on the stale module and every assertion against
    # the fake fails (order-dependent: passes in isolation, fails in a full run).
    parent = sys.modules.get("vllm.v1.sample")
    if parent is None:
        parent = types.ModuleType("vllm.v1.sample")
        sys.modules["vllm.v1.sample"] = parent
    prev_attr = getattr(parent, "rejection_sampler", None)
    prev_submod = sys.modules.get("vllm.v1.sample.rejection_sampler")
    sys.modules["vllm.v1.sample.rejection_sampler"] = fake
    parent.rejection_sampler = fake

    yield fake

    sdm._reset_module_state()
    # Restore the prior parent-attribute + submodule binding so the fake never
    # leaks into tests that run after this file.
    if prev_attr is not None:
        parent.rejection_sampler = prev_attr
    else:
        try:
            delattr(parent, "rejection_sampler")
        except AttributeError:
            pass
    if prev_submod is not None:
        sys.modules["vllm.v1.sample.rejection_sampler"] = prev_submod
    else:
        sys.modules.pop("vllm.v1.sample.rejection_sampler", None)


@pytest.fixture
def metric_on(monkeypatch):
    monkeypatch.setenv("SNDR_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC", "1")
    yield


def _counter_value(name, **labels):
    from prometheus_client import REGISTRY
    return float(REGISTRY.get_sample_value(name, labels) or 0.0)


# ─── Apply behavior ─────────────────────────────────────────────────────


class TestApply:
    def test_apply_skipped_without_env(self, _module_state_reset):
        from sndr.engines.vllm.patches.observability import (
            pn282_spec_decode_acceptance_metric as pn282,
        )
        status, reason = pn282.apply()
        assert status == "skipped"
        assert "SNDR_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC" in reason
        # Original rejection_sample NOT replaced
        assert not getattr(
            _module_state_reset.rejection_sample,
            "_genesis_pn282_wrapped",
            False,
        )

    def test_apply_succeeds_with_env(self, metric_on, _module_state_reset):
        from sndr.engines.vllm.patches.observability import (
            pn282_spec_decode_acceptance_metric as pn282,
        )
        status, reason = pn282.apply()
        assert status == "applied"
        assert pn282.is_applied() is True
        # rs.rejection_sample now carries the marker
        wrapped = _module_state_reset.rejection_sample
        assert getattr(wrapped, "_genesis_pn282_wrapped", False) is True

    def test_apply_idempotent(self, metric_on, _module_state_reset):
        from sndr.engines.vllm.patches.observability import (
            pn282_spec_decode_acceptance_metric as pn282,
        )
        status1, _ = pn282.apply()
        status2, _ = pn282.apply()
        assert status1 == "applied"
        assert status2 == "applied"
        # Verify only one layer of wrap by checking that the wrapped
        # callable is the SAME object on both apply()s.
        first_wrap = _module_state_reset.rejection_sample
        pn282.apply()  # third time for good measure
        assert _module_state_reset.rejection_sample is first_wrap


# ─── Revert ─────────────────────────────────────────────────────────────


class TestRevert:
    def test_revert_restores_original(self, metric_on, _module_state_reset):
        from sndr.engines.vllm.patches.observability import (
            pn282_spec_decode_acceptance_metric as pn282,
        )
        original = _module_state_reset.rejection_sample
        pn282.apply()
        assert _module_state_reset.rejection_sample is not original
        assert pn282.revert() is True
        assert _module_state_reset.rejection_sample is original
        assert pn282.is_applied() is False

    def test_revert_noop_when_not_applied(self, _module_state_reset):
        from sndr.engines.vllm.patches.observability import (
            pn282_spec_decode_acceptance_metric as pn282,
        )
        assert pn282.revert() is False


# ─── Metric emission ────────────────────────────────────────────────────


class TestMetricEmission:
    def test_wrap_emits_metrics_on_call(
        self, metric_on, monkeypatch, _module_state_reset,
    ):
        monkeypatch.setenv("SNDR_SPEC_DECODE_PROFILE_LABEL", "wrap-test-A")
        from sndr.engines.vllm.patches.observability import (
            pn282_spec_decode_acceptance_metric as pn282,
        )
        pn282.apply()

        # PLACEHOLDER_TOKEN_ID = -1. Rows of shape (batch, K+1):
        # row 0: bonus=100, then [42, 17, -1, -1] → 2 accepted
        # row 1: bonus=101, then [-1, -1, -1, -1] → 0 accepted
        # row 2: bonus=102, then [55, 56, 57, 58] → 4 accepted (full)
        _module_state_reset.next_rows = [
            [100, 42, 17, -1, -1],
            [101, -1, -1, -1, -1],
            [102, 55, 56, 57, 58],
        ]
        result = _module_state_reset.rejection_sample(
            None, None, 4, None, None, None, None, None,
        )
        assert result.tolist() == _module_state_reset.next_rows

        profile = "wrap-test-A"
        assert _counter_value(
            "sndr_spec_decode_accepted_per_call_total",
            k="0", profile=profile,
        ) == pytest.approx(1.0)
        assert _counter_value(
            "sndr_spec_decode_accepted_per_call_total",
            k="2", profile=profile,
        ) == pytest.approx(1.0)
        assert _counter_value(
            "sndr_spec_decode_accepted_per_call_total",
            k="4", profile=profile,
        ) == pytest.approx(1.0)
        assert _counter_value(
            "sndr_spec_decode_calls_total", profile=profile,
        ) == pytest.approx(1.0)

    def test_no_metric_when_off(self, monkeypatch, _module_state_reset):
        """Critical gate from user message: no metric emitted when env off.

        Prometheus counters are process-global so we can't assert
        absolute zero across tests; assert delta=0 with a unique profile
        label that no other test touches.
        """
        unique_profile = "off-test-unique-9af3b1"
        monkeypatch.setenv("SNDR_SPEC_DECODE_PROFILE_LABEL", unique_profile)
        before = _counter_value(
            "sndr_spec_decode_calls_total", profile=unique_profile,
        )
        from sndr.engines.vllm.patches.observability import (
            pn282_spec_decode_acceptance_metric as pn282,
        )
        status, _ = pn282.apply()
        assert status == "skipped"
        # Original (unwrapped) function call should NOT touch metrics
        original = _module_state_reset.rejection_sample
        _module_state_reset.next_rows = [[100, 42, 42, 42, 42]]
        original(None, None, 4, None, None, None, None, None)
        after = _counter_value(
            "sndr_spec_decode_calls_total", profile=unique_profile,
        )
        assert after == pytest.approx(before)


# ─── Coexistence with PN248-style outer wrap ───────────────────────────


class TestCoexistence:
    def test_outer_wrap_after_pn282_still_calls_metric(
        self, metric_on, monkeypatch, _module_state_reset,
    ):
        """Simulate PN248 wrapping AFTER PN282. PN282's wrap is inner;
        the outer (PN248-style) wrap calls PN282's wrap which calls the
        original. Metric must still be emitted exactly once per call.
        """
        monkeypatch.setenv("SNDR_SPEC_DECODE_PROFILE_LABEL", "coexist-A")
        from sndr.engines.vllm.patches.observability import (
            pn282_spec_decode_acceptance_metric as pn282,
        )
        pn282.apply()
        pn282_wrap = _module_state_reset.rejection_sample
        assert getattr(pn282_wrap, "_genesis_pn282_wrapped", False)

        # Now install a fake PN248-style outer wrap that adds a marker
        # and calls through.
        outer_calls = {"count": 0}

        def outer(*args, **kwargs):
            outer_calls["count"] += 1
            return pn282_wrap(*args, **kwargs)

        outer._genesis_pn248_wrapped = True  # type: ignore[attr-defined]
        _module_state_reset.rejection_sample = outer

        # Markers from both layers should be observable on the stack
        assert getattr(_module_state_reset.rejection_sample,
                       "_genesis_pn248_wrapped", False)
        assert getattr(pn282_wrap, "_genesis_pn282_wrapped", False)

        _module_state_reset.next_rows = [[100, 42, 17, 9, -1]]  # 3 accepted
        _module_state_reset.rejection_sample(
            None, None, 4, None, None, None, None, None,
        )

        assert outer_calls["count"] == 1
        assert _counter_value(
            "sndr_spec_decode_accepted_per_call_total",
            k="3", profile="coexist-A",
        ) == pytest.approx(1.0)
        assert _counter_value(
            "sndr_spec_decode_calls_total", profile="coexist-A",
        ) == pytest.approx(1.0)

    def test_re_apply_after_outer_wrap_recognises_existing_pn282(
        self, metric_on, _module_state_reset,
    ):
        """When PN248-style outer wrap is on top, re-applying PN282 must
        NOT install another PN282 underneath — the marker on the inner
        wrap is still accessible only if apply() inspects ``rs.rejection_sample``
        attribute. Since the outer wrap doesn't carry PN282's marker,
        apply() WILL wrap again — verify this stays bounded by the
        module-level _APPLIED flag (idempotent at module level)."""
        from sndr.engines.vllm.patches.observability import (
            pn282_spec_decode_acceptance_metric as pn282,
        )
        pn282.apply()
        inner = _module_state_reset.rejection_sample

        def outer(*args, **kwargs):
            return inner(*args, **kwargs)

        outer._genesis_pn248_wrapped = True  # type: ignore[attr-defined]
        _module_state_reset.rejection_sample = outer

        # Second apply() — module-level _APPLIED is True so it
        # short-circuits without re-wrapping. The outer wrap stays.
        status, _ = pn282.apply()
        assert status == "applied"
        assert _module_state_reset.rejection_sample is outer
