# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-host reachability tracker + circuit-breaker states."""
from sndr.product_api.legacy.reliability import ReliabilityTracker


def test_uptime_and_samples():
    t = ReliabilityTracker()
    for i, ok in enumerate([True, True, False, True]):
        t.record("h", ok, now=100.0 + i)
    snap = t.snapshot("h", now=104.0)
    assert snap["checks"] == 4
    assert snap["uptime_pct"] == 75.0
    assert snap["samples"] == [1, 1, 0, 1]
    assert snap["state"] == "closed"          # last probe OK
    assert snap["last_ok"] == 103.0


def test_breaker_opens_then_half_opens():
    t = ReliabilityTracker(fail_threshold=3, recovery=60.0)
    for i in range(3):
        t.record("h", False, now=200.0 + i)   # 3 consecutive fails
    assert t.snapshot("h", now=205.0)["state"] == "open"
    assert t.allow("h", now=205.0) is False
    # After the recovery window, the breaker half-opens (trial allowed).
    assert t.snapshot("h", now=300.0)["state"] == "half_open"
    assert t.allow("h", now=300.0) is True


def test_below_threshold_stays_closed():
    t = ReliabilityTracker(fail_threshold=3)
    t.record("h", False, now=1.0)
    t.record("h", False, now=2.0)            # only 2 fails < threshold
    assert t.snapshot("h", now=3.0)["state"] == "closed"


def test_unknown_key():
    t = ReliabilityTracker()
    assert t.snapshot("nope", now=1.0) is None
    assert t.allow("nope", now=1.0) is True
    assert t.snapshot_all(now=1.0) == {}
