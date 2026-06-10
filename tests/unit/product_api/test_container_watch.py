# SPDX-License-Identifier: Apache-2.0
"""Tests for the container health-watch transition detector (pure)."""
from __future__ import annotations

from sndr.product_api.legacy.container_watch import Watcher, detect_transitions


def _c(name, state, status="Up"):
    return {"name": name, "state": state, "status": status}


def test_first_tick_is_baseline_no_alerts():
    alerts, snap = detect_transitions({}, [_c("vllm-35b", "running"), _c("sndr-daemon", "running")])
    assert alerts == []
    assert snap == {"vllm-35b": "running", "sndr-daemon": "running"}


def test_running_to_down_is_critical():
    prev = {"vllm-35b": "running"}
    alerts, snap = detect_transitions(prev, [_c("vllm-35b", "exited", "Exited (137) 1s ago")])
    assert len(alerts) == 1
    assert alerts[0]["level"] == "critical" and "DOWN" in alerts[0]["message"]
    assert snap == {"vllm-35b": "down"}


def test_down_to_running_is_recovery():
    alerts, _ = detect_transitions({"vllm-35b": "down"}, [_c("vllm-35b", "running")])
    assert alerts[0]["level"] == "ok" and "recovered" in alerts[0]["message"]


def test_no_change_no_alert():
    alerts, _ = detect_transitions({"vllm-35b": "running"}, [_c("vllm-35b", "running")])
    assert alerts == []


def test_watcher_holds_state_across_ticks():
    w = Watcher()
    assert w.tick([_c("vllm-35b", "running")]) == []       # baseline
    crit = w.tick([_c("vllm-35b", "dead", "Dead")])
    assert crit and crit[0]["level"] == "critical"
    assert w.tick([_c("vllm-35b", "running")])[0]["level"] == "ok"  # recovered
    assert w.watched == 1
