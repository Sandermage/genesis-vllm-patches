# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pin-gated self-updater (no real git / no mutation)."""
from __future__ import annotations

# Import the CANONICAL module (not the sndr.* shim): build_plan /
# apply_plan read their own module globals (collect_status, run_steps), so
# monkeypatching must target the same module the code actually executes in.
from sndr.product_api.legacy import updater


def test_supported_pins_scans_yaml(tmp_path):
    (tmp_path / "a.yaml").write_text("vllm_pin_required: 0.21.1rc0+gABC\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("foo: bar\n  vllm_pin_required: 0.21.1rc0+gABC\n", encoding="utf-8")
    (tmp_path / "c.yaml").write_text("vllm_pin_required: 0.20.2rc1.dev209+gOLD  # comment\n", encoding="utf-8")
    pins = updater.supported_pins(tmp_path)
    # Most-common first; the trailing comment is stripped.
    assert pins[0] == "0.21.1rc0+gABC"
    assert "0.20.2rc1.dev209+gOLD" in pins


def test_pin_gate_blocks_unsupported_pin():
    supported = ["0.21.1rc0+gABC", "0.20.2rc1.dev209+gOLD"]
    # No target -> defaults to the canonical supported pin.
    g0 = updater.pin_gate(None, supported)
    assert g0["ok"] is True and g0["target_pin"] == "0.21.1rc0+gABC"
    # Supported target -> ok.
    assert updater.pin_gate("0.20.2rc1.dev209+gOLD", supported)["ok"] is True
    # Unsupported (newer, not declared) -> blocked. This is the policy guard:
    # never move to a pin the patcher does not support.
    bad = updater.pin_gate("0.99.0+gNEWER", supported)
    assert bad["ok"] is False and "not declared supported" in bad["reason"]


def test_build_plan_is_pin_gated_and_keeps_server_step_manual(monkeypatch):
    monkeypatch.setattr(updater, "collect_status", lambda: {
        "sndr_core_version": "11.0.0",
        "supported_pins": ["0.21.1rc0+gABC", "0.20.2rc1.dev209+gOLD"],
        "canonical_pin": "0.21.1rc0+gABC",
        "git": {"is_repo": True, "branch": "dev", "commit": "abc1234", "dirty": False},
        "gui_build": {"published": True}, "apply_enabled": False,
    })
    plan = updater.build_plan()
    assert plan["valid"] is True
    assert plan["target_pin"] == "0.21.1rc0+gABC"
    # The docker pin step is server-manual, not a local auto-run step.
    server = [s for s in plan["steps"] if s["kind"] == "server-manual"]
    assert len(server) == 1 and "0.21.1rc0+gABC" in server[0]["cmd"]
    # An unsupported target blocks the whole plan.
    blocked = updater.build_plan("0.99+gNEW")
    assert blocked["valid"] is False and any("not declared supported" in r for r in blocked["blocked_reasons"])


def test_build_plan_blocks_dirty_tree(monkeypatch):
    monkeypatch.setattr(updater, "collect_status", lambda: {
        "sndr_core_version": "11.0.0", "supported_pins": ["0.21.1rc0+gABC"], "canonical_pin": "0.21.1rc0+gABC",
        "git": {"is_repo": True, "branch": "dev", "commit": "abc", "dirty": True},
        "gui_build": {}, "apply_enabled": True,
    })
    plan = updater.build_plan()
    assert plan["valid"] is False
    assert any("uncommitted changes" in r for r in plan["blocked_reasons"])


def test_apply_refuses_without_apply_enabled():
    out = updater.apply_plan(confirm=True, apply_enabled=False)
    assert out["applied"] is False and out["status"] == "disabled"


def test_apply_refuses_without_confirm():
    out = updater.apply_plan(confirm=False, apply_enabled=True)
    assert out["applied"] is False and out["status"] == "needs_confirm"


def test_apply_refuses_when_plan_blocked(monkeypatch):
    monkeypatch.setattr(updater, "build_plan", lambda target_pin=None: {"valid": False, "blocked_reasons": ["dirty tree"]})
    out = updater.apply_plan(confirm=True, apply_enabled=True)
    assert out["applied"] is False and out["status"] == "blocked" and "dirty tree" in out["message"]


def test_apply_runs_only_local_steps(monkeypatch):
    monkeypatch.setattr(updater, "build_plan", lambda target_pin=None: {
        "valid": True, "blocked_reasons": [], "pin_gate": {"ok": True}, "target_pin": "P",
        "current_version": "11.0.0",
        "steps": [
            {"order": 1, "title": "pull", "kind": "local", "cmd": "git pull"},
            {"order": 5, "title": "server pin", "kind": "server-manual", "cmd": "# docker pull"},
        ],
    })
    captured = {}
    from sndr.product_api.legacy import runtime_exec

    class _R:
        def __init__(self, order, title, command, status="ok"):
            self.order, self.title, self.command, self.status = order, title, command, status
            self.exit_code, self.stdout, self.stderr = 0, "ok", ""

    def fake_run_steps(steps, *, timeout=900, **kw):
        captured["steps"] = steps
        return [_R(i, t, c) for i, (t, c) in enumerate(steps)]

    monkeypatch.setattr(runtime_exec, "run_steps", fake_run_steps)
    out = updater.apply_plan(confirm=True, apply_enabled=True)
    assert out["applied"] is True and out["status"] == "done"
    # Only the local step is executed; the server-manual step is returned, not run.
    assert captured["steps"] == [("pull", "git pull")]
    assert len(out["manual_steps"]) == 1 and out["manual_steps"][0]["kind"] == "server-manual"
