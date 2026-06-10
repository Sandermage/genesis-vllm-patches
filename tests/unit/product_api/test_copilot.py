# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the scoped ops-copilot (tool-calling loop, read-only tools)."""
from __future__ import annotations

import json

from sndr.product_api.legacy import copilot, installer


def _tool_call(name, args, call_id="c1"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _scripted_chat(*responses):
    """A fake chat_fn that returns the scripted assistant messages in order."""
    seq = iter(responses)

    def chat_fn(messages, tools=None):  # noqa: ARG001 - signature parity with engine_client.chat_raw
        return next(seq)

    return chat_fn


def test_tool_specs_are_openai_shaped_and_read_only():
    specs = copilot.tool_specs()
    assert specs and all(s["type"] == "function" and "name" in s["function"] for s in specs)
    # Every registered tool is read-only or a dry-run planner — nothing mutating.
    assert {t["category"] for t in copilot.tool_catalog()} <= {"read", "plan"}


def test_execute_unknown_tool_returns_error_not_raise():
    out = copilot.execute_tool("definitely_not_a_tool", {})
    assert out["ok"] is False and "unknown tool" in out["error"]


def test_execute_overview_tool_returns_real_counts():
    out = copilot.execute_tool("get_overview", {})
    assert out["ok"] is True
    r = out["result"]
    assert "presets" in r and "models" in r and isinstance(r["presets"], int)


def test_loop_runs_a_tool_then_answers():
    # Round 1: the model asks for the overview tool. Round 2: it answers.
    chat = _scripted_chat(
        {"message": {"role": "assistant", "content": "", "tool_calls": [_tool_call("get_overview", {})]},
         "usage": {"total_tokens": 10}},
        {"message": {"role": "assistant", "content": "You have N presets."}, "usage": {"total_tokens": 5}},
    )
    res = copilot.run_copilot([{"role": "user", "content": "how many presets?"}], chat_fn=chat)
    assert res["stopped"] == "final"
    assert res["reply"] == "You have N presets."
    assert [s["tool"] for s in res["steps"]] == ["get_overview"]
    assert res["steps"][0]["ok"] is True
    assert res["usage"]["total_tokens"] == 15  # accumulated across both turns


def test_loop_answers_without_tools():
    chat = _scripted_chat({"message": {"role": "assistant", "content": "Hi, I'm read-only."}, "usage": {}})
    res = copilot.run_copilot([{"role": "user", "content": "hello"}], chat_fn=chat)
    assert res["stopped"] == "final" and res["reply"] == "Hi, I'm read-only." and res["steps"] == []


def test_loop_feeds_tool_error_back_and_keeps_going():
    chat = _scripted_chat(
        {"message": {"role": "assistant", "tool_calls": [_tool_call("get_preset", {})]}},  # missing required arg
        {"message": {"role": "assistant", "content": "That preset id was missing."}},
    )
    res = copilot.run_copilot([{"role": "user", "content": "show preset"}], chat_fn=chat)
    assert res["steps"][0]["tool"] == "get_preset" and res["steps"][0]["ok"] is False
    assert res["reply"] == "That preset id was missing."


def test_plan_install_surfaces_proposed_action(monkeypatch):
    # The dry-run planner must hand the UI a proposed action — never apply.
    monkeypatch.setattr(installer, "build_install_plan", lambda **kw: {
        "target_label": "Docker Compose", "steps": [1, 2, 3], "danger_count": 1,
        "provisions_infra": False, "artifact": {"filename": "docker-compose.yml"},
    })
    chat = _scripted_chat(
        {"message": {"role": "assistant", "tool_calls": [
            _tool_call("plan_install", {"preset_id": "p", "target": "compose"})]}},
        {"message": {"role": "assistant", "content": "Here is the plan — review it in the Installer."}},
    )
    res = copilot.run_copilot([{"role": "user", "content": "deploy p"}], chat_fn=chat)
    assert len(res["proposed_actions"]) == 1
    act = res["proposed_actions"][0]
    assert act["kind"] == "install" and act["section"] == "setup"
    assert act["params"]["preset_id"] == "p" and act["params"]["target"] == "compose"


def test_loop_respects_max_steps_budget():
    # The model keeps calling tools forever; the loop must stop and close out.
    looping = {"message": {"role": "assistant", "tool_calls": [_tool_call("get_overview", {})]}}
    final = {"message": {"role": "assistant", "content": "stopping."}}
    chat = _scripted_chat(looping, looping, final)  # 2 tool rounds then forced close
    res = copilot.run_copilot([{"role": "user", "content": "loop"}], chat_fn=chat, max_steps=2)
    assert res["stopped"] == "max_steps" and res["reply"] == "stopping."
    assert len(res["steps"]) == 2


def test_copilot_endpoints(monkeypatch, tmp_path):
    import pytest
    pytest.importorskip("fastapi")
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    from fastapi.testclient import TestClient

    from sndr.product_api.legacy import engine_client
    from sndr.product_api.legacy.http_app import create_app

    calls = {"n": 0}

    def fake_chat_raw(msgs, tools=None, **kw):  # noqa: ARG001
        calls["n"] += 1
        if calls["n"] == 1 and tools:
            return {"message": {"role": "assistant", "tool_calls": [_tool_call("get_overview", {})]}, "usage": {"total_tokens": 3}}
        return {"message": {"role": "assistant", "content": "Catalog summarized."}, "usage": {}}

    monkeypatch.setattr(engine_client, "chat_raw", fake_chat_raw)
    client = TestClient(create_app(allowed_origins=()))

    # tool catalog is read-only / dry-run only.
    tools = client.get("/api/v1/copilot/tools").json()["tools"]
    assert any(t["name"] == "get_overview" for t in tools)
    assert {t["category"] for t in tools} <= {"read", "plan"}

    # the loop runs the tool then answers.
    r = client.post("/api/v1/copilot/chat", json={"messages": [{"role": "user", "content": "summary?"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == "Catalog summarized."
    assert [s["tool"] for s in body["steps"]] == ["get_overview"]

    # empty messages -> 400.
    assert client.post("/api/v1/copilot/chat", json={}).status_code == 400
