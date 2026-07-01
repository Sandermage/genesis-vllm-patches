# SPDX-License-Identifier: Apache-2.0
"""Tests for the persistent-memory CLI: ``sndr mem remember/recall/search/stats``.

These commands drive the running memory daemon over HTTP (the same
``/api/v1/memory/*`` the GUI calls), so the unit tests inject a fake client via
the module-level ``_make_client`` seam — no daemon, no network. A separate
argparse round-trip pins the parser surface.
"""
from __future__ import annotations

import argparse
import json
import urllib.error

import pytest

from sndr.cli.commands import mem as mem_cli
from sndr.memory.model import MemoryNode, SearchHit


class _FakeClient:
    """Records calls and returns canned MemoryEngine-compatible results."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def remember(self, *, owner_id, text, kind="note", importance=0.0, properties=None):
        self.calls.append(("remember", owner_id, text, kind, importance))
        return 42

    def recall(self, *, owner_id, query, limit=10, expand_depth=2, reinforce=True):
        self.calls.append(("recall", owner_id, query, limit, expand_depth, reinforce))
        return [SearchHit(node=MemoryNode(id=7, owner_id=owner_id, kind="note",
                                          content="postgres vector memory"), score=0.91)]

    def search(self, *, owner_id, query, limit=10):
        self.calls.append(("search", owner_id, query, limit))
        return [SearchHit(node=MemoryNode(id=3, owner_id=owner_id, kind="note",
                                          content="hello world"), score=0.5)]

    def stats(self, *, owner_id):
        self.calls.append(("stats", owner_id))
        return {"nodes": 12, "edges": 8}


@pytest.fixture
def fake_client(monkeypatch) -> _FakeClient:
    fc = _FakeClient()
    monkeypatch.setattr(mem_cli, "_make_client", lambda args: fc)
    return fc


def _ns(**kw) -> argparse.Namespace:
    base = {"output": "text", "url": None, "owner": 1, "token": None,
            "kind": "note", "importance": 0.0, "limit": 10, "depth": 2, "no_reinforce": False}
    base.update(kw)
    return argparse.Namespace(**base)


# ── argparse round-trip ────────────────────────────────────────────────
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sndr-test")
    parser.add_argument("--output", default="text")
    sub = parser.add_subparsers()
    for cmd in (mem_cli.MemRememberCommand(), mem_cli.MemRecallCommand(),
                mem_cli.MemSearchCommand(), mem_cli.MemStatsCommand()):
        p = sub.add_parser(cmd.name)
        cmd.configure_parser(p)
        p.set_defaults(_cmd=cmd)
    return parser


def test_remember_parses_text_and_options():
    args = _parser().parse_args(["mem.remember", "a fact", "--importance", "0.5"])
    assert args.text == "a fact"
    assert args.importance == 0.5


def test_recall_parses_query_and_limit():
    args = _parser().parse_args(["mem.recall", "what is x", "--limit", "3"])
    assert args.query == "what is x"
    assert args.limit == 3


# ── execute behavior (fake client) ─────────────────────────────────────
def test_remember_calls_client_and_prints_id(capsys, fake_client):
    rc = mem_cli.MemRememberCommand().execute(_ns(text="a new fact", importance=0.5))
    assert rc == 0
    assert fake_client.calls[0] == ("remember", 1, "a new fact", "note", 0.5)
    assert "42" in capsys.readouterr().out


def test_recall_prints_hits(capsys, fake_client):
    rc = mem_cli.MemRecallCommand().execute(_ns(query="postgres", limit=5))
    assert rc == 0
    assert fake_client.calls[0] == ("recall", 1, "postgres", 5, 2, True)
    out = capsys.readouterr().out
    assert "postgres vector memory" in out
    assert "7" in out


def test_recall_no_reinforce_flag(fake_client):
    mem_cli.MemRecallCommand().execute(_ns(query="q", no_reinforce=True))
    assert fake_client.calls[0][5] is False  # reinforce=False


def test_search_prints_hits(capsys, fake_client):
    rc = mem_cli.MemSearchCommand().execute(_ns(query="hello"))
    assert rc == 0
    assert fake_client.calls[0] == ("search", 1, "hello", 10)
    assert "hello world" in capsys.readouterr().out


def test_stats_json_output(capsys, fake_client):
    rc = mem_cli.MemStatsCommand().execute(_ns(output="json"))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"nodes": 12, "edges": 8}


def test_owner_scoping_is_passed_through(fake_client):
    mem_cli.MemSearchCommand().execute(_ns(query="q", owner=99))
    assert fake_client.calls[0][1] == 99


# ── error handling: daemon down / unauthorized ─────────────────────────
def test_daemon_unreachable_is_friendly(capsys, monkeypatch):
    class _Boom:
        def search(self, **kw):
            raise urllib.error.URLError("Connection refused")
    monkeypatch.setattr(mem_cli, "_make_client", lambda args: _Boom())
    rc = mem_cli.MemSearchCommand().execute(_ns(query="q"))
    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert any(s in err for s in ("daemon", "sndr up"))  # actionable hint, not a traceback
