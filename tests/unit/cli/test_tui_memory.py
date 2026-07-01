# SPDX-License-Identifier: Apache-2.0
"""TUI Memory pane — pure render + defensive data facade (no textual, no daemon).

Mirrors the cockpit's "view-over-the-CLI, defensive, never crashes" contract:
``render_memory`` turns a memory snapshot into pane text; ``data.memory_snapshot``
fetches counts from the running daemon and downgrades a down/refused daemon to a
calm ``reachable: False`` payload rather than raising.
"""
from __future__ import annotations

from sndr.cli.tui import data as _data
from sndr.cli.tui.render import render_memory


# ── render_memory (pure) ────────────────────────────────────────────────
def test_render_memory_offline_is_calm_hint():
    text = render_memory({"reachable": False, "stats": {}, "error": "refused"})
    assert "memory" in text
    assert "sndr up" in text          # actionable, not a traceback
    assert "refused" not in text      # raw error is not dumped into the pane


def test_render_memory_reachable_shows_counts():
    text = render_memory({"reachable": True, "stats": {"nodes": 12, "edges": 8}, "error": None})
    assert "12" in text
    assert "8" in text
    assert "node" in text.lower()
    assert "edge" in text.lower()


def test_render_memory_reachable_missing_counts_defaults_zero():
    text = render_memory({"reachable": True, "stats": {}, "error": None})
    assert "0" in text  # partial payload → 0, never a KeyError


def test_render_memory_none_is_safe():
    assert isinstance(render_memory(None), str)  # defensive: never crashes


# ── data.memory_snapshot (defensive facade) ─────────────────────────────
def test_memory_snapshot_ok(monkeypatch):
    class _Client:
        def __init__(self, *a, **k): ...
        def stats(self, *, owner_id):
            return {"nodes": 5, "edges": 3}
    monkeypatch.setattr("sndr.memory.client.MemoryHTTPClient", _Client)
    snap = _data.memory_snapshot()
    assert snap["reachable"] is True
    assert snap["stats"] == {"nodes": 5, "edges": 3}


def test_memory_snapshot_down_is_reachable_false(monkeypatch):
    class _Boom:
        def __init__(self, *a, **k): ...
        def stats(self, *, owner_id):
            raise OSError("connection refused")
    monkeypatch.setattr("sndr.memory.client.MemoryHTTPClient", _Boom)
    snap = _data.memory_snapshot()
    assert snap["reachable"] is False
    assert snap["stats"] == {}
    assert "refused" in (snap.get("error") or "")


def test_memory_snapshot_never_raises(monkeypatch):
    def _explode(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr("sndr.memory.client.MemoryHTTPClient", _explode)
    snap = _data.memory_snapshot()  # must not raise
    assert snap["reachable"] is False
