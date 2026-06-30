# SPDX-License-Identifier: Apache-2.0
"""TDD for Obsidian vault import.

An Obsidian vault IS a knowledge graph: each note is a node, each `[[wikilink]]`
is an edge. Importing maps that directly into our memory graph — notes become
memory nodes (deduped), wikilinks become edges, tags land in properties.
"""
from __future__ import annotations

from sndr.memory.embedder import HashEmbedder
from sndr.memory.engine import MemoryEngine
from sndr.memory.inmemory import InMemoryStore
from sndr.memory.obsidian import import_vault


def _engine() -> MemoryEngine:
    return MemoryEngine(store=InMemoryStore(), embedder=HashEmbedder(dim=64))


def _by_title(eng, owner=1) -> dict[str, int]:
    return {n.properties.get("title"): n.id for n in eng.store.iter_nodes(owner)}


def test_import_creates_node_per_note(tmp_path):
    (tmp_path / "Alpha.md").write_text("# Alpha\nThe alpha note body.\n", encoding="utf-8")
    (tmp_path / "Beta.md").write_text("Beta note body here.\n", encoding="utf-8")
    eng = _engine()
    report = import_vault(engine=eng, owner_id=1, vault_path=str(tmp_path))
    assert report["notes"] == 2
    titles = _by_title(eng)
    assert set(titles) == {"Alpha", "Beta"}


def test_wikilinks_become_edges(tmp_path):
    (tmp_path / "Alpha.md").write_text(
        "Links to [[Beta]] and [[Gamma|an alias]] and missing [[Nope]].\n", encoding="utf-8"
    )
    (tmp_path / "Beta.md").write_text("beta\n", encoding="utf-8")
    (tmp_path / "Gamma.md").write_text("gamma\n", encoding="utf-8")
    eng = _engine()
    report = import_vault(engine=eng, owner_id=1, vault_path=str(tmp_path))
    t = _by_title(eng)
    # Beta and Gamma exist -> edges; Nope is missing -> no edge
    assert eng.store.edge_weight(t["Alpha"], t["Beta"], "wikilink") > 0.0
    assert eng.store.edge_weight(t["Alpha"], t["Gamma"], "wikilink") > 0.0
    assert report["links"] == 2
    assert report["missing"] == 1  # [[Nope]] had no target


def test_tags_captured_in_properties(tmp_path):
    (tmp_path / "Note.md").write_text("body #project #genesis\n", encoding="utf-8")
    eng = _engine()
    import_vault(engine=eng, owner_id=1, vault_path=str(tmp_path))
    node = next(eng.store.iter_nodes(1))
    assert set(node.properties.get("tags", [])) == {"project", "genesis"}
    assert node.properties.get("source") == "obsidian"


def test_reimport_is_idempotent(tmp_path):
    (tmp_path / "A.md").write_text("[[B]]\n", encoding="utf-8")
    (tmp_path / "B.md").write_text("b\n", encoding="utf-8")
    eng = _engine()
    import_vault(engine=eng, owner_id=1, vault_path=str(tmp_path))
    import_vault(engine=eng, owner_id=1, vault_path=str(tmp_path))
    assert eng.store.count_nodes(owner_id=1) == 2  # no duplicates


def test_owner_scoped(tmp_path):
    (tmp_path / "X.md").write_text("x\n", encoding="utf-8")
    eng = _engine()
    import_vault(engine=eng, owner_id=7, vault_path=str(tmp_path))
    assert eng.store.count_nodes(owner_id=7) == 1
    assert eng.store.count_nodes(owner_id=1) == 0


def test_missing_vault_raises(tmp_path):
    import pytest
    with pytest.raises((FileNotFoundError, NotADirectoryError)):
        import_vault(engine=_engine(), owner_id=1, vault_path=str(tmp_path / "nope"))
