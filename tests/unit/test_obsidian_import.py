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


def test_wikilinks_resolve_case_insensitively_and_by_h1_title(tmp_path):
    # Obsidian resolves [[Note]] ignoring case and against the display title,
    # not just the exact filename — so a link can differ in case or use the H1.
    (tmp_path / "genesis.md").write_text(
        "# Genesis\nUses [[Memory]] and [[Qwen3.6-35B]].\n", encoding="utf-8"
    )
    (tmp_path / "memory.md").write_text("# Memory\nPart of [[genesis]].\n", encoding="utf-8")
    (tmp_path / "qwen.md").write_text("# Qwen3.6-35B\nA model.\n", encoding="utf-8")
    eng = _engine()
    report = import_vault(engine=eng, owner_id=1, vault_path=str(tmp_path))
    t = _by_title(eng)
    assert eng.store.edge_weight(t["genesis"], t["memory"], "wikilink") > 0.0   # [[Memory]] -> memory.md (case)
    assert eng.store.edge_weight(t["genesis"], t["qwen"], "wikilink") > 0.0     # [[Qwen3.6-35B]] -> qwen.md (H1)
    assert eng.store.edge_weight(t["memory"], t["genesis"], "wikilink") > 0.0   # [[genesis]] -> genesis.md (case)
    assert report["missing"] == 0


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


def test_symlink_escaping_vault_is_skipped(tmp_path):
    import os
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Inside.md").write_text("inside note\n", encoding="utf-8")
    secret = tmp_path / "secret.md"
    secret.write_text("SECRET outside the vault\n", encoding="utf-8")
    try:
        os.symlink(secret, vault / "Leak.md")  # in-vault symlink pointing outside
    except (OSError, NotImplementedError):
        import pytest
        pytest.skip("symlinks not supported here")
    eng = _engine()
    report = import_vault(engine=eng, owner_id=1, vault_path=str(vault))
    contents = [n.content for n in eng.store.iter_nodes(1)]
    assert any("inside note" in c for c in contents)
    assert not any("SECRET" in c for c in contents)  # escaping symlink not ingested
    assert report["notes"] == 1


def test_missing_vault_raises(tmp_path):
    import pytest
    with pytest.raises((FileNotFoundError, NotADirectoryError)):
        import_vault(engine=_engine(), owner_id=1, vault_path=str(tmp_path / "nope"))


def test_identical_content_files_keep_both_link_lists(tmp_path):
    """Two files with IDENTICAL content dedup to one node — but each file's
    wikilinks used to OVERWRITE the previous one's in the per-node dict, so the
    first file's edges were silently never created."""
    (tmp_path / "A.md").write_text("Same body. [[Target1]]\n", encoding="utf-8")
    # identical content -> same node id via dedup; different filename/link
    (tmp_path / "B.md").write_text("Same body. [[Target1]]\n", encoding="utf-8")
    (tmp_path / "C.md").write_text("Links to [[A]] and [[B]].\n", encoding="utf-8")
    (tmp_path / "Target1.md").write_text("t1\n", encoding="utf-8")
    eng = _engine()
    report = import_vault(engine=eng, owner_id=1, vault_path=str(tmp_path))
    t = _by_title(eng)
    # A+B deduped to ONE node (title = first file); its Target1 link must exist
    assert eng.store.edge_weight(t["A"], t["Target1"], "wikilink") > 0.0
    # C's [[A]] and [[B]] BOTH resolve to the deduped node (stem aliases) ...
    assert eng.store.edge_weight(t["C"], t["A"], "wikilink") > 0.0
    # ... which is exactly what missing==0 proves for the [[B]] alias.
    assert report["missing"] == 0
