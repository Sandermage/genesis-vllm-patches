# SPDX-License-Identifier: Apache-2.0
"""TDD contract for `MemoryEngine` (Phase 1, verifiable without a DB).

The engine is the storage-agnostic, text-level facade the proxy middleware and
the product-API will call: it owns the embedder and delegates persistence +
brain mechanics to a `MemoryStore`. It also runs the semantic auto-linker
(kNN -> `similar_to` edges) that turns isolated notes into a connected graph.

Backed here by `HashEmbedder` + `InMemoryStore`, so it is fully verified
locally; the same engine runs unchanged on the Postgres backend.
"""
from __future__ import annotations

from sndr.memory.embedder import HashEmbedder
from sndr.memory.engine import MemoryEngine
from sndr.memory.inmemory import InMemoryStore


def _engine() -> MemoryEngine:
    return MemoryEngine(store=InMemoryStore(), embedder=HashEmbedder(dim=512))


class TestRememberRecall:
    def test_remember_returns_id_and_recall_finds_it(self):
        eng = _engine()
        nid = eng.remember(owner_id=1, text="postgres vector memory graph")
        hits = eng.recall(owner_id=1, query="postgres vector memory graph", limit=5)
        assert nid in [h.id for h in hits]

    def test_recall_ranks_topical_over_unrelated(self):
        eng = _engine()
        topical = eng.remember(owner_id=1, text="postgres vector memory engine")
        eng.remember(owner_id=1, text="banana orange weather guitar")
        hits = eng.recall(owner_id=1, query="postgres vector memory graph",
                          limit=5, expand_depth=0)
        assert hits[0].id == topical

    def test_remember_is_owner_scoped(self):
        eng = _engine()
        eng.remember(owner_id=1, text="shared topic words here")
        hits = eng.recall(owner_id=2, query="shared topic words here", limit=5)
        assert hits == []


class TestCommunitiesAndImportance:
    def test_detect_communities_separates_clusters(self):
        eng = _engine()
        # two disjoint fully-linked triplets -> two "clouds"
        a = [eng.remember(owner_id=1, text=f"alpha topic note number {i}") for i in range(3)]
        b = [eng.remember(owner_id=1, text=f"beta unrelated subject row {i}") for i in range(3)]
        # wire each triplet internally (no cross edges)
        for grp in (a, b):
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    eng.store.add_edge(min(grp[i], grp[j]), max(grp[i], grp[j]), "similar_to", weight=0.9)
        comms = eng.detect_communities(owner_id=1)
        ca = {comms[n] for n in a}
        cb = {comms[n] for n in b}
        assert len(ca) == 1 and len(cb) == 1   # each triplet shares one community
        assert ca.isdisjoint(cb)                # the two clouds are different
        # persisted on the nodes
        assert eng.store.get_node(a[0]).community_id == comms[a[0]]

    def test_recompute_importance_ranks_hub_over_leaf(self):
        eng = _engine()
        hub = eng.remember(owner_id=1, text="central hub fact")
        leaves = [eng.remember(owner_id=1, text=f"leaf fact {i}") for i in range(4)]
        for lf in leaves:
            eng.store.add_edge(min(hub, lf), max(hub, lf), "similar_to", weight=0.9)
        isolated = eng.remember(owner_id=1, text="lonely disconnected fact")
        eng.recompute_importance(owner_id=1)
        assert eng.store.get_node(hub).importance > eng.store.get_node(leaves[0]).importance
        assert eng.store.get_node(leaves[0]).importance > eng.store.get_node(isolated).importance

    def test_importance_rewards_connection_count_not_just_strength(self):
        eng = _engine()
        # hub: many (weak) connections; rival: one very strong connection
        hub = eng.remember(owner_id=1, text="hub")
        rival = eng.remember(owner_id=1, text="rival")
        for i in range(5):
            leaf = eng.remember(owner_id=1, text=f"leaf {i}")
            eng.store.add_edge(min(hub, leaf), max(hub, leaf), "similar_to", weight=0.2)
        one = eng.remember(owner_id=1, text="single strong partner")
        eng.store.add_edge(min(rival, one), max(rival, one), "similar_to", weight=1.0)
        eng.recompute_importance(owner_id=1)
        # the well-connected hub outranks the single-strong-edge rival (centrality)
        assert eng.store.get_node(hub).importance > eng.store.get_node(rival).importance

    def test_delete_node_forgets_node_and_its_edges(self):
        eng = _engine()
        a = eng.remember(owner_id=1, text="fact to keep around")
        b = eng.remember(owner_id=1, text="fact to forget entirely")
        eng.store.add_edge(min(a, b), max(a, b), "similar_to", weight=0.9)
        assert eng.store.edge_weight(min(a, b), max(a, b), "similar_to") > 0.0
        assert eng.store.delete_node(b, owner_id=1) is True
        assert eng.store.get_node(b) is None            # gone
        assert eng.store.get_node(a) is not None         # survivor kept
        assert eng.store.edge_weight(min(a, b), max(a, b), "similar_to") == 0.0  # edge dropped
        assert eng.store.delete_node(b, owner_id=1) is False  # idempotent

    def test_delete_node_is_owner_scoped(self):
        eng = _engine()
        n = eng.remember(owner_id=1, text="owner-one private fact")
        assert eng.store.delete_node(n, owner_id=2) is False  # other owner can't delete
        assert eng.store.get_node(n) is not None
        assert eng.store.delete_node(n, owner_id=1) is True

    def test_count_communities_after_consolidate(self):
        eng = _engine()
        # two disjoint fully-linked triplets -> two communities
        a = [eng.remember(owner_id=1, text=f"alpha topic note number {i}") for i in range(3)]
        b = [eng.remember(owner_id=1, text=f"beta unrelated subject row {i}") for i in range(3)]
        for grp in (a, b):
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    eng.store.add_edge(min(grp[i], grp[j]), max(grp[i], grp[j]), "similar_to", weight=0.9)
        assert eng.store.count_communities(owner_id=1) == 0  # none until detected
        eng.detect_communities(owner_id=1)
        assert eng.store.count_communities(owner_id=1) == 2
        assert eng.store.count_communities(owner_id=2) == 0  # owner-scoped

    def test_consolidate_links_and_clusters(self):
        eng = _engine()
        eng.remember(owner_id=1, text="postgres vector memory graph")
        eng.remember(owner_id=1, text="postgres vector memory engine")
        eng.remember(owner_id=1, text="postgres vector memory store")
        report = eng.consolidate(owner_id=1, tau=0.5)
        assert report["linked"] >= 1
        assert report["communities"] >= 1
        # similar notes ended up in one cloud
        comm = {eng.store.get_node(n.id).community_id for n in eng.store.iter_nodes(1)}
        assert None not in comm


class TestHybridAndDedup:
    def test_hybrid_surfaces_exact_token_match(self):
        eng = _engine()
        target = eng.remember(owner_id=1, text="the api owner header is X-Owner-Id exactly")
        eng.remember(owner_id=1, text="a general note about databases and storage")
        hits = eng.search_hybrid(owner_id=1, query="X-Owner-Id", limit=5)
        assert hits[0].id == target  # lexical component pins the exact-token doc

    def test_remember_dedup_returns_existing(self):
        eng = _engine()
        a = eng.remember(owner_id=1, text="a unique fact to store once")
        b = eng.remember(owner_id=1, text="a unique fact to store once", dedup=True)
        assert a == b
        assert eng.store.count_nodes(owner_id=1) == 1

    def test_remember_without_dedup_duplicates(self):
        eng = _engine()
        eng.remember(owner_id=1, text="repeated fact")
        eng.remember(owner_id=1, text="repeated fact", dedup=False)
        assert eng.store.count_nodes(owner_id=1) == 2


class TestMaintenance:
    def test_run_maintenance_consolidates_and_bounds_all_owners(self):
        from sndr.memory.engine import run_maintenance

        eng = _engine()
        for owner in (1, 2):
            for i in range(30):
                eng.remember(owner_id=owner, text=f"owner{owner} note number {i}")
        report = run_maintenance(eng, max_nodes=10)
        # every owner pruned to the cap (leak-bound) + consolidated
        assert eng.store.count_nodes(owner_id=1) == 10
        assert eng.store.count_nodes(owner_id=2) == 10
        assert set(report["owners"]) == {1, 2}
        assert report["pruned"] == 40  # 20 removed per owner


class TestSemanticLinking:
    def test_links_similar_notes_not_dissimilar(self):
        eng = _engine()
        a = eng.remember(owner_id=1, text="postgres vector memory graph")
        b = eng.remember(owner_id=1, text="postgres vector memory engine")
        c = eng.remember(owner_id=1, text="banana orange weather guitar")
        created = eng.link_semantic(owner_id=1, tau=0.5, k=10)
        assert created >= 1
        # a and b share 3/4 tokens (cosine ~0.75) -> linked
        assert eng.store.edge_weight(min(a, b), max(a, b), "similar_to") > 0.0
        # c shares nothing -> not linked to a
        assert eng.store.edge_weight(min(a, c), max(a, c), "similar_to") == 0.0
        assert eng.store.edge_weight(min(b, c), max(b, c), "similar_to") == 0.0

    def test_link_semantic_does_not_self_link(self):
        eng = _engine()
        a = eng.remember(owner_id=1, text="alpha beta gamma delta")
        eng.link_semantic(owner_id=1, tau=0.1, k=10)
        assert eng.store.edge_weight(a, a, "similar_to") == 0.0

    def test_recall_expand_reaches_semantically_linked_neighbor(self):
        eng = _engine()
        # x matches the query; y is similar to x but NOT to the query token
        x = eng.remember(owner_id=1, text="postgres vector memory graph")
        y = eng.remember(owner_id=1, text="postgres vector memory store")
        eng.link_semantic(owner_id=1, tau=0.5, k=10)
        direct = {h.id for h in eng.recall(owner_id=1, query="graph",
                                          limit=10, expand_depth=0, reinforce=False)}
        expanded = {h.id for h in eng.recall(owner_id=1, query="graph",
                                            limit=10, expand_depth=1, reinforce=False)}
        assert x in direct          # x is a direct ANN hit on the query token
        assert y not in direct      # y is not (it lacks the query token)
        assert y in expanded        # but it is reached via the similar_to edge


def test_graph_returns_most_important_nodes_not_oldest():
    """The GUI force-graph is bounded to `limit` nodes. It must show the most
    IMPORTANT (hub) nodes, not the first-inserted (oldest) ones — otherwise a
    large memory renders an arbitrary stale corner instead of the live cloud."""
    eng = _engine()
    ids = [eng.remember(owner_id=1, text=f"memory node number {i}") for i in range(5)]
    # oldest are low-importance; two of the NEWEST are the hubs
    eng.store.set_importance({ids[0]: 0.1, ids[1]: 0.1, ids[2]: 0.1, ids[3]: 0.9, ids[4]: 0.8})
    nodes, _edges = eng.graph(owner_id=1, limit=2)
    assert {n.id for n in nodes} == {ids[3], ids[4]}


def test_graph_degrades_to_insertion_order_when_importance_unset():
    """With no importance computed yet (all 0.0), the bound is stable — the first
    `limit` inserted nodes — so a fresh graph still renders deterministically."""
    eng = _engine()
    ids = [eng.remember(owner_id=1, text=f"fresh node {i}") for i in range(4)]
    nodes, _edges = eng.graph(owner_id=1, limit=2)
    assert [n.id for n in nodes] == ids[:2]
