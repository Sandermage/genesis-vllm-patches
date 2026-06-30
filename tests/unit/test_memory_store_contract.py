# SPDX-License-Identifier: Apache-2.0
"""TDD contract for the persistent-memory engine (Phase 1).

These tests define the behaviour of the storage interface
(`sndr.memory.store.MemoryStore`) independently of any backend. The
in-memory reference backend (`sndr.memory.inmemory.InMemoryStore`) must
satisfy every test here; the future Postgres backend will be held to the
same contract (via a parametrized fixture under the `integration` marker).

Design source: docs/design/memory-engine-production-design.md
  * mem_node / mem_edge model
  * ANN search (cosine), owner-scoped
  * Hebbian co-access  w <- (1-lambda)*w + eta   (eta=0.02, lambda=0.995)
  * lazy Ebbinghaus decay at read
  * two-phase retrieval (ANN seeds -> bounded cycle-safe graph expand)
  * salience prune under a per-owner cap (leak-bounded)

The backend under test is provided by the `store` fixture so the same
suite can later run against Postgres.
"""
from __future__ import annotations

import os
import time
import uuid

import pytest

from sndr.memory.inmemory import InMemoryStore
from sndr.memory.model import (
    EBBINGHAUS_S,
    HEBBIAN_ETA,
    HEBBIAN_LAMBDA,
    MemoryNode,
)

# Same contract, both backends. The Postgres backend runs only when
# MEMORY_TEST_DSN points at a live Postgres+pgvector (else it skips cleanly).
_PG_DSN = os.environ.get("MEMORY_TEST_DSN")
_PG_DIM = 8  # >= the largest test vector; embeddings are zero-padded to this


class _FakeClock:
    """Controllable epoch-seconds clock for deterministic decay tests."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture(params=["inmemory", "postgres"])
def store_factory(request):
    """Yields make(*, clock=...) -> a fresh, isolated MemoryStore of the
    parametrized backend. Postgres uses a throwaway schema dropped on teardown.
    """
    if request.param == "postgres":
        if not _PG_DSN:
            pytest.skip("MEMORY_TEST_DSN not set — Postgres backend")
        import psycopg

        from sndr.memory.postgres import PostgresStore

        schema = "memtest_" + uuid.uuid4().hex[:12]
        created: list = []

        def make(*, clock=time.time):
            st = PostgresStore(_PG_DSN, dim=_PG_DIM, schema=schema, clock=clock)
            created.append(st)
            return st

        yield make
        for st in created:
            st.close()
        with psycopg.connect(_PG_DSN, autocommit=True) as conn:
            conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    else:

        def make(*, clock=time.time):
            return InMemoryStore(clock=clock)

        yield make


@pytest.fixture
def store(store_factory):
    return store_factory()


def _vec(*xs: float) -> list[float]:
    return [float(x) for x in xs]


class TestNodeCrud:
    def test_add_returns_id_and_get_roundtrips(self, store: InMemoryStore):
        nid = store.add_node(owner_id=1, kind="note", content="hello",
                             embedding=_vec(1, 0, 0))
        assert isinstance(nid, int)
        node = store.get_node(nid)
        assert node is not None
        assert isinstance(node, MemoryNode)
        assert node.owner_id == 1
        assert node.kind == "note"
        assert node.content == "hello"
        assert node.strength == pytest.approx(1.0)
        assert node.access_count == 0

    def test_get_missing_returns_none(self, store: InMemoryStore):
        assert store.get_node(999) is None

    def test_ids_are_unique(self, store: InMemoryStore):
        a = store.add_node(owner_id=1, kind="note", content="a", embedding=_vec(1, 0))
        b = store.add_node(owner_id=1, kind="note", content="b", embedding=_vec(0, 1))
        assert a != b

    def test_iter_nodes_is_owner_scoped(self, store: InMemoryStore):
        a = store.add_node(owner_id=1, kind="note", content="a", embedding=_vec(1, 0))
        b = store.add_node(owner_id=1, kind="note", content="b", embedding=_vec(0, 1))
        store.add_node(owner_id=2, kind="note", content="other", embedding=_vec(1, 0))
        got = sorted(n.id for n in store.iter_nodes(owner_id=1))
        assert got == sorted([a, b])


class TestVectorSearch:
    def test_search_ranks_by_cosine(self, store: InMemoryStore):
        near = store.add_node(owner_id=1, kind="note", content="near",
                             embedding=_vec(1, 0, 0))
        far = store.add_node(owner_id=1, kind="note", content="far",
                            embedding=_vec(0, 1, 0))
        hits = store.search(owner_id=1, query=_vec(0.9, 0.1, 0.0), limit=2)
        assert [h.id for h in hits] == [near, far]
        # cosine similarity in [0,1] for these non-negative vectors, near > far
        assert hits[0].score > hits[1].score

    def test_search_is_owner_scoped(self, store: InMemoryStore):
        mine = store.add_node(owner_id=1, kind="note", content="mine",
                             embedding=_vec(1, 0, 0))
        store.add_node(owner_id=2, kind="note", content="theirs",
                       embedding=_vec(1, 0, 0))
        hits = store.search(owner_id=1, query=_vec(1, 0, 0), limit=10)
        assert [h.id for h in hits] == [mine]

    def test_search_respects_limit(self, store: InMemoryStore):
        for i in range(5):
            store.add_node(owner_id=1, kind="note", content=str(i),
                           embedding=_vec(1, 0, 0))
        assert len(store.search(owner_id=1, query=_vec(1, 0, 0), limit=3)) == 3


class TestHebbianCoAccess:
    def test_first_co_access_creates_edge_at_eta(self, store: InMemoryStore):
        a = store.add_node(owner_id=1, kind="note", content="a", embedding=_vec(1, 0))
        b = store.add_node(owner_id=1, kind="note", content="b", embedding=_vec(0, 1))
        store.reinforce_co_access([a, b])
        w = store.edge_weight(a, b, rel="co_access")
        assert w == pytest.approx(HEBBIAN_ETA)

    def test_repeated_co_access_follows_clamped_rule_then_saturates(self, store: InMemoryStore):
        # The documented rule is  w <- min(1, (1-lambda)*w + eta).
        # Its unclamped fixed point is eta/(1-lambda) = 4.0, so the clamp does
        # real work: weight rises along the formula, stays in [0,1], and
        # eventually saturates at the 1.0 ceiling.
        a = store.add_node(owner_id=1, kind="note", content="a", embedding=_vec(1, 0))
        b = store.add_node(owner_id=1, kind="note", content="b", embedding=_vec(0, 1))
        w = 0.0
        weights: list[float] = []
        for _ in range(8):
            store.reinforce_co_access([a, b])
            w = min(1.0, HEBBIAN_LAMBDA * w + HEBBIAN_ETA)  # mirror clamped rule
            got = store.edge_weight(a, b, rel="co_access")
            assert got == pytest.approx(w, abs=1e-9)
            weights.append(got)
        assert weights == sorted(weights)               # monotonic non-decreasing
        assert all(0.0 <= x <= 1.0 for x in weights)    # bounded
        for _ in range(2000):
            store.reinforce_co_access([a, b])
        assert store.edge_weight(a, b, rel="co_access") == pytest.approx(1.0)

    def test_co_access_is_undirected_pairing(self, store: InMemoryStore):
        a = store.add_node(owner_id=1, kind="note", content="a", embedding=_vec(1, 0))
        b = store.add_node(owner_id=1, kind="note", content="b", embedding=_vec(0, 1))
        store.reinforce_co_access([b, a])  # order must not matter
        assert store.edge_weight(a, b, rel="co_access") == pytest.approx(HEBBIAN_ETA)


class TestRecallDecayAndTouch:
    def test_fresher_node_outranks_stale_at_equal_similarity(self, store_factory):
        clock = _FakeClock(0.0)
        store = store_factory(clock=clock)
        stale = store.add_node(owner_id=1, kind="note", content="stale",
                              embedding=_vec(1, 0, 0))
        clock.advance(10 * EBBINGHAUS_S)          # stale ages 10 time-constants
        fresh = store.add_node(owner_id=1, kind="note", content="fresh",
                              embedding=_vec(1, 0, 0))
        hits = store.recall(owner_id=1, query=_vec(1, 0, 0), limit=2,
                            expand_depth=0, reinforce=False)
        assert [h.id for h in hits] == [fresh, stale]  # decay breaks the cosine tie
        assert hits[0].score > hits[1].score

    def test_recall_touches_returned_nodes(self, store_factory):
        clock = _FakeClock(100.0)
        store = store_factory(clock=clock)
        nid = store.add_node(owner_id=1, kind="note", content="x",
                            embedding=_vec(1, 0, 0))
        clock.advance(50.0)
        store.recall(owner_id=1, query=_vec(1, 0, 0), limit=1,
                     expand_depth=0, reinforce=False)
        node = store.get_node(nid)
        assert node.access_count == 1
        assert node.accessed_at == pytest.approx(150.0)  # touched to "now"

    def test_recall_reinforces_co_access_of_returned_set(self, store: InMemoryStore):
        a = store.add_node(owner_id=1, kind="note", content="a", embedding=_vec(1, 0, 0))
        b = store.add_node(owner_id=1, kind="note", content="b", embedding=_vec(0.9, 0.1, 0))
        store.recall(owner_id=1, query=_vec(1, 0, 0), limit=2,
                     expand_depth=0, reinforce=True)
        assert store.edge_weight(a, b, rel="co_access") == pytest.approx(HEBBIAN_ETA)


class TestExpandSpreadingActivation:
    def test_strongly_linked_neighbor_is_recalled_via_expand(self, store: InMemoryStore):
        # seed matches the query; neighbor is orthogonal (ANN would miss it)
        seed = store.add_node(owner_id=1, kind="note", content="seed",
                             embedding=_vec(1, 0, 0))
        neighbor = store.add_node(owner_id=1, kind="note", content="neighbor",
                                 embedding=_vec(0, 1, 0))
        store.add_edge(seed, neighbor, "co_access", weight=1.0)
        ids_depth0 = {h.id for h in store.recall(owner_id=1, query=_vec(1, 0, 0),
                                                limit=10, expand_depth=0, reinforce=False)}
        ids_depth1 = {h.id for h in store.recall(owner_id=1, query=_vec(1, 0, 0),
                                                limit=10, expand_depth=1, reinforce=False)}
        assert neighbor not in ids_depth0       # pure ANN can't surface it
        assert neighbor in ids_depth1           # spreading activation does

    def test_expand_is_bounded_and_cycle_safe(self, store: InMemoryStore):
        # a -> b -> a cycle plus self-loop must not hang or duplicate nodes
        a = store.add_node(owner_id=1, kind="note", content="a", embedding=_vec(1, 0, 0))
        b = store.add_node(owner_id=1, kind="note", content="b", embedding=_vec(0, 1, 0))
        store.add_edge(a, b, "co_access", weight=1.0)
        store.add_edge(b, a, "co_access", weight=1.0)
        hits = store.recall(owner_id=1, query=_vec(1, 0, 0), limit=10,
                            expand_depth=3, reinforce=False)
        assert sorted(h.id for h in hits) == [a, b]   # each node once


class TestPruneLeakBound:
    def test_prune_keeps_at_most_cap_nodes_per_owner(self, store: InMemoryStore):
        for i in range(50):
            store.add_node(owner_id=1, kind="note", content=str(i), embedding=_vec(1, 0))
        removed = store.prune(owner_id=1, max_nodes=10)
        assert removed == 40
        assert store.count_nodes(owner_id=1) == 10

    def test_prune_drops_edges_of_removed_nodes_no_dangling(self, store: InMemoryStore):
        keep = store.add_node(owner_id=1, kind="note", content="keep",
                             embedding=_vec(1, 0), importance=100.0)  # the survivor
        drop = store.add_node(owner_id=1, kind="note", content="drop",
                             embedding=_vec(1, 0))
        store.add_edge(keep, drop, "co_access", weight=1.0)
        store.prune(owner_id=1, max_nodes=1)
        assert store.get_node(drop) is None
        assert store.edge_weight(keep, drop, "co_access") == 0.0  # edge gone, no dangling

    def test_prune_is_owner_scoped(self, store: InMemoryStore):
        for i in range(20):
            store.add_node(owner_id=1, kind="note", content=str(i), embedding=_vec(1, 0))
        other = store.add_node(owner_id=2, kind="note", content="other", embedding=_vec(1, 0))
        store.prune(owner_id=1, max_nodes=5)
        assert store.count_nodes(owner_id=1) == 5
        assert store.get_node(other) is not None        # owner 2 untouched
        assert store.count_nodes(owner_id=2) == 1
