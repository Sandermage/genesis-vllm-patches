# SPDX-License-Identifier: Apache-2.0
"""Leak-soak: the memory store must stay BOUNDED under sustained churn.

This is the executable form of the operator requirement "без утечек памяти".
We hammer the store with thousands of add/recall/prune cycles and assert that
node and edge counts converge to the cap rather than growing with the step
count — and that pruning a node drops its edges (no dangling references that
would silently leak).

Pure stdlib + the in-memory reference backend, so it runs in CI in well under a
second. The same assertions will later be run against the Postgres backend
under the `integration` marker to prove the SQL DELETE + VACUUM path is equally
leak-bounded.
"""
from __future__ import annotations

import random

from sndr.memory.inmemory import InMemoryStore

CAP = 200
PRUNE_EVERY = 50
STEPS = 10_000
DIM = 8


def test_memory_bounded_under_sustained_churn():
    store = InMemoryStore()
    rng = random.Random(20260630)  # deterministic
    owner = 1

    peak_nodes = 0
    peak_edges = 0
    for step in range(STEPS):
        store.add_node(
            owner_id=owner,
            kind="note",
            content=f"n{step}",
            embedding=[rng.random() for _ in range(DIM)],
        )
        # recall forms co_access edges among the returned set (edge churn)
        store.recall(
            owner_id=owner,
            query=[rng.random() for _ in range(DIM)],
            limit=8,
            expand_depth=1,
            reinforce=True,
        )
        if step % PRUNE_EVERY == 0:
            store.prune(owner_id=owner, max_nodes=CAP)
        peak_nodes = max(peak_nodes, store.count_nodes(owner))
        peak_edges = max(peak_edges, store.count_edges())

    store.prune(owner_id=owner, max_nodes=CAP)

    # Final state is the cap, NOT the 10_000 nodes we inserted -> no leak.
    assert store.count_nodes(owner) <= CAP
    # Edges only ever connect surviving nodes -> bounded by the complete graph
    # on the cap; the final prune drops every edge of a removed node.
    assert store.count_edges() <= CAP * (CAP - 1) // 2
    # No runaway between prunes: peak bounded by cap + one inter-prune batch.
    assert peak_nodes <= CAP + PRUNE_EVERY
    # Sanity: we really did churn far more than the cap.
    assert STEPS > 10 * CAP


def test_prune_reclaims_edge_space_proportionally():
    """After heavy edge formation, pruning to a small cap collapses edge count
    toward the small graph — the anti-leak guarantee at the edge level."""
    store = InMemoryStore()
    owner = 1
    ids = [
        store.add_node(owner_id=owner, kind="note", content=str(i),
                       embedding=[1.0, 0.0])
        for i in range(60)
    ]
    # fully co-access them in batches to build many edges
    for _ in range(30):
        store.reinforce_co_access(ids)
    assert store.count_edges() == 60 * 59 // 2  # complete graph

    store.prune(owner_id=owner, max_nodes=10)
    assert store.count_nodes(owner) == 10
    assert store.count_edges() <= 10 * 9 // 2   # collapsed with the node set
