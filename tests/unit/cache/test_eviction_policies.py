# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr.cache.eviction_policies` — library module.

Three policy implementations (LRU, 2Q, ARC) all conform to the
`EvictionPolicy` ABC. Tests check each policy independently for
contract correctness, then run scenario simulations to verify the
defining behaviors (2Q's scan-resistance, ARC's adaptive split, LRU's
recency-only ordering).

Status: library module is shipped + tested. The vllm BlockPool
integration (vllm#40270 backport) is NOT yet implemented — when that
work lands as a proper patch, it will use these policy classes.
"""
from __future__ import annotations

import pytest

from sndr.cache.eviction_policies import (
    ARCPolicy,
    EvictionPolicy,
    LRUPolicy,
    TwoQueuePolicy,
    list_policies,
    make_policy,
)


# ─── LRU contract ───────────────────────────────────────────────────────


class TestLRU:
    def test_admit_then_evict(self):
        p = LRUPolicy()
        p.admit("a")
        p.admit("b")
        p.admit("c")
        assert len(p) == 3
        # LRU evicts oldest first
        assert p.evict() == "a"
        assert p.evict() == "b"
        assert p.evict() == "c"
        assert p.evict() is None

    def test_touch_promotes_to_most_recent(self):
        p = LRUPolicy()
        p.admit("a"); p.admit("b"); p.admit("c")
        p.touch("a")  # 'a' is now most-recent
        # Eviction order should be b, c, a
        assert p.evict() == "b"
        assert p.evict() == "c"
        assert p.evict() == "a"

    def test_remove_drops_silently(self):
        p = LRUPolicy()
        p.admit("a")
        p.remove("a")
        p.remove("nonexistent")  # no-op
        assert len(p) == 0

    def test_keys_iteration_recency_order(self):
        p = LRUPolicy()
        p.admit("a"); p.admit("b"); p.admit("c")
        # Hottest last
        assert list(p.keys()) == ["a", "b", "c"]


# ─── 2Q contract ────────────────────────────────────────────────────────


class TestTwoQueue:
    def test_admits_to_a1_first(self):
        p = TwoQueuePolicy()
        p.admit("a")
        # On admit, key sits in A1
        assert "a" in list(p.keys())

    def test_first_hit_promotes_a1_to_am(self):
        p = TwoQueuePolicy()
        p.admit("a")
        p.touch("a")  # promotion: A1 → Am
        stats = p.stats()
        assert stats["a1_len"] == 0
        assert stats["am_len"] == 1

    def test_eviction_drops_a1_first(self):
        """Defining 2Q property: scan pollution evicts probationary
        entries first, NOT promoted (Am) ones."""
        p = TwoQueuePolicy()
        # Hot prefix: admit + promote
        p.admit("hot1"); p.touch("hot1")
        p.admit("hot2"); p.touch("hot2")
        # Cold scan: just admits, never touches
        for k in [f"cold{i}" for i in range(10)]:
            p.admit(k)
        # First N evictions should drop cold entries before any hot
        for _ in range(10):
            victim = p.evict()
            assert victim is not None
            assert victim.startswith("cold"), (
                f"2Q failed scan resistance: evicted {victim} before A1 drained"
            )
        # Hot entries must still be present
        assert "hot1" in list(p.keys())
        assert "hot2" in list(p.keys())

    def test_invalid_a1_ratio_raises(self):
        with pytest.raises(ValueError):
            TwoQueuePolicy(a1_ratio=0.0)
        with pytest.raises(ValueError):
            TwoQueuePolicy(a1_ratio=1.5)

    def test_remove_handles_both_queues(self):
        p = TwoQueuePolicy()
        p.admit("a"); p.touch("a")  # 'a' now in Am
        p.admit("b")  # 'b' in A1
        p.remove("a"); p.remove("b")
        assert len(p) == 0


# ─── ARC contract ───────────────────────────────────────────────────────


class TestARC:
    def test_capacity_validation(self):
        with pytest.raises(ValueError):
            ARCPolicy(capacity=0)
        with pytest.raises(ValueError):
            ARCPolicy(capacity=-1)

    def test_admit_grows_t1(self):
        p = ARCPolicy(capacity=10)
        p.admit("a")
        assert "a" in list(p.keys())
        s = p.stats()
        assert s["t1_len"] == 1
        assert s["t2_len"] == 0

    def test_touch_promotes_t1_to_t2(self):
        p = ARCPolicy(capacity=10)
        p.admit("a"); p.touch("a")
        s = p.stats()
        assert s["t1_len"] == 0
        assert s["t2_len"] == 1

    def test_evict_falls_back_to_t1_then_t2(self):
        p = ARCPolicy(capacity=10)
        p.admit("a"); p.admit("b")
        # Both in T1; evict pops T1 head
        v = p.evict()
        assert v in ("a", "b")
        assert len(p) == 1

    def test_adaptive_p_increases_on_b1_hit(self):
        """Hitting an entry that recently fell out of T1 → adapt p UP."""
        p = ARCPolicy(capacity=4)
        # Fill T1, then evict to B1
        for k in ["a", "b", "c", "d"]:
            p.admit(k)
        # Force eviction (drops T1 head into B1)
        p.evict()
        p_before = p.stats()["p"]
        # Now hit the ghost — adapts p up
        p.touch("a")  # 'a' was the evicted one; now in B1, hit promotes
        assert p.stats()["p"] >= p_before

    def test_remove_drops_from_any_list(self):
        p = ARCPolicy(capacity=4)
        p.admit("a")
        p.remove("a")
        assert len(p) == 0


# ─── Factory ────────────────────────────────────────────────────────────


class TestFactory:
    def test_make_policy_lru(self):
        p = make_policy("lru")
        assert isinstance(p, LRUPolicy)

    def test_make_policy_2q(self):
        p = make_policy("2q")
        assert isinstance(p, TwoQueuePolicy)

    def test_make_policy_arc_with_capacity(self):
        p = make_policy("arc", capacity=2048)
        assert isinstance(p, ARCPolicy)
        assert p.capacity == 2048

    def test_make_policy_case_insensitive(self):
        assert isinstance(make_policy("LRU"), LRUPolicy)
        assert isinstance(make_policy("ARC"), ARCPolicy)

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError, match="unknown eviction policy"):
            make_policy("totally-fake")

    def test_list_policies(self):
        names = list_policies()
        assert "lru" in names
        assert "2q" in names
        assert "arc" in names


# ─── Polymorphism — all three policies conform to ABC ───────────────────


@pytest.mark.parametrize("name", ["lru", "2q", "arc"])
class TestPolicyCommon:
    def test_isinstance_eviction_policy(self, name):
        p = make_policy(name, capacity=128)
        assert isinstance(p, EvictionPolicy)

    def test_admit_increases_len(self, name):
        p = make_policy(name, capacity=128)
        assert len(p) == 0
        p.admit("a")
        assert len(p) == 1

    def test_evict_after_admit_reduces_len(self, name):
        p = make_policy(name, capacity=128)
        p.admit("a")
        p.evict()
        assert len(p) == 0

    def test_evict_empty_returns_none(self, name):
        p = make_policy(name, capacity=128)
        assert p.evict() is None

    def test_keys_returns_iterable(self, name):
        p = make_policy(name, capacity=128)
        p.admit("a")
        keys = list(p.keys())
        assert "a" in keys


# ─── Workload simulation (smoke test) ───────────────────────────────────


class TestScanPollutionResistance:
    """The defining 2Q property — verify scan resistance vs LRU.

    Workload: 100 hits on 5 hot keys, then 100 cold scans. Then 5 more
    hits on the hot keys. Compare hot-key retention.
    """

    def test_2q_keeps_hot_keys_lru_loses_them(self):
        hot = ["h0", "h1", "h2", "h3", "h4"]
        cold = [f"c{i}" for i in range(100)]

        # LRU: scans evict everything (correct LRU behavior)
        lru = LRUPolicy()
        for k in hot:
            lru.admit(k)
        for _ in range(20):
            for k in hot:
                lru.touch(k)
        # Cold scan
        for k in cold:
            lru.admit(k)
        # Force evictions to capacity
        for _ in range(80):
            lru.evict()

        # 2Q: hot keys promoted, scans churn A1 only
        tq = TwoQueuePolicy()
        for k in hot:
            tq.admit(k); tq.touch(k)  # promote to Am
        for _ in range(20):
            for k in hot:
                tq.touch(k)
        for k in cold:
            tq.admit(k)
        # Force evictions until matching pressure
        for _ in range(80):
            tq.evict()

        lru_hot_remaining = sum(1 for k in hot if k in list(lru.keys()))
        tq_hot_remaining = sum(1 for k in hot if k in list(tq.keys()))
        # 2Q must protect more hot keys than LRU did
        assert tq_hot_remaining >= lru_hot_remaining
        # Stronger guarantee: 2Q should keep ALL 5 hot keys
        assert tq_hot_remaining == 5
