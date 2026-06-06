# SPDX-License-Identifier: Apache-2.0
"""Pluggable KV cache eviction policies — T2.1 / vllm#40270 backport.

Three policies:

  - **LRU** — classic least-recently-used. Drop-in compatible with
    vLLM's existing FreeQueue. Baseline / fallback.
  - **2Q** — two-queue: A1 (probationary) + Am (protected). Defends
    hot prefixes from scan pollution. Best for agent / RAG / multi-turn
    where one cold scan would otherwise evict the system prompt cache.
  - **ARC** — adaptive replacement: T1+T2 lists with B1+B2 ghost
    entries. Self-tunes between recency and frequency. Best when the
    workload mix shifts (chat → batch → agent in same process).

Design notes
─────────────
We expose a uniform `EvictionPolicy` ABC with three operations:

  - `touch(key)`: record a hit on an existing entry.
  - `admit(key)`: record a fresh insertion.
  - `evict() -> key`: choose a victim to evict + remove from internal state.

vLLM's prefix-cache manager calls these on every block hash. We keep
the policy purely structural (no torch / no async) so it's testable
on a laptop and porting to other engines is trivial.

When PN91 is OFF, the policy stays inert — vLLM's native FreeQueue
runs untouched. When PN91 is ON, the patcher wires our `dispatch()`
in front of the FreeQueue so eviction decisions route through the
selected policy.

References
──────────
- vllm#40270 — pluggable eviction hook (CLOSED unmerged; we re-implement
  the API surface and ship our own LRU/2Q/ARC).
- 2Q paper — "2Q: A Low Overhead High Performance Buffer Management
  Replacement Algorithm" by Theodore Johnson and Dennis Shasha, 1994.
- ARC paper — "ARC: A Self-Tuning, Low Overhead Replacement Cache"
  by Megiddo and Modha, 2003.

Author: Sandermage(Sander)-Barzov Aleksandr.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Hashable, Iterable, Optional


class EvictionPolicy(ABC):
    """Common interface every policy implements."""

    @abstractmethod
    def touch(self, key: Hashable) -> None:
        """Record a cache hit for `key`. No-op if key not tracked yet."""

    @abstractmethod
    def admit(self, key: Hashable) -> None:
        """Record a fresh insertion of `key`."""

    @abstractmethod
    def evict(self) -> Optional[Hashable]:
        """Choose + remove a victim. Returns None when nothing to evict."""

    @abstractmethod
    def remove(self, key: Hashable) -> None:
        """Drop `key` from the policy's bookkeeping (e.g. on explicit
        invalidation). No-op if not tracked."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of currently-tracked live entries."""

    @abstractmethod
    def keys(self) -> Iterable[Hashable]:
        """Iterate live entries — recency order, hottest last."""


# ─── LRU ────────────────────────────────────────────────────────────────


class LRUPolicy(EvictionPolicy):
    """Least-recently-used. O(1) touch / admit / evict via OrderedDict."""

    def __init__(self) -> None:
        self._data: OrderedDict[Hashable, None] = OrderedDict()

    def touch(self, key: Hashable) -> None:
        if key in self._data:
            self._data.move_to_end(key)

    def admit(self, key: Hashable) -> None:
        self._data[key] = None
        self._data.move_to_end(key)

    def evict(self) -> Optional[Hashable]:
        if not self._data:
            return None
        # popitem(last=False) drops the LRU entry (head of order)
        key, _ = self._data.popitem(last=False)
        return key

    def remove(self, key: Hashable) -> None:
        self._data.pop(key, None)

    def __len__(self) -> int:
        return len(self._data)

    def keys(self) -> Iterable[Hashable]:
        return list(self._data.keys())


# ─── 2Q ─────────────────────────────────────────────────────────────────


class TwoQueuePolicy(EvictionPolicy):
    """2Q (Johnson + Shasha 1994).

    Two FIFO queues: `A1` (probationary, FIFO) and `Am` (protected, LRU).

      - `admit(k)` → put `k` at the tail of A1 (assumes new entries are
        cold; only stays if it gets a second hit).
      - `touch(k)` → if in A1, move to Am tail. If in Am, LRU-promote.
      - `evict()` → drop A1 head first; only fall through to Am LRU when
        A1 is empty.

    Configurable split: `a1_ratio` controls how big A1 is relative to
    total capacity. Default 0.25 (A1 = 25% of slots) per the paper.
    """

    def __init__(self, *, a1_ratio: float = 0.25) -> None:
        if not (0.0 < a1_ratio < 1.0):
            raise ValueError(f"a1_ratio must be in (0,1), got {a1_ratio}")
        self._a1: OrderedDict[Hashable, None] = OrderedDict()
        self._am: OrderedDict[Hashable, None] = OrderedDict()
        self._a1_ratio = a1_ratio

    def touch(self, key: Hashable) -> None:
        if key in self._a1:
            # Promote: A1 → Am
            del self._a1[key]
            self._am[key] = None
            self._am.move_to_end(key)
        elif key in self._am:
            self._am.move_to_end(key)

    def admit(self, key: Hashable) -> None:
        # New entries always go to A1 (probationary)
        self._a1[key] = None
        self._a1.move_to_end(key)

    def evict(self) -> Optional[Hashable]:
        # Prefer A1 (cold candidates) — protect Am from scan pollution.
        if self._a1:
            key, _ = self._a1.popitem(last=False)
            return key
        if self._am:
            key, _ = self._am.popitem(last=False)
            return key
        return None

    def remove(self, key: Hashable) -> None:
        self._a1.pop(key, None)
        self._am.pop(key, None)

    def __len__(self) -> int:
        return len(self._a1) + len(self._am)

    def keys(self) -> Iterable[Hashable]:
        return list(self._a1.keys()) + list(self._am.keys())

    # Diagnostics
    def stats(self) -> dict:
        return {
            "a1_len": len(self._a1),
            "am_len": len(self._am),
            "a1_ratio": self._a1_ratio,
        }


# ─── ARC ────────────────────────────────────────────────────────────────


class ARCPolicy(EvictionPolicy):
    """Adaptive Replacement Cache (Megiddo + Modha, 2003).

    Four lists:
      - T1: recently used once (LRU)
      - T2: recently used more than once (LRU)
      - B1: ghost — recently evicted from T1
      - B2: ghost — recently evicted from T2

    Adaptive parameter `p` shifts capacity between T1 and T2 based on
    which ghost list (B1 or B2) gets a hit. No external tuning needed.

    Capacity `c` bounds T1+T2; ghost lists also bounded to `c` each.
    """

    def __init__(self, *, capacity: int = 4096) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self._c = capacity
        self._p = 0  # adaptive split point in [0, c]
        self._t1: OrderedDict[Hashable, None] = OrderedDict()
        self._t2: OrderedDict[Hashable, None] = OrderedDict()
        self._b1: OrderedDict[Hashable, None] = OrderedDict()
        self._b2: OrderedDict[Hashable, None] = OrderedDict()

    @property
    def capacity(self) -> int:
        return self._c

    def touch(self, key: Hashable) -> None:
        # Hit in T1 → promote to T2 (used >1 time now)
        if key in self._t1:
            del self._t1[key]
            self._t2[key] = None
            self._t2.move_to_end(key)
            return
        # Hit in T2 → just LRU-update
        if key in self._t2:
            self._t2.move_to_end(key)
            return
        # Hit in B1 (ghost) → adapt p UP, then move to T2
        if key in self._b1:
            delta = max(1, len(self._b2) // max(len(self._b1), 1))
            self._p = min(self._c, self._p + delta)
            self._replace(key)
            del self._b1[key]
            self._t2[key] = None
            self._t2.move_to_end(key)
            return
        # Hit in B2 (ghost) → adapt p DOWN, move to T2
        if key in self._b2:
            delta = max(1, len(self._b1) // max(len(self._b2), 1))
            self._p = max(0, self._p - delta)
            self._replace(key)
            del self._b2[key]
            self._t2[key] = None
            self._t2.move_to_end(key)
            return
        # Miss — caller should call admit() instead
        self.admit(key)

    def admit(self, key: Hashable) -> None:
        # Fresh entry → goes into T1
        if (len(self._t1) + len(self._b1)) == self._c:
            if len(self._t1) < self._c:
                # Drop oldest from B1
                if self._b1:
                    self._b1.popitem(last=False)
                self._replace(key)
            else:
                # T1 alone is full → evict its head
                if self._t1:
                    self._t1.popitem(last=False)
        else:
            total = (len(self._t1) + len(self._t2)
                     + len(self._b1) + len(self._b2))
            if total >= self._c:
                if total == 2 * self._c and self._b2:
                    self._b2.popitem(last=False)
                self._replace(key)
        self._t1[key] = None
        self._t1.move_to_end(key)

    def _replace(self, key: Hashable) -> None:
        """Internal eviction subroutine — moves victim from T1/T2 to ghost."""
        if (self._t1 and
                ((key in self._b2 and len(self._t1) == self._p)
                 or len(self._t1) > self._p)):
            victim, _ = self._t1.popitem(last=False)
            self._b1[victim] = None
            self._b1.move_to_end(victim)
        elif self._t2:
            victim, _ = self._t2.popitem(last=False)
            self._b2[victim] = None
            self._b2.move_to_end(victim)

    def evict(self) -> Optional[Hashable]:
        """Force an eviction from T1+T2 (for capacity-pressure cases)."""
        if not (self._t1 or self._t2):
            return None
        if self._t1 and (len(self._t1) > self._p or not self._t2):
            victim, _ = self._t1.popitem(last=False)
            self._b1[victim] = None
            return victim
        if self._t2:
            victim, _ = self._t2.popitem(last=False)
            self._b2[victim] = None
            return victim
        return None

    def remove(self, key: Hashable) -> None:
        for d in (self._t1, self._t2, self._b1, self._b2):
            d.pop(key, None)

    def __len__(self) -> int:
        return len(self._t1) + len(self._t2)

    def keys(self) -> Iterable[Hashable]:
        return list(self._t1.keys()) + list(self._t2.keys())

    def stats(self) -> dict:
        return {
            "t1_len": len(self._t1),
            "t2_len": len(self._t2),
            "b1_len": len(self._b1),
            "b2_len": len(self._b2),
            "p": self._p,
            "capacity": self._c,
        }


# ─── Factory ────────────────────────────────────────────────────────────


_POLICY_REGISTRY: dict[str, type[EvictionPolicy]] = {
    "lru": LRUPolicy,
    "2q": TwoQueuePolicy,
    "arc": ARCPolicy,
}


def make_policy(name: str, *, capacity: Optional[int] = None) -> EvictionPolicy:
    """Construct an eviction policy by canonical name.

    Raises:
        ValueError: when `name` is not recognized.
    """
    name_lc = name.strip().lower()
    if name_lc not in _POLICY_REGISTRY:
        raise ValueError(
            f"unknown eviction policy {name!r} — choose one of "
            f"{sorted(_POLICY_REGISTRY.keys())}"
        )
    cls = _POLICY_REGISTRY[name_lc]
    # ARC needs a capacity; LRU/2Q don't
    if cls is ARCPolicy:
        return ARCPolicy(capacity=capacity if capacity is not None else 4096)
    return cls()


def list_policies() -> list[str]:
    """All registered policy names."""
    return sorted(_POLICY_REGISTRY.keys())


__all__ = [
    "EvictionPolicy",
    "LRUPolicy",
    "TwoQueuePolicy",
    "ARCPolicy",
    "make_policy",
    "list_policies",
]
