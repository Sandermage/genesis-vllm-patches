# SPDX-License-Identifier: Apache-2.0
"""Path C v7.73.x (PN95) — Tier-aware KV cache manager.

Owns the per-tier eviction policy + a host-pinned CPU pool.  Pure
runtime layer: schema (CacheTier / CacheConfig) lives in
`model_configs/schema.py`; eviction policies (LRU/2Q/ARC) live in
`cache/eviction_policies.py`; this module composes them into a
multi-tier hierarchy with promote/demote between adjacent tiers.

Design contract:
  - Tier 0 = closest to compute (typically GPU). Higher index = farther.
  - `admit(key, mm_origin=False)` — newly cached block enters tier 0.
  - `touch(key)` — hit on `key`. Returns None if still tier 0; returns
    a `bytes` view (CPU pool slot) if the page lives in tier 1; the
    caller is responsible for promoting it back to GPU.
  - `demote_to_threshold()` — called when GPU free-VRAM drops below
    `tier_low_water_pct`. Walks tier 0 candidates (vision-first if
    enabled), copies them to tier 1, marks them demoted in policy.
    Returns the number of pages moved.
  - `evict_terminal()` — last-resort drop from the coldest tier when
    even that's full.
  - `register_mamba_excluded(group_id)` — caller marks any
    `MambaSpec` group at startup; pages in those groups are filtered
    out of demote candidates.

The CPU "pool" in this skeleton is a `bytearray` slab divided into
fixed-size slots. The real text-patch into vLLM (PN95 wire-in, Day 3)
will swap this for `torch.empty(N, slot_nbytes, dtype=uint8,
pin_memory=True)` when torch is importable. The skeleton API is
identical so the patch + tests don't change.

Author: Sandermage(Sander)-Barzov Aleksandr.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Hashable, Iterable, Optional

from vllm.sndr_core.cache.eviction_policies import (
    EvictionPolicy, make_policy,
)


@dataclass
class _PageMeta:
    """Internal per-page bookkeeping inside a TierManager."""
    key: Hashable
    tier_idx: int       # which tier the page currently lives in
    mm_origin: bool     # True if from a multimodal/image span
    group_id: str       # KV group (for mamba exclusion)
    cpu_slot_idx: Optional[int] = None  # tier 1 slot when demoted


class _CpuSlab:
    """Fixed-size pinned-memory slot pool with LAZY allocation.

    Day 7a (Path C v7.73.x): bridges from skeleton bytearray to a real
    torch.empty(pin_memory=True) tensor when torch is importable.
    Behavior is byte-identical between the two backends; tests can
    use either path.

    Path C v1.0 Phase 1 fix: storage is allocated LAZILY on first
    store() call, NOT at __init__ time. This prevents OOM when
    `tier1.capacity_gib` is large (e.g. 40 GiB pinned host RAM × 2
    TP workers > container memory limit). For Phase 1 (observability
    only, no real demote), no allocation happens at all.

    When torch is missing (CPU-only Mac dev), falls back to bytearray
    so unit tests run without GPU. Real production deployments
    (vllm worker process) always have torch + CUDA available, so the
    pinned-memory path is taken on first use.
    """

    def __init__(self, n_slots: int, slot_nbytes: int,
                  *, force_bytearray: bool = False):
        self.n_slots = n_slots
        self.slot_nbytes = slot_nbytes
        self._free: list[int] = list(range(n_slots))
        self._uses_torch = False
        self._storage = None
        self._force_bytearray = force_bytearray

    def _ensure_storage(self) -> None:
        """Lazy allocation — fires only on first store() call."""
        if self._storage is not None:
            return
        if not self._force_bytearray:
            try:
                import torch
                self._storage = torch.empty(
                    (self.n_slots, self.slot_nbytes),
                    dtype=torch.uint8,
                    pin_memory=torch.cuda.is_available(),
                )
                self._uses_torch = True
            except Exception:
                pass
        if self._storage is None:
            self._storage = bytearray(self.n_slots * self.slot_nbytes)

    @property
    def uses_torch(self) -> bool:
        return self._uses_torch

    def alloc(self) -> Optional[int]:
        """Allocate one slot; return its index or None if pool is full."""
        if not self._free:
            return None
        return self._free.pop()

    def free(self, idx: int) -> None:
        """Release a slot back to the free list."""
        if 0 <= idx < self.n_slots:
            self._free.append(idx)

    def n_free(self) -> int:
        return len(self._free)

    def n_used(self) -> int:
        return self.n_slots - len(self._free)

    def store(self, idx: int, payload: bytes) -> None:
        """Write `payload` into slot `idx`.

        Real wire-in (Day 7b): caller passes a torch.Tensor view of the
        GPU block instead of bytes; we use `slot.copy_(gpu_view, non_blocking=True)`.
        For now (skeleton callers), bytes path stays.
        """
        if not (0 <= idx < self.n_slots):
            raise IndexError(f"slot {idx} out of range [0, {self.n_slots})")
        if len(payload) > self.slot_nbytes:
            raise ValueError(
                f"payload {len(payload)} > slot_nbytes {self.slot_nbytes}"
            )
        self._ensure_storage()
        if self._uses_torch:
            import torch
            # Convert payload bytes to a CPU uint8 tensor and copy
            src = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
            self._storage[idx, :len(payload)].copy_(src)
            if len(payload) < self.slot_nbytes:
                self._storage[idx, len(payload):].zero_()
        else:
            start = idx * self.slot_nbytes
            self._storage[start:start + len(payload)] = payload
            if len(payload) < self.slot_nbytes:
                self._storage[start + len(payload):start + self.slot_nbytes] = (
                    b"\0" * (self.slot_nbytes - len(payload))
                )

    def store_from_gpu(self, idx: int, gpu_view: Any) -> None:
        """Day 7b real path: cudaMemcpyAsync GPU→CPU pinned slot.

        `gpu_view` must be a torch.Tensor view of `slot_nbytes` bytes
        of GPU memory. Caller is responsible for the layout flatten.
        Falls back to bytes path when torch unavailable.
        """
        if not (0 <= idx < self.n_slots):
            raise IndexError(f"slot {idx} out of range")
        self._ensure_storage()
        if not self._uses_torch:
            # Best-effort: test mocks expose _ba directly, real torch
            # tensors get the .cpu().numpy() round-trip.
            try:
                if hasattr(gpu_view, "_ba"):
                    self.store(idx, bytes(gpu_view._ba))
                else:
                    self.store(idx, bytes(gpu_view.contiguous().cpu().view(-1).numpy()))
            except Exception:
                pass
            return
        import torch
        # Async copy GPU → pinned CPU slot
        flat = gpu_view.contiguous().view(-1)
        n = min(flat.numel(), self.slot_nbytes)
        self._storage[idx, :n].copy_(
            flat[:n].to(dtype=torch.uint8, copy=False),
            non_blocking=True,
        )
        if n < self.slot_nbytes:
            self._storage[idx, n:].zero_()

    def load(self, idx: int) -> bytes:
        """Read slot `idx` as bytes (synchronous, used by tests + skeleton)."""
        if self._storage is None:
            # Lazy slab not yet allocated → return empty bytes
            return b"\0" * self.slot_nbytes
        if self._uses_torch:
            return bytes(self._storage[idx].cpu().numpy())
        start = idx * self.slot_nbytes
        return bytes(self._storage[start:start + self.slot_nbytes])

    def load_into_gpu(self, idx: int, gpu_view: Any) -> None:
        """Day 7b real promote path: cudaMemcpyAsync CPU pinned slot → GPU.

        `gpu_view` is a writable torch.Tensor on GPU.
        """
        if not (0 <= idx < self.n_slots):
            raise IndexError(f"slot {idx} out of range")
        if self._storage is None:
            # Lazy slab not yet allocated → no-op
            return
        if not self._uses_torch:
            # Bytes path (test mock) — copy from bytearray slab to mock view._ba
            if hasattr(gpu_view, "_ba"):
                start = idx * self.slot_nbytes
                end = start + self.slot_nbytes
                src_bytes = bytes(self._storage[start:end])
                n = min(len(gpu_view._ba), len(src_bytes))
                gpu_view._ba[:n] = src_bytes[:n]
            return
        flat = gpu_view.contiguous().view(-1)
        n = min(flat.numel(), self.slot_nbytes)
        flat[:n].copy_(
            self._storage[idx, :n].to(flat.dtype, copy=False),
            non_blocking=True,
        )


class TierManager:
    """Multi-tier KV cache manager — composes EvictionPolicy per tier.

    Skeleton (Day 2): pure-Python admit/touch/demote/evict bookkeeping
    against a `_CpuSlab` byte pool. No real CUDA copies.

    Real-runtime contract (Day 3 wire-in):
      - `_CpuSlab._storage` becomes `torch.empty(... pin_memory=True)`.
      - `admit()` is a no-op other than tier 0 policy bookkeeping.
      - `touch()` returns the GPU block pointer when in tier 0; when
        in tier 1, fetches CPU bytes and triggers a `cuda_block.copy_(
        cpu_slot, non_blocking=True)` against a free GPU block.
      - `demote_to_threshold()` runs on a separate CUDA stream from
        the cudagraph capture stream.
    """

    def __init__(
        self,
        tiers: list,                  # list[CacheTier]
        slot_nbytes: int,
        *,
        vision_demote_first: bool = True,
        spec_decode_hot_ring: int = 0,  # last N admits never demote
        host_capacity_cap_gib: Optional[float] = None,
    ):
        if not tiers:
            raise ValueError("TierManager requires at least one tier")
        self.tiers = tuple(tiers)
        self.slot_nbytes = int(slot_nbytes)
        self.vision_demote_first = vision_demote_first
        self.spec_decode_hot_ring = int(spec_decode_hot_ring)
        self.host_capacity_cap_gib = host_capacity_cap_gib

        # Per-tier eviction policy (tier 0 = compute-side, tier ≥1 = spillover)
        self._policies: list[EvictionPolicy] = [
            make_policy(t.eviction_policy) for t in self.tiers
        ]

        # CPU spill pool sized from tier 1 capacity (when present).
        # If host_capacity_cap_gib is supplied, the effective capacity is
        # min(tier-1 declared, host cap). The host cap mirrors SGLang
        # HiCache's HICACHE_HOST_MEMORY_RESERVE_BYTES contract: the
        # operator (or factory) computes available_host - reserve and
        # passes it here so a misconfigured tier-1 cannot OOM the box.
        self._cpu_slab: Optional[_CpuSlab] = None
        if len(self.tiers) >= 2 and self.tiers[1].device == "cpu":
            declared_gib = float(self.tiers[1].capacity_gib)
            effective_gib = declared_gib
            if host_capacity_cap_gib is not None:
                effective_gib = min(declared_gib, float(host_capacity_cap_gib))
            self._effective_cpu_capacity_gib = effective_gib
            n_slots = max(1, int(effective_gib * (1 << 30) // self.slot_nbytes))
            self._cpu_slab = _CpuSlab(n_slots, self.slot_nbytes)
        else:
            self._effective_cpu_capacity_gib = 0.0

        # Page bookkeeping
        self._pages: dict[Hashable, _PageMeta] = {}
        self._mamba_excluded: set[str] = set()
        self._admit_order: list[Hashable] = []  # for hot-ring computation

        # Active-block protection: keys touched within the last
        # `_active_ttl_ticks` admit/touch operations are skipped during
        # demote-candidate selection. Mirrors LMCache's pin_count
        # concept but cheaper: no explicit reference counting, just a
        # rolling window. Demote never races with a worker that is
        # currently reading the block.
        self._active_ttl_ticks: int = 0
        self._tick_counter: int = 0
        self._active_last_tick: dict[Hashable, int] = {}

    # ─── API surface ────────────────────────────────────────────────────

    def admit(self, key: Hashable, *,
              mm_origin: bool = False, group_id: str = "") -> None:
        """Record a new page entering tier 0."""
        if group_id in self._mamba_excluded:
            # Mamba state: bookkeep but never demote candidate.
            self._pages[key] = _PageMeta(
                key=key, tier_idx=0, mm_origin=mm_origin, group_id=group_id,
            )
            self._policies[0].admit(key)
            return
        self._pages[key] = _PageMeta(
            key=key, tier_idx=0, mm_origin=mm_origin, group_id=group_id,
        )
        self._policies[0].admit(key)
        self._admit_order.append(key)

    def touch(self, key: Hashable) -> Optional[bytes]:
        """Record a hit on `key`. Returns CPU bytes if demoted, else None.

        When non-None, caller must promote (load to GPU) and call
        `mark_promoted(key)` to update bookkeeping.
        """
        meta = self._pages.get(key)
        if meta is None:
            return None
        # Always touch the policy of the residence tier.
        self._policies[meta.tier_idx].touch(key)
        if meta.tier_idx == 0:
            return None
        # Tier ≥ 1 hit: if promote_on_hit, return payload for caller to upload.
        tier = self.tiers[meta.tier_idx]
        if not getattr(tier, "promote_on_hit", True):
            return None
        if self._cpu_slab is None or meta.cpu_slot_idx is None:
            return None
        return self._cpu_slab.load(meta.cpu_slot_idx)

    def mark_promoted(self, key: Hashable) -> None:
        """Caller-side helper: page was just lifted back to tier 0."""
        meta = self._pages.get(key)
        if meta is None or meta.tier_idx == 0:
            return
        if self._cpu_slab is not None and meta.cpu_slot_idx is not None:
            self._cpu_slab.free(meta.cpu_slot_idx)
            meta.cpu_slot_idx = None
        # Move to tier 0 policy
        self._policies[meta.tier_idx].remove(key) if hasattr(
            self._policies[meta.tier_idx], "remove") else None
        self._policies[0].admit(key)
        meta.tier_idx = 0

    def demote_to_threshold(self, payloads: dict[Hashable, bytes]) -> int:
        """Demote tier-0 pages until tier 0 fill ≤ low_water_pct.

        `payloads` provides the bytes for any page eligible to demote
        (caller is the KV manager — it has the GPU block pointer +
        knows the per-key payload). Returns count of pages moved.

        Vision-first policy: when `vision_demote_first` is True, we
        drain mm_origin pages first.
        """
        if len(self.tiers) < 2 or self._cpu_slab is None:
            return 0  # no spillover tier configured
        tier0 = self.tiers[0]
        target_used_pct = tier0.low_water_pct
        # Approximate: count tier-0 pages in our bookkeeping.
        n_tier0 = sum(1 for m in self._pages.values()
                      if m.tier_idx == 0
                      and m.group_id not in self._mamba_excluded)
        if n_tier0 == 0:
            return 0
        capacity_pages = max(
            1, int(tier0.capacity_gib * (1 << 30) // self.slot_nbytes)
        )
        target_n = int(capacity_pages * target_used_pct)
        to_move = max(0, n_tier0 - target_n)
        if to_move == 0:
            return 0

        candidates = self._demote_candidates()
        moved = 0
        for key in candidates:
            if moved >= to_move:
                break
            meta = self._pages.get(key)
            if meta is None or meta.tier_idx != 0:
                continue
            payload = payloads.get(key)
            if payload is None:
                continue  # caller didn't provide the payload — skip
            slot = self._cpu_slab.alloc()
            if slot is None:
                # CPU slab full — try terminal eviction first
                self.evict_terminal()
                slot = self._cpu_slab.alloc()
                if slot is None:
                    break
            self._cpu_slab.store(slot, payload)
            meta.cpu_slot_idx = slot
            meta.tier_idx = 1
            self._policies[0].remove(key) if hasattr(
                self._policies[0], "remove") else None
            self._policies[1].admit(key)
            moved += 1
        return moved

    def demote_block(self, layer_name: str, block_idx: int) -> bool:
        """Path C v1.0 Phase 2 (UNIFIED_CONFIG plan 2026-05-09): real
        bytes movement — demote one (layer, block) GPU view to a CPU
        pinned slot via cudaMemcpyAsync.

        Requires `register_kv_caches()` to have populated
        `_attention_views[layer_name]` with the live tensor ref +
        per-block geometry. If not, returns False (silent no-op).

        Returns True if the demote happened; False on any error or
        when prerequisites are missing.
        """
        if len(self.tiers) < 2 or self._cpu_slab is None:
            return False
        views = getattr(self, "_attention_views", None)
        if not views or layer_name not in views:
            return False
        info = views[layer_name]
        tensor = info.get("tensor")
        num_blocks = info.get("num_blocks", 0)
        bytes_per_block = info.get("bytes_per_block", 0)
        if (tensor is None or block_idx < 0 or block_idx >= num_blocks
                or bytes_per_block <= 0):
            return False
        try:
            view = tensor[block_idx]
            slot = self._cpu_slab.alloc()
            if slot is None:
                # CPU pool full — try terminal eviction first
                self.evict_terminal()
                slot = self._cpu_slab.alloc()
                if slot is None:
                    return False
            self._cpu_slab.store_from_gpu(slot, view)
            key = (layer_name, block_idx)
            self._pages[key] = _PageMeta(
                key=key, tier_idx=1, mm_origin=False,
                group_id=info.get("group_id", "attn_eligible"),
                cpu_slot_idx=slot,
            )
            self._policies[1].admit(key)
            return True
        except Exception:
            return False

    def promote_block(self, layer_name: str, block_idx: int) -> bool:
        """Path C v1.0 Phase 2: reverse of demote_block — copy CPU
        pinned slot back to the GPU tensor's per-block view.

        Phase 3 — also bumps observability counter.

        Returns True iff the promote happened; False on any error.
        """
        if self._cpu_slab is None:
            return False
        views = getattr(self, "_attention_views", None)
        if not views or layer_name not in views:
            return False
        key = (layer_name, block_idx)
        meta = self._pages.get(key)
        if meta is None or meta.tier_idx != 1 or meta.cpu_slot_idx is None:
            return False
        info = views[layer_name]
        tensor = info.get("tensor")
        if tensor is None:
            return False
        try:
            view = tensor[block_idx]
            self._cpu_slab.load_into_gpu(meta.cpu_slot_idx, view)
            self._cpu_slab.free(meta.cpu_slot_idx)
            meta.cpu_slot_idx = None
            if hasattr(self._policies[1], "remove"):
                self._policies[1].remove(key)
            self._policies[0].admit(key)
            meta.tier_idx = 0
            # Phase 3 observability — bump global counter (best-effort,
            # avoid circular import via late lookup)
            try:
                import sys as _sys
                _rt = _sys.modules.get("vllm.sndr_core.cache._pn95_runtime")
                if _rt is not None:
                    _rt._PN95_STATS["blocks_promoted_total"] += 1
            except Exception:
                pass
            return True
        except Exception:
            return False

    def n_attention_layers_eligible(self) -> int:
        """How many attention layers are registered for demote
        (Path C v1.0 Phase 2)."""
        views = getattr(self, "_attention_views", None)
        return 0 if views is None else len(views)

    def evict_terminal(self) -> Optional[Hashable]:
        """Drop one page from the coldest tier. Returns evicted key."""
        if not self.tiers:
            return None
        last_idx = len(self.tiers) - 1
        last_pol = self._policies[last_idx]
        try:
            victim = last_pol.evict()
        except Exception:
            return None
        if victim is None:
            return None
        meta = self._pages.pop(victim, None)
        if meta is not None and meta.cpu_slot_idx is not None and self._cpu_slab is not None:
            self._cpu_slab.free(meta.cpu_slot_idx)
        return victim

    def register_mamba_excluded(self, group_id: str) -> None:
        """Mark `group_id` as containing Mamba/SSM state — never demote.

        Called at TierManager init by the wire-in patch when it walks
        `KVCacheGroupSpec` and finds `MambaSpec` instances.
        """
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("group_id must be non-empty str")
        self._mamba_excluded.add(group_id)

    def is_mamba_excluded(self, group_id: str) -> bool:
        return group_id in self._mamba_excluded

    # ─── Inspection helpers (for tests + observability) ──────────────────

    def n_pages_at_tier(self, tier_idx: int) -> int:
        return sum(1 for m in self._pages.values() if m.tier_idx == tier_idx)

    def n_pages(self) -> int:
        return len(self._pages)

    def cpu_slab_n_used(self) -> int:
        return 0 if self._cpu_slab is None else self._cpu_slab.n_used()

    def cpu_slab_n_free(self) -> int:
        return 0 if self._cpu_slab is None else self._cpu_slab.n_free()

    def stats(self) -> dict:
        return {
            "n_pages_total": self.n_pages(),
            "n_pages_per_tier": [
                self.n_pages_at_tier(i) for i in range(len(self.tiers))
            ],
            "cpu_slab_used": self.cpu_slab_n_used(),
            "cpu_slab_free": self.cpu_slab_n_free(),
            "n_mamba_excluded_groups": len(self._mamba_excluded),
        }

    # ─── Internal: candidate ordering ────────────────────────────────────

    # ─── Active-block protection (LMCache-inspired pin equivalent) ──────

    def set_active_ttl(self, ttl_ticks: int) -> None:
        """Configure how long a touched key is considered active.

        `ttl_ticks=0` disables the protection (default). A non-zero
        value treats keys touched within the last `ttl_ticks`
        admit/touch operations as in-flight: they are skipped from
        `_demote_candidates`. Mirrors LMCache's pin_count concept
        without per-block counters — operators are expected to call
        `mark_active(key)` from the worker hot path when reads begin.
        """
        if ttl_ticks < 0:
            raise ValueError("ttl_ticks must be non-negative")
        self._active_ttl_ticks = int(ttl_ticks)

    def mark_active(self, key: Hashable) -> None:
        """Stamp `key` as currently in-flight at the worker side.

        Effective only when `set_active_ttl` configured a non-zero
        window. Cheap: one dict assignment per call.
        """
        if self._active_ttl_ticks <= 0:
            return
        self._tick_counter += 1
        self._active_last_tick[key] = self._tick_counter

    def is_active(self, key: Hashable) -> bool:
        """True when `key` was marked active inside the TTL window."""
        if self._active_ttl_ticks <= 0:
            return False
        last = self._active_last_tick.get(key)
        if last is None:
            return False
        return (self._tick_counter - last) < self._active_ttl_ticks

    # ─── Internal: candidate ordering ────────────────────────────────────

    def _demote_candidates(self) -> Iterable[Hashable]:
        """Iterate tier-0 keys in demote order.

        Order:
          1. Skip mamba-excluded groups (HARD).
          2. Skip the spec-decode hot ring (last N admits).
          3. Skip keys marked active inside the protection TTL window.
          4. If vision_demote_first: mm_origin pages first.
          5. Within each bucket: LRU order (oldest admit first).
        """
        hot_ring = set(self._admit_order[-self.spec_decode_hot_ring:]) \
            if self.spec_decode_hot_ring > 0 else set()

        eligible: list[_PageMeta] = []
        for key in self._admit_order:
            meta = self._pages.get(key)
            if meta is None or meta.tier_idx != 0:
                continue
            if meta.group_id in self._mamba_excluded:
                continue
            if key in hot_ring:
                continue
            if self.is_active(key):
                continue
            eligible.append(meta)

        if self.vision_demote_first:
            eligible.sort(key=lambda m: (not m.mm_origin, ))
        return [m.key for m in eligible]

    # ─── Atomic demote selection (upstream PR #40020 inspired) ───────────

    def demote_candidates_atomic(
        self, n: int, protected: Optional[set] = None,
    ) -> Optional[list[Hashable]]:
        """All-or-nothing variant of _demote_candidates.

        Returns exactly `n` keys eligible for demote (skipping mamba groups,
        the spec-decode hot ring, active TTL, AND any key in `protected`).
        Returns None when fewer than `n` candidates can be found without
        violating the protection rules — caller MUST NOT proceed with
        partial demote in that case (the alternative — silently partial —
        leaves L1/L2/L3 in an inconsistent state where some block bytes
        are out of GPU but their hashes never made it to the prefix store).

        Mirrors upstream `vllm.v1.kv_offload.cpu.policies.base.evict`
        contract from PR #40020 — atomic eviction with `protected` set.
        """
        if n <= 0:
            return []
        protected_set = protected or set()
        hot_ring = set(self._admit_order[-self.spec_decode_hot_ring:]) \
            if self.spec_decode_hot_ring > 0 else set()
        out: list[Hashable] = []
        # Iterate in admit order (LRU first) — same ordering as
        # _demote_candidates so vision_demote_first still implicitly works
        # through the meta lookup pattern; here we sort by mm_origin if
        # vision_demote_first is set and we have headroom in our walk.
        candidates_pool: list[Hashable] = []
        for key in self._admit_order:
            meta = self._pages.get(key)
            if meta is None or meta.tier_idx != 0:
                continue
            if meta.group_id in self._mamba_excluded:
                continue
            if key in hot_ring or key in protected_set:
                continue
            if self.is_active(key):
                continue
            candidates_pool.append(key)
        if self.vision_demote_first:
            # mm_origin first, then LRU within each bucket
            candidates_pool.sort(
                key=lambda k: (not self._pages[k].mm_origin,
                               self._admit_order.index(k)
                               if k in self._admit_order else 1 << 30),
            )
        for key in candidates_pool[:n]:
            out.append(key)
        if len(out) < n:
            return None  # cannot satisfy atomically
        return out


# ─── Factory helper ─────────────────────────────────────────────────────


def _host_capacity_cap_gib() -> Optional[float]:
    """Compute the host RAM ceiling for the CPU spill tier.

    Mirrors SGLang HiCache's HICACHE_HOST_MEMORY_RESERVE_BYTES contract:
    available_host_RAM minus a configurable reserve. The result caps
    `tiers[1].capacity_gib` so a misconfigured config cannot OOM the
    host. Returns None when host inspection is unavailable (no
    /proc/meminfo and no psutil) — in that case the operator's
    declared value passes through unchanged.

    Env knobs:
      - GENESIS_PN95_HOST_RESERVE_GIB (default 8.0): GiB reserved for
        the OS and other processes.
      - GENESIS_PN95_HOST_CAP_GIB (override): explicit cap that wins
        over auto-computation.
    """
    override = os.environ.get("GENESIS_PN95_HOST_CAP_GIB")
    if override:
        try:
            v = float(override)
            return v if v > 0 else None
        except ValueError:
            pass
    try:
        reserve_gib = float(os.environ.get("GENESIS_PN95_HOST_RESERVE_GIB", "8"))
    except ValueError:
        reserve_gib = 8.0
    total_gib = _read_total_host_ram_gib()
    if total_gib is None:
        return None
    cap = total_gib - reserve_gib
    return cap if cap > 0 else None


def _read_total_host_ram_gib() -> Optional[float]:
    """Return total host RAM in GiB via /proc/meminfo, else None.

    Pure-stdlib; does not import psutil so the runtime stays light.
    """
    try:
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        kb = int(parts[1])
                        return kb / (1 << 20)
    except (OSError, ValueError):
        return None
    return None


def make_tier_manager(cfg, *, slot_nbytes: int = 1 << 16) -> Optional[TierManager]:
    """Build a TierManager from a ModelConfig, or return None.

    Returns None when:
      - cfg.cache_config is None (no PN91/PN95 declared)
      - cfg.cache_config.tiers is empty (single-tier PN91 mode)

    Default slot_nbytes = 64 KiB (~16 tokens of fp16 KV at head_dim=128
    × 2 K/V × num_kv_heads=4). Real wire-in computes this from the
    KV cache spec at runtime.

    The host RAM ceiling (`_host_capacity_cap_gib()`) is forwarded so
    a tier-1 declaration that exceeds available host RAM falls back to
    the safe ceiling instead of OOMing the host.
    """
    cc = getattr(cfg, "cache_config", None)
    if cc is None or not getattr(cc, "tiers", None):
        return None
    return TierManager(
        cc.tiers,
        slot_nbytes=slot_nbytes,
        vision_demote_first=cc.vision_demote_first,
        host_capacity_cap_gib=_host_capacity_cap_gib(),
    )
