# SPDX-License-Identifier: Apache-2.0
"""PN201 — threshold-gated empty_cache between scheduler ticks (Tier 1.C).

Defragments the PyTorch CUDA caching allocator when GPU free-block
count drops below threshold. The 319 MiB "reserved but unallocated"
observed in OOM logs is exactly this fragmentation — pageable cache
slabs the allocator holds in case a future torch.zeros matches that
exact shape, but which never does after chunked-prefill chunk size
varies.

Threshold-gated to keep the hot path fast — `empty_cache()` is a
synchronous syscall costing 2-10 ms. We only pay it when fragmentation
actually matters (≤8 free blocks left). With Tier 2/3 (CPU offload
landing later), this hook becomes the trigger for `pn203_demote_cold`
calls — same scheduler tick, same pressure signal.

Reuses existing PN95 scheduler_tick anchor (`_pn95_runtime.scheduler_tick`
is already hook'd into vllm's `Scheduler.schedule` via PN95's tier-aware
text-patch on `vllm/v1/core/sched/scheduler.py`). PN201 adds a post-
demote empty_cache call there.

Env gate: `GENESIS_ENABLE_PN201_SCHEDULER_EMPTY_CACHE=1`.
Threshold: `GENESIS_PN201_EMPTY_CACHE_FREE_BLOCKS_THRESHOLD` (default 8).
Cooldown ticks: `GENESIS_PN201_EMPTY_CACHE_COOLDOWN` (default 50 ticks
~= 5s — avoid back-to-back empty_cache when pool stays tight).

No text-patch; this is a runtime hook into the existing PN95 scheduler_
tick path. Activated via importing `pn201_maybe_empty_cache` from
_pn95_runtime where the tick already runs.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.wiring.pn201_scheduler_empty_cache")


def _enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN201_SCHEDULER_EMPTY_CACHE", "0",
    ).strip().lower() in ("1", "true", "yes", "on")


def apply() -> tuple[str, str]:
    """PN201 has no text-patch surface — it's a runtime hook activated by
    env-gated check inside _pn95_runtime.scheduler_tick. Apply just
    validates the env is set and reports."""
    if not _enabled():
        return "skipped", "PN201 disabled (set GENESIS_ENABLE_PN201_SCHEDULER_EMPTY_CACHE=1)"
    return "applied", (
        "PN201 runtime hook active — empty_cache will fire when "
        "free_blocks < threshold (default 8) with cooldown"
    )
