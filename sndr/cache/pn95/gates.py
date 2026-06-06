# SPDX-License-Identifier: Apache-2.0
"""PN95 env-gate predicates — pure os.environ readers.

Every function here is stateless: reads one env var, returns
bool/int. No module-level state, no side effects, no torch/CUDA
dependency. Safe to import from any module at any time.

Extracted from ``_pn95_runtime.py`` in M.4.1. The legacy module
re-exports each symbol so existing call sites and text-patch
anchors keep working without edit.
"""
from __future__ import annotations

import os


def _enabled() -> bool:
    """True iff GENESIS_ENABLE_PN95_TIER_AWARE_CACHE=1."""
    return os.environ.get(
        "GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", "0"
    ).strip().lower() in ("1", "true", "yes", "on")


def _phase5_virt_enabled() -> bool:
    """Path C v1.0 Phase 5 master gate — KV pool virtualization.

    Default OFF for safety. When enabled:
      - Anchor #9 inflates available_memory in pre-flight check by CPU tier capacity
      - Anchor #10 caps physical KVCacheTensor allocation to GPU-only memory
      - Anchor #11+ (later sessions) wire up logical/physical block split

    Independent from GENESIS_ENABLE_PN95_TIER_AWARE_CACHE — Phase 5 virt
    requires both: PN95 base infrastructure AND VIRT opt-in.
    """
    if not _enabled():
        return False  # virt requires PN95 base
    return os.environ.get(
        "GENESIS_PN95_VIRT_ENABLE", "0"
    ).strip().lower() in ("1", "true", "yes", "on")


def _pn95_async_enabled() -> bool:
    """Path C v1.0 Quality-First Sprint Q1 B1 — async stream gate.

    Default ON — async stream usage is lossless (stream sync ensures
    correctness). Operator can disable via GENESIS_PN95_ASYNC_STREAM=0
    for debugging or when cudagraph capture issues are observed.
    """
    return os.environ.get(
        "GENESIS_PN95_ASYNC_STREAM", "1"
    ).strip().lower() in ("1", "true", "yes", "on")


def _pn95_use_stream_pool() -> bool:
    """Stream-pool mode (upstream PR #40020-style event polling).

    When enabled, demote/promote submit work into the pooled-stream queue
    in _pn95_stream_pool and synchronize via end_event.synchronize() on
    that stream — no host-blocking torch.cuda.current_stream().synchronize()
    on the default stream. The default stream stays free to dispatch the
    next attention forward while PCIe DMA runs on the pooled stream.

    Default OFF (singleton stream path stays exact). Set
    GENESIS_PN95_USE_STREAM_POOL=1 to opt in.
    """
    return os.environ.get(
        "GENESIS_PN95_USE_STREAM_POOL", "0",
    ).strip().lower() in ("1", "true", "yes", "on")


def _pn95_prefetch_neighbors_enabled() -> bool:
    """Env gate: GENESIS_PN95_PREFETCH_NEIGHBORS=1 enables auto-warm.

    When a vllm prefix-cache hit lands in notify_touch, we know the
    request will likely traverse adjacent block_hashes next. If any of
    those neighbors are in L2 / disk (not yet in L1 pinned), pre-warm
    them into the pinned pool so the next promote_on_miss is fast.
    Default OFF — operators turn on when running multi-stream workloads
    where adjacency matters more than one-shot.
    """
    return os.environ.get(
        "GENESIS_PN95_PREFETCH_NEIGHBORS", "0",
    ).strip().lower() in ("1", "true", "yes", "on")


def _pn95_prefetch_window() -> int:
    """How many trailing block_hashes from the admit-order tail to pre-warm
    around a touch event. Default 8 — small enough that the warm-up cost
    stays under typical attention compute time.
    """
    try:
        return max(0, min(64, int(os.environ.get(
            "GENESIS_PN95_PREFETCH_WINDOW", "8"))))
    except (ValueError, TypeError):
        return 8


def _pn95_store_threshold() -> int:
    try:
        return max(0, int(os.environ.get("GENESIS_PN95_STORE_THRESHOLD", "0")))
    except (ValueError, TypeError):
        return 0


def _pn95_block_size_factor() -> int:
    try:
        v = int(os.environ.get("GENESIS_PN95_BLOCK_SIZE_FACTOR", "1"))
    except (ValueError, TypeError):
        v = 1
    # Clamp to sensible range; 8+ rarely helps and uses lots of pinned slots.
    return max(1, min(8, v))


def _pn95_layer_aware_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN95_LAYER_AWARE_DEMOTE", "0",
    ).strip().lower() in ("1", "true", "yes", "on")


def _read_env_int(name: str, default: int) -> int:
    """Read int env var with fail-silent fallback."""
    try:
        return int(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        return default
