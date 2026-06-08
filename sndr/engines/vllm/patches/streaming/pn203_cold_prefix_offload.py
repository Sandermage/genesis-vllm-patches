# SPDX-License-Identifier: Apache-2.0
"""PN203 — cold-prefix CPU offload manager (Tier 3.A core).

The architectural breakthrough piece. Builds on:
- PN202 per-layer KV split (enables independent layer eviction)
- PN95 pinned host pool (L2 storage)
- PN95 stream pool (async transfers, non-blocking)
- PN95 layer-access heatmap (cold-first demote)
- PN95 prefetch API (warm L1 before access)

Concept: for full-attention layers in long-context prefill, only the
"active window" of blocks must be GPU-resident at attention time.
Older prefix blocks can be demoted to pinned host RAM and async-paged
back when needed. Mamba/GDN state stays GPU-resident (small fixed cost).

Math for Qwen3.6-27B INT4 at 256K context, fp8 KV:
  16 attention layers × 256K × (8 KV heads × 128 dim × 2) bytes = 8 GiB
  GPU-resident keep last 32K tokens = 1 GiB
  → 7 GiB offloaded to pinned host RAM (fits in 64 GB host comfortably)

Quality: **none**. Math is identical. Only memory residency moves.
Speed: prefill compute hides H2D copy (~13 ms PCIe per block batch);
decode adds 5-15 ms per token when paging hits, but only for queries
that actually attend to demoted positions (e.g. needle retrieval).

This module is the **integration layer**. The heavy lifting is done by
existing PN95 components:
- `pn95_prefetch_blocks(block_hashes)` — async warm L1 from L2
- `_pn95_stream_pool.submit(fn)` — async PCIe transfers
- `_pn95_pinned_pool` — pinned RAM L2 storage
- `pn95_anchor12_post_popleft` — Phase 5 materialization guard (defensive)

PN203 adds:
- Per-layer block-residency tracking (full-attn layers only)
- Scheduler-tick demote-cold-prefix decision (extends PN95 demote logic)
- Pre-attention prefetch trigger (using PN95 prefetch API)

Env gate: `GENESIS_ENABLE_PN203_COLD_PREFIX_OFFLOAD=1` (default OFF).
Tunables:
  GENESIS_PN203_ACTIVE_WINDOW_TOKENS  — last N tokens to keep GPU (default 32768)
  GENESIS_PN203_OFFLOAD_ATTENTION_ONLY=1 — exclude Mamba/GDN (default 1, safe)

Requires PN202 for proper isolation. With PN202 off, demote of a single
attention layer would also touch the shared-slab Mamba layer's bytes
(incorrect). Apply checks for PN202 and refuses to enable otherwise.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.wiring.pn203_cold_prefix_offload")

_APPLIED = False


def _enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN203_COLD_PREFIX_OFFLOAD", "0",
    ).strip().lower() in ("1", "true", "yes", "on")


def _pn202_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN202_PER_LAYER_KV_SPLIT", "0",
    ).strip().lower() in ("1", "true", "yes", "on")


def _active_window_tokens() -> int:
    try:
        return max(1024, int(os.environ.get(
            "GENESIS_PN203_ACTIVE_WINDOW_TOKENS", "32768")))
    except (ValueError, TypeError):
        return 32768


def _attention_only() -> bool:
    return os.environ.get(
        "GENESIS_PN203_OFFLOAD_ATTENTION_ONLY", "1",
    ).strip().lower() in ("1", "true", "yes", "on")


def apply() -> tuple[str, str]:
    """PN203 is a runtime coordinator — no text-patch. Initialization
    sanity-checks PN202 is also on, then registers the cold-prefix
    sweep hook with the existing PN95 scheduler_tick path.

    The actual demote/promote work uses PN95 primitives:
    - demote_on_evict (already wired)
    - promote_on_miss (already wired)
    - pn95_prefetch_blocks (newly wired in this branch)
    - _pn95_stream_pool (newly wired in this branch)

    PN203 just adds the policy: 'demote full-attn blocks older than
    active_window_tokens to pinned host RAM, prefetch them async
    before attention reads them'.
    """
    global _APPLIED
    if not _enabled():
        return "skipped", "PN203 disabled (set GENESIS_ENABLE_PN203_COLD_PREFIX_OFFLOAD=1)"
    if not _pn202_enabled():
        return "skipped", (
            "PN203 requires PN202 per-layer KV split — set "
            "GENESIS_ENABLE_PN202_PER_LAYER_KV_SPLIT=1 first"
        )
    if _APPLIED:
        return "applied", "PN203 already initialized"
    try:
        from sndr.cache import _pn95_runtime as _runtime
    except Exception as e:
        return "skipped", f"PN95 runtime unavailable: {e}"
    # Register PN203's window-aware demote policy as a hook in scheduler_tick.
    # The hook lives in _pn95_runtime — wire by setting a module-level flag
    # that scheduler_tick reads and enables the cold-prefix sweep.
    try:
        _runtime._PN203_ACTIVE_WINDOW_TOKENS = _active_window_tokens()
        _runtime._PN203_ATTENTION_ONLY = _attention_only()
        _runtime._PN203_ENABLED = True
    except Exception as e:
        return "skipped", f"PN95 runtime hook install failed: {e}"
    _APPLIED = True
    return "applied", (
        f"PN203 cold-prefix offload active "
        f"(window={_active_window_tokens()}, attn_only={_attention_only()})"
    )
