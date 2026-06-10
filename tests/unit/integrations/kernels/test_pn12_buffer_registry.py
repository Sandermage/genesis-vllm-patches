# SPDX-License-Identifier: Apache-2.0
"""Byte-equivalent verification for PN12 buffer registry migration.

PN12 is a TEXT WIRING patch — it edits vLLM's silu_and_mul forward to
delegate output-buffer allocation to FFNIntermediateCache. The actual
torch.empty() call happens inside that cache (process-wide pool,
allocate-once-keep-forever).

The v11.1.0 P3.3 migration adds PersistentBufferRegistry as the
operator-visible lookup surface — the underlying FFNIntermediateCache
behavior is UNCHANGED. Same torch.empty() calls at the same time with
the same shapes/dtypes/devices.
"""
from __future__ import annotations

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="torch unavailable")
def test_pn12_registry_pool_visible_after_module_import():
    """After pn12 module import + ensure_pool_registered(), POOL_FFN_
    INTERMEDIATE_SCRATCH is visible in PersistentBufferRegistry."""
    from sndr.runtime.persistent_buffer_registry import (
        PersistentBufferRegistry,
        POOL_FFN_INTERMEDIATE_SCRATCH,
    )
    import sndr.engines.vllm.patches.kernels.pn12_ffn_intermediate_pool as pn12

    pn12.ensure_pool_registered()

    pools = PersistentBufferRegistry().all_pools()
    assert POOL_FFN_INTERMEDIATE_SCRATCH in pools, (
        f"PN12 pool not registered; pools={list(pools)}"
    )


def test_pn12_module_uses_registry_after_migration():
    """Source imports PersistentBufferRegistry + POOL_FFN_INTERMEDIATE_
    SCRATCH constant."""
    import sndr.engines.vllm.patches.kernels.pn12_ffn_intermediate_pool as pn12
    source = open(pn12.__file__).read()
    assert "PersistentBufferRegistry" in source
    assert "POOL_FFN_INTERMEDIATE_SCRATCH" in source


# ──────────────────────────────────────────────────────────────────────
# v11.3.0 regression guard — PN12 must use PersistentSlicePool
# (slice+grow semantics) NOT BufferPool (free-list semantics).
#
# Pre-fix production bug: when integration module's
# ensure_pool_registered() ran first (e.g., at module import), it
# created a BufferPool. Then when FFNIntermediateCache.acquire_silu_out
# called _get_backing_pool() (via get_slice_pool), the registry raised
# ValueError "pool was registered as BufferPool, not
# PersistentSlicePool" — causing the FFN cache to never engage and
# silently falling back to allocate-per-step behaviour.
# ──────────────────────────────────────────────────────────────────────


def test_pn12_registers_persistent_slice_pool_not_buffer_pool():
    """ensure_pool_registered() must register POOL_FFN_INTERMEDIATE_SCRATCH
    as a PersistentSlicePool — PN12's allocation pattern is grow+slice
    with key_dims=1 (variable rows, fixed intermediate_size)."""
    from sndr.runtime.persistent_buffer_registry import (
        PersistentBufferRegistry,
        PersistentSlicePool,
        POOL_FFN_INTERMEDIATE_SCRATCH,
        _reset_registry_for_tests,
    )
    _reset_registry_for_tests()
    import sndr.engines.vllm.patches.kernels.pn12_ffn_intermediate_pool as pn12_int
    pn12_int.ensure_pool_registered()
    pool = PersistentBufferRegistry().all_pools()[POOL_FFN_INTERMEDIATE_SCRATCH]
    assert isinstance(pool, PersistentSlicePool), (
        f"PN12 registered as {type(pool).__name__}, expected "
        f"PersistentSlicePool. v11.3.0 bug fix regression guard."
    )


def test_pn12_source_uses_get_slice_pool_not_get_pool():
    """Static check: ensure_pool_registered() uses get_slice_pool()."""
    import sndr.engines.vllm.patches.kernels.pn12_ffn_intermediate_pool as pn12_int
    source = open(pn12_int.__file__).read()
    assert "get_slice_pool(POOL_FFN_INTERMEDIATE_SCRATCH)" in source, (
        "PN12 must call get_slice_pool(POOL_FFN_INTERMEDIATE_SCRATCH) — "
        "the v11.3.0 bug fix"
    )
    assert "get_pool(POOL_FFN_INTERMEDIATE_SCRATCH)" not in source, (
        "PN12 must NOT use get_pool() — wrong pool type for slice+grow "
        "allocation pattern. v11.3.0 bug fix regression guard."
    )


def test_pn12_integration_and_storage_class_compose_without_raising():
    """End-to-end: integration ensure_pool_registered() then storage
    class _get_backing_pool() — the path that was broken pre-fix.

    Without the fix, the second call raised ValueError because the pool
    was registered as the wrong type."""
    from sndr.runtime.persistent_buffer_registry import (
        _reset_registry_for_tests, PersistentSlicePool,
    )
    _reset_registry_for_tests()
    # Step 1: integration registers pool
    import sndr.engines.vllm.patches.kernels.pn12_ffn_intermediate_pool as pn12_int
    pn12_int.ensure_pool_registered()
    # Step 2: storage class looks it up — must not raise
    from sndr.engines.vllm.kernels_legacy.ffn_intermediate_cache import (
        FFNIntermediateCache,
    )
    pool = FFNIntermediateCache._get_backing_pool()
    assert isinstance(pool, PersistentSlicePool), (
        "storage class got wrong pool type — registration/lookup type "
        "mismatch"
    )
