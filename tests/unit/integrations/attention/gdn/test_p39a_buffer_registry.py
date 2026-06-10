# SPDX-License-Identifier: Apache-2.0
"""Byte-equivalent verification for P39a buffer registry migration.

P39a is a MONKEY-PATCH wiring patch — it rebinds
chunk_scaled_dot_kkt_fwd to a pooled drop-in that calls
FlaKktBufferManager.acquire(). The actual torch.empty() call lives in
that manager (allocate-once-keep-forever, pointer-stable, CUDA-graph
safe via the reserve-before-cudagraph pattern).

The v11.1.0 P3.3 migration adds PersistentBufferRegistry as the
operator-visible lookup surface — the underlying FlaKktBufferManager
behavior is UNCHANGED. Same external API; same allocation path; same
torch.empty() call paths.
"""
from __future__ import annotations

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="torch unavailable")
def test_p39a_registry_pool_visible_after_module_import():
    """After p39a module import + ensure_pool_registered(),
    POOL_FLA_KKT_PERSISTENT_A is visible in PersistentBufferRegistry."""
    from sndr.runtime.persistent_buffer_registry import (
        PersistentBufferRegistry,
        POOL_FLA_KKT_PERSISTENT_A,
    )
    import sndr.engines.vllm.patches.attention.gdn.p39a_fla_kkt_buffer as p39a

    p39a.ensure_pool_registered()

    pools = PersistentBufferRegistry().all_pools()
    assert POOL_FLA_KKT_PERSISTENT_A in pools, (
        f"P39a pool not registered; pools={list(pools)}"
    )


def test_p39a_module_uses_registry_after_migration():
    """Source imports PersistentBufferRegistry + POOL_FLA_KKT_PERSISTENT_A
    constant."""
    import sndr.engines.vllm.patches.attention.gdn.p39a_fla_kkt_buffer as p39a
    source = open(p39a.__file__).read()
    assert "PersistentBufferRegistry" in source
    assert "POOL_FLA_KKT_PERSISTENT_A" in source


# ──────────────────────────────────────────────────────────────────────
# v11.3.0 regression guard — P39a must use PersistentSlicePool
# (slice+grow semantics) NOT BufferPool (free-list semantics).
# Pre-fix bug: ensure_pool_registered() called get_pool() which created
# a BufferPool — wrong type for P39a's grow-in-place + slice-on-acquire
# allocation pattern. Fixed in v11.3.0 by switching to get_slice_pool().
# ──────────────────────────────────────────────────────────────────────


def test_p39a_registers_persistent_slice_pool_not_buffer_pool():
    """ensure_pool_registered() must register POOL_FLA_KKT_PERSISTENT_A
    as a PersistentSlicePool — P39a's allocation pattern is grow + slice,
    not acquire/release free-list."""
    from sndr.runtime.persistent_buffer_registry import (
        PersistentBufferRegistry,
        PersistentSlicePool,
        POOL_FLA_KKT_PERSISTENT_A,
        _reset_registry_for_tests,
    )
    _reset_registry_for_tests()
    import sndr.engines.vllm.patches.attention.gdn.p39a_fla_kkt_buffer as p39a
    p39a.ensure_pool_registered()
    pool = PersistentBufferRegistry().all_pools()[POOL_FLA_KKT_PERSISTENT_A]
    assert isinstance(pool, PersistentSlicePool), (
        f"P39a registered as {type(pool).__name__}, expected "
        f"PersistentSlicePool (grow+slice). This is the v11.3.0 bug fix "
        f"regression guard — do not switch to get_pool()."
    )


def test_p39a_source_uses_get_slice_pool_not_get_pool():
    """Static check: the ensure_pool_registered() body uses
    `.get_slice_pool(` not `.get_pool(`."""
    import sndr.engines.vllm.patches.attention.gdn.p39a_fla_kkt_buffer as p39a
    source = open(p39a.__file__).read()
    # Extract the ensure_pool_registered function body
    assert "get_slice_pool(POOL_FLA_KKT_PERSISTENT_A)" in source, (
        "P39a must call get_slice_pool(POOL_FLA_KKT_PERSISTENT_A) — "
        "the v11.3.0 bug fix"
    )
    # Verify the buggy call pattern doesn't accidentally come back
    assert "get_pool(POOL_FLA_KKT_PERSISTENT_A)" not in source, (
        "P39a must NOT use get_pool(POOL_FLA_KKT_PERSISTENT_A) "
        "— that's the pre-v11.3.0 bug (wrong pool type)"
    )
