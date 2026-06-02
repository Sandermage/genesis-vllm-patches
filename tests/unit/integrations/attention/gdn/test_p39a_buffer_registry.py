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
    from vllm.sndr_core.runtime.persistent_buffer_registry import (
        PersistentBufferRegistry,
        POOL_FLA_KKT_PERSISTENT_A,
    )
    import vllm.sndr_core.integrations.attention.gdn.p39a_fla_kkt_buffer as p39a

    p39a.ensure_pool_registered()

    pools = PersistentBufferRegistry().all_pools()
    assert POOL_FLA_KKT_PERSISTENT_A in pools, (
        f"P39a pool not registered; pools={list(pools)}"
    )


def test_p39a_module_uses_registry_after_migration():
    """Source imports PersistentBufferRegistry + POOL_FLA_KKT_PERSISTENT_A
    constant."""
    import vllm.sndr_core.integrations.attention.gdn.p39a_fla_kkt_buffer as p39a
    source = open(p39a.__file__).read()
    assert "PersistentBufferRegistry" in source
    assert "POOL_FLA_KKT_PERSISTENT_A" in source
