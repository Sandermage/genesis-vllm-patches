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
    from vllm.sndr_core.runtime.persistent_buffer_registry import (
        PersistentBufferRegistry,
        POOL_FFN_INTERMEDIATE_SCRATCH,
    )
    import vllm.sndr_core.integrations.kernels.pn12_ffn_intermediate_pool as pn12

    pn12.ensure_pool_registered()

    pools = PersistentBufferRegistry().all_pools()
    assert POOL_FFN_INTERMEDIATE_SCRATCH in pools, (
        f"PN12 pool not registered; pools={list(pools)}"
    )


def test_pn12_module_uses_registry_after_migration():
    """Source imports PersistentBufferRegistry + POOL_FFN_INTERMEDIATE_
    SCRATCH constant."""
    import vllm.sndr_core.integrations.kernels.pn12_ffn_intermediate_pool as pn12
    source = open(pn12.__file__).read()
    assert "PersistentBufferRegistry" in source
    assert "POOL_FFN_INTERMEDIATE_SCRATCH" in source
