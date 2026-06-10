# SPDX-License-Identifier: Apache-2.0
"""Byte-equivalent verification for P46 buffer registry migration.

P46 is a TEXT WIRING patch — it edits vLLM's gdn_linear_attn.py to
delegate g + beta_output allocation to GdnGatingBufferManager. The
actual torch.empty() call lives in that manager (process-wide pool,
allocate-once-keep-forever, pointer-stable for CUDA-graph capture).

The v11.1.0 P3.3 migration adds PersistentBufferRegistry as the
operator-visible lookup surface — the underlying
GdnGatingBufferManager behavior is UNCHANGED.
"""
from __future__ import annotations

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="torch unavailable")
def test_p46_registry_pool_visible_after_module_import():
    """After p46 module import + ensure_pool_registered(), POOL_GDN_GATING
    is visible in PersistentBufferRegistry."""
    from sndr.runtime.persistent_buffer_registry import (
        PersistentBufferRegistry,
        POOL_GDN_GATING,
    )
    import sndr.engines.vllm.patches.attention.gdn.p46_gdn_gating_buffers as p46

    p46.ensure_pool_registered()

    pools = PersistentBufferRegistry().all_pools()
    assert POOL_GDN_GATING in pools, (
        f"P46 pool not registered; pools={list(pools)}"
    )


def test_p46_module_uses_registry_after_migration():
    """Source imports PersistentBufferRegistry + POOL_GDN_GATING constant."""
    import sndr.engines.vllm.patches.attention.gdn.p46_gdn_gating_buffers as p46
    source = open(p46.__file__).read()
    assert "PersistentBufferRegistry" in source
    assert "POOL_GDN_GATING" in source


# ──────────────────────────────────────────────────────────────────────
# v11.3.0 regression guard — P46 must use PersistentSlicePool
# (matches GdnGatingBufferManager's slice pool backing).
# ──────────────────────────────────────────────────────────────────────


def test_p46_registers_persistent_slice_pool_not_buffer_pool():
    """ensure_pool_registered() must register POOL_GDN_GATING as a
    PersistentSlicePool — P46's allocation is fixed-shape with
    key_dims=3 (all dims keyed)."""
    from sndr.runtime.persistent_buffer_registry import (
        PersistentBufferRegistry,
        PersistentSlicePool,
        POOL_GDN_GATING,
        _reset_registry_for_tests,
    )
    _reset_registry_for_tests()
    import sndr.engines.vllm.patches.attention.gdn.p46_gdn_gating_buffers as p46_int
    p46_int.ensure_pool_registered()
    pool = PersistentBufferRegistry().all_pools()[POOL_GDN_GATING]
    assert isinstance(pool, PersistentSlicePool), (
        f"P46 registered as {type(pool).__name__}, expected "
        f"PersistentSlicePool. v11.3.0 bug fix regression guard."
    )


def test_p46_source_uses_get_slice_pool_not_get_pool():
    """Static check: ensure_pool_registered() uses get_slice_pool()."""
    import sndr.engines.vllm.patches.attention.gdn.p46_gdn_gating_buffers as p46_int
    source = open(p46_int.__file__).read()
    assert "get_slice_pool(POOL_GDN_GATING)" in source, (
        "P46 must call get_slice_pool(POOL_GDN_GATING) — "
        "the v11.3.0 bug fix"
    )
    assert "get_pool(POOL_GDN_GATING)" not in source, (
        "P46 must NOT use get_pool() — wrong pool type. "
        "v11.3.0 bug fix regression guard."
    )


def test_p46_integration_and_storage_class_compose_without_raising():
    """End-to-end: integration ensure_pool_registered() then storage
    class _get_backing_pool() — the path that was broken pre-fix."""
    from sndr.runtime.persistent_buffer_registry import (
        _reset_registry_for_tests, PersistentSlicePool,
    )
    _reset_registry_for_tests()
    import sndr.engines.vllm.patches.attention.gdn.p46_gdn_gating_buffers as p46_int
    p46_int.ensure_pool_registered()
    from sndr.engines.vllm.kernels_legacy.gdn_gating_buffer import (
        GdnGatingBufferManager,
    )
    pool = GdnGatingBufferManager._get_backing_pool()
    assert isinstance(pool, PersistentSlicePool), (
        "storage class got wrong pool type — registration/lookup type "
        "mismatch"
    )
