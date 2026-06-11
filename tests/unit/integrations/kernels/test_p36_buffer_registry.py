# SPDX-License-Identifier: Apache-2.0
"""Byte-equivalent verification for P36 buffer registry migration.

P36 is a TEXT WIRING patch — it edits vLLM's attention.py to delegate
shared-decode-buffer allocation to TurboQuantBufferManager. The actual
torch.empty() call lives in
sndr.runtime.prealloc.GenesisPreallocBuffer.get_or_create,
which is process-global, pointer-stable (CUDA-graph safe), and
allocate-once-keep-forever.

The v11.1.0 P3.3 migration adds the registry as the operator-visible
lookup surface — the underlying GenesisPreallocBuffer behavior is
UNCHANGED. Same torch.empty() calls happen at the same time with the
same shapes/dtypes/devices. The registry just exposes the pool name
for `sndr patches show buffer_registry` (future CLI).

Byte-equivalence is preserved because the registry hook does not
allocate new storage — it only registers the pool name in
PersistentBufferRegistry so operators can see it. The original
GenesisPreallocBuffer cache continues to own the tensor storage.
"""
from __future__ import annotations

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="torch unavailable")
def test_p36_registry_pool_visible_after_module_import():
    """After p36 module import, POOL_TQ_DECODE_SHARED is visible in
    PersistentBufferRegistry — operator can see the named pool exists.

    This is purely a lookup-surface check; no byte-level allocation
    semantics changed by the registry hook itself."""
    from sndr.runtime.persistent_buffer_registry import (
        PersistentBufferRegistry,
        POOL_TQ_DECODE_SHARED,
    )
    import sndr.engines.vllm._archive.p36_tq_shared_decode_buffers as p36  # noqa: F401

    # Trigger the registry-visibility hook (no allocation — just registers the name).
    p36.ensure_pool_registered()

    pools = PersistentBufferRegistry().all_pools()
    assert POOL_TQ_DECODE_SHARED in pools, (
        f"P36 pool not registered after import; pools={list(pools)}"
    )


@pytest.mark.skipif(not HAS_TORCH, reason="torch unavailable")
def test_p36_registry_pool_acquire_byte_equivalent():
    """PersistentSlicePool.acquire() returns a tensor with the requested
    shape/dtype/device — byte-equivalent to torch.empty() with the same
    args. Uses key_dims=4 for the full-fixed-shape case (the simplest
    P36 sub-pool semantic)."""
    from sndr.runtime.persistent_buffer_registry import (
        PersistentBufferRegistry,
        POOL_TQ_DECODE_SHARED,
        _reset_registry_for_tests,
    )
    _reset_registry_for_tests()
    # ensure_pool_registered() must run first (test ordering — pytest
    # may not have run the previous test in this same fixture scope)
    import sndr.engines.vllm._archive.p36_tq_shared_decode_buffers as p36
    p36.ensure_pool_registered()
    pool = PersistentBufferRegistry().get_slice_pool(POOL_TQ_DECODE_SHARED)
    shape = (8, 32, 16, 64)
    dtype = torch.float32
    # Full-fixed-shape (no variable dims) → key_dims=4
    t = pool.acquire(shape, dtype, "cpu", key_dims=4)
    assert t.shape == shape
    assert t.dtype == dtype
    assert t.device.type == "cpu"


def test_p36_module_uses_registry_after_migration():
    """The migrated p36 module imports PersistentBufferRegistry + the
    POOL_TQ_DECODE_SHARED constant — operator-visible lookup surface."""
    import sndr.engines.vllm._archive.p36_tq_shared_decode_buffers as p36
    source = open(p36.__file__).read()
    assert "PersistentBufferRegistry" in source, \
        "p36 must use PersistentBufferRegistry post-migration"
    assert "POOL_TQ_DECODE_SHARED" in source, \
        "p36 must reference POOL_TQ_DECODE_SHARED constant"


# ──────────────────────────────────────────────────────────────────────
# v11.3.0 regression guard — P36 must use PersistentSlicePool
# (slice+grow semantics) NOT BufferPool (free-list semantics).
# Pre-fix bug: ensure_pool_registered() called get_pool() which created
# a BufferPool — wrong type for P36's grow-in-place + slice-on-acquire
# allocation pattern (TurboQuant decode + prefill buffers).
# ──────────────────────────────────────────────────────────────────────


def test_p36_registers_persistent_slice_pool_not_buffer_pool():
    """ensure_pool_registered() must register POOL_TQ_DECODE_SHARED as
    a PersistentSlicePool — P36's TurboQuant decode + prefill buffers
    use grow-in-place + slice-on-acquire (NOT free-list acquire/release)."""
    from sndr.runtime.persistent_buffer_registry import (
        PersistentBufferRegistry,
        PersistentSlicePool,
        POOL_TQ_DECODE_SHARED,
        _reset_registry_for_tests,
    )
    _reset_registry_for_tests()
    import sndr.engines.vllm._archive.p36_tq_shared_decode_buffers as p36
    p36.ensure_pool_registered()
    pool = PersistentBufferRegistry().all_pools()[POOL_TQ_DECODE_SHARED]
    assert isinstance(pool, PersistentSlicePool), (
        f"P36 registered as {type(pool).__name__}, expected "
        f"PersistentSlicePool (grow+slice). This is the v11.3.0 bug "
        f"fix regression guard — do not switch to get_pool()."
    )


def test_p36_source_uses_get_slice_pool_not_get_pool():
    """Static check: the ensure_pool_registered() body uses
    `.get_slice_pool(` not `.get_pool(`."""
    import sndr.engines.vllm._archive.p36_tq_shared_decode_buffers as p36
    source = open(p36.__file__).read()
    assert "get_slice_pool(POOL_TQ_DECODE_SHARED)" in source, (
        "P36 must call get_slice_pool(POOL_TQ_DECODE_SHARED) — "
        "the v11.3.0 bug fix"
    )
    assert "get_pool(POOL_TQ_DECODE_SHARED)" not in source, (
        "P36 must NOT use get_pool(POOL_TQ_DECODE_SHARED) "
        "— that's the pre-v11.3.0 bug (wrong pool type)"
    )
