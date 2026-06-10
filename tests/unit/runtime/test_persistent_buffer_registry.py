# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PersistentBufferRegistry + BufferPool — v11.1.0 P3.3."""
from __future__ import annotations

import threading

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def test_persistent_buffer_registry_is_singleton():
    """Registry() is Registry() — same identity across calls."""
    from sndr.runtime.persistent_buffer_registry import (
        PersistentBufferRegistry,
    )
    r1 = PersistentBufferRegistry()
    r2 = PersistentBufferRegistry()
    assert r1 is r2


def test_get_pool_returns_same_instance_for_same_name():
    """Repeated get_pool('x') returns the same BufferPool."""
    from sndr.runtime.persistent_buffer_registry import (
        PersistentBufferRegistry,
    )
    r = PersistentBufferRegistry()
    p1 = r.get_pool("test_pool_same_instance")
    p2 = r.get_pool("test_pool_same_instance")
    assert p1 is p2


def test_get_pool_creates_distinct_pools_for_distinct_names():
    """Different names = different BufferPool instances."""
    from sndr.runtime.persistent_buffer_registry import (
        PersistentBufferRegistry,
    )
    r = PersistentBufferRegistry()
    p1 = r.get_pool("test_pool_a")
    p2 = r.get_pool("test_pool_b")
    assert p1 is not p2
    assert p1.name == "test_pool_a"
    assert p2.name == "test_pool_b"


def test_all_pools_lists_registered():
    """all_pools() returns dict[name -> BufferPool]."""
    from sndr.runtime.persistent_buffer_registry import (
        PersistentBufferRegistry,
    )
    r = PersistentBufferRegistry()
    r.get_pool("pool_listing_test_1")
    r.get_pool("pool_listing_test_2")
    pools = r.all_pools()
    assert "pool_listing_test_1" in pools
    assert "pool_listing_test_2" in pools


@pytest.mark.skipif(not HAS_TORCH, reason="torch unavailable")
def test_buffer_pool_acquire_returns_tensor_with_requested_shape():
    """acquire((4, 8), torch.float32, 'cpu') returns a float32 tensor of shape (4, 8) on cpu."""
    from sndr.runtime.persistent_buffer_registry import (
        PersistentBufferRegistry,
    )
    pool = PersistentBufferRegistry().get_pool("test_acquire_shape")
    t = pool.acquire((4, 8), torch.float32, "cpu")
    assert t.shape == (4, 8)
    assert t.dtype == torch.float32
    assert t.device.type == "cpu"


@pytest.mark.skipif(not HAS_TORCH, reason="torch unavailable")
def test_buffer_pool_reuses_after_release():
    """After release, next acquire with same shape/dtype/device returns the SAME storage."""
    from sndr.runtime.persistent_buffer_registry import (
        PersistentBufferRegistry,
    )
    pool = PersistentBufferRegistry().get_pool("test_reuse")
    t1 = pool.acquire((4, 8), torch.float32, "cpu")
    t1_id = id(t1)
    pool.release(t1)
    t2 = pool.acquire((4, 8), torch.float32, "cpu")
    # Same storage (reused) — data_ptr identical
    assert t2.data_ptr() == t1.data_ptr()


def test_buffer_pool_stats_increment_correctly():
    """stats() reflects acquires + releases."""
    from sndr.runtime.persistent_buffer_registry import (
        PersistentBufferRegistry,
    )
    pool = PersistentBufferRegistry().get_pool("test_stats")
    initial = pool.stats()
    if HAS_TORCH:
        t = pool.acquire((2,), torch.float32, "cpu")
        pool.release(t)
        after = pool.stats()
        assert after["acquires"] == initial["acquires"] + 1
        assert after["releases"] == initial["releases"] + 1


def test_buffer_pool_thread_safe_acquire():
    """Concurrent acquires don't crash; each returns valid tensor."""
    if not HAS_TORCH:
        pytest.skip("torch unavailable")
    from sndr.runtime.persistent_buffer_registry import (
        PersistentBufferRegistry,
    )
    pool = PersistentBufferRegistry().get_pool("test_thread_safety")
    results = []
    errors = []

    def worker():
        try:
            t = pool.acquire((4,), torch.float32, "cpu")
            results.append(t)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent acquire raised: {errors}"
    assert len(results) == 8
    for t in results:
        assert t.shape == (4,)
        assert t.dtype == torch.float32


# ──────────────────────────────────────────────────────────────────────
# PersistentSlicePool tests (v11.2.0+ — grow + slice semantics for
# CUDA-graph-safe allocator migration).
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
def test_slice_pool_first_acquire_allocates():
    """First acquire on empty pool allocates a tensor of exact shape."""
    from sndr.runtime.persistent_buffer_registry import (
        PersistentSlicePool,
    )
    pool = PersistentSlicePool("test_slice_pool_alloc")
    t = pool.acquire((10, 20), torch.float32, "cpu", key_dims=1)
    assert t.shape == (10, 20)
    assert t.dtype == torch.float32
    stats = pool.stats()
    assert stats["allocations"] == 1
    assert stats["grows"] == 0


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
def test_slice_pool_same_shape_returns_same_data_ptr():
    """Re-acquire at same shape returns pointer-stable tensor."""
    from sndr.runtime.persistent_buffer_registry import (
        PersistentSlicePool,
    )
    pool = PersistentSlicePool("test_slice_pool_stable_ptr")
    t1 = pool.acquire((10, 20), torch.float32, "cpu", key_dims=1)
    t2 = pool.acquire((10, 20), torch.float32, "cpu", key_dims=1)
    assert t1.data_ptr() == t2.data_ptr()


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
def test_slice_pool_smaller_acquire_returns_slice():
    """Re-acquire with smaller var-dim returns a slice view of the
    same backing buffer (pointer stable)."""
    from sndr.runtime.persistent_buffer_registry import (
        PersistentSlicePool,
    )
    pool = PersistentSlicePool("test_slice_pool_smaller")
    t_large = pool.acquire((100, 20), torch.float32, "cpu", key_dims=1)
    t_small = pool.acquire((50, 20), torch.float32, "cpu", key_dims=1)
    assert t_small.shape == (50, 20)
    # Slice should share storage with the original
    assert t_small.data_ptr() == t_large.data_ptr()
    stats = pool.stats()
    assert stats["allocations"] == 1, "should not realloc on smaller request"
    assert stats["grows"] == 0


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
def test_slice_pool_larger_acquire_grows_in_place():
    """Larger var-dim triggers grow — pointer changes ONCE, then stable."""
    from sndr.runtime.persistent_buffer_registry import (
        PersistentSlicePool,
    )
    pool = PersistentSlicePool("test_slice_pool_grow")
    t_first = pool.acquire((10, 20), torch.float32, "cpu", key_dims=1)
    ptr_first = t_first.data_ptr()
    t_grown = pool.acquire((100, 20), torch.float32, "cpu", key_dims=1)
    assert t_grown.shape == (100, 20)
    assert t_grown.data_ptr() != ptr_first, "grow should reallocate"
    # Subsequent acquires (same or smaller) should be stable against grown
    t_third = pool.acquire((50, 20), torch.float32, "cpu", key_dims=1)
    assert t_third.data_ptr() == t_grown.data_ptr()
    stats = pool.stats()
    assert stats["grows"] == 1
    assert stats["allocations"] == 1


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
def test_slice_pool_multi_variable_dims():
    """Multi-dim grow (e.g. P39a B+T variable, H+BT fixed) works."""
    from sndr.runtime.persistent_buffer_registry import (
        PersistentSlicePool,
    )
    pool = PersistentSlicePool("test_slice_pool_multi_dim")
    # B=2, T=100, H=16, BT=64
    t1 = pool.acquire((2, 100, 16, 64), torch.float32, "cpu", key_dims=2)
    assert t1.shape == (2, 100, 16, 64)
    # Same H+BT, smaller B+T -> slice
    t2 = pool.acquire((1, 50, 16, 64), torch.float32, "cpu", key_dims=2)
    assert t2.shape == (1, 50, 16, 64)
    assert t2.data_ptr() == t1.data_ptr()
    # Same H+BT, larger T only -> grow
    t3 = pool.acquire((1, 200, 16, 64), torch.float32, "cpu", key_dims=2)
    assert t3.shape == (1, 200, 16, 64)
    assert t3.data_ptr() != t1.data_ptr()
    # Different H+BT -> new pool entry, distinct allocation
    t4 = pool.acquire((2, 100, 32, 128), torch.float32, "cpu", key_dims=2)
    assert t4.data_ptr() != t3.data_ptr()
    assert pool.num_entries() == 2


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
def test_slice_pool_fixed_only_no_grow():
    """key_dims = full shape (e.g. P46 GDN gating) = no variable dims =
    pure fixed-shape pool. acquire returns the pool tensor itself."""
    from sndr.runtime.persistent_buffer_registry import (
        PersistentSlicePool,
    )
    pool = PersistentSlicePool("test_slice_pool_fixed")
    t1 = pool.acquire((1, 8, 16), torch.float16, "cpu", key_dims=3)
    t2 = pool.acquire((1, 8, 16), torch.float16, "cpu", key_dims=3)
    assert t1 is t2  # exact same object
    stats = pool.stats()
    assert stats["allocations"] == 1, "fixed shape should alloc once"
    assert stats["grows"] == 0


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
def test_slice_pool_distinct_dtypes_distinct_pools():
    """Different dtypes → distinct pool entries (no implicit cast)."""
    from sndr.runtime.persistent_buffer_registry import (
        PersistentSlicePool,
    )
    pool = PersistentSlicePool("test_slice_pool_dtype_separation")
    t_fp32 = pool.acquire((10, 20), torch.float32, "cpu", key_dims=1)
    t_fp16 = pool.acquire((10, 20), torch.float16, "cpu", key_dims=1)
    assert t_fp32.data_ptr() != t_fp16.data_ptr()
    assert pool.num_entries() == 2


def test_slice_pool_invalid_key_dims_raises():
    """key_dims out of range raises ValueError."""
    from sndr.runtime.persistent_buffer_registry import (
        PersistentSlicePool,
    )
    pool = PersistentSlicePool("test_slice_pool_bad_key_dims")
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="key_dims"):
        pool.acquire((10, 20), torch.float32, "cpu", key_dims=5)
    with pytest.raises(ValueError, match="key_dims"):
        pool.acquire((10, 20), torch.float32, "cpu", key_dims=-1)


def test_registry_get_slice_pool_returns_singleton():
    """Registry.get_slice_pool returns same instance for same name."""
    from sndr.runtime.persistent_buffer_registry import (
        PersistentBufferRegistry, _reset_registry_for_tests,
    )
    _reset_registry_for_tests()
    r = PersistentBufferRegistry()
    p1 = r.get_slice_pool("test_registry_slice_a")
    p2 = r.get_slice_pool("test_registry_slice_a")
    assert p1 is p2


def test_registry_cannot_mix_pool_types_for_same_name():
    """Same name used for BufferPool then PersistentSlicePool → ValueError."""
    from sndr.runtime.persistent_buffer_registry import (
        PersistentBufferRegistry, _reset_registry_for_tests,
    )
    _reset_registry_for_tests()
    r = PersistentBufferRegistry()
    r.get_pool("test_mixed_type")
    with pytest.raises(ValueError, match="registered as BufferPool"):
        r.get_slice_pool("test_mixed_type")

    _reset_registry_for_tests()
    r = PersistentBufferRegistry()
    r.get_slice_pool("test_mixed_type2")
    with pytest.raises(ValueError, match="registered as PersistentSlicePool"):
        r.get_pool("test_mixed_type2")


def test_registry_summary_includes_both_pool_types():
    """summary() reports type=BufferPool vs PersistentSlicePool."""
    from sndr.runtime.persistent_buffer_registry import (
        PersistentBufferRegistry, _reset_registry_for_tests,
    )
    _reset_registry_for_tests()
    r = PersistentBufferRegistry()
    r.get_pool("summary_test_buffer_pool")
    r.get_slice_pool("summary_test_slice_pool")
    s = r.summary()
    assert s["pool_count"] == 2
    assert s["pools"]["summary_test_buffer_pool"]["type"] == "BufferPool"
    assert (
        s["pools"]["summary_test_slice_pool"]["type"]
        == "PersistentSlicePool"
    )
