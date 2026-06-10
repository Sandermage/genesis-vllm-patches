# SPDX-License-Identifier: Apache-2.0
"""TDD for LayoutIntrospect — backend-aware KV cache layout helpers.

Pure-Python tests. No torch / vllm required at module import time —
torch is imported only inside individual tests that need a real tensor.
This matches the family-contract collection rule (torch-less safe).

Strategy: build small fake backend classes mimicking each canonical
``get_kv_cache_shape`` signature (HND pre-#42095, NHD post-#42095, TQ
4-dim packed, Mamba state). Verify ``block_dim_of`` returns the right
index via both the explicit API path and the sentinel-fallback path.
Verify ``classify_layout`` correctly identifies every layout from a
live tensor. Verify ``build_warmup_kv_cache`` honors backend shape.

Provenance: K.1.R.R.3 (2026-05-29). Author: Sander, Odessa.
"""
from __future__ import annotations

import pytest

# Top-level import is library-only (no torch dep).
from sndr.engines.vllm.patches.spec_decode.layout_introspect import (
    Layout,
    block_dim_from_tensor,
    block_dim_of,
    build_warmup_kv_cache,
    classify_layout,
)


# ----------------------- Fake backend fixtures -----------------------


class _FakeHNDBackend:
    """Pre-#42095 layout: (2, num_blocks, block_size, num_kv_heads, head_size)."""

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        return (2, num_blocks, block_size, num_kv_heads, head_size)


class _FakeNHDBackend:
    """Post-#42095 layout: (num_blocks, 2, block_size, num_kv_heads, head_size)."""

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        return (num_blocks, 2, block_size, num_kv_heads, head_size)


class _FakeNHDBackendWithExplicitAPI:
    """Same NHD shape but with explicit block_dim API (post-#42095 native)."""

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @classmethod
    def get_kv_cache_block_dim(
        cls, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        # Explicit answer — must take precedence over shape introspection.
        return 0


class _FakeTQBackend:
    """TurboQuant packed: (num_blocks, block_size, num_kv_heads, slot_size)."""

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        # slot_size_aligned = head_size * 2 for k8v4 (illustrative)
        return (num_blocks, block_size, num_kv_heads, head_size * 2)


class _FakeBrokenBackend:
    """Returns garbage on both API surfaces — exercise the conservative
    fallback."""

    @staticmethod
    def get_kv_cache_shape(*args, **kwargs):
        raise RuntimeError("intentionally broken for fallback test")

    @classmethod
    def get_kv_cache_block_dim(cls, *args, **kwargs):
        raise RuntimeError("intentionally broken for fallback test")


class _FakeNoIntrospectBackend:
    """Has no introspection methods at all — exercise default fallback."""
    pass


# ----------------------- block_dim_of -----------------------


class TestBlockDimOf:
    """``block_dim_of`` must return the correct num_blocks index for
    each canonical backend layout, via explicit API or fallback."""

    def test_hnd_via_shape_inspection(self):
        # HND has num_blocks at index 1 (after the leading 2).
        assert block_dim_of(
            _FakeHNDBackend,
            block_size=16, num_kv_heads=4, head_size=128,
        ) == 1

    def test_nhd_via_shape_inspection(self):
        # NHD has num_blocks at index 0.
        assert block_dim_of(
            _FakeNHDBackend,
            block_size=16, num_kv_heads=4, head_size=128,
        ) == 0

    def test_nhd_explicit_api_overrides_inspection(self):
        # Explicit API should be called even if shape inspection
        # would also produce a correct answer. Verifies preference
        # for the explicit upstream surface.
        assert block_dim_of(
            _FakeNHDBackendWithExplicitAPI,
            block_size=16, num_kv_heads=4, head_size=128,
        ) == 0

    def test_tq_packed_via_shape_inspection(self):
        # TQ 4-dim cache has num_blocks at index 0.
        assert block_dim_of(
            _FakeTQBackend,
            block_size=16, num_kv_heads=4, head_size=128,
            cache_dtype_str="turboquant_k8v4",
        ) == 0

    def test_broken_backend_falls_back_to_conservative_default(self):
        # When both API surfaces raise, return the pre-#42095 default
        # (1 = leading-2 HND layout). Better wrong-but-safe than crash.
        assert block_dim_of(
            _FakeBrokenBackend,
            block_size=16, num_kv_heads=4, head_size=128,
        ) == 1

    def test_no_introspection_falls_back_to_conservative_default(self):
        # Backend without any introspection methods returns the default.
        assert block_dim_of(
            _FakeNoIntrospectBackend,
            block_size=16, num_kv_heads=4, head_size=128,
        ) == 1


# ----------------------- classify_layout -----------------------


class TestClassifyLayout:
    """``classify_layout`` inspects live tensors and assigns the right
    ``Layout`` verdict. Each test imports torch lazily."""

    def test_none_is_unknown(self):
        assert classify_layout(None) == Layout.UNKNOWN

    def test_hnd_tensor(self):
        torch = pytest.importorskip("torch")
        cache = torch.zeros(
            (2, 16, 16, 4, 128), dtype=torch.bfloat16
        )
        assert classify_layout(cache) == Layout.HND

    def test_nhd_tensor(self):
        torch = pytest.importorskip("torch")
        cache = torch.zeros(
            (16, 2, 16, 4, 128), dtype=torch.bfloat16
        )
        assert classify_layout(cache) == Layout.NHD

    def test_tq_packed_4d_tensor(self):
        torch = pytest.importorskip("torch")
        cache = torch.zeros(
            (16, 16, 4, 256), dtype=torch.uint8
        )
        assert classify_layout(cache) == Layout.TQ_PACKED_4D

    def test_tq_packed_4d_rejects_non_uint8(self):
        # 4D + last dim > 2 but wrong dtype is not a TQ slot.
        torch = pytest.importorskip("torch")
        cache = torch.zeros(
            (16, 16, 4, 256), dtype=torch.bfloat16
        )
        # 4D bf16 doesn't match any of our patterns → UNKNOWN.
        assert classify_layout(cache) == Layout.UNKNOWN

    def test_mamba_state_tensor(self):
        torch = pytest.importorskip("torch")
        state = torch.zeros(
            (16, 32, 128), dtype=torch.bfloat16
        )
        assert classify_layout(state) == Layout.MAMBA_STATE

    def test_scalar_is_unknown(self):
        torch = pytest.importorskip("torch")
        scalar = torch.zeros(())
        assert classify_layout(scalar) == Layout.UNKNOWN

    def test_1d_is_unknown(self):
        torch = pytest.importorskip("torch")
        vec = torch.zeros(10)
        assert classify_layout(vec) == Layout.UNKNOWN


# ----------------------- block_dim_from_tensor -----------------------


class TestBlockDimFromTensor:
    """``block_dim_from_tensor`` is a thin wrapper around
    ``classify_layout``; verify each mapping."""

    def test_hnd_returns_1(self):
        torch = pytest.importorskip("torch")
        cache = torch.zeros((2, 16, 16, 4, 128), dtype=torch.bfloat16)
        assert block_dim_from_tensor(cache) == 1

    def test_nhd_returns_0(self):
        torch = pytest.importorskip("torch")
        cache = torch.zeros((16, 2, 16, 4, 128), dtype=torch.bfloat16)
        assert block_dim_from_tensor(cache) == 0

    def test_tq_packed_returns_0(self):
        torch = pytest.importorskip("torch")
        cache = torch.zeros((16, 16, 4, 256), dtype=torch.uint8)
        assert block_dim_from_tensor(cache) == 0

    def test_mamba_state_returns_0(self):
        torch = pytest.importorskip("torch")
        state = torch.zeros((16, 32, 128), dtype=torch.bfloat16)
        assert block_dim_from_tensor(state) == 0

    def test_unknown_returns_none(self):
        assert block_dim_from_tensor(None) is None


# ----------------------- build_warmup_kv_cache -----------------------


class TestBuildWarmupKVCache:
    """``build_warmup_kv_cache`` allocates a tiny zero tensor with the
    backend's declared shape, using sensible dtype defaults."""

    def test_native_backend_returns_bf16(self):
        torch = pytest.importorskip("torch")
        cache = build_warmup_kv_cache(
            _FakeHNDBackend,
            block_size=16, num_kv_heads=4, head_size=128,
            device="cpu",  # CI may not have GPU
        )
        assert tuple(cache.shape) == (2, 2, 16, 4, 128)
        assert cache.dtype == torch.bfloat16
        # All zeros — warmup tensor must not carry stale data.
        assert cache.abs().sum().item() == 0

    def test_tq_backend_returns_uint8(self):
        torch = pytest.importorskip("torch")
        cache = build_warmup_kv_cache(
            _FakeTQBackend,
            num_blocks=2,
            block_size=16, num_kv_heads=4, head_size=128,
            cache_dtype_str="turboquant_k8v4",
            device="cpu",
        )
        # Shape per the fake TQ backend: (num_blocks, block_size,
        # num_kv_heads, head_size * 2)
        assert tuple(cache.shape) == (2, 16, 4, 256)
        assert cache.dtype == torch.uint8

    def test_custom_num_blocks(self):
        pytest.importorskip("torch")  # build_warmup_kv_cache imports torch internally
        # Caller can override num_blocks for warmup.
        cache = build_warmup_kv_cache(
            _FakeHNDBackend,
            num_blocks=8,
            block_size=16, num_kv_heads=4, head_size=128,
            device="cpu",
        )
        assert tuple(cache.shape) == (2, 8, 16, 4, 128)

    def test_no_get_kv_cache_shape_raises(self):
        with pytest.raises(RuntimeError, match="get_kv_cache_shape"):
            build_warmup_kv_cache(
                _FakeNoIntrospectBackend,
                block_size=16, num_kv_heads=4, head_size=128,
                device="cpu",
            )

    def test_explicit_dtype_override(self):
        torch = pytest.importorskip("torch")
        # Caller can override dtype regardless of cache_dtype_str.
        cache = build_warmup_kv_cache(
            _FakeHNDBackend,
            block_size=16, num_kv_heads=4, head_size=128,
            dtype=torch.float16,
            device="cpu",
        )
        assert cache.dtype == torch.float16


# ----------------------- Cross-cutting invariants -----------------------


class TestInvariants:
    """Behavioral invariants the library promises across all
    backends."""

    def test_round_trip_block_dim_matches_tensor_classification(self):
        """The block_dim declared by the backend must match what the
        live tensor would classify as — verifying our shape/inspection
        path agrees with our tensor-classification path.

        Uses ``num_blocks=4`` to avoid the documented (2,2) ambiguity
        in ``classify_layout`` where shape[0]==shape[1]==2 cannot be
        distinguished without a backend handle.
        """
        torch = pytest.importorskip("torch")

        # HND backend → block_dim_of returns 1 → built tensor
        # classifies as HND → block_dim_from_tensor returns 1.
        block_dim_decl = block_dim_of(
            _FakeHNDBackend,
            block_size=16, num_kv_heads=4, head_size=128,
        )
        cache = build_warmup_kv_cache(
            _FakeHNDBackend,
            num_blocks=4,
            block_size=16, num_kv_heads=4, head_size=128,
            device="cpu",
        )
        block_dim_real = block_dim_from_tensor(cache)
        assert block_dim_decl == block_dim_real == 1

        # Same round-trip for NHD.
        block_dim_decl = block_dim_of(
            _FakeNHDBackend,
            block_size=16, num_kv_heads=4, head_size=128,
        )
        cache = build_warmup_kv_cache(
            _FakeNHDBackend,
            num_blocks=4,
            block_size=16, num_kv_heads=4, head_size=128,
            device="cpu",
        )
        block_dim_real = block_dim_from_tensor(cache)
        assert block_dim_decl == block_dim_real == 0

        # And TQ.
        block_dim_decl = block_dim_of(
            _FakeTQBackend,
            block_size=16, num_kv_heads=4, head_size=128,
            cache_dtype_str="turboquant_k8v4",
        )
        cache = build_warmup_kv_cache(
            _FakeTQBackend,
            num_blocks=4,
            block_size=16, num_kv_heads=4, head_size=128,
            cache_dtype_str="turboquant_k8v4",
            device="cpu",
        )
        block_dim_real = block_dim_from_tensor(cache)
        assert block_dim_decl == block_dim_real == 0

    def test_warmup_num_blocks_2_documented_ambiguity(self):
        """When ``num_blocks == 2``, the warmup tensor's shape becomes
        ``(2, 2, ...)`` for NHD and HND alike — pure tensor inspection
        cannot disambiguate. ``classify_layout`` picks HND (pre-#42095
        convention) in this corner case. Callers with backend handles
        should use ``block_dim_of`` instead.
        """
        torch = pytest.importorskip("torch")

        # Build NHD warmup with num_blocks=2 (the library default).
        # block_dim_of returns 0 (unambiguous via sentinel).
        block_dim_decl = block_dim_of(
            _FakeNHDBackend,
            block_size=16, num_kv_heads=4, head_size=128,
        )
        assert block_dim_decl == 0

        # But the tensor (2, 2, 16, 4, 128) looks like HND to the
        # naive classifier. This is the documented ambiguity.
        cache = build_warmup_kv_cache(
            _FakeNHDBackend,
            num_blocks=2,  # the ambiguity-trigger
            block_size=16, num_kv_heads=4, head_size=128,
            device="cpu",
        )
        assert tuple(cache.shape) == (2, 2, 16, 4, 128)
        block_dim_real = block_dim_from_tensor(cache)
        assert block_dim_real == 1, (
            "ambiguity should classify as HND (block_dim==1) — change "
            "this test only if classify_layout's tiebreak rule changes"
        )
