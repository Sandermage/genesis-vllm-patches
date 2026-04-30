# SPDX-License-Identifier: Apache-2.0
"""pytest configuration for Genesis tests.

Fixtures and helpers shared across all test modules.
"""
from __future__ import annotations

import pytest
import torch


# ═══════════════════════════════════════════════════════════════════════════
#                          PLATFORM DETECTION FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def cuda_available() -> bool:
    """True if CUDA is available for testing."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def rocm_available() -> bool:
    """True if running on ROCm (PyTorch built for HIP)."""
    if not torch.cuda.is_available():
        return False
    try:
        return torch.version.hip is not None
    except AttributeError:
        return False


@pytest.fixture(scope="session")
def nvidia_cuda_available() -> bool:
    """True if NVIDIA CUDA specifically (NOT ROCm)."""
    if not torch.cuda.is_available():
        return False
    try:
        # ROCm's torch.version.hip is a string; NVIDIA's is None
        return torch.version.hip is None
    except AttributeError:
        # Old PyTorch without torch.version.hip = NVIDIA-only build
        return torch.cuda.is_available()


# ═══════════════════════════════════════════════════════════════════════════
#                       PYTEST MARKERS
# ═══════════════════════════════════════════════════════════════════════════

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "cuda_required: test requires NVIDIA CUDA device",
    )
    config.addinivalue_line(
        "markers",
        "rocm_required: test requires AMD ROCm device",
    )
    config.addinivalue_line(
        "markers",
        "gpu_required: test requires any GPU (CUDA or ROCm)",
    )
    config.addinivalue_line(
        "markers",
        "slow: test takes >5 seconds",
    )


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests automatically on CPU-only hosts."""
    cuda = torch.cuda.is_available()
    for item in items:
        if "cuda_required" in item.keywords and not cuda:
            item.add_marker(pytest.mark.skip(reason="CUDA not available"))
        if "gpu_required" in item.keywords and not cuda:
            item.add_marker(pytest.mark.skip(reason="GPU not available"))


# ═══════════════════════════════════════════════════════════════════════════
#                         FIXTURE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _reset_genesis_prealloc_state() -> None:
    """Internal helper: drop all module-cached state used by Genesis preallocs."""
    from vllm._genesis.prealloc import GenesisPreallocBuffer
    GenesisPreallocBuffer.clear_for_tests()
    try:
        from vllm._genesis.kernels.dequant_buffer import TurboQuantBufferManager
        TurboQuantBufferManager.clear_for_tests()
    except Exception:
        pass
    try:
        from vllm._genesis.kernels.gdn_core_attn_manager import GdnCoreAttnManager
        GdnCoreAttnManager.clear_for_tests()
    except Exception:
        pass
    # The central token-budget resolver caches its decision at module
    # scope. Tests that probe the default-fallback path need a fresh
    # cache, otherwise they see whatever an earlier test resolved.
    try:
        from vllm._genesis import prealloc_budget as _pb
        _pb._CACHED = None
    except Exception:
        pass


@pytest.fixture
def reset_genesis_prealloc():
    """Clear ALL Genesis buffer registries before/after each test.

    Covers:
      - `GenesisPreallocBuffer._REGISTRY` (universal framework)
      - `TurboQuantBufferManager._K_BUFFERS / _V_BUFFERS / _CU_* /
         _SYNTH_* / _PREFILL_OUT_BUFFERS / _DECODE_*` (P22/P26/P32/P33/P36)
      - `GdnCoreAttnManager._BUFFERS` + `_SHOULD_APPLY_CACHED` (P28)
      - `prealloc_budget._CACHED` (P73 token budget resolver)

    Test isolation is critical since these are class-level state on
    module-scoped singletons. If one test allocates and another asserts
    the registry is empty, a stale entry leaks and the assertion fails.

    Usage:
        def test_something(reset_genesis_prealloc):
            # all Genesis registries are clean
            ...
            # and cleaned again after test
    """
    _reset_genesis_prealloc_state()
    yield
    _reset_genesis_prealloc_state()


@pytest.fixture(autouse=True)
def _autoreset_token_budget_cache():
    """Always-on hygiene: drop the central P73 _CACHED before AND after
    every test in this directory. The fixture is cheap (one attribute
    write) and prevents cross-test pollution from any test that touches
    `prealloc_budget.resolve_token_budget()` directly or indirectly."""
    try:
        from vllm._genesis import prealloc_budget as _pb
        _pb._CACHED = None
    except Exception:
        pass
    yield
    try:
        from vllm._genesis import prealloc_budget as _pb
        _pb._CACHED = None
    except Exception:
        pass


@pytest.fixture
def deterministic_seed():
    """Set deterministic torch seed for reproducible tests."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    yield 42
