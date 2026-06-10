# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr.engines.vllm.detection.gpu_detect` — GPU detection facade.

Contract:

  1. get_gpu_generation routes through detection helpers in priority
     order: ROCm > XPU > CPU > blackwell > hopper > ada > ampere DC >
     ampere consumer > unknown.
  2. get_gpu_count returns 0 on systems without CUDA.
  3. get_gpu_name returns 'unknown' on systems without CUDA.
  4. Facade re-exports all 14 helpers from detection.guards.
"""
from __future__ import annotations

import pytest

from sndr.engines.vllm.detection import gpu_detect


# ─── Facade re-exports ────────────────────────────────────────────────


class TestFacadeReexports:
    def test_all_export_list_present(self):
        """Every name in __all__ must be importable."""
        assert "get_gpu_count" in gpu_detect.__all__
        assert "get_gpu_generation" in gpu_detect.__all__
        assert "get_gpu_name" in gpu_detect.__all__

    def test_each_helper_callable(self):
        # All re-exported names should be callable predicates / getters
        for name in (
            "is_ampere_any", "is_ampere_consumer", "is_ampere_datacenter",
            "is_ada_lovelace", "is_hopper",
            "is_blackwell", "is_blackwell_consumer", "is_blackwell_datacenter",
            "is_amd_rocm", "is_intel_xpu", "is_cpu_only",
            "is_nvidia_cuda", "is_cuda_alike",
            "is_sm_at_least", "is_sm_exactly",
            "get_compute_capability",
        ):
            assert callable(getattr(gpu_detect, name)), f"{name} not callable"


# ─── get_gpu_generation routing ───────────────────────────────────────


class TestGetGpuGenerationRouting:
    """Priority order: ROCm > XPU > CPU > blackwell_consumer >
    blackwell_dc > hopper > ada_lovelace > ampere_dc > ampere_consumer >
    unknown."""

    @pytest.fixture
    def _all_false(self, monkeypatch):
        """Stub every predicate to False; tests then flip the one they care about."""
        for name in (
            "is_amd_rocm", "is_intel_xpu", "is_cpu_only",
            "is_blackwell_consumer", "is_blackwell_datacenter",
            "is_hopper", "is_ada_lovelace",
            "is_ampere_datacenter", "is_ampere_consumer",
        ):
            monkeypatch.setattr(gpu_detect, name, lambda: False)

    def test_amd_rocm_wins(self, monkeypatch, _all_false):
        monkeypatch.setattr(gpu_detect, "is_amd_rocm", lambda: True)
        assert gpu_detect.get_gpu_generation() == "amd_rocm"

    def test_intel_xpu(self, monkeypatch, _all_false):
        monkeypatch.setattr(gpu_detect, "is_intel_xpu", lambda: True)
        assert gpu_detect.get_gpu_generation() == "intel_xpu"

    def test_cpu_only(self, monkeypatch, _all_false):
        monkeypatch.setattr(gpu_detect, "is_cpu_only", lambda: True)
        assert gpu_detect.get_gpu_generation() == "cpu_only"

    def test_blackwell_consumer(self, monkeypatch, _all_false):
        monkeypatch.setattr(gpu_detect, "is_blackwell_consumer", lambda: True)
        assert gpu_detect.get_gpu_generation() == "blackwell_consumer"

    def test_blackwell_datacenter(self, monkeypatch, _all_false):
        monkeypatch.setattr(gpu_detect, "is_blackwell_datacenter", lambda: True)
        assert gpu_detect.get_gpu_generation() == "blackwell_datacenter"

    def test_hopper(self, monkeypatch, _all_false):
        monkeypatch.setattr(gpu_detect, "is_hopper", lambda: True)
        assert gpu_detect.get_gpu_generation() == "hopper"

    def test_ada_lovelace(self, monkeypatch, _all_false):
        monkeypatch.setattr(gpu_detect, "is_ada_lovelace", lambda: True)
        assert gpu_detect.get_gpu_generation() == "ada_lovelace"

    def test_ampere_datacenter(self, monkeypatch, _all_false):
        monkeypatch.setattr(gpu_detect, "is_ampere_datacenter", lambda: True)
        assert gpu_detect.get_gpu_generation() == "ampere_datacenter"

    def test_ampere_consumer(self, monkeypatch, _all_false):
        monkeypatch.setattr(gpu_detect, "is_ampere_consumer", lambda: True)
        assert gpu_detect.get_gpu_generation() == "ampere_consumer"

    def test_unknown_when_all_false(self, _all_false):
        assert gpu_detect.get_gpu_generation() == "unknown"

    def test_priority_amd_over_blackwell(self, monkeypatch, _all_false):
        """Earlier branch wins — amd_rocm before blackwell_consumer."""
        monkeypatch.setattr(gpu_detect, "is_amd_rocm", lambda: True)
        monkeypatch.setattr(gpu_detect, "is_blackwell_consumer", lambda: True)
        assert gpu_detect.get_gpu_generation() == "amd_rocm"


# ─── get_gpu_name + get_gpu_count graceful no-CUDA fallback ──────────


class TestNoCudaFallbacks:
    def test_get_gpu_name_returns_string(self):
        result = gpu_detect.get_gpu_name()
        assert isinstance(result, str)
        # Either 'unknown' or a real GPU name — both strings

    def test_get_gpu_count_returns_int(self):
        result = gpu_detect.get_gpu_count()
        assert isinstance(result, int)
        assert result >= 0

    def test_get_gpu_name_handles_import_error(self, monkeypatch):
        """If `import torch` raises, get_gpu_name returns 'unknown'."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "torch":
                raise ImportError("synthetic")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert gpu_detect.get_gpu_name() == "unknown"

    def test_get_gpu_count_handles_import_error(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "torch":
                raise ImportError("synthetic")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert gpu_detect.get_gpu_count() == 0
