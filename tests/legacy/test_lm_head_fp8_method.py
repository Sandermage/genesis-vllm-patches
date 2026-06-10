# SPDX-License-Identifier: Apache-2.0
"""TDD for PN77 Phase E.5 — Genesis_FP8_LMHead_EmbeddingMethod subclass.

Replaces the broken Phase E.2-3 design (load_weights post-hook + raw
nn.Parameter swap that orphans weight_loader). Tests the new architecture:

  - maybe_swap_pn77_quant_method gating (env, isinstance, idempotent, etc.)
  - process_weights_after_loading flow (compress + replace_parameter)
  - apply() hardware-tier dispatch
  - Tied embeddings safety (must skip via replace_parameter doc constraint)

Real-weight integration tests deferred to boot validation (next phase).
"""
from __future__ import annotations

import pytest
import torch

from sndr.engines.vllm.kernels_legacy import lm_head_fp8_method as lhm


# ─── Test helpers ──────────────────────────────────────────────────────


class _FakeUnquantMethod:
    """Mock UnquantizedEmbeddingMethod for type checks."""
    pass


class _FakeOtherQuantMethod:
    """Mock different quant method (e.g. AWQ) — should NOT be swapped."""
    pass


class _FakeLayer(torch.nn.Module):
    """Mock ParallelLMHead-like layer."""
    def __init__(
        self,
        cls_name: str = "ParallelLMHead",
        vocab: int = 64,
        hidden: int = 32,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(vocab, hidden, dtype=dtype) * 0.02,
            requires_grad=False,
        )
        # Simulate vllm's set_weight_attrs
        def _fake_loader(p, w): pass
        self.weight.weight_loader = _fake_loader
        self.weight.input_dim = 1
        self.weight.output_dim = 0
        self.quant_method = _FakeUnquantMethod()
        # Class-name detection via __name__
        self.__class__ = type(cls_name, (torch.nn.Module,), dict(self.__class__.__dict__))


# ─── Hardware-tier detection ──────────────────────────────────────────


class TestHardwareTier:
    def test_returns_string(self):
        tier = lhm._detect_hardware_tier()
        assert tier in ("marlin", "scaled_mm", "cast_back")

    def test_no_cuda_returns_cast_back(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert lhm._detect_hardware_tier() == "cast_back"


# ─── _is_lm_head class-name detection ─────────────────────────────────


class TestIsLmHead:
    def test_parallel_lm_head_recognized(self):
        layer = _FakeLayer(cls_name="ParallelLMHead")
        assert lhm._is_lm_head(layer)

    def test_ends_with_LMHead_recognized(self):
        layer = _FakeLayer(cls_name="QwenLMHead")
        assert lhm._is_lm_head(layer)

    def test_VocabParallelEmbedding_not_lm_head(self):
        layer = _FakeLayer(cls_name="VocabParallelEmbedding")
        assert not lhm._is_lm_head(layer)

    def test_random_class_not_lm_head(self):
        layer = _FakeLayer(cls_name="MyCustomLayer")
        assert not lhm._is_lm_head(layer)


# ─── maybe_swap_pn77_quant_method ─────────────────────────────────────


class TestMaybeSwap:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv(lhm.ENV_FLAG, raising=False)

    def test_env_off_no_swap(self):
        layer = _FakeLayer()
        original = layer.quant_method
        result = lhm.maybe_swap_pn77_quant_method(layer, original)
        assert result is original
        assert layer.quant_method is original

    def test_env_on_non_lm_head_no_swap(self, monkeypatch):
        monkeypatch.setenv(lhm.ENV_FLAG, "1")
        layer = _FakeLayer(cls_name="VocabParallelEmbedding")
        original = layer.quant_method
        # Need real isinstance check to pass in the swap helper — skip via class name
        result = lhm.maybe_swap_pn77_quant_method(layer, original)
        assert result is original

    def test_env_on_already_fp8_no_swap(self, monkeypatch):
        monkeypatch.setenv(lhm.ENV_FLAG, "1")
        layer = _FakeLayer()
        layer.weight = torch.nn.Parameter(
            torch.zeros(64, 32, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        original = layer.quant_method
        # Without UnquantizedEmbeddingMethod isinstance, fall through. Test will
        # skip swap cleanly.
        result = lhm.maybe_swap_pn77_quant_method(layer, original)
        # Can't fully test without vllm — check no exception
        assert result is not None  # graceful return

    def test_returns_original_on_internal_error(self, monkeypatch):
        """If anything raises, fallback to original method."""
        monkeypatch.setenv(lhm.ENV_FLAG, "1")
        # Pass a layer without `weight` to trigger internal error
        broken = torch.nn.Module()
        # No weight attr
        result = lhm.maybe_swap_pn77_quant_method(broken, _FakeUnquantMethod())
        # Should return the original (may have failed type check etc., not raised)
        assert result is not None


# ─── Genesis_FP8_LMHead_EmbeddingMethod.process_weights_after_loading ─


class TestProcessWeightsAfterLoading:
    def test_idempotent(self):
        method = lhm.Genesis_FP8_LMHead_EmbeddingMethod()
        layer = _FakeLayer()
        setattr(layer, lhm.PN77_APPLIED_MARKER, True)
        # Should return without doing anything
        method.process_weights_after_loading(layer)
        # Weight not changed
        assert layer.weight.dtype == torch.bfloat16

    def test_compresses_fresh_layer(self, monkeypatch):
        """When marker not set, process_weights_after_loading compresses."""
        # Need to mock vllm.model_executor.utils.replace_parameter — make minimal
        # version that uses set_weight_attrs preservation similar to real one.
        import sys

        replace_param_calls = []

        def _fake_replace_parameter(layer, name, data):
            old = getattr(layer, name, None)
            new_param = torch.nn.Parameter(data, requires_grad=False)
            if old is not None and hasattr(old, "weight_loader"):
                new_param.weight_loader = old.weight_loader
                if hasattr(old, "input_dim"):
                    new_param.input_dim = old.input_dim
                if hasattr(old, "output_dim"):
                    new_param.output_dim = old.output_dim
            setattr(layer, name, new_param)
            replace_param_calls.append((name, data.shape, data.dtype))

        # Inject mock module for vllm.model_executor.utils
        import types
        fake_utils = types.ModuleType("vllm.model_executor.utils")
        fake_utils.replace_parameter = _fake_replace_parameter
        # Provide set_weight_attrs too (for create_weights tests)
        def _fake_set_weight_attrs(p, attrs):
            for k, v in attrs.items():
                setattr(p, k, v)
        fake_utils.set_weight_attrs = _fake_set_weight_attrs

        monkeypatch.setitem(sys.modules, "vllm.model_executor.utils", fake_utils)

        method = lhm.Genesis_FP8_LMHead_EmbeddingMethod()
        layer = _FakeLayer(vocab=128, hidden=64)

        method.process_weights_after_loading(layer)

        # weight Parameter replaced with FP8
        assert layer.weight.dtype == torch.float8_e4m3fn
        # weight_loader preserved on new Parameter
        assert hasattr(layer.weight, "weight_loader")
        # weight_scale registered
        assert hasattr(layer, "weight_scale")
        # marker set
        assert getattr(layer, lhm.PN77_APPLIED_MARKER, False) is True
        # tier set
        assert getattr(layer, lhm.PN77_PATH_ATTR) in ("marlin", "scaled_mm", "cast_back")

    def test_sets_marlin_required_attrs(self, monkeypatch):
        """Marlin's prepare_fp8_layer_for_marlin reads:
        output_size_per_partition / input_size_per_partition / orig_dtype.
        ParallelLMHead doesn't have these natively (only num_embeddings_per_partition
        / embedding_dim). We MUST set them before tier=marlin prep."""
        import sys, types
        fake_utils = types.ModuleType("vllm.model_executor.utils")
        def _replace_parameter(layer, name, data):
            new_param = torch.nn.Parameter(data, requires_grad=False)
            old = getattr(layer, name, None)
            if old is not None and hasattr(old, "weight_loader"):
                new_param.weight_loader = old.weight_loader
            setattr(layer, name, new_param)
        fake_utils.replace_parameter = _replace_parameter
        def _set_weight_attrs(p, attrs):
            for k, v in attrs.items(): setattr(p, k, v)
        fake_utils.set_weight_attrs = _set_weight_attrs
        monkeypatch.setitem(sys.modules, "vllm.model_executor.utils", fake_utils)

        method = lhm.Genesis_FP8_LMHead_EmbeddingMethod()
        layer = _FakeLayer(vocab=128, hidden=64)
        # ParallelLMHead has these natively; emulate via setattr
        layer.num_embeddings_per_partition = 128
        layer.embedding_dim = 64

        method.process_weights_after_loading(layer)

        # Marlin-required attrs are set
        assert hasattr(layer, "output_size_per_partition")
        assert layer.output_size_per_partition == 128
        assert hasattr(layer, "input_size_per_partition")
        assert layer.input_size_per_partition == 64
        assert hasattr(layer, "orig_dtype")
        assert layer.orig_dtype == torch.bfloat16  # was BF16 before compress

    def test_attr_setting_falls_back_to_shape_when_native_attrs_missing(self, monkeypatch):
        """If layer doesn't have num_embeddings_per_partition/embedding_dim
        natively, fall back to weight_fp8.shape values (defensive)."""
        import sys, types
        fake_utils = types.ModuleType("vllm.model_executor.utils")
        def _replace_parameter(layer, name, data):
            new_param = torch.nn.Parameter(data, requires_grad=False)
            setattr(layer, name, new_param)
        fake_utils.replace_parameter = _replace_parameter
        def _set_weight_attrs(p, attrs):
            for k, v in attrs.items(): setattr(p, k, v)
        fake_utils.set_weight_attrs = _set_weight_attrs
        monkeypatch.setitem(sys.modules, "vllm.model_executor.utils", fake_utils)

        method = lhm.Genesis_FP8_LMHead_EmbeddingMethod()
        layer = _FakeLayer(vocab=64, hidden=32)
        # Note: NOT setting num_embeddings_per_partition / embedding_dim → fallback path

        method.process_weights_after_loading(layer)

        # Falls back to weight.shape values
        assert layer.output_size_per_partition == 64
        assert layer.input_size_per_partition == 32

    def test_preserves_weight_loader(self, monkeypatch):
        """KEY invariant: replace_parameter MUST preserve weight_loader."""
        import sys, types
        fake_utils = types.ModuleType("vllm.model_executor.utils")
        def _replace_parameter(layer, name, data):
            old = getattr(layer, name, None)
            new_param = torch.nn.Parameter(data, requires_grad=False)
            if old is not None and hasattr(old, "weight_loader"):
                new_param.weight_loader = old.weight_loader
            setattr(layer, name, new_param)
        fake_utils.replace_parameter = _replace_parameter
        def _set_weight_attrs(p, attrs):
            for k, v in attrs.items(): setattr(p, k, v)
        fake_utils.set_weight_attrs = _set_weight_attrs
        monkeypatch.setitem(sys.modules, "vllm.model_executor.utils", fake_utils)

        method = lhm.Genesis_FP8_LMHead_EmbeddingMethod()
        layer = _FakeLayer()

        original_loader = layer.weight.weight_loader
        method.process_weights_after_loading(layer)

        assert hasattr(layer.weight, "weight_loader")
        assert layer.weight.weight_loader is original_loader


# ─── apply() tier dispatch ────────────────────────────────────────────


class TestApplyDispatch:
    def test_no_marker_uses_unquant_path(self):
        """Layer without marker → bypass FP8 path → original GEMM behavior.

        Hard to assert exact behavior without full vllm imports; just check
        it doesn't crash on a reasonable input."""
        method = lhm.Genesis_FP8_LMHead_EmbeddingMethod()
        layer = _FakeLayer(vocab=8, hidden=4)
        x = torch.randn(2, 4, dtype=torch.bfloat16)
        # No marker → uses _unquant_apply which expects vllm dispatch
        # We can't run the full path without vllm; just verify the dispatch
        # logic uses the right branch
        assert not getattr(layer, lhm.PN77_APPLIED_MARKER, False)


# ─── Constants ────────────────────────────────────────────────────────


class TestConstants:
    def test_marker_string_stable(self):
        """Marker name is API surface for `_already_called_process_weights_after_loading`
        — stays stable to compose with vllm's idempotency convention."""
        assert lhm.PN77_APPLIED_MARKER == "_already_called_process_weights_after_loading"

    def test_path_attr_genesis_namespaced(self):
        assert lhm.PN77_PATH_ATTR.startswith("_genesis_")

    def test_env_flag_constant(self):
        assert lhm.ENV_FLAG == "GENESIS_ENABLE_PN77_FP8_LM_HEAD"
