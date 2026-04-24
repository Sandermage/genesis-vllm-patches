# SPDX-License-Identifier: Apache-2.0
"""Tests for Genesis v7.9 model-architecture dispatch detection (P52 + P53).

Covers:
  - MoE detection: Qwen3-MoE style (num_experts), Mixtral (num_local_experts),
    DeepSeek (n_routed_experts), model_type suffix, architecture name
  - Hybrid detection: Qwen3-Next layer_types, Mamba model_type, architecture
  - TurboQuant detection: kv_cache_dtype string prefix
  - Conservative fallback: no config → returns True for all
  - Caching: second call returns cached profile, `clear_for_tests` resets
  - Robustness: malformed hf_config doesn't crash
"""
from __future__ import annotations

import pytest

from vllm._genesis import model_detect


class FakeHFConfig:
    """Stand-in for transformers PretrainedConfig."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class FakeCacheConfig:
    def __init__(self, kv_cache_dtype=None):
        self.kv_cache_dtype = kv_cache_dtype


class FakeModelConfig:
    def __init__(self, hf_config):
        self.hf_config = hf_config


class FakeVllmConfig:
    def __init__(self, hf_config, kv_cache_dtype=None):
        self.model_config = FakeModelConfig(hf_config)
        self.cache_config = FakeCacheConfig(kv_cache_dtype)


@pytest.fixture(autouse=True)
def _reset_cache():
    model_detect.clear_for_tests()
    yield
    model_detect.clear_for_tests()


# ════════════════════════════════════════════════════════════════════════
#                            MoE DETECTION
# ════════════════════════════════════════════════════════════════════════

def test_moe_qwen3moe_num_experts(monkeypatch):
    cfg = FakeVllmConfig(
        FakeHFConfig(
            num_experts=128, model_type="qwen3_moe",
            architectures=["Qwen3MoeForCausalLM"],
        ),
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    model_detect.clear_for_tests()
    profile = model_detect.get_model_profile()
    assert profile["resolved"] is True
    assert profile["moe"] is True
    assert profile["hybrid"] is False
    assert "num_experts" in profile["moe_details"]


def test_moe_mixtral_num_local_experts(monkeypatch):
    cfg = FakeVllmConfig(
        FakeHFConfig(
            num_local_experts=8, num_experts_per_tok=2,
            model_type="mixtral", architectures=["MixtralForCausalLM"],
        ),
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    model_detect.clear_for_tests()
    assert model_detect.is_moe_model() is True


def test_moe_deepseek_n_routed_experts(monkeypatch):
    cfg = FakeVllmConfig(
        FakeHFConfig(
            n_routed_experts=64, model_type="deepseek_v2",
            architectures=["DeepseekV2ForCausalLM"],
        ),
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    model_detect.clear_for_tests()
    assert model_detect.is_moe_model() is True


def test_moe_detected_via_model_type_suffix(monkeypatch):
    # No num_experts attr but model_type contains 'moe'
    cfg = FakeVllmConfig(
        FakeHFConfig(model_type="custom_moe", architectures=["CustomForCausalLM"]),
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    model_detect.clear_for_tests()
    profile = model_detect.get_model_profile()
    assert profile["moe"] is True
    assert profile["moe_details"].get("moe_source") == "model_type_name"


def test_moe_detected_via_architecture_name(monkeypatch):
    cfg = FakeVllmConfig(
        FakeHFConfig(
            model_type="custom", architectures=["CustomMoEForCausalLM"],
        ),
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    model_detect.clear_for_tests()
    assert model_detect.is_moe_model() is True


def test_moe_false_on_dense_qwen3(monkeypatch):
    cfg = FakeVllmConfig(
        FakeHFConfig(
            model_type="qwen3", architectures=["Qwen3ForCausalLM"],
            hidden_size=4096, num_hidden_layers=32,
        ),
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    model_detect.clear_for_tests()
    profile = model_detect.get_model_profile()
    assert profile["moe"] is False
    assert profile["hybrid"] is False


def test_moe_false_when_num_experts_is_one(monkeypatch):
    # Edge case: some configs have num_experts=1 which is NOT MoE.
    cfg = FakeVllmConfig(
        FakeHFConfig(num_experts=1, model_type="some_arch", architectures=[]),
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    model_detect.clear_for_tests()
    assert model_detect.is_moe_model() is False


# ════════════════════════════════════════════════════════════════════════
#                          HYBRID DETECTION
# ════════════════════════════════════════════════════════════════════════

def test_hybrid_qwen3_next_layer_types(monkeypatch):
    # Qwen3-Next alternates linear_attention + full_attention
    layer_types = (
        ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3
    )
    cfg = FakeVllmConfig(
        FakeHFConfig(
            layer_types=layer_types, model_type="qwen3_next",
            architectures=["Qwen3NextForCausalLM"],
        ),
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    model_detect.clear_for_tests()
    profile = model_detect.get_model_profile()
    assert profile["hybrid"] is True
    assert profile["hybrid_details"].get("hybrid_source") == "layer_types"


def test_hybrid_mamba_model_type(monkeypatch):
    cfg = FakeVllmConfig(
        FakeHFConfig(
            model_type="falcon_mamba", architectures=["FalconMambaForCausalLM"],
        ),
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    model_detect.clear_for_tests()
    assert model_detect.is_hybrid_model() is True


def test_hybrid_false_pure_attention(monkeypatch):
    cfg = FakeVllmConfig(
        FakeHFConfig(
            model_type="qwen3", architectures=["Qwen3ForCausalLM"],
        ),
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    model_detect.clear_for_tests()
    assert model_detect.is_hybrid_model() is False


def test_hybrid_moe_combined_qwen3_next_a3b(monkeypatch):
    # Qwen3.6-35B-A3B has BOTH MoE + hybrid linear attention.
    layer_types = ["linear_attention", "full_attention"] * 24
    cfg = FakeVllmConfig(
        FakeHFConfig(
            num_experts=128, layer_types=layer_types,
            model_type="qwen3_next_moe",
            architectures=["Qwen3NextMoeForCausalLM"],
        ),
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    model_detect.clear_for_tests()
    profile = model_detect.get_model_profile()
    assert profile["moe"] is True
    assert profile["hybrid"] is True


# ════════════════════════════════════════════════════════════════════════
#                         TURBOQUANT DETECTION
# ════════════════════════════════════════════════════════════════════════

def test_turboquant_k8v4_detected(monkeypatch):
    cfg = FakeVllmConfig(
        FakeHFConfig(model_type="qwen3", architectures=[]),
        kv_cache_dtype="turboquant_k8v4",
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    model_detect.clear_for_tests()
    profile = model_detect.get_model_profile()
    assert profile["turboquant"] is True
    assert profile["kv_cache_dtype"] == "turboquant_k8v4"


def test_turboquant_fp8_not_detected(monkeypatch):
    cfg = FakeVllmConfig(
        FakeHFConfig(model_type="qwen3", architectures=[]),
        kv_cache_dtype="fp8",
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    model_detect.clear_for_tests()
    assert model_detect.is_turboquant_active() is False


def test_turboquant_auto_not_detected(monkeypatch):
    cfg = FakeVllmConfig(
        FakeHFConfig(model_type="qwen3", architectures=[]),
        kv_cache_dtype="auto",
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    model_detect.clear_for_tests()
    assert model_detect.is_turboquant_active() is False


# ════════════════════════════════════════════════════════════════════════
#                       CONSERVATIVE FALLBACK
# ════════════════════════════════════════════════════════════════════════

def test_conservative_true_on_no_config(monkeypatch):
    # get_current_vllm_config raises — we return True for everything.
    def _raise():
        raise RuntimeError("no config context")
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", _raise, raising=False,
    )
    model_detect.clear_for_tests()
    profile = model_detect.get_model_profile()
    assert profile["resolved"] is False
    assert profile["moe"] is True
    assert profile["hybrid"] is True
    assert profile["turboquant"] is True


def test_conservative_true_on_missing_hf_config(monkeypatch):
    class _Bad:
        @property
        def model_config(self):
            raise AttributeError("no model_config")
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: _Bad(), raising=False,
    )
    model_detect.clear_for_tests()
    profile = model_detect.get_model_profile()
    assert profile["resolved"] is False
    assert profile["moe"] is True
    assert profile["hybrid"] is True


# ════════════════════════════════════════════════════════════════════════
#                            CACHING BEHAVIOUR
# ════════════════════════════════════════════════════════════════════════

def test_profile_cached_across_calls(monkeypatch):
    call_count = {"n": 0}
    cfg = FakeVllmConfig(
        FakeHFConfig(
            num_experts=64, model_type="moe_arch", architectures=[],
        ),
    )

    def _fake_getter():
        call_count["n"] += 1
        return cfg

    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", _fake_getter, raising=False,
    )
    model_detect.clear_for_tests()

    _ = model_detect.get_model_profile()
    _ = model_detect.is_moe_model()
    _ = model_detect.is_hybrid_model()
    _ = model_detect.is_turboquant_active()

    # Only one config query despite 4 API calls
    assert call_count["n"] == 1


def test_clear_for_tests_resets_cache(monkeypatch):
    cfg1 = FakeVllmConfig(
        FakeHFConfig(num_experts=64, model_type="moe", architectures=[]),
    )
    cfg2 = FakeVllmConfig(
        FakeHFConfig(model_type="qwen3", architectures=["Qwen3ForCausalLM"]),
    )
    current = {"cfg": cfg1}
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config",
        lambda: current["cfg"], raising=False,
    )

    model_detect.clear_for_tests()
    assert model_detect.is_moe_model() is True

    current["cfg"] = cfg2
    # Without clear, still cached as True
    assert model_detect.is_moe_model() is True

    model_detect.clear_for_tests()
    assert model_detect.is_moe_model() is False


# ════════════════════════════════════════════════════════════════════════
#                          LOG HELPER
# ════════════════════════════════════════════════════════════════════════

def test_log_skip_emits_without_raising(caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="genesis.model_detect"):
        model_detect.log_skip("P37", "dense model")
    assert any(
        "P37" in rec.message and "dense model" in rec.message
        for rec in caplog.records
    )
