# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Genesis Gemma 4 Triton kernel reference implementations.

We test the **reference** (torch-only) paths exhaustively. The Triton
kernels themselves are validated against the reference on the server
(CUDA required). These tests run in CI without CUDA.
"""
from __future__ import annotations

import math

import pytest

try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

pytestmark = pytest.mark.skipif(not _TORCH_OK, reason="torch not installed")


# ─── Fused RMSNorm reference ─────────────────────────────────────────


def test_g4_rmsnorm_residual_scalar_reference_matches_naive():
    from vllm.sndr_core.integrations.gemma4.kernels.g4_fused_rmsnorm_triton import (
        g4_rmsnorm_residual_scalar_reference,
    )
    torch.manual_seed(0)
    M, N = 8, 256
    x = torch.randn(M, N, dtype=torch.float32) * 0.5
    w = torch.randn(N, dtype=torch.float32) * 0.1 + 1.0
    r = torch.randn(M, N, dtype=torch.float32) * 0.2
    s = torch.tensor(1.5, dtype=torch.float32)
    eps = 1e-6

    fused = g4_rmsnorm_residual_scalar_reference(x, w, residual=r, scalar=s, eps=eps)

    # Naive implementation
    var = (x * x).mean(dim=-1, keepdim=True)
    rrms = torch.rsqrt(var + eps)
    naive = (x * rrms * w + r) * s

    assert fused.shape == (M, N)
    assert torch.allclose(fused, naive, atol=1e-5)


def test_g4_rmsnorm_residual_scalar_reference_no_residual_no_scalar():
    from vllm.sndr_core.integrations.gemma4.kernels.g4_fused_rmsnorm_triton import (
        g4_rmsnorm_residual_scalar_reference,
    )
    x = torch.randn(4, 128, dtype=torch.float32)
    w = torch.randn(128, dtype=torch.float32) + 1.0
    fused = g4_rmsnorm_residual_scalar_reference(x, w)
    var = (x * x).mean(dim=-1, keepdim=True)
    naive = x * torch.rsqrt(var + 1e-6) * w
    assert torch.allclose(fused, naive, atol=1e-5)


def test_g4_qkv_rmsnorm_reference_per_head():
    from vllm.sndr_core.integrations.gemma4.kernels.g4_fused_rmsnorm_triton import (
        g4_qkv_rmsnorm_reference,
    )
    torch.manual_seed(0)
    M = 4
    num_q_heads, num_kv_heads, head_dim = 8, 2, 256
    q = torch.randn(M, num_q_heads * head_dim, dtype=torch.float32) * 0.3
    k = torch.randn(M, num_kv_heads * head_dim, dtype=torch.float32) * 0.3
    v = torch.randn(M, num_kv_heads * head_dim, dtype=torch.float32) * 0.3
    qw = torch.randn(head_dim, dtype=torch.float32) + 1.0
    kw = torch.randn(head_dim, dtype=torch.float32) + 1.0

    # Save original V for verification (V is normalized in-place WITHOUT scale)
    q_orig = q.clone()
    v_orig = v.clone()

    g4_qkv_rmsnorm_reference(q, k, v, qw, kw, num_q_heads, num_kv_heads, head_dim)

    # Verify Q was normalized per-head with qw scale
    q_check = q_orig.view(M, num_q_heads, head_dim)
    var = (q_check * q_check).mean(dim=-1, keepdim=True)
    expected_q = (q_check * torch.rsqrt(var + 1e-6) * qw).view(M, num_q_heads * head_dim)
    assert torch.allclose(q, expected_q, atol=1e-5)

    # Verify V was normalized per-head WITHOUT scale
    v_check = v_orig.view(M, num_kv_heads, head_dim)
    var = (v_check * v_check).mean(dim=-1, keepdim=True)
    expected_v = (v_check * torch.rsqrt(var + 1e-6)).view(M, num_kv_heads * head_dim)
    assert torch.allclose(v, expected_v, atol=1e-5)


def test_g4_dual_rmsnorm_residual_scalar_reference():
    from vllm.sndr_core.integrations.gemma4.kernels.g4_fused_rmsnorm_triton import (
        g4_dual_rmsnorm_residual_scalar_reference,
    )
    torch.manual_seed(0)
    M, N = 4, 256
    x1 = torch.randn(M, N) * 0.3
    x2 = torch.randn(M, N) * 0.3
    w1 = torch.randn(N) + 1.0
    w2 = torch.randn(N) + 1.0
    w3 = torch.randn(N) + 1.0
    r = torch.randn(M, N) * 0.1
    s = torch.tensor(0.5)

    out = g4_dual_rmsnorm_residual_scalar_reference(x1, w1, x2, w2, w3, r, s)
    assert out.shape == (M, N)
    assert torch.isfinite(out).all()


# ─── Softcap reference ───────────────────────────────────────────────


def test_g4_softcap_reference_matches_naive():
    from vllm.sndr_core.integrations.gemma4.kernels.g4_softcap_triton import (
        g4_softcap_reference,
    )
    torch.manual_seed(0)
    x = torch.randn(8, 64, dtype=torch.float32) * 50.0  # Large values to exercise tanh saturation
    softcap = 30.0
    fused = g4_softcap_reference(x, softcap)
    naive = torch.tanh(x / softcap) * softcap
    assert torch.allclose(fused, naive, atol=1e-5)


def test_g4_softcap_reference_zero_softcap_is_noop():
    from vllm.sndr_core.integrations.gemma4.kernels.g4_softcap_triton import (
        g4_softcap_reference,
    )
    x = torch.randn(4, 8)
    out = g4_softcap_reference(x, 0.0)
    assert out is x


def test_g4_softcap_reference_none_softcap_is_noop():
    from vllm.sndr_core.integrations.gemma4.kernels.g4_softcap_triton import (
        g4_softcap_reference,
    )
    x = torch.randn(4, 8)
    out = g4_softcap_reference(x, None)
    assert out is x


def test_g4_softcap_reference_inplace_out():
    from vllm.sndr_core.integrations.gemma4.kernels.g4_softcap_triton import (
        g4_softcap_reference,
    )
    x = torch.randn(4, 8)
    x_copy = x.clone()
    out = torch.empty_like(x)
    g4_softcap_reference(x, 30.0, out=out)
    # Original x untouched
    assert torch.allclose(x, x_copy)
    # out has the result
    assert torch.allclose(out, torch.tanh(x / 30.0) * 30.0, atol=1e-5)


# ─── Detection utilities ─────────────────────────────────────────────


def test_is_gemma4_arch_recognizes_known_names():
    from vllm.sndr_core.integrations.gemma4._gemma4_detect import is_gemma4_arch
    assert is_gemma4_arch("Gemma4ForConditionalGeneration")
    assert is_gemma4_arch("Gemma4ForCausalLM")
    assert is_gemma4_arch(["Gemma4ForCausalLM"])
    assert is_gemma4_arch("gemma4")
    assert is_gemma4_arch("gemma4_assistant")
    assert not is_gemma4_arch("Qwen3ForCausalLM")
    assert not is_gemma4_arch("LlamaForCausalLM")
    assert not is_gemma4_arch(None)


def test_marlin_kdim_supported_python_vs_cpp_check():
    from vllm.sndr_core.integrations.gemma4._gemma4_detect import marlin_kdim_supported
    # K=352 — fails BOTH because 352%128 != 0 AND 352%64 != 0
    assert not marlin_kdim_supported(352, strict_python_check=True)
    assert not marlin_kdim_supported(352, strict_python_check=False)
    # K=384 — passes C++ (384%64=0) AND Python (384%128=0)
    assert marlin_kdim_supported(384, strict_python_check=True)
    assert marlin_kdim_supported(384, strict_python_check=False)
    # K=192 — passes C++ (192%64=0) but NOT Python (192%128=64)
    assert not marlin_kdim_supported(192, strict_python_check=True)
    assert marlin_kdim_supported(192, strict_python_check=False)


def test_detect_fp8_block_format():
    from vllm.sndr_core.integrations.gemma4._gemma4_detect import detect_fp8_block_format

    # Mock checkpoint config matching FP8_BLOCK signature
    class _Cfg:
        quant_method = "compressed-tensors"
        format = "float-quantized"
        weight_quant = {"strategy": "block", "block_structure": [128, 128]}

    assert detect_fp8_block_format(_Cfg())

    # Standard FP8 e4m3 channel-wise — NOT block
    class _Cfg2:
        quant_method = "compressed-tensors"
        format = "float-quantized"
        weight_quant = {"strategy": "channel"}

    assert not detect_fp8_block_format(_Cfg2())
    assert not detect_fp8_block_format(None)


def test_detect_non_causal_drafter():
    from vllm.sndr_core.integrations.gemma4._gemma4_detect import detect_non_causal_drafter

    class _Cfg:
        method = "eagle3"

    assert detect_non_causal_drafter(_Cfg()) == "eagle3"

    class _Cfg2:
        method = "dflash"

    assert detect_non_causal_drafter(_Cfg2()) == "dflash"

    class _Cfg3:
        method = "mtp"  # causal — not affected

    assert detect_non_causal_drafter(_Cfg3()) is None
    assert detect_non_causal_drafter(None) is None


# ─── K-padding helper (G4_08 pre-step) ───────────────────────────────


def test_pad_moe_weight_to_aligned_k_pads_correctly():
    from vllm.sndr_core.integrations.gemma4.kernels.g4_kpad_moe_gemm_triton import (
        pad_moe_weight_to_aligned_k,
    )
    # Mock MoE weight: [num_experts=4, hidden_size=512, K_real=352]
    w = torch.randn(4, 512, 352)
    K_real = 352
    align = 64
    padded, mask, K_padded = pad_moe_weight_to_aligned_k(w, K_real, align)
    assert K_padded == 384  # next multiple of 64 ≥ 352
    assert padded.shape == (4, 512, 384)
    # Padding zone should be zero
    assert padded[:, :, K_real:].abs().sum().item() == 0.0
    # Real zone preserved
    assert torch.allclose(padded[:, :, :K_real], w)
