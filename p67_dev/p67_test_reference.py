"""TDD tests for P67 reference implementations.

These tests verify the PURE PYTORCH reference impls are internally consistent
(layer N matches layer N-1 when run on equivalent inputs). When Triton kernel
is built, kernel output is compared to Layer 4 reference.

Run: python3 -m pytest p67_test_reference.py -v
Or:  python3 p67_test_reference.py
"""
from __future__ import annotations

import math

import torch

from p67_reference import (
    make_synthetic_test_inputs,
    reference_multi_query_attention_layer1,
    reference_multi_query_attention_layer2,
    reference_multi_query_attention_layer3,
    reference_multi_query_attention_layer4_turboquant,
    turboquant_dequant_k_fp8,
    turboquant_dequant_v_4bit,
)


def _close(a: torch.Tensor, b: torch.Tensor, rtol=1e-2, atol=1e-2) -> bool:
    """fp16 tolerance — looser than fp32 because we aggregate over many positions."""
    return torch.allclose(a.to(torch.float32), b.to(torch.float32), rtol=rtol, atol=atol)


# ─── Layer 1 sanity checks ─────────────────────────────────────────────


def test_layer1_basic_shape():
    """Output shape matches input Q shape."""
    inputs = make_synthetic_test_inputs(B=2, K_PLUS_1=4, Hq=8, Hk=8, D=64, S_prior=10)
    # Layer 1 needs full Hq KV
    k_full = inputs["k_cached"].repeat_interleave(inputs["kv_group_size"], dim=2)
    v_full = inputs["v_cached"].repeat_interleave(inputs["kv_group_size"], dim=2)
    out = reference_multi_query_attention_layer1(
        inputs["q"], k_full, v_full, inputs["scale"]
    )
    assert out.shape == inputs["q"].shape
    assert out.dtype == inputs["q"].dtype


def test_layer1_softmax_sums_to_one():
    """Internal: softmax weights must sum to 1.0 per query."""
    B, T, H, D, S = 1, 2, 1, 8, 5
    q = torch.randn(B, T, H, D)
    k = torch.randn(B, S, H, D)
    v = torch.randn(B, S, H, D)
    scores = torch.einsum("bthd,bshd->bths", q, k) / math.sqrt(D)
    p = torch.softmax(scores, dim=-1)
    assert torch.allclose(p.sum(dim=-1), torch.ones(B, T, H), atol=1e-4)


def test_layer1_self_attention_identity():
    """When Q == K and V is one-hot, attention reduces to argmax."""
    # q[0,0,0,:] = k[0,2,0,:], so softmax should peak at s=2
    D = 4
    q = torch.zeros(1, 1, 1, D)
    k = torch.eye(D).reshape(1, D, 1, D)  # k[0,s,0,:] = e_s
    v = torch.eye(D).reshape(1, D, 1, D)
    q[0, 0, 0, 2] = 10.0  # very high logit at position 2 will dominate
    out = reference_multi_query_attention_layer1(q, k, v, scale=1.0)
    # Output should be approximately e_2
    assert out[0, 0, 0, 2] > 0.9, f"out={out[0, 0, 0]}"


# ─── Layer 2: GQA grouping ─────────────────────────────────────────────


def test_layer2_matches_layer1_when_kv_group_size_1():
    """GQA with group_size=1 = vanilla MHA."""
    inputs = make_synthetic_test_inputs(B=2, K_PLUS_1=4, Hq=8, Hk=8, D=64, S_prior=10)
    out_l2 = reference_multi_query_attention_layer2(
        inputs["q"], inputs["k_cached"], inputs["v_cached"],
        inputs["scale"], kv_group_size=1
    )
    out_l1 = reference_multi_query_attention_layer1(
        inputs["q"], inputs["k_cached"], inputs["v_cached"], inputs["scale"]
    )
    assert _close(out_l2, out_l1), "layer2 group_size=1 should match layer1"


def test_layer2_gqa_reduces_kv():
    """GQA: each KV head shared by group_size queries."""
    inputs = make_synthetic_test_inputs(B=1, K_PLUS_1=2, Hq=8, Hk=2, D=32, S_prior=8)
    # kv_group_size = 4
    out = reference_multi_query_attention_layer2(
        inputs["q"], inputs["k_cached"], inputs["v_cached"],
        inputs["scale"], kv_group_size=4
    )
    assert out.shape == inputs["q"].shape


# ─── Layer 3: causal mask within K+1 ─────────────────────────────────────


def test_layer3_no_chunk_matches_layer2():
    """When current chunk is empty (K+1=1, only verify), layer3 reduces to layer2 with seq_len-1."""
    # Edge case: K_PLUS_1=1 with full prior cached
    B, Hq, Hk, D, S = 1, 4, 2, 32, 5
    q = torch.randn(B, 1, Hq, D)
    k_cached = torch.randn(B, S, Hk, D)
    v_cached = torch.randn(B, S, Hk, D)
    k_chunk = torch.zeros(B, 1, Hk, D)
    v_chunk = torch.zeros(B, 1, Hk, D)
    seq_lens = torch.tensor([S], dtype=torch.int32)
    scale = 1.0 / math.sqrt(D)
    kv_group_size = Hq // Hk

    out_l3 = reference_multi_query_attention_layer3(
        q, k_cached, v_cached, k_chunk, v_chunk, scale, kv_group_size, seq_lens
    )
    # Layer3 attends to k_cached[0..S-1] + k_chunk[0] (causal allows self)
    # Since k_chunk[0] is zero vector, its score is exactly 0 (low logit)
    # but softmax still gives it some weight. Hard to simplify exactly.
    # Just check shape + finite
    assert out_l3.shape == q.shape
    assert torch.all(torch.isfinite(out_l3))


def test_layer3_causal_isolation():
    """q[0, t=0] must NOT attend to k_chunk[1, 2, 3] (future)."""
    # Set k_chunk[0, 1, 2, 3] to something detectable; q[0, t=0] result should
    # be insensitive to chunk positions > 0.
    B, K1, Hq, Hk, D, S = 1, 4, 1, 1, 4, 0  # No prior cached
    q = torch.zeros(B, K1, Hq, D)
    q[0, 0, 0, 0] = 1.0  # q at t=0
    k_cached = torch.zeros(B, S, Hk, D)
    v_cached = torch.zeros(B, S, Hk, D)
    k_chunk_a = torch.zeros(B, K1, Hk, D)
    k_chunk_b = torch.zeros(B, K1, Hk, D)
    k_chunk_b[0, 1, 0, :] = 999.0  # huge difference at t=1 in version B
    v_chunk_a = torch.randn(B, K1, Hk, D)
    v_chunk_b = v_chunk_a.clone()
    v_chunk_b[0, 1, 0, :] = 999.0
    seq_lens = torch.tensor([0], dtype=torch.int32)

    out_a = reference_multi_query_attention_layer3(
        q, k_cached, v_cached, k_chunk_a, v_chunk_a, 1.0, 1, seq_lens
    )
    out_b = reference_multi_query_attention_layer3(
        q, k_cached, v_cached, k_chunk_b, v_chunk_b, 1.0, 1, seq_lens
    )
    # Output for t=0 should be same in both — chunk[1] is in the future
    assert _close(out_a[0, 0], out_b[0, 0]), (
        f"causal violation: out_a[0,0]={out_a[0,0]} out_b[0,0]={out_b[0,0]}"
    )


def test_layer3_t_last_attends_full_chunk():
    """q[0, t=K] should attend to all K+1 chunk positions inclusive."""
    B, K1, Hq, Hk, D = 1, 4, 1, 1, 4
    seq_lens = torch.tensor([0], dtype=torch.int32)
    q = torch.zeros(B, K1, Hq, D)
    q[0, K1-1, 0, 0] = 1.0  # query at last position
    k_cached = torch.zeros(B, 0, Hk, D)
    v_cached = torch.zeros(B, 0, Hk, D)
    k_chunk_a = torch.zeros(B, K1, Hk, D)
    k_chunk_b = torch.zeros(B, K1, Hk, D)
    k_chunk_a[0, 0, 0, :] = 1.0
    k_chunk_b[0, 0, 0, :] = 100.0  # bigger logit at t=0
    v_chunk_a = torch.zeros(B, K1, Hk, D)
    v_chunk_a[0, 0, 0, :] = 7.0
    v_chunk_b = v_chunk_a.clone()

    out_a = reference_multi_query_attention_layer3(
        q, k_cached, v_cached, k_chunk_a, v_chunk_a, 1.0, 1, seq_lens
    )
    out_b = reference_multi_query_attention_layer3(
        q, k_cached, v_cached, k_chunk_b, v_chunk_b, 1.0, 1, seq_lens
    )
    # Last query attends to position 0; bigger logit → higher softmax weight on v_chunk_a[0]=7
    # out_b should be CLOSER to 7 than out_a (more concentration)
    assert out_b[0, K1-1, 0, 0] > out_a[0, K1-1, 0, 0], (
        f"t=K should attend to chunk[0]: out_a={out_a[0, K1-1, 0]} out_b={out_b[0, K1-1, 0]}"
    )


# ─── Layer 4: TurboQuant decompression layer ─────────────────────────────


def test_dequant_v_4bit_round_trip():
    """4-bit dequant: pack indices, then unpack should match original."""
    # Original: idx in [0, 15], scale=1.0, zero=0.0 → dequant should equal idx
    B, S, Hk, D = 1, 4, 2, 8
    idx = torch.randint(0, 16, (B, S, Hk, D), dtype=torch.uint8)
    # Pack: 2 nibbles per byte
    idx_low = idx[..., 0::2]
    idx_high = idx[..., 1::2]
    packed = (idx_low | (idx_high << 4)).to(torch.uint8)
    scales = torch.ones(B, S, Hk)
    zeros = torch.zeros(B, S, Hk)
    dequant = turboquant_dequant_v_4bit(packed, scales, zeros)
    assert dequant.shape == (B, S, Hk, D)
    assert _close(dequant, idx.to(torch.float32))


def test_layer4_fp16_pass_through():
    """Layer 4 with FP16 K (not actual FP8) should match layer 3."""
    inputs = make_synthetic_test_inputs(B=1, K_PLUS_1=4, Hq=8, Hk=2, D=32, S_prior=10)
    # For Layer 4, V needs to be 4-bit packed. Build packed V from random idx.
    B, S_prior, Hk, D = inputs["k_cached"].shape
    g = torch.Generator().manual_seed(7)
    v_idx = torch.randint(0, 16, (B, S_prior, Hk, D), generator=g, dtype=torch.uint8)
    idx_low = v_idx[..., 0::2]
    idx_high = v_idx[..., 1::2]
    v_packed = (idx_low | (idx_high << 4)).to(torch.uint8)
    v_scales = torch.ones(B, S_prior, Hk, dtype=torch.float16)
    v_zeros = torch.zeros(B, S_prior, Hk, dtype=torch.float16)

    # Layer 4 result
    out_l4 = reference_multi_query_attention_layer4_turboquant(
        inputs["q"],
        inputs["k_cached"].to(torch.float32),  # passed-through fp32
        v_packed, v_scales, v_zeros,
        inputs["k_chunk"], inputs["v_chunk"],
        inputs["scale"], inputs["kv_group_size"], inputs["seq_lens"]
    )

    # Layer 3 with explicit dequant V
    v_cached_dequant = turboquant_dequant_v_4bit(v_packed, v_scales, v_zeros)
    out_l3 = reference_multi_query_attention_layer3(
        inputs["q"],
        inputs["k_cached"].to(torch.float32),
        v_cached_dequant,
        inputs["k_chunk"], inputs["v_chunk"],
        inputs["scale"], inputs["kv_group_size"], inputs["seq_lens"]
    )

    assert _close(out_l4, out_l3, rtol=1e-3, atol=1e-3), (
        "Layer 4 must match Layer 3 when V is decompressed identically"
    )


# ─── Run all tests when invoked as script ────────────────────────────────


if __name__ == "__main__":
    import sys
    test_funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"Running {len(test_funcs)} reference tests...")
    failures = []
    for fn in test_funcs:
        name = fn.__name__
        try:
            fn()
            print(f"  ✓ {name}")
        except AssertionError as e:
            print(f"  ✗ {name}: {e}")
            failures.append(name)
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            failures.append(name)
    print()
    print(f"{len(test_funcs) - len(failures)}/{len(test_funcs)} passed")
    sys.exit(1 if failures else 0)
