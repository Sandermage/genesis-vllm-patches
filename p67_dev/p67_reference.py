"""P67 reference implementation — pure PyTorch multi-query attention against
TurboQuant-compressed KV cache. Slow but correct ground truth for kernel TDD.

Builds in 4 layers:
  Layer 1: pure multi-query attention (no compression, no GQA, no causal)
  Layer 2: + GQA grouping (Hq heads sharing Hk KV heads)
  Layer 3: + causal mask within K+1 query tile
  Layer 4: + TurboQuant compression layer (FP8 K, 3/4-bit V + scales/zeros)

Each layer has its own test in p67_test.py. Triton kernel is built layer by
layer matching the same staged correctness proof.

Key definitions:
  B          batch size (typically 1-8 in production)
  K          num_speculative_tokens (typically 3 for MTP n=3)
  K_PLUS_1   K + 1 = query tokens per request (1 verify + K drafts)
  Hq         total query heads
  Hk         KV heads (Hq // KV_GROUP_SIZE for GQA)
  D          head dimension (typically 128)
  S          sequence length (prior cached context, typically up to 256K)

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import math
import torch


# ─── Layer 1: vanilla multi-query attention ──────────────────────────────


def reference_multi_query_attention_layer1(
    q: torch.Tensor,    # [B, K_PLUS_1, Hq, D] fp16/fp32
    k: torch.Tensor,    # [B, S, Hq, D] fp16/fp32 (full prior + current chunk)
    v: torch.Tensor,    # [B, S, Hq, D] fp16/fp32
    scale: float,       # 1/sqrt(D)
) -> torch.Tensor:
    """No GQA, no causal — just multi-query attention.

    Each q[b, t, h, :] attends to all k[b, :, h, :].
    Returns o[b, t, h, :] shape [B, K_PLUS_1, Hq, D].
    """
    B, K_PLUS_1, Hq, D = q.shape
    S = k.shape[1]
    assert k.shape == (B, S, Hq, D)
    assert v.shape == (B, S, Hq, D)

    # Promote to fp32 for stability
    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)

    # scores[b, t, h, s] = sum_d q[b, t, h, d] * k[b, s, h, d]
    scores = torch.einsum("bthd,bshd->bths", q_f, k_f) * scale

    # softmax over s
    p = torch.softmax(scores, dim=-1)

    # out[b, t, h, d] = sum_s p[b, t, h, s] * v[b, s, h, d]
    out = torch.einsum("bths,bshd->bthd", p, v_f)
    return out.to(q.dtype)


# ─── Layer 2: + GQA grouping ─────────────────────────────────────────────


def reference_multi_query_attention_layer2(
    q: torch.Tensor,    # [B, K_PLUS_1, Hq, D]
    k: torch.Tensor,    # [B, S, Hk, D] — KV has fewer heads
    v: torch.Tensor,    # [B, S, Hk, D]
    scale: float,
    kv_group_size: int, # Hq // Hk
) -> torch.Tensor:
    """GQA: each KV head is shared by kv_group_size query heads.

    q_head h attends to k_head (h // kv_group_size).
    """
    B, K_PLUS_1, Hq, D = q.shape
    S, Hk = k.shape[1], k.shape[2]
    assert Hq == Hk * kv_group_size
    assert k.shape == (B, S, Hk, D)
    assert v.shape == (B, S, Hk, D)

    # Expand KV to per-q-head view: repeat each Hk head kv_group_size times
    k_expanded = k.repeat_interleave(kv_group_size, dim=2)  # [B, S, Hq, D]
    v_expanded = v.repeat_interleave(kv_group_size, dim=2)
    return reference_multi_query_attention_layer1(q, k_expanded, v_expanded, scale)


# ─── Layer 3: + causal mask within K+1 ───────────────────────────────────


def reference_multi_query_attention_layer3(
    q: torch.Tensor,         # [B, K_PLUS_1, Hq, D]
    k_cached: torch.Tensor,  # [B, S_prior, Hk, D] prior cached KV
    v_cached: torch.Tensor,  # [B, S_prior, Hk, D]
    k_chunk: torch.Tensor,   # [B, K_PLUS_1, Hk, D] current chunk K (just-quantized)
    v_chunk: torch.Tensor,   # [B, K_PLUS_1, Hk, D]
    scale: float,
    kv_group_size: int,
    seq_lens: torch.Tensor,  # [B] int — prior cached length per request
) -> torch.Tensor:
    """Multi-query causal attention against prior cached KV + current chunk.

    For each batch b, query token t, query head h:
      q_pos[t] = seq_lens[b] + t
      attends to:
        - all prior cached k_cached[b, 0..seq_lens[b]-1, h//kv_group_size, :]
        - current chunk k_chunk[b, 0..t, h//kv_group_size, :] (causal)
      ignores k_chunk[b, t+1..K, ...] (future)

    Returns o[b, t, h, :] shape [B, K_PLUS_1, Hq, D].
    """
    B, K_PLUS_1, Hq, D = q.shape
    S_prior = k_cached.shape[1]
    Hk = k_cached.shape[2]
    assert Hq == Hk * kv_group_size

    out = torch.zeros_like(q, dtype=torch.float32)

    for b in range(B):
        sl = int(seq_lens[b].item())
        # Concatenate cached + chunk for this request
        k_full = torch.cat([k_cached[b, :sl], k_chunk[b]], dim=0)  # [sl + K_PLUS_1, Hk, D]
        v_full = torch.cat([v_cached[b, :sl], v_chunk[b]], dim=0)

        # Expand to Hq view
        k_exp = k_full.repeat_interleave(kv_group_size, dim=1)  # [sl+K+1, Hq, D]
        v_exp = v_full.repeat_interleave(kv_group_size, dim=1)

        for t in range(K_PLUS_1):
            q_pos = sl + t
            # Attend to k positions [0, q_pos] inclusive (= sl + t + 1 positions)
            kv_end = q_pos + 1  # = sl + t + 1
            k_sub = k_exp[:kv_end]  # [kv_end, Hq, D]
            v_sub = v_exp[:kv_end]

            for h in range(Hq):
                q_vec = q[b, t, h].to(torch.float32)  # [D]
                k_h = k_sub[:, h].to(torch.float32)   # [kv_end, D]
                v_h = v_sub[:, h].to(torch.float32)

                scores = (q_vec @ k_h.T) * scale  # [kv_end]
                p = torch.softmax(scores, dim=-1)
                out[b, t, h] = p @ v_h

    return out.to(q.dtype)


# ─── Layer 4: + TurboQuant decompression ─────────────────────────────────
# This is where it gets fun. TurboQuant encodes K as FP8 (when key_fp8=True)
# or as MSE-quantized centroid indices (when key_fp8=False), and V as
# 3-bit or 4-bit indices + per-token scale/zero (FP16). Layer 4 takes
# the COMPRESSED representations and decompresses, then calls Layer 3.


def turboquant_dequant_k_fp8(k_compressed: torch.Tensor) -> torch.Tensor:
    """FP8 K storage → fp32. k_compressed is uint8 representing FP8 e4nv values."""
    # In Triton kernel: k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    # PyTorch equivalent: bitcast uint8 -> float8_e4m3fn (close to e4nv) -> fp32
    # For testing simplicity, accept fp32 already (no actual fp8 quant in test)
    if k_compressed.dtype == torch.float32 or k_compressed.dtype == torch.float16:
        return k_compressed.to(torch.float32)
    # Real FP8 path
    return k_compressed.view(torch.float8_e4m3fn).to(torch.float32)


def turboquant_dequant_v_4bit(
    v_idx_packed: torch.Tensor,  # [B, S, Hk, D//2] uint8 (2 nibbles per byte)
    v_scales: torch.Tensor,      # [B, S, Hk] fp16
    v_zeros: torch.Tensor,       # [B, S, Hk] fp16
) -> torch.Tensor:
    """4-bit value dequant: v[b,s,h,d] = idx[b,s,h,d] * scale[b,s,h] + zero[b,s,h]."""
    B, S, Hk, D_half = v_idx_packed.shape
    D = D_half * 2

    # Unpack 4-bit indices
    low = (v_idx_packed & 0xF).to(torch.float32)  # [..., D_half]
    high = ((v_idx_packed >> 4) & 0xF).to(torch.float32)
    v_idx = torch.stack([low, high], dim=-1).reshape(B, S, Hk, D)  # [..., D]

    return (
        v_idx * v_scales[:, :, :, None].to(torch.float32)
        + v_zeros[:, :, :, None].to(torch.float32)
    )


def reference_multi_query_attention_layer4_turboquant(
    q: torch.Tensor,             # [B, K_PLUS_1, Hq, D] fp16
    k_cached_compressed: torch.Tensor,  # [B, S_prior, Hk, D] (FP8 stored as uint8)
    v_cached_idx: torch.Tensor,         # [B, S_prior, Hk, D//2] uint8 (4-bit packed)
    v_cached_scales: torch.Tensor,      # [B, S_prior, Hk] fp16
    v_cached_zeros: torch.Tensor,       # [B, S_prior, Hk] fp16
    k_chunk: torch.Tensor,              # [B, K_PLUS_1, Hk, D] fp16 (current chunk uncompressed)
    v_chunk: torch.Tensor,              # [B, K_PLUS_1, Hk, D] fp16
    scale: float,
    kv_group_size: int,
    seq_lens: torch.Tensor,             # [B] int
) -> torch.Tensor:
    """Layer 4: dequantize cached KV from TurboQuant format then call Layer 3."""
    k_cached = turboquant_dequant_k_fp8(k_cached_compressed)
    v_cached = turboquant_dequant_v_4bit(
        v_cached_idx, v_cached_scales, v_cached_zeros
    )
    return reference_multi_query_attention_layer3(
        q, k_cached, v_cached, k_chunk, v_chunk, scale, kv_group_size, seq_lens
    )


# ─── Convenience: synthetic test data generator ──────────────────────────


def make_synthetic_test_inputs(
    B: int = 1,
    K_PLUS_1: int = 4,
    Hq: int = 32,
    Hk: int = 8,  # kv_group_size = 4
    D: int = 128,
    S_prior: int = 290,
    dtype: torch.dtype = torch.float16,
    seed: int = 42,
) -> dict:
    """Generate synthetic test tensors. Uses fp16 for K/V (no compression)."""
    g = torch.Generator().manual_seed(seed)
    return {
        "q": torch.randn((B, K_PLUS_1, Hq, D), generator=g, dtype=dtype),
        "k_cached": torch.randn((B, S_prior, Hk, D), generator=g, dtype=dtype),
        "v_cached": torch.randn((B, S_prior, Hk, D), generator=g, dtype=dtype),
        "k_chunk": torch.randn((B, K_PLUS_1, Hk, D), generator=g, dtype=dtype),
        "v_chunk": torch.randn((B, K_PLUS_1, Hk, D), generator=g, dtype=dtype),
        "seq_lens": torch.full((B,), S_prior, dtype=torch.int32),
        "scale": 1.0 / math.sqrt(D),
        "kv_group_size": Hq // Hk,
    }
